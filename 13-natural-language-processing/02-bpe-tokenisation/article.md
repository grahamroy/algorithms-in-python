# Tokenisation — Who Decides the Pieces?

### *Algorithms in Python --- Natural Language Processing, Part 2*

---

Part 1 ended with an insurance policy. fastText built every
word from character 3–5-grams, and when three words that had
never appeared in the corpus arrived for pricing, their n-gram
vectors landed beside their singulars anyway. Morphology was
the insurance — but the pieces were hand-chosen: every
trigram, every 4-gram, every 5-gram, mechanically enumerated.
A six-letter word generates 28 of them. Most are noise.

This article asks a sharper question:

```
    can the corpus decide which pieces to keep?
```

Three algorithms, three answers. **Byte-pair encoding**
(Sennrich et al., 2016) starts from individual characters and
greedily merges the most frequent adjacent pair — repeating
until a vocabulary budget is spent. **WordPiece** (Schuster &
Nakajima, 2012; used by BERT) runs the same loop but scores
by mutual information instead of raw count: which pair is most
*surprising* given how common each part is on its own?
**Unigram LM** (Kudo, 2018; used by SentencePiece, T5, Llama)
goes the other way entirely: start with every substring, then
*prune* by how much each token's removal hurts corpus
likelihood, until the budget is met.

The stage is Part 1's synthetic micro-language — the same
three topics, 106 words, 165,166 tokens — so the same words
whose embeddings we measured can now be measured through a
different lens: not *what neighbours does a word have?* but
*what pieces should a word be built from?*

---

## A worked example: merge, score, prune, compress

### BPE: greedy merging from characters

Start from individual characters plus an end-of-word marker
(`_`). Count every adjacent pair in the corpus. Merge the
most frequent. Repeat.

The first fifteen merges on Part 1's corpus:

```
    merge  1:  'h'  + 'e'    ->  'he'
    merge  2:  't'  + 'he'   ->  'the'
    merge  3:  'the'+ '_'    ->  'the_'
    merge  4:  's'  + '_'    ->  's_'
    merge  5:  'r'  + '_'    ->  'r_'
    merge  6:  'a'  + 'n'    ->  'an'
    merge  7:  'e'  + '_'    ->  'e_'
    merge  8:  's'  + 't'    ->  'st'
    merge  9:  'o'  + 'r'    ->  'or'
    merge 10:  'the'+ 'i'    ->  'thei'
    merge 11:  'thei'+'r_'   ->  'their_'
    merge 12:  'n'  + '_'    ->  'n_'
    merge 13:  'd'  + '_'    ->  'd_'
    merge 14:  'r'  + 'o'    ->  'ro'
    merge 15:  'a'  + 'r'    ->  'ar'
```

The corpus speaks immediately. Merge 1–3 build `the_` — the
most common word. Merges 10–11 build `their_` — the third
most common. No rule tells BPE about English articles or
possessives; frequency alone promotes them. By merge 200 the
entire corpus vocabulary fits in 141 tokens, and the average
word is 1.61 tokens long:

```
    king           -> king_
    queen          -> queen_
    harvests       -> harvests_
    navigates      -> navigates_
```

Single tokens for common words, exactly as you would hope.
But the real test is the words BPE has *never seen*. Part 1's
three held-out plurals — never generated, zero occurrences —
decompose into known pieces:

```
    princesses     -> prince | s | s | es_   (OOV)
    roosters       -> roost | e | rs_        (OOV)
    compasses      -> compa | s | s | es_    (OOV)
```

Every piece was learned from other words in the corpus.
`prince` from *prince*, `es_` from every word ending in *-es*.
The merge table *is* the tokeniser: 200 lines that say
"whenever you see X next to Y, join them." Apply them in
order to any new string and it decomposes — no dictionary
lookup, no hand-written rules. The pieces the corpus chose
are the pieces the model gets to see.

### WordPiece: the same loop, a different score

WordPiece replaces raw frequency with a mutual-information
criterion:

```
    score(a, b) = freq(ab) / (freq(a) * freq(b))
```

A pair scores high when it appears together more often than
you would expect from each part's individual frequency. On the
same corpus, the first fifteen merges diverge completely:

```
     #   BPE                  WordPiece
    ---  ----                 ---------
     1   'h'+'e'              'q'+'u'
     2   't'+'he'             'u'+'g'
     3   'the'+'_'            'o'+'ug'
     4   's'+'_'              'l'+'oug'
     5   'r'+'_'              'p'+'loug'
     6   'a'+'n'              'u'+'l'
     7   'e'+'_'              'b'+'ul'
     8   's'+'t'              'bul'+'l'
     9   'o'+'r'              'g'+'ul'
    10   'the'+'i'            'gul'+'l'
    11   'thei'+'r_'          'm'+'u'
    12   'n'+'_'              'mu'+'d'
    13   'd'+'_'              'mud'+'d'
    14   'r'+'o'              'mudd'+'y'
    15   'a'+'r'              'o'+'u'
```

BPE builds *the* first — the most frequent word. WordPiece
builds *bull*, *gull*, *muddy* — rare words whose letter
sequences are perfectly predictable. `q` appears only before
`u` in this corpus, so the pair `q+u` has maximum
mutual information despite modest count. The rare-pair bonus
makes WordPiece greedy on morphologically informative
fragments: it learns whole stems and suffixes earlier because
they co-occur reliably, even when their raw count is modest.

Of 200 merges, only 35 are shared. The two algorithms
segment the same words differently:

```
    princess    BPE:       princess_
                WordPiece: princ | e | s | s | _

    compasses   BPE:       compa | s | s | es_
                WordPiece: compass | e | s | _
```

WordPiece builds `compass` as a single piece — the stem is
informative — while BPE keeps `compa` and `es_` separately,
stitched from frequency. BERT chose WordPiece; GPT chose BPE.
The difference is in which fragments get promoted, not in the
promise: both decompose anything into known pieces.

### Unigram LM: start big, prune by likelihood

BPE and WordPiece build *up* from characters. Unigram LM
(the algorithm inside SentencePiece) goes the other way:
start with every observed substring up to a length cap, then
iteratively *prune*.

At each round: (1) estimate token probabilities from the
current best segmentation of every training word (EM), (2)
score each token by how much corpus log-likelihood would drop
if it were removed, (3) drop the least painful fraction —
but never single characters. Repeat until the vocabulary
budget is met.

Segmentation is not merge-by-merge but global: the **Viterbi
algorithm** (dynamic programming) finds the highest-probability
tokenisation of every word under the current unigram model.

Trained on Part 1's corpus with a target of 80 tokens:

```
    Top-5 tokens by probability:
      'the_'    p = 0.1296
      '_'       p = 0.0979
      's'       p = 0.0731
      'e'       p = 0.0662
      'r'       p = 0.0428
```

The in-vocabulary words segment cleanly:

```
    king       -> king_
    queen      -> queen_
    princess   -> p | rincess_
    navigates  -> n | a | vigates_
```

But the held-out words reveal a genuine difference in
character. With 80 vocabulary slots and roughly 128 unique
word forms, the Unigram model *memorises* the most frequent
whole words and spends nothing on intermediate subword
pieces — it never needs them for any training word. The OOV
fallback is pure character-level:

```
    princesses -> p|r|i|n|c|e|s|s|e|s|_  (OOV)
    roosters   -> r|o|o|s|t|e|r|s|_      (OOV)
    compasses  -> c|o|m|p|a|s|s|e|s|_    (OOV)
```

This is not a bug — it is the honest consequence of the
algorithm's optimality criterion. BPE and WordPiece *must*
pass through intermediate pieces (like `prince`, `roost`,
`es_`) because they build bottom-up; those intermediates
happen to be useful for OOV. Unigram, building top-down,
never pays for pieces it does not need on the training
corpus. At web scale — millions of unique words against tens
of thousands of vocabulary slots — the math reverses: whole
words will not fit, so the model learns genuinely useful
subword pieces and OOV handling improves. The difference is
one of *scale*, not design.

Comparing all three on OOV:

```
    word          BPE                WordPiece
    ----          ---                ---------
    princesses    prince s s es_     princ e s s e s _
    roosters      roost e rs_        roost e r s _
    compasses     compa s s es_      compass e s _

    word          Unigram (80 tokens)
    ----          -------
    princesses    p r i n c e s s e s _
    roosters      r o o s t e r s _
    compasses     c o m p a s s e s _
```

Three algorithms, three segmentation styles, one shared
promise: no word is unsayable.

### The compression curve

Every merge trades one vocabulary slot for shorter sequences:

```
    merges  vocab  tok/word  corpus tokens
    ------  -----  --------  -------------
         0     25      6.65        970,502
        25     48      4.68        578,521
        50     70      3.87        470,663
       100     98      2.92        349,226
       150    119      2.17        268,047
       200    133      1.61        206,161
       300    128      1.00        165,166
```

At zero merges every character is a token: maximum sequence
length, minimum vocabulary. At 300 merges every word is a
single token: minimum length, maximum vocabulary. Real
tokenisers sit somewhere in between. GPT-4 uses roughly
100,000 BPE tokens; Llama uses 32,000 Unigram tokens. The
choice is architectural: a larger vocabulary means shorter
sequences (faster attention — which is `O(n²)` in length —
and more context per window) but a bigger embedding matrix.
BPE's merge count is the single knob that sets this tradeoff.

At 300 merges the vocabulary *shrinks* from 200-merge levels
(133 → 128) because late merges subsume earlier intermediate
tokens that are no longer needed. The vocabulary curve is not
monotone — it peaks and then contracts as whole words replace
their parts.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

BPE learning scans every word's symbol sequence once per
merge — `O(M · N)` where `M` is the merge count and `N` the
number of symbol pairs in the corpus. Application (encoding a
new word) replays `M` merges over the word's characters:
`O(M · L)` for a word of length `L`. WordPiece has the same
complexity with a different scoring function. Unigram's
pruning loop is heavier — each round runs Viterbi over every
training word (`O(L²)` per word per vocabulary candidate) —
but runs offline; at inference it is a single Viterbi pass,
`O(L² · |V|)` in the worst case, `O(L²)` with a trie.

In practice all three run in seconds on corpora with millions
of words. Tokenisation is a *preprocessing* step: it runs
once before training and once per input at inference, so even
quadratic-in-word-length costs vanish against the model's
own `O(n² · d)` attention. The real cost is indirect:
vocabulary size sets the embedding matrix's row count, and
sequence length sets the attention bill. The compression
curve above *is* the complexity story — every merge shortens
every sequence at the price of one more embedding row.

---

## Where tokenisation sits

Tokenisation is the invisible first decision of every
language model. GPT-2 (Radford et al., 2019) adopted BPE as
its tokeniser and the field followed; BERT (Devlin et al.,
2019) chose WordPiece; T5 (Raffel et al., 2020) and Llama
(Touvron et al., 2023) chose SentencePiece's Unigram. The
choice shapes what the model can and cannot see: the same
sentence becomes a different number of tokens under each
scheme, and attention's `O(n²)` cost means that difference
compounds through every layer.

The connection to Part 1 is direct. BPE's merge loop is
fastText's character n-grams with a learned filter: instead
of enumerating every 3-to-5-gram, BPE keeps only the
substrings the corpus promotes. The Levy–Goldberg equivalence
(SGNS = shifted PMI factorisation) showed that word embeddings
are implicitly compressing co-occurrence statistics; BPE
compresses the words themselves, by the same principle —
frequent patterns get shorter codes. Sennrich et al. borrowed
the algorithm from data compression (Gage, 1994), and the
analogy is exact: BPE was invented to shrink files, and it
shrinks vocabularies by the same greedy logic.

---

## What comes next

Part 3, **Sequence-to-Sequence Models**, takes these token
sequences and asks the first generative question: given an
input sequence, produce an output sequence. The encoder
reads, the decoder writes, and the bridge between them —
attention — becomes the subject. From machine translation to
summarisation, Seq2Seq is where NLP became generative.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**bpe_tokenisation.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/13-natural-language-processing/02-bpe-tokenisation/bpe_tokenisation.py)

Run it with:

```bash
pip install numpy
python bpe_tokenisation.py
```

It needs only `numpy` and runs in about thirty seconds.
Everything is from scratch: BPE's greedy merge loop,
WordPiece's mutual-information scoring, the Unigram LM with
Viterbi segmentation and EM-trained probabilities, and the
compression curve that shows the vocabulary-vs-length
tradeoff. The headline insight worth pinning to the wall:
**the corpus decides the pieces — BPE promotes frequent pairs
(building `the_` first), WordPiece promotes surprising pairs
(building `bull` and `gull` before `the`), and Unigram
prunes from above by likelihood cost; all three decompose
unseen words into known fragments, but the bottom-up methods
(BPE, WordPiece) keep useful intermediate subword pieces by
construction, while top-down Unigram invests its budget in
whole words and falls back to characters for the unknown —
an honest consequence of optimising for the training corpus,
not for words it has never seen; the compression curve
(6.65 tokens per word at zero merges, 1.00 at 300) is the
single architectural knob that decides how large a model's
vocabulary should be, trading embedding rows for sequence
length against attention's quadratic cost**.

---

*This is Part 2 of the Natural Language Processing track in the Algorithms in Python series. The companion script `bpe_tokenisation.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It reuses Part 1's synthetic micro-language ([word embeddings](https://medium.com/p/164f7f542a3c)) and its held-out OOV words. Part 3 will look at Sequence-to-Sequence Models.*
