# Natural Language Generation — The Cost of Sounding Correct

### *Algorithms in Python --- Natural Language Processing, Part 7*

---

Every article in this track so far has compared a prediction
against an answer. Part 4 recovered a masked word that was
really there. Part 6 assigned one of three labels that a
function computed. Even Part 5, which generated text, judged
it by asking whether each sentence was grammatical.

Generation proper has no answer to compare against. There are
many good continuations of a prompt, the model does not emit
one sentence but a distribution over sentences, and the thing
you ship is not the model — it is the model plus a rule for
turning that distribution into words. Part 5 shipped the
simplest such rule and deliberately stopped there:

```
temperature 0   always take the likeliest token
temperature 1   sample from the model as it is
temperature 2   flatten first, then sample
```

Top-k, nucleus sampling and beam search were left for this
article. So was the question underneath them: when there is
no single right answer, what exactly is being measured?

Here, uniquely, that question has an arithmetic answer.

---

## A worked example: the decoder is a distribution

### The move this article is built on

Part 1's micro-language is **enumerable**. All 31,740 of its
sentences can be listed, their probabilities sum to exactly
one, and its entropy is 9.2538 nats. Part 5 used that to put
a floor under the decoder's loss.

Now notice what temperature, top-k and nucleus sampling
actually are. Each one rewrites the next-token distribution
and then samples from the rewrite. The rewrite is
deterministic. So the probability that the whole strategy
produces a particular sentence is just a product:

```
q(s) = prod_t q_trunc(s_t | s_<t)
```

and that product can be **evaluated** for every sentence in
the language rather than estimated by sampling. A decoding
strategy is a distribution, and in this universe we can
compute it exactly.

One pre-trained decoder fixes everything that follows. It is
Part 5's, unchanged — 114,286 parameters, held-out loss
1.4706 nats per token against the language's floor of 1.4190,
with 0.9770 of its probability mass sitting on real sentences.
From here the model never changes again. Only the decoding
does.

### What truncation actually costs

![[FRONTIER TABLE IMAGE]]

Four columns, no sampling error in any of them. *Grammatical*
is the exact probability the output is a sentence of the
language. *Can say* counts the sentences the strategy is
capable of producing at all. *Effective sentences* is the
exponential of its entropy. *TV* is total-variation distance
from the truth.

Read the top-k 2 row twice. It is **99.97% grammatical** and
it can produce **16 of the 31,740 sentences** — three tenths
of one percent of the language. Its grammar is nearly perfect
because there is almost nothing left for it to say.

That is the trade in its rawest form, and it does not go away
at sensible settings. Nucleus 0.90 is 99.93% grammatical and
has quietly made **more than half the language unsayable**:
14,032 sentences remain reachable out of 31,740.

The TV column makes the point that the quality column hides.
Sampling the model untruncated sits 0.2880 away from the
truth. **Every truncation setting in the table except the
gentlest moves it further away**, while making the output look
better. Truncation does not repair a model. It trades breadth
for the appearance of competence, and the exchange rate is
steep.

A consistency check, because the whole article rests on the
exact computation being right: drawing 1,000 samples per
strategy gives 0.9810 against the computed 0.9770 for
temperature 1.0, 0.9950 against 0.9961 for top-k 10, and
0.9990 against 0.9993 for nucleus 0.90 — each inside one
standard error of sampling.

### Why nucleus beats top-k

The two methods are usually compared by showing people
paragraphs and asking which they prefer. Here they can be
compared by arithmetic:

```
top-k 10      gramm 0.9961 | eff sentences 2257
nucleus 0.90  gramm 0.9993 | eff sentences 5555
```

Nucleus wins on quality, wins on diversity, and sits less than
half as far from the truth. The reason is in the mechanism. A
fixed `k` has to be chosen for the model's *flattest*
prediction, because that is where cutting too hard does the
damage — but the same `k` then applies where the next token is
nearly determined, keeping nine words that should never have
been candidates. A fixed probability mass adapts: two
candidates where the model is confident, many where it is not.

That is Holtzman et al.'s argument for nucleus sampling, and
it reproduces here as measurement rather than as preference.

### Beam search finds the wrong sentence

![[BEAM TABLE IMAGE]]

Beam search is not a sampler. It searches for the sequence of
highest total probability, and it succeeds — which is the
problem.

A sentence sampled from this model costs 9.2793 nats. The
language's entropy is 9.2538. Beam search returns sentences at
6.1840, about **three nats more probable than a typical
sentence** — a factor of roughly 22. The sentence it finds is
not a representative draw from the distribution; it is an
extreme point of it.

This is a property of distributions, not a bug in beam search.
In a distribution over 31,740 sentences, nearly all the mass
sits on sentences that are individually unlikely. The mode is
a tiny, atypical corner. Asking for the single likeliest
output and hoping for a representative one is asking for two
different things.

Watch what the larger beams do, too. Beam 25 finds a *shorter,
blander* sentence than beam 1, because every extra word
multiplies in another probability below one. Length
normalisation is the standard patch, and it is visibly a
patch: dividing by length puts the six-word sentence back on
top without any principle saying it should be there.

None of this makes beam search wrong. It makes it right for a
different job — translation, where one input has roughly one
correct output and probability is genuinely what you want.
Part 3's sequence-to-sequence model was exactly that job. It
is wrong for open-ended generation, where the output wanted is
a sample and the mode is not one.

### The metric problem

Now the harder half. Suppose you want to score generated text.
The prompt `the king` has **92 valid completions** in this
language, and because the language is enumerable, that is not
an estimate — it is the complete set.

A real benchmark shows the scorer one of them. Sometimes four.
So the metric itself can be put on trial here, which is rarely
possible: pair every valid sentence with a twin broken by
exactly **one token**, confirmed outside the language by the
same oracle that defines it, and score both against references
drawn from the other valid completions.

![[METRIC TABLE IMAGE]]

Against a single reference a **perfect** generator scores
0.4899. It is not wrong. It is differently right, and BLEU has
no way to tell those apart.

The second column is the serious one. The sentence the oracle
*rejects* scores 0.4178 and matches or beats its valid twin
**55.07% of the time**. On one reference, BLEU ranks broken
output at or above correct output more often than not.

More references help — at sixteen, the broken sentence wins
only 16.67% of the time. But sixteen references per example is
not what benchmarks have, and the deficit never closes: even
there, a perfect generator scores 0.7921.

Which breaks BLEU catches is the mechanism, not a mystery:

![[BREAK TABLE IMAGE]]

The possessive break is caught every single time. The topic
break is caught almost never — 93.48% of the time the
sentence that mixes a sea noun into a royal sentence scores at
or above a valid alternative. The reason is pure n-gram
bookkeeping. Changing `his` to `her` destroys a bigram the
reference shares; swapping the object for a word from another
topic replaces a word the reference did not contain either, so
nothing is lost that was ever counted.

Here is the cleanest case. Against one fixed reference,
**157 of 273 pairs score exactly the same** for the valid and
the broken sentence:

```
reference : the king commands his lord
valid     : the king commands his banner    BLEU 0.7521
broken    : the king commands his captain   BLEU 0.7521
```

Both differ from the reference in exactly one word, so the
n-gram counts are identical and so are the scores. BLEU is not
malfunctioning. It is counting overlapping word strings, which
is what it has always said it does. The failure is in reading
that count as a measure of whether the output is any good.

### What a sample cannot see

One more reason the exact computation was worth the trouble.

![[PROXY TABLE IMAGE]]

The usual diversity proxy is the share of samples that come
out distinct. Read along any row: the number falls as you draw
more samples, because distinctness is a birthday problem and
not a property of the decoder. At 100 samples, temperature 1.0
and nucleus 0.90 are **indistinguishable** — both 0.9800 —
while their effective sentence counts differ by a factor of
1.6. The proxy goes blind exactly where the decision between
two strategies gets made.

The effective sentence count does not move, because it is a
property of the distribution rather than of the experiment.
That is the whole argument for building a universe you can
enumerate: the quantities that actually decide a decoding
strategy — how much of the language it can still reach, how
far it has moved from the truth — are not estimable from any
feasible number of samples. Here they are arithmetic.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Compare the two middle rows. Sampling generates a sentence in
`n` forward passes. Beam search costs `B` times that, for an
output that is less representative, not more — you pay
twenty-five times over to move further from a typical sample.
Every truncation rule is free by comparison: a sort or a
partial sort over the vocabulary, dwarfed by the forward pass
that produced the logits it is sorting.

The last row is the one that does not generalise. Computing a
strategy's exact distribution costs a pass over every sentence
in the language, which is possible here and impossible
anywhere real — that is the price of the yardstick, and it is
why the rest of the field estimates.

Which is worth stating plainly: the decoding strategy is the
cheapest part of the system and one of the most consequential.
It is also the part most often chosen by copying a default out
of someone else's configuration file.

---

## Where NLG sits

BLEU is from 2002, built at IBM for machine translation, where
its central assumption — that there is essentially one right
answer and overlap with it measures quality — was defensible.
It then spread to every text generation task, including the
ones where that assumption is simply false.

Beam search arrived in neural NLP through translation too, in
the sequence-to-sequence models of 2014. For a decade it was
*the* decoding algorithm, and generated text was known to be
strangely repetitive and flat without anyone quite naming why.
Fan et al. introduced top-k sampling in 2018. Holtzman et al.
named the problem in 2019 — neural text degeneration — and
showed that the flatness came from searching for high
probability at all, not from any defect in the models. Nucleus
sampling is their fix, and it is still the default in most
systems you will use today. The entropy argument in this
article is the one Meister et al. later made formally under
the name typical decoding.

The lineage in this series is direct. Part 5 built the decoder
and measured its distribution against the language's; this
article leaves that distribution untouched and changes only
how it is read, which turns out to matter as much as anything
pre-training did. Part 3's beam-search-shaped task — one
input, one correct output — is exactly the job beam search is
still right for, and exactly the job this one is not.

---

## What comes next

Part 8, **Retrieval-Augmented Generation**, gives the
generator something to generate *from*. The bi-encoder that
Part 6 ended on becomes the retrieval half, the decoder
becomes the generation half, and a new failure mode appears
that neither article could produce on its own: output that is
fluent, confident and contradicted by the document it was
supposedly reading.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**nlg.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/13-natural-language-processing/07-natural-language-generation/nlg.py)

Run it with:

```bash
pip install numpy
python nlg.py
```

It needs only `numpy` and runs in about seventy-five seconds.
Everything is from scratch: the causal transformer, the
truncation rules, batched sampling, beam search with length
normalisation, BLEU, and the exact closed-form analysis that
computes each strategy's distribution over all 31,740
sentences — all hand-differentiated and audited against
central differences at startup (worst error 1.76e-09 over 104
sampled gradients), with the oracle checked against the
generator it claims to describe and each truncation rule
checked against its own definition. The headline insight worth
pinning to the wall: **a decoding strategy is a distribution,
not a setting, and truncation does not improve a model but
trades away what it can say — top-k 2 reaches 99.97%
grammaticality by making 31,724 of 31,740 sentences impossible,
nucleus 0.90 quietly removes more than half the language, and
every setting that makes output look better moves it further
from the truth; meanwhile the metric most used to score the
result cannot distinguish a valid sentence from one broken by
a single word, scoring 157 of 273 such pairs exactly the
same**.

---

*This is Part 7 of the Natural Language Processing track in the Algorithms in Python series. The companion script `nlg.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It reuses Part 1's synthetic micro-language ([word embeddings](https://medium.com/p/164f7f542a3c), [BPE tokenisation](https://medium.com/p/5e6a51341455), [sequence-to-sequence](https://medium.com/p/bbdf65112819), [BERT-style encoders](https://medium.com/p/a0ba729fe954), [GPT-style decoders](https://medium.com/p/6da33883e00d), [natural language inference](https://medium.com/p/d4f9c54be0d4)). Part 8 will look at Retrieval-Augmented Generation.*
