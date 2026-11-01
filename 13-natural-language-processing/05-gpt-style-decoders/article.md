# GPT-style Decoders — Writing One Token at a Time

### *Algorithms in Python --- Natural Language Processing, Part 5*

---

Part 4 kept the encoder and threw the decoder away. This one
does the opposite, and the surprising part is how little
code changes hands.

In the companion script the encoder and the decoder are the
same class. Exactly two things differ:

```
    causal=True       one boolean on the attention mask
    (targets, sel)    what you hand the loss function
```

BERT hands that loss the original tokens at the 15% of
positions it corrupted. GPT hands it the same tokens shifted
one place to the left, at every position. Same trunk, same
output head, same loss function. The whole argument between
the two architectures is a flag and a slice.

And corruption turns out to be something only an encoder
needs. A bidirectional model must hide a word, or every
position would simply read its own answer off the input. The
causal mask already does that work, so a decoder trains on
the corpus untouched:

```
sentence: the thrones crown their queen
input:    [BOS] the thrones crown their queen [EOS] [PAD]
target:   the thrones crown their queen [EOS] . .
```

The dots are positions with no target: once `[EOS]` is
predicted there is nothing left to guess.

Every position carries a target, including the one that
predicts `[EOS]` — which is how the model learns where a
sentence stops. On the 8,000-sentence corpus that works out
at **6.49 targets per sequence against masking's 1.24** —
over the whole pre-training run, 5.25 times the gradient
signal. Masked language modelling throws most of each
forward pass away.

The stage is Part 1's synthetic micro-language again, and it
has one property this article leans on hard: it is small
enough to **enumerate**.

---

## A worked example: the shift, the floor, the sampler

### A loss with a floor you can compute

Run the generator over every choice it could make and you
get all 31,740 sentences the language can produce, with
their exact probabilities, summing to 1. So its entropy is
not estimated from a sample — it is computed. **9.2538 nats
per sentence**, 1.4190 per token.

That changes what a loss curve means. Ordinarily a loss of
1.47 is a number you can only compare to another run. Here
there is a floor, and we know where it is:

![[FLOOR CURVE IMAGE]]

The decoder lands at 1.4706 nats per token against an
irreducible 1.4190 — **3.6% above the floor**, a perplexity
of 4.352 where 4.133 is the best any model could do. Not
*approximately* the best; the best, provably.

No masked model gets this yardstick. A causal model's
sentence probability is a factorisation — a product of
next-token probabilities that sums to one over sequences —
so it can be compared with the thing that produced the text.
Masked prediction is not a factorisation of anything, which
is why Part 4's loss curve could only be read against
another loss curve.

The comparison also works slot by slot, because the
conditional entropy at each position is computable too:

![[FLOOR TABLE IMAGE]]

Read the floor column first. Three slots are deterministic —
every sentence opens with `the`, the possessive is fixed the
moment the subject is chosen, and the length is fixed by the
template — and the model drives all three to within 0.0013
nats of zero. The verb is a uniform choice among four, so
its floor is exactly `ln 4 = 1.3863`, and the model scores
1.4334.

The object is the widest gap at +0.2011, and that is Part 1's
doing: half the time the object is the subject's signature
partner, and a role signature is a harder thing to learn
than a rule. Remember that row — it comes back.

One more detail worth a glance. The vocabulary contains four
structural tokens the model should never emit mid-sentence,
and the total probability it assigns to `[PAD]`, `[BOS]` and
`[MASK]` as a next token averages **2.20e-05**. Nobody told
it those were off limits. It learned which tokens are
grammar and which are plumbing.

### What the flood buys, and what it costs

Same trunk, same 1,500 steps, same corpus, 5.25x the
targets. The obvious prediction is that the decoder learns
everything faster. It does not, and the way it fails to is
the interesting part.

Two yardsticks, both label-free apart from a single
threshold fitted on 20 labels. The first is verb-number
agreement: does the subject govern the verb? The evidence
sits to the left, so both models can reach it. The second is
whole-sentence acceptability, scored by whatever label-free
scorer each model naturally has.

![[EFFICIENCY CURVE IMAGE]]

On the verb they are level. Both reach a perfect score at
step 100, and the encoder is briefly *ahead* at step 75.
Five times the gradient does not buy the agreement rule any
sooner, because the rule is a single local fact and both
objectives put it in front of the model constantly.

On whole sentences the gap is enormous. The decoder passes
95% at step 50, where the encoder is still at chance; the
encoder does not get there until step 300. What the flood
buys is not knowledge, it is **calibration** — a probability
for the whole sentence rather than a good guess at one
position.

Those scores also cost different amounts. The decoder's
probabilities factorise, so one forward pass gives the whole
sentence. Part 4's encoder had to blank one position at a
time and add up the results: six passes for these sentences,
`O(n)` in general.

Now the bill. The causal mask cannot be made free:

![[RECEIPT TABLE IMAGE]]

Those are not bad scores, they are base rates. The subject's
number comes out at 0.6670, which is precisely the rate of
guessing *singular* every time. Subject topic comes out at
0.6650, and half these sentences carry an adjective that
gives the topic away while the other half start bare — so
the ceiling on a left-to-right guess is 0.5 + 0.5/3 =
0.6667. The decoder is not looking in the wrong place. There
is no place to look, exactly as Part 4's causal encoder
found.

So pre-training buys the decoder a calibrated score and
costs it everything to the right. What does that mean for an
actual labelled task? Here is the same acceptability task
Part 4 used, by five routes:

![[LABELS TABLE IMAGE]]

Two results that want separating. **Fine-tuned, the encoder
wins at every label count** — 0.8200 against 0.6267 at 100
labels. That is not a quirk of this stage, it is the
history: BERT owned the fine-tuning era, and a bidirectional
reader with a fresh head on top really was the better way to
spend a few hundred labels.

**Scored with no new parameters at all, the decoder wins.**
Twenty labels spent fitting one threshold over its one-pass
sentence score reach 0.9900 — beating four hundred labels
spent fine-tuning the encoder, and beating its own
fine-tuned self by a mile. The randomly-initialised head is
the bottleneck, which is Part 4's pseudo-log-likelihood
result turned up: there, skipping the head got you 90.5%,
and here it gets you 99.0% for a sixth of the compute.

That is the whole pre-train-and-prompt idea in miniature.
The knowledge was in the model; the head was in the way.

### Generation, which an encoder cannot do

An encoder scores text. A decoder writes it: sample a token,
append it, run the model again. And because the language is
enumerable we do not have to squint at the output and call
it fluent — the oracle says whether a sentence is *in* the
language, and if not, which rule broke.

![[TEMPERATURE TABLE IMAGE]]

Temperature 0 is the trap. Every sample is grammatical and
they are all the same sentence: one distinct string in a
thousand. Greedy decoding is not careful, it is stuck. Push
the other way and grammar dissolves in a readable order:
slot order goes first and stays the biggest failure, topic
mixing second, and the agreement rules are the last thing
left standing.

Nothing in the loss curve says where the useful band is.
Only sampling finds it, and here it is 0.5 to 1.0.

At temperature 1.0 the samples look like this:

```
the fertile orchard ploughs the seed
the ancient ladies crown their banner
the salty compass sails the wave
the wooden tractors plough their fields
the bull grazes his tractor
```

Grammatical is the low bar, though, and sampling only
estimates it. We can do better: the model assigns a
probability to every one of the 31,740 sentences, so add
them up and the question is answered exactly.

![[DISTRIBUTION TABLE IMAGE]]

The model spends **0.9770** of its probability on real
sentences — and sampling at temperature 1.0 said 0.9780,
which is the check that the sampler really is drawing from
the model. The KL divergence from the language to the model
is 0.2987 nats. That is the same quantity the held-out test
set estimated at 0.3358; computed rather than sampled, the
test set was a touch pessimistic.

The marginals are exact, so their deviations are real rather
than noise — and they land where the floor table said they
would. The model under-generates six-word sentences by 4.2
points and the signature partner by 2.4, and those are
precisely the two slots with the widest loss gap. It learned
the grammar almost exactly and the frequencies only
approximately.

Two limits worth naming. The vocabulary is a hard wall: the
plurals Part 1 held out — `princesses`, `roosters`,
`compasses` — are not tokens, so this model cannot utter
them however well it understands plurals. That is the
problem Part 2's subword pieces exist to solve.

And generation is sequential where training was parallel.
One training pass scored every position of a sentence at
once; writing one takes a pass per token, each re-reading
the whole prefix — 21 position-reads for a sentence that
training handled in one pass over 6. Caching the keys and
values is the standard repair.

Prompted, the same machine continues instead of inventing:

```
the noble -> the noble castle commands the queen
the salty captains -> the salty captains anchor their tide
the muddy hen -> the muddy hen feeds her seeds
the kings -> the kings command their queen
```

Note the agreement it supplies unprompted: a plural subject
takes a plural verb and `their`, and `hen` takes `her`.
Nobody labelled that. It is simply what predicting the next
token requires.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The forward pass is the same arithmetic as Part 4's encoder
— `O(n² · d)` for attention, `O(n · d²)` for the
feed-forward block — because the causal mask overwrites
scores that were computed either way. Masking is free.

What differs is everything around it. **Signal per pass** is
`O(n)` rather than `O(0.15 · n)`, which is the 5.25x
measured above. **Scoring** inverts Part 4's cost: one
forward pass instead of `O(n)` masked ones.

And **generation** inverts the inversion. Training is
parallel: one pass handles every position of a sequence at
once, because the targets are all known in advance.
Inference is sequential: `n` passes, each one depending on
the token the last one produced. That asymmetry is the
central engineering fact of modern language models, and
every trick in a serving stack — KV caches, speculative
decoding, batching — exists to soften it.

---

## Where GPT sits

GPT (Radford et al., 2018) did not introduce the
transformer, the causal mask, or next-token training. Part 5
of the Deep Learning track built all three. What the GPT
line introduced was a bet: that one decoder, trained on
enough text, would not need a new head for each new task.

GPT-1 still fine-tuned, and on that ground BERT beat it
within months — which is exactly the column that wins in the
labels table above. GPT-2 (2019) dropped the fine-tuning and
asked for the task in the prompt. GPT-3 (2020) made that the
whole interface. The architecture barely moved across the
three; the bet was about what you do with the thing
afterwards, and the thing that made the bet payable was the
property measured here — a decoder's output is a calibrated
distribution over continuations, so a task can be posed as
*what comes next* rather than as a new classifier.

The lineage in this series is direct. Part 1 built static
word vectors. Part 2 decided what the tokens are, and set
the wall this model cannot speak past. Part 3 built the
encoder-decoder pair and named its halves. Part 4 took the
encoder, deleted the mask, and reconstructed. This article
takes the decoder, keeps the mask, and predicts — and gets,
in exchange for a blindness to its right, a machine that can
be scored against the truth and can write.

---

## What comes next

Part 6, **Natural Language Inference**, changes the shape of
the question. Everything so far has judged one sentence at a
time: is it likely, is it well formed, what comes next. NLI
asks what one sentence tells you about another — does the
second follow from the first, contradict it, or neither? It
is the task that made sentence *pairs* the unit of work, and
it is where the encoder and the decoder meet again.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**gpt_decoder.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/13-natural-language-processing/05-gpt-style-decoders/gpt_decoder.py)

Run it with:

```bash
pip install numpy
python gpt_decoder.py
```

It needs only `numpy` and runs in about three and a half
minutes. Everything is from scratch: multi-head causal
self-attention with a padding mask, LayerNorm, learned
positional embeddings, the shared output head driving both
the next-token and the masked objective, temperature
sampling, and the exact enumeration of the language —
all hand-differentiated and audited against central
differences at startup (worst error 1.77e-09 over 550
sampled gradients, with the enumeration itself checked
against the generator on 6,000 strings). The headline
insight worth pinning to the wall: **a causal model's loss
is a calibrated quantity in a way a masked model's never is
— this decoder lands 3.6% above the computed entropy of its
own universe and spends 97.70% of its probability on real
sentences, both exact rather than estimated; the five times
more gradient it gets does not buy the agreement rule any
sooner than masking does, it buys a sentence-level score,
which is why the decoder passes 95% at step 50 where the
encoder needs 300; and the route to a labelled task matters
more than the objective, because fine-tuned the encoder wins
at every label count, while scored with no new parameters at
all, twenty labels spent on a single threshold beat four
hundred spent fine-tuning the encoder**.

---

*This is Part 5 of the Natural Language Processing track in the Algorithms in Python series. The companion script `gpt_decoder.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It reuses Part 1's synthetic micro-language ([word embeddings](https://medium.com/p/164f7f542a3c), [BPE tokenisation](https://medium.com/p/5e6a51341455), [sequence-to-sequence](https://medium.com/p/bbdf65112819), [BERT-style encoders](https://medium.com/p/a0ba729fe954)). Part 6 will look at Natural Language Inference.*
