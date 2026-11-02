# Natural Language Inference — When the Label Is in the Hypothesis

### *Algorithms in Python --- Natural Language Processing, Part 6*

---

Every article in this track so far has judged one sentence at
a time. Is it likely, is it well formed, what comes next.
This one takes two sentences and asks what the first tells
you about the second:

```
premise:    the ancient throne commands the banner
hypothesis: the throne does not command the banner
label:      contradiction
```

Three answers are allowed — the hypothesis **follows** from
the premise, **contradicts** it, or neither, which is called
**neutral**. That is natural language inference, and it is
the task that made the sentence *pair* the unit of work.

It is also the task that taught the field to distrust its own
benchmarks, and that is where this article spends most of its
time.

The micro-language from Part 1 gets a semantics so the labels
can be computed rather than collected. A sentence now asserts
a polarised relation plus whatever adjectives it carries:

```
the noble king rules his crown
  -> rule(king, crown) and noble(king)
the king does not rule his crown
  -> NOT rule(king, crown)
```

Entailment is set inclusion over those facts; contradiction
is the same relation with the polarity flipped; everything
else is neutral. Negation is new — `does not` and `do not` —
and it is the only new machinery the stage needs.

---

## A worked example: a function, two readers, and a tell

### The label is a function, not an opinion

Because the label is computed, the task can be taken apart
and each piece priced. Here is what you score with only part
of the rule, on 2,000 held-out pairs:

![[RULE TABLE IMAGE]]

Relation matching — find the premise clause about the same
subject, verb and object, then compare polarity — gets
94.90%. The remaining five points are the adjective check:
whether the hypothesis asserted something the premise never
did. Guessing the majority class gets 34.35%.

Keep the third row in mind. **Hypothesis only** scores
32.95%, which is chance. That is not an accident; it is a
property the generator was built to have, and the rest of
this article is about what happens when a dataset does not
have it.

### Two ways to read a pair

There are two ways to put a pair through a transformer, and
the choice turns out to matter more than anything else here.

A **cross-encoder** reads them joined together —
`[CLS] premise [SEP] hypothesis [SEP]` — in one pass, so any
hypothesis token may attend to any premise token. A third
embedding table, alongside tokens and positions, marks which
side each token is on. This is BERT's recipe.

A **bi-encoder** encodes each sentence alone, pools each to a
vector, and compares the two with `[u; v; |u-v|; u*v]`. The
two sentences never share a forward pass. This is InferSent's
recipe, and SBERT's after it.

Same trunk, same 8,000 labels, same budget:

![[READERS TABLE IMAGE]]

The cross-encoder trained from scratch is the interesting
failure. It scores 65.30%, and the ceiling for a model that
gets entailment and contradiction right while abandoning
neutral entirely is 66.7%. Its neutral recall is 9.6%. It
found the cheap half of the task — *is there a negation
mismatch* — and stopped looking.

That is the same shortcut-taking this article is about,
appearing one level down. The model is not lazy; it is doing
what the gradient pays for.

Now read the two pre-training rows. **Pre-training is worth
+30.0 points to the cross-encoder and +0.9 to the
bi-encoder.** That asymmetry is the whole result. The
bi-encoder's comparison is wired into its architecture — the
subtraction in `|u-v|` is not learned, it is soldered on, so
there is almost nothing for pre-training to add. The
cross-encoder's comparison has to be *discovered* inside the
attention pattern, and 8,000 labels are nowhere near enough
to discover it.

What pre-training installs can be measured on its own. Blank
a token in the hypothesis and ask the encoder to recover it,
once with the premise visible and once with it hidden:

![[ALIGNMENT IMAGE]]

The premise is worth **2.2156 nats**. That gap is
cross-attention doing its job, and it is the one thing a
bi-encoder can never have, because its two sentences never
meet.

An honesty note, because the ordering here is not the
published one. In real NLI leaderboards cross-encoders beat
bi-encoders; in this article the bi-encoder still wins,
98.50% to 95.30%. Both facts point the same way. Joint
attention is a capability you *pay for*, the bill is paid in
pre-training, and a budget that fits in a twelve-minute
script buys most of the gap but not the crossover. The
measured +30.0 is the shape of that bill.

### The artefact

SNLI, the dataset that started modern NLI, was built by
showing annotators a photo caption and asking for three
sentences: one that must be true, one that must be false,
and one that might be true. Hundreds of thousands of times.

People took the shortest path. For *must be false* they
negated. For *must be true* they deleted a detail. For *might
be true* they added one. Nobody was being careless on
purpose; it is simply what the task rewards at speed.

So this article ships two generators. The **careless** one
does exactly that. The **balanced** one puts a negation in
half the hypotheses of every class and an adjective in half
of them, and matches the lengths:

![[ARTEFACT TABLE IMAGE]]

Look at the careless block. Negation appears in 100% of
contradictions and 0% of everything else. An adjective
appears in 100% of neutrals and nowhere else. Mean
hypothesis length runs 5.0, 6.0, 7.0 — one word per class.
The label is written on the hypothesis three separate ways.

Now train a model that is shown **only the hypothesis**. The
premise is not summarised or truncated; it is removed:

![[TRANSFER TABLE IMAGE]]

**100.00%** on a three-way entailment task, from a model
that has never seen a premise. It has learned nothing
whatsoever about entailment. It has learned how the data was
made.

Then move that same model to the balanced data and it falls
to 33.00%, against a chance rate of 33.3%. Everything it
knew was the tell.

This is not a hypothetical. Gururangan et al. and Poliak et
al. independently found in 2018 that a hypothesis-only model
gets around 67% on SNLI — against 34% chance — and the
leaderboard had been running for three years by then. Every
number published on that benchmark included an unknown amount
of this.

It is also why the gold labels in this article are computed
by a function. A generator can be audited line by line. A
pool of annotators cannot.

### The label as a token

Part 5's decoder answered questions by predicting the next
word. Nothing stops it answering this one the same way: read
the pair, then predict `yes`, `no` or `maybe`.

```
[CLS] the ancient throne commands the banner [SEP]
the throne does not command the banner [SEP] no
```

No classification head anywhere — the answer comes out of the
same vocabulary the text did. Given the same trunk, the same
pre-training budget and the same labels:

![[DECODER TABLE IMAGE]]

The decoder reaches 66.45% against the cross-encoder's
95.30%, and almost all of the difference is neutral, where
its recall is 24.0%. The answer's *format* is not the
problem: the position that predicts the label sits at the
very end and can see the whole pair.

What it lacks is the other direction. In a bidirectional
encoder, every premise token is re-read once the hypothesis
arrives — `throne` in the premise gets to notice that
`throne` appears again on the other side of the `[SEP]`.
Causality forbids exactly that. The premise is encoded before
the hypothesis exists, and comparison is a two-way operation.

That is Part 4's lesson arriving somewhere new. There, the
evidence that a causal model could not reach lay to the right
within one sentence. Here it lies in the *other sentence*,
and it is the premise, not the hypothesis, that needed to see
ahead.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The row that decides architectures in practice is the last
one. A bi-encoder's sentence vectors do not depend on what
they are compared against, so they can be computed once and
cached; scoring a new hypothesis against `N` stored premises
is `N` dot products. A cross-encoder has to run the whole
transformer `N` times, because its representation of the
premise depends on the hypothesis sitting next to it.

For a leaderboard with 10,000 test pairs that difference is
an inconvenience. For search over ten million documents it is
the difference between a product and a research demo — which
is why the retrieval half of the next few articles is built
on bi-encoders, and why anything that needs real accuracy
re-ranks the top fifty with a cross-encoder afterwards.

---

## Where NLI sits

NLI is old — it was the Recognising Textual Entailment
challenge in 2005, with a few thousand hand-built pairs.
SNLI (Bowman et al., 2015) made it a deep-learning task by
making it large: 570,000 pairs, enough to train on. MultiNLI
added genres. For about three years it was the benchmark the
field steered by, and GLUE put it at the centre.

Then came the artefact papers, and the lesson generalised
well beyond NLI: a benchmark measures the dataset as much as
the model, and the only way to know which is to run the
ablation that removes the input. Hypothesis-only here;
image-only in visual question answering; question-only in
reading comprehension. The ablation is cheap and almost
nobody ran it for three years.

The lineage in this series is direct. Part 4 built the
encoder and showed what pre-training buys a single sentence;
this article shows the same purchase being made for a *pair*,
and the price tag is larger. Part 5 built the decoder and
scored sentences with no head at all; this article asks it a
three-way question and finds the one shape of evidence its
mask forbids.

---

## What comes next

Part 7, **Natural Language Generation**, turns from judging
text to producing it, and runs into a problem this article
did not have: no single right answer. Part 5 deliberately
stopped at temperature sampling and left top-k, nucleus
sampling and beam search alone. They belong here, along with
the harder question of how you score a generated sentence
when the gold standard is a set rather than a string.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**nli.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/13-natural-language-processing/06-natural-language-inference/nli.py)

Run it with:

```bash
pip install numpy
python nli.py
```

It needs only `numpy`. It runs in about twelve minutes, which
is the longest script in this series, because pre-training is
the thing being measured and there is no way to measure it
quickly — set `FAST = True` at the top of the file for a
three-minute run with the same shape and weaker numbers.
Everything is from scratch: segment embeddings, the
cross-encoder, the bi-encoder with its `[u; v; |u-v|; u*v]`
comparison, the hypothesis-only ablation, masked and
next-token pre-training, and the label-as-a-token decoder —
all hand-differentiated and audited against central
differences at startup (worst error 7.78e-10 over 372 sampled
gradients, with the semantics checked against both generators
on all 20,000 pairs). The headline insight worth pinning to
the wall: **a model shown only the hypothesis and never the
premise scores 100.00% on three-way entailment when the data
is built the way SNLI was built, and 33.00% — chance — once
the negations and lengths are balanced across classes, which
means a benchmark number is a statement about a dataset
before it is a statement about a model; and the architecture
that can in principle read both sentences together is the one
that cannot do it without help, because pre-training is worth
+30.0 points to a cross-encoder and +0.9 to a bi-encoder
whose comparison is soldered in rather than learned**.

---

*This is Part 6 of the Natural Language Processing track in the Algorithms in Python series. The companion script `nli.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It reuses Part 1's synthetic micro-language ([word embeddings](https://medium.com/p/164f7f542a3c), [BPE tokenisation](https://medium.com/p/5e6a51341455), [sequence-to-sequence](https://medium.com/p/bbdf65112819), [BERT-style encoders](https://medium.com/p/a0ba729fe954), [GPT-style decoders](https://medium.com/p/6da33883e00d)). Part 7 will look at Natural Language Generation.*
