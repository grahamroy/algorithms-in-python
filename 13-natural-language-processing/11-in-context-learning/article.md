# In-Context Learning — Behaviour Without a Gradient

### *Algorithms in Python --- Natural Language Processing, Part 11*

---

Every article in this track has changed a model by changing
its weights. Pre-training, fine-tuning, adapters, policy
gradients — different objectives, same mechanism. Part 9 went
as far as taking the SVD of the update to see exactly which
weights moved.

This one does not touch them.

A trained model is shown a few examples of a rule it has
never seen, in its context, and asked to apply it. No
gradient step happens. The weights afterwards are bit-for-bit
the weights before. And the behaviour changes.

That is in-context learning, and the first question is
whether a model this small can do it at all.

---

## A worked example: a rule that cannot be memorised

### The stage

A sequence is a few demonstrations, then a query:

```
tide grazes anchor
banner crowns bull
prince navigates prince
tide navigates ->        anchor
```

The subject-to-object pairing is a **fresh random permutation
of the 36 nouns in every sequence**. It is not a fact about
the language, and it is not in the weights. A model that
answers correctly has read it out of its own prompt.

Guessing scores 0.0278. The script asserts both halves of
that at startup: the answer *is* in the context whenever a
supporting demonstration exists, and provably *is not* when
none does — so chance really is the ceiling for a model that
ignores its prompt.

Then two models are trained, with the same trunk, the same
108,403 parameters, the same budget and the same seeds. The
only difference is how their **training** sequences were
built:

```
bursty  one pairing per sequence, redrawn between them
IID     every object independent; context says nothing
```

### It works — and only one of them can do it

![[ICL TABLE IMAGE]]

With a single demonstration the bursty model is right
**1.0000** of the time, with a standard deviation across
seeds of **0.0000**. The IID model scores 0.0275, which is
chance to three decimals.

Same architecture. Same number of gradient steps. Same
initialisation. **The capability is a property of the
distribution the model was trained on, not of the model.**

That is the third time this section has landed on the same
shape. Part 8 found that grounding is learned rather than
inherited. Part 10 found that forgetting is governed by the
learning rate rather than the method. Here the most
celebrated emergent capability in the field turns out to be
something you put in the training data, or fail to.

### What "number of examples" was hiding

Read down that table, though, and accuracy *falls*: 1.0000 at
one demonstration, 0.1308 at eight. Which looks like
in-context learning getting worse as you add examples.

It is not. In that table only one demonstration ever shared
the query's subject — every other was noise. Adding
demonstrations added **distractors**, not evidence. Holding
the context at eight and varying how many demonstrations
actually support the query gives the opposite curve:

![[AXES TABLE IMAGE]]

So the model copies, reliably, and what degrades as the
prompt grows is not the copying but the **finding**. A longer
prompt is only a better prompt if what you added was
evidence.

This is worth separating carefully because the two are
routinely conflated in practice. "More examples in the
prompt" is not one knob. It is a ratio, and the denominator
is everything you added that did not bear on the question.

### The mechanism, read off the attention map

If the model is copying, the query position has to be
attending to the object token that followed the matching
demonstration. That is the only place in the whole sequence
where the answer is written down.

![[ATTENTION TABLE IMAGE]]

The bursty model puts **0.6872** of its attention on exactly
that token. The IID model, handed the identical prompt, sits
at 0.1407 — it is not looking anywhere in particular, because
nothing in its training ever made looking worthwhile.

Two layers is the minimum a circuit like this needs: one to
mark what preceded each token, one to match the query against
those marks and copy what follows. This model has exactly
two, which is why it can do this at all and also why it
cannot do it in a crowded context.

Part 4 made the case that attention maps are readable
receipts rather than decoration. This is that thread's last
appearance, and the clearest: a capability that looks like
inference, and is a lookup you can point at.

### It arrives all at once

![[EMERGENCE CURVE IMAGE]]

The curve is the reason this measurement was worth doing
properly. For the first thirty steps the bursty model sits at
chance — 0.0350, 0.0683, 0.0317, 0.0600, 0.0583. At step 30
it is at 0.0983, barely above the 0.0278 it would score by
guessing.

Fifty steps later it is at **0.9933**.

Nothing about the data changes across that window. Nothing
about the objective changes. The loss was falling smoothly
the whole time. What happened is that a circuit finished
assembling, and the behaviour it supports went from absent to
complete in about fifty gradient steps. The IID model, probed
at the same points, never moves off chance at all.

An aside on how that plot nearly did not exist. My first
version probed every hundred steps — and the model was
already at 1.0000 at the first probe, so the transition
happened entirely inside the gap between step 0 and step 100
and I never saw it. The narration I had written asserted a
sharp onset anyway. It was right, as it turns out, which is
not the same as being supported.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The row that explains the shape of the industry is the first
one. In-context learning costs **no training at all** — the
weights are read-only — but it pays on every single forward
pass, because the demonstrations sit in the context and
attention is quadratic in context length. Fine-tuning
inverts that: pay once, then nothing.

Which is the real trade. Prompting is not cheaper than
fine-tuning; it is cheaper *to start* and more expensive
*per query*, and the crossover depends entirely on how many
times you intend to ask.

---

## Where this sits

In-context learning was not designed. It was noticed. GPT-3's
authors reported in 2020 that a sufficiently large language
model, given a few examples in its prompt, would perform
tasks it was never trained on — and the capability had not
been an objective of the training.

Explaining it took longer. Olsson et al. identified induction
heads in 2022: a two-layer attention circuit that matches the
current token against earlier occurrences and copies what
followed, forming abruptly during training at the same moment
in-context learning appears. Chan et al., the same year,
showed the capability depends on properties of the training
*distribution* — burstiness and a long tail — rather than on
scale alone. This article is their experiment in miniature,
and the attention measurement above is the induction circuit
being caught in the act.

One finding deliberately not reproduced here: Min et al.
showed in 2022 that replacing the labels in a prompt with
random ones barely hurts performance on many tasks, implying
the demonstrations were teaching *format* rather than
content. That cannot happen in this stage, because the
mapping is genuinely novel and genuinely needed. Both things
are true of real prompts, in proportions that vary by task,
and telling them apart requires exactly the kind of control
this universe makes cheap and real benchmarks make hard.

---

## What comes next

This closes the Natural Language Processing track. Eleven
parts ago it began with a synthetic micro-language and a
count of word co-occurrences; it ends with a model reading a
rule out of its own prompt. The same 31,740-sentence universe
ran underneath all of it, which is what made every claim
checkable: an oracle that knows the true distribution, the
true label, the true reward.

Section 14 turns to **Computer Vision**, where the inputs
stop being discrete and the first thing that has to be
rebuilt is what a neighbourhood means.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**icl.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/13-natural-language-processing/11-in-context-learning/icl.py)

Run it with:

```bash
pip install numpy
python icl.py
```

It needs only `numpy` and runs in about seventeen minutes.
Everything is from scratch: the bursty and IID sequence
generators, the loss taken on the final object alone, the
attention extraction that reads the copying head directly out
of the forward pass, and the probe schedule that catches the
transition — all hand-differentiated and audited against
central differences at startup (worst error 7.50e-10 over 78
sampled gradients), with a self-test that proves the task
cannot be solved without reading the context. The headline
insight worth pinning to the wall: **two models with identical
architecture, parameter count, budget and seeds differ only
in whether their training sequences were bursty, and that
alone decides whether in-context learning exists at all —
1.0000 against a chance rate of 0.0278, with a standard
deviation across seeds of 0.0000 — and the capability does
not fade in, it assembles, going from 0.0983 at step 30 to
0.9933 at step 80 while nothing about the data or the
objective changes**.

---

*This is Part 11 of the Natural Language Processing track in the Algorithms in Python series. The companion script `icl.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It reuses Part 1's synthetic micro-language ([word embeddings](https://medium.com/p/164f7f542a3c), [BPE tokenisation](https://medium.com/p/5e6a51341455), [sequence-to-sequence](https://medium.com/p/bbdf65112819), [BERT-style encoders](https://medium.com/p/a0ba729fe954), [GPT-style decoders](https://medium.com/p/6da33883e00d), [natural language inference](https://medium.com/p/d4f9c54be0d4), [natural language generation](https://medium.com/p/48a45ec5b396), [retrieval-augmented generation](https://medium.com/p/0686445e3d97), [fine-tuning and LoRA](https://medium.com/p/048d50f039aa), [RLHF and DPO](https://medium.com/p/ef2a4ca1e10c)). Section 14 will turn to Computer Vision.*
