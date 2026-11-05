# Fine-tuning and LoRA — The Update Is Not Low Rank

### *Algorithms in Python --- Natural Language Processing, Part 9*

---

Part 8 put the knowledge beside the model. A retriever found a
document, a generator read it, and nothing in the weights
had to change to add a fact.

It also left a bruise. The fine-tuning that taught the
generator to defer to documents dropped its closed-book
accuracy from 0.6111 to 0.0833. Something was written over,
and the article could only watch it happen.

This one asks what, exactly — and whether all of it needed to
move. The fine-tune is Part 8's, unchanged. What is new is
that the weights here are small enough to look at directly:

```
dW = W_after - W_before
```

That matrix is the entire record of what fine-tuning did. In
a real model it is far too large to examine. In this one you
can take its SVD.

Which matters, because the most popular idea in fine-tuning
rests on a claim about it.

---

## A worked example: reading an update

### LoRA's premise, and how to check it

LoRA freezes `W` and learns a low-rank correction instead:

```
W_eff = W + (alpha / r) * A @ B
```

with `A` random, `B` zero, so the adapted model starts exactly
where the base model stopped. Only `A` and `B` ever receive a
gradient. The premise is that this is enough because `dW` has
low intrinsic rank — that fine-tuning, despite touching every
weight, really only moves the model in a handful of
directions.

That premise is normally argued from results. Here it can be
decomposed.

![[SPECTRUM TABLE IMAGE]]

`r90%` is the rank a truncated SVD needs in order to keep 90%
of the update's energy. Out of 64 available directions it
runs into the twenties almost everywhere. Rank 8 — a
perfectly ordinary LoRA setting — captures between 52% and
70% of what full fine-tuning actually did, the output head
aside.

On the strong reading of the claim, that `dW` **is** low rank,
the answer here is plainly no.

One row stands apart. The output head `Wv` moved furthest in
relative terms, 0.4461 against 0.09 to 0.19 for the attention
matrices, and **46.23% of its change sits in a single
direction**. That is the fine-tune learning which tokens to
emit in the answer slot, and it is nearly a rank-one edit.
Remember that row; it sets a trap two sections from now.

### Two claims that sound like one

Here is the distinction the spectrum forces:

> **The update is low rank** — false here.
> **A low-rank update suffices** — a different claim entirely.

They sound alike and they are not. LoRA never tries to
reproduce `dW`. It is not compressing the full fine-tune; it
is solving the same problem from scratch inside a smaller
space. Whether that space is big enough is an empirical
question that the SVD of someone else's solution does not
answer.

So: does it suffice?

![[PLACEMENT TABLE IMAGE]]

A warning before reading the order. The same configuration
varies by up to eleven points across initialisations, because
`B` starts at zero and where `A` happens to point decides
which subspace the adapter gets to search. **Every LoRA
figure in this article is a mean over four seeds**, with the
spread printed beside it. The standard error on each mean is
about 0.013, so differences smaller than roughly 0.04 are not
differences.

With that said, **where** the adapters go survives the test
easily. Adapting everything scores 0.5417 against attention
alone at 0.4549 — and both bury the head on its own at
0.2569.

Which is the trap. The head is the matrix that moved
*furthest*, and it is the one place where `dW` really is
close to rank one. Adapting it alone is still the worst thing
you can do. The spectrum tells you where the change
**happened**; it does not tell you where the change had to be
**made**. A fine-tune rewrites the output head because the
layers beneath it have already done the work of representing
the answer — give the adapter only the head and there is
nothing underneath for it to read.

### How much rank

![[RANK TABLE IMAGE]]

The column rises to rank 8 and stops. Rank 16 scores 0.5660
against rank 8's 0.5903 on twice the parameters — a gap
inside the standard error, so the honest reading is that more
rank stops helping, not that it hurts.

**Rank 8 reaches 98.84% of full fine-tuning on 15.45% of the
parameters.** Rank 2 gets 87.21% of it for 3.86%. That is the
trade, and it is a good one, even though full fine-tuning
stays ahead throughout.

Note what the spectrum did and did not predict. It said rank
8 would capture barely half the update — and rank 8 recovers
almost all of the *accuracy*. Those are consistent, and the
gap between them is the whole point: the adapter is not
approximating the update, so the energy it would have
captured was never the right thing to measure.

### The learning rate, and a piece of advice that fails here

Common practice is to raise the learning rate for LoRA, on
the reasoning that a factorisation starting from zero has
further to travel. Measured over four seeds:

```
lr 5e-04:  0.5417   (sd 0.0260)
lr 2e-03:  0.5035   (sd 0.0690)
```

No better in the mean, and **2.7 times the spread**. At this
scale the larger step buys variance, not accuracy.

This is worth dwelling on because of how it nearly got
written up the other way. Here are the four seeds behind
those two means:

```
lr 5e-04 : 0.5694 0.5000 0.5417 0.5556
lr 2e-03 : 0.5972 0.4028 0.5139 0.5000
```

A single run at the larger step could have reported 0.5972 —
matching full fine-tuning, the best result anywhere in this
article — or 0.4028, which is nearly the worst. A single run
is what most such comparisons are, and the first seed I drew
happened to be the lucky one.

### What it costs the model underneath

Part 8 could only watch closed-book accuracy fall. Here the
language enumerates, so the damage is exact: held-out nats
per token against the oracle's floor of 1.4310, and
`KL(language ‖ model)` summed over all 31,740 sentences.

![[FORGETTING TABLE IMAGE]]

Read the `nats` column down. The learning rate moves it from
about 2.3 to about 3.9. The *method* moves it by 0.009.

**Forgetting here is a property of how far you stepped, not
of which parameters you stepped in.** The folk claim is that
LoRA forgets less because it writes to fewer weights. On this
measurement it does not forget less — it forgets the same,
and the thing that actually governs the damage is the one
knob the comparison usually holds constant by accident.

The `mass` column disagrees, and it is printed rather than
buried: LoRA keeps 0.5552 of the language's probability
against full fine-tuning's 0.3111. The two columns answer
different questions — how *well* the model still predicts the
language, and how much of its output is still spent on it.
LoRA ties on the first and wins the second.

So what does LoRA actually guarantee? Something structural,
not statistical. **The base weights were never written to.**
Detach the adapter and the original model is back, bit for
bit — which the script asserts at startup. No amount of
measuring forgetting can give you that, and it is the
property that lets fifty fine-tunes be served from one copy
of the weights.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The row people get wrong is the fourth. LoRA does **not**
make the backward pass cheaper. To obtain the gradient for
`A` and `B` you first need the gradient with respect to the
effective matrix, which costs exactly what full fine-tuning
costs — the factor gradients are then projections of it,
`dA = s·dW·Bᵀ` and `dB = s·Aᵀ·dW`. That is why the
implementation needs no new derivation, and also why LoRA is
not a training-speed optimisation.

What it saves is everything that scales with the *number of
parameters you own*: the optimiser state, which for Adam is
two more copies of every trainable weight, and the storage
per task. The last row is the one that built an industry —
`N` tasks cost `N` adapters rather than `N` models.

---

## Where this sits

Adapters are from 2019: Houlsby et al. inserted small trained
bottlenecks between frozen layers and got most of fine-tuning
for a few percent of the parameters. The missing piece was
that their adapters changed the architecture, so they cost
latency at inference.

The claim this article tests came from Aghajanyan et al. in
2020, who measured the *intrinsic dimension* of fine-tuning
and found that models could be tuned within surprisingly
small random subspaces. Hu et al. turned that into LoRA in
2021, with the decisive engineering detail: because the
correction is linear, `W + sAB` can be multiplied out and
folded into `W` before serving, so a merged adapter costs
nothing at inference. QLoRA added quantised base weights in
2023 and made the whole thing run on one consumer GPU.

The lineage in this series is direct. Part 8 made a generator
defer to a document and recorded the collateral damage. This
article identifies the damage, finds that most of it follows
from step size rather than method, and shows that the
cheapest version of the same fine-tune recovers almost all of
it. Part 6's discipline shows up again, in a new place: do
not read an ordering off a single run when the thing you are
ordering has a standard deviation.

---

## What comes next

Part 10, **RLHF and DPO**, changes what the gradient is made
of. Everything in this track so far has trained against a
target that was known — the next word, a computed label, a
document's answer. Preference learning has no target at all,
only the statement that one output was better than another,
and the first thing that has to be built is a loss that can
be assembled out of comparisons.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**lora.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/13-natural-language-processing/09-fine-tuning-and-lora/lora.py)

Run it with:

```bash
pip install numpy
python lora.py
```

It needs only `numpy`. It is the slowest script in this
series, at about twenty-two minutes, because almost every number
in it is a mean over four seeds and there was no honest way
to make it cheaper. Everything is from scratch: the frozen
base plus `W + (alpha/r)AB` with `B` initialised to zero, the
projection of the base gradient onto the two factors, the
singular-value machinery for reading an update, and the exact
forgetting measurement over all 31,740 sentences — all
hand-differentiated and audited against central differences
at startup, through the LoRA factors as well as the base head
(worst error 1.20e-09 over 96 sampled gradients). The
headline insight worth pinning to the wall: **the update that
full fine-tuning produces is not low rank — reaching 90% of
it takes a rank in the twenties out of 64 — and LoRA works
anyway, because it never tries to reproduce that update;
rank 8 recovers 98.84% of full fine-tuning's accuracy on
15.45% of the parameters, while which matrices you adapt
matters more than how much rank you give them, and the
forgetting everyone attributes to the method turns out to be
a property of the learning rate instead**.

---

*This is Part 9 of the Natural Language Processing track in the Algorithms in Python series. The companion script `lora.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It reuses Part 1's synthetic micro-language ([word embeddings](https://medium.com/p/164f7f542a3c), [BPE tokenisation](https://medium.com/p/5e6a51341455), [sequence-to-sequence](https://medium.com/p/bbdf65112819), [BERT-style encoders](https://medium.com/p/a0ba729fe954), [GPT-style decoders](https://medium.com/p/6da33883e00d), [natural language inference](https://medium.com/p/d4f9c54be0d4), [natural language generation](https://medium.com/p/48a45ec5b396), [retrieval-augmented generation](https://medium.com/p/0686445e3d97)). Part 10 will look at RLHF and DPO.*
