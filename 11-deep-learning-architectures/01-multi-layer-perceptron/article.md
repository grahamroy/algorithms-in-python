# Multi-Layer Perceptron — Width Memorises, Depth Composes

### *Algorithms in Python --- Deep Learning Architectures, Part 1*

---

A new section deserves an honest confession: this series has
been using its subject for months without introducing it. The
semi-supervised track hand-rolled this exact machine twice — as
the network VAT attacked and as the decoders that drew the
moons — and the reinforcement learning track before it carried
`dQ/da` through the same hidden layers. The **multi-layer
perceptron** is deep learning's atom, and it has been doing the
series' heavy lifting in disguise. This article finally takes
it apart on its own terms — and it opens a section (CNNs, RNNs,
attention, transformers, GANs, diffusion) in which every
architecture to come is a *rearrangement of this one idea*.

The MLP is three ideas stacked:

**A layer is an affine map plus a pointwise nonlinearity**:
`a = act(xW + b)`. The nonlinearity is not decoration — it is
the entire reason depth exists. Stack affine maps alone and
they collapse into a single affine map: a hundred linear layers
*is* one linear layer, algebraically. The little `tanh` or
`ReLU` between them is what stops the tower from folding flat.

**Backprop is the chain rule with bookkeeping.** Run forward,
keep every activation; then walk the error backwards,
multiplying local derivatives as you go. For softmax with
cross-entropy the starting error is famously clean — just
`p − onehot` — and each layer hands the one below `d·Wᵀ` gated
by its activation's slope. Nothing more is happening in any
framework you have ever used.

**Training is gradient descent** on the surface those gradients
describe — Adam here, the series' workhorse.

And the new section gets a new stage: **two spirals**, 500
points, two arms wound 2.5 turns around each other. No straight
line touches it. That is precisely the point.

---

## A worked example: the ladder, the audit, and two deaths

### The ladder

One codebase, one training recipe (ReLU, He initialisation,
8,000 epochs of Adam); the only thing that changes is the list
of layer widths:

```
DEMO 1 --- The ladder: width buys wiggles, depth buys structure
    architecture        params   train acc   test acc
    linear (no hidden)       6      59.4%      57.2%
    [2]                     12      58.6%      55.2%
    [4]                     22      59.4%      55.4%
    [8]                     42      64.4%      57.6%
    [16]                    82      74.0%      67.8%
    [64]                   322      95.6%      84.2%
```

No hidden layer is a linear model — 57.2%, a straight line laid
across a spiral. Each hidden ReLU contributes one *crease* in
the decision surface, and the honest lower rungs are worth
staring at: two or four creases are **worse** than the straight
line they distract. The climb only starts once there are enough
creases to spend.

Then the twist — averaged over three seeds, because a single
run of a small network can flatter or slander an architecture:

```
    architecture        params   test acc (3 seeds)      mean
    [64]                   322   84.2%  92.2%  90.4%    88.9%
    [256]                 1282   93.2%  93.2%  89.6%    92.0%
    [16, 16]               354   94.8%  94.0%  94.4%    94.4%
```

`[16, 16]` beats both wide nets on **every seed**, with a
quarter of `[256]`'s parameters. The theory chapter says one
wide-enough layer can represent anything (universal
approximation, Cybenko 1989) — and this table is the practical
rebuttal's exhibit A: *representable* is not *learnable at a
budget*. One wide layer must draw the spiral in a single stroke
of 256 creases; two storeys **compose** — the first builds
curved strokes out of creases, the second builds the spiral out
of strokes. Width memorises; depth composes. That trade is the
founding reason this section exists, and every architecture in
it makes the trade deeper.

### The audit

```
DEMO 2 --- The audit: is the chain rule actually right?
    parameters checked : 49
    max |analytic - numerical| : 1.25e-10
```

Backprop's only failure mode is *silent*: a slightly wrong
gradient still trains, just worse, and no error message will
ever tell you. The ritual that separates "it trains" from "it
is correct" is the numerical gradient check — nudge every
parameter by ±h, difference the losses, compare with the
analytic gradient. Agreement to ten decimal places, limited by
floating-point arithmetic alone. (A tanh net for the audit:
smooth everywhere, whereas ReLU's kink would blur the
comparison exactly at zero.) Every derivative in the companion
script has passed it; every hand-rolled network this series
ships gets this test before it gets an article.

### Two classic ways a network dies

```
DEMO 3 --- Two classic ways a network dies
  DEATH 1: the stillborn network (all weights = 0)
    gradient norms at init : 0.0  0.0  0.0
    accuracy after 8000 epochs : 50.0%   (the class prior)

  DEATH 2: the vanishing gradient ([16] x 6, norms per layer,
           output side on the right)
    sigmoid : 1e-05  4e-04  2e-03  1e-02  4e-02  2e-01  8e-01
    tanh    : 3e-02  8e-02  1e-01  1e-01  8e-02  8e-02  8e-02
    relu    : 5e-02  1e-01  2e-01  2e-01  2e-01  2e-01  1e-01

    trained anyway:  sigmoid 66.8%   tanh 94.6%   relu 90.6%
```

Death one is the answer to a question every beginner asks: why
random initialisation? Set every weight to zero and hidden
units are identical twins — identical outputs, identical
gradients, no way to *ever* differ. The gradients are not
small; they are **exactly zero**, and after 8,000 epochs the
network still predicts the class prior. Symmetry must be broken
at birth or never.

Death two is the wall deep learning actually hit in the 1990s.
Sigmoid's derivative is at most 0.25, so each layer multiplies
the error signal down as it travels backwards: by the input
layer the gradient is ~25,000× smaller than at the output. The
tower learns its top and starves its bottom — 66.8% after the
same training budget. The honest nuance is in the other two
rows: tanh survives six storeys comfortably (its derivative
touches 1.0 at the centre, so the decay is gentle — 94.6%
here). ReLU's derivative is a clean 0-or-1 — **no decay at all
through active units** — which is what survives *sixty*
storeys, and why every architecture in this section inherits
it as the default.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The elegant fact of backprop — the reason it beat every rival
scheme for computing gradients — is that **the backward pass
costs the same as the forward pass**: one more matrix multiply
per layer, run in reverse. Training is `O(E · B · W)` for `E`
epochs, batch `B`, and `W` weights, plus `O(W)` for Adam's two
moment buffers. Memory is the forward activations — the
"pantry" the backward pass eats — `O(B · H)` for `H` hidden
units. The gradient check is `O(P)` *forward passes* (two per
parameter): a unit test you run once on a toy batch, never a
way to train.

---

## Where the MLP lives now

The lineage runs: Rosenblatt's perceptron (1958), the
Minsky–Papert freeze (a single layer cannot even do XOR,
1969), the backprop renaissance (Rumelhart, Hinton & Williams,
1986), the vanishing-gradient wall, and the ReLU + sensible
initialisation + GPU thaw that opened the modern era. But the
MLP is not a museum piece. Strip any transformer and look
inside: between every attention layer sits a two-layer MLP —
the "FFN block" — and it holds roughly **two-thirds of the
model's parameters**. The machine in this article is not the
ancestor of modern architecture; it is most of it, still on
duty, wrapped in increasingly clever plumbing. The rest of this
section is a tour of the plumbing.

---

## What comes next

Part 2, **Convolutional Neural Networks**, confronts the MLP's
blind spot: it treats input dimensions as unrelated columns —
shuffle the pixels of every image identically and an MLP
learns exactly as well, which is absurd for images. The fix is
weight sharing: one small crease-maker slid across the input,
so that what is learned *here* is known *everywhere*.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**mlp.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/11-deep-learning-architectures/01-multi-layer-perceptron/mlp.py)

Run it with:

```bash
pip install numpy
python mlp.py
```

It needs only `numpy` and runs in about a minute. Everything is
from scratch: the layer stack with three swappable activations,
the full chain-rule backward pass, Adam, He/Xavier/zero
initialisation, the numerical gradient audit, and the spiral
stage the whole section will reuse. The headline insight worth
pinning to the wall: **an MLP is an affine map and a
nonlinearity, stacked — without the nonlinearity the stack
collapses into one linear layer, and with it each hidden unit
buys one crease in the decision surface; on the two-spirals
stage a straight line gets 57.2%, creases climb the ladder,
and the deep twist is that [16,16] beats [256] on every seed
with a quarter of the parameters, because width memorises
while depth composes — provided the network is born alive
(zero initialisation gives exactly-zero gradients forever) and
its gradients survive the stairs (sigmoid decays ×25,000 over
six layers; ReLU's 0-or-1 derivative is why deep works)**.

---

*This is Part 1 of the Deep Learning Architectures track in the Algorithms in Python series. The companion script `mlp.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). The network it dissects already starred, uncredited, in the semi-supervised track — as [VAT](https://medium.com/p/5ad699fa2684)'s defender and as the [deep generative models](https://medium.com/p/59e50bb71435)' decoders. Part 2 will look at Convolutional Neural Networks.*
