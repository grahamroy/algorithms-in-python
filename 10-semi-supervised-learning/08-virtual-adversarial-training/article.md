# Virtual Adversarial Training — The Boundary May Not Live Here

### *Algorithms in Python --- Semi-Supervised Learning, Part 8*

---

Every method in this track so far has manufactured its safety
from *other things*. Extra judges. A second view. A graph. A
generative story. Each one an external structure bolted around
the classifier to keep its pseudo-labels honest.

**Virtual Adversarial Training (VAT)** (Miyato et al., 2018)
needs none of them, because it puts the track's founding
assumption — smoothness — directly **inside the loss function**.
For every training point, labelled or not, it asks a single
paranoid question: *what is the smallest nudge that would most
change my prediction here?* Then it penalises that change:

```
loss  =  CE(labelled)   +   α · KL( p(y|x)  ‖  p(y|x + r_adv) )
```

where `r_adv` is the most damaging perturbation of size at most
`ε`. The adversary is **virtual** because it attacks the model's
*own current belief* rather than a true label — which is
precisely why every unlabelled point can join the training set.
No pseudo-labels are ever minted. The unlabelled data
contributes something subtler: **places the decision boundary is
not allowed to go**. A boundary that passes near any data point
is expensive — a tiny nudge there flips predictions — so
training evicts the boundary from wherever data lives, into the
low-density gap. The cluster assumption, enforced by an
adversary.

This article builds VAT from scratch — the network, the
input-gradient backprop, the power-iteration attack — and runs
it on the same eight-label draws that have stress-tested every
method in this track. It posts the track's best committee-free
scoreboard, and its dial demo breaks the method in the most
instructive way yet.

---

## Finding the worst nudge: one step of power iteration

The whole algorithm rides on computing `r_adv` cheaply. The
trick (and the reason VAT scaled to real deep learning) is that
the direction of maximum prediction change is the dominant
eigenvector of the local curvature — and **one step of power
iteration** approximates it with two extra network passes:

1. Take a tiny random unit direction `d` and nudge: `x + ξd`.
2. Compute the KL between the clean and nudged predictions, and
   **backpropagate it to the input**. (For a softmax network
   this gradient is delightfully simple: the output-side error
   is just `q − p`, carried backwards to the input layer.)
3. Normalise that input gradient: this is the attack direction.
   Scale it to length `ε`: `r_adv`.

The companion script's network returns input gradients from the
same hand-written backward pass that returns parameter gradients
— the same machinery the RL track used to carry `dQ/da` from a
critic into an actor, doing adversarial duty here. Total
overhead: roughly two extra forward/backward passes per batch,
no extra parameters, nothing to store.

If this smells like **adversarial examples** — Goodfellow's
fast-gradient attacks — it should. VAT is adversarial training
with the labels removed: instead of attacking the *loss against
the truth* (which needs labels), it attacks the *consistency of
the model with itself* (which doesn't).

---

## A worked example: the same eight labels, one new loss

The stage is unchanged — Part 1's exact two-moons data and its
five random 8-label draws — but the model is new: a small neural
network (2 → 16 → 2), the track's first since Part 1's cautionary
tale about softmax confidence.

```
DEMO 1 --- The baseline: a neural network and 8 labels
    draw:  74.2%   72.8%   72.6%   85.0%   82.2%    mean 77.4%
```

Trained on eight points alone, the network does what unregularised
networks do: fits the labels perfectly, parks its boundary
wherever those eight points permit — usually straight through
both moons — and reports 99% confidence while doing it.

### Add the attack

Same network, same labels, plus the VAT penalty over all 500
points:

```
DEMO 2 --- The scoreboard: add the attack, evict the boundary
    draw   supervised only   + VAT penalty
      0         74.2%           90.8%
      1         72.8%           97.4%
      2         72.6%           98.0%
      3         85.0%           93.8%
      4         82.2%           97.2%
     mean        77.4%           95.4%

  Mean prediction confidence across the pool:
    supervised only: 0.99   (99% sure, boundary through the moons)
    with VAT       : 0.95   (hedges exactly where the data thins)
```

A **+18-point mean jump**, with every draw at 90% or better —
including draw 1, the one that broke self-training (59.6%) and
resisted tri-training (74.6%). That draw's problem was always
*coverage*: its labels left whole stretches of moon closer to
the wrong frontier, and every committee method inherited the
blind spot through its error estimates. VAT never estimates
errors. The unlabelled points in those uncovered stretches
simply forbid the boundary from crossing them, whichever side
the labels happen to sit on. This is the strongest 8-label
scoreboard of any *inductive* method in this track — only
Part 5's transductive label propagation (98.3%) sits higher,
and VAT hands you a trained network at the end rather than
answers for one fixed dataset.

### The dial: how hard to shove

`ε` — the attack radius — is VAT's one important knob, and it is
wonderfully interpretable: *the distance within which
predictions must not change.*

```
DEMO 3 --- The eps dial: how hard to shove
    eps     per-draw accuracies              mean
    0.05     85%   74%   93%   87%   85%       85.0%
    0.2      91%   97%   98%   94%   97%       95.4%
    0.5      92%   95%   95%   88%   93%       92.6%
    1.0      84%   94%   87%   90%   78%       86.7%
    2.0      49%   74%   80%   71%   77%       70.5%
```

A clean inverted U, and both ends fail for stated, geometric
reasons. At `ε = 0.05` the forbidden zone around each point is
smaller than the spacing between points, so the boundary can
still thread between them. At `ε = 0.2` — roughly the data's
own noise scale — the only region not claimed by some point's
bubble is the gap itself: 95.4%. And at `ε = 2.0` the attack
radius **spans the gap**: smoothness now demands the same
prediction on *both* moons, and the classes weld together (one
draw collapses to 49%). The dial is the cluster assumption made
quantitative: `ε` must be larger than the within-cluster
spacing and smaller than the between-cluster gap. When no such
`ε` exists, VAT has nothing to work with — which is exactly the
condition under which the assumption itself is false.

One honest footnote from this article's experiments: the
literature typically pairs VAT with **entropy minimisation**,
because a large model can satisfy smoothness by simply refusing
to commit away from the labels (uniformly uncertain predictions
are perfectly smooth). On this problem the cross-entropy on
eight labelled points supplied all the commitment needed — we
measured the EntMin variant at 91.0%, slightly *below* pure
VAT — so the script ships the pure form, and the flatness
escape-hatch stands as a warning for bigger models rather than
a demonstrated failure here.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

VAT's price is a constant factor on training, and nothing else.

**Per batch**: one clean forward pass, two passes for the power
iteration (nudged forward + backward to the input), and one
attacked forward/backward for the penalty — roughly **3× plain
supervised training**, independent of how much unlabelled data
you fold in per batch.

**Parameters and memory**: zero extra. No second model, no
buffer of pseudo-labels, no graph. The regulariser exists only
at training time; inference is untouched.

**The knobs**: `ε` (the dial above), `α` (penalty weight), `ξ`
(the power-iteration probe, robustly tiny). One honest caveat:
`ε` lives in *input units*, so feature scaling changes its
meaning — normalise first, or the forbidden bubbles are
ellipsoids you didn't intend.

---

## The consistency family

VAT is the sharpest member of the family that took over deep
semi-supervised learning: **consistency regularisation** — the
demand that a model answer the same under perturbation. The
Π-model and Mean Teacher perturb with dropout and augmentation
noise and average weights; **UDA** and **FixMatch** perturb with
strong data augmentations and threshold the confident ones;
VAT perturbs with the *provably worst* small nudge, no
augmentation engineering required. That's its enduring niche:
domains where you don't know good augmentations (tabular data,
molecules, signals) but do have a differentiable model — the
adversary finds the weak direction for you. And the same
smoothness-under-attack idea, aimed at labels instead of
consistency, is adversarial robustness training — one idea,
two literatures.

---

## What comes next

Part 9, **Deep Generative Models for SSL**, returns to Part 4's
worldview — model how the data was *made* — but with neural
machinery: variational autoencoders whose latent spaces carry
the class structure, so that a handful of labels can name
clusters the generative model already found. EM's philosophy,
after the deep learning upgrade.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**vat.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/08-virtual-adversarial-training/vat.py)

Run it with:

```bash
pip install numpy
python vat.py
```

It needs only `numpy` and runs in about half a minute.
Everything is from scratch: the MLP whose backward pass returns
input gradients, the one-step power iteration that turns those
gradients into the worst-case nudge, the KL penalty, and the
three experiments on Part 1's exact data. The headline insight
worth pinning to the wall: **VAT writes the smoothness
assumption into the loss — for every point, find the most
damaging nudge of size ε by one step of power iteration and
penalise the prediction change; unlabelled points never receive
labels, they become places the boundary may not go, lifting the
same network from 77.4% to 95.4% on the draws that broke the
committee methods — and the ε dial is the cluster assumption
made quantitative: bigger than the within-cluster spacing
(85.0% at ε=0.05), smaller than the gap (70.5% and a welded
49% at ε=2.0), just right at the noise scale (95.4%)**.

---

*This is Part 8 of the Semi-Supervised Learning track in the Algorithms in Python series. The companion script `vat.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It is evaluated on [Part 1](https://medium.com/p/eeca5accd031)'s exact label draws, and takes the committee-free road that [Part 7](https://medium.com/p/296af8f95185)'s tri-training approached with three judges. Part 9 will look at Deep Generative Models for SSL.*
