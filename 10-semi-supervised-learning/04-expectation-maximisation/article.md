# Expectation-Maximisation — When Missing Labels Are Just Missing Data

### *Algorithms in Python --- Semi-Supervised Learning, Part 4*

---

The first three parts of this track shared a worldview: train a
classifier, and be careful what you feed it. Pseudo-labels,
vetoes, learned representations — all machinery bolted around a
discriminative model to keep it honest.

**Expectation-Maximisation (EM)** comes from a different
tradition entirely — statistics — and it changes the question.
Instead of asking *"how do I classify these points?"*, it asks
*"how were these points **produced**?"* Posit a **generative
story**: each class is a Gaussian cloud; a data point is born by
nature picking a class, then sampling from that class's cloud.
Under a story like that, a dataset where most labels are absent
is nothing exotic. It is simply a dataset with **missing data**
— and EM (Dempster, Laird & Rubin, 1977) is the canonical recipe
for maximum-likelihood estimation when part of the data is
missing. Semi-supervised learning stops being a trick and
becomes a special case.

The recipe alternates two steps you can write in four lines of
NumPy, and it carries the field's most famous convergence
guarantee. It also carries this track's most instructive
warning, and the companion script demonstrates both: the
guarantee holding perfectly while the classifier climbs to
oracle level — and the *same* guarantee holding perfectly while
the classifier sinks to a coin flip.

---

## The two steps

Say the story has parameters `θ` — for a two-class Gaussian
model, a prior, a mean, and a covariance per class. A few points
come with labels; most don't. EM repeats:

**E-step (Expectation).** With the current `θ`, compute each
unlabelled point's **responsibilities**: `P(class | x, θ)`, the
probability of each class given the point. This is a *soft*,
fractional assignment — a point can be 58% class A and 42%
class B. Labelled points keep responsibility 1 for their known
class.

**M-step (Maximisation).** Refit `θ` by **weighted** maximum
likelihood, every point counting fractionally toward every
class: a 58%-A point contributes 0.58 of itself to A's mean and
covariance, 0.42 to B's.

Repeat until nothing moves. The celebrated theorem: each
iteration can only **increase** (never decrease) the likelihood
of the observed data. EM climbs a hill, provably, without a
learning rate, without gradients, in closed form.

### Self-training, done softly

Look again at the E-step and you will recognise Part 1.
**Self-training is EM with hard assignments**: it computes the
same per-point class probabilities, then *commits* — promotes
the confident ones to full labels and retrains as if they were
certain. EM never commits. The 58/42 point stays 58/42, its
fractional weight forever revisable as the parameters improve.
Where self-training's confident mistakes locked in and
compounded (the Part 1 catastrophe), EM's soft assignments can
be walked back on the next iteration. Hard assignment is a
special case; softness is the upgrade.

---

## A worked example: ten labels and a true story

The companion script builds the world to match the model: two
overlapping tilted-Gaussian classes, 500 points, **10 labelled**.

```
DEMO 1 --- The setting: a generative story and 10 labels
  Gaussians fit to the 10 labels     :  77.4% test accuracy
  Oracle: fit to all 500 true labels :  87.6% test accuracy
```

The gap has a precise diagnosis here: a Gaussian per class needs
a mean *and a 2×2 covariance*, and five labelled points per
class is a hopeless budget for estimating a covariance matrix.
The 490 unlabelled points know those covariances — they just
don't know their classes.

### The loop, when the model is right

```
DEMO 2 --- The loop, when the model is right
    iter   log-likelihood   test accuracy
       1        -1593.3         81.8%
       2        -1573.3         83.4%
       3        -1563.1         84.4%
       4        -1558.8         85.2%
       5        -1556.7         85.2%
       6        -1555.4         85.8%
      30        -1552.2         88.4%

  A borderline unlabelled point: P(A) = 0.58, P(B) = 0.42.

    seed   10 labels only   semi-sup EM   oracle (500 labels)
      0        88.0%          90.2%         89.6%
      1        77.4%          88.4%         87.6%
      2        85.0%          87.0%         88.8%
```

Read the middle column first: the log-likelihood rises at every
single iteration — the Dempster-Laird-Rubin guarantee, visible
in a table — and the accuracy rides up with it, from 77.4% to
**88.4%, statistically at the all-labels oracle**. Across three
datasets, ten labels plus EM matches (twice slightly exceeds —
sampling noise) what 500 labels achieve. The unlabelled points
did exactly the job the diagnosis predicted: they estimated the
covariances that ten points never could, with the soft
responsibilities apportioning them between the classes.

And note the borderline point held at `P(A) = 0.58`: not
promoted, not discarded — counted as 0.58 of a point. That
restraint is the entire difference from Part 1.

### The catch: the most likely story can be wrong

Now change the world and keep the model. Each class is *really
two clusters*, interleaved `A, B, A, B` along a line. The
one-Gaussian-per-class story cannot express that — and the best
two-Gaussian description of this data is "left half vs right
half", a split that crosses **both** classes.

```
DEMO 3 --- The catch: the most likely story can be wrong
    iter   log-likelihood   test accuracy
       1        -1498.1         56.2%
       2        -1487.4         56.2%
       3        -1481.2         55.8%
       4        -1477.9         55.2%
      40        -1473.0         52.4%

  Log-likelihood of EM's fitted story      :   -1473.0
  Log-likelihood of the TRUE class Gaussians:   -1521.6
```

Read the two columns *against* each other this time. The
likelihood rises at every iteration, exactly as promised — while
the accuracy **falls**, from 56.2% to a coin-flip 52.4%. The 490
unlabelled points, so helpful a moment ago, are now actively
pulling the parameters toward the left-right split, because
under this model that split genuinely explains the data better:
EM's final story scores −1473.0 against the *true* class
parameters' −1521.6. The wrong answer is **more likely than the
truth**.

There is no bug here, and that is the lesson. EM's guarantee is
about *likelihood under your model*, and it delivered. It was
the model that couldn't say the true thing — so the most likely
sayable story was a false one, and every unlabelled point made
it more confident. Generative semi-supervised learning is a
bet on your story of the data: when the story is right,
unlabelled data is nearly free labels; when it is wrong,
unlabelled data is fuel for a fiction.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

EM's steps are closed-form and linear in the data.

**E-step**: `O(n · k · d²)` — a Gaussian density per point per
class (the `d²` from the covariance solve).

**M-step**: `O(n · k · d²)` — weighted means and covariances,
one pass.

**Per iteration**: no gradients, no learning rate, no step-size
tuning — each M-step lands exactly on the weighted-likelihood
optimum. Convergence to a plateau typically takes tens of
iterations (`I ≈ 30` here), giving `O(I · n · k · d²)` overall.

**The guarantee**: observed-data log-likelihood is
non-decreasing every iteration — the property both demo tables
display, in triumph and in failure alike.

---

## Where EM lives

This article used EM for semi-supervised classification, but the
recipe is one of the most reused in statistics and ML. Fitting
**Gaussian mixture models** with no labels at all is pure EM
(the unsupervised track used exactly that). **Hidden Markov
models** are trained by EM under the name Baum-Welch. Any
problem with latent variables — missing survey answers, mixed
populations, topic mixtures — tends to have an EM at the bottom
of it. And the classic semi-supervised success story is Nigam et
al. (2000): naive-Bayes text classification where EM over
thousands of unlabelled documents cut error dramatically with a
few dozen labels — DEMO 2, at 1990s web scale.

**When to reach for it**: you can write a credible generative
story, labels are scarce, and the model's assumptions are worth
betting on. **When to hesitate**: DEMO 3. Check the fit —
residuals, held-out likelihood, per-class cluster structure —
because a misspecified generative model doesn't fail loudly; it
converges beautifully to the wrong answer.

---

## What comes next

EM bet everything on a parametric story of the data. Part 5,
**Label Propagation**, makes the opposite bet: no distributions,
no parameters — just geometry. Build a similarity graph over all
points, labelled and not, and let the labels *flow* along its
edges until they cover the graph — the smoothness assumption
(nearby points share labels) in its purest algorithmic form, and
the gateway to the graph-based family that Part 6 completes.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**em.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/04-expectation-maximisation/em.py)

Run it with:

```bash
pip install numpy
python em.py
```

It needs only `numpy` and runs in under a second. Everything is
from scratch: the Gaussian log-densities, the weighted
maximum-likelihood M-step, the soft E-step, the observed-data
log-likelihood that the tables track, and both worlds — the one
that matches the model and the one that breaks it. The headline
insight worth pinning to the wall: **EM treats missing labels as
missing data in a generative model, alternating soft
responsibilities (E) with weighted refits (M), and provably
increases the data's likelihood every iteration — with the model
right, 490 unlabelled points estimated what 10 labels couldn't
and matched the oracle (77.4% → 88.4%); with the model wrong,
the same guarantee held while accuracy fell to a coin flip,
because EM's job is the most likely story, and under a
misspecified model the most likely story (−1473) genuinely beats
the truth (−1522)**.

---

*This is Part 4 of the Semi-Supervised Learning track in the Algorithms in Python series. The companion script `em.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 1](https://medium.com/p/eeca5accd031)'s self-training is this article's E-step with hard assignments; [Part 3](https://medium.com/p/348afbf5d79f) covered Multiview Learning. Part 5 will look at Label Propagation, where labels flow over a similarity graph instead of through a parametric story.*
