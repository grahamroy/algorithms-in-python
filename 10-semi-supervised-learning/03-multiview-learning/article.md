# Multiview Learning — What Two Views Agree On Is Probably Real

### *Algorithms in Python --- Semi-Supervised Learning, Part 3*

---

Co-training (Part 2) treated its two views as a lucky accident:
two feature sets, two classifiers, labels traded between them.
But hiding inside that algorithm is a deeper principle, and
**multiview learning** drags it into the open: *if two
independent views of the same example agree on something, that
something is probably real.* View-specific quirks — the
lighting in one sensor, the phrasing in one document field —
don't survive the comparison. What survives is what the views
**share**. And in most data, what paired views share is
precisely the thing you care about.

The payoff for this track is that agreement can be measured
**without a single label**. You don't need to know what class an
example is to check whether its two views move together. So the
entire unlabelled mountain — useless for supervised training —
becomes the training set for a *representation*: find the
directions on which the views agree, project into them, and let
your ten precious labels do their work in a space where the
signal is finally louder than the noise.

The classical engine for this is **Canonical Correlation
Analysis (CCA)** — Hotelling, 1936, an algorithm older than the
electronic computer — and this article builds it from scratch,
uses it to turn a near-chance problem into a solved one, shows
ten labels in the learned space **beating six hundred labels in
the raw one**, and then demonstrates its one great trap:
agreement is not the same thing as relevance.

---

## The problem: the loudest directions are the wrong ones

Real features are dominated by structure that has nothing to do
with the label. A camera's images vary more with lighting than
with the object; a document's words vary more with author style
than with topic. In the language of this article: each view
contains the shared **signal**, buried under view-specific
**nuisance** that is *louder*.

That loudness is what kills the obvious tools:

- **Distance-based classifiers** (like k-NN) compare points
  along *all* dimensions, so the nuisance dominates every
  distance. More labels barely help — the geometry itself is
  wrong.
- **PCA**, the default dimensionality reducer, keeps the
  directions of *maximum variance*. But maximum variance is
  exactly where the nuisance lives. PCA doesn't remove the
  noise; it curates it.

What we need is a criterion that distinguishes signal from
nuisance without labels. The multiview insight: **nuisance is
view-specific, signal is shared.** The lighting in view A has no
counterpart in view B; the class does. So don't ask "what is
loud?" — ask "what do the views agree on?"

---

## CCA: the mathematics of agreement

Given paired views `A` and `B` (matrices with one row per
example), CCA finds a projection direction for each view —
`w_A`, `w_B` — maximising the **correlation** between the
projected views:

```
maximise   corr( A · w_A ,  B · w_B )
```

Then it finds a second pair of directions, uncorrelated with the
first, and so on — each with its **canonical correlation**, a
number in [0, 1] saying how strongly the views agree along that
axis. Correlation is computed from the data alone: **no labels
enter anywhere**.

The from-scratch solution (exactly what the companion script
does): centre both views, form the covariance matrices
`C_AA`, `C_BB`, `C_AB`; **whiten** each view by
`C_AA^(-1/2)`, `C_BB^(-1/2)` so that within-view structure
can't masquerade as agreement; and take the **SVD** of the
whitened cross-covariance `C_AA^(-1/2) C_AB C_BB^(-1/2)`. The
singular vectors are the canonical directions; the singular
values are the canonical correlations. A dozen lines of NumPy.

---

## A worked example: ten labels against the noise

The companion script builds the situation deliberately: 600
example pairs, each a hidden class latent expressed in two 10-D
views, with view-specific nuisance **three times louder** than
the class signal. Ten examples are labelled.

```
DEMO 1 --- The setting: loud views, quiet signal, 10 labels
  k-NN in raw view A (10-D), 10 labels :  53.8%
  k-NN in PCA-2 of view A,   10 labels :  53.5%
  k-NN in raw view A, ALL 600 labels   :  67.5%
```

Ten labels in the raw space: a coin flip. PCA's two loudest
directions: still a coin flip — it kept the nuisance, exactly as
advertised. And the sobering row is the third: even **all 600
labels** only reach 67.5% in the raw space. When the geometry is
wrong, labels can't fix it.

### Agreement finds the signal

Now run CCA on the 600 **unlabelled** pairs:

```
DEMO 2 --- Agreement finds the signal: CCA from unlabelled pairs
  Canonical correlations found : 0.90, 0.26   (one strong shared direction)
  |corr| with the hidden class latent:  PCA-1 0.05   CCA-1 0.95

    seed   10 labels, CCA space   600 labels, raw space
      0           92.5%                 67.5%
      1           96.0%                 78.0%
      2           97.2%                 72.8%
```

Two things to read off. First, the diagnosis: CCA found **one
strong shared direction** (canonical correlation 0.90; the next
is 0.26 — noise), and that direction correlates **0.95** with
the hidden class latent we generated the data from. PCA's top
direction: **0.05**. Same data, no labels for either — one asked
"what is loud?", the other asked "what is shared?", and only the
second question has the class as its answer.

Second, the headline: with the space fixed, **ten labels reach
92–97% — beating six hundred labels in the raw space on every
seed**. The unlabelled mountain did the heavy lifting: it taught
the representation. The labels only had to draw one line in it.
This is semi-supervised learning's promise in its purest form —
and it is the shape of the modern recipe: *pretrain a
representation on unlabelled data, fine-tune with a handful of
labels.*

### The catch: agreement is not relevance

CCA finds what the views share. Nobody promised that what they
share is what you want. Give both views a common **background**
latent — same lighting, same season, same recording device —
four times louder than the class:

```
DEMO 3 --- The catch: agreement is not relevance
    seed   corr 1   corr 2   |comp-1 vs class|   k=1 acc    k=2 acc
      0     0.99     0.89        0.03              48.2%      96.0%
      1     0.99     0.91        0.04              53.2%      93.8%
      2     0.99     0.91        0.03              50.5%      82.5%
```

CCA does its job perfectly — and that's the problem. The
strongest shared direction (canonical correlation 0.99) is now
the *background*: its alignment with the class is 0.03. The
class hasn't vanished — it survives as **component 2** (still a
healthy 0.89–0.91 correlation) — but it has been **demoted**,
because CCA ranks shared directions by agreement strength, not
by usefulness. Keep only the top component and your classifier
is a coin flip (48–53%). Keep two, and the class is back
(82–96%).

The lesson generalises far beyond CCA: any method that learns
representations from view agreement — including the modern
contrastive ones — will faithfully learn *whatever the views
share*. Know what that is before you trust it, and don't
truncate the representation harder than your knowledge of it.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

CCA is refreshingly cheap — closed-form linear algebra, no
iterations, no learning rates.

**Covariances**: `O(n · d²)` for `n` pairs of `d`-dimensional
views — one pass over the data.

**Whitening and SVD**: `O(d³)` — on the *feature* dimension, not
the sample count. For views of tens or hundreds of dimensions
this is microseconds.

**Projecting** a new point: `O(d · k)` — a matrix multiply.
After that, your classifier works in `k` dimensions instead of
`d`, which makes the downstream few-label model cheaper too.

**Labels consumed: zero.** The representation is trained
entirely on unlabelled pairs — that column is the entire reason
this article is in a semi-supervised track.

---

## From Hotelling to CLIP

The agreement principle did not stay in 1936. **Kernel CCA**
made the projections nonlinear; **Deep CCA** (Andrew et al.,
2013) replaced them with neural networks trained to maximise the
same correlation. And the modern self-supervised wave rediscovered
the principle at scale: contrastive methods like SimCLR
manufacture two "views" of an image by augmentation and train a
network to make their representations agree; **CLIP** treats an
image and its caption as two views of one thing and aligns them
— hundreds of millions of unlabelled pairs teaching a
representation, exactly the CCA recipe with the linear algebra
swapped for transformers. When you next hear "pretrain on
unlabelled data, fine-tune with a few labels," you are hearing
this article's DEMO 2, industrialised.

---

## When to use multiview methods

**Reach for CCA (and its descendants) when** your examples come
in natural pairs — two sensors, two modalities, image and text,
before and after — and unlabelled pairs are plentiful while
labels are scarce. It is the cheapest serious representation
learner there is.

**Check what the views share.** DEMO 3 is the standing warning:
shared background structure (same device, same season, same
site) will out-rank the signal if it is louder. Either remove
known confounders first, or keep enough canonical components to
survive the demotion.

**Mind the linearity.** Plain CCA finds linear agreement; if the
shared structure is nonlinear, you need the kernel or deep
variants — same principle, heavier machinery.

---

## What comes next

Parts 1–3 all shared one worldview: train a classifier, be
careful about what you feed it. Part 4 changes the worldview:
**Expectation-Maximisation (EM)** treats the missing labels as
*missing data* in a probabilistic model — positing how the data
was generated, then alternating between guessing the hidden
labels (softly, with probabilities) and refitting the model to
those guesses. It is the statistician's answer to
semi-supervision, it is the algorithm self-training was quietly
approximating all along, and it comes with the field's most
famous convergence guarantee.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**multiview.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/03-multiview-learning/multiview.py)

Run it with:

```bash
pip install numpy
python multiview.py
```

It needs only `numpy` and runs in under a second. Everything is
from scratch: the two-view generator with its loud nuisance, CCA
via whitening and SVD, PCA for the contrast, and the three
experiments — with the learned directions audited against the
hidden generating latent. The headline insight worth pinning to
the wall: **what two views agree on is probably real — CCA finds
the directions along which paired views correlate, using no
labels at all, so the unlabelled mountain trains the
representation and ten labels finish the job (92–97%, beating
600 labels in the raw space, where PCA's loudest-direction
prior aligned 0.05 with the class against CCA's 0.95); but
agreement is ranked by strength, not relevance — a shared
background can take the top slot and demote the signal, so know
what your views share besides the thing you care about**.

---

*This is Part 3 of the Semi-Supervised Learning track in the Algorithms in Python series. The companion script `multiview.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 2](https://medium.com/p/6f581df978ff) covered Co-Training, which trades labels between views; this article learns from the views directly. Part 4 will look at Expectation-Maximisation, where the missing labels become missing data in a generative model.*
