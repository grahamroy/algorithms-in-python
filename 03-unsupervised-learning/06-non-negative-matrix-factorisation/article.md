# Non-Negative Matrix Factorisation — Where Parts Beat Mixtures

### *Algorithms in Python --- Unsupervised Learning, Part 6*

---

In Part 3 we used PCA to decompose the digits dataset into a
small number of *principal components* — directions of maximum
variance in the original 64-dimensional pixel space. Each
component was a weighted combination of the original features,
and the weights could be *any* real number, positive or
negative. The first principal component of a face dataset, for
example, often includes positive contributions from "bright
pixels in the cheek region" and negative contributions from
"dark pixels in the eye sockets". The component is a real
direction in feature space, but it is not interpretable as a
*thing*; it is a mixture of signed adjustments to many features
at once.

Today we look at the dimensionality-reduction technique that
takes the opposite design choice. **Non-negative Matrix
Factorisation** (NMF; Lee & Seung, 1999) keeps the same
"decompose `X` into components plus coefficients" framework as
PCA, but adds one constraint: all the components and all the
coefficients must be **non-negative**. Negative values are
forbidden. The data, the components, and the way the components
combine — all entries are required to be `≥ 0`.

The non-negativity constraint sounds technical. The consequence
is profound: it forces the decomposition to be **additive and
parts-based**. The components stop being "directions in feature
space" and start being "parts of the data". On a face dataset
NMF rediscovers things that look like eyes, noses, mouths,
hairlines — *anatomical parts that combine additively* to form
a face — where PCA gives you "eigenfaces" that look like
ghostly amalgamations of multiple faces averaged with positive
and negative weights. On a text dataset (word counts ≥ 0), NMF
rediscovers things that look like *topics* — co-occurring word
clusters — where PCA gives mathematically valid but
hard-to-interpret directions.

NMF underlies modern topic modelling, image-parts decomposition,
audio source separation, and a substantial fraction of
recommender systems. It is computationally heavier than PCA
and lacks PCA's clean theoretical guarantees, but on
non-negative data — counts, intensities, frequencies,
amplitudes — it routinely produces decompositions a human can
actually look at and explain.

This article builds NMF from first principles. We will derive
the multiplicative-update algorithm that powers most
implementations, walk through both flagship applications (text
topic modelling and image parts decomposition), compare with
scikit-learn's `NMF`, and finish with the cases where NMF is
worth the extra computational cost and the cases where PCA or
LDA are the better tool.

---

## The setup

Suppose your data matrix is `X` with `n` rows (samples) and
`d` columns (features), and every entry is `≥ 0`. NMF asks for
an approximate decomposition:

```
X ≈ W · H
```

where:

- `W` is `n × k` with all entries `≥ 0` — the "encodings" of
  each sample (one row per sample, `k` non-negative
  coefficients).
- `H` is `k × d` with all entries `≥ 0` — the "components"
  (one row per component, each being a non-negative vector in
  feature space).
- `k` is the number of components, chosen in advance (usually
  much smaller than `d`).

The `i`-th row of `X` is then approximately:

```
X[i, :] ≈ W[i, 0] · H[0, :] + W[i, 1] · H[1, :] + ... + W[i, k-1] · H[k-1, :]
```

— a *non-negative weighted sum* of `k` non-negative components.
There is no subtraction. Every sample is a sum of "parts"
(rows of `H`), weighted by how strongly each part is present in
that sample (the corresponding row of `W`).

The objective is to minimise the reconstruction error, usually
the squared Frobenius norm:

```
minimise   ‖X − W · H‖²_F
subject to W ≥ 0, H ≥ 0
```

(Alternative objectives — KL divergence, Itakura–Saito — exist
and are appropriate for different data types; the squared
Frobenius norm is the default in sklearn and the easiest to
explain.)

This is a non-convex problem in `(W, H)` jointly, but convex in
each block separately. Like K-Means, NMF is solved by
alternating optimisation: fix `W`, update `H`; fix `H`, update
`W`; repeat.

---

## The multiplicative-update algorithm

Lee & Seung's 1999 *Nature* paper gave NMF its breakthrough
algorithm: **multiplicative updates** that automatically
preserve non-negativity at every step.

Start with `W` and `H` randomly initialised to positive
values. Then alternate:

```
H_{ij} ← H_{ij} · (Wᵀ X)_{ij} / (Wᵀ W H)_{ij}
W_{ij} ← W_{ij} · (X Hᵀ)_{ij} / (W H Hᵀ)_{ij}
```

Two observations.

**The updates are guaranteed non-negative.** Since `W` and `H`
start non-negative, the numerators and denominators are
non-negative, the multiplicative factor is non-negative, and
the new values stay non-negative. No projection step is needed.

**The objective is guaranteed to decrease.** Lee & Seung
proved that each update monotonically decreases the
reconstruction error. Convergence to a local minimum is
guaranteed; convergence to the *global* minimum is not (the
problem is non-convex jointly in `W` and `H`).

Modern implementations also support:

- **Projected gradient descent** — faster convergence on some
  problems, slightly more complex.
- **Coordinate descent** — block updates of individual columns;
  what sklearn uses by default.
- **Sparse NMF variants** — add L1 penalties to `W` or `H` to
  encourage sparse encodings or sparse components.
- **Regularised NMF** — Frobenius (L2) penalties for
  smoother solutions.

The multiplicative-update algorithm is slower but conceptually
the cleanest, and the easiest to write from scratch.

---

## A worked example: topic modelling on text

NMF's flagship application is **topic modelling**: given a
collection of documents, discover a small number of "topics"
each defined by its top words. The companion script applies
NMF to the 20-newsgroups dataset (a classical corpus of ~18,000
newsgroup posts from the early 1990s), restricted to 4
categories — autos, atheism, religion-christian, baseball — to
keep the topics readable.

```
DEMO 1 --- NMF from scratch on 20-newsgroups TF-IDF (4 categories)
  Data shape : 2270 documents × 1000 vocabulary (TF-IDF)
  Components : 4
  Iterations : 200 (multiplicative updates)
  Final reconstruction error : 45.05 (Frobenius)
  Topic words (top 8 per component):
    Topic 0: people don think just know like religion say
    Topic 1: car cars engine new like dealer good price
    Topic 2: god jesus believe faith christ does hell bible
    Topic 3: year team game games runs hit pitching good
```

```
DEMO 2 --- Same data, scikit-learn NMF (coordinate descent)
  Iterations : 200
  Final reconstruction error : 45.05 (Frobenius)
  Topic words (top 8 per component):
    Topic 0: people don think just know like say religion
    Topic 1: year team game games runs hit pitching baseball
    Topic 2: car cars engine like new dealer good price
    Topic 3: god jesus christ faith believe hell bible sin
```

```
DEMO 3 --- How many topics? Reconstruction error vs k
     k   reconstruction error
   ---   --------------------
     2                  45.44
     4                  45.05
     6                  44.79
     8                  44.57
    10                  44.35
    15                  43.87
    20                  43.43
```

Three observations.

**NMF rediscovered three of the four newsgroup categories
without labels.** The four discovered topics correspond
clearly to **autos** (car, engine, dealer), **christianity**
(god, jesus, faith, bible), **baseball** (team, game, hit,
pitching), and a fourth **general discussion / religion**
topic that mixes the atheism category with discursive-language
features (people, think, religion, say). The category-to-topic
mapping is not perfect — the atheism category lacks a vocabulary
distinctive enough to claim its own topic at `k = 4`, so it
gets folded into a general meta-discussion topic — but the
three well-vocabularied categories come through cleanly. PCA
on the same data does not produce topics this readable; its
components are mixtures of positive and negative TF-IDF
directions and the top "words" (by absolute weight) include
high-magnitude common terms shared across categories.

**The from-scratch implementation matches sklearn.** The two
algorithms reach the same reconstruction error (`45.05`) and
identify the same topics with the same top words (the topic
ordering differs because NMF is unique only up to permutation).
Multiplicative updates and coordinate descent converge to
different local optima in principle, but on this dataset they
land in essentially the same solution.

**The reconstruction error keeps falling as `k` grows** — but
slowly, and the *topic interpretability* peaks somewhere around
`k = 4` to `k = 8`. With `k = 20` you get finer-grained topics
that subdivide categories (e.g. "baseball" into "pitching" and
"hitting") — useful sometimes, but eventually you are just
re-discovering noise. Picking `k` is the standard
unsupervised-learning judgement call: reconstruction error
gives a lower bound on the right value, interpretability gives
an upper bound, and the answer is somewhere in between.

---

## Why parts, not mixtures?

PCA's components can have negative entries. To reconstruct a
sample, PCA *adds and subtracts* contributions from each
component. On data that is inherently non-negative — pixel
intensities, word counts, gene expression, audio amplitudes —
this means the decomposition cancels positive and negative
contributions to express each sample. The components themselves
are not interpretable as anything physical; they are
mathematical directions that happen to span the variance.

NMF's non-negativity constraint forbids cancellation. To
reconstruct a sample, NMF can only *add* contributions. The
components must therefore be *constituents* of the data — each
one a recognisable sub-structure that genuinely appears in the
samples. On faces this manifests as components that look like
eyes, mouths, noses, hair, foreheads. On text as topics. On
audio as harmonic groups. On gene expression as co-regulated
gene modules.

This is not magic — it is what the constraint *requires*. There
is no mathematical theorem that says NMF components correspond
to semantic parts. On well-behaved data they often do. On data
where the parts assumption is violated (e.g. data that
genuinely has negative values, or where signals interfere
destructively) NMF either fails or has to be applied to a
transformed representation.

Lee & Seung's original *Nature* paper made this point
visually with face decompositions. The "NMF parts" figure that
accompanies most introductions to the algorithm is the single
clearest illustration of why parts-based decomposition matters.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Per iteration cost: each multiplicative update is dominated by
matrix multiplications `Wᵀ · X`, `Wᵀ · W · H`, `X · Hᵀ`, and
`W · H · Hᵀ`. With dimensions `n × d`, `n × k`, and `k × d`,
the dominant cost is `O(n · d · k)` per iteration.

With `I` iterations (typically 200 to 1000 for convergence),
total cost is `O(I · n · d · k)`. For text corpora with `n ≈
10⁵`, `d ≈ 10⁵`, `k ≈ 50`, that's `≈ 10¹³` operations per full
fit — minutes on a modern machine with optimised BLAS, but
substantially heavier than PCA's `O(n · d²)` single SVD.

Memory is `O(n · d)` for the data (sparse if applicable), plus
`O((n + d) · k)` for `W` and `H`. NMF's memory profile is
friendlier than PCA's on sparse high-dimensional data, because
the dense covariance matrix is never formed.

For very large datasets, **online NMF** variants update `W` and
`H` from mini-batches of rows, reducing memory and enabling
streaming fits. Scikit-learn's `MiniBatchNMF` and the `online`
solver in some libraries implement this.

---

## Real-world ML and AI connections

NMF is everywhere on non-negative data:

**Topic modelling.** NMF is the simpler, faster, and often
just-as-good cousin of Latent Dirichlet Allocation (LDA).
Where LDA is a generative probabilistic model with
hyperparameters and inference algorithms, NMF is straight
matrix factorisation with a non-negativity constraint. For
many practical topic-modelling tasks the two produce nearly
identical topic lists, and NMF is easier to deploy. Most
production topic-modelling pipelines use one or the other.

**Recommender systems.** Implicit-feedback recommender
systems — "users × items × clicks" matrices, where clicks
are non-negative counts — fit naturally into the NMF
framework. The decomposition `X ≈ W · H` gives latent user
factors (`W`) and latent item factors (`H`); recommendations
are computed by `W · H` evaluated at unobserved cells.
Variants like *implicit-feedback ALS* and *weighted NMF*
underlie much of recommendations infrastructure at Spotify
(circa 2014), Amazon, and Netflix's earlier systems.

**Audio source separation.** A spectrogram (frequency × time
matrix of audio energies) is non-negative. NMF decomposes it
into `k` spectral components and their time-activations.
Setting `k` = number of expected sources gives a separation
of mixed audio into per-source spectrograms — the foundation
of many audio source-separation systems before deep learning
took over.

**Hyperspectral unmixing.** Remote-sensing data captures
per-pixel spectral signatures that are mixtures of multiple
materials. NMF decomposes the data into endmember spectra
(`H`) and per-pixel abundance maps (`W`) — the literal
"how much of each material is in this pixel" answer.

**Single-cell genomics.** Gene-expression matrices (cells ×
genes, non-negative counts) decompose into "gene programs"
(`H`) and per-cell activations (`W`) that often correspond
to interpretable cell-state signatures. Tools like cNMF and
NMFP shipped specifically for this application.

**Image parts decomposition.** Lee & Seung's original face
example, plus many subsequent applications to medical imaging,
microscopy, and astronomy. The "discover the visual primitives"
use case.

**Chemometrics.** Spectroscopic data (NMR, mass spec,
chromatography) decomposes into pure-component spectra and
mixture coefficients via NMF, used heavily in analytical
chemistry.

The pattern: NMF is the right tool when your data is genuinely
non-negative *and* you want the decomposition to be
interpretable. That is a large class of real problems.

---

## When NOT to use NMF

NMF's strengths come with constraints:

**When the data has genuinely negative values.** Centred
financial returns, accelerometer signals, anything where the
sign carries meaning — NMF cannot represent these. Use PCA,
ICA, or sparse coding.

**When you need a unique solution.** NMF is non-convex; runs
with different random initialisations produce different
decompositions. The components are unique only up to scaling
and permutation. Sklearn's `init='nndsvd'` mitigates this by
using a deterministic initialisation, but truly unique answers
require the SVD-based methods of PCA.

**When you need an orthogonal decomposition.** NMF components
are not orthogonal in general. PCA's orthogonality is useful
for downstream applications that need a true basis. NMF does
not give you one.

**When the data is too sparse and high-dimensional to
converge well.** A 10-million-document TF-IDF matrix with
millions of terms is at the edge of practical NMF. Use
incremental / online variants, or accept that LDA's
probabilistic framework may converge better.

**When interpretability is not actually the goal.** If you
just want compression or visualisation, PCA / UMAP are
faster and have stronger guarantees. Use NMF only when the
"parts" interpretation matters.

**When the multiplicative update gets stuck on zeros.** Once
a `W_{ij}` or `H_{ij}` hits zero, the multiplicative update
keeps it there forever (the update factor multiplies by
something ≥ 0 but if the current value is 0 it stays 0).
Modern implementations add small regularisation to prevent
this; from-scratch implementations need a manual epsilon
floor.

---

## What comes next

Part 7 of the Unsupervised Learning track is **Spectral
Clustering** — a clustering algorithm built on the
eigendecomposition of a similarity graph's Laplacian matrix.
Where K-Means assumes Euclidean spherical clusters and
hierarchical clustering uses linkage criteria, spectral
clustering uses the *spectrum* (eigenvalues) of a graph
representation to find clusters of arbitrary shape — the
mathematics that also underlies UMAP's spectral
initialisation and many modern manifold-learning techniques.

After spectral clustering the track wraps with Association
Rule Mining and the track-level overview.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**nmf.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/03-unsupervised-learning/06-non-negative-matrix-factorisation/nmf.py)

Run it with:

```bash
pip install numpy scikit-learn
python nmf.py
```

It needs `numpy` and `scikit-learn`. The script implements
NMF from scratch using Lee & Seung's multiplicative-update
algorithm, fits it to a 4-category subset of the 20-newsgroups
dataset (`autos`, `atheism`, `christianity`, `baseball`),
displays the top words per discovered topic, and compares
against scikit-learn's `NMF` (which finds essentially the same
topics). A small `k`-sweep shows how reconstruction error
falls with more components. The headline insight worth pinning
to the wall: **NMF decomposes non-negative data into a sum of
non-negative parts; on text the parts are topics, on faces
they are facial features, on audio they are spectral
templates, and on gene expression they are co-regulated gene
programmes**.

---

*This is Part 6 of the Unsupervised Learning track in the Algorithms in Python series. The companion script `nmf.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 5](https://medium.com/p/baae695bcfd6) covered UMAP. Part 7 will look at Spectral Clustering — graph-Laplacian-based clustering for arbitrary cluster shapes.*
