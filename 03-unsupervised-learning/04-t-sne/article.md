# t-SNE — When Local Neighbourhoods Matter More Than Global Geometry

### *Algorithms in Python --- Unsupervised Learning, Part 4*

---

In Part 3 we reduced the dimensionality of the digits dataset
with PCA. The first two principal components captured 28% of
the variance and gave us a 2D scatterplot in which the digit
classes were *starting* to be visible — but several digits
overlapped heavily, the boundaries were fuzzy, and the global
geometry of the 64-dimensional pixel space had to be approximated
by two straight axes. PCA is *linear*: it can only find
directions that are linear combinations of the original
features. When the structure in your data lives on a curved
*manifold* — handwritten digits being one of the canonical
examples — a linear projection can only do so much.

**t-SNE** (t-distributed Stochastic Neighbour Embedding;
van der Maaten & Hinton, 2008) is the algorithm that took
over scientific poster sessions and machine-learning blog posts
in the 2010s by going non-linear. It throws away the idea of
preserving variance or global geometry and instead asks one
question: in the high-dimensional space, *which points are each
other's neighbours?* — then arranges the low-dimensional
embedding so that the neighbour relationships are preserved as
faithfully as possible. The result is a 2D or 3D scatter where
visually distinct clusters appear *because* the algorithm has
worked hard to keep neighbours together and distant points
apart, ignoring exactly how far apart they actually were.

t-SNE is the reason MNIST digit clusters look beautifully
separable in every tutorial. It is the workhorse visualisation
for word embeddings, sentence embeddings, image embeddings,
and — overwhelmingly — for single-cell RNA-seq, where almost
every paper since 2014 has used t-SNE or its successor UMAP
to display cell types in 2D. It is also one of the most
*misunderstood* algorithms in machine learning, because its
output looks like a clustering result but does not have the
mathematical properties of one. We will spend a section on
exactly what t-SNE preserves and what it does not.

This article builds t-SNE from first principles. We will derive
the high-dimensional and low-dimensional similarity
distributions, explain why the low-dimensional one uses a
Student t-distribution rather than a Gaussian, write a
minimal from-scratch implementation in numpy, compare with
scikit-learn's `TSNE` and with PCA on the same digits dataset,
and finish with the long list of *what t-SNE is not* — most
importantly, not a feature extractor.

---

## The intuition: preserve who-is-near-whom

Imagine a 64-dimensional point cloud — handwritten-digit images
flattened into pixel vectors. Two images of the digit "3"
written by different people are *close* in this space; an image
of "3" and an image of "8" are *farther apart*. Distance in the
original space is the signal we want to capture.

PCA preserves variance by rotating and projecting. The cost is
that any two points that were close in 64-D can end up far
apart in 2-D if the direction connecting them is perpendicular
to the top two principal components. Local neighbourhoods get
shredded.

t-SNE inverts the priorities. The high-dimensional distances
between *nearby* points are preserved at all costs; the
distances between distant points are sacrificed. The 2-D
embedding is constructed by simulating a system where every
point *attracts* its high-dimensional neighbours and *repels*
the rest, until the system settles into a stable arrangement.

Mathematically: define a probability distribution `P` over
*pairs* of high-dimensional points where `P_{ij}` is high if
`x_i` and `x_j` are close. Define a similar distribution `Q`
over pairs of *low-dimensional* points `y_i, y_j`. Optimise the
low-dimensional points so that `Q` matches `P` as closely as
possible. "As closely as possible" means: minimise the
Kullback–Leibler divergence `KL(P || Q)`.

---

## The maths

The high-dimensional similarity between points `i` and `j`:

```
p_{j|i} = exp(-‖x_i - x_j‖² / 2σ_i²) /
          Σ_{k ≠ i} exp(-‖x_i - x_k‖² / 2σ_i²)
```

This is a Gaussian centred on `x_i`, scaled by a bandwidth
`σ_i` chosen so the *effective number of neighbours* (the
**perplexity**) is a target value — typically 5 to 50.
Perplexity is the most important knob in t-SNE; we will come
back to it.

Symmetrise across `i` and `j`:

```
p_{ij} = (p_{j|i} + p_{i|j}) / (2n)
```

So `P` is a joint distribution over pairs, with `Σ p_{ij} = 1`.

The low-dimensional similarity uses a **Student t-distribution
with one degree of freedom** (equivalent to a Cauchy
distribution):

```
q_{ij} = (1 + ‖y_i - y_j‖²)⁻¹ /
         Σ_{k ≠ l} (1 + ‖y_k - y_l‖²)⁻¹
```

The objective is the KL divergence:

```
C = Σ_{i, j} p_{ij} log(p_{ij} / q_{ij})
```

The low-dimensional `y_i` are initialised randomly and updated
by gradient descent. The gradient with respect to `y_i` has a
clean form:

```
∂C/∂y_i = 4 Σ_j (p_{ij} - q_{ij}) (y_i - y_j) (1 + ‖y_i - y_j‖²)⁻¹
```

Iterate until convergence (typically ~1000 iterations with
momentum and a few standard heuristics like "early exaggeration"
that multiply the high-d probabilities by 4× early in training —
the canonical implementations use the first 250 iterations; our
from-scratch script uses 100 — to encourage tight cluster
formation).

---

## Why the t-distribution?

t-SNE's predecessor SNE (Hinton & Roweis, 2003) used Gaussians
in both spaces. The result suffered from the **crowding
problem**: in a faithful low-dimensional embedding of
high-dimensional data, there simply isn't enough room to
accurately represent the distances *between* clusters of points
— moderate-distance pairs all collapse to roughly the same
distance, and clusters merge visually.

The Student t-distribution with one degree of freedom has heavy
tails. A pair of points at moderate low-dimensional distance
gets a much smaller `q_{ij}` than they would under a Gaussian
— which means the gradient pushes them *farther apart* than
SNE would. This deliberate over-spreading of moderate-distance
pairs is what gives t-SNE its characteristic well-separated
clusters. The "t" in t-SNE is doing the work that makes the
visualisation legible.

---

## Perplexity, the knob that matters

The bandwidth `σ_i` is set per-point such that the conditional
distribution `p_{j|i}` has a specified *perplexity*. Perplexity
is `2^H` where `H` is the entropy of `p_{j|i}`; it can be
interpreted as the *effective number of neighbours* each point
"feels". Typical values:

- **Perplexity 5** — very local; the embedding focuses on
  immediate neighbours and may shatter weakly-connected
  clusters into fragments.
- **Perplexity 30** — the default in most implementations;
  balances local and mid-range structure.
- **Perplexity 50** — broader; preserves more of the
  intermediate-scale geometry.
- **Perplexity > n/3** — meaningless; the algorithm tries to
  treat almost every point as a neighbour and the visualisation
  flattens into noise.

The choice of perplexity changes the visualisation. **Different
perplexities reveal different aspects of the data.** Best
practice is to run t-SNE at three or four perplexity values
(say 5, 30, 50) and look at all of them; treating any single
t-SNE plot as ground truth is a recipe for over-confident
storytelling.

---

## A worked example

The companion script uses the same `digits` dataset from
Part 3 — 1,797 grey-scale 8×8 images of handwritten 0–9 — and
runs four reductions side-by-side: from-scratch t-SNE (slow,
educational), scikit-learn's t-SNE (Barnes–Hut, fast), and PCA
for comparison.

```
DEMO 1 --- t-SNE from scratch on the digits dataset
  Subsample for speed : 300 points
  Perplexity          : 30
  Iterations          : 600
  Final KL divergence : 0.417
  KNN accuracy in 2D  : 0.897  (15-NN classifier on the embedding)
```

```
DEMO 2 --- Same data (full 1797), scikit-learn TSNE (Barnes-Hut)
  Perplexity          : 30
  Iterations          : 1000
  Final KL divergence : 0.754
  KNN accuracy in 2D  : 0.969
```

```
DEMO 3 --- Same data, PCA (2 components, from Part 3)
  Cumulative variance explained : 0.285
  KNN accuracy in 2D            : 0.624
```

The KNN-in-2D score is a standard way to *quantify* how good a
visualisation is at preserving class structure: project to 2D,
classify each point by majority vote of its 15 nearest 2D
neighbours, measure accuracy against true labels. It is not how
the embeddings were trained — t-SNE and PCA are both
unsupervised — but it tells you whether the structure they
discovered lines up with the structure that actually matters.

The results are striking: PCA's 2D projection lets a 15-NN
classifier get 62% of digits right (well above chance — `1/10
= 10%` — but not great). t-SNE's embedding pushes that to
**97%**. The non-linear visualisation has organised the same
64-D point cloud so that almost every digit ends up surrounded
by other copies of the same digit. PCA cannot do this; the
manifold is too curved to be captured by a linear projection.

---

## What t-SNE preserves and does not

This is the section that makes t-SNE *useful instead of
misleading*. The algorithm is widely misused. The honest list:

**Preserved.** Local neighbourhoods. If two points were among
each other's top-`k` nearest neighbours in the original space
(for `k` related to perplexity), they will usually be close
in the embedding too.

**Not preserved.** Distances between clusters. The visual
distance between two clusters in a t-SNE plot does *not*
correspond to their distance in the original space — t-SNE
deliberately spreads moderate-distance pairs apart. A cluster
on one side of the plot and a cluster on the other could be
geometric neighbours in the original space, or could be very
far apart. The plot does not tell you.

**Not preserved.** Cluster *sizes*. A big visual blob in a
t-SNE plot is not necessarily a big cluster in feature space;
t-SNE expands sparse clusters to fill available space. Two
visually similar-sized clusters in the embedding may have
wildly different point counts in reality.

**Not preserved.** Densities. The local point density in the
embedding does not reflect the density in the original space.
t-SNE tries to keep each point's neighbourhood the same *size*
in the low-dimensional plot, regardless of how dense or sparse
the local region was originally.

**Not preserved.** Global geometry. Don't read continuity,
"this cluster is between those two", or "this region is empty"
from a t-SNE plot. These conclusions are not licensed by the
algorithm.

**Not preserved across runs.** Different random initialisations
produce different layouts. The cluster *memberships* are
typically stable, but the relative *positions* of clusters
change. Always run t-SNE multiple times before reading
structure from the layout.

The honest summary: **t-SNE is a tool for seeing clusters that
exist in the data, not for measuring distances or detecting
spatial relationships**. Treat it like a microscope, not like a
map.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The cost of t-SNE depends on the implementation:

**Naive t-SNE** (the original 2008 algorithm) is `O(n²)` per
iteration because computing `q_{ij}` requires summing over all
pairs. With ~1000 iterations and `n = 5000` this becomes
genuinely slow.

**Barnes–Hut t-SNE** (van der Maaten, 2014) — sklearn's
default — approximates the repulsive forces using a quad-tree
over the low-dimensional points and brings per-iteration cost
to `O(n log n)`. Practical up to `n ≈ 10⁵`.

**FIt-SNE** (Linderman et al, 2019) — uses the Fast Fourier
Transform to evaluate the repulsive forces on a regular grid.
Sub-linear scaling in `n` past `n ≈ 10⁴`; the only practical
choice for `n > 10⁵`.

Memory in all variants is `O(n²)` for the `P` matrix (or
`O(n · k)` if you restrict each row to its `k` nearest
neighbours — which is the standard optimisation in fast
implementations).

For datasets above `n ≈ 10⁵`, **UMAP** (the subject of Part 5
in this track) typically wins on both speed and quality. For
`n < 10⁵`, t-SNE remains the de facto standard for
visualisation in most published research.

---

## Real-world ML and AI connections

t-SNE became famous in the deep-learning era as the
visualisation tool for *learned representations*:

**Word embeddings.** Mikolov's 2013 Word2Vec paper was popularised
in part by stunning t-SNE plots of 100-dimensional word vectors
arranged in 2D so that semantically related words (countries,
colours, professions) clustered together. The same recipe is
applied to sentence embeddings (Sentence-BERT), document
embeddings (Doc2Vec), and modern LLM embedding spaces.

**Image embeddings.** ConvNet features (Krizhevsky's AlexNet
2012, every subsequent ResNet / EfficientNet / ViT) are
routinely visualised by passing a dataset through the network,
collecting the penultimate-layer activations, and running t-SNE.
The resulting plots show how the network organised the input
space — frequently used as evidence that a model has "learned
meaningful representations".

**Single-cell RNA-seq.** The dominant application of t-SNE in
science. Cell-by-gene expression matrices, after normalisation
and PCA, are visualised with t-SNE (or UMAP) to identify cell
types. Almost every single-cell paper published since 2015
includes such a plot.

**TensorBoard's embedding projector.** Google's TensorBoard
visualisation tool ships with built-in t-SNE for exploring any
embedding space — a feature widely used for debugging
production ML systems.

**Anomaly inspection.** Run t-SNE on a dataset and look for
points that end up isolated from any cluster. Useful as a
qualitative anomaly-detection tool, especially when paired
with domain expertise.

**Quality assurance for deep representations.** During model
development, t-SNE on the model's activations is the canonical
"does the model know what it's doing?" check. If similar inputs
cluster together, the representation is probably good;
otherwise the representation is probably broken.

The pattern: t-SNE is almost always used as a *diagnostic*
or *presentation* tool rather than as a step inside a
production ML pipeline. It is the chart, not the model.

---

## When NOT to use t-SNE

t-SNE's weaknesses are the flip side of its strengths:

**As a feature extractor.** t-SNE embeddings are not
generalisable — there is no fitted model, just an arrangement
of points. New points cannot be embedded into an existing
t-SNE plot without re-running the algorithm. (Parametric t-SNE
exists but is rarely used.) Use PCA, autoencoders, or
self-supervised embeddings if you need transferable features.

**For clustering decisions.** A t-SNE plot is a visualisation,
not a clustering. Running K-Means on t-SNE coordinates "to find
clusters" is a common anti-pattern — the resulting clusters
reflect t-SNE's layout heuristics as much as the data. If you
need clusters, use HDBSCAN, K-Means, or hierarchical
clustering directly on the original (or PCA-reduced) data.

**For datasets beyond ~10⁵ points.** Barnes–Hut and FIt-SNE
help, but even FIt-SNE struggles past n = 10⁶. Use **UMAP**
instead (see Part 5).

**When you need to compare distances between clusters.** The
inter-cluster distances in a t-SNE plot are meaningless. If
those distances matter, do PCA or multidimensional scaling.

**When determinism matters.** Different random seeds produce
different layouts. For papers and dashboards where the figure
must be stable across re-runs, this is awkward. Pin the seed
and accept that the visualisation is one of many valid
arrangements.

**For data that is genuinely linear.** If PCA already
separates the classes well, t-SNE is over-engineering. Use the
simpler tool.

---

## What comes next

Part 5 of the Unsupervised Learning track is **UMAP**
(Uniform Manifold Approximation and Projection; McInnes,
Healy & Melville, 2018). UMAP solves most of t-SNE's
limitations: it is faster (often 10× to 100×), it scales to
millions of points, it preserves *both* local and global
structure better, and unlike t-SNE it produces a re-usable
transformation — new points can be embedded without re-fitting.
For visualisation of large datasets in 2026, UMAP is the
default; t-SNE remains the safer choice on smaller datasets
and in scientific publishing where its longer track record
matters.

After UMAP the track turns to NMF (Non-negative Matrix
Factorisation) for parts-based decomposition, then Spectral
Clustering and Association Rule Mining.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**tsne.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/03-unsupervised-learning/04-t-sne/tsne.py)

Run it with:

```bash
pip install numpy scikit-learn
python tsne.py
```

It needs `numpy` and `scikit-learn`. The script implements a
minimal from-scratch t-SNE — Gaussian high-dimensional
similarities, t-distributed low-dimensional similarities,
gradient descent on KL divergence — on a 300-point subsample
of the digits dataset (the full naive `O(n²)` implementation
would take minutes on the full 1797). It then runs
scikit-learn's Barnes–Hut t-SNE on the full dataset and PCA
for comparison, reporting a `KNN-in-2D` accuracy that
quantifies how cleanly each method preserves digit-class
neighbourhoods. The headline insight worth pinning to the
wall: **t-SNE is a non-linear visualisation that preserves
local neighbourhoods at the cost of global geometry; it makes
manifold structure visible and is the right tool for
*looking* at high-dimensional data, but not for measuring,
clustering, or feature-engineering from the result**.

---

*This is Part 4 of the Unsupervised Learning track in the Algorithms in Python series. The companion script `tsne.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 3](https://medium.com/p/20ea2e6745d4) covered Principal Component Analysis. Part 5 will look at UMAP — t-SNE's faster, scalable, and better-behaved successor.*
