# Principal Component Analysis — The Directions Your Data Cares About

### *Algorithms in Python --- Unsupervised Learning, Part 3*

---

In Parts 1 and 2 we asked clustering algorithms to find
**groups** in the data — points that look similar to each
other and different from the rest. Today we ask a different
question. Forget the groups. Look at the *whole point cloud*
and tell me: which directions in feature space carry the most
information?

That is the question **Principal Component Analysis** (PCA)
answers. Pearson posed it in 1901; Hotelling re-derived and
popularised it in 1933. The answer is geometrically simple
— rotate the coordinate system so the new axes line up with
the directions of greatest variance, in decreasing order — and
operationally devastating: most real datasets put 80% or 90% of
their variance into a small number of directions, so a few
"principal components" can replace dozens of original features
with almost no information loss.

PCA is the foundational dimensionality-reduction technique. It
underlies every face-recognition system from the 1990s
("eigenfaces"), every "compress your high-dimensional features
before clustering" preprocessing step, every visualisation of a
high-dimensional dataset projected onto two axes for a
scatterplot, the whitening step that helps neural networks
train, and the LSA / LSI text-retrieval methods that pre-dated
modern embeddings. It is older than scikit-learn and older than
machine learning as a discipline, and it is still the right
first thing to try when your dataset has too many features.

This article builds PCA from first principles. We will
motivate it as variance maximisation, derive the algorithm via
the Singular Value Decomposition (SVD) of the centred data
matrix, implement it from scratch in numpy, fit it to
scikit-learn's handwritten-digits dataset to compress 64 pixel
features into 2 principal components good enough to *see*
digit clusters, compare with scikit-learn's `PCA`, and finish
with what PCA assumes about your data and where to reach for
something else (t-SNE, UMAP, autoencoders, ICA) when those
assumptions break.

---

## The intuition: maximum-variance directions

Suppose you have data with two highly correlated features —
say, height and weight of adults. Plot points on the
height-vs-weight plane and they will form a long elongated
cloud running diagonally from short-and-light to tall-and-heavy.
Most of the variation in the cloud is along that diagonal; very
little is perpendicular to it.

Standard `(x, y)` coordinates are a poor description of this
data. A *better* coordinate system would have one axis aligned
with the diagonal (the direction of most variance) and a second
axis perpendicular to it (the direction of least variance). In
those coordinates, almost all the information about each point
sits in the first axis; the second is nearly zero. We have
*reduced* a two-dimensional cloud to a one-dimensional summary
without losing meaningful structure.

PCA generalises this picture to any number of dimensions. Find
the direction of maximum variance — call it `PC1`. Then find
the direction of maximum variance *among directions orthogonal
to `PC1`* — that is `PC2`. Continue until you have `d`
principal components in `d`-dimensional data, each one
orthogonal to all the previous, each one carrying less variance
than the one before. The components are a *new orthonormal
basis* for the data, ordered by how much information they
carry.

The compression win comes from the fact that, on real data, the
later principal components usually carry very little variance.
Drop them, keep only the first few, and you have replaced a
high-dimensional dataset with a low-dimensional one that
preserves the structure you cared about.

---

## The math: SVD of the centred data matrix

The clean way to derive PCA is via the **Singular Value
Decomposition** of the centred data matrix.

Centre your data so each feature has mean zero — subtract the
column means:

```
X_c = X - mean(X, axis=0)
```

Now compute the SVD:

```
X_c = U · Σ · Vᵀ
```

where `U` is `n × n` orthonormal, `V` is `d × d` orthonormal, and
`Σ` is `n × d` with the **singular values** `σ_1 ≥ σ_2 ≥ ... ≥ 0`
on its diagonal and zeros elsewhere.

The principal components are the columns of `V`. The variance
explained by the *i*-th principal component is
`σ_i² / (n − 1)`. The "score" of each data point on that
component — its coordinate in the new basis — is the
corresponding column of `U · Σ`, equivalently `X_c · V[:, i]`.

The connection to the covariance picture: the sample
covariance matrix is `C = X_cᵀ · X_c / (n − 1)`. Substituting
`X_c = U Σ Vᵀ` gives `C = V · (Σ² / (n − 1)) · Vᵀ`, which is
the eigendecomposition of `C` — eigenvectors `V`, eigenvalues
`σ_i² / (n − 1)`. SVD-of-data and
eigen-decomposition-of-covariance are the same calculation; SVD is numerically more
stable and avoids ever forming the `d × d` covariance matrix
explicitly, which matters when `d` is large.

---

## The algorithm

Putting it together:

```
fit(X, n_components):
    # 1. Centre the data
    mean_ = X.mean(axis=0)
    X_c = X - mean_

    # 2. Singular Value Decomposition
    U, S, Vt = svd(X_c, full_matrices=False)

    # 3. Components are rows of Vt; explained variance from S
    components_ = Vt[:n_components]
    explained_variance_ = S**2 / (n - 1)
    explained_variance_ratio_ = explained_variance_ / explained_variance_.sum()

    # 4. Resolve sign ambiguity (SVD is unique only up to sign)
    sign_correct(components_)
    return mean_, components_, explained_variance_ratio_

transform(X):
    return (X - mean_) @ components_.T

inverse_transform(X_proj):
    return X_proj @ components_ + mean_
```

A few subtleties worth pulling out.

The **centring** step is mandatory. PCA without centring measures
"distance from origin" rather than "variance around the mean",
which is rarely what you want. Most implementations centre by
default; if you do PCA from scratch you must remember to
subtract the mean.

The **inverse transform** is what makes PCA a *compression*: a
point projected to `k` components, then transformed back, is
the closest possible reconstruction in the original space using
only `k` directions. The reconstruction error per point is
exactly the variance carried by the dropped components.

The **standardisation** question is more subtle. PCA is sensitive
to feature scale: a feature in metres will have larger variance
than the same feature in millimetres just because of units. If
your features are on different scales, standardise first
(`StandardScaler` in sklearn) so each contributes equally to
the variance budget. If your features are already in the same
units (pixel intensities, gene-expression z-scores), centring
without scaling is usually right.

---

## A worked example

The companion script loads scikit-learn's `digits` dataset —
1,797 grey-scale 8×8 images of handwritten digits 0–9, flattened
into 64-dimensional feature vectors. The task: reduce 64
dimensions to 2, keep enough variance to see the digit clusters.

```
DEMO 1 --- PCA from scratch on scikit-learn's digits dataset
  Data shape          : 1797 samples, 64 features
  Centring            : per-feature mean
  Method              : SVD of centred data
  Components fitted   : 10
  Explained variance ratio (top 10):
    PC  1: 0.149   |   cumulative 0.149
    PC  2: 0.136   |   cumulative 0.285
    PC  3: 0.118   |   cumulative 0.403
    PC  4: 0.084   |   cumulative 0.487
    PC  5: 0.058   |   cumulative 0.545
    PC  6: 0.049   |   cumulative 0.594
    PC  7: 0.043   |   cumulative 0.637
    PC  8: 0.037   |   cumulative 0.674
    PC  9: 0.034   |   cumulative 0.707
    PC 10: 0.031   |   cumulative 0.738
```

```
DEMO 2 --- Same data, scikit-learn PCA
  Components fitted     : 10
  Explained variance ratio (top 5): 0.149 0.136 0.118 0.084 0.058
  Reconstruction error (10 components) MSE : 4.9143
  Maximum |difference| in projections vs from-scratch : 2.58e-14
```

```
DEMO 3 --- How many components do we need?
    k    cumulative variance   reconstruction MSE
   ---   -------------------   ------------------
     2                 0.285              13.4210
     5                 0.545               8.5424
    10                 0.738               4.9143
    20                 0.894               1.9843
    30                 0.959               0.7681
    40                 0.988               0.2215
    64                 1.000               0.0000
```

Three things to pull out.

**The first PC carries 15% of the variance, the second another
13%.** Together those two PCs explain 28% of the total variance
in 64-dimensional pixel space. That sounds modest, but the *2D
projection* it gives you is good enough to *visually*
distinguish most of the ten digit classes — see the header
image. PCA does not need to capture *all* the variance to
produce a useful low-dimensional view; it just needs to capture
the variance that distinguishes the structure you care about.

**95% variance is reached at ~30 components.** The standard
heuristic — "keep enough components to explain 95% of the
variance" — would compress 64 features to 30 here, with
reconstruction MSE about 0.77 (in original pixel-intensity
units, against an original feature range of 0–16). For
preprocessing pipelines (e.g. PCA before logistic regression
or before a neural net) this is the typical recipe: pick the
elbow on the cumulative-variance curve, throw away the rest.

**The from-scratch and sklearn implementations agree to
floating-point precision.** Maximum absolute difference in
projected coordinates is `~10⁻¹⁴` — the bit-level rounding
limit of 64-bit floats (the exact value varies with the BLAS
library version). PCA is one of the rare ML algorithms
where two correct implementations produce *byte-identical*
results, modulo the sign ambiguity inherent in SVD.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The cost story for PCA is dominated by the SVD step:

**Full SVD on the centred data matrix is `O(min(n · d², n² · d))`**
— typically reported as `O(n · d²)` when `n > d` (the common
case for tall data) and `O(n² · d)` when `d > n` (wide data,
e.g. genomics with thousands of features and hundreds of
samples). Memory is `O(n · d)` for the data plus `O(d²)` for
the eventual covariance matrix.

For very high-dimensional data (`d > 10⁴`) full SVD becomes
expensive. Two standard speedups:

**Truncated SVD** computes only the top-`k` singular values and
vectors, with cost roughly `O(n · d · k)`. Sklearn's
`PCA(n_components=k)` for moderate `k` and sklearn's
`TruncatedSVD` use this internally.

**Randomised SVD** (Halko, Martinsson & Tropp, 2011) — sample a
random projection of the data, do a small SVD on that, refine.
Cost `O(n · d · log k + (n + d) · k²)`, asymptotically much
faster for `k ≪ d`. This is what `PCA(svd_solver="randomized")`
runs and is the default in sklearn for large `n` or `d`.

For sparse data (text bag-of-words, gene expression with many
zeros), `TruncatedSVD` is the right tool. It does not centre
the data — centring would destroy sparsity — and is
mathematically equivalent to **Latent Semantic Analysis**
applied to the term-document matrix.

---

## Real-world ML and AI connections

PCA is everywhere — usually quietly, as the preprocessing step
that makes some other algorithm work:

**Visualisation of high-dimensional data.** The five-line recipe
"standardise → PCA(n_components=2) → scatter plot" is the
default first look at any new dataset with more than a few
features. It is fast, deterministic, and gives an honest answer
to "is there obvious structure here?" — which t-SNE and UMAP
sometimes hallucinate.

**Eigenfaces and classical face recognition.** Turk & Pentland's
1991 *Eigenfaces for Recognition* paper showed that the first
~50 principal components of a face-image database capture
enough information to recognise faces by nearest-neighbour
search in PC space. Eigenfaces dominated face recognition for
a decade and is still a teaching example in every computer
vision course.

**Whitening for neural network input.** Inputs with strongly
correlated features can slow neural network training. Whitening
— PCA followed by scaling each component to unit variance —
decorrelates the inputs and historically improved CNN training
significantly (Krizhevsky's ImageNet preprocessing pipeline,
for example).

**Latent Semantic Analysis (LSA / LSI).** Apply truncated SVD to
the term-document matrix and the resulting low-dimensional
"topic" representations were the dominant text-retrieval
technique through the 1990s and 2000s. Modern dense
embeddings have largely displaced LSA, but LSA is still a
strong baseline and the conceptual ancestor of every
embedding-based retrieval system.

**Compression and storage.** Dimensionality reduction with PCA
is a lossy compression scheme — keep `k` components, drop the
rest. The reconstruction error is the variance of the dropped
components. JPEG's DCT and modern image codecs do not use PCA
directly but use closely-related basis transforms.

**Preprocessing for clustering and classification.** Run PCA to
reduce 1000 features to 50, then run K-Means or logistic
regression on the 50-dim representation. This is the standard
recipe when the original features suffer from the curse of
dimensionality.

**Genomics and high-throughput biology.** Population structure
in genome-wide association studies is summarised by the first
few PCs of the genotype matrix. Single-cell RNA-seq pipelines
routinely apply PCA before clustering or visualisation.
Bioinformatics is where PCA gets perhaps the heaviest use in
2026.

**Anomaly detection.** Reconstruct each point from the top-`k`
PCs; points with high reconstruction error are unusual relative
to the dominant variance directions of "normal" data. A
straightforward, interpretable, and surprisingly effective
anomaly score.

The pattern is consistent: PCA is rarely the final answer, but
it is one of the most-used *intermediate* steps in machine
learning. If your dataset has more than a few dozen features
and you have not run PCA on it, you do not yet understand it.

---

## When NOT to use PCA

PCA's assumptions are strong, and when they break it gives
misleading results:

**When the structure is genuinely non-linear.** PCA is a *linear*
projection — it can only find directions that are linear
combinations of the original features. Data on a curved
manifold (digits drawn at varying angles, faces under varying
illumination, points on a sphere) will look smeared and
uninformative under PCA. Use **t-SNE**, **UMAP**, or an
**autoencoder** for non-linear manifolds.

**When you need interpretable individual features.** Each
principal component is a weighted combination of *all* the
original features. "PC1 = 0.31 × age + 0.27 × income − 0.18 ×
zip_code + ..." is not an interpretable summary you can show a
domain expert. Sparse PCA and other variants try to fix this,
but the unmodified PCs are linear blends that resist
interpretation.

**When your data is sparse.** Centring a sparse matrix destroys
the sparsity (zeros become non-zero means). For text
bag-of-words or one-hot-encoded categoricals, use
`TruncatedSVD` (no centring, preserves sparsity) instead of
plain PCA.

**When your features are non-Gaussian and you want to find
independent latent factors.** PCA finds *uncorrelated*
directions; it does not find *independent* directions unless the
data is jointly Gaussian. For independence you want
**Independent Component Analysis (ICA)** — the standard tool for
blind source separation in audio and EEG.

**When the variance you care about is small.** PCA orders
components by total variance. If the structure that matters to
your downstream task lives in a low-variance direction (a rare
class concentrated in a small region of feature space, for
example), PCA will discard it as noise. Supervised dimensionality
reduction methods (Linear Discriminant Analysis, partial least
squares, supervised autoencoders) preserve task-relevant
variance instead.

**When you have categorical features.** PCA assumes continuous
numeric features. For categorical data use **Multiple
Correspondence Analysis (MCA)**, the categorical analogue of
PCA, or one-hot encode and use truncated SVD.

---

## What comes next

Part 4 of the Unsupervised Learning track is **t-SNE**
(t-distributed Stochastic Neighbour Embedding) — the
non-linear visualisation algorithm that took over scientific
poster sessions in the early 2010s. Where PCA preserves global
structure and linearity, t-SNE preserves *local* neighbourhoods
and reveals manifold structure that PCA misses. The two are
complementary tools, often run side-by-side on the same dataset.

After t-SNE comes **UMAP** — newer, faster, and better at
preserving global structure than t-SNE — and then NMF, DBSCAN,
spectral clustering, and the rest of the unsupervised toolkit.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**pca.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/03-unsupervised-learning/03-principal-component-analysis/pca.py)

Run it with:

```bash
pip install numpy scikit-learn
python pca.py
```

It needs `numpy` and `scikit-learn`. The script implements PCA
from scratch via SVD of the centred data matrix, fits it to
the 64-dimensional digits dataset, compares against
scikit-learn's `PCA` (projections agree to floating-point
precision), and walks through the cumulative-variance curve
that tells you how many components to keep. The headline
insight worth pinning to the wall: **PCA finds the orthogonal
directions of maximum variance via SVD of the centred data
matrix, gives you a faithful low-dimensional summary when the
structure is linear, and is the universal first preprocessing
step on any dataset with more features than you trust**.

---

*This is Part 3 of the Unsupervised Learning track in the Algorithms in Python series. The companion script `pca.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 2](https://medium.com/p/37727954df03) covered Hierarchical Clustering. Part 4 will look at t-SNE — the non-linear visualisation algorithm that picks up where PCA's linearity assumption gives out.*
