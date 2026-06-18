# Spectral Clustering — When Clusters Don't Look Like Blobs

### *Algorithms in Python --- Unsupervised Learning, Part 7*

---

K-Means (Part 1) and hierarchical clustering with Ward linkage
(Part 2) share a quiet assumption: clusters are roughly
spherical blobs in Euclidean space. On the 3-Gaussian-blob
dataset we used for both, that assumption was correct and both
algorithms recovered the true structure perfectly. On the
two-moons dataset that K-Nearest Neighbours, decision trees,
and the rest of the supervised track all handled comfortably,
K-Means asked to find two clusters draws a straight line right
through both moons and produces two perfectly-wrong groups.

The reason is structural. K-Means' decision boundary is the
perpendicular bisector between cluster centroids — a straight
line in 2D, a hyperplane in higher dimensions. Hierarchical
clustering with Ward linkage shares the variance-minimisation
objective and behaves similarly. Both fail on data where the
true clusters are *connected* but not *compact* — half-moons,
nested circles, S-shapes, anything where the natural cluster
structure is topological rather than geometric.

**Spectral Clustering** (Shi & Malik, 2000; Ng, Jordan &
Weiss, 2002) is the algorithm that handles this. The key move
is to stop thinking of the data as points in Euclidean space
and start thinking of it as a *graph*. Each data point is a
node; an edge connects two nodes if the points are similar
(close in feature space). The cluster structure of the graph
is then captured by the **eigenvectors of the graph Laplacian**
— a matrix that encodes the graph's connectivity. The bottom
few eigenvectors give a new "spectral embedding" of the data
in which K-Means can finally find the right partitions.

Spectral clustering is the workhorse of image segmentation, the
foundation of community detection in social-network analysis,
and the mathematical engine that powers UMAP's spectral
initialisation (Part 5) and many other modern manifold-learning
techniques. The Laplacian eigenvectors that drive it are the
same objects that show up in graph signal processing, in
diffusion maps, and in the random-walk analysis of Markov
chains. Spectral clustering is the entry point to all of this
machinery.

This article builds spectral clustering from first principles.
We will define the affinity matrix and the graph Laplacian,
derive the algorithm from the relaxation of the *normalised
cut* objective, implement it from scratch, run it on the
two-moons dataset where K-Means famously fails, compare with
scikit-learn's `SpectralClustering`, and finish with the
practical considerations that determine when the algorithm
shines and when it falls over.

---

## The two-line summary

1. Build a weighted similarity graph from your data: nodes =
   data points, edge weights = pairwise similarities.
2. Take the bottom `K` eigenvectors of the graph Laplacian,
   stack them as columns of an `n × K` matrix, and run K-Means
   on the rows.

That is the algorithm. The hard work is hidden in two places:
constructing the right affinity graph (which similarity metric,
how to sparsify), and computing the Laplacian eigenvectors
(numerically delicate at scale).

Why does it work? Eigenvectors of the graph Laplacian
corresponding to small eigenvalues are *nearly constant within
connected components of the graph*. If the graph has `K`
disconnected components, the bottom `K` eigenvectors are
exactly the component indicator vectors (each one is `1` on
one component, `0` elsewhere). For graphs that are "almost
disconnected" — strong connections within clusters, weak
connections between — the bottom `K` eigenvectors are nearly
indicator vectors, and K-Means in this `K`-dim eigenvector
space trivially separates them.

---

## The affinity matrix

The first design choice is how to turn `n` data points into a
weighted graph.

**k-nearest-neighbour graph.** Connect each point to its `k`
nearest neighbours. The edge weight is either `1` (binary k-NN
graph) or a similarity score like `exp(-‖x_i - x_j‖² / σ²)`.
Default `k` is something like 10. This produces a sparse graph,
which makes the eigendecomposition tractable on larger
datasets.

**ε-neighbourhood graph.** Connect every pair of points within
distance `ε`. Simple but `ε` is hard to tune — too small and
the graph fragments, too large and everything connects to
everything.

**Fully-connected RBF graph.** Connect every pair, weight by
`exp(-‖x_i - x_j‖² / (2σ²))`. Dense `n × n` affinity matrix.
Best statistical properties; worst scalability. The natural
choice for small datasets.

The bandwidth `σ` of the RBF kernel is the most important knob
in spectral clustering. Too small and the affinities are
all-or-nothing (the algorithm finds singleton clusters); too
large and every point is similar to every other (the algorithm
collapses to one cluster). A common heuristic: set `σ` to the
median pairwise distance, or use a per-point adaptive scale
(Zelnik-Manor & Perona, 2004).

For the companion script we use a k-NN affinity graph (k = 10)
with a per-point local Gaussian scale (Zelnik-Manor & Perona,
2004) — on curved cluster shapes the sparse locally-scaled
graph separates the structure far more reliably than a dense
RBF kernel with a single global bandwidth.

---

## The graph Laplacian

Given the `n × n` affinity matrix `W`, the **degree matrix**
`D` is diagonal with `D_{ii} = Σ_j W_{ij}` — the total weight
incident to node `i`.

The **unnormalised Laplacian** is:

```
L = D - W
```

It is symmetric, positive semi-definite (all eigenvalues ≥ 0),
and its smallest eigenvalue is exactly `0` with eigenvector
`(1, 1, ..., 1) / √n`. For a graph with `K` connected
components, the multiplicity of the `0` eigenvalue is exactly
`K` — and the corresponding eigenvectors are indicator vectors
for the components. This is the foundational property: *the
spectrum of the Laplacian reveals the connectivity structure
of the graph*.

The **symmetric normalised Laplacian** is:

```
L_sym = I - D^(-1/2) · W · D^(-1/2)
```

Better-conditioned numerically and almost always preferred in
practice. Ng, Jordan & Weiss's algorithm (2002) uses
`L_sym`; Shi & Malik's original formulation uses the
random-walk normalised Laplacian `L_rw = I - D^(-1) · W`
which has the same eigenvalues and very similar eigenvectors.

For the rest of this article we use `L_sym`; it is the
sklearn default and gives the cleanest derivation.

---

## The full algorithm

Putting it together:

```
spectral_cluster(X, K, sigma, k_nn=None):
    # 1. Build the affinity matrix W
    W = rbf_affinity(X, sigma) (sparsify with k_nn if requested)

    # 2. Degree and normalised Laplacian
    d = W.sum(axis=1)
    D_inv_sqrt = diag(1 / sqrt(d))
    L_sym = I - D_inv_sqrt @ W @ D_inv_sqrt

    # 3. Take the K smallest eigenvectors of L_sym
    eigvals, eigvecs = eigh(L_sym)  # ascending order
    U = eigvecs[:, :K]               # n x K matrix

    # 4. Normalise rows of U (Ng-Jordan-Weiss recipe)
    U_norm = U / norm(U, axis=1, keepdims=True)

    # 5. K-Means in the K-dim spectral embedding
    labels = KMeans(n_clusters=K).fit_predict(U_norm)
    return labels
```

Two things worth noting.

The K-Means in step 5 is *not* clustering in the original
feature space. It is clustering in the K-dimensional space
spanned by the bottom K Laplacian eigenvectors. On this
transformed representation the clusters are well-separated
spheres even when they were curved manifolds in the original
space.

The row-normalisation in step 4 is the Ng-Jordan-Weiss trick.
Without it the algorithm still works but the eigenvector
coordinates have very different scales across rows and K-Means
struggles. The normalisation puts every row on the unit sphere
in `K` dimensions, which makes the K-Means clusters tight and
well-defined.

---

## A worked example

The companion script runs spectral clustering on three
datasets that span K-Means' failure modes:

- **3 Gaussian blobs** (where K-Means succeeds) — sanity check.
- **2 interleaving moons** (where K-Means fails) — the
  textbook spectral-clustering use case.
- **2 concentric circles** (where K-Means fails badly) — even
  harder structure.

```
DEMO 1 --- Spectral clustering from scratch on 3 datasets
  Affinity      : k-NN graph with local Gaussian scale (k=10)
  Laplacian     : symmetric normalised
  Clustering    : K-Means in eigenvector space

  Dataset          K   Spectral ARI     K-Means ARI
  ---------------  -   ------------     -----------
  3 blobs          3          1.000           1.000
  2 moons          2          1.000           0.234
  2 circles        2          1.000          -0.003
```

```
DEMO 2 --- Same data, scikit-learn SpectralClustering
  Dataset          K  Sklearn ARI     Agreement vs from-scratch
  ---------------  -  -----------     -------------------------
  3 blobs          3        1.000                         1.000
  2 moons          2        1.000                         1.000
  2 circles        2        1.000                         1.000
```

```
DEMO 3 --- Why does it work? Eigenvector inspection on moons
  Bottom 4 eigenvalues of L_sym (moons, n=300):
    lambda_0 = 0.0000  (always; constant vector)
    lambda_1 = 0.0000  (the cluster-separating eigenvector)
    lambda_2 = 0.0018  (start of "within-cluster" structure)
    lambda_3 = 0.0021

  Gap between lambda_1 and lambda_2 : 0.0018
  (Large gap = clean K=2 structure)
```

Three observations.

**Spectral clustering recovers the true labels on all three
datasets** — including the two that K-Means cannot solve.
On the moons, K-Means gets ARI 0.23 (essentially random with
a slight class-imbalance lift); on the concentric circles
K-Means gets ARI −0.003 (worse than random — the inner and
outer circle share a centroid, so K-Means' partition has no
relationship to the true structure). Spectral clustering
gets 1.000 on both.

**The eigenvalue gap predicts when spectral clustering will
work.** On the k-NN moons graph the two moons are *almost*
disconnected components, so `lambda_0` and `lambda_1` are
both essentially zero — the graph behaves as two near-isolated
pieces. `lambda_2` is the first "within-cluster" eigenvalue at
`0.0018`. The gap between `lambda_1` and `lambda_2` is what
makes the K=2 partition robust. The "eigengap heuristic" for
choosing `K` is exactly this: count the near-zero eigenvalues
before the first big jump.

**Spectral clustering agrees with sklearn on every example.**
The from-scratch implementation matches `SpectralClustering`
on all three datasets, which is reassuring since spectral
clustering has more parameter choices (affinity kernel,
normalisation variant) than most algorithms.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The two big costs:

**Affinity matrix construction.** Dense RBF is `O(n² · d)` in
time and `O(n²)` in memory. For sparse k-NN graphs the cost
drops to roughly `O(n · k · d)` using approximate nearest
neighbours, with `O(n · k)` memory. Above a few thousand
points the sparse graph is mandatory.

**Eigendecomposition of the Laplacian.** A full
eigendecomposition is `O(n³)`. The good news: we only need the
bottom `K` eigenvectors, and sparse iterative solvers (Lanczos,
ARPACK — what sklearn uses) compute these in roughly `O(n² · K)`
for dense matrices, or `O(n · K · log n)` for sparse ones.

In practice spectral clustering is limited to `n ≈ 10⁴` to
`10⁵` points without specialised infrastructure. For larger
datasets the field has produced **Nyström-extension spectral
clustering** (approximate the eigenvectors from a small random
sample of points), **landmark spectral clustering** (cluster
a small set of "landmarks" then propagate), and **anchor-graph
spectral clustering** — all of which trade exactness for
scalability.

The model itself is not parametric. There is no `predict`
method for new points (the same limitation t-SNE has). You can
extend a fitted clustering to new data via Nyström
approximation or by 1-NN against the original training points;
both are workarounds rather than first-class support.

---

## Real-world ML and AI connections

Spectral methods are the mathematical foundation of a large
chunk of modern ML:

**Image segmentation.** Shi & Malik's 2000 *Normalized Cuts
and Image Segmentation* paper kicked off the modern wave of
spectral methods in computer vision. Treat each pixel as a
node, edge weights from colour/texture similarity, partition
the resulting graph with normalised cuts. Pre-deep-learning
image segmentation pipelines were heavily spectral; modern
SAM-style segmenters use neural networks, but graph-cut
post-processing of their outputs often still uses spectral
ideas.

**Community detection in social networks.** Spectral clustering
on the adjacency matrix of a social network finds
communities — groups of users with dense internal connections
and sparse external ones. The Newman-Girvan modularity and the
many spectral-based community-detection algorithms that
followed (spectral modularity maximisation, Bethe Hessian
methods) all live in this neighbourhood.

**Manifold learning.** Laplacian Eigenmaps (Belkin & Niyogi,
2003), Locally Linear Embedding, Diffusion Maps, and UMAP all
use Laplacian-eigenvector machinery. UMAP's *spectral init*
step is literally the spectral-clustering eigenvector
computation; we hit it in Part 5 without unpacking the
mathematics, which we have now done here.

**Single-cell RNA-seq.** Some single-cell pipelines use
spectral clustering on a k-NN graph in PCA-reduced
gene-expression space — particularly in workflows where Louvain
or Leiden community-detection (which is what Scanpy / Seurat
default to) is too coarse.

**Graph signal processing.** The Laplacian eigenvectors are
the "Fourier basis" for graph signals — the foundation of an
entire field that generalises classical Fourier analysis to
arbitrary graph structures. Graph Neural Networks build on
exactly this machinery.

**Initialisation and dimensionality reduction.** UMAP's
spectral init, the spectral-embedding step in Laplacian
eigenmaps, and similar methods are spectral clustering
without the final K-Means step — useful whenever you want a
low-dimensional embedding that respects graph connectivity.

The pattern: spectral clustering is rarely the headline
algorithm in production, but the *Laplacian spectrum* it
relies on is one of the most widely-used mathematical objects
in unsupervised learning and graph machine learning generally.

---

## When NOT to use spectral clustering

The algorithm's limitations are real:

**When K-Means already works.** On compact spherical clusters,
K-Means is faster, simpler, and just as accurate. Reach for
spectral clustering only when you suspect the clusters have
non-trivial shapes.

**When `n` is very large.** Without approximation methods,
spectral clustering does not scale past ~10⁴ points cleanly.
For larger datasets use sparse k-NN graphs, Nyström
approximations, or community-detection algorithms (Louvain,
Leiden) on a k-NN graph instead.

**When the affinity kernel is hard to tune.** RBF bandwidth
choice can dominate the results. If you don't have a principled
way to set `σ`, the algorithm is fragile.

**When you need to embed new points.** Like t-SNE and
hierarchical clustering, spectral clustering does not
generalise to unseen data. For workflows that need fast
inference on new points, train a parametric model on the
spectral-clustering labels as a surrogate.

**When you need probabilistic cluster assignments.** Spectral
clustering produces hard assignments. For soft probabilities
use Gaussian Mixture Models or fuzzy variants of spectral
clustering.

**When the data has very different cluster sizes.** The
normalised Laplacian helps but does not fully solve this; a
dominant cluster can still bias the embedding. Hierarchical
clustering with average linkage is often more forgiving.

---

## What comes next

Part 8 of the Unsupervised Learning track is **Association
Rule Mining** — the classical "people who bought X also bought
Y" recipe behind market-basket analysis. Apriori and FP-Growth
extract frequent itemsets from transaction data and turn them
into actionable rules. It is the most domain-specific of the
unsupervised algorithms we cover — useful for transactional /
categorical data, less applicable to numeric vectors — but it
is the algorithm that built recommendation systems before
collaborative filtering took over, and it still ships in every
serious business-analytics stack.

After Association Rule Mining the Unsupervised Learning track
wraps. The next big track — Advanced Unsupervised Learning —
opens with DBSCAN, Gaussian Mixture Models, and friends.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**spectral_clustering.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/03-unsupervised-learning/07-spectral-clustering/spectral_clustering.py)

Run it with:

```bash
pip install numpy scipy scikit-learn
python spectral_clustering.py
```

It needs `numpy`, `scipy`, and `scikit-learn`. The script implements
spectral clustering from scratch — k-NN affinity with local
Gaussian scale, symmetric
normalised Laplacian, eigendecomposition, K-Means on the
row-normalised eigenvectors — and applies it to three datasets
spanning K-Means' failure modes: 3 Gaussian blobs (control),
2 interleaving moons (curved manifolds), and 2 concentric
circles (nested structure). On all three it matches the true
labels with ARI 1.000. The K-Means baseline matches on the
blobs and fails on the other two. The headline insight worth
pinning to the wall: **spectral clustering converts data into
a graph, decomposes the graph's Laplacian, and runs K-Means
in the eigenvector space where curved clusters become
spherical and easy**.

---

*This is Part 7 of the Unsupervised Learning track in the Algorithms in Python series. The companion script `spectral_clustering.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 6](https://medium.com/p/4bddfe26f8f7) covered Non-Negative Matrix Factorisation. Part 8 will look at Association Rule Mining — the market-basket-analysis algorithms behind classical recommender systems.*
