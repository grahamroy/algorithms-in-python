# K-Means Clustering — The Most-Used (and Most-Misused) Clustering Algorithm

### *Algorithms in Python --- Unsupervised Learning, Part 1*

---

The supervised-learning track is behind us. From linear
regression to support vector machines, every algorithm we built
shared the same problem setup: each training example came with
a label `y`, and the model's job was to learn the mapping
`x ↦ y`. Today the labels go away.

In **unsupervised learning** the algorithm sees only `X` and has
to discover the structure of the data on its own — which points
group together, which directions in feature space carry the
most variance, which examples are outliers, which features are
redundant. There is no `y` to score against, no accuracy on a
held-out set, no neat right-or-wrong answer. The success
criterion is itself part of the problem definition.

The classic entry point to unsupervised learning is **K-Means
Clustering**. Pick a number of clusters `K`. Drop `K` centroids
into the feature space. Assign every data point to the nearest
centroid. Recompute each centroid as the mean of the points
assigned to it. Repeat until the assignments stop changing. The
algorithm — Lloyd's algorithm, named after Stuart Lloyd's
unpublished 1957 Bell Labs memo and the published 1982 version
— is so simple it fits in fifteen lines of numpy, and so
useful that it has been re-invented in every applied field that
ever needed to group things.

This article builds K-Means from first principles. We will
derive Lloyd's algorithm, explain why the initial centroid
placement matters and how `k-means++` fixes it, walk through
the objective the algorithm is actually minimising (and where
that objective leads it astray), implement it from scratch,
compare with scikit-learn's `KMeans`, and finish with the
surprisingly long list of places K-Means is *quietly* doing
real work — image compression, vector quantisation, the IVF
index inside every vector database, customer segmentation, and
the initialisation step of half of the more sophisticated
clustering and density-estimation algorithms.

---

## The algorithm

Given data `X` with `n` rows in `d` dimensions and a chosen
number of clusters `K`, Lloyd's algorithm alternates two steps
until convergence.

**Step 1 — Assignment.** For each point `x_i`, find the
nearest centroid `μ_k` (by Euclidean distance) and assign the
point to cluster `k`.

```
for i in 1..n:
    cluster[i] = argmin_k  ‖x_i − μ_k‖²
```

**Step 2 — Update.** For each cluster `k`, recompute its
centroid as the mean of the points currently assigned to it.

```
for k in 1..K:
    μ_k = mean(x_i for i in cluster k)
```

Iterate Step 1 then Step 2 until the assignments stop changing
(or a maximum iteration count is hit). The output is `K`
centroids and a cluster label `1..K` for every input point.

The whole algorithm is:

```
fit(X, K, max_iter):
    μ = initialise K centroids (see below)
    for iter in 1..max_iter:
        old_clusters = clusters
        clusters = assign(X, μ)        # Step 1
        μ = recompute(X, clusters)     # Step 2
        if clusters == old_clusters:
            break
    return μ, clusters
```

That is the entire algorithm. No gradient, no learning rate, no
loss landscape — just alternation between two cheap operations
that each have a closed-form solution given the other.

---

## What is actually being minimised?

K-Means looks heuristic, but it is doing coordinate descent on
a perfectly well-defined objective: the **within-cluster sum
of squares** (WCSS), also called *inertia*:

```
J(μ, clusters) = Σ_k  Σ_{i ∈ cluster k}  ‖x_i − μ_k‖²
```

Read this as: for each cluster, sum the squared distance from
every member to that cluster's centroid; then sum across
clusters. Lower `J` means tighter, more compact clusters.

Lloyd's algorithm minimises `J` by alternating optimisation
over its two variable groups:

- **Fix `μ`, optimise clusters.** Each point's contribution
  `‖x_i − μ_k‖²` is minimised by assigning it to the closest
  centroid. That is exactly Step 1.
- **Fix clusters, optimise `μ`.** For each cluster the sum
  `Σ ‖x_i − μ_k‖²` is minimised (over `μ_k`) by setting `μ_k`
  to the mean of the assigned points. That is exactly Step 2.

Because each step is a *minimiser* of `J` (given the other
variable held fixed), `J` is guaranteed to decrease at every
iteration. The algorithm always converges — in a finite number
of steps, since there are finitely many possible assignments.

But it converges to a **local** minimum of `J`, not the global
one. The objective `J` is non-convex in `μ` (the
cluster-assignment choice is combinatorial), and Lloyd's
algorithm can settle into bad local minima if the initial
centroids are placed badly. Which is why the initialisation
step matters more than the algorithm itself.

---

## Initialisation matters — `k-means++`

The simplest initialisation is to pick `K` random data points
as the initial centroids. It works most of the time, but on
adversarial inputs (long thin clusters, very different sizes,
or rare classes) it can land in dramatically suboptimal local
minima. Multiple restarts (sklearn's `n_init`) help by running
the whole thing many times and keeping the best, but each
restart costs a full convergence.

**k-means++** (Arthur & Vassilvitskii, 2007) is a smarter
initialisation that spreads the initial centroids out before
Lloyd's algorithm runs:

```
1. Pick the first centroid uniformly at random from the data.
2. For each subsequent centroid k = 2..K:
   - Compute D(x) = distance from each point to its nearest
     already-chosen centroid.
   - Pick the next centroid by sampling a data point with
     probability proportional to D(x)².
```

Points far from the existing centroids are exponentially more
likely to be chosen as new centroids. The squared-distance
weighting means that any region of the dataset not yet covered
by a centroid is overwhelmingly likely to attract one. After
`K` iterations of this rule, the initial centroids are
well-spread and Lloyd's algorithm typically converges quickly
to a near-global minimum.

Arthur & Vassilvitskii proved that k-means++ gives an
*expected* `O(log K)`-competitive solution with respect to the
optimal — a remarkable result given how cheap the procedure is.
In practice, k-means++ is so much better than random init that
it has been the default in sklearn (and every other major
library) for over a decade.

---

## The hard problem: picking K

Lloyd's algorithm tells you the best partition into `K`
clusters. It does not tell you what `K` should be. Picking `K`
is the central practical difficulty of K-Means and the source
of most of the "K-Means doesn't work for my data" stories that
people tell at meetups.

A handful of standard tools:

**Elbow method.** Plot the WCSS objective `J` as a function of
`K`. As `K` increases, `J` decreases monotonically (more
clusters = tighter clusters = lower WCSS). The "elbow" is the
value of `K` past which adding another cluster barely reduces
`J`. Reading off the elbow is subjective and works best when
the true `K` is small (2 to 6) and the clusters are
well-separated. On real data the elbow is often ambiguous.

**Silhouette score.** For each point, compute `(b − a) / max(a, b)`
where `a` is the mean distance to other points in the same
cluster and `b` is the mean distance to points in the nearest
other cluster. The score is in `[-1, +1]`; higher is better.
Average across the dataset to get a single quality number for
each `K`. Pick the `K` with the highest silhouette. Works
better than the elbow on ambiguous data but is expensive on
large `n`.

**Gap statistic** (Tibshirani et al, 2001). Compare the WCSS
of your data at each `K` to the WCSS of uniform-random data of
the same shape, also clustered at each `K`. The optimal `K` is
the one with the largest gap between observed and reference
WCSS. Theoretically principled but computationally heavy.

**Davies–Bouldin index**, **Calinski–Harabasz**, **BIC** (after
fitting a GMM rather than K-Means) — all variations on the
same theme: pick the `K` that balances cluster tightness
against cluster separation.

Honest summary: every one of these gives an answer, and the
answers disagree on real data. The pragmatic recipe is to use
the elbow as a starting hypothesis, run a silhouette score to
confirm, and reach for domain knowledge to break ties. K is
ultimately a *choice* — and the right choice depends on what
you intend to *do* with the clusters.

---

## A worked example

The companion script generates a 3-cluster Gaussian-blob
dataset, fits a from-scratch K-Means with k-means++ init,
compares against scikit-learn's `KMeans`, and finishes with an
elbow plot in tabular form.

```
DEMO 1 --- K-Means from scratch on Gaussian blobs
  Data shape : 600 points, 2 features
  True K     : 3
  Chosen K   : 3
  Init       : k-means++
  Converged in 2 iterations
  Final inertia (WCSS) : 918.7
  Adjusted Rand Index vs true labels : 1.000
```

```
DEMO 2 --- Same data, scikit-learn KMeans
  Converged in 2 iterations
  Final inertia (WCSS) : 918.7
  Adjusted Rand Index vs true labels : 1.000
  Agreement (ARI) between from-scratch and sklearn: 1.000
```

```
DEMO 3 --- Elbow method: WCSS vs K
     K       WCSS
   ---   --------
     2     4598.9
     3      918.7
     4      819.1
     5      711.8
     6      639.9
     7      560.2
     8      449.0
```

The Adjusted Rand Index (ARI) compares the clustering against
the true labels we secretly generated the data with. ARI of
`1.000` means the clusters are *exactly* the ground truth up
to relabelling — perfect recovery, which is the easy outcome
you expect on a clean 3-blob problem with good initialisation.

The elbow plot in Demo 3 is textbook. From `K = 2` to `K = 3`
the WCSS drops by 80% (`4599 → 919`) — adding the third
cluster genuinely captures a third group in the data. From
`K = 3` onwards each additional cluster reduces WCSS by only
10–15% — the algorithm is now slicing already-tight blobs into
sub-blobs, which costs degrees of freedom for marginal return.
The elbow at `K = 3` is unambiguous. On real data it usually
is not.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

K-Means is cheap. Each iteration is `O(n · K · d)` — every
point computes its distance to every centroid, every centroid
recomputes from its assigned points. Lloyd's algorithm
typically converges in 10 to 50 iterations, even on large
datasets, so total cost is `O(I · n · K · d)` with `I` small.
Compare against tree ensembles (`O(N · n · d · depth)` for
training a forest of `N` trees) or kernel SVMs (`O(n²)` or
worse): K-Means is one of the cheapest non-trivial algorithms
in machine learning.

For very large `n` there is **mini-batch K-Means** (Sculley,
2010), which updates centroids on small random batches rather
than recomputing the full means each iteration. Sub-linear in
the dataset size and converges quickly. It is the workhorse
implementation behind `MiniBatchKMeans` and the basis for
streaming clustering at scale.

---

## What K-Means quietly assumes

The reason K-Means is "most misused" is that it makes four
quiet assumptions about the data that are routinely violated:

**Clusters are spherical.** Lloyd's algorithm uses Euclidean
distance, which is rotation-invariant but not scale-invariant.
A cluster that is genuinely elongated along one direction (a
narrow cigar shape, not a sphere) will be carved up into
multiple K-Means clusters even though it should obviously be
one. The fix is feature standardisation, or — if that does not
help — switching to a model class that handles non-spherical
clusters (Gaussian Mixture Models with full covariance,
DBSCAN, spectral clustering).

**Clusters are roughly equal-size.** The means-based update
gives every point equal weight in its cluster's centroid. If
one true cluster has 10× as many points as another, the larger
cluster's centroid will be pulled toward its own centre and
will tend to absorb the nearby smaller cluster. Subsampling
the larger cluster, or switching to Gaussian Mixture Models
(which model cluster size explicitly via mixing weights), fixes
this.

**Clusters have similar variance.** A cluster with high
variance ("noisy") and a cluster with low variance ("tight")
side by side: Euclidean distance puts roughly equal weight on
both, but the noisy cluster's true membership will spread far
into the tight cluster's territory and absorb its outliers.

**There are no obvious outliers.** A single far-away point will
become its own centroid in any restart that happens to seed on
it, distorting the rest of the assignment. Remove outliers
upstream or use a robust variant (k-medoids, which uses medians
or actual data points as centres rather than means).

When these assumptions fail, the answer is usually not "tweak
K-Means parameters" — it is "use a different clustering
algorithm". The next article in this track (Hierarchical
Clustering) handles unequal-size clusters; Part 4 (t-SNE) and
Part 5 (UMAP) are visualisation tools that often reveal cluster
structure K-Means alone misses; Part 7 (Spectral Clustering)
and the DBSCAN article in track 04 handle non-convex shapes.

---

## Real-world ML and AI connections

K-Means is one of the most-deployed unsupervised algorithms,
usually hiding inside something else:

**Image compression and colour quantisation.** Run K-Means with
`K = 16` or `K = 256` on the pixels of an image (treating each
pixel as a 3-vector of RGB values). Replace every pixel with
its assigned centroid. The result is a palette-quantised image
indistinguishable from the original at common bit depths, with
a much smaller storage footprint. This is the classical
**Lloyd–Max quantiser** (and Lloyd's algorithm is named after
exactly this application).

**Vector quantisation for compression.** Long predates ML —
LBG (Linde–Buzo–Gray, 1980) is K-Means for vector quantisation
in audio compression. Modern codecs (Opus, AAC) still use VQ
codebooks derived from K-Means-like procedures.

**Inverted file (IVF) index for vector databases.** Foundations
Part 12 (Vector Indexes) introduced IVF as the workhorse
"coarse quantiser" inside FAISS and other vector databases.
The "training" step of IVF is K-Means on the database vectors,
producing `K = √N` (or so) centroids that partition the space.
Queries find the nearest centroids, then search only those
partitions. The big idea behind every modern vector DB has
K-Means at the bottom.

**Customer segmentation.** The most common business-analytics
use case. RFM (Recency, Frequency, Monetary) feature
engineering followed by K-Means on the resulting 3D feature
space, with `K = 4` or `K = 5` chosen by elbow, gives the
canonical "VIPs / loyal / new / lapsed / churned" customer
segments that drive marketing decisions. Probably the most
economically valuable single application of K-Means.

**Document and word clustering.** K-Means on TF-IDF vectors
(or, more recently, sentence-embedding vectors) is the
default approach to topic discovery on a document collection
when LDA is overkill and supervised classification is
unavailable. The same approach powers RAG-pipeline chunk
clustering for retrieval optimisation.

**Initialisation for harder algorithms.** Gaussian Mixture
Models are typically initialised with K-Means centroids
(sklearn's default). Self-organising maps, several
deep-learning quantisation methods (Product Quantisation in
Foundations Part 12, VQ-VAEs, residual vector quantisation in
modern audio codecs like SoundStream) all use K-Means as the
training step for their codebooks.

**Anomaly detection.** Distance to the nearest cluster centroid
is a cheap, interpretable anomaly score. Points far from every
centroid are unusual relative to the modelled structure of the
"normal" data.

The pattern: K-Means is rarely the headline algorithm in
production. But it is almost always the workhorse at the
bottom of something more interesting — and a non-trivial
fraction of the world's running ML systems contain a K-Means
step somewhere in their pipeline.

---

## When NOT to use K-Means

The places K-Means is the wrong tool:

**Non-convex / elongated / nested clusters.** Two interleaving
half-moons (the dataset we used through the supervised track)
have two genuine clusters, but K-Means with `K = 2` will draw
a straight line through them and produce two perfectly-wrong
clusters. Use DBSCAN, spectral clustering, or a Gaussian
mixture with full covariance.

**Clusters of very different sizes.** A dominant class plus a
rare class is a hard problem for K-Means. The rare class either
gets absorbed into the nearest dominant cluster or gets its own
centroid by luck. Either way, the rare class is under-served.

**High-dimensional sparse data.** Bag-of-words and similar
sparse vectors are not Euclidean-friendly. Euclidean distance
between two sparse vectors is dominated by their lengths
rather than their semantic similarity. Switch to cosine
distance (spherical K-Means) or to clustering algorithms
designed for sparse data.

**When you don't know K and the elbow is ambiguous.** If the
elbow plot does not have a clean elbow and the silhouette
disagrees, K-Means is asking you a question you cannot answer.
Try **hierarchical clustering** (no `K` required, returns a
dendrogram you can cut at any level) or **DBSCAN** (decides
clusters by density, not by `K`).

**When the cluster assignments need to be probabilistic.**
K-Means gives hard assignments. If you want "this point is 70%
cluster A and 30% cluster B" — use a Gaussian Mixture Model
fit by EM, which gives soft per-cluster probabilities.

**Categorical or mixed data.** K-Means' mean update only makes
sense for continuous features. For categorical data use
**K-modes** (mode update instead of mean) or **K-prototypes**
(mix of K-Means and K-modes for mixed data types).

---

## What comes next

Part 2 of the Unsupervised Learning track is **Hierarchical
Clustering** — the family of algorithms that build a tree (the
*dendrogram*) of cluster merges and let you choose `K` *after*
fitting by cutting the tree at any level. Hierarchical
clustering handles unequal-size clusters more gracefully than
K-Means, makes no assumption about the number of clusters, and
produces a much richer summary of the data's structure — at the
cost of being substantially slower (`O(n² log n)` for the
classical agglomerative version).

After hierarchical clustering, the unsupervised track turns to
**dimensionality reduction** with PCA, t-SNE, and UMAP — three
algorithms that compress high-dimensional data into 2D for
visualisation and downstream analysis. Then DBSCAN for
density-based clustering, Gaussian Mixture Models for soft
probabilistic clusters, and the rest of the unsupervised
toolkit.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**k_means.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/03-unsupervised-learning/01-k-means-clustering/k_means.py)

Run it with:

```bash
pip install numpy scikit-learn
python k_means.py
```

It needs `numpy` and `scikit-learn`. The script implements
Lloyd's algorithm with `k-means++` initialisation from scratch,
fits it to a 3-cluster Gaussian-blob dataset, compares against
scikit-learn's `KMeans` (clusters agree perfectly up to
relabelling), and walks through an elbow plot in tabular form
that shows the unambiguous `K = 3` answer for this dataset. The
headline insight worth pinning to the wall: **K-Means is
coordinate descent on within-cluster sum of squares; it works
beautifully on spherical, equal-size clusters with k-means++
initialisation, and fails on everything else — which is most
real-world data**.

---

*This is Part 1 of the Unsupervised Learning track in the Algorithms in Python series, and the first article after the supervised track concluded. The companion script `k_means.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). The previous article (the final supervised one) covered Support Vector Machines. Part 2 of this track will look at Hierarchical Clustering — the algorithm family that builds a tree of cluster merges and lets you pick `K` after fitting.*
