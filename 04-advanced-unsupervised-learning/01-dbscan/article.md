# DBSCAN — No K, No Centroids, Just Density

### *Algorithms in Python --- Advanced Unsupervised Learning, Part 1*

---

The basic Unsupervised Learning track gave us eight algorithms
for finding structure in unlabelled data — K-Means and
hierarchical clustering for compact blobs, PCA / t-SNE / UMAP
for dimensionality reduction, NMF for parts-based decomposition,
spectral clustering for arbitrary shapes, association rule
mining for transactional sets. The clustering members of that
family share one
quiet assumption: *you tell the algorithm how many clusters to
look for*. K-Means demands `K` up front; hierarchical
clustering builds a dendrogram you cut at a chosen height;
spectral and the matrix-factorisation methods all need an
explicit number of components. Only association rule mining
sidesteps the question — and it does so by abandoning
clustering altogether.

Today we open the Advanced Unsupervised Learning track with the
algorithm that *finally* takes "how many clusters?" off the
hyperparameter list. **DBSCAN** (Density-Based Spatial
Clustering of Applications with Noise; Ester, Kriegel, Sander
& Xu, 1996) discovers clusters by *density* rather than count.
A cluster is wherever the data is locally dense; the
boundaries between clusters are wherever the data thins out;
and the points scattered in between — the ones that don't
belong to any dense region — are simply labelled **noise**
rather than forced into a cluster they don't fit. The number of
clusters falls out of the data instead of being declared.

DBSCAN is the workhorse of geospatial clustering (where you
genuinely don't know how many "hotspots" your data has), the
foundation of one of the most-used outlier-detection
strategies in production ML (DBSCAN's noise points are *exactly*
the anomalies), and the parent of HDBSCAN — its modern
successor that has displaced t-SNE-with-K-Means-on-top as the
default analysis pipeline for single-cell RNA-seq and many
modern embedding-clustering workflows.

This article builds DBSCAN from first principles. We will
define core, border, and noise points; walk through the
density-connectedness relation that defines DBSCAN clusters;
implement the algorithm from scratch in numpy; run it on the
two-moons dataset that K-Means famously fails on (this time
with added Gaussian noise to show DBSCAN's noise-detection in
action); compare with scikit-learn's `DBSCAN`; and finish with
the eps-and-min_samples tuning question, the algorithm's
limits, and a quick look at HDBSCAN as the right modern
default.

---

## The core idea: density defines clusters

K-Means' implicit definition of a cluster is *"the set of
points closest to a particular centroid"*. Hierarchical
clustering's is *"a connected subtree of merges below a chosen
height"*. Spectral clustering's is *"a connected component in
a similarity graph"*. All three are *partition-based* — every
point ends up in some cluster, no exceptions.

DBSCAN's definition is *density-based* — *"a connected region
where every point has at least a minimum number of other
points within a small radius"*. Two consequences fall out:

- **A cluster has no fixed size or shape.** It is whatever
  shape the dense region happens to take. Curved manifolds,
  elongated cigars, nested rings — DBSCAN handles all of them
  as long as the density inside the cluster is consistent and
  the density between clusters drops noticeably.
- **Some points belong to no cluster at all.** Points in
  sparse regions — outliers, far from any dense neighbourhood
  — are simply labelled *noise*. This is the cleanest treatment
  of outliers in any clustering algorithm we have built so
  far.

Two parameters control the density definition:

- **`eps`** — the neighbourhood radius. Two points are
  "neighbours" if they are within Euclidean distance `eps` of
  each other.
- **`min_samples`** — the density threshold. A point with at
  least `min_samples` neighbours within `eps` is in a dense
  region.

Together they define what "dense" means. The algorithm has no
opinion on how many clusters that produces; it just finds
whatever satisfies the density rule.

---

## Three categories of point

DBSCAN classifies every training point as one of three things:

**Core point.** A point with at least `min_samples` other
points (including itself) within `eps`. Core points sit
inside dense regions.

**Border point.** Not a core point, but within `eps` of some
core point. Border points are on the edge of a cluster —
attached to the dense interior but not themselves dense.

**Noise point.** Neither core nor border. No core points
within `eps`. These are the outliers.

Two points `p` and `q` are **density-connected** if there is a
chain of core points `p = p_0, p_1, ..., p_k = q` where every
consecutive pair `(p_i, p_{i+1})` is within `eps`. A **DBSCAN
cluster** is a maximal set of density-connected points. Every
core point belongs to exactly one cluster; every border point
attaches to one of its neighbouring clusters (which one is
implementation-defined when borders touch multiple cores);
every noise point belongs to no cluster.

The clusters fall out of these definitions. There is no
explicit count; the algorithm visits every point exactly once
and discovers however many clusters the density rule supports.

---

## The algorithm

In pseudocode:

```
dbscan(X, eps, min_samples):
    label = array of "unvisited" for every point
    cluster_id = 0
    for p in X:
        if label[p] != "unvisited":
            continue
        neighbours = points within eps of p
        if len(neighbours) < min_samples:
            label[p] = "noise"
            continue
        # p is a core point; start a new cluster
        cluster_id += 1
        label[p] = cluster_id
        seeds = neighbours - {p}
        while seeds:
            q = seeds.pop()
            if label[q] == "noise":
                # Border point: attach to this cluster
                label[q] = cluster_id
            if label[q] != "unvisited":
                continue
            label[q] = cluster_id
            q_neighbours = points within eps of q
            if len(q_neighbours) >= min_samples:
                # q is also a core point; expand the seed set
                seeds.update(q_neighbours)
    return label
```

The outer loop visits every point. The inner `while` loop
*expands* each cluster by chasing density-connectedness — when
a new core point is added, all of its neighbours become
candidates for the same cluster too. The recursion (here
implemented as a worklist of seeds) is what gives DBSCAN its
ability to follow arbitrary cluster shapes through feature
space.

The whole algorithm is a single pass with neighbourhood
queries. Done naively, computing "points within `eps`" for
every query is `O(n)` and the whole algorithm is `O(n²)`.
Done with a spatial index (KD-tree or ball tree), each query
drops to `O(log n)` and the algorithm becomes `O(n log n)` —
which is what sklearn does by default.

---

## A worked example

The companion script generates the two-moons dataset (the
classical curved-cluster benchmark from Part 2 of the basic
track) plus a handful of scattered noise points, runs DBSCAN
on it, and compares with K-Means on the same data.

```
DEMO 1 --- DBSCAN from scratch on moons + noise
  Data shape : 320 points, 2 features
            (300 moons + 20 uniform-random noise points)
  eps          : 0.2
  min_samples  : 5
  Clusters     : 2
  Noise points : 14
  ARI vs true labels : 0.961
```

```
DEMO 2 --- Same data, scikit-learn DBSCAN
  Clusters     : 2
  Noise points : 14
  ARI vs true labels : 0.961
  Agreement (ARI) with from-scratch DBSCAN : 1.000
```

```
DEMO 3 --- K-Means on the same data (for comparison)
  K            : 2 (forced)
  Noise points : 0 (K-Means cannot mark noise)
  ARI vs true labels : 0.205
```

Three observations.

**DBSCAN found exactly 2 clusters without being told.** The
algorithm has no `K` parameter — it discovered the structure
on its own and labelled 14 of the 20 injected random points as
noise (the other 6 happened to land close enough to a moon to
be absorbed as border points). The ARI against the true
labels — treating noise as its own "cluster" — is 0.961, which
is essentially perfect given that noise points have arbitrary
true labels from the dataset generator.

**The from-scratch and sklearn implementations agree
perfectly.** ARI 1.000 between them on every test. DBSCAN is
deterministic given the input data and parameters (no random
initialisation), so two correct implementations must produce
identical clusterings up to label permutation. Sklearn's
implementation is just faster.

**K-Means gets ARI 0.205.** As in Part 1 of the basic track,
K-Means draws a straight bisector through the moons and ends
up with two clusters that have no relationship to the true
labels. Worse, it has to assign every point to one of those
clusters — including the 20 noise points it should have
flagged. This is the canonical "DBSCAN handles outliers,
K-Means doesn't" demonstration.

---

## The hard part: picking eps and min_samples

DBSCAN's two parameters are easy to describe and hard to set.
`min_samples` is the more forgiving one — the rule of thumb
is `min_samples = 2 · d` for `d`-dimensional data (so 4 in 2D,
6 in 3D), increased modestly for noisy data. `eps` is much
harder. Set it too small and every point becomes noise
(everyone is isolated). Set it too large and every point ends
up in one giant cluster (everyone is connected).

The standard tool for picking `eps` is the **k-distance plot**:

1. For each point, compute the distance to its `k`-th nearest
   neighbour (where `k = min_samples - 1`).
2. Sort those distances ascending.
3. Plot the sorted distances.

The plot typically shows a long flat region (most points are
close to their neighbours — these are inside dense clusters)
followed by a sharp "elbow" where the distances begin to
climb steeply (these are points on the boundary between dense
and sparse regions). Set `eps` to the distance at the elbow.

In practice this gives a reasonable starting `eps` that needs
some tuning. For datasets where clusters have *different
densities*, no single `eps` will work for all of them — which
is why HDBSCAN was invented.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

DBSCAN's cost is dominated by neighbourhood queries:

**Naive implementation.** Computing the distance from every
point to every other point gives `O(n²)` queries each
costing `O(d)`, for `O(n² · d)` total. The memory cost is
`O(n²)` for the full distance matrix.

**With spatial index.** A KD-tree (low dimensions, `d ≲ 20`)
or ball tree (higher dimensions) lets each "points within
`eps`" query run in `O(log n)` rather than `O(n)`. The total
cost drops to `O(n · log n · d)`. Sklearn uses this by
default.

**Memory** is `O(n)` for the labels and `O(n · k)` for the
spatial index (with `k` = average neighbours per point).
Dramatically more scalable than spectral clustering or
hierarchical clustering.

**Practical limit** depends on dimensionality. In 2D or 3D
DBSCAN scales comfortably to millions of points. In high
dimensions (`d > 50`) the *curse of dimensionality* kicks in —
all points end up roughly equidistant, and the density
estimate based on `eps` becomes meaningless. For
high-dimensional data, run dimensionality reduction (PCA, UMAP)
first, then cluster the projection.

DBSCAN has no `predict` for new points — it is non-parametric.
You can extend by 1-NN against the original training points
(assign a new point to the cluster of its nearest training
neighbour, treating it as noise if no training point is within
`eps`), but this is post-hoc, not part of the algorithm.

---

## Real-world ML and AI connections

DBSCAN is the right tool whenever clusters and outliers matter
together:

**Geospatial clustering.** GPS pings, social-media check-ins,
traffic incident reports, retail-customer locations — all
naturally clustered by density, with many isolated points
that don't belong to any "hotspot". DBSCAN's density model
matches the data; you don't have to know how many hotspots
exist in advance. Production location-analytics pipelines at
companies from Uber to Foursquare use DBSCAN-family
algorithms heavily.

**Anomaly and fraud detection.** The noise points from a
DBSCAN run are *literally* the outliers — observations that
don't fit the dense patterns. Many anomaly-detection systems
build on top of this exact idea, sometimes with HDBSCAN or
LOF (Local Outlier Factor) for refinement. DBSCAN-as-anomaly
is a strong baseline in monitoring, network intrusion
detection, and credit-card fraud.

**Image segmentation.** Treat each pixel as a point in
position+colour space, cluster with DBSCAN, get segments of
similar contiguous pixels — a classical segmentation approach
that pre-dates SAM-style deep models but still works well on
medical imaging and microscopy where compute budgets are
tight.

**Astronomy and physics.** Galaxy clustering, particle-shower
analysis in collider experiments — domains where the number
of clusters is genuinely unknown and where outliers are
either interesting in themselves or need to be removed before
downstream analysis.

**Single-cell biology (via HDBSCAN).** Modern single-cell
RNA-seq pipelines often cluster cell embeddings with HDBSCAN
rather than with K-Means or Louvain — particularly when cell
types vary in abundance and density. The
`hdbscan` library, combined with UMAP for the embedding, is a
standard pipeline.

**As a noise filter.** Run DBSCAN, throw away the noise
points, then run your "real" clustering algorithm on what is
left. A surprisingly effective preprocessing pattern when
your data has a lot of stray points that confuse
compactness-based algorithms.

The pattern: DBSCAN is the right answer when (a) you don't
know how many clusters there are, (b) some points genuinely
don't belong to any cluster, and (c) the density is
reasonably consistent within clusters. Where any of those
fail, the next sections discuss alternatives.

---

## When NOT to use DBSCAN

DBSCAN's weaknesses are the flip side of its assumptions:

**When clusters have very different densities.** A single
global `eps` cannot accommodate both a dense cluster and a
sparse one. You'll either lose the sparse cluster (everyone
becomes noise) or merge dense clusters with their sparse
neighbours. **HDBSCAN** (Campello, Moulavi & Sander, 2013;
McInnes & Healy's widely-used 2017 implementation) solves this
by treating density at multiple scales simultaneously — the
modern recommendation when density varies.

**In high dimensions.** Above ~50 dimensions, Euclidean
distances stop discriminating between near and far points
(the curse of dimensionality), and DBSCAN's density model
breaks. Run PCA or UMAP first, then DBSCAN on the projection.

**When you need a fixed number of clusters.** If your
downstream pipeline needs *exactly* `K` clusters, DBSCAN
might give you 1, 3, or 7 depending on the data — and there's
no parameter that forces a specific count. Use K-Means or
spectral clustering instead.

**On very large datasets without an efficient spatial
index.** The default sklearn implementation uses a ball tree
which scales reasonably, but for truly huge datasets
(`n > 10⁶`) you may want approximate-nearest-neighbour
acceleration or a streaming variant.

**When you cannot tolerate noise points.** Some applications
require *every* point to be classified into some group. If
that's you, K-Means or Gaussian Mixture Models are better
matches; alternatively, post-process DBSCAN noise points by
1-NN assignment to the nearest cluster.

**When `eps` is genuinely hard to set.** If your domain has
no natural distance scale and the k-distance plot has no
clear elbow, DBSCAN's sensitivity to `eps` becomes a
liability. HDBSCAN removes the `eps` parameter entirely;
consider it.

---

## What comes next

Part 2 of the Advanced Unsupervised Learning track is
**Gaussian Mixture Models (GMMs)** — a *probabilistic*
clustering algorithm where each cluster is modelled as a
multivariate Gaussian distribution, and every point gets a
*soft* assignment (probabilities of belonging to each cluster
rather than a single hard label). GMMs are the natural
extension of K-Means when you need uncertainty estimates,
elliptical cluster shapes, or a proper probabilistic
framework — and they introduce the **EM (Expectation–Maximisation)
algorithm** that underlies a wide range of later methods.

After GMMs the track turns to **autoencoders**
(neural-network-based dimensionality reduction and anomaly
detection), **anomaly detection** as a topic in its own
right, and **Latent Dirichlet Allocation** for probabilistic
topic modelling.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**dbscan.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/04-advanced-unsupervised-learning/01-dbscan/dbscan.py)

Run it with:

```bash
pip install numpy scikit-learn
python dbscan.py
```

It needs `numpy` and `scikit-learn`. The script implements
DBSCAN from scratch with the standard core/border/noise
classification and density-connectedness expansion, fits it
to the two-moons dataset with added uniform-random noise,
compares against scikit-learn's `DBSCAN` (perfect agreement,
ARI 1.000 between them), and shows the K-Means baseline on
the same data for contrast (ARI 0.205 because the straight
bisector mangles both moons and absorbs the noise points). The
headline insight worth pinning to the wall: **DBSCAN defines
clusters by density rather than centroids, discovers the
number of clusters from the data instead of demanding it as a
parameter, and treats outliers as a first-class concept rather
than forcing every point into some cluster**.

---

*This is Part 1 of the Advanced Unsupervised Learning track in the Algorithms in Python series. The companion script `dbscan.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). The previous article (the final one in the basic Unsupervised Learning track) covered Association Rule Mining. Part 2 of this track will look at Gaussian Mixture Models — probabilistic clustering via Expectation–Maximisation.*
