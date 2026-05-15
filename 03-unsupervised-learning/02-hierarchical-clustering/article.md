# Hierarchical Clustering — A Dendrogram Instead of a Number

### *Algorithms in Python --- Unsupervised Learning, Part 2*

---

In Part 1 we built K-Means: pick `K`, drop `K` centroids,
iterate Lloyd's algorithm until convergence. The whole thing
was elegant and fast — but it forced us to commit to a value
of `K` *before* looking at the data. The elbow method, the
silhouette score, the gap statistic all helped us pick `K`,
but they were post-hoc tools layered on top of an algorithm
that had already made the choice for us.

**Hierarchical Clustering** turns the problem inside out. It
does not ask you for `K`. Instead it builds a *tree* — the
**dendrogram** — that records every possible clustering of the
data, from one cluster of size `n` down to `n` clusters of size
one. You inspect the tree, find the level that looks right for
your problem, cut horizontally, and read off the clusters.
Picking `K` becomes a choice you make *after* you have seen
the structure of the data, not before.

The classical algorithm — agglomerative hierarchical clustering
— pre-dates K-Means and is older than computational statistics
itself. Sokal & Sneath formalised it for biological taxonomy in
their 1963 textbook *Principles of Numerical Taxonomy*. It is
still the default clustering method in genomics, phylogenetics,
single-cell biology, and any field where the cluster *hierarchy*
itself carries scientific meaning. The cost is computational:
hierarchical clustering is `O(n²)` in memory and at best
`O(n² log n)` in time, which puts a hard ceiling of roughly
`n ≈ 10⁴` on the dataset sizes it handles cleanly.

This article builds agglomerative clustering from first
principles. We will walk through the four standard *linkage
criteria* (single, complete, average, Ward), implement
agglomerative clustering from scratch, plot a dendrogram, cut
it at different heights to recover different clusterings,
compare with scikit-learn's `AgglomerativeClustering`, and
finish with the places hierarchical clustering is the right
tool — and the (large) places it is not.

---

## Agglomerative clustering — bottom-up

The most-used hierarchical algorithm is **agglomerative**: start
with every point as its own singleton cluster, repeatedly merge
the two closest clusters, stop when only one cluster remains.

```
agglomerative(X):
    clusters = [[i] for i in range(n)]   # singletons
    merges = []
    while len(clusters) > 1:
        i, j = argmin over pairs (a, b) of distance(clusters[a], clusters[b])
        merge_distance = distance(clusters[i], clusters[j])
        merges.append((i, j, merge_distance))
        clusters[i] = clusters[i] + clusters[j]
        delete clusters[j]
    return merges
```

The `merges` list — `n - 1` triples `(cluster_a, cluster_b,
distance)` — is the dendrogram. It records the entire merge
history. Cutting the dendrogram at any height `h` is equivalent
to undoing every merge with `merge_distance > h`, which leaves
you with whatever clusters existed just before that height.

The whole algorithm hinges on what `distance(cluster_a,
cluster_b)` means — the **linkage criterion**.

---

## Four ways to measure cluster distance

The Euclidean distance between two *points* is unambiguous. The
distance between two *clusters* of points is not — and this is
where the four standard linkage criteria come from.

**Single linkage** — distance between the two *closest* points,
one from each cluster:

```
d_single(A, B) = min { ‖a - b‖ : a ∈ A, b ∈ B }
```

Captures "are these two clusters touching anywhere?" Tends to
produce long, chain-like clusters when points form curved or
elongated structures — a feature on data with non-convex shapes,
a bug ("chaining") on noisy data where a few intermediate points
can stitch two unrelated clusters together.

**Complete linkage** — distance between the two *farthest*
points:

```
d_complete(A, B) = max { ‖a - b‖ : a ∈ A, b ∈ B }
```

Conservative: refuses to merge two clusters until every pair
across them is reasonably close. Produces compact, roughly
spherical clusters but is sensitive to outliers (one far-away
point inflates every distance involving its cluster).

**Average linkage** — average of all pairwise distances:

```
d_average(A, B) = mean { ‖a - b‖ : a ∈ A, b ∈ B }
```

A compromise between single and complete — less prone to
chaining than single, less outlier-sensitive than complete.
Often a sensible default when you have no prior reason to
prefer the others.

**Ward linkage** — minimises the *increase in within-cluster
sum of squares* caused by the merge:

```
d_ward(A, B) = (|A| · |B|) / (|A| + |B|)  ·  ‖μ_A − μ_B‖²
```

This is the agglomerative analogue of K-Means' objective. Ward
linkage tends to produce balanced, compact, roughly equal-size
clusters and is the default in scikit-learn. It is the
linkage to reach for when you want the dendrogram to look like
something K-Means would produce — but with the freedom to pick
`K` after seeing the merge history.

The choice of linkage matters as much as the choice of distance
metric. Same data + same algorithm + different linkage = often
substantially different clusterings.

---

## The dendrogram

The dendrogram is a binary tree drawn with the *merge distance*
on the y-axis. Every leaf is a data point. Every internal node
represents a merge, drawn at the y-coordinate equal to the
distance at which that merge happened. The longer the vertical
line above a node, the more "expensive" the next merge — i.e.
the more *separated* the two children were before being joined.

```
distance
   |
6  |               ┌──────────────────────────┐
   |               │                          │
4  |    ┌──────────┴──────┐         ┌─────────┴──────┐
   |    │                 │         │                │
2  |  ┌─┴─┐           ┌───┴───┐   ┌─┴─┐         ┌────┴───┐
   |  │   │           │       │   │   │         │        │
0  |  A   B           C       D   E   F         G        H
```

Two ways to read it.

**Choose K visually.** Cut the dendrogram with a horizontal
line at any height; the number of vertical lines it crosses is
the number of clusters at that height. A cut at `y = 3` in the
sketch above gives 4 clusters: `{A,B}`, `{C,D}`, `{E,F}`,
`{G,H}`. A cut at `y = 5` gives 2 clusters: `{A,B,C,D}` and
`{E,F,G,H}`.

**Choose K from the gaps.** The "right" `K` is often the one
just *below* a big vertical gap in the dendrogram. A long
unbroken vertical line means the next merge cost a lot — the
two clusters being merged were genuinely separated. Cut just
below that gap and the algorithm tells you: *these were the
real clusters; merging across this gap is artificial*.

This is the conceptual win over K-Means. K-Means picks one
clustering; hierarchical clustering produces *all* clusterings
simultaneously, lets you visualise the structure, and lets you
pick the level that matches your problem.

---

## A worked example

The companion script reuses the 3-cluster Gaussian-blob
dataset from Part 1 (600 points, three well-separated blobs in
2D), fits agglomerative clustering with Ward linkage, and
extracts clusterings at multiple cut heights.

```
DEMO 1 --- Hierarchical clustering from scratch
  Data shape  : 600 points, 2 features
  Linkage     : Ward
  Lance-Williams updates, fit once, cut at any K

  Cut at K =  2 clusters: ARI vs ground truth = 0.571
  Cut at K =  3 clusters: ARI vs ground truth = 1.000
  Cut at K =  4 clusters: ARI vs ground truth = 0.869
  Cut at K =  6 clusters: ARI vs ground truth = 0.577
```

```
DEMO 2 --- Same data, scikit-learn AgglomerativeClustering
  Linkage     : Ward, K = 3
  ARI vs ground truth : 1.000
  Agreement (ARI) with from-scratch clustering : 1.000
```

```
DEMO 3 --- Linkage criterion comparison (K = 3, ARI vs ground truth)
  single    1.000
  complete  0.995
  average   1.000
  ward      1.000

  --- on harder data (two interleaving moons, n = 300) ---
  single    1.000
  complete  0.597
  average   0.416
  ward      0.416
```

Three things to pull out.

**At `K = 3`, the Ward clustering is perfect.** ARI 1.000
against ground truth — the agglomerative algorithm recovered
the same three blobs K-Means did in Part 1, without ever being
told that `K` was 3.

**Cutting at the wrong K reveals structure rather than failure.**
Cut at `K = 2` and the ARI drops to 0.57 — not because the
algorithm got it wrong, but because two true clusters got
merged together (they had to; you asked for fewer clusters than
exist). Cut at `K = 4` and ARI drops to 0.87 — one true cluster
got over-split into two sub-clusters. The dendrogram shows
*all* of these clusterings simultaneously and lets you see why
none of them except `K = 3` is natural.

**Linkage choice matters most when data is non-convex.** On the
clean blobs every linkage recovers an essentially perfect
clustering (single, average, and Ward at 1.000; complete at
0.995). On the two interleaving half-moons (which K-Means
famously butchers), only **single** linkage gets ARI 1.000 —
its chaining behaviour follows each moon all the way around.
Ward and average collapse to ARI ≈ 0.42, complete to 0.60 —
they all prefer compact clusters, and the moons are anything
but. This is the rare case where chaining is exactly what you
want.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The headline numbers are stark:

**Memory is `O(n²)`.** The pairwise distance matrix has `n(n−1)/2`
entries. At `n = 10⁴` that is 50 million pairs — about
400 MB of doubles. At `n = 10⁵` it is 40 GB. There is no way
around this for a true hierarchical clustering — you need *all*
the pairs because any pair could be involved in a merge.

**Time is `O(n² log n)`** in the standard implementations
(SLINK for single, CLINK for complete, NN-chain for Ward;
sklearn uses these under the hood). The naive implementation
scans the full distance matrix at every merge, giving `O(n³)`
— what the from-scratch script does — fine up to `n ≈ 10³`,
slow above that.

**Prediction is awkward.** Hierarchical clustering is
non-parametric — it assigns clusters to the training set
directly, with no model that generalises to new points.
Predicting the cluster of a new query point usually means
"compute its distance to existing cluster centroids and assign
to the nearest", which is K-Means-style prediction grafted on
top, not the algorithm's natural output.

For datasets above ~10⁴ points the practical answer is one of:
**HDBSCAN** (a fast hierarchical-density variant we will meet
in track 04), **BIRCH** (which compresses the data into a
clustering feature tree first), or just K-Means with multiple
restarts.

---

## Real-world ML and AI connections

Hierarchical clustering's footprint is largest in fields where
the *hierarchy itself* matters scientifically:

**Phylogenetics and genomics.** UPGMA (Unweighted Pair Group
Method with Arithmetic Mean) — average linkage on genetic
distances — built the original molecular phylogenies in the
1970s. Modern phylogenetic trees use more sophisticated
likelihood-based methods (RAxML, BEAST), but agglomerative
clustering on distance matrices remains a standard quick first
pass.

**Single-cell RNA-seq.** Identifying cell types in
single-cell data routinely uses hierarchical clustering on
dimensionally-reduced gene-expression vectors, often with
Ward linkage. Tools like Seurat ship with several
agglomerative-clustering options as standard analysis paths.

**Document and topic taxonomies.** Hierarchical clustering of
TF-IDF or sentence-embedding vectors is the standard way to
build a tree of topics for navigation, faceted search, and
content recommendation. The hierarchy is often the *deliverable*
— "here is your topic taxonomy" — not just an intermediate
artefact.

**Customer segmentation hierarchies.** Where K-Means gives you
4 segments, hierarchical clustering gives you the full nested
structure: 2 broad segments → 4 sub-segments → 8 fine segments,
all consistent with each other. Useful when different parts of
the business want different segmentation granularities from
the same data.

**Image segmentation.** Hierarchical region merging is a
classical computer-vision technique for segmenting images into
nested regions of similar pixels. Modern deep-learning
segmentation (Mask R-CNN, SAM) has displaced it for object
boundaries, but hierarchical region-merging is still used in
medical imaging where interpretability matters.

**Anomaly detection.** A point that joins the dendrogram only
at very high distance (a merge near the top of the tree) is, by
construction, far from every other cluster. The dendrogram
gives you a free anomaly score: how high you had to go before
the point was absorbed.

The pattern: hierarchical clustering is rarely the *fastest*
unsupervised algorithm or the most-deployed in
high-throughput ML systems, but it is consistently the
algorithm of choice when scientists want to *see* the cluster
structure, not just consume it.

---

## When NOT to use hierarchical clustering

Walk away when:

**Your dataset has more than ~10⁴ points.** The `O(n²)` memory
makes pairwise hierarchical clustering impractical above that.
Use HDBSCAN, BIRCH, or fall back to K-Means with multiple
restarts.

**You need streaming / online clustering.** Hierarchical
clustering is fundamentally batch — it needs to see all the
data to compute the full distance matrix. Streaming variants
exist (CURE, BIRCH) but lose much of the elegance.

**You need probabilistic cluster assignments.** Agglomerative
clustering produces hard assignments. For "this point is 70%
cluster A, 30% cluster B" use a Gaussian Mixture Model.

**Density matters more than distance.** If your "clusters" are
defined by being dense regions of feature space (the way DBSCAN
thinks about it), hierarchical clustering with metric distances
will get the boundaries wrong. Use DBSCAN or HDBSCAN instead.

**You need to predict cluster membership for new data.** No
natural prediction step. K-Means or Gaussian Mixtures do this
out of the box.

**You don't trust your distance metric.** Hierarchical
clustering is *highly* sensitive to the distance/linkage
combination. Mis-specified distances can produce dendrograms
that look reasonable but reflect noise rather than structure.

---

## What comes next

Part 3 of the Unsupervised Learning track is **Principal
Component Analysis (PCA)** — the foundational dimensionality
reduction technique. Where K-Means and hierarchical clustering
look for *groups* in the data, PCA looks for *directions*: the
axes along which the data varies most. The two questions are
complementary, and PCA is often the preprocessing step that
makes clustering work in high-dimensional spaces (where
distance metrics break down for the reasons K-Means struggled
with).

After PCA the track turns to t-SNE and UMAP — non-linear
dimensionality reduction designed specifically for
visualisation — then DBSCAN, Gaussian Mixture Models, and the
rest of the unsupervised toolkit.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**hierarchical_clustering.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/03-unsupervised-learning/02-hierarchical-clustering/hierarchical_clustering.py)

Run it with:

```bash
pip install numpy scikit-learn
python hierarchical_clustering.py
```

It needs `numpy` and `scikit-learn`. The script implements
agglomerative hierarchical clustering from scratch with all
four standard linkages (single, complete, average, Ward),
fits the same 3-blob dataset from Part 1 to recover the
clustering at multiple cut heights, compares against
scikit-learn's `AgglomerativeClustering`, and demonstrates
on the two-moons dataset the one case where single linkage's
chaining behaviour beats the others. The headline insight
worth pinning to the wall: **agglomerative clustering builds a
dendrogram of every possible clustering at once, the linkage
criterion controls what shape the resulting clusters take, and
the algorithm lets you choose `K` after looking at the data
rather than before**.

---

*This is Part 2 of the Unsupervised Learning track in the Algorithms in Python series. The companion script `hierarchical_clustering.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 1](https://medium.com/p/52dbacba3b5a) covered K-Means Clustering. Part 3 will look at Principal Component Analysis — the foundational dimensionality reduction technique and a frequent preprocessing step for clustering algorithms in high-dimensional spaces.*
