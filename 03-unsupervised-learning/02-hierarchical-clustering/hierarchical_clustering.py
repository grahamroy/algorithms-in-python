"""
hierarchical_clustering.py --- companion code for "Hierarchical Clustering"
(Unsupervised Learning, Part 2).

Three demos:
  1. From-scratch agglomerative clustering with Lance-Williams
     distance updates, recovering 2/3/4/6 clusterings from a single
     fitted dendrogram by cutting at different heights.
  2. Comparison with scikit-learn's AgglomerativeClustering.
  3. The single-linkage chaining trick: on two interleaving
     half-moons, single linkage perfectly separates them while
     the other linkages collapse to ARI ~ 0.25.

Dependencies: numpy, scikit-learn. Runs in a few seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def make_blobs_dataset(seed=RNG_SEED):
    centres = np.array([
        [-3.0, -3.0],
        [ 3.0, -3.0],
        [ 0.0,  3.5],
    ])
    return make_blobs(n_samples=600, centers=centres,
                      cluster_std=0.9, random_state=seed)


def make_moons_dataset(seed=RNG_SEED):
    return make_moons(n_samples=300, noise=0.05, random_state=seed)


# ---------------------------------------------------------------------------
# From-scratch agglomerative clustering with Lance-Williams updates
# ---------------------------------------------------------------------------

class AgglomerativeFromScratch:
    """Agglomerative hierarchical clustering using Lance-Williams
    distance updates between active clusters. Pure numpy.

    The fit method runs all the way down to one cluster, recording
    the full merge history. labels(K) extracts a clustering at any K
    from the recorded merges in O(n) without re-fitting.
    """

    def __init__(self, linkage="ward"):
        if linkage not in ("single", "complete", "average", "ward"):
            raise ValueError(f"unknown linkage: {linkage}")
        self.linkage = linkage

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        # Initial pairwise (squared) Euclidean distance matrix.
        # We store squared distances for Ward; sqrt for the others.
        diff = X[:, None, :] - X[None, :, :]
        sq = np.einsum("ijk,ijk->ij", diff, diff)
        if self.linkage == "ward":
            D = sq.copy()  # squared
        else:
            D = np.sqrt(sq)
        np.fill_diagonal(D, np.inf)

        sizes = np.ones(n, dtype=float)
        # active[k] = True if cluster k is still alive
        active = np.ones(n, dtype=bool)
        # Each row in merges_: (cluster_a, cluster_b, distance, new_id)
        merges = []

        # Track the cluster id assigned at each merge: starts at n
        next_id = n
        # cluster_id_of[k] = the latest id assigned to active row k
        cluster_id = np.arange(n, dtype=int)

        for _ in range(n - 1):
            # Find the closest pair (i, j) among active clusters.
            # Mask inactive rows/columns by setting them to +inf
            mask = active[:, None] & active[None, :]
            D_masked = np.where(mask, D, np.inf)
            np.fill_diagonal(D_masked, np.inf)
            flat = np.argmin(D_masked)
            i, j = divmod(int(flat), n)
            if i > j:
                i, j = j, i
            d_ij = D[i, j]

            ni, nj = sizes[i], sizes[j]
            ck = next_id
            merges.append((cluster_id[i], cluster_id[j],
                           float(d_ij), ck))

            # Lance-Williams update: new distance from merged (i, j)
            # to every other active cluster m.
            new_d = np.full(n, np.inf)
            for m in np.where(active)[0]:
                if m == i or m == j:
                    continue
                nm = sizes[m]
                if self.linkage == "ward":
                    a = (nm + ni) / (nm + ni + nj) * D[i, m]
                    b = (nm + nj) / (nm + ni + nj) * D[j, m]
                    c = nm / (nm + ni + nj) * D[i, j]
                    new_d[m] = a + b - c
                elif self.linkage == "single":
                    new_d[m] = min(D[i, m], D[j, m])
                elif self.linkage == "complete":
                    new_d[m] = max(D[i, m], D[j, m])
                else:  # average
                    new_d[m] = (ni * D[i, m] + nj * D[j, m]) / (ni + nj)

            # Replace row i with the new merged cluster's distances;
            # mark row j inactive.
            D[i, :] = new_d
            D[:, i] = new_d
            D[i, i] = np.inf
            sizes[i] = ni + nj
            cluster_id[i] = ck
            active[j] = False
            next_id += 1

        self.merges_ = merges
        self.X_shape_ = X.shape
        self.n_ = n
        return self

    def labels(self, n_clusters):
        """Return per-row cluster labels by undoing the last
        (n - n_clusters) merges."""
        n = self.n_
        # parent[k] = id of the cluster that absorbed k
        parent = list(range(2 * n - 1))

        # Undo only merges that should HAPPEN -- the first
        # (n - n_clusters) merges are kept; the rest are undone.
        keep = n - n_clusters
        for k, (a, b, _d, ck) in enumerate(self.merges_):
            if k >= keep:
                break
            parent[a] = ck
            parent[b] = ck

        # Find ultimate root (within kept merges) of each original point
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        roots = np.array([find(i) for i in range(n)])
        # Re-map roots to dense 0..K-1 labels
        unique_roots = np.unique(roots)
        remap = {r: i for i, r in enumerate(unique_roots)}
        return np.array([remap[r] for r in roots])


# ---------------------------------------------------------------------------
# Demo 1 --- from scratch, multiple cut heights
# ---------------------------------------------------------------------------

def demo_from_scratch(X, y_true):
    banner("DEMO 1 --- Hierarchical clustering from scratch")

    print(f"  Data shape  : {X.shape[0]} points, {X.shape[1]} features")
    print(f"  Linkage     : Ward")
    print(f"  Lance-Williams updates, fit once, cut at any K")
    print()

    model = AgglomerativeFromScratch(linkage="ward").fit(X)
    for K in (2, 3, 4, 6):
        labels = model.labels(K)
        ari = adjusted_rand_score(y_true, labels)
        print(f"  Cut at K = {K:>2} clusters: ARI vs ground truth = "
              f"{ari:.3f}")
    return model


# ---------------------------------------------------------------------------
# Demo 2 --- sklearn comparison
# ---------------------------------------------------------------------------

def demo_sklearn(X, y_true, our_model):
    banner("DEMO 2 --- Same data, scikit-learn AgglomerativeClustering")

    sk = AgglomerativeClustering(n_clusters=3, linkage="ward")
    sk.fit(X)
    ari = adjusted_rand_score(y_true, sk.labels_)
    print(f"  Linkage     : Ward, K = 3")
    print(f"  ARI vs ground truth : {ari:.3f}")

    ours = our_model.labels(3)
    ari_pair = adjusted_rand_score(sk.labels_, ours)
    print(f"  Agreement (ARI) with from-scratch clustering : "
          f"{ari_pair:.3f}")


# ---------------------------------------------------------------------------
# Demo 3 --- linkage criterion comparison
# ---------------------------------------------------------------------------

def demo_linkages(X_blobs, y_blobs, X_moons, y_moons):
    banner("DEMO 3 --- Linkage criterion comparison "
           "(K = 3, ARI vs ground truth)")

    for lk in ("single", "complete", "average", "ward"):
        model = AgglomerativeClustering(n_clusters=3, linkage=lk)
        model.fit(X_blobs)
        ari = adjusted_rand_score(y_blobs, model.labels_)
        print(f"  {lk:<9} {ari:.3f}")

    print()
    print("  --- on harder data (two interleaving moons, n = 300) ---")
    for lk in ("single", "complete", "average", "ward"):
        model = AgglomerativeClustering(n_clusters=2, linkage=lk)
        model.fit(X_moons)
        ari = adjusted_rand_score(y_moons, model.labels_)
        print(f"  {lk:<9} {ari:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    X_blobs, y_blobs = make_blobs_dataset()
    X_moons, y_moons = make_moons_dataset()
    model = demo_from_scratch(X_blobs, y_blobs)
    demo_sklearn(X_blobs, y_blobs, model)
    demo_linkages(X_blobs, y_blobs, X_moons, y_moons)
    print()


if __name__ == "__main__":
    main()
