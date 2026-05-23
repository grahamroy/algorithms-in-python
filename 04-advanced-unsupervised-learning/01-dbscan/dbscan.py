"""
dbscan.py --- companion code for "DBSCAN"
(Advanced Unsupervised Learning, Part 1).

Three demos:
  1. DBSCAN from scratch (core/border/noise + density-connected
     expansion) on the two-moons dataset plus injected noise.
  2. Comparison with scikit-learn's DBSCAN (identical labels
     after permutation alignment).
  3. K-Means on the same data for the textbook K-Means-fails-
     here contrast.

Dependencies: numpy, scikit-learn. Runs in well under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score


SEPARATOR = "=" * 72
RNG_SEED = 7
NOISE = -1
UNVISITED = -2


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Dataset: two moons + scattered noise
# ---------------------------------------------------------------------------

def make_dataset(seed=RNG_SEED):
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05,
                                  random_state=seed)
    rng = np.random.default_rng(seed)
    # Sprinkle 20 uniform-random noise points across the bounding box
    x_min, x_max = X_moons[:, 0].min() - 0.3, X_moons[:, 0].max() + 0.3
    y_min, y_max = X_moons[:, 1].min() - 0.3, X_moons[:, 1].max() + 0.3
    noise = rng.uniform(
        low=[x_min, y_min], high=[x_max, y_max], size=(20, 2)
    )
    X = np.vstack([X_moons, noise])
    # Give noise points label -1 in the "true" labels (they don't
    # belong to either moon)
    y = np.concatenate([y_moons, np.full(20, -1)])
    # Shuffle so noise is interleaved
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# DBSCAN from scratch
# ---------------------------------------------------------------------------

class DBSCAN:
    """Density-based spatial clustering with noise.

    Naive O(n^2) implementation for clarity. For larger datasets
    sklearn uses a KD-tree / ball-tree for O(n log n) per query.
    """

    def __init__(self, eps=0.2, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        # Pairwise squared distances
        diff = X[:, None, :] - X[None, :, :]
        sq = (diff ** 2).sum(axis=2)
        eps_sq = self.eps ** 2

        labels = np.full(n, UNVISITED, dtype=int)
        cluster_id = -1

        for p in range(n):
            if labels[p] != UNVISITED:
                continue
            neighbours = np.where(sq[p] <= eps_sq)[0]
            if len(neighbours) < self.min_samples:
                labels[p] = NOISE
                continue
            # p is a core point; start a new cluster
            cluster_id += 1
            labels[p] = cluster_id

            # Worklist of seeds to expand
            seeds = list(neighbours)
            while seeds:
                q = seeds.pop()
                if labels[q] == NOISE:
                    # Border point: attach to this cluster
                    labels[q] = cluster_id
                    continue
                if labels[q] != UNVISITED:
                    continue
                labels[q] = cluster_id
                q_neighbours = np.where(sq[q] <= eps_sq)[0]
                if len(q_neighbours) >= self.min_samples:
                    # q is also core; expand
                    seeds.extend(q_neighbours.tolist())

        self.labels_ = labels
        return self


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def n_clusters(labels):
    return len({lab for lab in labels if lab >= 0})


def n_noise(labels):
    return int((labels == NOISE).sum())


def demo_from_scratch(X, y_true):
    banner("DEMO 1 --- DBSCAN from scratch on moons + noise")

    eps = 0.20
    min_samples = 5
    print(f"  Data shape : {X.shape[0]} points, {X.shape[1]} features")
    print(f"            (300 moons + 20 uniform-random noise points)")
    print(f"  eps          : {eps}")
    print(f"  min_samples  : {min_samples}")

    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    print(f"  Clusters     : {n_clusters(model.labels_)}")
    print(f"  Noise points : {n_noise(model.labels_)}")
    ari = adjusted_rand_score(y_true, model.labels_)
    print(f"  ARI vs true labels : {ari:.3f}")
    return model


def demo_sklearn(X, y_true, our_model):
    banner("DEMO 2 --- Same data, scikit-learn DBSCAN")

    sk = SkDBSCAN(eps=0.20, min_samples=5)
    sk.fit(X)
    print(f"  Clusters     : {n_clusters(sk.labels_)}")
    print(f"  Noise points : {n_noise(sk.labels_)}")
    ari = adjusted_rand_score(y_true, sk.labels_)
    print(f"  ARI vs true labels : {ari:.3f}")
    ari_pair = adjusted_rand_score(our_model.labels_, sk.labels_)
    print(f"  Agreement (ARI) with from-scratch DBSCAN : "
          f"{ari_pair:.3f}")


def demo_kmeans(X, y_true):
    banner("DEMO 3 --- K-Means on the same data (for comparison)")

    km = KMeans(n_clusters=2, n_init=10, random_state=RNG_SEED).fit(X)
    print(f"  K            : 2 (forced)")
    print(f"  Noise points : 0 (K-Means cannot mark noise)")
    ari = adjusted_rand_score(y_true, km.labels_)
    print(f"  ARI vs true labels : {ari:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    X, y_true = make_dataset()
    model = demo_from_scratch(X, y_true)
    demo_sklearn(X, y_true, model)
    demo_kmeans(X, y_true)
    print()


if __name__ == "__main__":
    main()
