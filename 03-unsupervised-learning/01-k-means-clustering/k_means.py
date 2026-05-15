"""
k_means.py --- companion code for "K-Means Clustering"
(Unsupervised Learning, Part 1).

Three demos:
  1. Lloyd's algorithm with k-means++ initialisation, from scratch,
     on a 3-cluster Gaussian-blob dataset.
  2. Comparison with scikit-learn's KMeans (cluster labels agree
     up to relabelling).
  3. Elbow method: WCSS as a function of K.

Dependencies: numpy, scikit-learn. Runs in well under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans as SkKMeans
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
# Dataset --- three well-separated Gaussian blobs
# ---------------------------------------------------------------------------

def make_dataset(seed=RNG_SEED):
    centres = np.array([
        [-3.0, -3.0],
        [ 3.0, -3.0],
        [ 0.0,  3.5],
    ])
    X, y_true = make_blobs(n_samples=600, centers=centres,
                           cluster_std=0.9, random_state=seed)
    return X, y_true


# ---------------------------------------------------------------------------
# K-Means from scratch with k-means++ initialisation
# ---------------------------------------------------------------------------

class KMeans:
    """Lloyd's algorithm with k-means++ initialisation."""

    def __init__(self, n_clusters=3, max_iter=300,
                 tol=1e-6, random_state=RNG_SEED):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _kmeans_plus_plus_init(self, X, rng):
        n = X.shape[0]
        first = rng.integers(0, n)
        centres = [X[first]]
        # closest squared distance from each point to any chosen centre
        d2 = np.sum((X - centres[0]) ** 2, axis=1)
        for _ in range(1, self.n_clusters):
            probs = d2 / d2.sum()
            idx = rng.choice(n, p=probs)
            centres.append(X[idx])
            new_d2 = np.sum((X - X[idx]) ** 2, axis=1)
            d2 = np.minimum(d2, new_d2)
        return np.array(centres)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.cluster_centers_ = self._kmeans_plus_plus_init(X, rng)

        prev_labels = None
        for iteration in range(self.max_iter):
            # Assignment step
            dists = np.sum(
                (X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2,
                axis=2,
            )
            labels = np.argmin(dists, axis=1)
            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            # Update step
            new_centres = np.array([
                X[labels == k].mean(axis=0)
                if (labels == k).any()
                else self.cluster_centers_[k]
                for k in range(self.n_clusters)
            ])
            shift = np.linalg.norm(new_centres - self.cluster_centers_)
            self.cluster_centers_ = new_centres
            prev_labels = labels
            if shift < self.tol:
                break

        self.labels_ = labels
        self.n_iter_ = iteration + 1
        self.inertia_ = float(np.sum(
            (X - self.cluster_centers_[labels]) ** 2
        ))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = np.sum(
            (X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(dists, axis=1)


# ---------------------------------------------------------------------------
# Demo 1 --- from scratch
# ---------------------------------------------------------------------------

def demo_from_scratch(X, y_true):
    banner("DEMO 1 --- K-Means from scratch on Gaussian blobs")

    K = 3
    print(f"  Data shape : {X.shape[0]} points, {X.shape[1]} features")
    print(f"  True K     : 3")
    print(f"  Chosen K   : {K}")
    print(f"  Init       : k-means++")

    km = KMeans(n_clusters=K, random_state=RNG_SEED)
    km.fit(X)
    ari = adjusted_rand_score(y_true, km.labels_)
    print(f"  Converged in {km.n_iter_} iterations")
    print(f"  Final inertia (WCSS) : {km.inertia_:.1f}")
    print(f"  Adjusted Rand Index vs true labels : {ari:.3f}")
    return km


# ---------------------------------------------------------------------------
# Demo 2 --- sklearn comparison
# ---------------------------------------------------------------------------

def demo_sklearn(X, y_true, our_km):
    banner("DEMO 2 --- Same data, scikit-learn KMeans")

    sk = SkKMeans(n_clusters=3, init="k-means++", n_init=10,
                  random_state=RNG_SEED)
    sk.fit(X)
    ari = adjusted_rand_score(y_true, sk.labels_)
    print(f"  Converged in {sk.n_iter_} iterations")
    print(f"  Final inertia (WCSS) : {sk.inertia_:.1f}")
    print(f"  Adjusted Rand Index vs true labels : {ari:.3f}")

    # Relabel-aware agreement: ARI between the two clusterings
    ari_pair = adjusted_rand_score(sk.labels_, our_km.labels_)
    print(f"  Agreement (ARI) between from-scratch and sklearn: "
          f"{ari_pair:.3f}")


# ---------------------------------------------------------------------------
# Demo 3 --- elbow method
# ---------------------------------------------------------------------------

def demo_elbow(X):
    banner("DEMO 3 --- Elbow method: WCSS vs K")

    print(f"  {'K':>4}   {'WCSS':>8}")
    print(f"  {'---':>4}   {'--------':>8}")
    for K in range(2, 9):
        km = KMeans(n_clusters=K, random_state=RNG_SEED)
        km.fit(X)
        print(f"  {K:>4}   {km.inertia_:>8.1f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    X, y_true = make_dataset()
    km = demo_from_scratch(X, y_true)
    demo_sklearn(X, y_true, km)
    demo_elbow(X)
    print()


if __name__ == "__main__":
    main()
