"""
spectral_clustering.py --- companion code for "Spectral Clustering"
(Unsupervised Learning, Part 7).

Three demos:
  1. From-scratch spectral clustering (RBF affinity, symmetric
     normalised Laplacian, K-Means on row-normalised eigenvectors)
     on three datasets that span K-Means failure modes: 3 blobs,
     2 moons, 2 circles.
  2. Comparison with scikit-learn's SpectralClustering.
  3. Eigenvalue inspection on the moons: the eigengap that
     predicts when spectral clustering will work.

Dependencies: numpy, scipy, scikit-learn. Runs in a few seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_blobs, make_moons, make_circles
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

def dataset_blobs():
    centres = np.array([[-3, -3], [3, -3], [0, 3.5]])
    X, y = make_blobs(n_samples=300, centers=centres,
                      cluster_std=0.9, random_state=RNG_SEED)
    return X, y, "3 blobs", 3


def dataset_moons():
    X, y = make_moons(n_samples=300, noise=0.05, random_state=RNG_SEED)
    return X, y, "2 moons", 2


def dataset_circles():
    X, y = make_circles(n_samples=300, factor=0.5, noise=0.05,
                        random_state=RNG_SEED)
    return X, y, "2 circles", 2


# ---------------------------------------------------------------------------
# From-scratch spectral clustering
# ---------------------------------------------------------------------------

def knn_affinity(X, k=10):
    """Symmetric k-nearest-neighbour affinity. Edge weight is
    a Gaussian similarity using a per-point local scale
    (Zelnik-Manor & Perona 2004) to handle varying density."""
    n = X.shape[0]
    D2 = squareform(pdist(X, "sqeuclidean"))
    D = np.sqrt(D2)

    # Per-point local scale: distance to k-th neighbour
    sorted_d = np.sort(D, axis=1)
    sigma = sorted_d[:, k]  # k-th nearest (after self at index 0)
    sigma = np.maximum(sigma, 1e-12)

    # Locally-scaled affinity
    A = np.exp(-D2 / (sigma[:, None] * sigma[None, :]))
    np.fill_diagonal(A, 0.0)

    # Keep only k-NN edges, symmetrise
    idx_sort = np.argsort(D, axis=1)
    knn_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        knn_mask[i, idx_sort[i, 1 : k + 1]] = True
    knn_mask = knn_mask | knn_mask.T  # symmetric: edge if either picks it

    W = np.where(knn_mask, A, 0.0)
    return W


def spectral_cluster(X, n_clusters, k=10,
                     random_state=RNG_SEED):
    """Ng-Jordan-Weiss spectral clustering on a k-NN graph
    with the symmetric normalised Laplacian."""
    n = X.shape[0]
    W = knn_affinity(X, k=k)

    d = W.sum(axis=1)
    D_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
    # L_sym = I - D^{-1/2} W D^{-1/2}
    L_sym = np.eye(n) - (D_inv_sqrt[:, None] * W * D_inv_sqrt[None, :])

    eigvals, eigvecs = np.linalg.eigh(L_sym)
    # Smallest n_clusters eigenvectors (ascending order)
    U = eigvecs[:, :n_clusters]
    # Row-normalise
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    U_norm = U / np.where(row_norms > 1e-12, row_norms, 1.0)

    km = KMeans(n_clusters=n_clusters, n_init=10,
                random_state=random_state)
    labels = km.fit_predict(U_norm)
    return labels, eigvals


# ---------------------------------------------------------------------------
# Demo 1 --- from scratch on three datasets
# ---------------------------------------------------------------------------

def demo_from_scratch():
    banner("DEMO 1 --- Spectral clustering from scratch on 3 datasets")

    print(f"  Affinity      : k-NN graph with local Gaussian scale (k=10)")
    print(f"  Laplacian     : symmetric normalised")
    print(f"  Clustering    : K-Means in eigenvector space")
    print()
    print(f"  {'Dataset':<15}  {'K':>1}  "
          f"{'Spectral ARI':>13}    {'K-Means ARI':>12}")
    print(f"  {'-' * 15:<15}  {'-':>1}  "
          f"{'-' * 12:>13}    {'-' * 11:>12}")

    results = {}
    for ds in (dataset_blobs(), dataset_moons(), dataset_circles()):
        X, y, name, K = ds
        labels_spec, _ = spectral_cluster(X, n_clusters=K)
        ari_spec = adjusted_rand_score(y, labels_spec)

        km = KMeans(n_clusters=K, n_init=10,
                    random_state=RNG_SEED).fit(X)
        ari_km = adjusted_rand_score(y, km.labels_)

        print(f"  {name:<15}  {K:>1}  "
              f"{ari_spec:>13.3f}    {ari_km:>12.3f}")
        results[name] = labels_spec
    return results


# ---------------------------------------------------------------------------
# Demo 2 --- sklearn comparison
# ---------------------------------------------------------------------------

def demo_sklearn(scratch_results):
    banner("DEMO 2 --- Same data, scikit-learn SpectralClustering")

    print(f"  {'Dataset':<15}  {'K':>1}  "
          f"{'Sklearn ARI':>11}    "
          f"{'Agreement vs from-scratch':>26}")
    print(f"  {'-' * 15:<15}  {'-':>1}  "
          f"{'-' * 11:>11}    {'-' * 25:>26}")

    for ds in (dataset_blobs(), dataset_moons(), dataset_circles()):
        X, y, name, K = ds
        sk = SpectralClustering(n_clusters=K,
                                affinity="nearest_neighbors",
                                n_neighbors=10,
                                random_state=RNG_SEED,
                                assign_labels="kmeans")
        sk_labels = sk.fit_predict(X)
        ari_sk = adjusted_rand_score(y, sk_labels)
        ari_pair = adjusted_rand_score(scratch_results[name], sk_labels)
        print(f"  {name:<15}  {K:>1}  "
              f"{ari_sk:>11.3f}    {ari_pair:>26.3f}")


# ---------------------------------------------------------------------------
# Demo 3 --- eigenvalue inspection
# ---------------------------------------------------------------------------

def demo_eigenvalues():
    banner("DEMO 3 --- Why does it work? Eigenvector inspection on moons")

    X, y, _, _ = dataset_moons()
    _, eigvals = spectral_cluster(X, n_clusters=2)
    print(f"  Bottom 4 eigenvalues of L_sym (moons, n={len(X)}):")
    for i in range(4):
        suffix = ""
        if i == 0:
            suffix = "  (always; constant vector)"
        elif i == 1:
            suffix = "  (the cluster-separating eigenvector)"
        elif i == 2:
            suffix = "  (start of \"within-cluster\" structure)"
        print(f"    lambda_{i} = {eigvals[i]:.4f}{suffix}")
    gap = eigvals[2] - eigvals[1]
    print()
    print(f"  Gap between lambda_1 and lambda_2 : {gap:.4f}")
    print(f"  (Large gap = clean K=2 structure)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    results = demo_from_scratch()
    demo_sklearn(results)
    demo_eigenvalues()
    print()


if __name__ == "__main__":
    main()
