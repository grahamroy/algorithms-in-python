"""
pca.py --- companion code for "Principal Component Analysis"
(Unsupervised Learning, Part 3).

Three demos:
  1. From-scratch PCA via SVD of the centred data matrix on the
     scikit-learn digits dataset (1797 samples, 64 features).
  2. Comparison with scikit-learn's PCA (projections match to
     floating-point precision after sign alignment).
  3. Cumulative variance and reconstruction error as a function
     of the number of retained components.

Dependencies: numpy, scikit-learn. Runs in well under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as SkPCA


SEPARATOR = "=" * 72


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# From-scratch PCA via SVD
# ---------------------------------------------------------------------------

class PCA:
    """Principal Component Analysis via SVD of the centred data
    matrix. Returns components as rows of self.components_, in
    decreasing order of explained variance."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        # 1. Centre
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        # 2. SVD (economy form)
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        # 3. Sign convention: flip so each component's largest-
        #    magnitude entry is positive (matches sklearn).
        for i in range(Vt.shape[0]):
            j = np.argmax(np.abs(Vt[i]))
            if Vt[i, j] < 0:
                Vt[i] = -Vt[i]
                U[:, i] = -U[:, i]
        # 4. Keep top-k
        self.components_ = Vt[: self.n_components]
        var_full = (S ** 2) / (n - 1)
        self.explained_variance_ = var_full[: self.n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / var_full.sum()
        )
        # Stash for cumulative reporting
        self._all_var_ = var_full
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_proj):
        return X_proj @ self.components_ + self.mean_


# ---------------------------------------------------------------------------
# Demo 1 --- from scratch on digits
# ---------------------------------------------------------------------------

def demo_from_scratch(X):
    banner("DEMO 1 --- PCA from scratch on scikit-learn's digits dataset")

    n_components = 10
    print(f"  Data shape          : {X.shape[0]} samples, "
          f"{X.shape[1]} features")
    print(f"  Centring            : per-feature mean")
    print(f"  Method              : SVD of centred data")
    print(f"  Components fitted   : {n_components}")
    print(f"  Explained variance ratio (top {n_components}):")

    pca = PCA(n_components=n_components).fit(X)
    cum = 0.0
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        cum += ratio
        print(f"    PC {i + 1:>2}: {ratio:.3f}   |   cumulative {cum:.3f}")
    return pca


# ---------------------------------------------------------------------------
# Demo 2 --- sklearn comparison
# ---------------------------------------------------------------------------

def demo_sklearn(X, our_pca):
    banner("DEMO 2 --- Same data, scikit-learn PCA")

    n_components = our_pca.n_components
    sk = SkPCA(n_components=n_components, svd_solver="full").fit(X)
    print(f"  Components fitted     : {n_components}")
    top5 = " ".join(f"{r:.3f}" for r in sk.explained_variance_ratio_[:5])
    print(f"  Explained variance ratio (top 5): {top5}")

    # Reconstruction error using k components
    X_proj = sk.transform(X)
    X_back = sk.inverse_transform(X_proj)
    mse = float(((X - X_back) ** 2).mean())
    print(f"  Reconstruction error ({n_components} components) MSE : "
          f"{mse:.4f}")

    # Compare projections to from-scratch (after our sign alignment,
    # they should match sklearn's choice of sign too).
    ours_proj = our_pca.transform(X)
    sk_proj = sk.transform(X)
    # Account for residual sign ambiguity per component
    diff = np.abs(np.abs(ours_proj) - np.abs(sk_proj)).max()
    print(f"  Maximum |difference| in projections vs from-scratch : "
          f"{diff:.2e}")


# ---------------------------------------------------------------------------
# Demo 3 --- how many components do we need?
# ---------------------------------------------------------------------------

def demo_components_sweep(X):
    banner("DEMO 3 --- How many components do we need?")

    print(f"  {'k':>3}    {'cumulative variance':>19}   "
          f"{'reconstruction MSE':>18}")
    print(f"  {'---':>3}   {'-------------------':>19}   "
          f"{'------------------':>18}")

    n = X.shape[0]
    full_pca = PCA(n_components=X.shape[1]).fit(X)
    var_full = full_pca._all_var_
    total_var = var_full.sum()

    for k in (2, 5, 10, 20, 30, 40, 64):
        pca = PCA(n_components=k).fit(X)
        cum = pca.explained_variance_.sum() / total_var
        X_proj = pca.transform(X)
        X_back = pca.inverse_transform(X_proj)
        mse = float(((X - X_back) ** 2).mean())
        print(f"  {k:>3}   {cum:>19.3f}   {mse:>18.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    digits = load_digits()
    X = digits.data
    pca = demo_from_scratch(X)
    demo_sklearn(X, pca)
    demo_components_sweep(X)
    print()


if __name__ == "__main__":
    main()
