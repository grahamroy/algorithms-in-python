"""
gmm.py --- companion code for "Gaussian Mixture Models"
(Advanced Unsupervised Learning, Part 2).

Four demos:
  1. GMM-EM from scratch on 3 anisotropic Gaussian clusters
     (stretched ellipses K-Means cannot recover).
  2. Comparison with scikit-learn's GaussianMixture.
  3. K-Means on the same data for the baseline contrast.
  4. BIC sweep over K to demonstrate model selection.

Dependencies: numpy, scikit-learn. Runs in a few seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SkGMM


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Dataset: 3 anisotropic Gaussian clusters
# ---------------------------------------------------------------------------

def make_dataset(seed=RNG_SEED):
    # Cluster centres placed close enough together that K-Means'
    # spherical assumption hurts when combined with elongated shapes
    centres = np.array([[-1.5, 0.0], [1.5, 0.0], [0.0, 2.5]])
    X, y = make_blobs(n_samples=600, centers=centres,
                      cluster_std=0.8, random_state=seed)
    # Strongly anisotropic: each cluster is a long, narrow ellipse
    # at a different orientation
    transforms = [
        np.array([[3.0,  0.0], [0.0, 0.2]]),    # horizontal cigar
        np.array([[0.2,  0.0], [0.0, 3.0]]),    # vertical cigar
        np.array([[2.0, -1.5], [1.5, 0.2]]),    # diagonal cigar
    ]
    for k in range(3):
        mask = y == k
        Xk_centered = X[mask] - centres[k]
        X[mask] = Xk_centered @ transforms[k] + centres[k]
    return X, y


# ---------------------------------------------------------------------------
# GMM-EM from scratch
# ---------------------------------------------------------------------------

LOG_2PI = math.log(2 * math.pi)


def _log_gaussian(X, mu, Sigma, eps=1e-6):
    """log N(X | mu, Sigma) for every row of X."""
    d = X.shape[1]
    # Regularise for numerical stability
    Sigma = Sigma + eps * np.eye(d)
    sign, logdet = np.linalg.slogdet(Sigma)
    inv = np.linalg.inv(Sigma)
    diff = X - mu
    quad = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return -0.5 * (d * LOG_2PI + logdet + quad)


class GaussianMixture:
    """GMM with full covariance, fit by EM. Initialised from
    K-Means centroids; covariances start as the empirical
    covariance of K-Means' clusters."""

    def __init__(self, n_components, max_iter=200, tol=1e-4,
                 random_state=RNG_SEED, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.reg_covar = reg_covar

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        K = self.n_components

        # K-Means++ initialisation
        km = KMeans(n_clusters=K, n_init=10,
                    random_state=self.random_state).fit(X)
        labels = km.labels_

        self.weights_ = np.zeros(K)
        self.means_ = np.zeros((K, d))
        self.covariances_ = np.zeros((K, d, d))
        for k in range(K):
            pts = X[labels == k]
            self.weights_[k] = len(pts) / n
            self.means_[k] = pts.mean(axis=0)
            cov = np.cov(pts, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            self.covariances_[k] = cov + self.reg_covar * np.eye(d)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            # E-step: responsibilities (log-space for stability)
            log_resp = np.zeros((n, K))
            for k in range(K):
                log_resp[:, k] = (
                    math.log(self.weights_[k] + 1e-12)
                    + _log_gaussian(X, self.means_[k],
                                    self.covariances_[k],
                                    eps=self.reg_covar)
                )
            # log-sum-exp normalisation per row
            row_max = log_resp.max(axis=1, keepdims=True)
            log_norm = row_max + np.log(
                np.exp(log_resp - row_max).sum(axis=1, keepdims=True)
            )
            log_resp = log_resp - log_norm
            resp = np.exp(log_resp)
            ll = float(log_norm.sum())

            # M-step
            Nk = resp.sum(axis=0) + 1e-12
            self.weights_ = Nk / n
            self.means_ = (resp.T @ X) / Nk[:, None]
            for k in range(K):
                diff = X - self.means_[k]
                self.covariances_[k] = (
                    (resp[:, k : k + 1] * diff).T @ diff
                ) / Nk[k] + self.reg_covar * np.eye(d)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.n_iter_ = it + 1
        self.lower_bound_ = ll
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        K = self.n_components
        log_resp = np.zeros((n, K))
        for k in range(K):
            log_resp[:, k] = (
                math.log(self.weights_[k] + 1e-12)
                + _log_gaussian(X, self.means_[k],
                                self.covariances_[k],
                                eps=self.reg_covar)
            )
        return np.argmax(log_resp, axis=1)

    def bic(self, X):
        """Bayesian Information Criterion: lower is better."""
        n, d = X.shape
        K = self.n_components
        # Parameters: K weights (K-1 free), K means (K*d), K
        # covariances (K * d*(d+1)/2)
        n_params = (K - 1) + K * d + K * d * (d + 1) // 2
        return n_params * math.log(n) - 2 * self.lower_bound_


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_from_scratch(X, y_true):
    banner("DEMO 1 --- GMM from scratch on 3 anisotropic clusters")

    K = 3
    print(f"  Data shape    : {X.shape[0]} points, {X.shape[1]} features")
    print(f"  Components    : {K}")
    print(f"  Covariance    : full")
    print(f"  Init          : K-Means centroids")

    gmm = GaussianMixture(n_components=K).fit(X)
    preds = gmm.predict(X)
    ari = adjusted_rand_score(y_true, preds)
    print(f"  Converged in  : {gmm.n_iter_} EM iterations")
    print(f"  Final log-likelihood : {gmm.lower_bound_:.2f}")
    print(f"  ARI vs true labels   : {ari:.3f}")
    return gmm


def demo_sklearn(X, y_true, our_gmm):
    banner("DEMO 2 --- Same data, scikit-learn GaussianMixture")

    sk = SkGMM(n_components=3, covariance_type="full",
               init_params="kmeans", random_state=RNG_SEED).fit(X)
    preds = sk.predict(X)
    ari = adjusted_rand_score(y_true, preds)
    print(f"  Converged in  : {sk.n_iter_} EM iterations")
    print(f"  Final log-likelihood : "
          f"{sk.score(X) * len(X):.2f}")
    print(f"  ARI vs true labels   : {ari:.3f}")

    our_preds = our_gmm.predict(X)
    ari_pair = adjusted_rand_score(our_preds, preds)
    print(f"  Agreement (ARI) with from-scratch GMM : "
          f"{ari_pair:.3f}")


def demo_kmeans(X, y_true):
    banner("DEMO 3 --- K-Means on the same data (for comparison)")

    km = KMeans(n_clusters=3, n_init=10,
                random_state=RNG_SEED).fit(X)
    ari = adjusted_rand_score(y_true, km.labels_)
    print(f"  K-Means ARI vs true labels  : {ari:.3f}")


def demo_bic(X):
    banner("DEMO 4 --- Model selection: BIC vs K")

    print(f"  {'K':>4}   {'BIC':>6}     {'log-likelihood':>14}")
    print(f"  {'---':>4}   {'-----':>6}    {'--------------':>14}")
    for K in range(1, 7):
        gmm = GaussianMixture(n_components=K).fit(X)
        print(f"  {K:>4}   {gmm.bic(X):>6.1f}     "
              f"{gmm.lower_bound_:>14.1f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    X, y_true = make_dataset()
    gmm = demo_from_scratch(X, y_true)
    demo_sklearn(X, y_true, gmm)
    demo_kmeans(X, y_true)
    demo_bic(X)
    print()


if __name__ == "__main__":
    main()
