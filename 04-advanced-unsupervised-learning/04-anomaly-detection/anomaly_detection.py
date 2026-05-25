"""
anomaly_detection.py --- companion code for "Anomaly Detection"
(Advanced Unsupervised Learning, Part 4).

Compare six anomaly detectors on a synthetic dataset:
  - Mahalanobis distance from the global mean
  - GMM log-likelihood
  - k-NN distance (k=5)
  - Local Outlier Factor
  - Isolation Forest
  - One-Class SVM (RBF kernel)

For each detector, score every point, compare top-25 flagged
points against the 25 planted outliers, report precision@25
and ROC AUC.

Dependencies: numpy, scikit-learn. Runs in a few seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Dataset: mixture of 2 Gaussians + uniform outliers
# ---------------------------------------------------------------------------

def make_dataset(seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    inliers, _ = make_blobs(n_samples=500,
                            centers=[[-2, -2], [3, 2]],
                            cluster_std=0.7, random_state=seed)
    n_out = 25
    outliers = rng.uniform(low=-6, high=6, size=(n_out, 2))
    X = np.vstack([inliers, outliers])
    y_true = np.zeros(len(X), dtype=int)
    y_true[-n_out:] = 1
    return X, y_true


# ---------------------------------------------------------------------------
# Detectors --- each returns a high-is-anomalous score per point
# ---------------------------------------------------------------------------

def mahalanobis_scores(X):
    cov = EmpiricalCovariance().fit(X)
    return cov.mahalanobis(X)


def gmm_scores(X):
    gmm = GaussianMixture(n_components=2, covariance_type="full",
                          random_state=RNG_SEED).fit(X)
    # Lower log-likelihood = more anomalous; negate
    return -gmm.score_samples(X)


def knn_scores(X, k=5):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, _ = nn.kneighbors(X)
    return d[:, -1]


def lof_scores(X, k=20):
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
    lof.fit_predict(X)
    return -lof.negative_outlier_factor_


def iforest_scores(X):
    iso = IsolationForest(random_state=RNG_SEED,
                          contamination="auto").fit(X)
    return -iso.score_samples(X)


def ocsvm_scores(X):
    sk = OneClassSVM(kernel="rbf", gamma="auto").fit(X)
    return -sk.score_samples(X)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name, scores, y_true, top_k=25):
    order = np.argsort(-scores)
    top = set(order[:top_k].tolist())
    truth = set(np.where(y_true == 1)[0].tolist())
    precision = len(top & truth) / top_k
    auc = roc_auc_score(y_true, scores)
    return name, precision, auc


def main() -> None:
    banner("DEMO --- Five anomaly detectors on synthetic data")

    X, y_true = make_dataset()
    n_out = int(y_true.sum())

    print(f"  Dataset       : {(y_true == 0).sum()} inliers "
          f"(mixture of 2 Gaussians) + {n_out} outliers")
    print(f"  Top-{n_out} flagged points compared against true outliers")
    print()

    detectors = [
        ("Mahalanobis distance",       mahalanobis_scores),
        ("GMM log-likelihood",         gmm_scores),
        ("k-NN distance (k=5)",        knn_scores),
        ("Local Outlier Factor",       lof_scores),
        ("Isolation Forest",           iforest_scores),
        ("One-Class SVM (RBF)",        ocsvm_scores),
    ]

    print(f"  {'Detector':<26}    {'Precision@25':>12}    "
          f"{'AUC':>5}")
    print(f"  {'-' * 22:<26}    {'-' * 12:>12}    "
          f"{'-' * 5:>5}")
    for name, fn in detectors:
        scores = fn(X)
        _, prec, auc = evaluate(name, scores, y_true, top_k=n_out)
        print(f"  {name:<26}    {prec:>12.2f}    {auc:>5.3f}")
    print()


if __name__ == "__main__":
    main()
