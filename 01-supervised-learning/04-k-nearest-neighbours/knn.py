"""
knn.py --- companion code for "K-Nearest Neighbours" (Supervised Learning, Part 4).

Three demos:
  1. KNN from scratch on a 2D three-class synthetic dataset.
  2. Comparison with scikit-learn's KNeighborsClassifier (predictions agree).
  3. Bias-variance sweep over k, showing the textbook U-shape:
     small k overfits, large k underfits, the middle wins.

Dependencies: numpy, scikit-learn. Runs in well under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import Counter

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Synthetic 2D 3-class dataset
# ---------------------------------------------------------------------------

def make_dataset():
    centres = np.array([
        [-1.7, -1.7],
        [ 1.7, -1.7],
        [ 0.0,  2.2],
    ])
    X, y = make_blobs(
        n_samples=500, centers=centres, cluster_std=1.35,
        random_state=RNG_SEED,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED, stratify=y,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Demo 1 --- KNN from scratch
# ---------------------------------------------------------------------------

class KNearestNeighbours:
    """Brute-force KNN classifier."""

    def __init__(self, k: int = 5, weights: str = "uniform"):
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k = k
        self.weights = weights

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def _distances(self, x_query):
        return np.linalg.norm(self.X - x_query, axis=1)

    def predict_one(self, x_query):
        d = self._distances(x_query)
        idx = np.argpartition(d, self.k)[:self.k]
        labels = self.y[idx]
        if self.weights == "uniform":
            return Counter(labels).most_common(1)[0][0]
        # distance-weighted vote
        w = 1.0 / (d[idx] + 1e-12)
        scores = {}
        for lab, wi in zip(labels, w):
            scores[lab] = scores.get(lab, 0.0) + wi
        return max(scores, key=scores.get)

    def predict(self, X_query):
        X_query = np.asarray(X_query, dtype=float)
        return np.array([self.predict_one(x) for x in X_query])


def demo_from_scratch(X_train, X_test, y_train, y_test):
    banner("DEMO 1 --- KNN from scratch on 3-class synthetic data")

    k = 5
    print(f"  Training set: {len(X_train)} examples, {X_train.shape[1]} features")
    print(f"  Test set    : {len(X_test)} examples")
    print(f"  k = {k}, weights = uniform")

    model = KNearestNeighbours(k=k, weights="uniform")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    correct = int((preds == y_test).sum())
    total = len(y_test)
    print(f"  Accuracy    : {correct / total:.3f}  ({correct}/{total})")
    return preds


# ---------------------------------------------------------------------------
# Demo 2 --- scikit-learn comparison
# ---------------------------------------------------------------------------

def demo_sklearn(X_train, X_test, y_train, y_test, our_preds):
    banner("DEMO 2 --- Same data, scikit-learn KNeighborsClassifier")

    model = KNeighborsClassifier(n_neighbors=5, weights="uniform")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    correct = int((preds == y_test).sum())
    total = len(y_test)
    print(f"  Accuracy    : {correct / total:.3f}  ({correct}/{total})")

    agree = int((preds == our_preds).sum())
    print(f"  Agreement with from-scratch implementation: "
          f"{agree}/{total} predictions identical")


# ---------------------------------------------------------------------------
# Demo 3 --- Bias-variance sweep over k
# ---------------------------------------------------------------------------

def demo_k_sweep(X_train, X_test, y_train, y_test):
    banner("DEMO 3 --- Bias-variance sweep over k")

    print("Smaller k = lower bias, higher variance (jagged boundary).")
    print("Larger  k = higher bias, lower variance (smooth-but-wrong boundary).")
    print()
    print(f"  {'k':>4}  {'accuracy':>10}  notes")

    n_train = len(X_train)
    ks = [1, 3, 5, 7, 15, 31, 75, 199, 399]
    for k in ks:
        if k > n_train:
            continue
        model = KNearestNeighbours(k=k, weights="uniform")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = (preds == y_test).mean()
        if k == 1:
            note = "(high variance, jagged boundary)"
        elif k >= n_train - 1:
            note = "(majority-class baseline)"
        elif k >= 100:
            note = "(high bias, oversmoothed boundary)"
        else:
            note = ""
        print(f"  {k:>4}  {acc:>10.3f}  {note}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    X_train, X_test, y_train, y_test = make_dataset()
    our_preds = demo_from_scratch(X_train, X_test, y_train, y_test)
    demo_sklearn(X_train, X_test, y_train, y_test, our_preds)
    demo_k_sweep(X_train, X_test, y_train, y_test)
    print()


if __name__ == "__main__":
    main()
