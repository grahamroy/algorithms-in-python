"""
svm.py --- companion code for "Support Vector Machines"
(Advanced Supervised Learning, Part 3).

Three demos:
  1. Linear SVM from scratch via subgradient descent on hinge
     loss with L2 regularisation, on the two-moons dataset.
  2. Comparison with scikit-learn's LinearSVC (predictions agree).
  3. Kernelised SVM with the RBF kernel and a C sweep showing the
     bias-variance trade-off and how the kernel trick restores
     accuracy on a non-linearly-separable problem.

Dependencies: numpy, scikit-learn. Runs in well under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Dataset (same moons used in Parts 5, 1, 2)
# ---------------------------------------------------------------------------

def make_dataset(seed=RNG_SEED):
    X, y = make_moons(n_samples=500, noise=0.25, random_state=seed)
    return train_test_split(X, y, test_size=0.2,
                            random_state=seed, stratify=y)


# ---------------------------------------------------------------------------
# Linear SVM from scratch (hinge loss + L2, subgradient descent)
# ---------------------------------------------------------------------------

class LinearSVM:
    """Soft-margin linear SVM trained by subgradient descent on
    L(w, b) = 1/2 ||w||^2 + C * sum max(0, 1 - y_i (w.x_i + b)).
    Labels y must be 0/1 (converted internally to ±1)."""

    def __init__(self, C=1.0, n_epochs=2000, lr=0.001):
        self.C = C
        self.n_epochs = n_epochs
        self.lr = lr

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        y_signed = np.where(y == 1, 1.0, -1.0)

        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0

        for _ in range(self.n_epochs):
            margins = y_signed * (X @ self.w_ + self.b_)
            violating = margins < 1.0
            grad_w = self.w_ - self.C * (y_signed[violating, None] *
                                         X[violating]).sum(axis=0)
            grad_b = -self.C * y_signed[violating].sum()
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b

        # Identify support vectors (margins <= 1)
        margins = y_signed * (X @ self.w_ + self.b_)
        self.support_indices_ = np.where(margins <= 1.0 + 1e-6)[0]
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w_ + self.b_

    def predict(self, X):
        return (self.decision_function(X) >= 0.0).astype(int)


# ---------------------------------------------------------------------------
# Demo 1 --- from scratch linear SVM
# ---------------------------------------------------------------------------

def demo_from_scratch(X_train, X_test, y_train, y_test):
    banner("DEMO 1 --- Linear SVM from scratch on the moons dataset")

    print(f"  Training set : {len(X_train)} examples, "
          f"{X_train.shape[1]} features")
    print(f"  Test set     : {len(X_test)} examples")
    print(f"  Loss         : hinge + L2,  optimiser: subgradient descent")

    svm = LinearSVM(C=1.0, n_epochs=2000, lr=0.001)
    svm.fit(X_train, y_train)
    test_acc = (svm.predict(X_test) == y_test).mean()
    print(f"  Test accuracy : {test_acc:.3f}")
    print(f"  Support vectors (training points inside or on margin) : "
          f"{len(svm.support_indices_)}")
    return svm


# ---------------------------------------------------------------------------
# Demo 2 --- sklearn LinearSVC
# ---------------------------------------------------------------------------

def demo_sklearn_linear(X_train, X_test, y_train, y_test, our_svm):
    banner("DEMO 2 --- Same data, scikit-learn LinearSVC")

    sk = LinearSVC(C=1.0, loss="hinge", max_iter=5000,
                   random_state=RNG_SEED)
    sk.fit(X_train, y_train)
    test_acc = (sk.predict(X_test) == y_test).mean()
    print(f"  Test accuracy : {test_acc:.3f}")

    agree = (sk.predict(X_test) == our_svm.predict(X_test)).sum()
    print(f"  Agreement with from-scratch model on test set: "
          f"{agree}/{len(X_test)} predictions identical")


# ---------------------------------------------------------------------------
# Demo 3 --- Non-linear SVM with RBF kernel + C sweep
# ---------------------------------------------------------------------------

def demo_rbf(X_train, X_test, y_train, y_test):
    banner("DEMO 3 --- Non-linear SVM with RBF kernel (sklearn SVC)")

    print(f"  {'C':>6} {'gamma':>6}   {'train_acc':>9}   "
          f"{'test_acc':>8}   {'#support_vectors':>16}")
    print(f"  {'------':>6} {'-----':>6}   {'---------':>9}   "
          f"{'--------':>8}   {'----------------':>16}")

    for C in [0.1, 1.0, 10, 100, 1000]:
        svm = SVC(C=C, kernel="rbf", gamma="auto",
                  random_state=RNG_SEED)
        svm.fit(X_train, y_train)
        train_acc = (svm.predict(X_train) == y_train).mean()
        test_acc = (svm.predict(X_test) == y_test).mean()
        n_sv = int(svm.support_.size)
        c_str = f"{C}"
        print(f"  {c_str:>6} {'auto':>6}   {train_acc:>9.3f}   "
              f"{test_acc:>8.3f}   {n_sv:>16}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    X_train, X_test, y_train, y_test = make_dataset()
    our_svm = demo_from_scratch(X_train, X_test, y_train, y_test)
    demo_sklearn_linear(X_train, X_test, y_train, y_test, our_svm)
    demo_rbf(X_train, X_test, y_train, y_test)
    print()


if __name__ == "__main__":
    main()
