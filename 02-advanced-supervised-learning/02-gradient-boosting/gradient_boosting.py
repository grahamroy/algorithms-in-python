"""
gradient_boosting.py --- companion code for "Gradient Boosting"
(Advanced Supervised Learning, Part 2).

Three demos:
  1. Gradient-boosted regression trees for binary classification
     with log loss, from scratch on the two-moons dataset.
  2. Comparison with scikit-learn's GradientBoostingClassifier
     (predictions agree on every test example).
  3. An M-sweep producing the textbook boosting curve --- fast
     bias reduction up to a sweet spot, then slow overfitting
     as more trees are added.

Dependencies: numpy, scikit-learn. Runs in a few seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import math

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as SkGBC


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Two-moons dataset (same as Parts 5 and 1)
# ---------------------------------------------------------------------------

def make_dataset(seed=RNG_SEED):
    X, y = make_moons(n_samples=500, noise=0.25, random_state=seed)
    return train_test_split(X, y, test_size=0.2,
                            random_state=seed, stratify=y)


# ---------------------------------------------------------------------------
# Regression tree (squared-error splits, leaf-value override hook)
# ---------------------------------------------------------------------------

class _Leaf:
    __slots__ = ("value", "indices")

    def __init__(self, value, indices):
        self.value = value
        self.indices = indices  # training-row indices in this leaf

    def is_leaf(self):
        return True


class _Node:
    __slots__ = ("feature", "threshold", "left", "right")

    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    def is_leaf(self):
        return False


class RegressionTree:
    """CART-style regression tree using squared-error splits."""

    def __init__(self, max_depth=3, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.root = self._build(X, y, np.arange(len(y)), depth=0)
        return self

    @staticmethod
    def _sse(y):
        if len(y) == 0:
            return 0.0
        return float(((y - y.mean()) ** 2).sum())

    def _best_split(self, X, y):
        n, d = X.shape
        parent_sse = self._sse(y)
        best_score = parent_sse
        best_feat = None
        best_thr = None
        for feat in range(d):
            xs = X[:, feat]
            order = np.argsort(xs)
            xs_sorted = xs[order]
            ys_sorted = y[order]
            cum = np.cumsum(ys_sorted)
            cum_sq = np.cumsum(ys_sorted * ys_sorted)
            total = cum[-1]
            total_sq = cum_sq[-1]
            for i in range(1, n):
                if xs_sorted[i] == xs_sorted[i - 1]:
                    continue
                if (i < self.min_samples_leaf
                        or n - i < self.min_samples_leaf):
                    continue
                # SSE of left and right via cumulative sums
                lsum = cum[i - 1]
                lsq = cum_sq[i - 1]
                rsum = total - lsum
                rsq = total_sq - lsq
                lsse = lsq - (lsum * lsum) / i
                rsse = rsq - (rsum * rsum) / (n - i)
                score = lsse + rsse
                if score < best_score - 1e-12:
                    best_score = score
                    best_feat = feat
                    best_thr = 0.5 * (xs_sorted[i] + xs_sorted[i - 1])
        return best_feat, best_thr

    def _build(self, X, y, indices, depth):
        if (len(np.unique(y)) == 1
                or len(y) < 2 * self.min_samples_leaf
                or (self.max_depth is not None
                    and depth >= self.max_depth)):
            return _Leaf(float(y.mean()) if len(y) else 0.0, indices)

        feat, thr = self._best_split(X, y)
        if feat is None:
            return _Leaf(float(y.mean()), indices)

        mask = X[:, feat] <= thr
        left = self._build(X[mask], y[mask],
                           indices[mask], depth + 1)
        right = self._build(X[~mask], y[~mask],
                            indices[~mask], depth + 1)
        return _Node(feat, thr, left, right)

    def leaves(self):
        out = []
        def _walk(node):
            if node.is_leaf():
                out.append(node)
            else:
                _walk(node.left)
                _walk(node.right)
        _walk(self.root)
        return out

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._descend(self.root, x) for x in X])

    def _descend(self, node, x):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold \
                              else node.right
        return node.value


# ---------------------------------------------------------------------------
# Gradient-boosting classifier for binary log loss
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class GradientBoostingClassifier:
    """Binary gradient boosting with log loss."""

    def __init__(self, n_estimators=200, learning_rate=0.1,
                 max_depth=3, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Initial log-odds
        p_mean = float(y.mean())
        p_mean = min(max(p_mean, 1e-6), 1 - 1e-6)
        self.F0_ = math.log(p_mean / (1 - p_mean))

        F = np.full_like(y, self.F0_, dtype=float)
        self.trees_ = []

        for _ in range(self.n_estimators):
            p = _sigmoid(F)
            residuals = y - p

            tree = RegressionTree(max_depth=self.max_depth,
                                  min_samples_leaf=self.min_samples_leaf)
            tree.fit(X, residuals)

            # Newton step: optimal leaf value for log loss
            for leaf in tree.leaves():
                idx = leaf.indices
                num = residuals[idx].sum()
                den = (p[idx] * (1 - p[idx])).sum()
                leaf.value = num / (den + 1e-12)

            update = tree.predict(X)
            F = F + self.learning_rate * update
            self.trees_.append(tree)

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.F0_, dtype=float)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0.0).astype(int)


# ---------------------------------------------------------------------------
# Demo 1 --- from scratch
# ---------------------------------------------------------------------------

def demo_from_scratch(X_train, X_test, y_train, y_test):
    banner("DEMO 1 --- Gradient boosting from scratch on the moons dataset")

    M = 200
    eta = 0.1
    md = 3
    print(f"  Training set : {len(X_train)} examples, "
          f"{X_train.shape[1]} features")
    print(f"  Test set     : {len(X_test)} examples")
    print(f"  M = {M}   eta = {eta}   max_depth = {md}   loss = log")

    gb = GradientBoostingClassifier(n_estimators=M,
                                    learning_rate=eta,
                                    max_depth=md)
    gb.fit(X_train, y_train)
    train_acc = (gb.predict(X_train) == y_train).mean()
    test_acc = (gb.predict(X_test) == y_test).mean()
    print(f"  Train accuracy : {train_acc:.3f}")
    print(f"  Test accuracy  : {test_acc:.3f}")
    return gb


# ---------------------------------------------------------------------------
# Demo 2 --- sklearn comparison
# ---------------------------------------------------------------------------

def demo_sklearn(X_train, X_test, y_train, y_test, our_gb):
    banner("DEMO 2 --- Same data, scikit-learn GradientBoostingClassifier")

    sk = SkGBC(n_estimators=200, learning_rate=0.1,
               max_depth=3, random_state=RNG_SEED)
    sk.fit(X_train, y_train)
    train_acc = (sk.predict(X_train) == y_train).mean()
    test_acc = (sk.predict(X_test) == y_test).mean()
    print(f"  M = 200   eta = 0.1   max_depth = 3")
    print(f"  Train accuracy : {train_acc:.3f}")
    print(f"  Test accuracy  : {test_acc:.3f}")

    agree = (sk.predict(X_test) == our_gb.predict(X_test)).sum()
    print(f"  Agreement with from-scratch model on test set: "
          f"{agree}/{len(X_test)} predictions identical")


# ---------------------------------------------------------------------------
# Demo 3 --- M-sweep
# ---------------------------------------------------------------------------

def demo_m_sweep(X_train, X_test, y_train, y_test):
    banner("DEMO 3 --- Number of trees vs accuracy (the boosting curve)")

    print(f"  {'M':>5}    {'train_acc':>9}   {'test_acc':>8}")
    print(f"  {'----':>5}    {'---------':>9}   {'--------':>8}")

    M_values = [1, 5, 20, 50, 100, 200, 500, 1000]
    # Fit once at the largest M, then snapshot at each value
    gb = GradientBoostingClassifier(n_estimators=max(M_values),
                                    learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)

    # Score by truncating the ensemble
    for M in M_values:
        F_train = np.full(len(X_train), gb.F0_, dtype=float)
        F_test = np.full(len(X_test), gb.F0_, dtype=float)
        for t in range(M):
            F_train += gb.learning_rate * gb.trees_[t].predict(X_train)
            F_test += gb.learning_rate * gb.trees_[t].predict(X_test)
        train_acc = ((F_train >= 0).astype(int) == y_train).mean()
        test_acc = ((F_test >= 0).astype(int) == y_test).mean()
        print(f"  {M:>5}    {train_acc:>9.3f}   {test_acc:>8.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    X_train, X_test, y_train, y_test = make_dataset()
    gb = demo_from_scratch(X_train, X_test, y_train, y_test)
    demo_sklearn(X_train, X_test, y_train, y_test, gb)
    demo_m_sweep(X_train, X_test, y_train, y_test)
    print()


if __name__ == "__main__":
    main()
