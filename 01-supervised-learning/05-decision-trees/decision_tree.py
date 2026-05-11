"""
decision_tree.py --- companion code for "Decision Trees"
(Supervised Learning, Part 5).

Three demos:
  1. CART-style decision tree from scratch on the two-moons dataset.
  2. Comparison with scikit-learn's DecisionTreeClassifier
     (predictions agree).
  3. Depth sweep showing the textbook overfitting curve --- train
     accuracy climbing to 1.0 while test accuracy peaks and falls.

Dependencies: numpy, scikit-learn. Runs in well under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Two-moons dataset
# ---------------------------------------------------------------------------

def make_dataset():
    X, y = make_moons(n_samples=500, noise=0.25, random_state=RNG_SEED)
    return train_test_split(X, y, test_size=0.2,
                            random_state=RNG_SEED, stratify=y)


# ---------------------------------------------------------------------------
# Demo 1 --- decision tree from scratch
# ---------------------------------------------------------------------------

class _Leaf:
    __slots__ = ("label",)
    def __init__(self, label):
        self.label = label
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


class DecisionTreeClassifier:
    """CART-style binary decision tree using Gini impurity."""

    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    # ---------------- public API ----------------

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.root = self._build(np.asarray(X, dtype=float),
                                np.asarray(y), depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._descend(self.root, x) for x in X])

    # ---------------- helpers ----------------

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1.0 - (p * p).sum()

    @staticmethod
    def _majority(y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[counts.argmax()]

    def _best_split(self, X, y):
        n, d = X.shape
        parent_gini = self._gini(y)
        best_score = parent_gini
        best_feat = None
        best_thr = None
        for feat in range(d):
            xs = X[:, feat]
            order = np.argsort(xs)
            xs_sorted = xs[order]
            ys_sorted = y[order]
            # Candidate thresholds = midpoints between consecutive
            # distinct values
            for i in range(1, n):
                if xs_sorted[i] == xs_sorted[i - 1]:
                    continue
                if (i < self.min_samples_leaf
                        or n - i < self.min_samples_leaf):
                    continue
                thr = 0.5 * (xs_sorted[i] + xs_sorted[i - 1])
                left_y = ys_sorted[:i]
                right_y = ys_sorted[i:]
                score = (i / n) * self._gini(left_y) \
                      + ((n - i) / n) * self._gini(right_y)
                if score < best_score:
                    best_score = score
                    best_feat = feat
                    best_thr = thr
        return best_feat, best_thr, best_score

    def _build(self, X, y, depth):
        if (len(np.unique(y)) == 1
                or len(y) < 2 * self.min_samples_leaf
                or (self.max_depth is not None
                    and depth >= self.max_depth)):
            return _Leaf(self._majority(y))

        feat, thr, _ = self._best_split(X, y)
        if feat is None:
            return _Leaf(self._majority(y))

        mask = X[:, feat] <= thr
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _Node(feat, thr, left, right)

    def _descend(self, node, x):
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    # ---------------- introspection ----------------

    def depth(self):
        def _d(node):
            if node.is_leaf():
                return 0
            return 1 + max(_d(node.left), _d(node.right))
        return _d(self.root)

    def leaves(self):
        def _l(node):
            if node.is_leaf():
                return 1
            return _l(node.left) + _l(node.right)
        return _l(self.root)


def demo_from_scratch(X_train, X_test, y_train, y_test):
    banner("DEMO 1 --- Decision tree from scratch on the moons dataset")

    max_depth = 6
    print(f"  Training set : {len(X_train)} examples, "
          f"{X_train.shape[1]} features")
    print(f"  Test set     : {len(X_test)} examples")
    print(f"  max_depth    : {max_depth}")

    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X_train, y_train)

    train_acc = (tree.predict(X_train) == y_train).mean()
    test_acc = (tree.predict(X_test) == y_test).mean()
    print(f"  Train accuracy : {train_acc:.3f}")
    print(f"  Test accuracy  : {test_acc:.3f}")
    print(f"  Tree depth     : {tree.depth()}   "
          f"leaves : {tree.leaves()}")
    return tree


# ---------------------------------------------------------------------------
# Demo 2 --- scikit-learn comparison
# ---------------------------------------------------------------------------

def demo_sklearn(X_train, X_test, y_train, y_test, our_tree):
    banner("DEMO 2 --- Same data, scikit-learn DecisionTreeClassifier")

    sk = SkDecisionTree(criterion="gini", max_depth=6,
                        random_state=RNG_SEED)
    sk.fit(X_train, y_train)
    train_acc = (sk.predict(X_train) == y_train).mean()
    test_acc = (sk.predict(X_test) == y_test).mean()
    print(f"  Train accuracy : {train_acc:.3f}")
    print(f"  Test accuracy  : {test_acc:.3f}")
    print(f"  Tree depth     : {sk.get_depth()}   "
          f"leaves : {sk.get_n_leaves()}")

    agree = (sk.predict(X_test) == our_tree.predict(X_test)).sum()
    print(f"  Agreement with from-scratch tree on test set: "
          f"{agree}/{len(X_test)} predictions identical")


# ---------------------------------------------------------------------------
# Demo 3 --- Depth sweep
# ---------------------------------------------------------------------------

def demo_depth_sweep(X_train, X_test, y_train, y_test):
    banner("DEMO 3 --- The depth-vs-overfitting tradeoff")

    print(f"  {'depth':>5}   {'train_acc':>9}   "
          f"{'test_acc':>8}   {'leaves':>6}")
    print(f"  {'-----':>5}   {'---------':>9}   "
          f"{'--------':>8}   {'------':>6}")

    depths = [1, 2, 3, 4, 6, 10, None]
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d)
        tree.fit(X_train, y_train)
        train_acc = (tree.predict(X_train) == y_train).mean()
        test_acc = (tree.predict(X_test) == y_test).mean()
        d_str = "None" if d is None else str(d)
        print(f"  {d_str:>5}   {train_acc:>9.3f}   "
              f"{test_acc:>8.3f}   {tree.leaves():>6}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    X_train, X_test, y_train, y_test = make_dataset()
    tree = demo_from_scratch(X_train, X_test, y_train, y_test)
    demo_sklearn(X_train, X_test, y_train, y_test, tree)
    demo_depth_sweep(X_train, X_test, y_train, y_test)
    print()


if __name__ == "__main__":
    main()
