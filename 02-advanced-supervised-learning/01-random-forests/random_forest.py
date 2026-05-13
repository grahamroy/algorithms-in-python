"""
random_forest.py --- companion code for "Random Forests"
(Advanced Supervised Learning, Part 1).

Three demos:
  1. Random forest from scratch on the two-moons dataset, with
     OOB accuracy as a free held-out estimate.
  2. Comparison with scikit-learn's RandomForestClassifier.
  3. Head-to-head against a single decision tree, including the
     variance reduction across re-seeds.

Dependencies: numpy, scikit-learn. Runs in a few seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import math

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkTree
from sklearn.ensemble import RandomForestClassifier as SkForest


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Two-moons dataset (same as Part 5)
# ---------------------------------------------------------------------------

def make_dataset(seed=RNG_SEED):
    X, y = make_moons(n_samples=500, noise=0.25, random_state=seed)
    return train_test_split(X, y, test_size=0.2,
                            random_state=seed, stratify=y)


# ---------------------------------------------------------------------------
# Decision tree with feature subsampling at every split
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


class _Tree:
    """Decision tree with optional per-split feature subsampling."""

    def __init__(self, max_depth=None, min_samples_leaf=1,
                 max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # int or None
        self.rng = rng if rng is not None else np.random.default_rng()

    def fit(self, X, y):
        self.root = self._build(np.asarray(X, dtype=float),
                                np.asarray(y), depth=0)
        return self

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

    def _candidate_features(self, d):
        if self.max_features is None or self.max_features >= d:
            return np.arange(d)
        return self.rng.choice(d, size=self.max_features, replace=False)

    def _best_split(self, X, y):
        n, d = X.shape
        parent_gini = self._gini(y)
        best_score = parent_gini
        best_feat = None
        best_thr = None
        for feat in self._candidate_features(d):
            xs = X[:, feat]
            order = np.argsort(xs)
            xs_sorted = xs[order]
            ys_sorted = y[order]
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
        return best_feat, best_thr

    def _build(self, X, y, depth):
        if (len(np.unique(y)) == 1
                or len(y) < 2 * self.min_samples_leaf
                or (self.max_depth is not None
                    and depth >= self.max_depth)):
            return _Leaf(self._majority(y))

        feat, thr = self._best_split(X, y)
        if feat is None:
            return _Leaf(self._majority(y))

        mask = X[:, feat] <= thr
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _Node(feat, thr, left, right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._descend(self.root, x) for x in X])

    def _descend(self, node, x):
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold \
                              else node.right
        return node.label


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------

class RandomForestClassifier:
    """Random forest of CART-style trees with bagging and
    per-split feature subsampling."""

    def __init__(self, n_trees=200, max_depth=6,
                 max_features="sqrt", min_samples_leaf=1,
                 random_state=RNG_SEED):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    @staticmethod
    def _resolve_max_features(max_features, d):
        if max_features is None:
            return d
        if max_features == "sqrt":
            return max(1, int(round(math.sqrt(d))))
        if max_features == "log2":
            return max(1, int(round(math.log2(d))))
        return int(max_features)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, d = X.shape
        self.classes_ = np.unique(y)
        k = self._resolve_max_features(self.max_features, d)

        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        self.in_bag_ = []  # per-tree array of training-row indices
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, size=n)  # bootstrap with replacement
            tree = _Tree(max_depth=self.max_depth,
                         min_samples_leaf=self.min_samples_leaf,
                         max_features=k, rng=rng)
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            self.in_bag_.append(idx)
        # cache training data for OOB
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.stack([t.predict(X) for t in self.trees_])
        # Majority vote per column
        out = np.empty(X.shape[0], dtype=self._y_train.dtype)
        for i in range(X.shape[0]):
            vals, counts = np.unique(votes[:, i], return_counts=True)
            out[i] = vals[counts.argmax()]
        return out

    def oob_score(self):
        n = len(self._X_train)
        in_bag_sets = [set(idx.tolist()) for idx in self.in_bag_]
        # For each row, find trees that did not see it
        votes = [[] for _ in range(n)]
        for tree, in_bag in zip(self.trees_, in_bag_sets):
            oob_idx = np.array([i for i in range(n) if i not in in_bag])
            if len(oob_idx) == 0:
                continue
            preds = tree.predict(self._X_train[oob_idx])
            for i, p in zip(oob_idx, preds):
                votes[i].append(p)
        # Aggregate
        correct = 0
        counted = 0
        for i in range(n):
            if not votes[i]:
                continue
            vals, counts = np.unique(votes[i], return_counts=True)
            pred = vals[counts.argmax()]
            counted += 1
            if pred == self._y_train[i]:
                correct += 1
        return correct / counted


# ---------------------------------------------------------------------------
# Demo 1 --- from scratch
# ---------------------------------------------------------------------------

def demo_from_scratch(X_train, X_test, y_train, y_test):
    banner("DEMO 1 --- Random forest from scratch on the moons dataset")

    n_trees = 200
    max_depth = 6
    print(f"  Training set : {len(X_train)} examples, "
          f"{X_train.shape[1]} features")
    print(f"  Test set     : {len(X_test)} examples")
    print(f"  n_trees      : {n_trees}   max_depth : {max_depth}   "
          f"max_features : sqrt")

    forest = RandomForestClassifier(n_trees=n_trees,
                                    max_depth=max_depth,
                                    max_features="sqrt",
                                    random_state=RNG_SEED)
    forest.fit(X_train, y_train)

    train_acc = (forest.predict(X_train) == y_train).mean()
    test_acc = (forest.predict(X_test) == y_test).mean()
    oob_acc = forest.oob_score()
    print(f"  Train accuracy : {train_acc:.3f}")
    print(f"  Test accuracy  : {test_acc:.3f}")
    print(f"  OOB accuracy   : {oob_acc:.3f}")
    return forest


# ---------------------------------------------------------------------------
# Demo 2 --- sklearn comparison
# ---------------------------------------------------------------------------

def demo_sklearn(X_train, X_test, y_train, y_test, our_forest):
    banner("DEMO 2 --- Same data, scikit-learn RandomForestClassifier")

    sk = SkForest(n_estimators=200, max_depth=6,
                  max_features="sqrt", oob_score=True,
                  bootstrap=True, random_state=RNG_SEED)
    sk.fit(X_train, y_train)
    train_acc = (sk.predict(X_train) == y_train).mean()
    test_acc = (sk.predict(X_test) == y_test).mean()
    print(f"  Train accuracy : {train_acc:.3f}")
    print(f"  Test accuracy  : {test_acc:.3f}")
    print(f"  OOB accuracy   : {sk.oob_score_:.3f}")

    agree = (sk.predict(X_test) == our_forest.predict(X_test)).sum()
    print(f"  Agreement with from-scratch forest on test set: "
          f"{agree}/{len(X_test)} predictions identical")


# ---------------------------------------------------------------------------
# Demo 3 --- variance reduction over re-seeds
# ---------------------------------------------------------------------------

def demo_variance(X_train, X_test, y_train, y_test):
    banner("DEMO 3 --- Random forest vs single decision tree")

    # Single tree at depth 6 (the depth from Part 5's best run)
    tree = SkTree(criterion="gini", max_depth=6,
                  random_state=RNG_SEED)
    tree.fit(X_train, y_train)
    tree_acc = (tree.predict(X_test) == y_test).mean()
    print(f"  Single tree (depth 6) test acc : {tree_acc:.3f}")

    forest = SkForest(n_estimators=200, max_depth=6,
                      max_features="sqrt", bootstrap=True,
                      random_state=RNG_SEED)
    forest.fit(X_train, y_train)
    forest_acc = (forest.predict(X_test) == y_test).mean()
    print(f"  Random forest         test acc : {forest_acc:.3f}")

    # Variance under training-set perturbations: bootstrap the
    # training set 10 times, train tree vs forest on each, measure
    # spread of test accuracy. Single trees should be unstable.
    n_reseeds = 10
    n_train = len(X_train)
    tree_scores, forest_scores = [], []
    rng = np.random.default_rng(123)
    for _ in range(n_reseeds):
        idx = rng.integers(0, n_train, size=n_train)
        Xb, yb = X_train[idx], y_train[idx]

        tr = SkTree(criterion="gini", max_depth=6)
        tr.fit(Xb, yb)
        tree_scores.append((tr.predict(X_test) == y_test).mean())

        fr = SkForest(n_estimators=200, max_depth=6,
                      max_features="sqrt", bootstrap=True,
                      random_state=RNG_SEED)
        fr.fit(Xb, yb)
        forest_scores.append((fr.predict(X_test) == y_test).mean())
    print(f"  Test-set variance over {n_reseeds} training-set perturbations:")
    print(f"      single tree   : ±{np.std(tree_scores):.3f}")
    print(f"      random forest : ±{np.std(forest_scores):.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    X_train, X_test, y_train, y_test = make_dataset()
    forest = demo_from_scratch(X_train, X_test, y_train, y_test)
    demo_sklearn(X_train, X_test, y_train, y_test, forest)
    demo_variance(X_train, X_test, y_train, y_test)
    print()


if __name__ == "__main__":
    main()
