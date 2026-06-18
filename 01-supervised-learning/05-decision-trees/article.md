# Decision Trees — Twenty Questions, Learned from Data

### *Algorithms in Python --- Supervised Learning, Part 5*

---

In Part 4 we built a K-Nearest Neighbours classifier whose
decision boundary bent itself to fit the data — locally,
implicitly, by averaging the labels of nearby training points.
Today we look at a model that takes the opposite approach:
it draws the decision boundary *explicitly*, as a sequence of
axis-aligned splits, and the resulting "if-then" rules are so
human-readable that small trees can be drawn on a whiteboard
and discussed by hand.

The algorithm is the **Decision Tree**. Pick a feature, pick a
threshold, ask "is `x_i ≤ t`?" — if yes, go left; if no, go
right. Repeat in each child until the local subset is pure
enough to stop, then label the leaf with the majority class
(for classification) or the mean target (for regression).
The whole model is a flowchart of yes/no questions, and the
"learning" reduces to a single question we keep asking
recursively: *which split makes the children as pure as
possible?*

This article builds a CART-style decision tree from first
principles. We will derive Gini impurity (the splitting
criterion sklearn uses by default), implement the recursive
algorithm in numpy, train it on a synthetic 2D dataset where
linear classifiers fail, compare against scikit-learn's
`DecisionTreeClassifier`, and finish with the *real* reason
decision trees matter: as the building block of the random
forests, gradient-boosting machines, and XGBoost-style models
that win most tabular ML benchmarks. A single tree is a baseline
with personality. An ensemble of them is often state of the art.

---

## The basic idea

A decision tree partitions the feature space into a set of
non-overlapping axis-aligned rectangles, and assigns one
prediction to every rectangle. Internally it is just a binary
tree:

- An **internal node** holds a question — `(feature_index, threshold)` —
  and routes the example to its left or right child.
- A **leaf** holds a prediction — a class label, a probability
  vector, or a regression value.

Prediction is trivial. Start at the root, evaluate the question,
walk left or right, repeat until you hit a leaf, return what is
there. Total cost is `O(depth)` — for a balanced tree on
`n` examples, that is `O(log n)`.

Training is the interesting part. We need to learn:

- *Which* feature to split on at each node.
- *What* threshold to use.
- *When* to stop splitting and turn the node into a leaf.

The classical answer — and the one CART (Breiman et al, 1984)
made the de-facto standard — is to be greedy. At every node, try
every feature and every candidate threshold, score how good the
resulting split is, take the best one, recurse. Stop when the
node is pure, or when a stopping rule kicks in (max depth,
minimum samples per leaf, minimum impurity decrease).

---

## Picking the best split

The whole algorithm hinges on one number: how do we score a
candidate split?

The intuition is the *purity* of the children. A split is good
if the two subsets it produces are each more class-homogeneous
than the parent. Pure children mean the split has separated the
classes well; mixed children mean the split was uninformative.

Two impurity measures dominate.

### Gini impurity

For a node with class proportions `p_1, p_2, ..., p_K`:

```
Gini(node) = 1 - Σ_k p_k²
```

It ranges from `0` (perfectly pure — one class) to `1 - 1/K`
(uniform across all `K` classes). Intuition: it is the
probability that a randomly drawn pair of examples from the node
have different labels.

### Entropy and information gain

```
Entropy(node) = - Σ_k p_k · log₂ p_k
```

Ranges from `0` (pure) to `log₂ K` (uniform). It measures how
surprised you would be on average by the next label. **Information
gain** is the entropy of the parent minus the weighted entropy of
the children — the reduction in uncertainty produced by the split.

Both criteria pick essentially the same splits in practice. CART
uses Gini (cheaper to compute, no log), C4.5 uses entropy.
Sklearn's default is Gini.

### The split score

For a candidate split that sends `n_L` examples left and `n_R`
examples right (with `n = n_L + n_R` total in the parent), we
score it as the *weighted impurity of the children*:

```
score(split) = (n_L / n) · Gini(left)
             + (n_R / n) · Gini(right)
```

The best split at this node is the one that minimises this
score. Equivalently, the one that maximises the *impurity
decrease* — the parent's Gini minus the score above.

---

## The recursive algorithm

Putting it together:

```
build_tree(X, y, depth):
    if stopping criteria met:
        return Leaf(majority_class(y))

    best_score = +inf
    best_split = None
    for feature in features:
        for threshold in candidate_thresholds(X[:, feature]):
            left_mask = X[:, feature] <= threshold
            score = weighted_gini(y[left_mask], y[~left_mask])
            if score < best_score:
                best_score = score
                best_split = (feature, threshold, left_mask)

    if best_split is None:
        return Leaf(majority_class(y))

    f, t, mask = best_split
    left  = build_tree(X[mask],  y[mask],  depth + 1)
    right = build_tree(X[~mask], y[~mask], depth + 1)
    return Node(feature=f, threshold=t, left=left, right=right)
```

Stopping criteria are usually a combination of:

- **Pure node** — every example has the same label.
- **Max depth** — caps overfitting and keeps prediction fast.
- **Min samples per leaf / per split** — refuses splits that
  would create tiny, noisy leaves.
- **Min impurity decrease** — refuses splits that barely improve
  purity.

Without any of these stopping rules a tree will keep growing
until every leaf has one example. That memorises the training
set perfectly and almost always overfits.

For the candidate thresholds, the classical choice is the
midpoints between consecutive sorted values of the feature in
the current node. With `m` distinct values that gives `m - 1`
candidates per feature.

---

## A worked example

The companion script generates a 2D two-class dataset shaped
like two interleaving half-moons — a problem where any linear
classifier is doomed but a tree can carve out the curved
boundary with a sequence of axis-aligned splits.

The from-scratch tree:

```python
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.root = self._build(np.asarray(X, float),
                                np.asarray(y), depth=0)
        return self

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1.0 - (p * p).sum()

    def _build(self, X, y, depth):
        # Stopping criteria
        if (len(np.unique(y)) == 1
                or len(y) < 2 * self.min_samples_leaf
                or (self.max_depth is not None
                    and depth >= self.max_depth)):
            return _Leaf(self._majority(y))
        # Greedy best split over every feature and threshold
        feat, thr, score = self._best_split(X, y)
        if feat is None:
            return _Leaf(self._majority(y))
        mask = X[:, feat] <= thr
        left  = self._build(X[mask],  y[mask],  depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return _Node(feat, thr, left, right)
```

The helper methods (`_best_split`, `_majority`) and a tiny
`_Node` / `_Leaf` pair are in the full script. Run it on the
moons dataset:

```
DEMO 1 --- Decision tree from scratch on the moons dataset
  Training set : 400 examples, 2 features
  Test set     : 100 examples
  max_depth    : 6
  Train accuracy : 0.963
  Test accuracy  : 0.940
  Tree depth     : 6   leaves : 19
```

```
DEMO 2 --- Same data, scikit-learn DecisionTreeClassifier
  Train accuracy : 0.963
  Test accuracy  : 0.940
  Tree depth     : 6   leaves : 19
  Agreement with from-scratch tree on test set: 100/100 predictions identical
```

```
DEMO 3 --- The depth-vs-overfitting tradeoff
  depth   train_acc   test_acc   leaves
  -----   ---------   --------   ------
      1       0.823      0.840        2
      2       0.907      0.930        4
      3       0.907      0.930        6
      4       0.907      0.930        9
      6       0.963      0.940       19
     10       0.990      0.930       34
   None       1.000      0.930       39
```

The pattern in the depth sweep is the bias-variance story all
over again. At depth 1 the tree is essentially a single
threshold — high bias, both scores around 0.83. By depth 4 the
tree has captured most of the moon-shape; train sits at 0.91
with test at 0.93. At depth 6 the test score peaks at 0.94. Past
that, the train score climbs steadily towards 1.0 (the tree
memorises the training set) while the test score *falls* —
classic overfitting. The fully grown tree perfectly fits 400
training examples in 39 leaves but generalises slightly worse
than the depth-6 cap.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Three things are worth pulling out. **Training is `O(n · d · depth)`
in the typical case** — at every depth level, every node has to
score every candidate split, and each scoring step looks at all
the data passing through it. Sorting the features once at the
root and reusing the order brings it to that bound; naive
implementations are an order worse. **Prediction is `O(depth)`
per example** — a tiny constant on a balanced tree, often a few
microseconds even on big data. **Memory is `O(nodes)`** — the tree
itself is small (kilobytes for a typical fitted tree), no matter
how big the training set was.

That last property is unusual and powerful: a fitted decision
tree is a *compressed* model of the training data, which is the
opposite of KNN's "the data is the model". You can train on a
million examples and ship a model that fits in a few hundred
KB.

---

## What trees give you for free

The reason decision trees keep showing up — and the reason their
ensemble cousins win every other tabular ML benchmark — is a
short list of properties no other simple model has all of at
once:

**No feature scaling required.** A split is `feature ≤
threshold`. Whether the feature is in millimetres or kilometres
makes no difference; the algorithm picks the threshold to
match. No standardisation, no normalisation. Compare with
distance-based models (KNN), gradient-based linear models
(logistic regression), and neural networks — all of which
need careful preprocessing.

**Mixed types and missing data.** Categorical features can be
handled with one-hot or with native categorical splits (CatBoost
and LightGBM do this directly). Missing values can be sent down
a default branch. None of this requires imputing or one-hot
encoding upstream.

**Non-linear interactions, automatically.** A linear model has
to be told about interactions (`x1 * x2`, `x1²`, etc.) by
feature engineering. A tree discovers them by stacking splits.
*If x1 ≤ 5 and x2 > 3 then class A* is two splits deep — no
feature engineering required.

**Built-in feature importance.** The total impurity decrease
contributed by each feature, summed across all splits using it,
is a free feature-importance score. You get an interpretable
ranking of which features the model relied on without any extra
work.

**Interpretability — when the tree is small.** A depth-3 tree
with 8 leaves can be drawn on one page and read like a
flowchart. Medical decision aids, credit-scoring rules, and
quality-control protocols are still routinely deployed as
hand-pruned shallow decision trees because a domain expert can
look at one and say "yes, that matches my intuition" or "no,
splitting on age there is wrong".

---

## The fundamental weakness

A single decision tree is greedy and local. It picks the locally
best split at every node, never reconsiders, and never explores
alternative trees. This has two unpleasant consequences:

**High variance.** A small change to the training data — drop a
few rows, re-shuffle — can produce a *very* different tree,
because a slightly different split at the root cascades into
completely different children. Compared with linear models,
trees are wildly unstable.

**Greedy is not globally optimal.** Finding the smallest
decision tree consistent with the data is NP-hard. The greedy
heuristic CART uses is fast and usually good but never
guaranteed to be optimal, and the trees it produces sometimes
miss obvious structure that a smarter search would find.

The fix for both problems is the same: **don't ship one tree;
ship hundreds, and aggregate them**. That is exactly what
random forests, extra-trees, gradient boosting, and the modern
GBM stack (XGBoost, LightGBM, CatBoost) all do. Each tree is
still greedy and unstable, but the ensemble is stable, accurate,
and frequently the best model on tabular data. The next track
of this series — Advanced Supervised Learning — opens with
Random Forests for exactly this reason.

---

## Real-world ML and AI connections

Decision trees and their ensembles are the workhorses of tabular
machine learning:

**Gradient-boosted trees on Kaggle and beyond.** The dominant
model class for tabular competitions for the last decade has
been XGBoost, LightGBM, and (more recently) CatBoost — all
ensembles of decision trees fit by gradient boosting. They
routinely beat deep learning on tabular data of the kind
businesses actually have.

**Credit scoring, insurance underwriting, churn prediction.**
Regulated industries that need both accuracy and a readable
model lean heavily on tree-based methods. A gradient-boosted
ensemble plus per-feature SHAP values is the standard stack.

**Medical decision support.** From the Pediatric Appendicitis
Score to APACHE-style ICU scoring rules, hand-pruned shallow
decision trees are still in active clinical use. They are
auditable, defensible, and explainable to a patient.

**Random forests in computational biology and ecology.** Random
Forests (Breiman 2001) became the default classifier in
bioinformatics and species-distribution modelling because they
handle small-n high-d data with mixed types and missing values
without complaining.

**Surrogate / explainability models.** When a complex black-box
model (a deep neural net, a large GBM) needs to be explained,
one common technique is to fit a small decision tree to mimic
its predictions on a sample of inputs. The tree gives a
human-readable approximation of the black box's behaviour.

**Rule extraction for production systems.** Some shipped systems
take the *splits* learned by a tree and translate them into
hand-edited business rules — keeping the model out of the
serving path entirely. This is rare in modern ML but still
common in compliance-heavy domains.

The pattern: a single decision tree is rarely the production
model in 2026, but a tree-based *ensemble* almost always is, on
tabular data.

---

## When NOT to use decision trees

Single decision trees are rarely the right shipping model.
Specifically, walk away when:

**The decision boundary is genuinely linear or smooth.** A tree
approximates a linear boundary with a staircase — many splits to
get a so-so fit. Logistic regression beats a tree on truly
linear problems with less data and a smoother boundary.

**The data is high-dimensional and sparse.** Text data
represented as bag-of-words — thousands of features, mostly
zero — is not tree territory. Linear models (logistic
regression, SVM) and Naive Bayes dominate here, partly because
each tree split touches one feature at a time and the search
becomes wasteful.

**You need a single, well-calibrated probability.** Trees
produce step-function probability estimates with limited
resolution (each leaf is one bucket of training labels).
Calibration on top, or a logistic regression instead, is often
cleaner.

**The data has structure trees can't see.** Image pixels, audio
samples, sequences — anything where adjacent features are
strongly related and the model needs to learn a continuous
function over them. Convolutions and attention beat trees on
these problems by orders of magnitude.

**You want a single shipping model and you are not bound to a
single tree.** Almost always, swap to a random forest or a
gradient-boosted ensemble. A single tree is a teaching aid; a
forest of trees is a model.

---

## What comes next

This is the last article in the introductory Supervised Learning
track. Part 6 begins **Advanced Supervised Learning** with
**Random Forests** — the bagging ensemble that takes everything
weak about a single decision tree and trades it for an averaged
prediction that is stable, accurate, and almost as easy to fit
as a single tree. Random forests will lead into gradient
boosting (XGBoost, LightGBM, CatBoost) and then to support
vector machines, rounding out the classical supervised toolkit
before we move on to unsupervised learning.

If you have followed the series from Linear Regression through
Logistic Regression, Naive Bayes, KNN, and now Decision Trees,
you have all five "first models to try" on a new tabular
problem. The next track is what to reach for when those five
are not enough.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**decision_tree.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/01-supervised-learning/05-decision-trees/decision_tree.py)

Run it with:

```bash
pip install numpy scikit-learn
python decision_tree.py
```

It needs `numpy` and `scikit-learn`. The script implements a
CART-style decision tree from scratch, fits it to the
two-moons dataset, compares against scikit-learn's
`DecisionTreeClassifier` (the predictions agree on every test
example), and runs a depth sweep that reproduces the textbook
overfitting curve — train accuracy climbing to 1.0 while test
accuracy peaks and then falls. The headline insight worth
pinning to the wall: **a decision tree is a flowchart whose
questions were chosen by greedy impurity minimisation; a
single tree overfits, but an ensemble of them is the dominant
model class for tabular data**.

---

*This is Part 5 of the Algorithms in Python series, Supervised Learning track. The companion script `decision_tree.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 4](https://medium.com/p/bf60b8986801) covered K-Nearest Neighbours. The next track — Advanced Supervised Learning — opens with Random Forests, the ensemble that takes a single tree's weaknesses and averages them away.*
