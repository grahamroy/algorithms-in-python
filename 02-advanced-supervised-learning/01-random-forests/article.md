# Random Forests — When a Hundred Greedy Trees Beat One

### *Algorithms in Python --- Advanced Supervised Learning, Part 1*

---

In Part 5 we built a CART-style decision tree, watched it carve
a two-moons dataset into 19 axis-aligned regions, and ran a
depth sweep that showed the textbook overfitting curve — train
accuracy climbing to 1.0 while the test score peaked at depth 6
and then drifted *down*. The diagnosis was clear: a single
decision tree is greedy, unstable, and prone to memorising the
training set. The treatment is the subject of this article.

Don't ship one tree. Ship many.

That is the headline insight of the **Random Forest** (Breiman,
2001) — fit hundreds of intentionally-different decision trees
to the same data and aggregate their predictions. A single
greedy tree is a high-variance classifier with personality; a
forest of hundreds of greedy trees, averaged, is a stable,
accurate, and very hard-to-beat baseline. For a decade and a
half (until the rise of gradient boosting in its modern form)
random forests were *the* default model class on tabular data,
and they are still the right first reach when you want
"accuracy out of the box with very few knobs to tune".

This article builds a Random Forest from first principles. We
will explain why averaging is the variance killer, why naive
averaging of trees doesn't work, why **bagging** and **random
feature subsampling** together fix that, implement the forest
from scratch on top of the tree we wrote in Part 5, look at the
free validation set (out-of-bag estimation) that forests give
you for nothing, and finish with the family of methods random
forests gave birth to — Extra-Trees, Isolation Forest, and the
gradient-boosting machines we tackle next time.

---

## The variance problem, restated

A single decision tree is *greedy*: at every node it picks the
locally-best split and never reconsiders. Tiny perturbations of
the training set produce wildly different first splits, which
cascade into completely different children. The result is high
variance — train two trees on slightly different samples of the
same problem and you get two different-looking models. Their
predictions on a held-out test set will disagree more than you
would like.

Statistics gives us a clean way to think about this. If we have
`N` independent predictors each with variance `σ²` and we
average their predictions, the variance of the average is:

```
Var(average) = σ² / N
```

Twenty independent predictors will have one twentieth the
variance of one. A hundred will have one hundredth. Averaging is
the strongest variance-reduction tool we have — *if the
predictors are independent*.

That is the catch. Train one hundred decision trees on the *same
data*, and you do not get one hundred independent predictors —
you get one hundred copies of essentially the same tree, because
the greedy algorithm picks the same dominant first split every
time. Their predictions are highly correlated, the effective
"N" in the variance formula collapses, and the averaging gains
very little.

To make averaging work we need the trees to be different. That
is the entire job of a random forest: inject just enough
randomness during training that the trees disagree, then average
them.

---

## Ingredient 1 — Bootstrap aggregation

The first trick is **bagging** (Breiman, 1996): instead of
training every tree on the full training set, train each tree on
a *bootstrap sample* — `n` rows drawn with replacement from the
original `n`-row training set.

```
bootstrap(X, y):
    indices = random.choice(n, size=n, replace=True)
    return X[indices], y[indices]
```

A bootstrap sample is the same size as the training set, but
because it samples with replacement, some rows appear multiple
times and others not at all. In expectation about `63.2%` of the
unique rows make it in (`1 - (1 - 1/n)^n → 1 - 1/e`). The other
`36.8%` are **out-of-bag** for that tree — unused at training,
free to use as a held-out evaluation set.

Bagging by itself helps a bit. Each tree now sees a slightly
different training distribution, picks slightly different
splits, and ends up slightly different from its peers. But not
*very* different — the strongest predictive feature is usually
strong in every bootstrap, so every tree still chooses it for
the root split. The trees are decorrelated but not enough.

---

## Ingredient 2 — Random feature subsampling

The second trick is what turns bagging into a random forest.
At every internal node, instead of searching over *all* features
for the best split, search only over a random subset of them.

```
choose_split(X, y, available_features):
    # Pick a random subset, then do the usual greedy search
    candidates = random.choice(available_features, size=k,
                               replace=False)
    return best_split_among(candidates, X, y)
```

Typical choices of `k`:

- **Classification:** `k = √d` (e.g. on 100 features, each split
  considers 10).
- **Regression:** `k = d / 3`.

The mtry parameter — as the R community has called it for
twenty years — is the most important knob in a random forest.
Small `k` produces more diverse trees and lower correlation;
large `k` produces stronger individual trees but more correlated
ones. The defaults above strike a good balance for most
problems.

What this does is *force* the trees to disagree. The dominant
feature can only be considered at a node if it happens to be in
that node's random subset. When it is not, the tree has to pick
the second-best (or worse) split — producing a structurally
different tree from its peers. Across hundreds of trees,
correlation drops sharply, the averaging actually works, and
the variance of the ensemble collapses.

Bagging *plus* feature subsampling is the random forest. Both
ingredients are necessary; either alone leaves a lot on the
table.

---

## The full algorithm

Putting the two ingredients together:

```
RandomForest.fit(X, y, n_trees, k_features, max_depth):
    forest = []
    for t in 1..n_trees:
        X_t, y_t, in_bag_indices = bootstrap(X, y)
        # bootstrap returns the sampled rows AND their indices
        tree = DecisionTree(max_depth, k_features=k_features)
        tree.fit(X_t, y_t)
        forest.append((tree, in_bag))
    return forest

RandomForest.predict(X):
    # Classification: each tree votes; majority wins
    votes = [tree.predict(X) for tree, _ in forest]
    return majority_per_row(votes)
```

The trees inside the forest are the same decision trees from
Part 5 — they just sample features at each split rather than
looking at all of them. Everything else (the recursive build,
the Gini criterion, the stopping rules) is unchanged.

Prediction is unweighted: every tree gets one equal vote
(classification) or contributes equally to the mean (regression).
Some implementations weight the vote by tree depth or by an
estimate of each tree's accuracy, but the standard random forest
keeps it simple.

---

## Out-of-bag estimation — a free validation set

Because each tree was trained on a bootstrap sample, each tree
has roughly `37%` of the training rows it never saw. For each
row in the original training set, we can predict its label using
only the trees that did not see it during training. That gives
us a held-out prediction for *every row*, without ever splitting
the data.

```
oob_predict(X_train):
    for row i in X_train:
        contributing_trees = [tree for tree, in_bag in forest
                              if i not in in_bag]
        prediction[i] = majority_vote([t.predict(X_train[i])
                                       for t in contributing_trees])
    return prediction

oob_score = accuracy(y_train, oob_predict(X_train))
```

The OOB accuracy is a nearly-free, slightly conservative estimate
of the random forest's generalisation accuracy (each row is
predicted by only the ~37% of trees that did not see it). For
most problems it tracks
cross-validation accuracy to within a fraction of a percent,
for free. This is one of the underappreciated wins of bagging
ensembles: you do not need a separate validation split to
monitor overfitting; the forest produces one as a side effect of
training.

---

## A worked example

The companion script reuses the two-moons dataset from Part 5 —
exactly 400 train + 100 test, same seed, same problem — and
fits a random forest of 200 trees on top of it.

```
DEMO 1 --- Random forest from scratch on the moons dataset
  Training set : 400 examples, 2 features
  Test set     : 100 examples
  n_trees      : 200   max_depth : 6   max_features : sqrt
  Train accuracy : 0.958
  Test accuracy  : 0.970
  OOB accuracy   : 0.930
```

```
DEMO 2 --- Same data, scikit-learn RandomForestClassifier
  Train accuracy : 0.958
  Test accuracy  : 0.970
  OOB accuracy   : 0.917
  Agreement with from-scratch forest on test set: 100/100 predictions identical
```

```
DEMO 3 --- Random forest vs single decision tree
  Single tree (depth 6) test acc : 0.940
  Random forest         test acc : 0.970
  Test-set variance over 10 training-set perturbations:
      single tree   : ±0.020
      random forest : ±0.012
```

Two things to pull out. First, the test accuracy *increased*
from `0.940` (single tree) to `0.970` (forest) on the same
dataset, even though every individual tree in the forest is
trained on a smaller bootstrap sample of the data with weaker
splits (only one of two features considered per split). That is
the magic of the ensemble. Second — and arguably more important
— the **variance** of the test accuracy under bootstrap
perturbations of the training set dropped from `±0.020` to
`±0.012`. The random forest is not just more accurate on this
run; it is more *reliably* accurate. That is the property
production systems care about, and the gap widens on harder
problems with more features.

The OOB accuracy (`0.930` from our implementation, `0.917` from
sklearn's) sits below the held-out test accuracy (`0.970`),
which is the expected pattern: OOB is conservative because each
training row is predicted by only the ~37% of trees that did
not see it, whereas test predictions use the entire forest.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The story is straightforward. **Training is `N · O(tree)`** —
the cost of one tree times the number of trees, and trees are
independent so this is *embarrassingly parallel*. A 200-tree
forest on 16 cores costs roughly the same as a 13-tree forest
serial. **Prediction is `N · O(depth)` per example** — every
tree has to walk root-to-leaf — which is fast in absolute
terms (each tree is microseconds) but does scale linearly in
the number of trees. **Memory is `N · O(nodes)`** — each fitted
tree is small, but a 200-tree forest is genuinely 200 times the
size of a single tree, often a few megabytes. The model is no
longer KB-sized.

Practical implication: random forests are excellent at training
time and good at prediction time, but the model is fat compared
with a single tree or a logistic regression. For very large
forests (`N = 1000+`), production deployments often distill the
forest into a smaller approximating model.

---

## Feature importance

Forests come with two built-in interpretability tools.

**Mean Decrease in Impurity** (the sklearn default) sums, for
each feature, the impurity decrease produced by every split
that uses it, averaged across trees. Features that produce big
splits high in many trees get high importance. It is fast and
free — falls out of training — but has a known bias toward
high-cardinality features (a continuous feature with many
possible split points will look more important than a binary
feature even when they are equally informative).

**Permutation importance** shuffles the values of one feature
in the test set and measures how much the forest's accuracy
drops. Repeat for each feature. Slower but model-agnostic and
unbiased — for honest comparisons across features of different
types, this is the importance metric to use, and the one most
modern interpretability libraries default to.

Both are diagnostic, not causal. A feature that scores high in
permutation importance is one the forest *relies on*; whether it
is genuinely causal in the underlying process is a different
question entirely.

---

## Real-world ML and AI connections

For most of the 2000s and 2010s, random forests were the model
class to beat on tabular data:

**Bioinformatics and genomics.** Breiman's RF became the default
classifier for gene-expression analysis, SNP association
studies, and metagenomics. The combination of small `n`, large
`d`, mixed feature types, and a desire for interpretable feature
importance made it a natural fit, and the field is still full of
RF baselines.

**Pre-deep-learning computer vision.** Microsoft's Kinect
real-time body-part recognition (Shotton et al, 2011) was a
deep, wide random forest predicting body parts per pixel from
depth images. Random forests were a standard CV tool for face
detection, gesture recognition, and segmentation before
convolutional networks displaced them.

**Anomaly and fraud detection.** Isolation Forest (Liu et al,
2008) is a random-forest variant where shorter average path
length to isolation = stronger anomaly score. It ships in every
production fraud-detection stack.

**Kaggle and competitive ML.** Random forests dominated tabular
Kaggle competitions through the early 2010s before being
overtaken by gradient-boosted ensembles. They remain the
five-minute baseline every Kaggler fits before reaching for
XGBoost.

**Causal forests and uplift modelling.** Athey & Wager's *Generalized
Random Forests* (2019) and related causal-forest variants are
the modern way to estimate heterogeneous treatment effects in
econometrics and personalised medicine. The
bagging-plus-subsampling recipe scales naturally to causal
estimands.

**Industrial baseline.** In regulated industries where
explainability matters (credit, insurance, regulated medical
devices), a random forest plus permutation importance plus
isotonic calibration is still a standard production stack. It
is unfashionable but rock-solid.

The pattern: random forests are the workhorse model. Gradient
boosting beats them on raw accuracy in most modern benchmarks,
but random forests are easier to tune, easier to debug, and
easier to explain — and they often still win when the data is
small or noisy.

---

## When NOT to use random forests

Random forests are a strong baseline but they are not always
the right answer:

**When latency or memory is tight.** A 500-tree forest with
depth-20 trees is megabytes of model and milliseconds of
prediction. If you need a sub-millisecond classifier on a
microcontroller, a single shallow tree or a logistic regression
will get you there; a forest will not.

**When gradient boosting would clearly win.** On well-curated
tabular data with `n > 10⁴`, XGBoost / LightGBM / CatBoost will
beat a random forest by a noticeable margin most of the time. If
you are already willing to deploy a large ensemble, deploy the
right one. The next article will get into why.

**When the decision boundary is smooth or linear.** Trees
approximate continuous functions with staircases. Linear models,
GAMs, or neural networks handle smooth functions with less data
and better extrapolation. (Random forests, like single trees,
*cannot* extrapolate beyond the range of the training data — the
prediction is bounded by the leaf values they have seen.)

**When you need a well-calibrated probability.** Forest
probabilities (the fraction of trees voting for each class) are
biased toward 0.5 because of the averaging. Calibration on top
(Platt scaling or isotonic regression) is usually mandatory if
you need real probabilities downstream.

**When you have very high-cardinality categoricals without
preprocessing.** A feature with 10,000 unique values will
dominate the split space and the impurity-based importance
scores. One-hot encoding helps a little; target encoding, or
switching to CatBoost (which handles native categoricals well),
helps more.

---

## What comes next

Part 2 of the Advanced Supervised Learning track is **Gradient
Boosting** — the family of methods that fits trees *sequentially*,
each new tree correcting the residuals of the ensemble so far.
Where random forests reduce variance by averaging independent
trees, gradient boosting reduces bias by adding dependent
trees, and the resulting model (XGBoost, LightGBM, CatBoost)
is the dominant model class on modern tabular ML benchmarks.
The two families share the underlying decision tree as their
atom but achieve different things with it.

After gradient boosting we look at **Support Vector Machines**,
the classical large-margin classifier that ruled the late 1990s
before trees ate everyone's lunch, and then move on to
unsupervised learning.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**random_forest.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/02-advanced-supervised-learning/01-random-forests/random_forest.py)

Run it with:

```bash
pip install numpy scikit-learn
python random_forest.py
```

It needs `numpy` and `scikit-learn`. The script implements a
random forest from scratch by composing the decision tree from
Part 5 with bootstrap sampling and per-split feature
subsampling, fits it to the same moons dataset Part 5 used,
compares against scikit-learn's `RandomForestClassifier` (which
agrees on 100 out of 100 test predictions, with a comparable OOB
accuracy), and finishes with a head-to-head against a single
decision tree showing both the accuracy lift and — more
importantly — the variance reduction across re-seeds. The
headline insight worth pinning to the wall: **bag the data,
subsample the features, average the predictions, and the
greedy-tree pathology that plagued Part 5 disappears**.

---

*This is Part 1 of the Advanced Supervised Learning track in the Algorithms in Python series. The companion script `random_forest.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 5](https://medium.com/p/9383044e2a57) (the previous article, in the introductory Supervised Learning track) covered Decision Trees. Part 2 of this track will look at Gradient Boosting — the sequential cousin of the random forest, and the family behind XGBoost, LightGBM, and CatBoost.*
