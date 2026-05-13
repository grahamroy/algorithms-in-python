# Gradient Boosting — Trees that Correct Each Other's Mistakes

### *Algorithms in Python --- Advanced Supervised Learning, Part 2*

---

In Part 1 we built a random forest — hundreds of decision trees,
each fit on a different bootstrap of the training data, each
considering only a random subset of features at every split,
their predictions averaged into a single answer. The forest's
job was to take a high-variance base model (a single deep tree)
and average that variance away. Many trees trained
*independently*, vote at the end.

Today we look at the other big tree ensemble — the one that has
been winning tabular ML benchmarks for the last decade. It
trains its trees *sequentially*. Each new tree is fit to the
*residuals* of the ensemble so far — the part of the answer the
previous trees got wrong. Add the new tree, recompute residuals,
fit another tree, repeat. The ensemble does not get more diverse
with each tree; it gets more *accurate*, because each tree is
explicitly trying to fix the previous trees' mistakes.

The algorithm is **Gradient Boosting**. Friedman's 1999 paper
*Greedy Function Approximation: A Gradient Boosting Machine* is
the foundational reference, but the modern names you have heard
are the implementations that made it practical at scale —
**XGBoost** (Chen & Guestrin, 2016), **LightGBM** (Microsoft,
2017), and **CatBoost** (Yandex, 2017). These three between them
have won an astonishing fraction of every tabular Kaggle
competition since 2015. On structured business data — credit,
fraud, click-through, ranking, demand forecasting —
gradient-boosted trees are still, in 2026, the model class to beat.

This article builds gradient boosting from first principles. We
will derive the algorithm as gradient descent in *function
space*, implement it from scratch on top of the regression tree
we have been carrying since Part 5, train it on the same moons
dataset our forest used, compare with scikit-learn's
`GradientBoostingClassifier`, and finish with the modern stack
— XGBoost / LightGBM / CatBoost — and where each one shines.

---

## The intuition: sequential error correction

Boosting starts from a deliberately weak model — often the
single best constant prediction (the class log-odds for
classification, the mean for regression). On any non-trivial
problem this baseline makes lots of errors. The boosting idea
is to attack those errors directly.

Compute the *residual* at every training point — the gap between
the truth and what the current model predicts. Fit a small
regression tree to those residuals: it does not have to be a
good classifier, just a function that points in the direction
of the missing signal. Add it (scaled by a small learning rate)
to the running prediction. Recompute residuals — they are now
smaller, because the tree picked up some of the missing signal.
Fit another tree, and another, and another.

After `M` trees the ensemble is:

```
F_M(x) = F_0(x) + η · h_1(x) + η · h_2(x) + ... + η · h_M(x)
```

Each `h_m(x)` is a small regression tree, and `η` is a learning
rate that controls how much each tree is allowed to move the
prediction. Crucially, every `h_m` is fit to the residuals of
`F_{m-1}` — to whatever the previous ensemble got wrong. Trees
are *not* independent. Trees are *chained*.

The model class is the same as a random forest's (sums of
decision trees), but the training objective and dynamics are
completely different. Random forests reduce *variance* via
averaging. Gradient boosting reduces *bias* via sequential
correction. Both end up at strong models; gradient boosting
typically wins on well-curated tabular data.

---

## The math: gradient descent in function space

The "gradient" in gradient boosting comes from the observation
that fitting to residuals is a special case of something more
general. We have a loss function `L(y, F(x))` and an ensemble
`F`. To minimise the loss we want to add a small update
`h(x)` to `F`. Standard gradient descent (in *parameter* space)
moves the parameters in the direction of `-∇L`. Gradient
boosting does the same thing — but in *function* space.

Concretely, at iteration `m`, compute the negative gradient of
the loss with respect to the current model at every training
point:

```
r_im = - ∂L(y_i, F(x_i)) / ∂F(x_i)   evaluated at F = F_{m-1}
```

These `r_im` values are the **pseudo-residuals**. For the
squared-error loss `L = ½(y − F)²` they are literally the
residuals `y − F`. For log loss / cross-entropy they are
`y − σ(F)` — the difference between the label and the sigmoid
of the current log-odds. For other losses they are something
else, but the recipe is the same.

The next tree `h_m(x)` is fit by ordinary regression — a CART
tree built to minimise squared error between its predictions
and the pseudo-residuals. The tree partitions the input space
into leaves and assigns each leaf a value; for a perfectly
chosen tree, walking down to a leaf and reading off the value
moves `F` in exactly the direction that decreases the loss the
most.

Once the tree is fit, we add it scaled by the learning rate:

```
F_m(x) = F_{m-1}(x) + η · h_m(x)
```

A learning rate `η = 0.1` means we take a 10% step in the
direction the tree pointed. Why not 100%? Because fitting one
tree well to noisy residuals is brittle. Taking small steps,
recomputing residuals each time, and fitting a new tree at every
step lets the ensemble correct itself if a tree was
over-confident. The cost is more trees (smaller `η` needs more `M`);
the benefit is far better generalisation.

This is, formally, **steepest-descent in function space**, with
the trees acting as a search procedure for a basis function
that approximates the gradient direction. Friedman's paper is
the rigorous version of the picture above.

---

## The full algorithm

For binary classification with log loss:

```
fit(X, y, M, η, max_depth):
    # 1. Initialise with the best constant log-odds
    F = log(mean(y) / (1 - mean(y)))    # a scalar
    F_train = full_like(y, F)

    trees = []
    for m in 1..M:
        # 2. Pseudo-residuals: y minus current probability
        p = sigmoid(F_train)
        r = y - p

        # 3. Fit a regression tree to the residuals
        h = RegressionTree(max_depth).fit(X, r)

        # 4. Replace each leaf's mean-residual value with the
        #    Newton-style optimal step for log loss
        for leaf in h.leaves:
            num = sum(r[i] for i in leaf)
            den = sum(p[i] * (1 - p[i]) for i in leaf)
            leaf.value = num / (den + 1e-12)

        # 5. Update the model
        F_train += η * h.predict(X)
        trees.append(h)

    return F0 = F, trees

predict_proba(X):
    F = F0 + η * sum(h.predict(X) for h in trees)
    return sigmoid(F)
```

A few things are worth pulling out.

The initial model `F_0` is a single scalar — the log-odds of the
positive class. After training, the ensemble's prediction at any
point is that scalar plus the sum of `M` trees, all scaled by
`η`.

The trees are *regression* trees, not classification trees. They
are fit to a continuous target (the pseudo-residuals) using
squared-error splits. The fact that the final ensemble does
classification comes from the loss function and the sigmoid at
prediction time, not from the trees themselves.

The Newton-step leaf-value update (step 4) is what makes this
*gradient boosting for log loss* specifically. A vanilla
gradient step would just use the mean residual in each leaf;
the Newton step divides by the local curvature of the loss
(`p(1−p)` for log loss) and gives a much better per-step move.
XGBoost generalises this with the full Newton update plus L1/L2
regularisation on leaf weights.

---

## Three knobs that matter

Boosting has three hyperparameters that dominate everything
else:

**Number of trees `M`.** More trees = more capacity, until the
ensemble starts memorising training-set noise. Boosting will
overfit if you let it — the train accuracy climbs forever, the
test accuracy peaks and then drifts down. Cross-validate, or
better, use **early stopping** on a held-out set: keep adding
trees until the validation loss has not improved for some
number of rounds.

**Learning rate `η`.** Typical values are `0.01 – 0.3`. Smaller
learning rates need proportionally more trees but generalise
better, because each tree's contribution is small enough that
the ensemble has many chances to correct any single tree's
over-confidence. The pragmatic rule of thumb: set `η` to the
smallest value you can afford the compute for, then use early
stopping to pick `M`.

**Tree depth.** Boosting works with *weak learners*. Shallow
trees (depth 3 to 8) repeated many times are far more effective
than deep trees boosted a few times. The intuition: a deep tree
captures too much in a single step, leaving nothing for later
trees to refine, and the ensemble loses the gradient-descent
character that makes boosting work.

Two secondary knobs matter for big problems:

- **Subsampling** (`subsample < 1`) — fit each tree on a random
  fraction of the training rows. This is *stochastic gradient
  boosting* (Friedman, 2002) and works exactly like SGD: noisier
  but better generalisation.
- **Column subsampling** (`colsample_bytree` / `bynode`) —
  consider only a random subset of features at each tree or
  each split. Same idea as random forests' feature subsampling,
  ported to boosting.

XGBoost added L1 (`alpha`) and L2 (`lambda`) regularisation on
the leaf weights themselves and a `min_child_weight` knob that
refuses splits which would produce a leaf with too-small Hessian
mass. Together with subsampling and early stopping, these are
the levers that make modern gradient-boosting competitions
winnable.

---

## A worked example

The companion script reuses the two-moons dataset from Parts 5
and 1 — same 400/100 train/test split, same random seed — and
fits a gradient-boosted ensemble of regression trees with log
loss, depth 3, learning rate 0.1, 200 boosting rounds.

```
DEMO 1 --- Gradient boosting from scratch on the moons dataset
  Training set : 400 examples, 2 features
  Test set     : 100 examples
  M = 200   eta = 0.1   max_depth = 3   loss = log
  Train accuracy : 1.000
  Test accuracy  : 0.950
```

```
DEMO 2 --- Same data, scikit-learn GradientBoostingClassifier
  M = 200   eta = 0.1   max_depth = 3
  Train accuracy : 1.000
  Test accuracy  : 0.950
  Agreement with from-scratch model on test set: 100/100 predictions identical
```

```
DEMO 3 --- Number of trees vs accuracy (the boosting curve)
     M    train_acc   test_acc
   ----   ---------   --------
      1       0.907      0.930
      5       0.907      0.930
     20       0.920      0.970
     50       0.958      0.960
    100       0.985      0.950
    200       1.000      0.950
    500       1.000      0.940
   1000       1.000      0.930
```

The boosting curve in Demo 3 is the textbook bias-then-variance
story. Even with one depth-3 tree the ensemble is already at
`0.93` test accuracy on this binary problem — a single shallow
tree captures most of the moon-shape's gross structure. By
`M = 20` boosting has reached the noise floor of the moons
problem (test accuracy `0.970` — close to the irreducible
error). Past that the train score climbs steadily to `1.000`
while the test score *drifts down*. This is overfitting: the
later trees are fitting label noise rather than signal. Early
stopping at `M ≈ 20–50` would land us at the sweet spot.

Two comparisons to keep in mind. The single decision tree from
Part 5 got `0.940` test accuracy on this dataset. The random
forest from Part 1 got `0.970`. Gradient boosting matches the
forest at the sweet spot (`M = 20`, test acc `0.970`), then
overfits past it — exactly the dynamic the regularisation knobs
above are designed to control.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The Big-O comparison with random forests is the most useful
take-away. Both fit `M` trees of similar depth on the same data,
so their *per-tree* training costs are the same. The crucial
difference: forest trees are **independent** and can be fit in
parallel; boosted trees are **sequential** because each tree
depends on the residuals of the previous ensemble. A 16-core
machine speeds up a random forest by ~16×; it does almost
nothing for vanilla gradient boosting. The within-tree
operations (split finding) can still be parallelised, and the
big libraries — XGBoost's `hist` mode, LightGBM, CatBoost — all
exploit this. But across trees, boosting is fundamentally
serial.

Prediction is `O(M · depth)` per example, the same as a random
forest. With small `M` and shallow trees it is very fast —
sub-millisecond on typical hardware. With `M = 5000` and `depth = 8`
it can become a bottleneck, which is why production systems
sometimes *truncate* the trained ensemble: keep only the first
`M' < M` trees that produce indistinguishable test loss.

Histogram-based variants (LightGBM, XGBoost `hist`, sklearn's
`HistGradientBoostingClassifier`) replace per-feature sort with
quantile binning. Training cost drops from `O(M · n · d ·
depth)` to roughly `O(M · #bins · d · depth)` — sublinear in `n`
beyond the bin count. This is what makes LightGBM and XGBoost
practical on tens of millions of rows.

---

## The modern GBM stack

In production in 2026, "gradient boosting" almost always means
one of three implementations:

**XGBoost** (Chen & Guestrin, 2016). The first implementation
that combined second-order gradients (Newton step), L1/L2
regularisation on leaf weights, sparse-aware splits (a feature
with many zeros gets a "default direction" branch), and a
distributed/GPU-aware training pipeline. Won the Higgs Boson
Kaggle competition and immediately became the new tabular
state-of-the-art. Still the most popular GBM library on
Kaggle.

**LightGBM** (Microsoft, 2017). Histogram-based splits plus
leaf-wise (rather than depth-wise) tree growth. Trees can grow
deeper on the more informative side and shallower elsewhere, so
the same parameter budget captures more signal. Adds native
handling of categorical features and is typically 2–10× faster
than XGBoost on large datasets at comparable accuracy.

**CatBoost** (Yandex, 2017). The big idea is **ordered boosting**
— a target-encoding strategy that avoids the target-leakage
that plagues naive mean-encoding of categoricals. Handles
arbitrary categorical features without one-hot encoding,
which matters when your `d` includes high-cardinality categorical
columns (zip codes, product IDs, user IDs). Typically the best
out-of-the-box accuracy on data with messy categoricals.

These three libraries differ in defaults and ergonomics but the
underlying algorithm is the same gradient-boosting machine
described above, with tighter loops, better split-finding, and
serious regularisation. They have been mature and well-tuned
for nearly a decade and the gap between them and the vanilla
sklearn `GradientBoostingClassifier` (which is correct but
slow) is now an order of magnitude in training time.

---

## Real-world ML and AI connections

Where do gradient-boosted trees actually run in production?

**Search and ad ranking.** LambdaMART (and its descendants)
trains a gradient-boosted regression tree against a ranking loss
(pairwise or listwise). It has been the workhorse ranker at
Bing, Yandex, LinkedIn, and Baidu, and a strong baseline at
Google long before transformer rankers. Modern click-through
rate models still routinely ensemble a transformer with an
XGBoost head over hand-crafted features.

**Credit scoring and fraud detection.** Most major banks and
fintechs train gradient-boosted trees on tabular features
(transaction history, behavioural signals, derived ratios)
because the model is explainable enough for regulators (SHAP
values per feature) and accurate enough to drive real money
decisions. The data is messy, the categoricals are
high-cardinality, the labels are imbalanced — exactly the
problem CatBoost and XGBoost were designed for.

**Click-through and conversion prediction.** Display-ad
auctions need predicted CTR / CVR in milliseconds for billions
of opportunities a day. Two-stage stacks (retrieval + ranking)
typically use a deep neural network for retrieval and a
gradient-boosted ensemble for the final ranking, because the
GBM is faster to score on a few thousand candidates with
hand-crafted features.

**Demand forecasting.** Time-series problems with lots of
features (lagged values, calendar effects, weather, promotion
flags) are gradient-boosted-tree territory. The M5 forecasting
competition (2020) — Walmart sales forecasting — was won by
LightGBM ensembles.

**Anomaly detection on tabular data.** Trees boosted against a
proxy loss (predict next observation, or autoencoder-style
reconstruction) are still widely used for industrial monitoring
and fraud detection when the data is structured.

**Tabular AutoML.** Modern AutoML systems — H2O, AutoGluon,
TPOT, sklearn's `HistGradientBoostingClassifier` defaults — all
treat gradient boosting as the universal baseline. If the model
class is not gradient boosting, you should have a specific
reason.

The pattern is consistent: **on well-curated tabular data with
mixed feature types, gradient boosting is usually the right
answer**. Deep learning rarely beats it without significant
feature engineering, and even then often loses on calibration
and interpretability.

---

## When NOT to use gradient boosting

Boosting is powerful but not universal:

**Image, audio, or text data.** Boosting on raw pixels or audio
samples is a non-starter. Use convolutions or transformers.
Boosting on top of *learned features* (embedding from a CNN /
transformer) is still common — that is a different problem.

**Very small datasets.** A 200-row dataset will overfit a
boosted ensemble before you have any reason to believe it. With
`n < 1000` simpler models (regularised logistic regression,
Naive Bayes, a single tree) often generalise better.

**Strict latency budgets at scale.** A 1000-tree XGBoost model
takes hundreds of microseconds per prediction. If your serving
budget is sub-microsecond per request, distil the ensemble into
a smaller approximating model — or use a single tree.

**When you need fast iteration on the loss.** Custom losses on
GBMs require either second-derivative analysis or numerical
gradients, both of which slow training. PyTorch with a custom
loss + autograd is sometimes a faster way to prototype.

**When the data has structure GBM cannot see.** Time series
with long-range dependencies, sequences, graphs, anything where
the relationship is not "function of a fixed-dimensional vector
of features" — gradient-boosted trees do not natively model
these structures.

In all of these cases the right answer is to either preprocess
the data into a tabular feature vector (and boost) or use a
model class designed for the structure (and not boost). The
"is boosting the right tool?" question reduces to "is my data
already a clean table of features I trust?"

---

## What comes next

Part 3 of the Advanced Supervised Learning track is **Support
Vector Machines** — the classical large-margin classifier that
dominated supervised learning in the late 1990s and early
2000s before tree-based methods took over. SVMs are the cleanest
worked example of two of the most consequential ideas in
machine learning: the *margin maximisation* principle and the
*kernel trick* that lifts linear classifiers into rich
non-linear feature spaces without ever computing the embedding
explicitly. After SVMs the supervised toolkit is complete and we
move on to unsupervised learning.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**gradient_boosting.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/02-advanced-supervised-learning/02-gradient-boosting/gradient_boosting.py)

Run it with:

```bash
pip install numpy scikit-learn
python gradient_boosting.py
```

It needs `numpy` and `scikit-learn`. The script implements a
gradient-boosting machine for binary classification with log
loss, built on top of a regression tree that re-uses the
splitter from Part 5. It fits the moons dataset, compares against
scikit-learn's `GradientBoostingClassifier` (100/100 predictions
agree), and runs the M-sweep that produces the textbook boosting
curve — fast bias reduction up to a sweet spot, then slow
overfitting as more trees are added. The headline insight worth
pinning to the wall: **gradient boosting is steepest-descent in
function space; each tree fits the negative gradient of the
loss, scaled by a small learning rate, and the ensemble grows
until early stopping says to stop**.

---

*This is Part 2 of the Advanced Supervised Learning track in the Algorithms in Python series. The companion script `gradient_boosting.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 1](https://medium.com/p/54cee072a674) of this track covered Random Forests — the bagging cousin of gradient boosting. Part 3 will look at Support Vector Machines — the classical large-margin classifier and the natural home of the kernel trick.*
