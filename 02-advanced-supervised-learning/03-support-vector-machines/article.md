# Support Vector Machines — Max Margins and the Kernel Trick

### *Algorithms in Python --- Advanced Supervised Learning, Part 3*

---

In Parts 1 and 2 we built tree ensembles — random forests
(averaging hundreds of independent trees) and gradient-boosted
machines (stacking dependent trees that correct each other's
mistakes). Both are non-parametric, both work well on tabular
data, and both have a clear "more trees = more capacity"
intuition.

Today we look at the model that ruled supervised learning for a
decade before tree ensembles took over and convolutional
networks displaced almost everything in computer vision. The
**Support Vector Machine** (Boser, Guyon & Vapnik, 1992;
generalised to soft margins by Cortes & Vapnik, 1995) is built
from two of the most consequential ideas in classical machine
learning: the **maximum-margin principle** and the **kernel
trick**. Together they give you a classifier that finds the
"widest possible gap" between classes, optionally lifted into a
non-linear feature space without ever computing the embedding
explicitly. SVMs won the digit-recognition benchmarks of the
1990s, drove a generation of text-classification research, and
are still — when calibrated and well-tuned — competitive on
modestly-sized problems with low-dimensional structure.

This article builds the SVM from first principles. We will
derive the linear maximum-margin classifier as a quadratic
program, soften it for non-separable data with slack variables
and the `C` parameter, walk through the kernel trick that
generalises everything from inner products to arbitrary
non-linear feature spaces, implement a from-scratch linear SVM
by subgradient descent on hinge loss, compare with scikit-learn's
`LinearSVC` and `SVC(kernel="rbf")` on the same moons dataset
the rest of the supervised track has been using, and finish with
the places SVMs are still the right answer in 2026.

---

## The maximum-margin idea

Consider binary classification with labels in `{-1, +1}` and
data linearly separable by some hyperplane `w · x + b = 0`. If
the data is separable then *many* hyperplanes split it. Which
one should we pick?

The SVM answer: pick the one with the largest **margin** — the
biggest gap between the hyperplane and the nearest training
points on either side. A wide-margin classifier has more
"breathing room" around the decision boundary; small
perturbations of the data are less likely to flip a prediction.
Formally, of all hyperplanes that classify the training set
correctly, the maximum-margin one is the most robust to noise
and the best generaliser (this is Vapnik's structural-risk
argument).

For a hyperplane `w · x + b = 0`, the distance from any point
`x_i` to the plane is `|w · x_i + b| / ‖w‖`. If we normalise so
that the closest training points lie on `w · x + b = ±1`, the
margin width becomes:

```
margin = 2 / ‖w‖
```

Maximising `2 / ‖w‖` is the same as minimising `‖w‖²`. So the
hard-margin SVM is the constrained optimisation:

```
minimise   ½ ‖w‖²
subject to y_i (w · x_i + b) ≥ 1   for every training point i
```

A convex quadratic objective with linear constraints — solvable
exactly by any quadratic-programming solver. The optimal `w`
defines the maximum-margin hyperplane.

---

## Soft margins and the C parameter

Real datasets are not linearly separable. Even when they are,
the maximum-margin solution can be brittle: one mislabelled
outlier near the boundary will dominate the entire fit. The
**soft-margin SVM** (Cortes & Vapnik, 1995) loosens the
constraint with slack variables `ξ_i ≥ 0`:

```
minimise   ½ ‖w‖² + C · Σ ξ_i
subject to y_i (w · x_i + b) ≥ 1 - ξ_i,   ξ_i ≥ 0
```

The slack `ξ_i` is the amount by which point `i` violates the
margin constraint. `ξ_i = 0` means the point is correctly
classified and outside the margin. `0 < ξ_i < 1` means it is
inside the margin but on the correct side of the hyperplane.
`ξ_i ≥ 1` means it is on the wrong side — misclassified.

The hyperparameter `C` controls the trade-off:

- **Large `C`** (e.g. 100, 1000) penalises violations heavily.
  The solver prioritises classifying every point correctly, even
  at the cost of a narrow margin. Risk: overfitting to noisy
  training points.
- **Small `C`** (e.g. 0.01, 0.1) tolerates violations. The
  solver prioritises a wide margin and accepts that some points
  will be misclassified or inside the margin. Risk: underfitting.

`C` is the SVM's most important regulariser, the analogue of
`λ` in ridge regression or `α` in Lasso. It is almost always
chosen by cross-validation.

A useful reformulation: the soft-margin objective is exactly
equivalent to minimising **hinge loss** with L2 regularisation:

```
minimise   (1 / (2 C)) ‖w‖² + Σ max(0, 1 - y_i (w · x_i + b))
```

The hinge loss `max(0, 1 - y · ŷ)` is zero when the point is
correctly classified with margin, linear in the violation
otherwise. This view makes SVMs a member of the same family as
logistic regression — both are linear classifiers fit by a
convex loss with an L2 penalty — they just use *different
losses* (hinge vs log loss). That equivalence is the bridge to
training SVMs by subgradient descent on big data.

---

## The dual and support vectors

The soft-margin primal can be solved directly, but the **dual
problem** is where SVM's elegance shows. Introducing Lagrange
multipliers `α_i ≥ 0` for each constraint and eliminating `w`
and `b` gives:

```
maximise   Σ α_i  -  ½ Σ_{i,j} α_i α_j y_i y_j (x_i · x_j)
subject to 0 ≤ α_i ≤ C,   Σ α_i y_i = 0
```

Two important properties:

**The objective only depends on inner products `x_i · x_j`.**
Once those are computed (the Gram matrix), the SVM does not look
at the raw `x` values again. This is the entry point for the
kernel trick.

**Most `α_i` end up exactly zero.** Only the training points on
or inside the margin have non-zero `α_i` — and those are exactly
the **support vectors**. The fitted model is:

```
w = Σ α_i y_i x_i        (sum is only over support vectors)
prediction(x) = sign(w · x + b)
              = sign(Σ_{i ∈ SV} α_i y_i (x_i · x) + b)
```

The decision function is a weighted sum over support vectors,
each contributing through its inner product with the query
point. On most problems the support vectors are a small fraction
of the training set — sparsity that makes SVMs memory-efficient
to deploy even when the training set was large.

---

## The kernel trick

Look at the dual one more time. The training data appears only
through the inner products `x_i · x_j`. The prediction step only
uses inner products `x_i · x`. *Nowhere* do we need the raw
coordinates of `x`.

So replace the inner product with a function `K(x_i, x_j)`. As
long as `K` is a valid kernel (corresponds to an inner product
in *some* feature space `ϕ`, i.e. `K(a, b) = ϕ(a) · ϕ(b)`), the
algebra still works. The classifier is now implicitly fitting a
hyperplane in the (possibly infinite-dimensional) feature space
`ϕ` — without ever computing `ϕ(x)`. This is the kernel trick,
and it is one of the cleverest constructions in classical ML.

Common kernels:

**Linear:** `K(a, b) = a · b`. No transformation. Use for
high-dimensional sparse data (text bag-of-words, gene
expression) where a linear boundary is already enough.

**Polynomial:** `K(a, b) = (γ · a · b + r)^d`. Implicit feature
space of all monomials up to degree `d`. Useful when you suspect
the boundary is a polynomial of low degree.

**Radial Basis Function (RBF / Gaussian):** `K(a, b) = exp(-γ ‖a
− b‖²)`. Infinite-dimensional implicit feature space. The
universal default for non-linear SVMs. `γ` controls how "local"
the kernel is — small `γ` means broad influence (smoother
boundaries), large `γ` means each support vector influences only
its immediate neighbourhood (wiggly boundaries, prone to
overfit).

**Sigmoid:** `K(a, b) = tanh(γ · a · b + r)`. Historically used
to approximate a 2-layer neural net; rarely the right choice
today.

For unstructured tabular data with non-linear interactions, RBF
with cross-validated `C` and `γ` is the SVM workhorse. For text
classification with TF-IDF features (millions of dimensions,
nearly always linearly separable), a linear kernel is faster
and just as accurate.

---

## A worked example

The companion script reuses the two-moons dataset from the
preceding articles (Parts 5, 1, and 2 of this series) — 400
training + 100 test, same seed — and walks an SVM through it
three ways.

```
DEMO 1 --- Linear SVM from scratch on the moons dataset
  Training set : 400 examples, 2 features
  Test set     : 100 examples
  Loss         : hinge + L2,  optimiser: subgradient descent
  Test accuracy : 0.900
  Support vectors (training points inside or on margin) : 143
```

```
DEMO 2 --- Same data, scikit-learn LinearSVC
  Test accuracy : 0.900
  Agreement with from-scratch model on test set: 100/100 predictions identical
```

```
DEMO 3 --- Non-linear SVM with RBF kernel (sklearn SVC)
       C  gamma   train_acc   test_acc   #support_vectors
  ------  -----   ---------   --------   ----------------
     0.1   auto       0.885      0.930                206
     1.0   auto       0.925      0.980                118
      10   auto       0.927      0.980                 83
     100   auto       0.932      0.970                 71
    1000   auto       0.932      0.980                 70
```

Three things to pull out.

**Linear is not enough.** The from-scratch linear SVM and
sklearn's `LinearSVC` both top out at `0.900` test accuracy on
the moons dataset, because no straight line in 2D can separate
two interleaving half-moons. This is a `d = 2` problem with a
curved boundary — exactly where the kernel trick earns its keep.

**RBF beats both forest and boosting on this dataset.** With
`C = 1.0` and `γ = "auto"`, the RBF SVM reaches `0.98` test
accuracy — *above* the random forest from Part 1 (`0.97`) and
the gradient-boosted ensemble from Part 2 at their sweet spots.
The "non-linear feature space" was infinite-dimensional in
theory and totally invisible in the code; the kernel function
did all the work.

**Capacity is controlled by `C` and `γ` together.** At
`C = 0.1` the soft-margin penalty is too weak and the model
underfits (`0.93` test) with a huge number of support vectors
(206 — over half the training set is unsettled). As `C` rises,
the SVM commits harder to the training data: the number of
support vectors drops to ~70, the train accuracy edges up, and
the test accuracy hovers around `0.97–0.98` — RBF SVMs are
remarkably stable across the useful range of `C` on this kind
of clean two-class problem.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The SVM complexity story is the reason it has been displaced on
big data:

**Linear SVM training is fast.** With LIBLINEAR or hinge-loss
subgradient methods, you can train a linear SVM on `n = 10⁶`,
`d = 10⁵` (sparse) in seconds. Linear SVM on bag-of-words text
classification is still a sensible default in 2026.

**Kernel SVM training scales badly.** The dual quadratic
programme is `O(n²)` in memory (the Gram matrix is `n × n`)
and roughly `O(n² · d)` to `O(n³ · d)` in compute, depending on
the solver. For `n = 10⁴` it is fine. For `n = 10⁵` it is
painful. For `n = 10⁶` it is impractical without
approximations (Nyström kernel approximation, random Fourier
features, or just switching model class).

**Prediction is `O(n_SV · d)`** — proportional to the number
of support vectors, not the full training set. On problems
where the support vectors are a small fraction of training, the
fitted SVM is fast at inference even when training was slow.

The pragmatic ranges: kernel SVMs are the right answer for
`n ≲ 10⁵` and `d` reasonable. Beyond that, switch to gradient
boosting or to a linear SVM (LIBLINEAR / SGDClassifier with
hinge loss) over learned features.

---

## Real-world ML and AI connections

For a decade SVMs were the dominant supervised classifier
outside of the deep-learning research community. Their footprint
in 2026 is smaller but still real:

**Pre-deep-learning computer vision.** The classic
pipeline — HOG features plus a linear SVM — was the
state-of-the-art for pedestrian detection (Dalal & Triggs,
2005), the basis of the original DPM object detectors, and the
default classifier on the MNIST and CIFAR benchmarks before
CNNs took over. Many production object-detection systems still
ship the SVM head.

**Text classification.** SVMs on TF-IDF features dominated
text classification through the 1990s and 2000s. Joachims's
1998 paper *Text Categorization with Support Vector Machines*
established this as the default, and it held for over a decade.
Modern text systems mostly use transformer encoders, but a
linear SVM is still a strong baseline and very common in
high-volume / low-latency stacks.

**Bioinformatics.** Protein classification, gene expression
analysis, and structure prediction problems were heavy SVM
users for the 2000s and into the 2010s. The combination of
small `n`, large `d`, and the kernel trick (string kernels,
spectrum kernels, graph kernels for chemistry) made SVM a
natural fit.

**One-class SVMs for anomaly detection.** A one-class SVM
learns a decision boundary that encloses most of the training
data; anything outside is flagged anomalous. Still in
production fraud-detection and intrusion-detection systems.

**Calibrated decision boundaries in regulated industries.**
Where a model needs to be both well-understood and
production-safe (medical imaging triage, certain credit
applications), an RBF SVM with a Platt-scaled probability layer
is a defensible, low-surprise choice.

**Tabular ML on small to medium data.** For `n ≲ 10⁴` and
non-linear boundaries, an RBF SVM with cross-validated `C`,
`γ` is competitive with gradient boosting and sometimes wins —
especially when the data is too small to support a 1000-tree
GBM without overfitting.

The pattern: SVMs are no longer the default *anywhere*, but
they remain a strong, surprisingly-competitive baseline in
exactly the small-to-medium tabular niche where calibration,
sparsity, and well-understood theory matter.

---

## When NOT to use SVMs

The places SVMs are dominated by other methods:

**Very large datasets.** `n > 10⁵` and a kernel SVM
needs special treatment (Nyström approximation, random Fourier
features, mini-batch hinge-loss SGD). Easier to switch to a
linear SVM over learned features or to a gradient-boosting
ensemble.

**When you need probabilities.** Raw SVM outputs are distances
from the hyperplane, not probabilities. Platt scaling (fitting
a logistic regression on the SVM scores using cross-validation)
recovers probabilities but adds training cost and is still
worse-calibrated than a logistic-regression model trained
directly with log loss. If probabilities matter, train logistic
regression instead.

**Multi-class with many classes.** SVMs are natively binary.
Multi-class is implemented as one-vs-rest or one-vs-one, both
of which scale poorly past a few dozen classes. Softmax
classifiers (logistic regression, neural nets) handle
multi-class natively.

**When you need feature importance.** Kernel SVMs do not give
per-feature importance scores — the decision function is a
sum over support vectors in an implicit feature space.
Permutation importance works but is expensive. Linear SVMs
have weight magnitudes, which is the same as logistic
regression's coefficient magnitudes.

**High-cardinality categoricals.** SVMs require a
fixed-dimensional numeric feature vector. Categorical features
need one-hot or learned-embedding encoding upstream — work
that tree-based methods skip.

**Image / audio / sequence data.** Use the appropriate
deep-learning architecture (CNN / transformer / RNN). SVMs on
raw pixels are a non-starter; SVMs over *features extracted by
a CNN* still occasionally appear as classification heads but
even there a softmax is usually fine.

---

## What comes next

This is the final article in the **Supervised Learning** track
(introductory + advanced). The five basic models — Linear
Regression, Logistic Regression, Naive Bayes, K-Nearest
Neighbours, Decision Trees — plus the three advanced — Random
Forests, Gradient Boosting, Support Vector Machines — give you
the standard supervised toolkit. On any new labelled-data
problem the question to ask is *which of these eight is the
right starting point*, and the rest of the rolling improvement
is incremental tuning, feature engineering, and (if the data is
unstructured) reaching for deep learning.

Next is the **Unsupervised Learning** track, beginning with
**K-Means Clustering** — the simplest and most-used clustering
algorithm, and the entry point to the wider unsupervised
toolkit (hierarchical clustering, PCA, t-SNE, UMAP, NMF,
spectral methods). The shift from supervised to unsupervised is
the shift from "learn the labelling function" to "learn the
structure of the data itself", and it brings a different
mathematical toolkit with it.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**svm.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/02-advanced-supervised-learning/03-support-vector-machines/svm.py)

Run it with:

```bash
pip install numpy scikit-learn
python svm.py
```

It needs `numpy` and `scikit-learn`. The script trains a linear
SVM from scratch using subgradient descent on hinge loss with
L2 regularisation, fits the moons dataset, compares against
scikit-learn's `LinearSVC` (predictions agree on every test
example), then trains a kernelised `SVC` with the RBF kernel
and sweeps `C` to show the bias-variance trade-off — the
linear model tops out at `0.90` because moons is not linearly
separable, while the RBF kernel lifts accuracy to `0.98`
without ever explicitly computing the non-linear feature space.
The headline insight worth pinning to the wall: **find the
widest gap between the classes, then replace inner products
with a kernel function to fit non-linear boundaries for free**.

---

*This is Part 3 of the Advanced Supervised Learning track in the Algorithms in Python series, and the final article of the Supervised Learning track overall. The companion script `svm.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 2 of this track covered Gradient Boosting. The next article opens the Unsupervised Learning track with K-Means Clustering — the simplest, most-used, and most-imitated unsupervised algorithm.*
