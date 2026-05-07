# Logistic Regression — Drawing a Curve That Outputs Probabilities

### *Algorithms in Python --- Supervised Learning, Part 2*

---

In Part 1 we fit a straight line to data and called it linear
regression. The output was a number — a predicted house price, a
predicted stock movement, anything that lives on a continuous
scale. Today we change the question. *Will this customer churn?
Is this email spam? Will this loan default?* The answer is no
longer a number. It is a category. And the moment the target
becomes a category, the straight line breaks.

Logistic regression is the algorithm that fixes it. It looks
almost identical to linear regression on the surface — the same
weighted sum of features, the same training-data flow — but it
wraps that sum in a function that squashes any real number into
the interval (0, 1) and interprets the result as a *probability*.
With one small change the algorithm goes from predicting numbers
to predicting class membership, and that is the most common
prediction problem in production ML.

This article builds logistic regression from the ground up,
following the same pattern as Linear Regression. We will derive
the sigmoid function, derive the loss that goes with it (and why
Mean Squared Error is the wrong choice here), implement the
algorithm from scratch with a gradient-descent loop, then run the
scikit-learn version on the same data. By the end you will see
why the same shape — *linear combination → squashing function →
probability* — shows up at the output layer of every neural
network classifier in the field.

---

## When the straight line breaks

Imagine you want to predict whether a student passes an exam from
their hours studied. You collect 100 students, you have a 0/1
label per student, and you fit a linear regression to it. The
line will happily go below 0 and above 1 for extreme inputs —
predicting a "negative probability" of passing, or a probability
greater than 1. Both are nonsense.

Worse, the linear model treats *any* increase in the input as the
same magnitude of change in the output. In the binary world, that
is wrong. Going from 0.49 to 0.51 (just over the decision
threshold) is meaningful; going from 0.91 to 0.93 (already
confident) is not. The probability of passing should *plateau* as
the input gets very large or very small, with most of the
sensitivity concentrated near the decision boundary.

You need a curve, not a line. Specifically you need a curve that:

- Outputs values bounded in (0, 1) — never below 0, never above 1
- Is steepest in the middle, asymptotes at the extremes
- Is smooth and differentiable, so gradient descent can train it

The function that does all three is the **sigmoid**.

---

## The sigmoid function

The sigmoid (or "logistic") function is:

```
σ(z) = 1 / (1 + e^(-z))
```

Plug in `z = 0` and you get `1/(1+1) = 0.5`. Plug in large
positive `z` and `e^(-z)` goes to 0, so `σ(z)` approaches 1.
Plug in large negative `z` and `e^(-z)` blows up, so `σ(z)`
approaches 0. The whole real line gets mapped into (0, 1), with
50% probability sitting at z = 0 and the curve steepest at that
midpoint.

```
       σ(z)
        1 │              ╭──────────
          │           ╭───
        ½ │       ╭───
          │   ╭───
        0 │───╯
          └─────────────────────  z
            -5      0       +5
```

The shape is exactly what we wanted: bounded, smooth, S-shaped.
And the function is differentiable everywhere with the lovely
property `σ'(z) = σ(z)(1 - σ(z))` — its derivative is expressed
in terms of itself, which makes the gradient calculation later
remarkably clean.

The model is then *linear* inside the sigmoid:

```
P(y = 1 | x) = σ(b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ)
             = σ(θᵀx)
```

A weighted sum of features (the same shape as
[Linear Regression](https://medium.com/@grahamjroy/linear-regression-23125eaefd29))
followed by a squashing function. That two-step pattern —
*linear combination, then non-linear squash* — is the building
block of every neural network: a fully-connected layer is exactly
this, repeated and stacked.

---

## From scores to probabilities (and back)

Logistic regression has two equivalent views, and you will see
both in textbooks.

**Probability view.** The model predicts `P(y = 1 | x)`
directly, as we just wrote.

**Log-odds view.** Take the inverse of the sigmoid. Define the
*odds* of class 1 as `p / (1 - p)`. Take the log:

```
log(p / (1 - p)) = θᵀx
```

The left side is the **logit**, or **log-odds**. So logistic
regression is "linear regression on the log-odds scale." The
model is linear in the *log-odds* of the positive class, even
though it is non-linear in the probability itself. That is why
logistic regression is part of the *generalised linear model*
family and shares much of its theory with linear regression.

The practical consequence: a one-unit increase in feature `x_i`
changes the log-odds by `b_i`, regardless of where you currently
are on the curve. Equivalently, it multiplies the odds by
`e^(b_i)`. *"This feature changes the odds of churn by a factor
of 1.3"* — that is the kind of statement logistic regression
makes natural.

---

## The cost function: why MSE is wrong here

In linear regression, the cost was Mean Squared Error and the
optimal parameters fell out of a closed-form Normal Equation.
Neither carries over to logistic regression cleanly.

Two reasons MSE is wrong for binary classification:

**It does not match the model.** Squaring a probability error
penalises being wrong about an unsure prediction (0.51 → 0.50)
much less harshly than being wrong about a confident one
(0.99 → 0.50). For classification we want the opposite:
confident-but-wrong predictions should be punished severely,
because they signal the model has misunderstood something
fundamental.

**It produces a non-convex loss surface.** The combination of
sigmoid + MSE is not convex in the parameters, so gradient
descent gets stuck in local minima.

The right loss is **log loss** (also called
**binary cross-entropy**, or **negative log-likelihood**):

```
L(θ) = -(1/n) Σ [yᵢ log(p̂ᵢ) + (1 - yᵢ) log(1 - p̂ᵢ)]
```

For each example, you take the log of the predicted probability
of the *correct* class, sum across examples, negate, and divide
by `n`. If `y = 1` you want `p̂` to be close to 1 (so `log p̂`
is close to 0); if `y = 0` you want `1 - p̂` close to 1
(so `log(1 - p̂)` is close to 0). Either way, getting a confident
prediction wrong sends the loss to infinity.

Two important properties:

- It is **convex** in the parameters when paired with the sigmoid
  link function, so gradient descent finds the global minimum.
- It is the **maximum-likelihood loss** for the Bernoulli
  distribution, which is the principled probabilistic
  justification for using it.

The gradient of log loss with respect to `θ` works out to a
beautifully simple expression:

```
∂L/∂θ = (1/n) Xᵀ (p̂ - y)
```

Predicted probabilities minus actual labels, multiplied by the
features and averaged. That is structurally identical to the
gradient of *linear* regression's MSE — the only difference is
that `p̂` here goes through a sigmoid. No wonder the same
gradient-descent code works for both.

---

## Training: gradient descent, not the Normal Equation

There is no closed-form solution for logistic regression. The
sigmoid in the loss makes `Xᵀ X` not the right object anymore.
We have to iterate.

The training loop is:

```
1. Initialise θ to zeros (or small random values)
2. Compute predictions:   p̂ = σ(Xθ)
3. Compute the gradient:  g = (1/n) Xᵀ(p̂ - y)
4. Update parameters:     θ ← θ - α·g
5. Repeat until the loss stops decreasing
```

Where `α` is the learning rate. In practice you also add a
regularisation term to prevent overfitting (we will return to
this in a moment). This loop is exactly the gradient descent we
covered in
[Maths Behind ML Part 4](https://github.com/grahamroy) — same
algorithm, applied to a different loss.

The companion script implements this loop from scratch in numpy
on a synthetic 2D classification dataset and reports the
training trajectory:

```
Logistic regression --- gradient descent on 2D toy data

  Epoch    0   loss=0.6931   accuracy=0.500
  Epoch  500   loss=0.0884   accuracy=0.975
  Epoch 1000   loss=0.0816   accuracy=0.975
  Epoch 1500   loss=0.0797   accuracy=0.975
  Epoch 2000   loss=0.0789   accuracy=0.975

Final parameters:
  Intercept (b0): -0.0449
  Weights:        [+2.9438, -3.0862]

Test set metrics:
  Accuracy:  0.960
  Precision: 0.942
  Recall:    0.980
  F1:        0.961
  ROC-AUC:   0.998
```

Loss starts at `log(2) ≈ 0.693` (random guessing on a balanced
binary problem) and drops fast — by epoch 500 the model is
already at 97.5% training accuracy and the loss has fallen by an
order of magnitude. The remaining 1500 epochs sharpen the
parameters by tiny amounts. On the held-out test set the model
hits 96% accuracy with a ROC-AUC of 0.998 — almost perfect
ranking of positives above negatives.

---

## Comparing to scikit-learn

Scikit-learn's `LogisticRegression` does the same thing under
the hood, with three production-grade tweaks: it adds an L2
regularisation penalty by default, it uses a smarter optimiser
(LBFGS or SAGA, not vanilla gradient descent), and it handles
multinomial classification automatically.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
```

`.predict()` returns hard class labels (after applying the 0.5
threshold); `.predict_proba()` returns calibrated probabilities,
which is what you want whenever the downstream system needs to
trade off precision and recall. *Always prefer probabilities to
class labels when the consumer is anything but a final answer.*

---

## The decision threshold and what it costs you

`predict()` thresholds at 0.5 by default. That is rarely the
right choice in production.

If your problem is fraud detection, missing a fraudulent
transaction is far costlier than flagging a legitimate one for
review. Lower the threshold to 0.2 and you catch more fraud at
the cost of more false positives. If your problem is rare-disease
screening, the opposite asymmetry: a false negative might be
fatal, a false positive is just an extra test.

The right threshold depends on the relative cost of the two
error types. The right *tool* for picking it is the **ROC
curve** (true positive rate vs false positive rate as you sweep
the threshold) or the **precision-recall curve** (more
informative for imbalanced classes). The companion script plots
both for the toy dataset and reports the **AUC** — area under
the ROC curve — which is the threshold-free measure of how well
the model ranks positives above negatives.

```
Confusion matrix at threshold = 0.5:
                     Predicted 0   Predicted 1
  Actual 0 (50):           47            3
  Actual 1 (50):            1           49

ROC-AUC: 0.998
```

A model with AUC = 0.5 is no better than coin-flipping; AUC = 1
is perfect ranking. Any production deployment monitors AUC, and
choosing the threshold is a separate, business-driven decision
made on top.

---

## Multinomial: the softmax generalisation

Two classes is binary classification. Three or more classes is
**multinomial** classification, and the natural generalisation
of the sigmoid is the **softmax**:

```
P(y = k | x) = exp(θ_kᵀ x) / Σ_j exp(θ_jᵀ x)
```

One weight vector per class, exponentiated and normalised so the
probabilities sum to 1. This is exactly what the final layer of
every neural network classifier does: a linear layer per class,
followed by a softmax. From MNIST digit classification through
ImageNet up to GPT-style language models predicting the next
token, the output head is always *some logits, then softmax* —
which is binary logistic regression generalised to thousands or
hundreds of thousands of classes.

The cost function generalises too: **categorical cross-entropy**
is binary log loss extended to K classes. Every modern
classifier in PyTorch, TensorFlow, and JAX is trained on
cross-entropy + softmax + some clever optimiser. Logistic
regression is the K=2 case.

---

## Regularisation: the L2 default

A logistic regression with no penalty can drift toward
arbitrarily large weights when the classes are perfectly
separable, which makes the model fragile and the probabilities
miscalibrated. The fix is **L2 regularisation** (also called
**ridge** or **weight decay**): add `λ · ||θ||²` to the loss
and the optimiser is pulled gently toward small weights.

```
L_reg(θ) = L(θ) + λ · ||θ||²
```

`λ` is a hyperparameter; scikit-learn parameterises it as the
inverse `C = 1/λ`, where smaller `C` means stronger
regularisation. The default `C = 1.0` is a sensible middle
ground. **L1 regularisation** (also called **Lasso**) replaces
the squared norm with an absolute-value norm, which has the
property of driving some weights *exactly* to zero — useful for
feature selection on high-dimensional data.

The same regularisation theory carries over to neural networks
(weight decay in optimiser configs), to recommender systems
(matrix factorisation with L2 priors), and to most of supervised
ML. Logistic regression is where the principle is easiest to see.

---

## Big-O and cost summary

[[BIG-O TABLE IMAGE]]

Three notes. **Training is linear in the number of examples and
features per epoch**, with the constant factor dominated by the
matrix multiplication `Xθ`. The number of epochs is usually
small (10² to 10⁴) so the total cost stays manageable up to
hundreds of millions of examples. **Prediction is essentially
free** — one matrix-vector product and one sigmoid per example.
**Memory is dominated by the data matrix**, not the model; the
parameter vector is just `n_features + 1` floats. Sparse
features, as covered in
[Foundations Part 11 — Sparse Matrices](https://medium.com/@grahamjroy/sparse-matrices-when-most-of-your-data-is-zero-85cebc669d78),
let scikit-learn's `LogisticRegression` train on millions of
TF-IDF features without breaking a sweat.

---

## Real-world ML and AI connections

Logistic regression is one of the most-deployed algorithms in
production ML. Once you start looking, it is everywhere.

**Credit scoring.** FICO and most major credit-scoring systems
are logistic regressions on tens to a few hundred features. The
interpretability is the point — regulators want each weight to
be defensible, and logistic regression gives you a coefficient
per feature with a confidence interval and a p-value. No deep
network can match that audit trail.

**Medical risk models.** Cardiovascular risk calculators,
diabetes-progression scores, sepsis-prediction in ICUs — almost
all logistic regressions, often updated quarterly as new patient
data accumulates. The same interpretability constraint as credit
scoring: clinicians need to know why a patient is flagged.

**Click-through-rate (CTR) prediction at ad networks.** Google,
Meta, and Bing all started ad-CTR with logistic regression on
billions of features (mostly hashed feature crosses). The first
neural-network ad-prediction systems were "logistic regression
plus a small embedding layer," and even today the linear
component remains a critical baseline.

**Spam, abuse, and fraud filters.** The first generation of
Gmail's spam filter was a logistic regression on millions of
text features (n-gram hashes, sender features). Modern
production fraud filters at banks and payment networks still
use logistic regression as the calibration layer on top of more
complex feature pipelines.

**The output layer of every neural classifier.** ResNet, ViT,
BERT classification head, ChatGPT's content filter, every
sentiment classifier ever — all end in a linear layer followed
by a sigmoid (binary) or softmax (multi-class). When you train
a neural network for classification, you are training a deep
non-linear feature extractor that feeds into a logistic
regression at the top.

**LLM token sampling.** GPT-style models predict the next token
by computing logits over the vocabulary and softmaxing.
Temperature, top-k, and top-p sampling are all manipulations of
this softmax distribution. The math and the training loss
(cross-entropy) are the multi-class generalisation of logistic
regression.

**A/B test analysis.** When you run an A/B test and want to know
*"is the conversion rate different between A and B?"*, the
underlying statistical model is a logistic regression with a
single binary feature for the variant. Most causal-inference
libraries (DoWhy, EconML, CausalML) lean on logistic regression
internally for binary outcome modelling.

---

## When NOT to use logistic regression

Logistic regression is excellent at what it does, but the
boundaries are real:

**When the decision boundary is highly non-linear.** Logistic
regression can only draw a *hyperplane* in feature space
(possibly with hand-crafted interaction features). If the true
boundary is curved, intersecting, or fractal, you need decision
trees, random forests, or neural networks. Polynomial features
extend logistic regression's reach but only modestly.

**When you need to capture complex feature interactions
automatically.** Tree-based models (next two articles) discover
interactions for free; logistic regression requires you to
engineer them as features. For most tabular ML problems with
non-trivial interactions, gradient-boosted trees beat logistic
regression handily.

**When calibration matters more than ranking.** Logistic
regression with L2 regularisation is well-calibrated by
construction. Neural networks are often *not* well-calibrated
out of the box — their probabilities tend to be over-confident
— and need post-hoc temperature scaling to recover usable
probabilities. If you need probabilities you can act on,
logistic regression is the safer baseline.

**When you have a tiny number of examples and a huge number of
features.** Logistic regression overfits in this regime unless
you use heavy L1 regularisation. Linear support vector machines
or naïve Bayes (Part 3, next article) can be more robust at the
extreme.

**When the classes are perfectly separable.** Without
regularisation, the optimiser will push weights to infinity to
get the loss to exactly zero, producing miscalibrated
probabilities. Always train with at least mild L2.

---

## What comes next

Part 3 of the supervised-learning track is **Naive Bayes** — a
probabilistic classifier built on a deliberately simplistic
independence assumption that, surprisingly, works well in
practice. It will give us a chance to revisit Bayes' rule (which
underpins much of the Maths Behind ML series) and to see how
text classification problems get solved when the feature space is
sparse and high-dimensional — exactly the regime where logistic
regression and Naive Bayes both shine.

After Naive Bayes comes **K-Nearest Neighbours**, the simplest
distance-based classifier and a callback to the KD-trees from
[Foundations Part 8 — Trees](https://medium.com/@grahamjroy/trees-hierarchical-structure-for-decisions-search-and-database-indexes-64767b20394f).
Then **Decision Trees** (the entry point to the random-forest /
gradient-boosting family that wins on most tabular ML
benchmarks).

---

## The complete code

The full script is on GitHub — grab it and run it:

[**logistic_regression.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/01-supervised-learning/02-logistic-regression/logistic_regression.py)

Run it with:

```bash
pip install numpy matplotlib scikit-learn
python logistic_regression.py
```

It needs `numpy`, `scikit-learn`, and `matplotlib`. The script
implements logistic regression from scratch with a
gradient-descent loop, trains it on a 2D synthetic two-class
dataset and watches the loss drop from 0.69 to 0.08 over 2000
epochs, compares against scikit-learn's `LogisticRegression`
which agrees on the test-set metrics to three decimal places
(both hit 96% accuracy and ROC-AUC 0.998), and saves a
four-panel visualisation showing the sigmoid curve, the training
loss trajectory, the decision boundary, and the ROC curve. The headline insight worth pinning to the wall:
**logistic regression is linear regression with a sigmoid on
top, trained with gradient descent on log loss — and the same
shape sits at the output of every neural network classifier
ever built**.

---

*This is Part 2 of the Algorithms in Python series, Supervised Learning track. The companion script `logistic_regression.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 1](https://medium.com/@grahamjroy/linear-regression-23125eaefd29) covered linear regression. Part 3 will look at Naive Bayes — a probabilistic classifier whose independence assumption is wrong but whose performance often is not.*
