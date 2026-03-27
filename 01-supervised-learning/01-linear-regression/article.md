# Linear Regression — The Algorithm That Draws a Line Through Your Data

### *Algorithms in Python — Supervised Learning, Part 1*

---

Every machine learning journey starts in the same place: a scatter of data points on a
chart, and a question. *Is there a pattern here?* If someone told you that house prices
tend to rise with square footage, you would instinctively reach for a ruler and try to
draw a straight line through the cloud. That instinct — fitting a line to data — is
linear regression. It is the simplest algorithm in machine learning, and also one of the
most important.

Linear regression does not just draw the line. It draws the *best* line — the one that
minimises the total error between its predictions and the actual data. The mathematics
behind it are clean and elegant, and the Python implementation takes just a few lines of
code. But there is more going on beneath the surface than most introductions reveal, and
understanding what "best" actually means will change how you think about every algorithm
that comes after it.

This article builds linear regression from the ground up. We will start with the
intuition, derive the mathematics, implement it from scratch in Python, then do it
again with scikit-learn. By the end, you will understand not just *how* to use linear
regression, but *why* it works.

---

## The idea: one equation, two unknowns

At its core, linear regression fits a straight line to a set of data points. The
equation for that line is:

> ŷ = b₀ + b₁x

where **x** is the input (the feature), **ŷ** is the prediction, **b₀** is the
y-intercept (where the line crosses the vertical axis), and **b₁** is the slope (how
much ŷ changes when x increases by one unit).

The job of the algorithm is to find the values of b₀ and b₁ that make the line fit the
data as closely as possible. "As closely as possible" needs a precise definition, and
that is where the cost function comes in.

---

## The cost function: what "best" means

Given a set of data points, any line will make errors — it will predict values that are
above some points and below others. The question is which line makes the *smallest*
total error.

Linear regression uses the **Mean Squared Error** (MSE) as its cost function:

> MSE = (1/n) × Σ (yᵢ − ŷᵢ)²

For each data point, take the difference between the actual value y and the predicted
value ŷ, square it, and average across all points. Squaring does two things: it makes
all errors positive (so negatives do not cancel out positives), and it penalises large
errors more heavily than small ones.

The best line is the one that minimises this number. Every other line — tilted a little
steeper, shifted a little higher — produces a larger MSE.

---

## The Normal Equation: solving it in one step

One of the elegant things about linear regression is that the optimal parameters can be
found in a single calculation. There is no iteration, no "learning rate", no gradient
descent (we will meet those later in this series). The solution is a closed-form
equation called the **Normal Equation**:

> θ = (Xᵀ X)⁻¹ Xᵀ y

where **X** is the matrix of input features (with a column of ones prepended for the
intercept), **y** is the vector of target values, and **θ** is the vector of optimal
parameters [b₀, b₁].

What this equation says: project the target values onto the column space of X, and find
the parameters that make the projection exact. If that sounds abstract, the code makes
it concrete.

---

## Step 1: Generate synthetic data

We start by creating data with a *known* linear relationship. This is a powerful
learning technique: because we choose the true parameters ourselves, we can check
whether the algorithm recovers them.

```python
import numpy as np

np.random.seed(42)
X = 2 * np.random.rand(100, 1)            # 100 values between 0 and 2
y = 3 + 4 * X + np.random.randn(100, 1)   # y = 3 + 4x + noise
```

The true intercept is 3, the true slope is 4, and we have added Gaussian noise to
simulate real-world messiness. The algorithm's job is to look at these 100 noisy points
and figure out that the underlying relationship is approximately y = 3 + 4x.

---

## Step 2: Split into training and test sets

This is the single most important habit in machine learning. We never evaluate a model
on the same data we trained it on. That would be like marking your own exam — you would
always score well, but it would not tell you whether you actually understand the
material.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

We reserve 20% of the data (20 points) as a test set that the model never sees during
training. When we evaluate later, we use these held-out points to measure how well the
model generalises to new data.

---

## Step 3: Implement it from scratch

Here is the Normal Equation in Python. We add a column of ones to X (for the intercept
term), then apply the formula directly:

```python
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]  # add bias column
theta = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

intercept = theta[0][0]
slope = theta[1][0]
```

That is the entire algorithm — three lines of linear algebra. The `@` operator is
matrix multiplication, `.T` is transpose, and `np.linalg.inv` computes the inverse.
The result is a vector θ containing the intercept and slope.

With the seed and data above, this gives:

```
Intercept (b₀): 3.1242
Slope (b₁):     3.8511
```

Close to the true values of 3 and 4. The small discrepancy is the noise — with only
80 training points and random variation, the estimate will never be exact. But it is
remarkably close.

---

## Step 4: Do it with scikit-learn

Scikit-learn wraps the same mathematics in a clean API. The result should be identical
(or nearly so — scikit-learn uses numerical optimisations for stability):

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print(model.intercept_[0])   # b₀
print(model.coef_[0][0])     # b₁
```

Two lines to create and train the model. The `fit()` method does all the work —
computing the optimal intercept and slope from the training data. The parameters are
stored in `model.intercept_` and `model.coef_`.

---

## Step 5: Evaluate on the test set

Now the critical question: how well does the model perform on data it has never seen?

```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

**Mean Squared Error (MSE)** tells you the average squared prediction error. Lower is
better, but the number is hard to interpret on its own because it is in squared units.

**Root Mean Squared Error (RMSE)** is the square root of MSE — it is in the same units
as y, making it easier to understand. An RMSE of 1.05 means the model's predictions
are off by about 1.05 units on average.

**R² Score** is the proportion of variance in y that the model explains. An R² of 0.83
means the model captures 83% of the variation in the data. The remaining 17% is noise
that a straight line cannot capture. A perfect model scores 1.0; a model that just
predicts the mean of y scores 0.0.

---

## Step 6: Predict on new data

The practical payoff — using the trained model to make predictions on inputs it has
never encountered:

```python
X_new = np.array([[0.5], [1.0], [1.5], [2.0]])
y_new_pred = model.predict(X_new)
```

For X = 1.0, the model predicts approximately 6.98. The true value (without noise) is
3 + 4(1.0) = 7.0. The model is right on target.

This is what a trained model is for: you give it new measurements, and it returns
predictions based on the pattern it learned from the training data.

---

## Step 7: Visualise the results

Two plots tell you everything you need to know about a linear regression:

**The fit plot** shows the regression line against the actual data points. Training
points and test points are plotted in different colours so you can visually check
whether the model generalises — if the test points sit near the line just as well as
the training points do, you are in good shape.

**The residual plot** shows prediction errors (y − ŷ) against predicted values. For
a well-behaved linear regression, residuals should scatter randomly around zero with
no visible pattern. If you see a curve, a funnel shape, or clusters, it means the
linear assumption is wrong and a more complex model may be needed.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Fit plot
axes[0].scatter(X_train, y_train, color='steelblue', alpha=0.6, label='Training data')
axes[0].scatter(X_test, y_test, color='coral', alpha=0.6, marker='s', label='Test data')
X_line = np.linspace(0, 2, 100).reshape(-1, 1)
axes[0].plot(X_line, model.predict(X_line), color='black', linewidth=2)
axes[0].set_title('Linear Regression — Fit')
axes[0].legend()

# Residual plot
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, color='steelblue', alpha=0.6)
axes[1].axhline(y=0, color='black', linestyle='--')
axes[1].set_title('Residual Plot — Test Set')
```

---

## What linear regression assumes (and when it breaks)

Linear regression is powerful but not universal. It makes several assumptions, and
knowing them tells you when to reach for a different tool:

**Linearity.** The relationship between x and y is a straight line. If the true
relationship is curved (exponential, polynomial, periodic), the model will
systematically miss.

**Independence.** Each data point is independent of the others. Time series data, where
each measurement depends on the previous one, often violates this.

**Homoscedasticity.** The spread of residuals is roughly constant across all values of
x. If errors fan out (larger variance for larger x), the model's confidence intervals
become unreliable.

**Normality of residuals.** The errors follow a roughly normal distribution. This
matters most for statistical inference (confidence intervals, p-values) rather than for
prediction accuracy.

When these assumptions hold, linear regression is hard to beat for its simplicity and
interpretability. When they break, the residual plot will usually show you.

---

## Where this fits in the bigger picture

Linear regression is the foundation of supervised learning. Nearly every algorithm you
will meet in this series is either an extension of it or a deliberate departure from it:

- **Logistic regression** (next article) replaces the straight line with a curve that
  outputs probabilities instead of continuous values — classification instead of
  regression.

- **Decision trees** abandon the idea of a single global line and instead split the
  data into regions, fitting a different value in each region.

- **Neural networks** stack many linear regressions together with nonlinear activation
  functions between them, allowing them to learn arbitrarily complex patterns.

- **Gradient descent** (which we covered in the maths series) is the general-purpose
  method for finding optimal parameters when the Normal Equation becomes too expensive
  to compute — which happens as soon as the number of features grows large.

Understanding linear regression deeply — the cost function, the train/test split, the
residual analysis — gives you a vocabulary and a set of tools that apply everywhere.
The algorithms change. The principles do not.

---

## The complete code

The full script is on GitHub: [**linear_regression.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/01-supervised-learning/01-linear-regression/linear_regression.py)

It runs both the from-scratch implementation and the scikit-learn version, prints all
metrics, and produces a two-panel visualisation saved as `linear_regression_results.png`.

Run it with:

```bash
pip install numpy matplotlib scikit-learn
python linear_regression.py
```

The full repository for this series is at [github.com/grahamroy/algorithms-in-python](https://github.com/grahamroy/algorithms-in-python).

---

## Summary

1. **Linear regression** fits the line ŷ = b₀ + b₁x to minimise the Mean Squared Error.

2. **The Normal Equation** θ = (XᵀX)⁻¹Xᵀy gives the exact optimal parameters in one
   step — no iteration required.

3. **Train/test split** is non-negotiable. Always evaluate on data the model has not
   seen during training.

4. **MSE, RMSE, and R²** are the standard metrics. MSE measures total error, RMSE puts
   it in interpretable units, and R² tells you the fraction of variance explained.

5. **The residual plot** is your diagnostic tool. Random scatter around zero means the
   model is well-specified. Patterns mean something is wrong.

6. **Linear regression is the starting point**, not the finish line. Every supervised
   learning algorithm in this series builds on or reacts against the ideas introduced
   here.

---

*Part of the series **Algorithms in Python**:*
*Supervised Learning — Part 1: Linear Regression (this article)*
*Supervised Learning — Part 2: [Logistic Regression]*

---

**Tags:** #MachineLearning #Python #LinearRegression #DataScience #ScikitLearn #SupervisedLearning #AI
