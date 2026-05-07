"""
logistic_regression.py --- companion code for "Logistic Regression"
(Supervised Learning, Part 2).

Three demos:
  1. From-scratch gradient descent on a synthetic 2D two-class dataset,
     reporting the loss trajectory and final parameters.
  2. Scikit-learn's LogisticRegression on the same data, with
     accuracy / precision / recall / F1 / ROC-AUC and a confusion matrix.
  3. A four-panel visualisation: the sigmoid curve, the training-loss
     trajectory, the decision boundary, and the ROC curve.

Dependencies: numpy, scikit-learn, matplotlib. Runs in well under a second.
"""

import sys

# Force UTF-8 stdout so Unicode characters print correctly on Windows
# consoles that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib

# Use a non-interactive backend so the script saves the plot and exits
# without trying to open a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)
from sklearn.model_selection import train_test_split


SEPARATOR = "=" * 64


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Synthetic 2D two-class dataset
# ---------------------------------------------------------------------------

def make_dataset(n=500, seed=42):
    """Two well-separated 2D Gaussian blobs, one per class."""
    rng = np.random.default_rng(seed)
    n_per_class = n // 2
    # Class 0: centred at (-1, +1)
    X0 = rng.normal(loc=[-1.0, 1.0], scale=0.8, size=(n_per_class, 2))
    # Class 1: centred at (+1, -1)
    X1 = rng.normal(loc=[1.0, -1.0], scale=0.8, size=(n_per_class, 2))
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).astype(np.int32)
    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Demo 1 --- From-scratch gradient descent
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def log_loss(y_true, p_pred, eps=1e-15):
    """Binary cross-entropy."""
    p = np.clip(p_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def fit_gradient_descent(X, y, lr=0.1, epochs=2000, verbose_every=None):
    """Train logistic regression by vanilla gradient descent."""
    n, d = X.shape
    # Add bias column
    X_b = np.c_[np.ones((n, 1)), X]
    theta = np.zeros(d + 1)
    history = []

    for epoch in range(epochs + 1):
        z = X_b @ theta
        p = sigmoid(z)
        loss = log_loss(y, p)
        history.append(loss)

        # Gradient: (1/n) X^T (p - y)
        grad = X_b.T @ (p - y) / n
        theta -= lr * grad

        if verbose_every and epoch % verbose_every == 0:
            preds = (p >= 0.5).astype(int)
            acc = (preds == y).mean()
            print(f"  Epoch {epoch:>4d}   loss={loss:.4f}   accuracy={acc:.3f}")

    return theta, history


def demo_from_scratch():
    banner("DEMO 1 --- Logistic regression from scratch")

    X, y = make_dataset(n=500, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training: {len(X_train)} examples, {X_train.shape[1]} features")
    print(f"Test:     {len(X_test)} examples")
    print()
    print("Training (gradient descent, lr=0.1):")
    theta, history = fit_gradient_descent(
        X_train, y_train, lr=0.1, epochs=2000, verbose_every=500,
    )
    print()
    print(f"Final parameters:")
    print(f"  Intercept (b0): {theta[0]:+.4f}")
    print(f"  Weights:        [{theta[1]:+.4f}, {theta[2]:+.4f}]")

    # Evaluate on test set
    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
    p_test = sigmoid(X_test_b @ theta)
    y_pred = (p_test >= 0.5).astype(int)
    print()
    print("Test set metrics (from-scratch model):")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"  F1:        {f1_score(y_test, y_pred):.3f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, p_test):.3f}")

    return X, y, X_train, X_test, y_train, y_test, theta, history


# ---------------------------------------------------------------------------
# Demo 2 --- scikit-learn comparison
# ---------------------------------------------------------------------------

def demo_sklearn(X_train, X_test, y_train, y_test):
    banner("DEMO 2 --- Same data, scikit-learn LogisticRegression")

    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(X_train, y_train)
    print(f"sklearn parameters:")
    print(f"  Intercept (b0): {model.intercept_[0]:+.4f}")
    print(f"  Weights:        [{model.coef_[0, 0]:+.4f}, {model.coef_[0, 1]:+.4f}]")

    y_pred = model.predict(X_test)
    p_test = model.predict_proba(X_test)[:, 1]
    print()
    print("Test set metrics (sklearn):")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"  F1:        {f1_score(y_test, y_pred):.3f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, p_test):.3f}")

    cm = confusion_matrix(y_test, y_pred)
    print()
    print("Confusion matrix at threshold = 0.5:")
    print(f"                    Predicted 0   Predicted 1")
    print(f"  Actual 0 ({cm[0].sum():>2d}):     {cm[0, 0]:>9d}     {cm[0, 1]:>9d}")
    print(f"  Actual 1 ({cm[1].sum():>2d}):     {cm[1, 0]:>9d}     {cm[1, 1]:>9d}")

    return model, p_test


# ---------------------------------------------------------------------------
# Demo 3 --- Four-panel visualisation
# ---------------------------------------------------------------------------

def make_visualisation(X, y, X_train, y_train, X_test, y_test,
                       history, model, p_test):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # --- Panel 1: sigmoid curve ---
    ax = axes[0, 0]
    z = np.linspace(-6, 6, 200)
    ax.plot(z, sigmoid(z), color="#3b82f6", linewidth=2.2)
    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#94A3B8", linestyle="--", linewidth=0.8)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("z = θᵀx")
    ax.set_ylabel("σ(z)")
    ax.set_title("The sigmoid function: σ(z) = 1 / (1 + e⁻ᶻ)")
    ax.grid(True, alpha=0.3)

    # --- Panel 2: training loss trajectory ---
    ax = axes[0, 1]
    ax.plot(history, color="#16a34a", linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log loss")
    ax.set_title("Training loss over 2000 gradient-descent steps")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: decision boundary ---
    ax = axes[1, 0]
    # Plot training data
    mask0 = y_train == 0
    mask1 = y_train == 1
    ax.scatter(X_train[mask0, 0], X_train[mask0, 1],
               color="#3b82f6", alpha=0.6, label="class 0", s=24)
    ax.scatter(X_train[mask1, 0], X_train[mask1, 1],
               color="#DC2626", alpha=0.6, label="class 1", s=24)
    # Decision boundary: σ(b0 + b1*x1 + b2*x2) = 0.5  ↔  b0 + b1*x1 + b2*x2 = 0
    b0 = model.intercept_[0]
    b1, b2 = model.coef_[0]
    x1_grid = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    x2_boundary = -(b0 + b1 * x1_grid) / b2
    ax.plot(x1_grid, x2_boundary, color="black", linewidth=2,
            label="decision boundary")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("Learned decision boundary on training data")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # --- Panel 4: ROC curve ---
    ax = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_test, p_test)
    auc = roc_auc_score(y_test, p_test)
    ax.plot(fpr, tpr, color="#8b5cf6", linewidth=2.2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#94A3B8", linestyle="--", linewidth=0.8,
            label="random (AUC = 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve on the test set")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "logistic_regression_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFour-panel visualisation saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    X, y, X_train, X_test, y_train, y_test, theta, history = demo_from_scratch()
    model, p_test = demo_sklearn(X_train, X_test, y_train, y_test)
    make_visualisation(X, y, X_train, y_train, X_test, y_test,
                       history, model, p_test)
    print()


if __name__ == "__main__":
    main()
