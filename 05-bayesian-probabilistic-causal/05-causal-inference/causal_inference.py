"""
causal_inference.py --- companion code for "Causal Inference"
(Bayesian, Probabilistic & Causal Methods, Part 5).

Simulate a confounded observational dataset and estimate the
average treatment effect (ATE) three ways:
  1. Naive: simple difference in means between treated and
     control (biased — confounded).
  2. Regression adjustment: fit E[Y | T, Z], predict both
     potential outcomes per unit, average the difference.
  3. Inverse-propensity weighting (IPW): estimate the
     propensity score, reweight observations to mimic an RCT.

The true ATE is 2.0. The naive estimate is wildly biased; both
adjustment methods recover the true effect to within statistical
noise.

Dependencies: numpy, scikit-learn. Runs in under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Generate confounded observational data
# ---------------------------------------------------------------------------

def make_dataset(n=500, true_ate=2.0, seed=RNG_SEED):
    """Two confounders Z1, Z2 both increase the probability of
    treatment AND increase the outcome — classic confounding."""
    rng = np.random.default_rng(seed)
    Z1 = rng.normal(0, 1, size=n)
    Z2 = rng.normal(0, 1, size=n)
    # Propensity score: depends on confounders
    logit = -0.5 + 1.5 * Z1 + 1.0 * Z2
    p_treat = 1.0 / (1.0 + np.exp(-logit))
    T = (rng.uniform(size=n) < p_treat).astype(int)
    # Outcome: depends on treatment AND confounders, plus noise
    Y = 1.0 + true_ate * T + 1.5 * Z1 + 1.2 * Z2 + rng.normal(0, 1, size=n)
    Z = np.column_stack([Z1, Z2])
    return T, Y, Z, true_ate


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def naive_diff_means(T, Y):
    """Simple difference in means — ignores confounding."""
    return float(Y[T == 1].mean() - Y[T == 0].mean())


def regression_adjustment(T, Y, Z):
    """Fit E[Y | T, Z]; predict both potential outcomes per
    unit; average the difference."""
    X = np.column_stack([T, Z])
    model = LinearRegression().fit(X, Y)
    X0 = np.column_stack([np.zeros_like(T), Z])
    X1 = np.column_stack([np.ones_like(T), Z])
    y0 = model.predict(X0)
    y1 = model.predict(X1)
    return float((y1 - y0).mean())


def inverse_propensity_weighting(T, Y, Z):
    """Estimate propensity score e(Z) = P(T=1 | Z), then
    reweight observations to mimic randomised assignment."""
    pscore_model = LogisticRegression().fit(Z, T)
    e = pscore_model.predict_proba(Z)[:, 1]
    # Clip to avoid division by zero in extreme propensities
    e = np.clip(e, 0.01, 0.99)
    w_treated = T / e
    w_control = (1 - T) / (1 - e)
    ate = float((w_treated * Y).mean() - (w_control * Y).mean())
    return ate


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    banner("DEMO --- Causal effect estimation on confounded synthetic data")

    T, Y, Z, true_ate = make_dataset()
    n = len(T)

    print(f"  Sample size : {n} units")
    print(f"  True ATE    : {true_ate:.2f}")
    print(f"  Treatment   : binary, propensity depends on Z_1, Z_2")
    print(f"  Outcome     : Y = 1.0 + 2.0 * T + 1.5 * Z_1 + 1.2 * Z_2 + noise")
    print()
    print(f"  Treated fraction : {T.mean():.2f}")
    print()

    print(f"  {'Method':<38}   {'Estimated ATE':>13}   {'Bias':>5}")
    print(f"  {'-' * 36:<38}   {'-' * 13:>13}   {'-' * 5:>5}")

    estimators = [
        ("Naive (simple difference in means)", naive_diff_means),
        ("Regression adjustment (linear)",      regression_adjustment),
        ("Inverse propensity weighting",        inverse_propensity_weighting),
    ]

    for name, fn in estimators:
        if fn is naive_diff_means:
            est = fn(T, Y)
        else:
            est = fn(T, Y, Z)
        bias = est - true_ate
        print(f"  {name:<38}   {est:>13.2f}   {bias:+.2f}")
    print()


if __name__ == "__main__":
    main()
