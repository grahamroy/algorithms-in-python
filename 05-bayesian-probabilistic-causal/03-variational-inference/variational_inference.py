"""
variational_inference.py --- companion code for "Variational Inference"
(Bayesian, Probabilistic & Causal Methods, Part 3).

Fit Bayesian linear regression y = a + b·x + noise three ways:
  1. Exact analytical posterior (conjugate Gaussian).
  2. Mean-field variational inference via coordinate ascent
     on the ELBO.
  3. MCMC with Metropolis-Hastings.

All three should agree on this conjugate model.

Dependencies: numpy. Runs in under a second.
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Generate data
# ---------------------------------------------------------------------------

def make_dataset(n=200, true_a=2.0, true_b=0.5,
                 sigma=0.5, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    y = true_a + true_b * x + rng.normal(0, sigma, size=n)
    return x, y, sigma


# ---------------------------------------------------------------------------
# Method 1: Exact analytical posterior
# ---------------------------------------------------------------------------
# Likelihood: y_i ~ N(a + b·x_i, sigma²)
# Prior:      a, b ~ N(0, tau²)  independent
# Posterior:  Gaussian, mean = (X^T X / sigma² + I/tau²)^{-1} X^T y / sigma²
#             cov  = (X^T X / sigma² + I/tau²)^{-1}

def exact_posterior(x, y, sigma, tau=10.0):
    X = np.column_stack([np.ones_like(x), x])  # [1, x]
    n, d = X.shape
    A = X.T @ X / sigma ** 2 + np.eye(d) / tau ** 2
    cov = np.linalg.inv(A)
    mean = cov @ X.T @ y / sigma ** 2
    return mean, np.sqrt(np.diag(cov))


# ---------------------------------------------------------------------------
# Method 2: Mean-field VI by coordinate ascent
# ---------------------------------------------------------------------------
# q(a) = N(μ_a, σ_a²), q(b) = N(μ_b, σ_b²) independent.
# Under mean-field, q*_a has closed form given current q(b), and vice versa.
# Compute updates derived from log p(y, a, b) under Gaussian prior + likelihood.

def mean_field_vi(x, y, sigma, tau=10.0, n_iter=100, tol=1e-7):
    n = len(x)
    sum_x = x.sum()
    sum_x2 = (x ** 2).sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    s2 = sigma ** 2
    t2 = tau ** 2

    mu_a, sigma2_a = 0.0, 1.0
    mu_b, sigma2_b = 0.0, 1.0

    prev_elbo = -np.inf
    for it in range(n_iter):
        # Optimal q(a) given current q(b)
        prec_a = n / s2 + 1 / t2
        mean_a = (sum_y - mu_b * sum_x) / s2 / prec_a
        sigma2_a = 1 / prec_a
        mu_a = mean_a

        # Optimal q(b) given current q(a)
        prec_b = sum_x2 / s2 + 1 / t2
        mean_b = (sum_xy - mu_a * sum_x) / s2 / prec_b
        sigma2_b = 1 / prec_b
        mu_b = mean_b

    return np.array([mu_a, mu_b]), np.array([np.sqrt(sigma2_a),
                                              np.sqrt(sigma2_b)])


# ---------------------------------------------------------------------------
# Method 3: Metropolis-Hastings MCMC
# ---------------------------------------------------------------------------

def log_posterior(theta, x, y, sigma, tau=10.0):
    a, b = theta
    pred = a + b * x
    log_lik = -0.5 * (((y - pred) / sigma) ** 2).sum()
    log_prior = -0.5 * (a ** 2 + b ** 2) / tau ** 2
    return log_lik + log_prior


def mcmc(x, y, sigma, n_samples=10000, burn=2000,
         proposal_sigma=0.1, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    theta = np.array([0.0, 0.0])
    lp = log_posterior(theta, x, y, sigma)
    samples = np.zeros((n_samples, 2))
    for t in range(n_samples):
        prop = theta + rng.normal(0, proposal_sigma, size=2)
        lp_new = log_posterior(prop, x, y, sigma)
        if np.log(rng.random()) < lp_new - lp:
            theta = prop
            lp = lp_new
        samples[t] = theta
    return samples[burn:]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    banner("DEMO --- VI vs MCMC vs exact posterior for Bayesian linear regression")

    x, y, sigma = make_dataset()
    print(f"  True parameters       : intercept = 2.00, slope = 0.50")
    print(f"  Training set          : {len(x)} noisy points")
    print(f"  Noise (sigma)         : {sigma}")
    print()

    print(f"  {'Method':<17} {'intercept (mean ± std)':<22}  "
          f"{'slope (mean ± std)':<18}  {'time (s)':>8}")
    print(f"  {'-' * 15:<17} {'-' * 21:<22}  "
          f"{'-' * 17:<18}  {'-' * 8:>8}")

    # Exact
    t0 = time.perf_counter()
    mu_e, sd_e = exact_posterior(x, y, sigma)
    dt = time.perf_counter() - t0
    print(f"  {'Exact analytical':<17} "
          f"{mu_e[0]:>5.2f} ± {sd_e[0]:.3f}{'':<8}  "
          f"{mu_e[1]:>5.2f} ± {sd_e[1]:.3f}{'':<4}  {dt:>8.3f}")

    # VI
    t0 = time.perf_counter()
    mu_v, sd_v = mean_field_vi(x, y, sigma)
    dt = time.perf_counter() - t0
    print(f"  {'Mean-field VI':<17} "
          f"{mu_v[0]:>5.2f} ± {sd_v[0]:.3f}{'':<8}  "
          f"{mu_v[1]:>5.2f} ± {sd_v[1]:.3f}{'':<4}  {dt:>8.3f}")

    # MCMC
    t0 = time.perf_counter()
    samples = mcmc(x, y, sigma)
    dt = time.perf_counter() - t0
    mu_m = samples.mean(axis=0)
    sd_m = samples.std(axis=0)
    print(f"  {'MCMC (10k samples)':<17} "
          f"{mu_m[0]:>5.2f} ± {sd_m[0]:.3f}{'':<8}  "
          f"{mu_m[1]:>5.2f} ± {sd_m[1]:.3f}{'':<4}  {dt:>8.3f}")
    print()


if __name__ == "__main__":
    main()
