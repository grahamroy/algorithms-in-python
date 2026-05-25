"""
probabilistic_programming.py --- companion code for "Probabilistic
Programming" (Bayesian, Probabilistic & Causal Methods, Part 4).

Fit Bayesian linear regression using PyMC. The model is defined
declaratively; PyMC compiles it, picks NUTS, runs 4 chains, and
returns posterior samples with convergence diagnostics.

Dependencies: numpy, pymc. Runs in a few seconds.
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pymc as pm
import arviz as az


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


def make_dataset(n=200, true_a=2.0, true_b=0.5,
                 sigma=0.5, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    y = true_a + true_b * x + rng.normal(0, sigma, size=n)
    return x, y


def main() -> None:
    banner("DEMO --- PyMC fit of Bayesian linear regression")

    x, y = make_dataset()
    print(f"  Model           : intercept ~ N(0, 10²), "
          f"slope ~ N(0, 10²),")
    print(f"                    sigma ~ HalfNormal(1²)")
    print(f"  Likelihood      : y ~ N(intercept + slope·x, sigma²)")
    print(f"  Sampler         : NUTS (default), 4 chains, "
          f"1000 draws + 500 tune")

    t0 = time.perf_counter()
    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        slope = pm.Normal("slope", mu=0, sigma=10)
        sigma_param = pm.HalfNormal("sigma", sigma=1)
        pm.Normal("y_obs", mu=intercept + slope * x,
                  sigma=sigma_param, observed=y)

        idata = pm.sample(1000, tune=500, chains=4,
                          progressbar=False,
                          random_seed=RNG_SEED)
    dt = time.perf_counter() - t0
    print(f"  Wall time       : {dt:.2f} s")
    print()
    print(f"  Posterior summary:")

    summary = az.summary(idata, var_names=["intercept", "slope", "sigma"],
                         round_to=3)
    available_cols = list(summary.columns)
    # Find hdi columns by prefix (arviz changes the format across versions)
    hdi_cols = [c for c in available_cols if c.startswith("hdi_")]
    cols = ["mean", "sd"] + hdi_cols + ["ess_bulk", "r_hat"]
    cols = [c for c in cols if c in available_cols]
    print(f"                  {'  '.join(f'{c:>8}' for c in cols)}")
    for name in ["intercept", "slope", "sigma"]:
        row = summary.loc[name]
        vals = "  ".join(f"{row[c]:>8.3f}" for c in cols)
        print(f"  {name:<14} {vals}")
    print()


if __name__ == "__main__":
    main()
