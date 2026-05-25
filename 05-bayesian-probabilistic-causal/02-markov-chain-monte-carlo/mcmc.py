"""
mcmc.py --- companion code for "Markov Chain Monte Carlo"
(Bayesian, Probabilistic & Causal Methods, Part 2).

Three demos:
  1. Metropolis-Hastings sampling from a 2-D banana posterior
     across 4 chains, with acceptance rate + ESS reporting.
  2. Convergence diagnostics: R-hat per parameter.
  3. Compare against a grid-based reference (only feasible
     in 2-D) to validate the sampler.

Dependencies: numpy, scipy. Runs in well under a minute.
"""

import sys
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
# Target: 2-D banana-shaped unnormalised log-density
# Rosenbrock-like: log p̃(x, y) = -(1 - x)^2 / 0.5 - (y - x^2)^2 / 0.5
# ---------------------------------------------------------------------------

def log_target(theta):
    x, y = theta[..., 0], theta[..., 1]
    return -(1 - x) ** 2 / 0.5 - (y - x ** 2) ** 2 / 0.5


# ---------------------------------------------------------------------------
# Metropolis-Hastings with Gaussian random-walk proposal
# ---------------------------------------------------------------------------

def metropolis_hastings(log_p, init, n_samples, proposal_sigma,
                        seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    d = len(init)
    samples = np.zeros((n_samples, d))
    theta = np.array(init, dtype=float)
    lp = log_p(theta)
    n_accept = 0
    for t in range(n_samples):
        proposal = theta + rng.normal(0, proposal_sigma, size=d)
        lp_new = log_p(proposal)
        log_a = lp_new - lp
        if np.log(rng.random()) < log_a:
            theta = proposal
            lp = lp_new
            n_accept += 1
        samples[t] = theta
    return samples, n_accept / n_samples


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def r_hat(chains):
    """Gelman-Rubin R-hat. chains shape: (n_chains, n_samples)."""
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    B = n * chain_means.var(ddof=1)
    W = chain_vars.mean()
    var_hat = (1 - 1 / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def autocorr(x, max_lag=200):
    x = x - x.mean()
    var = x.var()
    n = len(x)
    return np.array([
        (x[:n - k] * x[k:]).mean() / var
        for k in range(max_lag)
    ])


def effective_sample_size(samples):
    n = len(samples)
    ac = autocorr(samples)
    # Sum until autocorr first hits zero (or 200 lags max)
    rho_sum = 1.0
    for k in range(1, len(ac)):
        if ac[k] < 0:
            break
        rho_sum += 2 * ac[k]
    return int(n / rho_sum)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_sampling():
    banner("DEMO 1 --- Metropolis-Hastings on a 2-D banana posterior")

    n_chains = 4
    n_samples = 10000
    burn = 2000
    proposal_sigma = 0.5
    inits = [(0.0, 0.0), (-1.0, 1.0), (2.0, 2.0), (-2.0, 4.0)]

    all_chains = []
    accept_rates = []
    for c in range(n_chains):
        samples, acc = metropolis_hastings(
            log_target, init=inits[c],
            n_samples=n_samples,
            proposal_sigma=proposal_sigma,
            seed=RNG_SEED + c,
        )
        all_chains.append(samples[burn:])
        accept_rates.append(acc)

    print(f"  Target          : Rosenbrock-like banana (unnormalised)")
    print(f"  Proposal        : Gaussian random walk, "
          f"sigma = {proposal_sigma}")
    print(f"  Chains          : {n_chains}")
    print(f"  Iterations / chain : {n_samples}")
    print(f"  Burn-in         : {burn}")
    print(f"  Acceptance rate : {np.mean(accept_rates):.2f}")

    ess_per_chain = [effective_sample_size(c[:, 0])
                     for c in all_chains]
    print(f"  Effective sample size (per chain) : "
          f"≈{int(np.mean(ess_per_chain))}")
    return all_chains


def demo_diagnostics(all_chains):
    banner("DEMO 2 --- Convergence diagnostics")

    stacked = np.stack(all_chains)  # (n_chains, n_samples, 2)
    rhat_1 = r_hat(stacked[:, :, 0])
    rhat_2 = r_hat(stacked[:, :, 1])
    print(f"  R-hat (θ_1)     : {rhat_1:.3f}   (target: ≤ 1.01)")
    print(f"  R-hat (θ_2)     : {rhat_2:.3f}   (target: ≤ 1.01)")

    flat = np.vstack(all_chains)
    print(f"  Mean θ_1        : {flat[:, 0].mean():>5.2f}   "
          f"(well-mixed across chains)")
    print(f"  Mean θ_2        : {flat[:, 1].mean():>5.2f}   "
          f"(well-mixed across chains)")
    return flat


def demo_grid_reference(mcmc_samples):
    banner("DEMO 3 --- Compare against grid-based reference "
           "(only feasible in 2-D)")

    # Build a fine grid, compute (unnormalised) posterior, normalise
    xs = np.linspace(-3, 4, 400)
    ys = np.linspace(-2, 10, 400)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    log_w = log_target(grid)
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()

    grid_mean = (grid * w[:, None]).sum(axis=0)
    grid_var = ((grid - grid_mean) ** 2 * w[:, None]).sum(axis=0)
    grid_std = np.sqrt(grid_var)

    mcmc_mean = mcmc_samples.mean(axis=0)
    mcmc_std = mcmc_samples.std(axis=0)

    print(f"  Posterior mean (θ_1, θ_2)  : "
          f" ({mcmc_mean[0]:.2f}, {mcmc_mean[1]:.2f})")
    print(f"  Grid reference (θ_1, θ_2)  : "
          f" ({grid_mean[0]:.2f}, {grid_mean[1]:.2f})")
    print(f"  Posterior std  (θ_1, θ_2)  : "
          f" ({mcmc_std[0]:.2f}, {mcmc_std[1]:.2f})")
    print(f"  Grid reference             : "
          f" ({grid_std[0]:.2f}, {grid_std[1]:.2f})")


def main() -> None:
    all_chains = demo_sampling()
    flat = demo_diagnostics(all_chains)
    demo_grid_reference(flat)
    print()


if __name__ == "__main__":
    main()
