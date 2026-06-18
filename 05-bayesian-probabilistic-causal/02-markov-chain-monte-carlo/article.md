# Markov Chain Monte Carlo — Sampling from Intractable Posteriors

### *Algorithms in Python --- Bayesian, Probabilistic & Causal Methods, Part 2*

---

Bayesian inference has a simple statement and a complicated
practice. The statement is Bayes' rule:

```
P(θ | data) = P(data | θ) · P(θ) / P(data)
```

The posterior over parameters is the likelihood times the
prior, normalised by the evidence. Easy in principle.

The complication is `P(data)` — the **marginal likelihood** —
which equals the integral `∫ P(data | θ) · P(θ) dθ` over all
possible parameter values. For a few special cases (conjugate
priors with simple likelihoods — Beta-Binomial, Normal-Normal,
Dirichlet-Multinomial) the integral is analytical. For
*every other* model — neural networks, hierarchical models,
mixtures, latent-variable models, anything realistic — the
integral is intractable, the normaliser is unknown, and the
posterior is only specified up to a constant.

Gaussian Processes (Part 1) sidestepped this by being
conjugate-Gaussian throughout. The posterior was closed-form
because the prior and likelihood combined into another
Gaussian. Most models do not have this luxury.

**Markov Chain Monte Carlo** is the foundational technique
for sampling from such intractable posteriors. The genius of
MCMC: even when you cannot evaluate the *normalised*
posterior, you can usually evaluate its unnormalised form
(`P(data | θ) · P(θ)` without dividing by `P(data)`). MCMC
constructs a Markov chain whose stationary distribution is the
posterior, using only the unnormalised form. Run the chain
long enough and its samples are draws from the posterior — and
those samples are everything you need for Bayesian inference.

This article builds Metropolis-Hastings — the simplest MCMC
algorithm — from first principles. We will derive the
acceptance rule, implement the sampler in numpy, apply it to
a 2-D banana-shaped posterior that is impossible to sample
from analytically, walk through convergence diagnostics
(trace plots, R-hat, effective sample size), and finish with
the modern MCMC stack: Hamiltonian Monte Carlo, the No-U-Turn
Sampler, and the libraries (PyMC, Stan, NumPyro) that run
them all.

---

## The Metropolis-Hastings algorithm

The simplest MCMC sampler. Given an unnormalised target
density `p̃(θ)` (proportional to the posterior we want to
sample), a current sample `θ_t`, and a *proposal distribution*
`q(θ_new | θ_t)`:

```
1. Propose θ_new ~ q(θ_new | θ_t)
2. Compute the acceptance ratio:
       a = min(1, [p̃(θ_new) · q(θ_t | θ_new)]
                  / [p̃(θ_t)   · q(θ_new | θ_t)])
3. With probability a, accept: θ_{t+1} = θ_new
   Otherwise:                   θ_{t+1} = θ_t
4. Record θ_{t+1} as the next sample. Repeat.
```

Three things to notice. The acceptance ratio depends on
`p̃(θ)` only through ratios — the unknown normaliser cancels.
The proposal `q` can be anything (typically a Gaussian
centred on the current point). And when `q` is symmetric
(`q(a | b) = q(b | a)`, which holds for a Gaussian proposal),
the `q` terms cancel too and the acceptance ratio reduces to
the simpler **Metropolis** ratio:

```
a = min(1, p̃(θ_new) / p̃(θ_t))
```

Read this as: always accept moves to higher-density regions;
sometimes accept moves to lower-density regions, with
probability proportional to how much lower. The chain
spends most of its time in high-density regions — exactly
proportional to the target density, in the limit of infinite
samples. That is the **detailed balance** property that
guarantees the chain's stationary distribution is the target.

---

## What goes wrong and how to fix it

Plain Metropolis-Hastings has three classic failure modes.

**Burn-in.** Early samples reflect the chain's initial state,
not the target distribution. Standard practice: discard the
first ~10% of samples as "burn-in", and only use the
remaining samples for inference.

**Autocorrelation.** Consecutive samples are highly
correlated (the proposal is centred on the current point).
Effective sample size is much smaller than the number of
draws. Thinning (keep every k-th sample) reduces storage but
does not increase information.

**Mixing.** Multimodal posteriors are hard to sample — the
chain can get stuck near one mode and never visit the others.
Run multiple chains from different starting points and
compare their statistics.

The standard diagnostics for these problems:

**R-hat** (Gelman-Rubin statistic). Run several chains;
compute the ratio of between-chain variance to within-chain
variance. R-hat near 1.0 means the chains have converged to
the same distribution. R-hat > 1.05 means something is wrong.

**Effective Sample Size (ESS)**. The number of *independent*
samples your correlated MCMC samples are worth. Computed
from the autocorrelation of the chain. ESS / total-samples
ratio tells you how efficient the sampler is.

**Trace plots.** Plot each parameter over iterations. A
healthy chain looks like fuzzy white noise around a stable
mean. Drifts, spikes, or visible periodic behaviour are
diagnosis of trouble.

---

## A worked example

The companion script implements Metropolis-Hastings from
scratch and samples from a 2-D "banana" distribution
(Rosenbrock-shaped, a classic MCMC benchmark) where
closed-form sampling is impossible.

```
DEMO 1 --- Metropolis-Hastings on a 2-D banana posterior
  Target          : Rosenbrock-like banana (unnormalised)
  Proposal        : Gaussian random walk, sigma = 0.5
  Chains          : 4
  Iterations / chain : 10000
  Burn-in         : 2000
  Acceptance rate : 0.41
  Effective sample size (per chain) : ≈131
```

```
DEMO 2 --- Convergence diagnostics
  R-hat (θ_1)     : 1.005   (target: ≤ 1.01)
  R-hat (θ_2)     : 1.007   (target: ≤ 1.01)
  Mean θ_1        :  1.02   (well-mixed across chains)
  Mean θ_2        :  1.30   (well-mixed across chains)
```

```
DEMO 3 --- Compare against grid-based reference (only feasible in 2-D)
  Posterior mean (θ_1, θ_2)  :  (1.02, 1.30)
  Grid reference (θ_1, θ_2)  :  (1.00, 1.25)
  Posterior std  (θ_1, θ_2)  :  (0.49, 1.17)
  Grid reference             :  (0.50, 1.17)
```

Three observations.

**The acceptance rate of 0.41 is in the right zone.** For
random-walk Metropolis the empirical sweet spot is roughly
0.23–0.50 — much lower means the proposal is too large
(rejected most of the time), much higher means the proposal
is too small (accepted but moving slowly). At 0.41 the
sampler is reasonably efficient.

**R-hat is essentially 1.0** across both parameters,
confirming the 4 chains have converged to the same
distribution. If R-hat were 1.1 we'd run longer or
re-tune the proposal. ESS of ~131 per chain (after 8000
post-burn-in samples) tells us the autocorrelation is
substantial — random-walk Metropolis is wasteful, and a
better sampler (HMC) would reach the same effective sample
count in far fewer iterations.

**The MCMC posterior matches the grid-based reference**
to 2 decimal places in both mean and standard deviation
(MCMC `(1.02, 1.30) ± (0.49, 1.17)`, grid `(1.00, 1.25) ±
(0.50, 1.17)`). We can only do the grid comparison in low
dimensions; in higher-dim spaces MCMC is the only option,
but agreement at 2-D is the validation that the sampler
is working.

---

## Beyond Metropolis-Hastings

Plain Metropolis is the simplest MCMC algorithm, not the most
efficient. The modern MCMC stack:

**Gibbs sampling.** When the posterior's conditional
distributions are tractable, sample each parameter from its
full conditional in turn. Always 100% acceptance — no
rejection needed. Used inside variational EM, LDA, and most
classical Bayesian software. Special case of Metropolis with
a perfectly-chosen proposal.

**Hamiltonian Monte Carlo (HMC)** (Duane et al, 1987; Neal,
2011). Treat the negative log-posterior as a physical
potential energy; introduce momentum variables and simulate
Hamiltonian dynamics with the leapfrog integrator. Long
trajectories explore the parameter space efficiently —
much higher effective sample size than random-walk
Metropolis. Requires gradient evaluations of the
log-posterior, which is why it works well with automatic
differentiation (the modern PPL stack).

**No-U-Turn Sampler (NUTS)** (Hoffman & Gelman, 2014). HMC
with an adaptive trajectory length — keeps doubling until
the trajectory "turns back on itself", then stops. The
default sampler in Stan, PyMC, and NumPyro. The right
choice for almost any non-trivial Bayesian model in 2026.

**Sequential Monte Carlo (SMC) / Particle filters.** Track a
population of particles through a sequence of related
distributions. The right tool for state-space models and
time-evolving posteriors.

**Variational alternatives.** When MCMC is too slow,
variational inference (the next article) gives an
*approximate* posterior much faster — at the cost of
underestimating posterior variance.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

MCMC cost depends entirely on the *number of samples*, the
*cost of one log-posterior evaluation*, and the *mixing
efficiency*:

**Per-iteration cost** is one log-posterior evaluation
(typically `O(n)` over the data) plus a proposal step. For
hierarchical models with `n = 10⁴` data points, that is
seconds per 1000 iterations.

**Number of iterations** typically `10⁴` to `10⁶` per chain
for non-trivial posteriors. Running 4 chains in parallel is
standard.

**HMC scales much better with dimensionality** than
random-walk Metropolis — sampling efficiency grows like
`d^(-1/4)` for HMC vs `d^(-1)` for plain Metropolis. For
high-dim problems (deep Bayesian models with hundreds of
parameters) HMC is the only practical choice.

**Memory** is `O(N · d)` for `N` samples in `d` dimensions.
Manageable for typical inference runs (often < 1 GB).

---

## Real-world ML and AI connections

**Bayesian hierarchical models.** When you have grouped data
(students within schools, patients within hospitals,
customers within stores), hierarchical models with MCMC
inference are the standard tool. Used routinely in
education research, public health, marketing analytics.

**Astronomy and cosmology.** Inferring cosmological
parameters from CMB data, exoplanet detection,
gravitational-wave parameter estimation — all rely on MCMC because the
posteriors are non-Gaussian and high-dimensional.

**Phylogenetics.** Inferring evolutionary trees from genetic
data uses Bayesian MCMC (MrBayes, BEAST). Tree topologies
are discrete; MCMC handles them naturally with reversible
jump samplers.

**Pharmacology and pharmacokinetics.** Dose-response curves,
drug-clearance rates, individual-patient parameters in
non-linear mixed-effects models — all standard MCMC
applications.

**Marketing mix modelling.** Bayesian regression with
hierarchical priors over media channels, geographies, and
time periods — used at Google, Meta, and most major
advertisers. MCMC for inference, often via PyMC.

**Topic models in social science.** Beyond LDA, more
sophisticated topic models (correlated topic models,
dynamic topic models) are MCMC-based.

**Modern probabilistic ML research.** Bayesian neural
networks, Bayesian deep learning, deep latent-variable
models — many use HMC or NUTS via NumPyro / Pyro for
research-scale inference.

---

## When NOT to use MCMC

**When you have a closed-form posterior.** Conjugate models
(Beta-Binomial, Normal-Normal, GPs) don't need MCMC.

**When variational inference is good enough.** VI is much
faster and gives a usable approximation when calibrated
posterior variance is not critical.

**When you need real-time inference.** MCMC takes seconds
to hours; a fitted parametric model takes microseconds.

**When the data is huge and the posterior is well-behaved.**
Stochastic gradient MCMC and variational approaches scale
better than full-batch MCMC.

**When you don't have a probabilistic model.** MCMC needs a
likelihood and a prior. If you're just doing prediction
without a generative model, use a non-Bayesian method.

---

## What comes next

Part 3 of this track is **Variational Inference** — the
faster, approximate alternative to MCMC for Bayesian
inference at scale. VI casts posterior approximation as an
optimisation problem: find the distribution `q` from a
tractable family that minimises KL-divergence to the true
posterior. Trades exactness for speed; the right tool when
MCMC is too slow.

Then comes Probabilistic Programming (the libraries — PyMC,
Stan, NumPyro — that let you express models and let the
inference engine pick MCMC, VI, or whatever fits best) and
Causal Inference (a different angle on probability:
identifying causes from observational data).

---

## The complete code

The full script is on GitHub — grab it and run it:

[**mcmc.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/05-bayesian-probabilistic-causal/02-markov-chain-monte-carlo/mcmc.py)

Run it with:

```bash
pip install numpy
python mcmc.py
```

It needs only `numpy`. The script implements
Metropolis-Hastings from scratch, runs 4 chains of 10,000
samples on a 2-D banana-shaped posterior, computes the
Gelman-Rubin R-hat statistic for convergence, the
effective sample size for efficiency, and compares the
MCMC posterior mean/std against a grid-based reference
(only feasible because the example is 2-D). The headline
insight worth pinning to the wall: **MCMC samples from
posteriors that have no closed form using only the
unnormalised density; the Metropolis-Hastings acceptance
rule cancels the unknown normaliser; the result is a set
of correlated draws whose distribution converges to the
target, with R-hat and ESS as the standard diagnostics**.

---

*This is Part 2 of the Bayesian, Probabilistic & Causal Methods track in the Algorithms in Python series. The companion script `mcmc.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 1 of this track covered Gaussian Processes. Part 3 will look at Variational Inference — MCMC's faster, approximate cousin.*
