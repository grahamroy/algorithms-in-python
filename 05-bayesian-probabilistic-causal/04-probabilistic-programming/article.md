# Probabilistic Programming — Write the Model, Let the Engine Pick the Sampler

### *Algorithms in Python --- Bayesian, Probabilistic & Causal Methods, Part 4*

---

The previous three articles each implemented a *specific*
inference algorithm — Gaussian Processes, Metropolis-Hastings
MCMC, mean-field Variational Inference. They taught the
machinery, but they also illustrated the problem: every
new Bayesian model required new derivations, new sampler
code, new convergence diagnostics. For a working analyst or
researcher, this is a tax on every new project.

**Probabilistic Programming Languages** (PPLs) eliminate that
tax. You declare your model — priors, likelihood, observed
data — using a domain-specific language inside Python (or R,
Julia). The PPL compiles your model into a computational graph,
chooses an appropriate inference algorithm (NUTS for most
continuous models, mean-field VI for big data, custom
samplers for tricky cases), runs it, and gives you posterior
samples. You write 10 lines of model definition; the PPL
handles everything else.

The big PPL stack in 2026:

- **PyMC** (Python). The most popular Python PPL.
  Auto-differentiation under PyTensor, NUTS by default,
  rich variational and MCMC alternatives.
- **NumPyro** (Python, JAX). PyMC's faster GPU-friendly
  cousin. Same modelling API as Pyro; JAX-based inference
  loop.
- **Stan** (C++ with Python / R bindings). The gold
  standard for Bayesian inference. Hand-tuned C++ MCMC
  implementation, mature ecosystem, the workhorse of
  applied Bayesian statistics.
- **Pyro** (PyTorch). Deep PPL focused on variational
  methods and amortised inference.
- **Turing.jl** (Julia). The Julia ecosystem's PPL.

This article uses PyMC because it's the most accessible
Python PPL with the broadest user community. The same
concepts apply to Stan / NumPyro / Pyro with minor syntax
changes.

We will build a Bayesian linear regression model the *PPL
way* — declarative, no hand-coded sampler — and watch PyMC
fit it with NUTS, do convergence diagnostics automatically,
and give us posterior samples we can use for prediction with
calibrated uncertainty.

---

## What does a PPL actually do?

Given a model definition and observed data, the PPL pipeline:

1. **Parses the model** into a static computational graph
   representing the joint density `p(data, parameters)`.
2. **Compiles** it with auto-differentiation (PyTensor for
   PyMC; JAX for NumPyro; custom AD for Stan) so the
   inference engine can compute gradients of the
   log-posterior with respect to parameters.
3. **Picks an inference algorithm** — NUTS by default,
   variational inference if requested, custom samplers
   (Gibbs, Metropolis) if specified.
4. **Runs the algorithm** — manages multiple chains,
   parallelism, tuning, adaptive step sizes.
5. **Returns an InferenceData object** with samples,
   convergence diagnostics (R-hat, ESS), summary
   statistics, and tools for posterior prediction.

You write the *model*. The PPL handles everything else.

---

## A worked example

The companion script fits Bayesian linear regression on the
same dataset as Part 3 (200 noisy points from `y = 2 + 0.5 ·
x + ε`). This time we use PyMC instead of hand-coded MCMC
or hand-coded VI.

```python
import pymc as pm

with pm.Model() as model:
    # Priors
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    slope = pm.Normal("slope", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Likelihood
    pm.Normal("y_obs", mu=intercept + slope * x,
              sigma=sigma, observed=y)

    # Inference
    idata = pm.sample(1000, tune=500, chains=4,
                      progressbar=False)
```

Eleven lines of model definition. No sampler code, no
gradients, no convergence checking. PyMC compiles the
model, picks NUTS, runs 4 chains in parallel, and returns
posterior samples. The script then reports the summary:

```
DEMO --- PyMC fit of Bayesian linear regression
  Model           : intercept ~ N(0, 10²), slope ~ N(0, 10²),
                    sigma ~ HalfNormal(1²)
  Likelihood      : y ~ N(intercept + slope·x, sigma²)
  Sampler         : NUTS (default), 4 chains, 1000 draws + 500 tune
  Wall time       : 14.95 s

  Posterior summary:
                      mean        sd  ess_bulk     r_hat
  intercept         1.969     0.035  5382.648     1.001
  slope             0.501     0.020  5206.390     1.002
  sigma             0.489     0.025  4641.912     1.002
```

Four observations.

**The model definition is the algorithm.** Eleven lines of
Python become a fully-specified probabilistic model with
NUTS inference, parallel chains, convergence diagnostics,
and posterior summary statistics. The hand-coded
Metropolis-Hastings sampler from Part 2 was 80 lines for a
*simpler* model.

**NUTS converges fast** — R-hat is 1.001–1.002 across all
parameters, effective sample size is ~5000 (about 125% of
the nominal 4000 draws, because NUTS draws are nearly
uncorrelated thanks to its adaptive trajectories). The
"tuning" phase (500 iterations before the recorded chain)
is where NUTS adapts step sizes and the mass matrix;
after tuning, the sampler is in its efficient regime.

**Posterior summary matches the analytical answer from
Part 3.** Same data, same priors. The intercept (1.969
mean, 0.035 std) and slope (0.501 mean, 0.020 std)
posteriors agree to the third decimal with the exact and
mean-field-VI results from the previous article. PyMC's
posterior over `sigma` (0.489 mean) recovers the true value
(0.50) within the noise.

**Wall time is ~15 seconds** — slower than VI (~ms) and
slower than my hand-rolled MCMC (0.05 s for 10k samples)
*on this tiny problem*. The PyMC overhead is dominated by
graph compilation (one-time, several seconds). For
non-trivial models (hundreds of parameters, custom
likelihoods, hierarchical structure), PyMC's compiled
NUTS is *faster* than any hand-coded sampler — the
constant-factor speedup from the C++/PyTensor backend
overwhelms the compilation cost.

---

## What PPLs enable

The PPL abstraction unlocks several patterns that hand-coded
inference cannot easily match.

**Modular model construction.** Build complex hierarchical
models by composition: a per-school random effect plus a
per-state random effect plus a global trend, each
expressible in a few lines. The hand-coded equivalent is
dozens of lines of bookkeeping.

**Automatic sampler choice.** PyMC's `pm.sample()` defaults
to NUTS but can fall back to other samplers for parameters
NUTS cannot handle (e.g. discrete parameters get
Metropolis or Gibbs automatically).

**Posterior predictive checks.** `pm.sample_posterior_predictive()`
draws from the posterior, runs the likelihood forward, and
gives you simulated datasets matching your fitted model.
Standard diagnostic for model fit.

**Easy switching between MCMC and VI.** `pm.fit()` does
variational inference with the same model definition. Try
both, pick the better one.

**Reproducibility and code reuse.** A model written in PyMC
can be shared, modified, extended. The same model can be
run by anyone with PyMC installed, no custom code required.

**Auto-differentiation everywhere.** PyTensor (the backend)
computes log-posterior gradients for any user-defined model
without manual derivation. This is the main reason NUTS
became practical for non-experts.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

PPL cost is dominated by:

**Compilation.** Building the computational graph: typically
1–10 seconds for small models, longer for big ones. One-time
cost per model definition.

**Per-iteration cost.** Same as the underlying sampler —
NUTS evaluates the log-posterior and its gradient many
times per iteration (~`2^j` for adaptive tree depth `j`,
typically 5–10).

**Parallelism.** Multi-chain MCMC is embarrassingly parallel
— `M` chains on `M` cores is `M×` faster. PyMC handles this
automatically.

**Memory.** Same as raw MCMC — `O(N · d)` for `N` samples in
`d` dimensions.

For non-trivial models (>50 parameters, hierarchical
structure, custom likelihoods) the PPL is *faster* than
hand-coded MCMC because the underlying C++ / JAX / PyTensor
backends are extensively optimised.

---

## Real-world ML and AI connections

**Applied Bayesian statistics across science.** Astronomy,
ecology, psychology, epidemiology, pharmacology — most
modern Bayesian analyses are done in PyMC or Stan.
Pre-PPL, you'd hand-code samplers; now you write the
model and the PPL does the rest.

**Marketing mix modelling.** Bayesian hierarchical
regression with informative priors for ad-spend
effectiveness — Google's open-source LightweightMMM,
Robyn (R), and many proprietary stacks are all PPL-based
(PyMC, Stan, NumPyro).

**Pharmaceutical pharmacokinetics / pharmacodynamics.**
Non-linear mixed-effects models for drug clearance with
patient-level random effects. Stan and PyMC are the
standard tools.

**Election forecasting.** The Economist's election model
(by Andrew Gelman et al) is a Bayesian hierarchical model
written in Stan, with state-level effects, time trends,
and uncertainty quantification throughout.

**Operations research and decision analysis.** Cost-benefit
analyses under uncertainty, supply-chain risk modelling,
inventory optimisation — all Bayesian and increasingly
PPL-based.

**Bayesian deep learning research.** Pyro and NumPyro are
the research workhorses for variational Bayesian neural
networks, normalising flows, and other probabilistic deep
models.

**Causal inference.** Probabilistic causal models — with
latent confounders, instrumental variables, mediation — are
typically expressed in PPLs and fit with MCMC.

---

## When NOT to use a PPL

**When the model is small and conjugate.** Just compute the
closed-form posterior — no need for PyMC overhead.

**When you need sub-millisecond inference per query.** PPL
inference is built around full-batch MCMC. For production
serving, fit the posterior once with PyMC, then ship a
distilled point-estimate model.

**When the data is huge and the posterior is unimodal.**
Stochastic gradient methods (PyTorch with built-in
optimisers, JAX) are often a better fit than full Bayesian
MCMC.

**When the model is so simple that the abstraction is
overhead.** A one-line conjugate update doesn't need a PPL.

**When deployment constraints rule out heavy dependencies.**
PyMC + PyTensor is ~200 MB installed. For edge / embedded
deployment use a custom MCMC or just point estimates.

---

## What comes next

Part 5 of this track is **Causal Inference** — the family of
methods that move from "X is correlated with Y" to "X
*causes* Y given everything else we know". Causal inference
is the framework behind A/B-test analysis at scale, policy
evaluation in economics, treatment-effect estimation in
medicine, and counterfactual reasoning in ML
interpretability. The mathematical toolkit (do-calculus,
propensity scores, instrumental variables, regression
discontinuity) is largely orthogonal to the
sampling/optimisation tools we've covered so far.

After causal inference the Bayesian / Probabilistic /
Causal track closes, and the series moves to time series
forecasting.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**probabilistic_programming.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/05-bayesian-probabilistic-causal/04-probabilistic-programming/probabilistic_programming.py)

Run it with:

```bash
pip install numpy pymc
python probabilistic_programming.py
```

It needs `numpy`, `pymc`, and `arviz` (arviz installs as a
pymc dependency). The script fits Bayesian
linear regression on the same dataset from Part 3 — but
this time using PyMC instead of hand-coded MCMC or VI.
The model definition is 11 lines; PyMC compiles it,
selects NUTS as the sampler, runs 4 parallel chains, and
returns a posterior with R-hat 1.0 and effective sample
size ~5000 across all parameters, matching the analytical
posterior to two decimal places. The headline insight worth
pinning to the wall: **probabilistic programming abstracts
the inference machinery; you specify the model
declaratively (priors, likelihood, observed data), the PPL
handles compilation, sampler selection, parallelism, and
diagnostics; for non-trivial models the PPL is faster than
hand-coded inference and dramatically easier to write**.

---

*This is Part 4 of the Bayesian, Probabilistic & Causal Methods track in the Algorithms in Python series. The companion script `probabilistic_programming.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 3 of this track covered Variational Inference. Part 5 will look at Causal Inference — moving from statistical association to causal identification.*
