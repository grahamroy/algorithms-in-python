# Variational Inference — Bayesian Approximation as Optimisation

### *Algorithms in Python --- Bayesian, Probabilistic & Causal Methods, Part 3*

---

Part 2 introduced MCMC — the gold standard for Bayesian
inference when the posterior has no closed form. MCMC's big
strength: with enough samples, it converges to the true
posterior, with calibrated uncertainty. MCMC's big weakness:
it can be *very* slow. Sampling a posterior over a million
parameters with reasonable effective sample size can take
hours or days, and at internet scale it is impractical.

**Variational Inference** (VI) is the faster, approximate
alternative. Rather than sampling the posterior, VI casts
posterior approximation as an **optimisation problem**:
choose a tractable family of distributions `q(θ)`, then find
the member of that family that is closest (in KL-divergence)
to the true posterior `p(θ | data)`. The optimisation can be
done with standard gradient descent in seconds to minutes,
even on models where MCMC would take hours.

The price is approximation. The fitted `q` is the *best*
member of its family, not the true posterior. Two
approximation costs to be aware of: the family is usually
parametric (Gaussians, factorised distributions) which limits
what `q` can represent, and KL(`q` || `p`) under-estimates
posterior variance — VI's approximations are typically *too
confident*. For applications where confidence calibration
matters (medical, financial, safety-critical) MCMC is still
preferred; for applications where speed dominates (LDA on
millions of documents, variational autoencoders, Bayesian
neural networks at production scale) VI wins.

This article builds variational inference from first
principles. We will define the **ELBO** — the lower bound
that VI maximises — derive the **mean-field** algorithm in
which `q` factorises across parameters, implement
coordinate-ascent VI on a Bayesian linear regression
problem with a non-conjugate Gaussian prior, compare against
MCMC and against the exact analytical posterior, and finish
with the **reparameterisation trick** and amortised inference
that power modern deep VI methods like variational
autoencoders.

---

## The variational framework

Given an intractable posterior `p(θ | data)`, pick a family
of tractable distributions `Q` (typically Gaussian, or a
factorised product of simpler distributions). The variational
inference recipe:

```
q* = argmin_{q ∈ Q}  KL(q(θ) || p(θ | data))
```

KL-divergence measures how much one distribution differs
from another. KL(`q` || `p`) is zero when `q == p` and
positive otherwise. The closest `q` to the true posterior
within the chosen family is the variational answer.

Trouble: we cannot compute KL directly because it involves
the unknown true posterior `p(θ | data)` — which is the
quantity we are trying to approximate. The trick: minimising
KL(`q` || `p`) is equivalent to **maximising** a quantity
called the **ELBO** (Evidence Lower BOund) which depends only
on tractable terms:

```
ELBO(q) = E_q[log p(data, θ)] - E_q[log q(θ)]
        = E_q[log p(data | θ)]
          - KL(q(θ) || p(θ))
        ≤ log p(data)
```

The ELBO is a lower bound on the marginal log-likelihood
`log p(data)`. Two interpretations:

- **Reconstruction term**: `E_q[log p(data | θ)]` — how well
  the posterior approximation explains the data.
- **Regularisation term**: `KL(q(θ) || p(θ))` — how much
  the posterior approximation deviates from the prior.

Maximising the ELBO is exactly equivalent to minimising
KL(`q` || `p`). Since both depend only on the prior, the
likelihood, and `q` itself (no need for the intractable
normaliser), we can optimise it with standard methods —
typically coordinate ascent for simple models, stochastic
gradient ascent for big ones.

---

## The mean-field assumption

The simplest and most-used variational family is the
**mean-field** family — `q` factorises across all parameters:

```
q(θ) = q_1(θ_1) · q_2(θ_2) · ... · q_d(θ_d)
```

Each `q_i` is an independent distribution over a single
parameter. The optimisation breaks into per-parameter
updates that often have closed-form solutions (one
parameter at a time, given all others fixed).

For models in the **exponential family** (most common
distributions: Gaussian, Bernoulli, Poisson, etc.), the
optimal `q_i` has a closed form:

```
log q*_i(θ_i) = E_{q_{-i}}[log p(θ, data)] + const
```

where `q_{-i}` is `q` with the `i`-th factor removed. The
expectation can usually be computed analytically. Coordinate
ascent — update each `q_i` while holding the others fixed —
converges to a local optimum of the ELBO.

The mean-field approximation is wrong whenever posterior
parameters are *correlated* — and they usually are.
Mean-field VI captures the marginal distributions but loses
the correlations, which is one of the main reasons VI
underestimates posterior variance. For some applications
(point estimation, prediction) this doesn't matter; for
others (uncertainty calibration, sensitivity analysis) it
matters a lot.

---

## A worked example: VI for Bayesian linear regression

The companion script implements variational inference for
Bayesian linear regression — the simplest non-trivial
Bayesian model. Data is generated from `y = 2 + 0.5 · x + ε`
with Gaussian noise; we put Gaussian priors over the slope
and intercept and fit the posterior three ways:

1. **Analytical** (this model is conjugate-Gaussian so the
   exact posterior is closed-form).
2. **Mean-field VI** (assume the posterior factorises;
   coordinate ascent on the ELBO).
3. **MCMC** (the previous article's Metropolis-Hastings
   sampler).

```
DEMO --- VI vs MCMC vs exact posterior for Bayesian linear regression
  True parameters       : intercept = 2.00, slope = 0.50
  Training set          : 200 noisy points
  Noise (sigma)         : 0.5

  Method            intercept (mean ± std)  slope (mean ± std)  time (s)
  ----------------- ----------------------  -----------------   --------
  Exact analytical            1.97 ± 0.035       0.50 ± 0.020      0.000
  Mean-field VI               1.97 ± 0.035       0.50 ± 0.020      0.000
  MCMC (10k samples)          1.97 ± 0.035       0.50 ± 0.020      0.047
```

Three observations.

**On this conjugate-Gaussian model VI is exact.** The
mean-field family happens to *contain* the true posterior
(both marginals are Gaussian, both are independent under
the analytical posterior). The fitted variational
parameters match the exact ones to 3+ decimal places.

**MCMC also recovers the exact posterior.** With 10,000
samples, the MCMC posterior mean and std match the
analytical ones to roughly the same precision.

**VI is two orders of magnitude faster than MCMC** here —
sub-millisecond for VI vs 47 ms for MCMC — and that gap
widens dramatically on larger problems. By the time you
have a million data points or a thousand parameters, VI
is the only practical option for exploratory work.

For non-conjugate models (the realistic case) VI is
typically only an approximation — the mean-field family
cannot represent the true posterior, and the fitted `q` is
biased. The bias is usually predictable (under-estimated
variance, mode-seeking) and acceptable for most downstream
tasks.

---

## The modern stack: reparameterisation and amortisation

Two extensions made VI dominant in deep learning:

**The reparameterisation trick** (Kingma & Welling, 2014;
Rezende et al, 2014). When `q` is Gaussian, we can sample
`θ ~ q(θ; μ, σ)` as `θ = μ + σ · ε` with `ε ~ N(0, 1)`. This
makes the sampling *differentiable* with respect to `μ` and
`σ`, which lets us backpropagate through ELBO estimates and
optimise the variational parameters with standard gradient
descent. The technical foundation of **Variational
Autoencoders**.

**Amortised inference.** Instead of fitting a separate `q`
for each data point, train a single *inference network* that
maps any input `x` to its variational parameters `(μ(x),
σ(x))`. Now inference is a forward pass through the
inference network — cheap, parametric, generalisable to new
data. This is exactly what a VAE encoder does: it amortises
the per-image variational inference into a learned
network.

These two ideas — reparameterisation gradients plus
amortised inference — turn classical VI into the engine
behind every modern deep generative model. VAEs, normalising
flows, deep Bayesian neural networks all use the same
machinery scaled up.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

VI's cost is much lower than MCMC's:

**Coordinate-ascent VI** for mean-field exponential families
is roughly `O(I · n · d)` per iteration where `I` is the
number of coordinate sweeps, `n` is the data size, and `d`
is the parameter dimension. Typically converges in 10–100
iterations.

**Stochastic VI (SVI)** processes mini-batches — `O(B · d)`
per gradient step where `B` is the batch size. Scales to
millions of data points like SGD.

**Amortised VI** (VAE-style) adds the cost of training the
inference network — but inference at test time is a single
network forward pass.

Compare to MCMC: per-iteration costs are similar but the
*number of iterations* differs by 100–10000×. VI converges
in tens to hundreds of iterations; MCMC needs tens of
thousands.

---

## Real-world ML and AI connections

**Variational Autoencoders.** The deep-learning workhorse
for unsupervised representation learning. Encoder = inference
network; decoder = likelihood model; ELBO is the training
objective. Used everywhere from image generation (early
VAE generation pre-GAN, now revived in latent-diffusion
models) to molecular design.

**Latent Dirichlet Allocation.** Variational EM is the
standard training algorithm for LDA at scale (Hoffman et al,
2013). Powers production topic modelling at every major
search engine and content recommendation system.

**Bayesian neural networks.** VI gives a tractable way to
fit posteriors over neural network weights — used for
uncertainty estimation in deep learning, especially in
safety-critical and out-of-distribution detection
applications. The "mean-field Gaussian VI" recipe is the
standard baseline; more sophisticated families (normalising
flows, structured posteriors) push the frontier.

**Probabilistic programming languages.** PyMC, Pyro, and
NumPyro all offer VI as an alternative to MCMC — useful when
MCMC is too slow.

**Normalising flows.** Combine VI with invertible neural
networks for richer variational families that can capture
correlations and non-Gaussian shapes. State-of-the-art deep
generative modelling.

**Causal discovery.** Variational methods are used to fit
latent-variable causal models, particularly the differentiable
DAG-learning frameworks introduced in the late 2010s.

---

## When NOT to use VI

**When you need calibrated uncertainty.** Mean-field VI
consistently underestimates posterior variance. For
applications where the *width* of the posterior matters,
MCMC or full-rank Gaussian VI are better.

**When the posterior is multimodal.** Standard VI is
*mode-seeking* — it concentrates on a single mode and
ignores the rest. Mixture variational families help but
need careful design.

**When you don't trust the variational family.** Mean-field
is wrong whenever parameters are correlated (most of the
time). Pick a richer family or use MCMC.

**When the model is small and conjugate.** Closed-form
posteriors exist; just compute them.

---

## What comes next

Part 4 of this track is **Probabilistic Programming** —
the systems (PyMC, Stan, Pyro, NumPyro, Turing.jl) that let
you write probabilistic models declaratively and have the
framework choose the right inference algorithm (HMC, NUTS,
VI, mean-field VI, normalising flows) automatically. The
shift is from "implement my own MCMC / VI for this specific
model" to "write the model, hit fit, see what happens".

After Probabilistic Programming the track closes with
**Causal Inference** — the family of methods that move
beyond statistical association to causal identification.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**variational_inference.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/05-bayesian-probabilistic-causal/03-variational-inference/variational_inference.py)

Run it with:

```bash
pip install numpy scipy
python variational_inference.py
```

It needs `numpy` and `scipy`. The script fits Bayesian
linear regression three ways — exact closed-form posterior,
mean-field VI by coordinate ascent on the ELBO, and MCMC
with Metropolis-Hastings — and verifies all three agree on
mean and std of the slope and intercept posteriors on a
conjugate Gaussian model. The headline insight worth
pinning to the wall: **VI turns Bayesian inference into
optimisation by minimising KL-divergence to the true
posterior within a tractable family; the optimisation
maximises the ELBO; the result is much faster than MCMC at
the cost of approximation, typically under-estimating
posterior variance because mean-field families cannot
capture parameter correlations**.

---

*This is Part 3 of the Bayesian, Probabilistic & Causal Methods track in the Algorithms in Python series. The companion script `variational_inference.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 2 of this track covered MCMC. Part 4 will look at Probabilistic Programming — the libraries that let you write models and pick inference engines declaratively.*
