# Causal Inference — Identifying Causes from Observational Data

### *Algorithms in Python --- Bayesian, Probabilistic & Causal Methods, Part 5*

---

The previous four articles in this track were all about
inference *over parameters* — given a probabilistic model and
some data, what is the posterior distribution over the model's
unknowns? Gaussian Processes, MCMC, Variational Inference, and
Probabilistic Programming all share this framing.

**Causal Inference** asks a fundamentally different question.
Not "given that I observed `X = x`, what is `Y`?" but "if I
*intervene* and *set* `X = x`, what will `Y` be?" The shift
from observation to intervention is the conceptual bridge
between statistical learning and decision-making. Every A/B
test, every drug trial, every policy evaluation, every "what
would have happened if" counterfactual rests on this
distinction.

The framework is older than machine learning. Donald Rubin's
**potential outcomes** model goes back to the 1970s; Judea
Pearl's **do-calculus** and causal graphical models were
formalised in the 1990s. Both frameworks describe the same
underlying mathematics, and the modern causal-machine-learning
stack (causal forests, double machine learning, debiased
regression) sits on top of both.

This article builds causal inference from first principles. We
will explain why "correlation ≠ causation" is a deep statement
about the limits of observational data, formalise the
**potential outcomes framework** and the **fundamental problem
of causal inference**, walk through Pearl's **do-calculus** and
the **backdoor criterion** for identifying causal effects,
implement two of the standard adjustment methods —
**regression adjustment** and **inverse-propensity weighting**
— on a confounded synthetic dataset, show they recover the
true treatment effect while the naive association doesn't, and
finish with the modern causal-ML stack (causal forests,
DoWhy, EconML) and where causal inference is doing real work
in production.

---

## Why correlation isn't enough

The classic example: ice cream sales correlate with drowning
deaths. Eat more ice cream → die from drowning? Obviously not.
Both are driven by a third variable — hot weather — that we
haven't accounted for. The correlation is real; the causal
interpretation is wrong.

The general form: a **confounder** `Z` causes both `X` (the
putative cause) and `Y` (the putative effect). If we condition
on `Z`, the correlation between `X` and `Y` disappears. If we
ignore `Z`, the correlation is spurious.

```
        Z (weather)
       / \
      ↓   ↓
     X     Y
   (ice    (drowning)
    cream)
```

Real-world variants are everywhere:

- Higher hospital admissions correlate with worse patient
  outcomes (sicker patients go to hospital).
- Education correlates with income (and with intelligence,
  family background, motivation — the confounders).
- Ad clicks correlate with sales (showing ads to people more
  likely to buy anyway).

In every case, the observational correlation between `X` and
`Y` is an unreliable guide to what would happen if we
*intervened* on `X` (running an ad campaign vs not, sending
patients to hospital vs not, requiring more education vs not).

---

## The potential outcomes framework

Rubin's framework gives a clean way to define causal effects.
For each unit `i` (a patient, a customer, a country), define:

- `Y_i(1)` — the outcome if unit `i` were treated.
- `Y_i(0)` — the outcome if unit `i` were *not* treated.

The **individual treatment effect** is `Y_i(1) - Y_i(0)`. The
**average treatment effect (ATE)** is its average across the
population:

```
ATE = E[Y(1) - Y(0)]
```

This is what we want to estimate.

The **fundamental problem of causal inference**: for any given
unit, we observe *either* `Y_i(1)` *or* `Y_i(0)`, never both.
You took the drug or you didn't; you saw the ad or you didn't.
The counterfactual is genuinely missing data.

Causal inference is, formally, the problem of imputing the
missing counterfactual outcomes — using *other* units who
*did* receive the treatment we want to estimate as
stand-ins for the missing potential outcomes of the
*untreated* units, and vice versa.

For this imputation to be valid we need three assumptions:

**SUTVA (Stable Unit Treatment Value Assumption)**: one unit's
treatment doesn't affect another unit's outcome. Vaccinating
me doesn't change your potential outcomes. (This is violated
by network effects, herd immunity, market spillovers — and
fixing it is hard.)

**Unconfoundedness (a.k.a. ignorability)**: conditional on
observed covariates `Z`, treatment assignment is independent
of potential outcomes. Once you control for everything that
predicts both treatment and outcome, the residual
treatment-outcome relationship is causal. This is the big
assumption — and it's almost never literally true; the
question is how close to true it is.

**Positivity**: every unit has a non-zero probability of
receiving each treatment, conditional on covariates. If
nobody with a particular profile ever gets treated, you can't
estimate the effect for that subpopulation.

The three together — SUTVA, unconfoundedness, positivity —
let us identify the ATE from observational data.

---

## Pearl's do-calculus

The parallel framework: causal **graphical models** with
explicit `do()` operators.

Write the joint distribution as a causal DAG (directed
acyclic graph) where edges represent direct causal effects.
Then distinguish:

- `P(Y | X = x)` — the *observational* conditional. The
  distribution of `Y` among units that happened to have
  `X = x`.
- `P(Y | do(X = x))` — the *interventional* distribution.
  The distribution of `Y` if we *force* `X` to equal `x`,
  cutting it off from its causes.

In a confounded DAG these are different. In a randomised
experiment they're equal (because randomisation severs
the backdoor paths from `X` to `Y`).

**Pearl's backdoor criterion** gives a procedure: a set `Z`
of variables suffices to identify the causal effect of `X`
on `Y` if (a) `Z` blocks every backdoor path from `X` to `Y`,
and (b) `Z` contains no descendant of `X`. Adjust for `Z`
and the resulting estimate equals `P(Y | do(X))`.

In our ice-cream example, `Z = {weather}` is a backdoor
adjustment set: condition on weather, and ice-cream
consumption is no longer associated with drowning. The
backdoor criterion formalises what most working analysts
call "controlling for confounders".

---

## Three methods for estimating causal effects

Given a dataset with treatment `T ∈ {0, 1}`, outcome `Y`,
and covariates `Z` (assumed sufficient for
unconfoundedness), three workhorse methods:

### Regression adjustment

Fit `E[Y | T, Z]` with a regression — linear, gradient
boosting, neural net, whatever. The ATE estimate:

```
ATE_hat = mean( m(T=1, Z_i) - m(T=0, Z_i) )   over all i
```

For each unit, predict what `Y` would be under both
treatment and control using the fitted model, take the
difference, average over the dataset. Simple, interpretable,
and asymptotically unbiased if the model is correctly
specified.

### Inverse-propensity weighting (IPW)

Estimate the **propensity score** `e(Z) = P(T = 1 | Z)` — the
probability a unit is treated, given covariates. Then weight
observations to make the treated and control groups look
balanced:

```
ATE_hat = mean( (T_i / e(Z_i)) * Y_i  -  ((1 - T_i) / (1 - e(Z_i))) * Y_i )
```

Treated units with low propensity (unlikely to be treated)
get high weight; untreated units with high propensity get
high weight. The reweighted samples mimic what a randomised
experiment would have produced. Asymptotically unbiased if
the propensity model is correct.

### Doubly robust estimation

Combine both: use regression adjustment as the primary
estimator, then apply IPW as a correction term. If *either*
the outcome model or the propensity model is correct, the
ATE estimate is unbiased. This is "double robustness" — and
it's the foundation of the modern causal-ML stack.

---

## A worked example

The companion script simulates confounded observational data:
500 units with one binary treatment `T`, two confounders
`Z_1, Z_2`, an outcome `Y` that depends on both, and a true
ATE of 2.0. We then estimate the ATE three ways:

```
DEMO --- Causal effect estimation on confounded synthetic data
  Sample size : 500 units
  True ATE    : 2.00
  Treatment   : binary, propensity depends on Z_1, Z_2
  Outcome     : Y = 1.0 + 2.0 * T + 1.5 * Z_1 + 1.2 * Z_2 + noise

  Treated fraction : 0.36

  Method                                   Estimated ATE    Bias
  ------------------------------------     -------------   -----
  Naive (simple difference in means)                4.08   +2.08
  Regression adjustment (linear)                    1.93   -0.07
  Inverse propensity weighting                      2.14   +0.14
```

Three observations.

**The naive estimate is dramatically biased.** The
difference-in-means estimate is 4.08 — more than double the
true effect — because units with high `Z_1` and `Z_2` are both
more likely to be treated AND more likely to have high `Y`.
The naive estimate confuses correlation with causation.

**Regression adjustment recovers the true effect.** With a
correctly-specified linear outcome model, the regression
estimate (1.93) is within 0.07 of the true 2.00.

**IPW also recovers the true effect.** The IPW estimate
(2.14) is within 0.14 — slightly noisier than regression
adjustment on this dataset, but still well within sampling
error of the true value. The two methods agree because the
simulation honours both modelling assumptions.

On real data only one of the two models is usually
well-specified — which is why doubly-robust estimators
(combining both) are the production default. Both methods
fail catastrophically if a key confounder is *missing* from
the data; no statistical technique recovers from unmeasured
confounding.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Causal estimation cost is dominated by the underlying
machine-learning method:

**Regression adjustment** is the cost of training the chosen
regressor — `O(n · d²)` for linear, `O(n · d · T · depth)` for
gradient boosting, etc.

**IPW** trains a propensity classifier (same cost as
regression), then does an `O(n)` weighted average. The
expensive part is hyperparameter-tuning the propensity
model to get well-calibrated probabilities.

**Doubly-robust estimation** trains both models. Modern
double/debiased ML (Chernozhukov et al, 2018) uses
cross-fitting to avoid overfitting bias — split the data,
fit each model on one half, score on the other. Cost is
constant-factor more than a single model.

**Causal forests** (Athey & Wager, 2019) fit a random
forest with a specific splitting criterion that targets
*heterogeneous* treatment effects (different effects for
different subpopulations). Same `O(n · log n)` scaling as
random forests.

**Synthetic controls** (Abadie et al) construct a weighted
combination of untreated units as a counterfactual for each
treated unit. Quadratic in the number of units for naive
implementations, cheaper with approximations.

For modern tabular causal datasets (`n < 10⁶`, `d < 100`),
all of these run in seconds to minutes.

---

## The modern causal-ML stack

**DoWhy** (Microsoft). Python library that wraps the
end-to-end causal inference workflow: specify a causal DAG,
identify the estimand using do-calculus, estimate it with a
chosen method, run refutation tests. The standard for
introductory and applied causal analysis in Python.

**EconML** (Microsoft). Heavier on the ML side. Implements
double machine learning, causal forests, heterogeneous
treatment effect estimation, instrumental variables.
Production-grade.

**CausalML** (Uber). Similar scope, emphasis on uplift
modelling for marketing — "which users will *change*
behaviour if we send them a coupon?" rather than "which
users will buy?".

**PyMC and Bayesian causal models**. Express the causal
structure as a probabilistic program, do full Bayesian
inference over the treatment effect. The right tool when
prior knowledge matters or when uncertainty quantification
is critical.

**Synthetic controls (causalimpact, causalpy)**. The
standard for evaluating one-time interventions where a
randomised control is impossible (a country's policy
change, a regulatory event affecting one company).

The pattern: pick your causal estimand → write down the
assumptions you're willing to make → use the corresponding
library to estimate it → run refutation/sensitivity
analyses to check robustness.

---

## Real-world ML and AI connections

**A/B testing at scale.** Every major tech company runs
hundreds of A/B tests per week. The basic infrastructure
is randomised assignment (which makes causal estimation
trivial), but advanced topics — interference between users,
long-term vs short-term effects, heterogeneous effects across
subpopulations — are causal-inference territory and lean
heavily on EconML / DoWhy / CausalML.

**Policy evaluation in economics.** Synthetic controls,
difference-in-differences, regression discontinuity — these
econometric methods are the foundation of modern empirical
economics. The 2021 Nobel Prize in Economics went to Card,
Angrist, and Imbens for exactly this work.

**Medical treatment effects.** Comparative effectiveness
research — does drug A actually work better than drug B in
practice (not just in the trial)? — is almost entirely
observational causal inference, typically with propensity
score methods.

**Causal recommendation systems.** Recommend an item *because
it will change the user's behaviour*, not because the user
would have bought it anyway. Uber's CausalML and Spotify's
work on counterfactual recommendation are examples.

**ML interpretability.** What would the prediction be if I
changed this feature? The counterfactual question is causal,
not statistical, and modern interpretability tools (SHAP, LIME)
are increasingly framed in causal terms.

**Marketing attribution.** Which channel actually drove the
sale? Naive last-touch attribution is associational and
wrong; causal attribution methods give defensible answers.

**Personalised medicine.** Estimate the treatment effect for
*this specific patient* given their covariates. Causal
forests / meta-learners (S-learner, T-learner, X-learner)
are the standard tools.

---

## When NOT to use causal inference

**When you only need prediction.** Predict `Y` from `X` — no
intervention, no counterfactual? Use standard supervised
learning. Causal inference asks a strictly harder question
and pays for it in stronger assumptions.

**When randomisation is feasible.** An RCT side-steps
unconfoundedness entirely. If you can randomise the
treatment, do it; analyse with a simple difference in means;
go home.

**When critical confounders are unmeasured.** No method
saves you from unmeasured confounding. Sensitivity analyses
(Rosenbaum bounds, E-values) quantify the *vulnerability*
of your estimate to plausible hidden confounders — they
don't fix the problem.

**When the data is observational and noisy.** Causal
estimates with 95% CIs spanning zero are usually because
the dataset cannot identify the effect. Get more data, or
accept that the question is unanswerable.

**When the question is really about prediction, not
intervention.** "What is the expected income of someone with
a master's degree?" is associational. "What would my income
be if I got a master's degree?" is causal. The first is
descriptive analytics; the second is a hard problem.

---

## What comes next

This is the final article in the **Bayesian, Probabilistic
& Causal Methods** track. Five articles: Gaussian Processes
(closed-form Bayesian regression), MCMC (sampling-based
inference), Variational Inference (optimisation-based
inference), Probabilistic Programming (the libraries that
let you specify models declaratively), and Causal Inference
(moving beyond association to identifying causes).

The next track is **Time Series & Forecasting**, opening
with ARIMA — the classical, decades-old statistical
forecasting method that remains the right baseline for any
new time series problem. After ARIMA the track covers
exponential smoothing, state-space models, Prophet, and
modern Temporal Fusion Transformers.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**causal_inference.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/05-bayesian-probabilistic-causal/05-causal-inference/causal_inference.py)

Run it with:

```bash
pip install numpy scikit-learn
python causal_inference.py
```

It needs `numpy` and `scikit-learn`. The script simulates a
confounded observational dataset (500 units, true ATE of 2.0,
treatment assignment depending on two confounders), estimates
the average treatment effect three ways — naive
difference-in-means (biased), regression adjustment
(unbiased given correct outcome model), and
inverse-propensity weighting (unbiased given correct propensity
model) — and shows the naive estimate is dramatically wrong
while the two adjustment methods recover the true effect to
within statistical noise. The headline insight worth pinning
to the wall: **causal inference is the problem of imputing
counterfactuals you cannot observe; observational
identification rests on unconfoundedness (no important
omitted confounder); regression adjustment and
inverse-propensity weighting are the workhorse methods, and
doubly-robust estimators (combining both) are the modern
production default**.

---

*This is Part 5 of the Bayesian, Probabilistic & Causal Methods track in the Algorithms in Python series, and the final article of the track. The companion script `causal_inference.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 4 of this track covered Probabilistic Programming. The next track — Time Series & Forecasting — opens with ARIMA.*
