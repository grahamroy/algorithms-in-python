# State-Space Models — The Unifying Formalism for Time Series

### *Algorithms in Python --- Time Series & Forecasting, Part 3*

---

Parts 1 and 2 built ARIMA and Exponential Smoothing — the two
historical workhorses of statistical forecasting. They look
different on the surface: ARIMA writes the current observation
as a linear combination of past observations and past noise;
ETS writes recurrence equations over level, trend, and seasonal
components. But underneath they are the *same kind of model*,
and the abstraction that unifies them is the **state-space
form**.

A state-space model has two equations:

```
State transition:    x_t = A · x_{t-1} + B · u_t + w_t
Observation:         y_t = C · x_t + D · u_t + v_t
```

The **state** `x_t` is a vector of latent variables that
evolves over time according to a linear update with process
noise `w_t`. The **observation** `y_t` is a linear function of
the current state plus measurement noise `v_t`. Optional
inputs `u_t` represent exogenous covariates.

This abstraction subsumes ARIMA, ETS, and a long list of other
time-series models. Set the state to "the last `p` observations
plus the last `q` noise terms" and you have ARIMA. Set the
state to "level, trend, season" and you have ETS. Set it to
something custom — "underlying trend plus a 12-period cycle
plus a 7-day cycle plus an irregular component" — and you have
a **structural time series** model.

The state-space view comes with two unique algorithmic powers:

- The **Kalman filter** — a recursive linear-Gaussian
  inference procedure that gives you the posterior distribution
  over the latent state at every time step, optimal updates as
  new data arrives, and forecasts with calibrated uncertainty.
- Handling of **missing data, irregular sampling, and
  structural breaks** that ARIMA and ETS struggle with by
  default.

This article builds state-space modelling from first principles.
We will write down the linear-Gaussian state-space form,
derive the Kalman filter as the optimal Bayesian update for
the latent state, walk through the **Unobserved Components
Model (UCM)** — the statsmodels formulation that decomposes a
series into explicit trend, seasonal, and cycle components,
fit UCM to the airline-passengers dataset, and finish with
how ARIMA and ETS fit inside the state-space framework as
special cases.

---

## The model

A linear Gaussian state-space model is fully specified by:

- **State equation**: `x_t = A · x_{t-1} + w_t`,
  `w_t ~ N(0, Q)`. The state transition matrix `A` describes
  how the latent state evolves over time; `Q` is the process
  noise covariance.
- **Observation equation**: `y_t = C · x_t + v_t`,
  `v_t ~ N(0, R)`. The observation matrix `C` projects the
  latent state to what we measure; `R` is the measurement
  noise covariance.

Given an initial state distribution `x_0 ~ N(μ_0, P_0)`, the
joint distribution of `(x_{1:T}, y_{1:T})` is fully determined
by `(A, C, Q, R, μ_0, P_0)`. Choose these matrices and you
have a specific model.

---

## The Kalman filter

Given observations `y_{1:t}`, what is the posterior over the
current state `x_t`? In a linear-Gaussian model the answer is
itself Gaussian, and the **Kalman filter** computes its mean
and covariance recursively in `O(d³)` per time step for a
dense transition matrix (where `d` is the state dimension);
structured/sparse transitions bring this down toward `O(d²)`.

Two steps per observation:

**Predict** (given `p(x_{t-1} | y_{1:t-1}) = N(μ_{t-1}, P_{t-1})`):

```
μ̄_t = A · μ_{t-1}
P̄_t = A · P_{t-1} · Aᵀ + Q
```

**Update** (after observing `y_t`):

```
K_t = P̄_t · Cᵀ · (C · P̄_t · Cᵀ + R)⁻¹     # Kalman gain
μ_t = μ̄_t + K_t · (y_t - C · μ̄_t)
P_t = (I - K_t · C) · P̄_t
```

The **Kalman gain** `K_t` is the optimal weighting between
prior prediction and new observation, depending on which has
lower variance.

Two related algorithms:

- The **Kalman smoother** runs the filter forward, then
  refines all state estimates by combining them with future
  observations.
- The **EM algorithm** alternates between running the smoother
  (E-step) and re-estimating `(A, C, Q, R)` (M-step) to learn
  unknown parameters.

Statsmodels' `KalmanFilter` and `MLEModel` implement both;
PyMC, Pyro, and Stan all support state-space models with full
Bayesian inference.

---

## Unobserved Components Models

The most common state-space formulation in applied work
decomposes a series into interpretable components:

```
y_t = level_t + season_t + cycle_t + irregular_t
```

Each component has its own dynamics:

- **Level + trend** (the *local linear trend*): the level is a
  random walk *with drift*, and the drift (slope) itself
  evolves stochastically:
  `level_t = level_{t-1} + slope_{t-1} + η_t`,
  `slope_t = slope_{t-1} + ζ_t`. The slope feeds the level; it
  is not added to `y_t` directly.
- **Season**: sum of seasonal indices that integrate to zero
  over one period, with their own noise.
- **Cycle**: a damped sinusoid for non-seasonal periodic
  effects (e.g. business cycle).
- **Irregular**: white noise.

Statsmodels' `UnobservedComponents` lets you toggle each
component on or off and fits everything by maximum likelihood
via the Kalman filter. The result decomposes the observed
series into its underlying drivers — you can plot each
component separately, ask "what would `y` look like with the
seasonal stripped out?", and get a forecast that propagates
each component forward according to its own dynamics.

This is the model class behind most central-bank
macroeconomic-forecasting work, the BSTS library (Bayesian
Structural Time Series — Google's open-source library used in
its CausalImpact package), and most modern
decomposition-based business forecasting.

---

## ARIMA and ETS as state-space models

The unification:

**ARIMA in state-space form**: stack the last `p` observations
into the state, plus the last `q` noise terms. The state
transition matrix encodes the AR coefficients; the
observation matrix picks off the current value. With this
representation, fitting ARIMA via the Kalman filter is
*exactly* the maximum-likelihood estimator, and it handles
missing observations naturally (skip the update step).

**ETS in state-space form**: state is `(level, trend, season_1,
..., season_m)`. The transition matrix implements the recurrence
equations from Part 2; the observation matrix is `[1, 1, 1,
0, ..., 0]` (current level + trend + current seasonal). The
Hyndman-Koehler-Snyder-Ord book derives the full mapping.

Practical implication: **statsmodels actually fits ARIMA and
ETS via the Kalman filter under the hood**. The classical
algebraic formulations are pedagogical; the production
implementations are state-space.

---

## A worked example

The companion script fits an **Unobserved Components Model**
to the airline-passengers dataset with local linear trend +
dummy-variable seasonality, and compares the forecast against
ARIMA, SARIMA, and Holt-Winters from the previous two
articles.

```
DEMO 1 --- UCM (local linear trend + seasonal) on airline data
  Components       : local linear trend + seasonal(12) + irregular
  Fit method       : Kalman filter MLE
  AIC              : 1011.43
  Test MAPE        : 15.77%
  Test RMSE        : 91.29
```

```
DEMO 2 --- Decomposition: each component plotted separately
  Final estimated level : 425.4
  Final estimated trend : -4.24 per month
  Seasonal pattern peaks in: Aug; troughs in: Nov
```

```
DEMO 3 --- Side-by-side with ARIMA / SARIMA / Holt-Winters
  Method                       AIC    Test MAPE    Test RMSE
  -----------------     ----------   ----------   ----------
  ARIMA(2, 1, 2)           1225.56        8.22%        55.22
  SARIMA(...)               901.05        2.96%        17.21
  Holt-Winters mult         633.69        2.21%        15.81
  UCM (this article)       1011.43       15.77%        91.29
```

Three observations.

**The decomposition is the win, not the raw accuracy.** UCM
gives explicit estimates of the final level (425), trend
(−4.24 passengers per month) and per-month seasonal indices —
all extractable as separate time series. You can plot each
component, ask "what would the series look like without
seasonality?", or run "what if?" scenarios on the level
alone.

**On this multiplicative-seasonal dataset UCM's
additive decomposition fits poorly.** The seasonal swing grows
with the level, but additive seasonality assumes a constant
amplitude. The result: a fitted trend that flips negative
trying to compensate, and MAPE of 16% — worse than even
plain ARIMA. The fix is to log-transform the series before
fitting (making the multiplicative structure additive on the
log scale), or to use a model class explicitly built for
multiplicative seasonality (Holt-Winters multiplicative).
This is exactly the modelling-choice trade-off the chapter
emphasises: pick the form that matches your data.

**The Kalman filter handles missing data for free.** If
one of the months were `NaN`, the state-space implementation
would just skip the update step and propagate uncertainty —
ARIMA and ETS need imputation upstream.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

State-space models trade interpretability and flexibility
for higher per-iteration cost:

**Kalman filter** is `O(d³)` per time step (dense transition)
where `d` is the
state dimension. For UCM with trend (2 states) + 11 seasonal
states the total state is small, sub-millisecond per step.

**Full forward filter pass** is `O(T · d²)`. For `T = 10⁴`
and `d = 20` that is a few seconds.

**Smoother** is another `O(T · d²)` backward pass.

**Maximum likelihood fitting** runs the filter at every
iteration of the optimiser — `O(I · T · d²)` total.

For large state dimensions (`d > 100`) the cost becomes
appreciable; the **square-root Kalman filter** and
**unscented Kalman filter** are numerical-stability and
non-linear extensions used in robotics and signal processing.

---

## Real-world ML and AI connections

**Central-bank macroeconomic forecasting.** The Federal
Reserve's FRB/US model, the ECB's models, and most national
central banks' forecasting pipelines are state-space
formulations — interpretable decompositions of GDP, inflation,
unemployment into structural components.

**BSTS (Bayesian Structural Time Series).** Google's
open-source library — used in CausalImpact for measuring
the effect of marketing campaigns and product changes — is
built on Bayesian state-space modelling with MCMC over the
component priors.

**Robotics and tracking.** Kalman filters are the foundational
algorithm of robotic state estimation — sensor fusion, SLAM,
target tracking, autonomous-vehicle perception. The state is
"where am I and what am I moving"; the observation equation
maps sensors to measurements.

**Signal processing.** GPS, radar, navigation systems —
all use Kalman filtering for noise-reduced state estimation.
Pre-dates time-series forecasting historically.

**Financial econometrics.** Stochastic volatility models,
term-structure models, dynamic factor models — virtually
every modern econometric time-series technique is a
state-space model.

**Epidemiology.** Infectious-disease compartmental models
(SIR, SEIR) are state-space models with the disease-state
variables as the latent state.

**Sensor data analysis.** IoT sensors, industrial monitoring,
seismic data — state-space models are routine.

The pattern: when you need to *decompose* a time series into
interpretable latent drivers, or when you have missing data or
exogenous covariates with non-trivial dynamics, state-space
is the right framework.

---

## When NOT to use state-space models

**When you don't need decomposition.** If you just want
forecasts and don't care about understanding the trend vs
seasonal vs cycle structure, ARIMA or ETS is faster and
simpler.

**When the model is genuinely non-linear.** Standard
Kalman filters assume linear-Gaussian dynamics. For
non-linearities use the Extended or Unscented Kalman filter,
or particle filters — substantially more complex.

**When you have very high state dimensions.** `O(d²)` per
step becomes prohibitive past `d > 1000`. Approximate
methods (ensemble Kalman filters, variational inference)
are needed.

**When you have a million parallel series.** Per-series UCM
fits don't share information. Deep-learning sequence models
trained globally can exploit cross-series structure that
state-space cannot.

---

## What comes next

Part 4 of the Time Series & Forecasting track is **Prophet**
— Facebook's open-source forecasting library that wraps a
state-space-style decomposition (trend + multiple
seasonalities + holiday effects) in a friendly API designed
for analysts who are not time-series specialists. Prophet
gave a generation of data scientists a one-line
`m.fit(df).predict(future)` API that beats ad-hoc Excel
forecasts and competes with hand-tuned ARIMA on many
business datasets.

After Prophet comes the Temporal Fusion Transformer (TFT) —
the modern deep-learning answer for forecasting at scale
across many related series.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**state_space.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/06-time-series-forecasting/03-state-space-models/state_space.py)

Run it with:

```bash
pip install numpy pandas statsmodels
python state_space.py
```

It needs `numpy`, `pandas`, and `statsmodels`. The script
fits an Unobserved Components Model with local linear trend
and dummy-variable seasonality to the airline-passengers
dataset, decomposes the series into its latent components,
and compares forecast accuracy against ARIMA, SARIMA, and
Holt-Winters from the previous two articles. The headline
insight worth pinning to the wall: **state-space models
write a time series as a linear-Gaussian latent state plus
an observation equation; the Kalman filter does optimal
Bayesian inference recursively (O(d³) per step for a dense
transition); ARIMA and
ETS are special cases of this framework, and the explicit
decomposition makes state-space the right tool when
interpretability matters as much as forecast accuracy**.

---

*This is Part 3 of the Time Series & Forecasting track in the Algorithms in Python series. The companion script `state_space.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 2 of this track covered Exponential Smoothing. Part 4 will look at Prophet — Facebook's automatic forecasting library that wraps a state-space-style decomposition in a friendly API.*
