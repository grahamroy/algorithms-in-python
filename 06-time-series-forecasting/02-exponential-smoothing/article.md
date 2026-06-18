# Exponential Smoothing — The Other Classical Forecasting Workhorse

### *Algorithms in Python --- Time Series & Forecasting, Part 2*

---

Part 1 built ARIMA — the autoregressive-moving-average model
that dominated time series forecasting from the 1970s into the
2000s. ARIMA is powerful, mathematically elegant, and on
clean stationary data after differencing it is often the best
linear forecaster you can fit.

**Exponential Smoothing** is the parallel tradition. The
underlying idea is older than ARIMA — Brown's "exponentially
weighted moving average" was used in operations research in
the 1950s — and the modern Holt-Winters formulation has been
the production default for retail and supply-chain forecasting
for half a century. Its appeal: a few simple recurrence
equations, no stationarity assumption, very few
hyperparameters, and accuracy on small-to-medium business
data that competes with and often beats ARIMA. The M3, M4,
and M5 forecasting competitions all featured ETS variants in
their top-10 simple methods.

The 2008 Hyndman-Koehler-Snyder-Ord book *Forecasting with
Exponential Smoothing* unified the family into a state-space
framework (ETS: Error, Trend, Seasonality) and put it on a
rigorous statistical footing. The R `forecast` package and
Python's `statsmodels.tsa.holtwinters` are the standard
implementations.

This article builds exponential smoothing from first
principles. We will derive **simple exponential smoothing**
(no trend, no seasonality), extend it to **Holt's linear
method** (trend), then to **Holt-Winters** (trend +
seasonality, additive or multiplicative), describe the **ETS
state-space framework** that unifies them, fit several
variants on the airline-passengers dataset, and finish with a
side-by-side comparison against the SARIMA forecast from
Part 1.

---

## Simple exponential smoothing

The simplest case: a series with no trend, no seasonality,
just noisy observations around a level that may drift slowly.
The forecast is a weighted average of the previous forecast
and the latest observation, with the weight `α` (the
smoothing parameter, between 0 and 1) controlling how
responsive the forecast is to new data:

```
ŷ_{t+1} = α · y_t + (1 - α) · ŷ_t
```

Equivalently, expanding the recurrence:

```
ŷ_{t+1} = α · y_t + α(1-α) · y_{t-1} + α(1-α)² · y_{t-2} + ...
```

The forecast is a weighted sum of *all* past observations,
with exponentially decaying weights. Recent observations
matter most; old ones fade smoothly. `α = 0.1` is very
smooth (long memory); `α = 0.9` is very reactive (short
memory). The smoothing parameter is typically fit by maximum
likelihood — pick the `α` that minimises the sum of squared
forecast errors on the training data.

Simple ES is the right baseline for a level-only series like
a sensor reading or a stable inventory count.

---

## Holt's linear method (trend)

Add a second equation tracking the trend:

```
Level:      ℓ_t = α · y_t + (1 - α) · (ℓ_{t-1} + b_{t-1})
Trend:      b_t = β · (ℓ_t - ℓ_{t-1}) + (1 - β) · b_{t-1}
Forecast:   ŷ_{t+h} = ℓ_t + h · b_t
```

The level `ℓ_t` is the same exponentially-smoothed estimate
as before, except adjusted by the previous trend. The trend
`b_t` is itself exponentially smoothed, with its own
parameter `β`. The `h`-step-ahead forecast is the current
level plus `h` increments of the current trend.

Holt's method handles series with a constant or slowly
changing slope. There is a **damped trend** variant that
multiplies the trend by a damping factor `φ < 1` per step —
useful when extrapolating a long-running trend indefinitely
would be unrealistic.

---

## Holt-Winters (trend + seasonality)

Add a third equation for seasonality. Two variants depending
on whether the seasonal effect is **additive** (fixed
amplitude regardless of level) or **multiplicative**
(amplitude scales with level):

```
# Multiplicative seasonality (level scales the season)
Level:      ℓ_t = α · (y_t / s_{t-m}) + (1 - α) · (ℓ_{t-1} + b_{t-1})
Trend:      b_t = β · (ℓ_t - ℓ_{t-1}) + (1 - β) · b_{t-1}
Season:     s_t = γ · (y_t / ℓ_t) + (1 - γ) · s_{t-m}
Forecast:   ŷ_{t+h} = (ℓ_t + h · b_t) · s_{t+h-m}
```

`m` is the seasonal period (12 for monthly with annual
seasonality, 7 for daily with weekly, 24 for hourly with
daily, etc.). Three smoothing parameters now: `α`
(level), `β` (trend), `γ` (seasonality).

Multiplicative is the right choice for the airline-passengers
dataset, where the December peak grows in absolute size as
total passenger volume grows.

Additive seasonality (just replace the
divisions/multiplications above with subtractions/additions) is right when
the seasonal swing stays constant regardless of level — e.g.
electricity demand where the summer-winter difference is a
fixed number of MWh.

---

## The ETS state-space framework

Hyndman et al's contribution: every variant — simple ES,
Holt, Holt-Winters, damped versions, etc. — can be expressed
in a **state-space form** with three slots:

- **Error**: additive (A) or multiplicative (M)?
- **Trend**: none (N), additive (A), additive damped (Ad)?
- **Seasonality**: none (N), additive (A), multiplicative (M)?

A specific model is named **ETS(E, T, S)**: `ETS(A, N, N)` is
simple exponential smoothing; `ETS(A, A, A)` is Holt-Winters
additive; `ETS(M, A, M)` is Holt-Winters multiplicative (the
classical airline-passengers model).

The unified framework is what lets R's `forecast::ets()`
automatically pick the best `(E, T, S)` combination by AIC —
analogous to `auto_arima` for the ARIMA family. statsmodels'
`ETSModel` has no built-in auto-selection, so the companion
script hand-rolls the AIC search over a handful of candidate
combinations.

---

## A worked example

The companion script fits several exponential smoothing
variants on the airline-passengers dataset and compares
against the SARIMA model from Part 1.

```
DEMO 1 --- Simple exponential smoothing
  Smoothing parameter α : auto-fit
  AIC               : 912.38
  Test MAPE         : 14.25%
  Test RMSE         : 102.98
```

```
DEMO 2 --- Holt-Winters multiplicative (trend + seasonality)
  Seasonal period   : 12
  AIC               : 633.69
  Test MAPE         : 2.21%
  Test RMSE         : 15.81
```

```
DEMO 3 --- ETS auto-selection across candidate (E, T, S) combinations
  Best model         : ETS(M, A, M)
  AIC                : 967.98
  Test MAPE          : 4.28%
  Test RMSE          : 26.06
```

```
DEMO 4 --- Side-by-side with SARIMA from Part 1
  Method                  AIC      Test MAPE    Test RMSE
  -----------------    -------    ----------   ----------
  Simple ES             912.38       14.25%       102.98
  Holt-Winters mult     633.69        2.21%        15.81
  ETS(M, A, M) auto     967.98        4.28%        26.06
  SARIMA (Part 1)       901.05        2.96%        17.21
```

Three observations.

**Simple ES is hopeless on seasonal data.** Without a
seasonal component, the forecasts ignore the annual cycle and
the MAPE is 14% — far worse than the seasonally-aware
variants. The first lesson of ETS: pick the right family
member for the data.

**Holt-Winters multiplicative actually beats SARIMA on this
dataset.** MAPE of 2.21% (vs SARIMA's 2.96%) and a lower
AIC (634 vs 901) — and that on the canonical seasonal
benchmark airline-passengers data. ETS is not always the
runner-up; on data where the multiplicative seasonal
structure is genuinely right, it can win outright.

**The auto-ETS selection is interesting.** AIC picked
`ETS(M, A, M)` — multiplicative errors, additive trend,
multiplicative seasonality — but the MAPE is worse than the
manually-specified `ExponentialSmoothing(trend='add',
seasonal='mul')` because the two functions in statsmodels
differ in initialisation strategy. On most datasets they
agree; here the manual fit happens to be slightly better. In
production fit both and keep the lower MAPE.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

ETS is even cheaper than ARIMA:

**Fitting** is `O(I · n)` where `I` is the maximum-likelihood
optimiser iterations. The recurrence equations are linear in
`n`; only the smoothing parameters need to be fit.

**Forecasting** is `O(h)` — each step is a constant-time
recurrence application.

**Memory** is `O(n)` for the data and `O(1)` for the model
state.

For sub-day data (hourly, minute-level) with multiple
seasonalities, the **TBATS** extension (Trigonometric Box-Cox
ARMA Trend Seasonal) handles up to 4 simultaneous seasonal
periods. Cost grows with the seasonal periods but remains
practical.

---

## Real-world ML and AI connections

**Retail demand forecasting.** Holt-Winters has been the
default at most major retailers for SKU-level demand
forecasting for decades. Walmart, Target, Tesco, Carrefour —
all run hundreds of millions of ETS-style forecasts per day
for inventory replenishment.

**Supply chain and operations research.** ETS is the textbook
inventory-planning forecaster. Production schedules, raw
material orders, capacity planning — all powered by
exponential smoothing in most enterprise resource planning
(ERP) software.

**Web analytics.** Google Analytics, Adobe Analytics, and
most web-analytics products use ETS variants under the hood
to compute "expected" baseline values and flag deviations.

**Smart-meter forecasting.** Hourly or sub-hourly electricity
consumption — TBATS or its multi-seasonal ETS cousins handle
the daily + weekly + yearly seasonalities.

**Operations / SRE alerting.** Datadog, New Relic, and most
APM tools fit ETS to each tracked metric, generate confidence
intervals, and alert when actuals fall outside the
intervals.

**M4 / M5 forecasting competitions.** ETS is consistently
in the top 5 simple methods. Modern ensembles often
include ETS as a component.

---

## When NOT to use exponential smoothing

**When you have exogenous variables you want to incorporate.**
ETS has no native support for covariates. SARIMAX or
regression-with-ARIMA-errors are the right tools.

**When the series has multiple complex seasonalities.** TBATS
handles a few, but for very rich multi-seasonal data (energy
demand with daily + weekly + holiday effects + weather) you
want Prophet (Part 4) or a deep learning approach.

**When the underlying generative process is genuinely
non-linear.** Logistic growth, threshold effects, regime
changes — these need Prophet's logistic trend, or a neural
forecaster.

**When you have very long history and want to leverage it.**
ETS' exponentially-decaying memory throws away old data
quickly. Long-context attention-based models can exploit
patterns from years ago.

**When forecast accuracy matters more than transparency.**
On clean tabular forecasting benchmarks deep learning often
wins by 5–15%. ETS gives you transparency and speed at a
small accuracy cost.

---

## What comes next

Part 3 of the Time Series & Forecasting track is **State-Space
Models** — the more general formalism that subsumes both
ARIMA and ETS. State-space models let you decompose a series
into latent components (trend, seasonality, noise) more
explicitly than either ARIMA or ETS, and they handle
irregularly-sampled data, missing observations, and structural
changes that the two simpler families struggle with.

After state-space comes Prophet (Facebook's automatic
forecasting library) and the Temporal Fusion Transformer (the
modern deep-learning answer).

---

## The complete code

The full script is on GitHub — grab it and run it:

[**exponential_smoothing.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/06-time-series-forecasting/02-exponential-smoothing/exponential_smoothing.py)

Run it with:

```bash
pip install numpy pandas statsmodels
python exponential_smoothing.py
```

It needs `numpy`, `pandas`, and `statsmodels`. The script
fits simple exponential smoothing, Holt-Winters
multiplicative, and ETS auto-selected models on the airline
passengers dataset and reports MAPE / RMSE alongside the
SARIMA reference from Part 1. The headline insight worth
pinning to the wall: **exponential smoothing replaces ARIMA's
algebraic AR/MA formulation with simple recurrence equations
over level, trend, and seasonal components; the ETS
state-space framework unifies the family; and on many
practical business datasets Holt-Winters and SARIMA give
indistinguishable accuracy — fit both, pick the winner**.

---

*This is Part 2 of the Time Series & Forecasting track in the Algorithms in Python series. The companion script `exponential_smoothing.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 1 of this track covered ARIMA. Part 3 will look at State-Space Models — the unifying formalism that subsumes both ARIMA and ETS.*
