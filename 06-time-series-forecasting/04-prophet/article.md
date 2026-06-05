# Prophet — Facebook's Decompose-and-Forecast for Everyone

### *Algorithms in Python --- Time Series & Forecasting, Part 4*

---

ARIMA (Part 1), exponential smoothing (Part 2), and state-space
models (Part 3) are powerful and well-understood, but they
share a friction: tuning them well requires real time-series
expertise. ARIMA's `(p, d, q)` choices, ETS's `(E, T, S)`
configurations, UCM's component selection — each demands
domain knowledge that most data scientists at most companies
don't have time to develop for every series they need to
forecast.

**Prophet** (Taylor & Letham, Facebook, 2017) solved this
problem by packaging a state-space-style decomposition behind
a deliberately friendly API:

```python
from prophet import Prophet
m = Prophet()
m.fit(df)            # df has columns 'ds' (date) and 'y' (value)
forecast = m.predict(future_df)
```

Three lines. Auto-selected trend break points, automatic
detection of daily/weekly/yearly seasonality, optional
holiday effects, native support for missing data and
irregular sampling, full uncertainty intervals, and a plot
method that produces publication-grade charts. For the
non-time-series-specialist analyst, Prophet replaced "spend
two weeks learning ARIMA" with "fit and inspect the result".

The internal model is roughly:

```
y(t) = trend(t) + seasonality(t) + holidays(t) + noise
```

with piecewise-linear or logistic-growth trend, Fourier-series
seasonalities at multiple periods, and an indicator for each
configured holiday. Parameters are fit by maximum a posteriori
(or full Bayesian MCMC) via the Stan probabilistic
programming language. The mathematics is more state-space than
ARIMA-style, but the *experience* is closer to "scikit-learn
for time series".

This article builds Prophet from first principles. We will
walk through its three model components, describe the
changepoint detection that handles trend changes, fit it to
the airline-passengers dataset and compare against the
previous three articles' methods, then finish with the
critique it has accumulated — Prophet is now ten years old
and has well-known failure modes alongside its real
strengths.

---

## The model

Prophet decomposes `y(t)` into three additive (or
multiplicative) components plus noise.

### Trend

Two flavours.

**Piecewise linear** — the default, suitable for most data
without a saturating asymptote. The model places `S`
candidate **changepoints** (default 25, evenly spaced over
the first 80% of the history) where the slope can change.
The actual slopes between changepoints are fit by Bayesian
inference with a sparsity-inducing Laplace prior — most
changepoints end up with zero adjustment, and only the
genuinely meaningful trend changes are kept.

**Logistic growth** — for series with a known capacity (web
traffic on a planet of N people, market share, sales
penetration into a fixed addressable market). User supplies
the capacity `C`; Prophet fits a logistic curve whose slope
changes at the same automatic changepoints.

### Seasonality

Each seasonal period is modelled as a **Fourier series** —
sums of sines and cosines at the period's harmonics. For
yearly seasonality (default 10 Fourier terms), weekly (3
terms), daily (4 terms). User can add custom seasonalities
(monthly, quarterly, hourly).

Fourier-series seasonality is more flexible than the
single-cycle "seasonal index" approach of ETS or SARIMA — it
can represent smoothly-varying seasonal shapes. It is also
more parsimonious than dummy-coded month effects.

### Holidays and custom regressors

User-provided. Pass Prophet a dataframe of holidays (date +
event name + optional `lower_window` / `upper_window` for
multi-day effects) and the model fits a per-holiday additive
effect. Same for custom regressors — promotions, weather,
anything observed.

### Putting it together

The full model:

```
y(t) = g(t) + s(t) + h(t) + ε_t
```

where `g(t)` is the piecewise trend, `s(t)` the sum of all
seasonalities (Fourier-series), `h(t)` the holiday effects.
Inference is **MAP estimation** (default) or full **MCMC via
Stan** (set `mcmc_samples` to a positive number — much
slower, gives proper uncertainty).

---

## Why is it so easy?

Three deliberate API choices set Prophet apart:

**Sensible defaults that mostly work.** The default Prophet
fit handles most business datasets reasonably well without
any tuning — yearly + weekly seasonality, automatic changepoint
detection, 25 candidate changepoints, default smoothing
priors. Compare against ARIMA where the wrong `(p, d, q)`
gives terrible results.

**Native handling of missing data.** Prophet doesn't care if
your timestamps are irregular or have gaps; the
additive-decomposition formulation works as long as
observations exist at any time `t`.

**Interpretable component plots.** `m.plot_components(forecast)`
shows the trend, each seasonality, and the holiday effects as
separate panels — instant model diagnostics without needing
to know what to look for.

**Forgiving of edge cases.** Negative values, weird outliers,
sudden trend changes — Prophet returns *something* sensible
where ARIMA might silently produce garbage.

The downside, predictable from these design choices:
**Prophet's defaults are decent, but the model is hard to
tune for last-mile accuracy**. The 2024 Hyndman critique
showed Prophet underperforms simpler methods on many
benchmarks. The right framing: Prophet is the
no-effort baseline, not the production accuracy champion.

---

## A worked example

The companion script fits Prophet on the airline-passengers
dataset and compares against the methods from the previous
three articles.

```
DEMO 1 --- Prophet on airline passengers
  Fit time          : 0.45 s
  Components fitted : trend (linear) + yearly seasonality
  Test MAPE         : 6.61%
  Test RMSE         : 43.07
```

```
DEMO 2 --- Component decomposition
  Estimated trend at end of training : 444.2
  Yearly peak month: Jul
  Yearly trough month: Nov
```

```
DEMO 3 --- Side-by-side with ARIMA / SARIMA / Holt-Winters / UCM
  Method                       AIC    Test MAPE    Test RMSE
  -----------------     ----------   ----------   ----------
  ARIMA(2, 1, 2)           1225.56        8.22%        55.22
  SARIMA(...)               901.05        2.96%        17.21
  Holt-Winters mult         633.69        2.21%        15.81
  UCM (Part 3)             1011.43       15.77%        91.29
  Prophet                        —        6.61%        43.07
```

Three observations.

**Prophet is fast and easy.** The whole fit + forecast
pipeline runs in under a second with sensible defaults. The
output includes the decomposition into trend + seasonality
and uncertainty intervals — for free.

**Prophet is mid-pack on accuracy.** MAPE of 6.6% is better
than UCM, worse than the classical methods. On this dataset
SARIMA and Holt-Winters are clearly preferred if accuracy is
the only goal.

**Prophet's value is the workflow, not the headline number.**
On a corpus of 100 business time series where you don't have
time to hand-tune each one, Prophet's "one API, sensible
defaults" wins — even if the per-series accuracy is 1–2%
worse than a hand-tuned SARIMA, the *total* labour saved is
enormous.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Prophet's cost is dominated by Stan's MAP estimation:

**MAP fitting** is `O(L-BFGS iters · n · features)`. Features
include trend coefficients (one per changepoint + base), 2 ×
Fourier order per seasonality (10 yearly → 20 features), plus
one per holiday. Total typically <100 features.

**Full Bayesian fitting** (`mcmc_samples > 0`) runs Stan's
NUTS sampler, several orders of magnitude slower than MAP but
gives proper posterior uncertainty.

**Forecasting** is `O(h · features)` — evaluate the fitted
function at each future time step.

**Memory** is `O(n + features²)` during fitting.

The Stan backend can be brittle on Windows; the alternative
**cmdstanpy** backend and the **NeuralProphet** rewrite (which
replaces Stan with PyTorch) address some of those issues.

---

## Real-world ML and AI connections

**Internal forecasting at Meta.** Prophet was open-sourced
because Meta was using it at scale internally — site
traffic, ad performance, infrastructure capacity — across
thousands of teams. The library is still the official
Meta forecasting tool.

**Data-science-team default at most companies.** When a
non-time-series specialist needs to forecast something at a
business that doesn't have a dedicated forecasting team,
Prophet is the most-likely answer. Snowflake, Databricks,
and most cloud-data-platform vendors integrate Prophet.

**M5 forecasting competition.** Prophet was a baseline that
many top entries had to beat. Most ensembles included a
Prophet component, even if the headline method was a
gradient-boosted tree.

**Capacity planning and SRE.** Predicting server load,
network traffic, queue depth — Prophet is the standard tool
for "I need a forecast, I don't want to think about it".

**Marketing campaign analysis.** When the business says "what
would the metric have done absent this launch?", Prophet's
ability to add custom regressors and changepoints around
launch dates makes it a natural counterfactual baseline.
Many practitioners use Prophet inside CausalImpact-style
workflows.

**NeuralProphet, GreyKite, others.** Prophet inspired a small
ecosystem of "easy forecasting libraries" — NeuralProphet
(PyTorch-based, more flexible), LinkedIn's GreyKite, Uber's
Orbit, Amazon's GluonTS. All borrow Prophet's "decompose +
nice API" template.

---

## When NOT to use Prophet

**When you need accuracy more than convenience.** Hand-tuned
SARIMA or a Holt-Winters fit on the right transform usually
beats Prophet on a single carefully-modelled series.

**When the trend has structural breaks Prophet can't see.**
Default changepoints are placed in the first 80% of the
history. If a real change happened at month 95, Prophet won't
catch it unless you tell it where to look.

**When you have many parallel related series.** Prophet fits
each series independently; global deep-learning forecasters
(DeepAR, TFT) exploit cross-series structure.

**When you don't have weekly or yearly seasonality.** Prophet
is *built* for those defaults. If your series has only an
intra-day cycle, configure it manually or use a more general
tool.

**When the data is sub-hourly with multiple complex
seasonalities.** Energy demand, network traffic, financial
tick data — TBATS or Temporal Fusion Transformer (next
article) are better fits.

---

## What comes next

Part 5 of the Time Series & Forecasting track is **Temporal
Fusion Transformer (TFT)** — the modern deep-learning answer
for forecasting. TFT combines: variable selection networks
(automatic feature importance), LSTM encoders + decoders for
local temporal structure, multi-head self-attention for
long-range patterns, and quantile-loss outputs for calibrated
prediction intervals. Where ARIMA, ETS, UCM, and Prophet
each handle one series at a time, TFT trains globally across
hundreds or thousands of related series and exploits
shared structure.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**prophet_demo.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/06-time-series-forecasting/04-prophet/prophet_demo.py)

Run it with:

```bash
pip install numpy pandas prophet statsmodels
python prophet_demo.py
```

It needs `numpy`, `pandas`, `prophet`, and `statsmodels` (for
the comparison). The script fits Prophet on the
airline-passengers dataset, extracts the trend and seasonal
components, and compares forecast accuracy against ARIMA,
SARIMA, Holt-Winters multiplicative, and UCM. The headline
insight worth pinning to the wall: **Prophet wraps a
state-space-style decomposition (piecewise trend + Fourier
seasonality + holiday effects) in a deliberately friendly
API; its value is the workflow — sensible defaults,
missing-data handling, component plots — rather than headline
accuracy; hand-tuned classical methods usually beat it on a
single series, but Prophet wins on the cost of forecasting a
hundred series of unknown character**.

---

*This is Part 4 of the Time Series & Forecasting track in the Algorithms in Python series. The companion script `prophet_demo.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 3 of this track covered State-Space Models. Part 5 will look at Temporal Fusion Transformer — the modern deep-learning answer for forecasting at scale across many related series.*
