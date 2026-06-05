# ARIMA — The Workhorse Statistical Forecasting Method

### *Algorithms in Python --- Time Series & Forecasting, Part 1*

---

The previous track was about probabilistic inference over
unordered observations — Bayesian posteriors, MCMC samples,
causal effects. Time series have a structural property those
methods ignore: **the data points are ordered, and what comes
next depends on what came before**. A new track is needed.

Time series forecasting answers a deceptively simple question:
given a history `y_1, y_2, ..., y_t`, predict `y_{t+1},
y_{t+2}, ..., y_{t+h}`. Sales, sensor readings, web traffic,
weather, electricity demand — every business and scientific
domain has time series and every one of them needs forecasts.
The methods range from one-line statistical formulas to
billion-parameter transformers, and choosing among them is the
practitioner's day-job.

This track opens with **ARIMA** — AutoRegressive Integrated
Moving Average — the method Box and Jenkins formalised in 1970
and the method every working forecaster still tries first. It
is purely statistical (no machine learning, no neural nets),
fits in seconds on any dataset, gives interpretable
coefficients, and on stationary or near-stationary series with
modest noise it is genuinely hard to beat. Twenty-first century
methods (Prophet, deep-learning sequence models) win on
specific problem shapes — multiple seasonalities, exogenous
covariates, long contexts — but ARIMA remains the *baseline*
that every alternative is compared against.

This article builds ARIMA from first principles. We will
decompose the algorithm into its three components, walk through
the stationarity assumption and the differencing trick that
deals with non-stationary series, derive the parameter-selection
machinery (ACF, PACF, AIC/BIC), implement the whole forecasting
pipeline using statsmodels on the canonical airline-passengers
dataset, and finish with the seasonal variants (SARIMA, SARIMAX)
that handle real-world periodicities.

---

## The three components

ARIMA has three building blocks, each indexed by a single
integer hyperparameter.

### AR(p): autoregressive

The current value is a weighted sum of the previous `p` values
plus noise:

```
y_t = c + φ_1 · y_{t-1} + φ_2 · y_{t-2} + ... + φ_p · y_{t-p} + ε_t
```

AR(1) says "today is mostly yesterday plus noise". AR(2) adds
the day before yesterday. AR(`p`) lets the model remember `p`
lags. The `φ_i` are coefficients fit from the data.

### MA(q): moving average

The current value is a weighted sum of the previous `q` *noise*
terms plus the current noise:

```
y_t = c + ε_t + θ_1 · ε_{t-1} + θ_2 · ε_{t-2} + ... + θ_q · ε_{t-q}
```

Confusingly, this is *not* the same as a rolling-window average.
MA(q) means the current value is partly explained by the past
`q` *shocks* — surprises in the past that linger and decay.
ARIMA always has both AR and MA components combined.

### I(d): integration / differencing

The "integrated" part handles non-stationarity. If the raw
series has a trend or a unit root, the model is fit on the
`d`-th *difference* of the series:

```
∇y_t = y_t - y_{t-1}   (first difference)
∇²y_t = ∇y_t - ∇y_{t-1}  (second difference)
```

Most non-stationary series become stationary after one or two
differences. The Augmented Dickey-Fuller test is the standard
check.

The full **ARIMA(p, d, q)** model fits an ARMA(p, q) on the
`d`-th difference of the series, then integrates the forecasts
back up to the original scale.

---

## Stationarity: the core assumption

ARIMA assumes the differenced series is **stationary** — its
statistical properties (mean, variance, autocorrelation) don't
change over time. Without stationarity the model's parameters
don't have stable meanings and the forecasts are unreliable.

Real-world series rarely look stationary at first glance:

- **Trend.** Mean shifts over time. Fix with differencing
  (`d = 1` or `2`) or with a deterministic trend term.
- **Seasonality.** Mean cycles with a fixed period. Fix with
  *seasonal* differencing (`y_t - y_{t-s}` where `s` is the
  season length) — handled by the SARIMA extension.
- **Variance changes.** Variance grows or shrinks over time.
  Fix with a Box-Cox or log transformation before fitting.

Run an ADF test on the differenced series; if `p > 0.05` you
have not differenced enough.

---

## Picking p, d, q: the Box-Jenkins methodology

The classical approach:

1. **Plot the series.** Look for trend, seasonality, variance
   changes. Apply transformations as needed.
2. **Difference until stationary.** Try `d = 0`, then `d = 1`,
   then `d = 2`. Stop when ADF says stationary.
3. **Look at the ACF and PACF** of the differenced series:
   - **ACF** (autocorrelation function): correlation of the
     series with itself at lag `k`.
   - **PACF** (partial autocorrelation function): correlation
     at lag `k` after controlling for shorter lags.
   The rule of thumb:
   - **AR(p) only**: PACF cuts off after lag `p`; ACF decays
     gradually.
   - **MA(q) only**: ACF cuts off after lag `q`; PACF decays
     gradually.
   - **Mixed ARMA(p, q)**: both decay gradually; harder to
     read off.
4. **Fit several candidate models.** Compare with **AIC** or
   **BIC** — lower is better.
5. **Check residuals.** They should look like white noise. If
   not, the model is mis-specified.

This is the traditional manual procedure. In 2026 nobody
actually does steps 3–4 by hand: `pmdarima.auto_arima()` or
the R `forecast::auto.arima()` function search over candidate
`(p, d, q)` combinations and pick the best by AIC. Use the
auto-fit; review the result.

---

## A worked example

The companion script fits ARIMA on the **airline-passengers
dataset** — monthly international airline passenger numbers
from 1949 to 1960. It has a clear upward trend and 12-month
seasonality, the canonical test case for the field.

```
DEMO 1 --- ARIMA(2, 1, 2) on airline passengers
  Training set    : 132 months (Jan 1949 – Dec 1959)
  Test set        : 12 months (Jan 1960 – Dec 1960)
  Differencing    : d = 1 (first difference is stationary)
  AIC             : 1225.56

DEMO 2 --- 12-month forecast on held-out test set
  Test MAPE       : 8.22%
  Test RMSE       : 55.22

DEMO 3 --- SARIMA(2, 1, 2)(1, 1, 1, 12) for seasonal awareness
  AIC             : 901.05
  Test MAPE       : 2.96%
  Test RMSE       : 17.21
```

Three observations.

**Plain ARIMA captures the trend but misses the seasonality.**
The 12-month forecasts follow the upward direction of the
training data but miss the annual peak-and-trough pattern.
Test MAPE of 8.2% is mediocre — the forecasts drift toward a
smooth growing line instead of tracking the December-January
peak.

**SARIMA — adding seasonal differencing and seasonal AR/MA
terms — gets MAPE down to 3.0%.** Almost every real-world
series has seasonality; SARIMA is the version you actually
use. Statsmodels' `SARIMAX` is the production API.

**AIC drops sharply** from 1226 (ARIMA) to 901 (SARIMA),
confirming the seasonal extension is a meaningful improvement
on the model fit — not just on the held-out metric.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

ARIMA is cheap:

**Fitting** is dominated by maximum-likelihood estimation of
the `(p + q + 1)` parameters — `O(I · n · (p + q))` where `I`
is the number of optimiser iterations (typically <100). For
`n = 10⁴` and small `(p, q)` that is sub-second.

**Forecasting** is `O(h · (p + q))` for an `h`-step ahead
forecast. Microseconds in absolute terms.

**Memory** is `O(n + (p + q))` for the data and parameters.

`auto_arima` searches over `O(p_max · q_max · seasonal options)`
candidate models, each requiring a fit. With reasonable bounds
(`p, q ≤ 5`, `P, Q ≤ 2`) the search is on the order of
hundreds of fits — still seconds total.

The expensive variants are **vector ARIMA** (multi-series with
cross-effects), **ARFIMA** (long-memory differencing), and
**GARCH** extensions for changing variance — all
polynomial-factor more expensive but the asymptotic shape is
the same.

---

## Real-world ML and AI connections

ARIMA and SARIMA are everywhere:

**Demand forecasting.** Retail SKU sales, warehouse inventory,
electricity load, water consumption — all canonical SARIMA
territory. Power utilities run SARIMA models on hour-level
demand every day to plan generation. Walmart's classical
inventory-replenishment models were ARIMA-based for decades.

**Economic indicators.** GDP growth, inflation, unemployment —
the Bureau of Labor Statistics, ECB, and most central banks
publish SARIMA-based forecasts as part of their standard
output.

**Pharmaceutical sales forecasting.** Drug-launch revenue
modelling, prescription-trend forecasting at major pharma —
SARIMAX (with exogenous covariates like marketing spend) is
the standard tool.

**Web traffic and capacity planning.** Hourly visitor counts,
server load, CDN bandwidth — ARIMA gives quick reliable
forecasts that feed autoscaling decisions.

**Financial markets — modestly.** Daily returns are
notoriously hard to forecast (efficient markets), but ARIMA
appears in volatility forecasting (GARCH variants) and in
some macroeconomic-prediction-for-portfolio-tilting use
cases.

**Anomaly detection on time series.** Fit ARIMA, compare new
observations against the model's predictive distribution.
Large residuals = anomalies. The basis of most
application-monitoring tools' "metric just deviated from its
baseline" alerts.

**Causal inference for time series.** Difference-in-differences
and synthetic-controls methods (Part 5 of the previous track)
often use ARIMA as the counterfactual model — what would the
series have done absent the intervention?

The pattern: ARIMA is the *first* forecasting model to try.
Beating it requires either (a) clear non-linearity, (b)
multiple interacting seasonalities, (c) exogenous covariates
with non-linear effects, or (d) such large data volumes that
neural methods can train at scale.

---

## When NOT to use ARIMA

**When the series is genuinely non-linear.** Multiple
interacting seasonalities, regime changes, threshold effects
— ARIMA's linear-Gaussian assumption breaks. Try Prophet
(Part 4) or a deep learning model.

**When you have hundreds or thousands of related series.**
Per-series ARIMA fits don't share information; modern hierarchical
or global deep-learning forecasters do. M5 forecasting
competition (Walmart, 2020) — won by LightGBM, not ARIMA.

**When exogenous variables have non-linear effects.** SARIMAX
handles exogenous variables linearly. Real effects (promotions,
holidays, weather) interact in complex ways.

**When you have very short history.** ARIMA needs at least
~50 observations to estimate parameters reliably; SARIMA with
seasonality of 12 needs 2–3 full cycles minimum.

**When the future fundamentally differs from the past.**
Pandemics, regulatory changes, brand-new products. No
statistical model handles these; ARIMA is honest about its
predictive distribution but the distribution is irrelevant
when the data-generating process changes.

---

## What comes next

Part 2 of the Time Series & Forecasting track is **Exponential
Smoothing** — the family of methods (simple ES, Holt linear,
Holt-Winters seasonal) that compete with ARIMA on
small-to-medium business data. Exponential smoothing has fewer
hyperparameters, is even simpler to fit, and on many practical
problems beats ARIMA. The Holt-Winters method is the
forecasting world's other historical workhorse.

After exponential smoothing comes State-Space Models (the
mathematical formalism that subsumes both ARIMA and ETS),
Prophet (Facebook's open-source decomposition method), and
Temporal Fusion Transformer (the modern deep-learning answer).

---

## The complete code

The full script is on GitHub — grab it and run it:

[**arima.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/06-time-series-forecasting/01-arima/arima.py)

Run it with:

```bash
pip install numpy pandas statsmodels
python arima.py
```

It needs `numpy`, `pandas`, and `statsmodels`. The script
loads the airline passengers dataset, fits a plain
ARIMA(2, 1, 2) and a seasonal SARIMA(2, 1, 2)(1, 1, 1, 12),
forecasts the held-out final 12 months, and reports MAPE and
RMSE for each. The headline insight worth pinning to the
wall: **ARIMA combines autoregression (AR), differencing (I)
for non-stationarity, and moving-average noise terms (MA) into
a single linear-Gaussian model; SARIMA extends it with
seasonal versions of all three; and the combination handles a
large fraction of business-data forecasting at the cost of
linear-Gaussian assumptions that newer methods relax**.

---

*This is Part 1 of the Time Series & Forecasting track in the Algorithms in Python series. The companion script `arima.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). The previous track closed with Causal Inference. Part 2 will look at Exponential Smoothing — ARIMA's main historical competitor on small-to-medium business data.*
