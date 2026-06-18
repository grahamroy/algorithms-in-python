# Temporal Fusion Transformer — Deep Learning for Time Series at Scale

### *Algorithms in Python --- Time Series & Forecasting, Part 5*

---

The four previous articles in this track built classical
statistical forecasters — ARIMA, exponential smoothing,
state-space models, and Prophet. Each fits *one series at a
time*. None of them learn from related series, none scale to
hundreds of thousands of products, and none handle complex
exogenous covariates (weather, promotions, calendar events,
nested category hierarchies) without significant manual
feature engineering.

**Temporal Fusion Transformer (TFT)** (Lim et al, Google
Research, 2019) is the modern deep-learning answer. It is a
single neural network architecture that:

- Trains *globally* across hundreds or thousands of related
  series, sharing parameters and exploiting cross-series
  structure that per-series fits cannot see.
- Combines an LSTM encoder for local temporal patterns with
  multi-head self-attention for long-range dependencies.
- Includes a **variable selection network** that learns which
  features matter for each series, providing interpretable
  feature importance.
- Outputs **quantiles** rather than point forecasts —
  calibrated prediction intervals are a first-class output.
- Handles *static* (per-series) covariates, *known future*
  covariates (calendar, scheduled promotions), and *observed
  past* covariates (weather, sales) separately with
  appropriate gating.

On the M5 Walmart-sales forecasting competition (2020), the
winning entries were gradient-boosted trees, but the top
deep-learning approaches were all variants of TFT or its
contemporaries (DeepAR, N-BEATS). The Nixtla NeuralForecast
benchmark studies showed TFT and its
descendants matching or beating classical methods on the
majority of business forecasting tasks once `n_series > 100`.

This article builds TFT from first principles. We will walk
through the architecture — variable selection, LSTM encoder,
self-attention, quantile output — describe what each piece
does and why, implement a *simplified* version (an LSTM
sequence-to-one forecaster, the core TFT pattern stripped of
the variable-selection and gating machinery) on the
airline-passengers dataset, and finish with the trade-offs and when
to reach for TFT vs. the classical methods from earlier in
this track.

---

## The architecture, end to end

A real TFT is dense — the original paper's diagram fills a
page. The key conceptual blocks:

### Variable Selection Networks (VSN)

For each time step the model sees many features: the lagged
target value, calendar variables (hour, day-of-week, month),
known future covariates (planned promotions), observed past
covariates (weather), static per-series covariates (store
ID, category). The VSN learns per-feature gating weights —
"how much does each input matter at this time step for this
series?" — producing an interpretable feature-importance
score as a side product.

### LSTM encoder-decoder

The historical window goes into an LSTM encoder; the LSTM's
final hidden state is the *context* for the decoder, which
unrolls forecasts step by step. The LSTM captures local
temporal patterns (today's value depends on the last few
days).

### Multi-head self-attention

After the LSTM, a multi-head attention layer lets the model
look across all time steps in the input window. This captures
long-range patterns (a promotion last year affects sales
this year; the lockdown three years ago has lingering
effects on the level). Attention also produces an
interpretable attention-weight pattern showing *when* in the
history matters most.

### Gated Residual Networks (GRN) and Static Covariate Encoders

Throughout the architecture, gated residual networks
(built on the Gated Linear Units of Dauphin et al., 2017)
let the model bypass unnecessary computations. Static covariates (per-series
identifiers, category) feed into separate encoders that
condition the entire prediction.

### Quantile output

The output layer produces *multiple quantiles* of the
predictive distribution — typically the 10th, 50th, and 90th
percentiles. The loss function is the **pinball loss** at
each quantile, summed. The result: calibrated prediction
intervals out of the box, not point estimates that someone
later wraps in a heuristic CI.

The full TFT has ~500K to 5M parameters depending on
configuration — modest by deep-learning standards, but
substantial compared to ARIMA's few coefficients.

---

## Why does it beat the classical methods?

Two reasons.

**Cross-series learning.** When you have 10,000 related
series — say, daily SKU sales for a retailer — TFT trains
*one* model on all of them, sharing the parameters.
Sparse-sales SKUs benefit from the patterns the model learned
on dense-sales SKUs. SARIMA fits 10,000 separate models, none
of which can learn from each other.

**Rich covariate handling.** Want to incorporate Black
Friday, Easter dates, weather forecasts, in-store
promotions, and competitor pricing? In TFT they are all just
input features. In SARIMAX you can add exogenous regressors
but only linearly; in Prophet you can add holidays and
regressors but their interactions are limited; in ARIMA you
mostly cannot at all.

The downside: TFT needs **lots of data** (or many series),
**GPUs** for reasonable training time, and a **lot of
engineering effort** to feed it the right covariates and tune
hyperparameters. The NeuralForecast benchmarks show TFT
is *not* universally better — on small datasets it loses to
SARIMA, on highly idiosyncratic series it loses to per-series
fits. It wins specifically when you have many related series
and rich features.

---

## A simplified worked example

A full TFT implementation is hundreds of lines of PyTorch.
The companion script implements a *minimal* deep-learning
forecaster — a single-layer LSTM encoder followed by a linear
output — trained on a sliding-window representation of the
airline-passengers dataset. The simplification preserves the
core "global model on lagged windows" pattern while removing
the gating, attention, and variable-selection machinery that
distinguishes TFT.

```
DEMO 1 --- Simplified LSTM forecaster on airline passengers
  Architecture     : input(1) → LSTM(hidden=32) → Linear → output
  Loss             : MSE on next-step prediction
  Epochs           : 500
  Window size      : 12 (one year of history per prediction)
  Wall time (CPU)  : 2.4 s
  Test MAPE        : 9.01%
  Test RMSE        : 64.25
```

```
DEMO 2 --- Side-by-side with the rest of the track
  Method                       AIC    Test MAPE    Test RMSE
  -----------------     ----------   ----------   ----------
  ARIMA(2, 1, 2)           1225.56        8.22%        55.22
  SARIMA(...)               901.05        2.96%        17.21
  Holt-Winters mult         633.69        2.21%        15.81
  UCM (Part 3)             1011.43       15.77%        91.29
  Prophet (Part 4)               —        6.61%        43.07
  Simplified LSTM                —        9.01%        64.25
```

Three observations.

**The simplified LSTM is mid-pack on this small single-series
dataset** — MAPE 9.0%, slightly worse than plain ARIMA and
considerably worse than SARIMA or Holt-Winters. Deep
learning's advantage isn't visible on one short series. On
the M4 / M5 benchmarks with thousands of related series, the
ranking inverts.

**Where TFT would win.** Imagine 10,000 SKUs of monthly sales
data: SARIMA / Holt-Winters fits 10,000 separate models, each
on a tiny history; TFT fits one model on all of them and
shares strength across series. On the M5 competition such
shared learning produced 5–15% MAPE improvements over
classical per-series baselines on the harder SKUs.

**The simplified demo skips most of what makes TFT
distinctive.** Real TFT adds variable selection (per-feature
gating), multi-head attention (long-range patterns),
static-covariate encoders, and quantile loss. The LSTM-only demo
captures only the "global neural forecaster" pattern.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

TFT is expensive but parallelisable:

**Training** is `O(E · n_series · n_windows · d²)` per epoch
where `d` is the hidden dimension. With `n_series = 10⁴`,
`n_windows = 100` per series, `d = 64`, and `E = 100` epochs,
that is ~10⁹ operations per epoch — minutes on a single GPU,
hours on CPU.

**Inference** is `O(window_len · d²)` per forecast — much
faster than training, milliseconds on GPU.

**Memory** at training time is `O(batch · window · d)`, easily
fits in 16 GB GPU memory for typical configurations.

**Compare to classical methods**: SARIMA on one series fits
in seconds; SARIMA on 10,000 series fits in 10,000 × seconds
= hours. TFT trains once globally; per-series cost is
amortised. Below `n_series ≈ 100` classical methods are
faster; above `n_series ≈ 10³` TFT becomes increasingly
attractive.

---

## Real-world ML and AI connections

**Retail forecasting at scale.** Walmart, Target, Amazon
demand-forecasting pipelines have all adopted TFT-style
deep-learning forecasters for SKU-level forecasts. Forecast
accuracy improvements of 5–15% translate to billions in
inventory cost savings.

**Energy demand forecasting.** Electricity load,
photovoltaic generation, gas demand — utilities have moved
from ETS/SARIMA per-substation to global deep-learning
models. TFT and its derivatives are the workhorse for
day-ahead and hour-ahead forecasting at most major utilities
in 2026.

**Cloud resource forecasting.** AWS, Azure, GCP all run
internal TFT-style forecasters to predict cloud demand
across customer / region / instance-type pairs for capacity
planning.

**Pharmaceutical sales forecasting.** Pharma analytics teams
have shifted from per-product SARIMAX to globally-trained
deep-learning models, especially for new-product launches
where shared learning across analogue products matters
most.

**Web traffic and CDN load balancing.** Hourly request
volumes per region per service — TFT with calendar features,
known marketing events, and weather covariates is a typical
production deployment.

**Financial time-series.** Deep learning has been slow to win
in finance (efficient markets, low signal-to-noise), but TFT
is a serious tool for option pricing, volatility forecasting,
and order-book modelling.

**The wider ecosystem.** Beyond TFT, the deep time-series
forecasting field has rapidly produced N-BEATS, N-HiTS,
PatchTST, TimesFM (Google's pretrained foundation model for
time series, 2024), and Chronos (Amazon's pretrained
foundation model). All share the "global neural model,
attention-based, scaled training" recipe. TFT is the
representative example, not the only option.

---

## When NOT to use TFT

**When you have one short series.** Classical methods are
faster, simpler, and as accurate. TFT needs lots of data.

**When you don't have a GPU.** Training is impractical on
CPU for non-trivial models. Use a cloud GPU or stay with
classical methods.

**When interpretability is critical.** TFT has *some*
interpretability via VSN and attention weights, but it is no
match for ARIMA's coefficients or UCM's component
decomposition. For regulated domains, classical methods are
often required.

**When you have few related series.** Below `n_series ≈ 50`
the global-training advantage doesn't compensate for the
hyperparameter-tuning effort.

**When data quality is bad.** Deep learning amplifies bad
data. Garbage-in, garbage-out applies more sharply than for
classical methods, which have stronger inductive biases that
sometimes paper over poor inputs.

---

## What comes next

This is the final article in the **Time Series &
Forecasting** track. Five articles: ARIMA (the foundational
linear method), Exponential Smoothing (the recurrence-based
cousin), State-Space Models (the unifying formalism),
Prophet (the friendly-API decomposition tool), and TFT (the
deep-learning answer at scale).

The next track is **Recommender Systems**, opening with
Matrix Factorisation (SVD / ALS) — the classical method
behind Netflix's original recommendation pipeline and still
the strong baseline for collaborative filtering. After that
comes Neural Collaborative Filtering, Two-Tower Retrieval,
and Sequential Recommenders.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**tft.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/06-time-series-forecasting/05-temporal-fusion-transformer/tft.py)

Run it with:

```bash
pip install numpy pandas torch statsmodels
python tft.py
```

It needs `numpy`, `pandas`, `torch`, and `statsmodels` (for
the comparison row). The script implements a simplified
LSTM-based sequence-to-one forecaster on the
airline-passengers dataset, illustrating the core "global
neural forecaster" pattern that TFT generalises. The
headline insight worth pinning to the wall: **TFT trains
one neural network across many related time series, combines
local LSTM patterns with global self-attention, includes
variable selection networks for interpretable feature
importance, outputs quantiles for calibrated uncertainty,
and beats classical methods primarily when you have many
related series and rich covariates — not on a single short
benchmark series**.

---

*This is Part 5 of the Time Series & Forecasting track in the Algorithms in Python series, and the final article of the track. The companion script `tft.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 4 of this track covered Prophet. The next track — Recommender Systems — opens with Matrix Factorisation.*
