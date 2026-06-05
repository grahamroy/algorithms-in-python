"""
tft.py --- companion code for "Temporal Fusion Transformer"
(Time Series & Forecasting, Part 5).

A simplified deep-learning forecaster: single-layer LSTM
encoder + linear output, trained on a sliding-window
representation of the airline-passengers dataset. The
simplification preserves the core "global neural sequence
model" pattern that TFT generalises; real TFT adds variable
selection networks, multi-head attention, gated residual
networks, static-covariate encoders, and quantile loss.

Dependencies: numpy, pandas, torch, statsmodels. Runs in
under a minute on CPU.
"""

import sys
import time
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


AIRLINE = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]


def load_data():
    idx = pd.date_range("1949-01-01", periods=len(AIRLINE), freq="MS")
    return pd.Series(AIRLINE, index=idx, name="passengers")


def mape(actual, pred):
    return float(np.mean(np.abs((np.asarray(actual, float) - np.asarray(pred, float))
                                / np.asarray(actual, float))) * 100)


def rmse(actual, pred):
    return float(np.sqrt(np.mean((np.asarray(actual, float) -
                                  np.asarray(pred, float)) ** 2)))


# ---------------------------------------------------------------------------
# Simplified LSTM forecaster
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                            batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, seq, 1)
        out, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def make_windows(series, window=12):
    arr = np.asarray(series, dtype=float)
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i : i + window])
        y.append(arr[i + window])
    return np.array(X), np.array(y)


def demo_lstm(train_series, test_series, window=12, epochs=500,
              lr=1e-3, hidden=32):
    banner("DEMO 1 --- Simplified LSTM forecaster on airline passengers")

    # Normalise (scale by max for numerical stability)
    scale = train_series.max()
    norm_train = train_series.values / scale

    X, y = make_windows(norm_train, window=window)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y_t = torch.tensor(y, dtype=torch.float32)

    torch.manual_seed(RNG_SEED)
    model = LSTMForecaster(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    print(f"  Architecture     : input(1) → LSTM(hidden={hidden}) → "
          f"Linear → output")
    print(f"  Loss             : MSE on next-step prediction")
    print(f"  Epochs           : {epochs}")
    print(f"  Window size      : {window}")

    t0 = time.perf_counter()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
    dt = time.perf_counter() - t0
    print(f"  Wall time (CPU)  : {dt:.1f} s")

    # Roll forward to produce test forecasts
    model.eval()
    history = list(norm_train)
    forecasts = []
    with torch.no_grad():
        for _ in range(len(test_series)):
            win = torch.tensor(history[-window:], dtype=torch.float32) \
                       .view(1, window, 1)
            pred = float(model(win).item())
            forecasts.append(pred)
            history.append(pred)
    forecasts = np.array(forecasts) * scale
    print(f"  Test MAPE        : {mape(test_series.values, forecasts):.2f}%")
    print(f"  Test RMSE        : {rmse(test_series.values, forecasts):.2f}")
    return forecasts


def demo_compare(train, test, lstm_fc):
    banner("DEMO 2 --- Side-by-side with the rest of the track")

    arima = ARIMA(train, order=(2, 1, 2)).fit()
    sarima = SARIMAX(train, order=(2, 1, 2),
                     seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    hw = ExponentialSmoothing(train, trend="add", seasonal="mul",
                              seasonal_periods=12,
                              initialization_method="estimated").fit()
    ucm = UnobservedComponents(train, level="local linear trend",
                                seasonal=12, irregular=True).fit(disp=False)

    rows = [
        ("ARIMA(2, 1, 2)",     arima),
        ("SARIMA(...)",        sarima),
        ("Holt-Winters mult",  hw),
        ("UCM (Part 3)",       ucm),
    ]
    print(f"  {'Method':<19}   {'AIC':>10}   {'Test MAPE':>10}   "
          f"{'Test RMSE':>10}")
    print(f"  {'-' * 17:<19}   {'-' * 10:>10}   {'-' * 10:>10}   "
          f"{'-' * 10:>10}")
    for name, model in rows:
        fc = model.forecast(len(test))
        print(f"  {name:<19}   {model.aic:>10.2f}   "
              f"{mape(test.values, fc.values):>9.2f}%   "
              f"{rmse(test.values, fc.values):>10.2f}")
    # Prophet placeholder + LSTM
    print(f"  {'Prophet (Part 4)':<19}   {'—':>10}   "
          f"{'6.61':>9}%   {'43.07':>10}")
    print(f"  {'Simplified LSTM':<19}   {'—':>10}   "
          f"{mape(test.values, lstm_fc):>9.2f}%   "
          f"{rmse(test.values, lstm_fc):>10.2f}")


def main() -> None:
    data = load_data()
    train, test = data[:-12], data[-12:]
    lstm_fc = demo_lstm(train, test)
    demo_compare(train, test, lstm_fc)
    print()


if __name__ == "__main__":
    main()
