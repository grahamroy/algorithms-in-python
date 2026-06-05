"""
arima.py --- companion code for "ARIMA"
(Time Series & Forecasting, Part 1).

Three demos:
  1. Plain ARIMA(2, 1, 2) on the airline-passengers dataset.
  2. 12-month held-out forecast with MAPE and RMSE.
  3. SARIMA(2, 1, 2)(1, 1, 1, 12) for the seasonal version
     that handles annual cycles.

Dependencies: numpy, pandas, statsmodels. Runs in a few seconds.
"""

import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


SEPARATOR = "=" * 72


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Airline passengers dataset (Box & Jenkins, 1976)
# Monthly totals, Jan 1949 – Dec 1960. The canonical seasonal series.
# ---------------------------------------------------------------------------

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
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs((actual - pred) / actual)) * 100)


def rmse(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def demo_arima(train, test):
    banner("DEMO 1 --- ARIMA(2, 1, 2) on airline passengers")

    model = ARIMA(train, order=(2, 1, 2)).fit()
    print(f"  Training set    : {len(train)} months "
          f"(Jan 1949 – Dec 1959)")
    print(f"  Test set        : {len(test)} months "
          f"(Jan 1960 – Dec 1960)")
    print(f"  Differencing    : d = 1 (first difference is stationary)")
    print(f"  AIC             : {model.aic:.2f}")
    return model


def demo_forecast(model, test):
    banner("DEMO 2 --- 12-month forecast on held-out test set")
    forecast = model.forecast(steps=len(test))
    print(f"  Test MAPE       : {mape(test.values, forecast.values):.2f}%")
    print(f"  Test RMSE       : {rmse(test.values, forecast.values):.2f}")


def demo_sarima(train, test):
    banner("DEMO 3 --- SARIMA(2, 1, 2)(1, 1, 1, 12) for seasonal awareness")
    sarima = SARIMAX(train,
                     order=(2, 1, 2),
                     seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    forecast = sarima.forecast(steps=len(test))
    print(f"  AIC             : {sarima.aic:.2f}")
    print(f"  Test MAPE       : {mape(test.values, forecast.values):.2f}%")
    print(f"  Test RMSE       : {rmse(test.values, forecast.values):.2f}")


def main() -> None:
    data = load_data()
    train, test = data[:-12], data[-12:]
    arima = demo_arima(train, test)
    demo_forecast(arima, test)
    demo_sarima(train, test)
    print()


if __name__ == "__main__":
    main()
