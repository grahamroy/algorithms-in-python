"""
state_space.py --- companion code for "State-Space Models"
(Time Series & Forecasting, Part 3).

Three demos:
  1. Unobserved Components Model with local linear trend +
     dummy-variable seasonality on the airline-passengers data.
  2. Component decomposition: print final level, trend, and
     seasonal peak/trough.
  3. Side-by-side accuracy comparison with ARIMA, SARIMA, and
     Holt-Winters.

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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents


SEPARATOR = "=" * 72


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


def demo_ucm(train, test):
    banner("DEMO 1 --- UCM (local linear trend + seasonal) on airline data")
    ucm = UnobservedComponents(train,
                                level="local linear trend",
                                seasonal=12,
                                irregular=True).fit(disp=False)
    fc = ucm.forecast(len(test))
    print(f"  Components       : local linear trend + seasonal(12) + irregular")
    print(f"  Fit method       : Kalman filter MLE")
    print(f"  AIC              : {ucm.aic:.2f}")
    print(f"  Test MAPE        : {mape(test.values, fc.values):.2f}%")
    print(f"  Test RMSE        : {rmse(test.values, fc.values):.2f}")
    return ucm


def demo_decomposition(ucm):
    banner("DEMO 2 --- Decomposition: each component plotted separately")
    levels = ucm.level.smoothed
    trends = ucm.trend.smoothed
    seasons = ucm.seasonal.smoothed
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    last_year_season = seasons[-12:]
    peak_idx = int(np.argmax(last_year_season))
    trough_idx = int(np.argmin(last_year_season))
    print(f"  Final estimated level : {levels[-1]:.1f}")
    print(f"  Final estimated trend : {trends[-1]:.2f} per month")
    print(f"  Seasonal pattern peaks in: {months[peak_idx]}; "
          f"troughs in: {months[trough_idx]}")


def demo_compare(train, test, ucm):
    banner("DEMO 3 --- Side-by-side with ARIMA / SARIMA / Holt-Winters")
    arima = ARIMA(train, order=(2, 1, 2)).fit()
    sarima = SARIMAX(train, order=(2, 1, 2),
                     seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    hw = ExponentialSmoothing(train, trend="add", seasonal="mul",
                              seasonal_periods=12,
                              initialization_method="estimated").fit()

    rows = [
        ("ARIMA(2, 1, 2)",     arima),
        ("SARIMA(...)",        sarima),
        ("Holt-Winters mult",  hw),
        ("UCM (this article)", ucm),
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


def main() -> None:
    data = load_data()
    train, test = data[:-12], data[-12:]
    ucm = demo_ucm(train, test)
    demo_decomposition(ucm)
    demo_compare(train, test, ucm)
    print()


if __name__ == "__main__":
    main()
