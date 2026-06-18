"""
exponential_smoothing.py --- companion code for "Exponential Smoothing"
(Time Series & Forecasting, Part 2).

Four demos:
  1. Simple exponential smoothing (no trend, no seasonality).
  2. Holt-Winters multiplicative (trend + multiplicative seasonality).
  3. ETS auto-selection over (E, T, S) configurations.
  4. Side-by-side comparison with SARIMA from Part 1.

Dependencies: numpy, pandas, statsmodels. Runs in a few seconds.
"""

import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing, ExponentialSmoothing
)
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX


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


def demo_simple(train, test):
    banner("DEMO 1 --- Simple exponential smoothing")
    model = SimpleExpSmoothing(train,
                                initialization_method="estimated").fit()
    fc = model.forecast(len(test))
    print(f"  Smoothing parameter α : auto-fit")
    print(f"  AIC               : {model.aic:.2f}")
    print(f"  Test MAPE         : {mape(test.values, fc.values):.2f}%")
    print(f"  Test RMSE         : {rmse(test.values, fc.values):.2f}")
    return model.aic, mape(test.values, fc.values), rmse(test.values, fc.values)


def demo_holt_winters(train, test):
    banner("DEMO 2 --- Holt-Winters multiplicative (trend + seasonality)")
    model = ExponentialSmoothing(train,
                                 trend="add",
                                 seasonal="mul",
                                 seasonal_periods=12,
                                 initialization_method="estimated").fit()
    fc = model.forecast(len(test))
    print(f"  Seasonal period   : 12")
    print(f"  AIC               : {model.aic:.2f}")
    print(f"  Test MAPE         : {mape(test.values, fc.values):.2f}%")
    print(f"  Test RMSE         : {rmse(test.values, fc.values):.2f}")
    return model.aic, mape(test.values, fc.values), rmse(test.values, fc.values)


def demo_ets_auto(train, test):
    banner("DEMO 3 --- ETS auto-selection across candidate (E, T, S) combinations")
    candidates = [
        ("add", "add", "add"),
        ("mul", "add", "mul"),
        ("add", "add", None),
        ("add", None, None),
    ]
    best = None
    best_aic = np.inf
    best_name = None
    for err, trend, seas in candidates:
        try:
            model = ETSModel(train, error=err, trend=trend,
                             seasonal=seas, seasonal_periods=12).fit(
                disp=False
            )
            if model.aic < best_aic:
                best_aic = model.aic
                best = model
                short = lambda x: 'N' if x is None else x[0].upper()
                best_name = f"ETS({short(err)}, {short(trend)}, {short(seas)})"
        except Exception:
            pass
    fc = best.forecast(len(test))
    print(f"  Best model         : {best_name}")
    print(f"  AIC                : {best_aic:.2f}")
    print(f"  Test MAPE          : {mape(test.values, fc.values):.2f}%")
    print(f"  Test RMSE          : {rmse(test.values, fc.values):.2f}")
    return best_aic, mape(test.values, fc.values), rmse(test.values, fc.values), best_name


def demo_summary(simple, hw, ets, ets_name, train, test):
    banner("DEMO 4 --- Side-by-side with SARIMA from Part 1")
    sarima = SARIMAX(train, order=(2, 1, 2),
                     seasonal_order=(1, 1, 1, 12)).fit(disp=False)
    fc = sarima.forecast(len(test))
    sa_aic = sarima.aic
    sa_mape = mape(test.values, fc.values)
    sa_rmse = rmse(test.values, fc.values)

    rows = [
        ("Simple ES",        simple),
        ("Holt-Winters mult", hw),
        (f"{ets_name} auto",  ets[:3]),
        ("SARIMA (Part 1)",  (sa_aic, sa_mape, sa_rmse)),
    ]
    print(f"  {'Method':<19}  {'AIC':>10}   {'Test MAPE':>10}   "
          f"{'Test RMSE':>10}")
    print(f"  {'-' * 17:<19}  {'-' * 10:>10}   {'-' * 10:>10}   "
          f"{'-' * 10:>10}")
    for name, (aic, m, r) in rows:
        print(f"  {name:<19}  {aic:>10.2f}   {m:>9.2f}%   {r:>10.2f}")


def main() -> None:
    data = load_data()
    train, test = data[:-12], data[-12:]
    simple = demo_simple(train, test)
    hw = demo_holt_winters(train, test)
    ets = demo_ets_auto(train, test)
    demo_summary(simple, hw, ets, ets[3], train, test)
    print()


if __name__ == "__main__":
    main()
