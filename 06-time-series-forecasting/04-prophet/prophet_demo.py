"""
prophet_demo.py --- companion code for "Prophet"
(Time Series & Forecasting, Part 4).

Three demos:
  1. Fit Prophet on airline-passengers, report wall time and MAPE.
  2. Component decomposition (trend, yearly seasonality).
  3. Side-by-side comparison against ARIMA, SARIMA, Holt-Winters,
     and UCM from the previous three articles.

Dependencies: numpy, pandas, prophet, statsmodels.
Install prophet with: pip install prophet
Runs in a few seconds.
"""

import sys
import time
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from prophet import Prophet
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


def demo_prophet(train, test):
    banner("DEMO 1 --- Prophet on airline passengers")

    df = pd.DataFrame({'ds': train.index, 'y': train.values})
    future = pd.DataFrame({'ds': test.index})

    t0 = time.perf_counter()
    m = Prophet(yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False)
    m.fit(df)
    fc = m.predict(future)
    dt = time.perf_counter() - t0

    print(f"  Fit time          : {dt:.2f} s")
    print(f"  Components fitted : trend (linear) + yearly seasonality")
    print(f"  Test MAPE         : {mape(test.values, fc['yhat'].values):.2f}%")
    print(f"  Test RMSE         : {rmse(test.values, fc['yhat'].values):.2f}")
    return m, fc


def demo_decomposition(m, train):
    banner("DEMO 2 --- Component decomposition")

    full_df = pd.DataFrame({'ds': train.index})
    fc = m.predict(full_df)
    end_trend = float(fc['trend'].iloc[-1])
    yearly = fc['yearly'].values[-12:]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    peak_idx = int(np.argmax(yearly))
    trough_idx = int(np.argmin(yearly))
    print(f"  Estimated trend at end of training : {end_trend:.1f}")
    print(f"  Yearly peak month: {months[peak_idx]}")
    print(f"  Yearly trough month: {months[trough_idx]}")


def demo_compare(train, test, prophet_fc):
    banner("DEMO 3 --- Side-by-side with ARIMA / SARIMA / Holt-Winters / UCM")

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
    # Prophet row
    p_mape = mape(test.values, prophet_fc['yhat'].values)
    p_rmse = rmse(test.values, prophet_fc['yhat'].values)
    print(f"  {'Prophet':<19}   {'—':>10}   "
          f"{p_mape:>9.2f}%   {p_rmse:>10.2f}")


def main() -> None:
    data = load_data()
    train, test = data[:-12], data[-12:]
    m, fc = demo_prophet(train, test)
    demo_decomposition(m, train)
    demo_compare(train, test, fc)
    print()


if __name__ == "__main__":
    main()
