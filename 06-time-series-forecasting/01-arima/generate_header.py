"""Generate the header image for the ARIMA article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'
HIST_COLOR = '#1F2937'
TEST_COLOR = '#16A34A'
ARIMA_COLOR = '#DC2626'
SARIMA_COLOR = '#3B82F6'


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

idx = pd.date_range("1949-01-01", periods=len(AIRLINE), freq="MS")
data = pd.Series(AIRLINE, index=idx)
train, test = data[:-12], data[-12:]

arima = ARIMA(train, order=(2, 1, 2)).fit()
arima_fc = arima.forecast(steps=12)
sarima = SARIMAX(train, order=(2, 1, 2),
                 seasonal_order=(1, 1, 1, 12)).fit(disp=False)
sarima_fc = sarima.forecast(steps=12)


fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('auto')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45,
        'ARIMA: Decompose Time Into Lags, Differences, and Noise',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Airline passengers 1949–1960. Plain ARIMA captures trend; SARIMA captures trend + seasonality.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# Time series plot inside panel
plot_x0 = PANEL[0] + 0.9
plot_x1 = PANEL[0] + PANEL[2] - 0.5
plot_y0 = PANEL[1] + 1.0
plot_y1 = PANEL[1] + PANEL[3] - 0.8

n_total = len(data)
y_min, y_max = data.min() - 30, data.max() + 80

def to_x(i):
    return plot_x0 + (i / (n_total - 1)) * (plot_x1 - plot_x0)
def to_y(v):
    return plot_y0 + (v - y_min) / (y_max - y_min) * (plot_y1 - plot_y0)

# Training history
xs = [to_x(i) for i in range(len(train))]
ys = [to_y(v) for v in train.values]
ax.plot(xs, ys, color=HIST_COLOR, linewidth=1.4,
        zorder=2, label='training history')

# True test (held out)
xs = [to_x(i) for i in range(len(train) - 1, n_total)]
ys = [to_y(v) for v in [train.values[-1]] + list(test.values)]
ax.plot(xs, ys, color=TEST_COLOR, linewidth=2.4,
        zorder=3, label='actual')

# ARIMA forecast (red)
xs = [to_x(i) for i in range(len(train) - 1, n_total)]
ys = [to_y(v) for v in [train.values[-1]] + list(arima_fc.values)]
ax.plot(xs, ys, color=ARIMA_COLOR, linewidth=2.0,
        linestyle='--', zorder=3, label='ARIMA forecast')

# SARIMA forecast (blue)
xs = [to_x(i) for i in range(len(train) - 1, n_total)]
ys = [to_y(v) for v in [train.values[-1]] + list(sarima_fc.values)]
ax.plot(xs, ys, color=SARIMA_COLOR, linewidth=2.0,
        linestyle='-.', zorder=3, label='SARIMA forecast')

# Vertical line at train/test boundary
boundary_x = to_x(len(train) - 0.5)
ax.plot([boundary_x, boundary_x], [plot_y0, plot_y1],
        color='#94A3B8', linewidth=0.8, linestyle=':')
ax.text(boundary_x, plot_y1 + 0.15, 'train | test',
        fontsize=9, fontstyle='italic',
        ha='center', va='bottom', color=SUBTLE_TEXT)

# Year ticks
years = [1949, 1953, 1957, 1961]
months_per_year = 12
for y in years:
    i = (y - 1949) * months_per_year
    if i < n_total:
        tx = to_x(i)
        ax.text(tx, plot_y0 - 0.35, str(y),
                fontsize=9, ha='center', va='center',
                color=SUBTLE_TEXT, fontfamily='sans-serif')

# Legend
lx = PANEL[0] + 0.6
ly = PANEL[1] + 0.45
entries = [
    (HIST_COLOR, '-', 'history'),
    (TEST_COLOR, '-', 'actual'),
    (ARIMA_COLOR, '--', 'ARIMA forecast'),
    (SARIMA_COLOR, '-.', 'SARIMA forecast'),
]
for i, (c, s, t) in enumerate(entries):
    sx = lx + i * 3.4
    ax.plot([sx, sx + 0.45], [ly, ly], color=c, linewidth=2.0,
            linestyle=s)
    ax.text(sx + 0.55, ly, t, fontsize=10,
            ha='left', va='center', color=c, fontweight='bold')

ax.text(8, 0.3,
        'Algorithms in Python  |  Time Series & Forecasting Part 1',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '06-time-series-forecasting/'
       '01-arima/header_arima.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
