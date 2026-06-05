"""Generate the header image for the Exponential Smoothing article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing, ExponentialSmoothing
)
from statsmodels.tsa.statespace.sarimax import SARIMAX


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'
HIST_COLOR = '#1F2937'
TEST_COLOR = '#16A34A'
SES_COLOR = '#94A3B8'
HW_COLOR = '#3B82F6'
SARIMA_COLOR = '#DC2626'


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

ses_fc = SimpleExpSmoothing(train,
                            initialization_method="estimated").fit().forecast(12)
hw_fc = ExponentialSmoothing(train, trend="add", seasonal="mul",
                             seasonal_periods=12,
                             initialization_method="estimated").fit().forecast(12)
sarima_fc = SARIMAX(train, order=(2, 1, 2),
                    seasonal_order=(1, 1, 1, 12)).fit(disp=False).forecast(12)

fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('auto')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45,
        'Exponential Smoothing: Simple Recurrence Equations Beat ARIMA',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Same airline-passengers benchmark. Holt-Winters multiplicative — three smoothing parameters — wins.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

plot_x0 = PANEL[0] + 0.9
plot_x1 = PANEL[0] + PANEL[2] - 0.5
plot_y0 = PANEL[1] + 1.0
plot_y1 = PANEL[1] + PANEL[3] - 0.8

n_total = len(data)
y_min, y_max = 0, data.max() + 80

def to_x(i): return plot_x0 + (i / (n_total - 1)) * (plot_x1 - plot_x0)
def to_y(v): return plot_y0 + (v - y_min) / (y_max - y_min) * (plot_y1 - plot_y0)

# History (black)
xs = [to_x(i) for i in range(len(train))]
ys = [to_y(v) for v in train.values]
ax.plot(xs, ys, color=HIST_COLOR, linewidth=1.4, zorder=2)

# Actuals (green)
xs = [to_x(i) for i in range(len(train) - 1, n_total)]
ys = [to_y(v) for v in [train.values[-1]] + list(test.values)]
ax.plot(xs, ys, color=TEST_COLOR, linewidth=2.4, zorder=3)

# Forecasts
fc_xs = [to_x(i) for i in range(len(train) - 1, n_total)]
ax.plot(fc_xs, [to_y(v) for v in [train.values[-1]] + list(ses_fc.values)],
        color=SES_COLOR, linewidth=2.0, linestyle=':', zorder=3)
ax.plot(fc_xs, [to_y(v) for v in [train.values[-1]] + list(hw_fc.values)],
        color=HW_COLOR, linewidth=2.2, linestyle='-.', zorder=4)
ax.plot(fc_xs, [to_y(v) for v in [train.values[-1]] + list(sarima_fc.values)],
        color=SARIMA_COLOR, linewidth=2.0, linestyle='--', zorder=3)

boundary_x = to_x(len(train) - 0.5)
ax.plot([boundary_x, boundary_x], [plot_y0, plot_y1],
        color='#94A3B8', linewidth=0.8, linestyle=':')
ax.text(boundary_x, plot_y1 + 0.15, 'train | test',
        fontsize=9, fontstyle='italic',
        ha='center', va='bottom', color=SUBTLE_TEXT)

years = [1949, 1953, 1957, 1961]
for y in years:
    i = (y - 1949) * 12
    if i < n_total:
        ax.text(to_x(i), plot_y0 - 0.35, str(y),
                fontsize=9, ha='center', va='center', color=SUBTLE_TEXT)

# Legend
lx = PANEL[0] + 0.6
ly = PANEL[1] + 0.45
entries = [
    (HIST_COLOR, '-',  'history'),
    (TEST_COLOR, '-',  'actual'),
    (SES_COLOR,  ':',  'Simple ES (MAPE 14.2%)'),
    (HW_COLOR,   '-.', 'Holt-Winters mult (2.2%)'),
    (SARIMA_COLOR, '--', 'SARIMA (3.0%)'),
]
for i, (c, s, t) in enumerate(entries):
    sx = lx + i * 2.85
    ax.plot([sx, sx + 0.4], [ly, ly], color=c, linewidth=2.0, linestyle=s)
    ax.text(sx + 0.48, ly, t, fontsize=9.5,
            ha='left', va='center', color=c, fontweight='bold')

ax.text(8, 0.3,
        'Algorithms in Python  |  Time Series & Forecasting Part 2',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '06-time-series-forecasting/'
       '02-exponential-smoothing/header_ets.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
