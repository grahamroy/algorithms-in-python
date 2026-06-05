"""Generate the header image for the Prophet article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from prophet import Prophet


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
TREND_COLOR = '#3B82F6'
SEAS_COLOR = '#F59E0B'
ACTUAL_COLOR = '#1F2937'
FORECAST_COLOR = '#16A34A'


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
train = data[:-12]
test = data[-12:]

df = pd.DataFrame({'ds': train.index, 'y': train.values})
m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
            daily_seasonality=False).fit(df)
future = pd.concat([
    pd.DataFrame({'ds': train.index}),
    pd.DataFrame({'ds': test.index}),
])
fc = m.predict(future)

fig, axes = plt.subplots(3, 1, figsize=(1600/150, 900/150), dpi=150,
                         gridspec_kw={'height_ratios': [0.6, 1.3, 1]})
fig.patch.set_facecolor(BG_COLOR)

axes[0].axis('off')
axes[0].text(0.5, 0.85,
             'Prophet: Decomposition + Friendly API',
             fontsize=18, fontweight='bold', ha='center', va='center',
             color=TEXT_COLOR, transform=axes[0].transAxes)
axes[0].text(0.5, 0.30,
             'Piecewise trend + Fourier seasonalities + holiday effects. Sub-second fit, native missing-data, easy decomposition.',
             fontsize=11, fontstyle='italic', ha='center', va='center',
             color=SUBTLE_TEXT, transform=axes[0].transAxes)

# Forecast vs actual
axes[1].plot(data.index, data.values, color=ACTUAL_COLOR, linewidth=1.4,
             label='actual')
axes[1].plot(fc['ds'], fc['yhat'], color=FORECAST_COLOR,
             linewidth=2.0, linestyle='--', label='Prophet forecast')
axes[1].fill_between(fc['ds'], fc['yhat_lower'], fc['yhat_upper'],
                     color=FORECAST_COLOR, alpha=0.15)
axes[1].axvline(train.index[-1], color='#94A3B8', linewidth=0.8,
                linestyle=':')
axes[1].set_title('observed + forecast (with 80% interval)',
                  loc='left', fontsize=11, fontweight='bold',
                  color=ACTUAL_COLOR)
axes[1].legend(loc='upper left', fontsize=9, frameon=False)
axes[1].set_yticks([])
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].tick_params(axis='x', colors=SUBTLE_TEXT, labelsize=8)
axes[1].set_facecolor('#F8FAFC')

# Trend + seasonality decomposition
fc_train_only = m.predict(pd.DataFrame({'ds': train.index}))
ax2 = axes[2]
ax2.plot(train.index, fc_train_only['trend'], color=TREND_COLOR,
         linewidth=2.0, label='trend')
ax2_twin = ax2.twinx()
ax2_twin.plot(train.index, fc_train_only['yearly'], color=SEAS_COLOR,
              linewidth=1.2, label='yearly seasonality')
ax2_twin.axhline(0, color='#94A3B8', linewidth=0.5, linestyle=':')
ax2.set_title('components: trend (blue) + yearly seasonality (orange)',
              loc='left', fontsize=11, fontweight='bold',
              color=TREND_COLOR)
ax2.set_yticks([])
ax2_twin.set_yticks([])
ax2.spines['top'].set_visible(False)
ax2_twin.spines['top'].set_visible(False)
ax2.tick_params(axis='x', colors=SUBTLE_TEXT, labelsize=8)
ax2.set_facecolor('#F8FAFC')

fig.text(0.5, 0.02,
         'Algorithms in Python  |  Time Series & Forecasting Part 4',
         fontsize=8, ha='center', va='center',
         color='#aaaaaa', fontfamily='sans-serif')

plt.tight_layout(rect=[0.02, 0.04, 0.98, 1.0])
out = ('D:/Projects/Medium/algorithms-in-python/'
       '06-time-series-forecasting/'
       '04-prophet/header_prophet.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
