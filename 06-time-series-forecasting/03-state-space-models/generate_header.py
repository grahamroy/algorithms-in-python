"""Generate the header image for the State-Space Models article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'
HIST_COLOR = '#1F2937'
LEVEL_COLOR = '#3B82F6'
TREND_COLOR = '#16A34A'
SEAS_COLOR = '#F59E0B'

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
# Use log for cleaner decomposition
data = pd.Series(np.log(AIRLINE), index=idx)

ucm = UnobservedComponents(data, level="local linear trend",
                            seasonal=12, irregular=True).fit(disp=False)
level = ucm.level.smoothed
seasonal = ucm.seasonal.smoothed
n = len(data)

fig, axes = plt.subplots(4, 1, figsize=(1600/150, 900/150), dpi=150,
                         gridspec_kw={'height_ratios': [0.5, 1, 1, 1]})
fig.patch.set_facecolor(BG_COLOR)

# Title row (axes[0])
axes[0].axis('off')
axes[0].text(0.5, 0.85,
             'State-Space Models: Decompose a Series Into Latent Components',
             fontsize=17, fontweight='bold', ha='center', va='center',
             color=TEXT_COLOR, fontfamily='sans-serif',
             transform=axes[0].transAxes)
axes[0].text(0.5, 0.35,
             'Kalman-filtered Unobserved Components Model on log(airline passengers): level, seasonal, decomposed.',
             fontsize=11, fontstyle='italic', ha='center', va='center',
             color=SUBTLE_TEXT, fontfamily='sans-serif',
             transform=axes[0].transAxes)

# Observed (axes[1])
axes[1].plot(idx, data.values, color=HIST_COLOR, linewidth=1.5)
axes[1].set_title('observed log(y)', loc='left',
                  fontsize=11, fontweight='bold', color=HIST_COLOR)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_facecolor(PANEL_BG)

# Level (axes[2])
axes[2].plot(idx, level, color=LEVEL_COLOR, linewidth=2.0)
axes[2].set_title('level (smoothed)', loc='left',
                  fontsize=11, fontweight='bold', color=LEVEL_COLOR)
axes[2].set_xticks([])
axes[2].set_yticks([])
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].set_facecolor(PANEL_BG)

# Seasonal (axes[3])
axes[3].plot(idx, seasonal, color=SEAS_COLOR, linewidth=1.5)
axes[3].axhline(0, color='#94A3B8', linewidth=0.6, linestyle=':')
axes[3].set_title('seasonal (12-month cycle)', loc='left',
                  fontsize=11, fontweight='bold', color=SEAS_COLOR)
axes[3].set_yticks([])
axes[3].spines['top'].set_visible(False)
axes[3].spines['right'].set_visible(False)
axes[3].set_facecolor(PANEL_BG)
axes[3].tick_params(axis='x', colors=SUBTLE_TEXT, labelsize=8)

fig.text(0.5, 0.02,
         'Algorithms in Python  |  Time Series & Forecasting Part 3',
         fontsize=8, ha='center', va='center',
         color='#aaaaaa', fontfamily='sans-serif')

plt.tight_layout(rect=[0.02, 0.04, 0.98, 1.0])
out = ('D:/Projects/Medium/algorithms-in-python/'
       '06-time-series-forecasting/'
       '03-state-space-models/header_state_space.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
