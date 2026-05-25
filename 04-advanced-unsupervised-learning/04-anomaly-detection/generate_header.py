"""Generate the header image for the Anomaly Detection article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.datasets import make_blobs


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

INLIER_FILL = '#dbeafe'
INLIER_BORDER = '#3B82F6'
OUTLIER_FILL = '#fee2e2'
OUTLIER_BORDER = '#DC2626'

fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45, 'Anomaly Detection: Spotting the Points That Don’t Belong',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        '500 inliers + 25 outliers. Five algorithms, four families, one ranking problem.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# Generate data
rng = np.random.default_rng(7)
inliers, _ = make_blobs(n_samples=500, centers=[[-2, -2], [3, 2]],
                        cluster_std=0.7, random_state=7)
outliers = rng.uniform(low=-6, high=6, size=(25, 2))

x_min, x_max = -7, 7
y_min, y_max = -5, 5

plot_x0 = PANEL[0] + 0.6
plot_x1 = PANEL[0] + PANEL[2] - 0.6
plot_y0 = PANEL[1] + 0.6
plot_y1 = PANEL[1] + PANEL[3] - 0.6

def to_panel(p):
    fx = (p[0] - x_min) / (x_max - x_min)
    fy = (p[1] - y_min) / (y_max - y_min)
    return plot_x0 + fx * (plot_x1 - plot_x0), \
           plot_y0 + fy * (plot_y1 - plot_y0)

# Inliers
in_coords = np.array([to_panel(p) for p in inliers])
ax.scatter(in_coords[:, 0], in_coords[:, 1],
           s=22, c=INLIER_FILL, edgecolors=INLIER_BORDER,
           linewidths=0.8, alpha=0.7, zorder=2,
           label='inliers (500)')

# Outliers
out_coords = np.array([to_panel(p) for p in outliers])
ax.scatter(out_coords[:, 0], out_coords[:, 1],
           s=80, c=OUTLIER_FILL, edgecolors=OUTLIER_BORDER,
           linewidths=2.0, marker='X', zorder=3,
           label='planted outliers (25)')

# Legend
ax.text(PANEL[0] + 1.6, PANEL[1] + 0.45, 'inliers',
        fontsize=10, ha='left', va='center',
        color=INLIER_BORDER, fontweight='bold')
ax.scatter([PANEL[0] + 1.3], [PANEL[1] + 0.45],
           s=22, c=INLIER_FILL, edgecolors=INLIER_BORDER,
           linewidths=1.0)
ax.text(PANEL[0] + 3.8, PANEL[1] + 0.45, 'planted outliers',
        fontsize=10, ha='left', va='center',
        color=OUTLIER_BORDER, fontweight='bold')
ax.scatter([PANEL[0] + 3.4], [PANEL[1] + 0.45],
           s=60, c=OUTLIER_FILL, edgecolors=OUTLIER_BORDER,
           linewidths=1.6, marker='X')

ax.text(8, 0.3,
        'Algorithms in Python  |  Advanced Unsupervised Learning Part 4',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '04-advanced-unsupervised-learning/'
       '04-anomaly-detection/header_anomaly.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
