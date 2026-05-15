"""Generate the header image for the K-Means Clustering article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CLUSTER_BORDER = ['#3B82F6', '#DC2626', '#16A34A', '#F59E0B', '#7C3AED']
CLUSTER_FILL = ['#dbeafe', '#fee2e2', '#dcfce7', '#fef3c7', '#ede9fe']
CENTROID_COLOR = '#1F2937'
UNCLUSTERED_FILL = '#E5E7EB'
UNCLUSTERED_BORDER = '#9CA3AF'


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, "K-Means: Pick K Centres, Iterate Until Settled",
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        "Lloyd's algorithm: assign every point to the nearest centroid, recompute centroids, repeat.",
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# --- Two side-by-side panels --------------------------------------------
LEFT_PANEL = (0.4, 0.9, 7.2, 6.0)
RIGHT_PANEL = (8.4, 0.9, 7.2, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle='round,pad=0.02,rounding_size=0.15',
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)


# ========================================================================
# Shared data
# ========================================================================
X, _ = make_blobs(n_samples=180,
                  centers=[[-3.0, -2.5], [3.0, -2.5], [0.0, 3.0]],
                  cluster_std=0.9, random_state=7)

# Fit final K-Means for the right panel
km_final = KMeans(n_clusters=3, n_init=10, random_state=7).fit(X)


def panel_project(panel, data_x, data_y, x_min, x_max, y_min, y_max,
                  x_pad=0.55, y_top_pad=1.0, y_bottom_pad=0.55):
    px, py, pw, ph = panel
    plot_x0 = px + x_pad
    plot_x1 = px + pw - x_pad
    plot_y0 = py + y_bottom_pad
    plot_y1 = py + ph - y_top_pad
    fx = (data_x - x_min) / (x_max - x_min)
    fy = (data_y - y_min) / (y_max - y_min)
    return plot_x0 + fx * (plot_x1 - plot_x0), \
           plot_y0 + fy * (plot_y1 - plot_y0)


# ========================================================================
# LEFT PANEL: before clustering — just the raw points
# ========================================================================
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4,
        "Before: unlabelled points",
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

for p in X:
    cx, cy = panel_project(LEFT_PANEL, p[0], p[1],
                           x_min, x_max, y_min, y_max)
    ax.scatter([cx], [cy], s=30, c=UNCLUSTERED_FILL,
               edgecolors=UNCLUSTERED_BORDER,
               linewidths=0.9, zorder=2)

ax.text(lpx + lpw/2, lpy + 0.3,
        "What groups are in this data?",
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# ========================================================================
# RIGHT PANEL: after K-Means with K=3
# ========================================================================
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4,
        "After K-Means (K = 3): groups + centroids",
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Same axis bounds as left for direct comparison
for p, c in zip(X, km_final.labels_):
    cx, cy = panel_project(RIGHT_PANEL, p[0], p[1],
                           x_min, x_max, y_min, y_max)
    ax.scatter([cx], [cy], s=30, c=CLUSTER_FILL[c],
               edgecolors=CLUSTER_BORDER[c],
               linewidths=0.9, zorder=2)

# Plot centroids
for k, mu in enumerate(km_final.cluster_centers_):
    cx, cy = panel_project(RIGHT_PANEL, mu[0], mu[1],
                           x_min, x_max, y_min, y_max)
    # Large X marker for centroid
    ax.scatter([cx], [cy], s=260, marker='X',
               c=CLUSTER_BORDER[k], edgecolors='white',
               linewidths=2.0, zorder=4)
    ax.text(cx + 0.18, cy + 0.18, f"μ_{k+1}",
            fontsize=10, fontweight='bold',
            ha='left', va='bottom',
            color=CLUSTER_BORDER[k], fontfamily='monospace', zorder=4)

ax.text(rpx + rpw/2, rpy + 0.3,
        "Each point belongs to its nearest centroid.",
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Unsupervised Learning Part 1',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '03-unsupervised-learning/'
       '01-k-means-clustering/header_k_means.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
