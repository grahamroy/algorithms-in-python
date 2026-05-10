"""Generate the header image for the K-Nearest Neighbours article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
from sklearn.datasets import make_blobs


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

# Three class colours
CLASS_COLOURS = ['#3B82F6', '#16A34A', '#DC2626']  # blue, green, red
CLASS_FILLS   = ['#dbeafe', '#dcfce7', '#fee2e2']

QUERY_COLOR = '#7C3AED'  # purple
RADIUS_COLOR = '#7C3AED'
HIGHLIGHT_COLOR = '#1F2937'

# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 'K-Nearest Neighbours: Vote of the Closest k',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'No training. At predict time, find the k nearest training points and let them vote.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# --- Two side-by-side panels --------------------------------------------
LEFT_PANEL = (0.4, 0.9, 9.6, 6.0)
RIGHT_PANEL = (10.4, 0.9, 5.2, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle='round,pad=0.02,rounding_size=0.15',
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)


# --- LEFT PANEL: scatter + query + k-circle -----------------------------
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4, 'The algorithm in one picture',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Generate 3 Gaussian clusters in panel-local space
rng = np.random.default_rng(7)
centres = np.array([
    [-1.7, -1.4],
    [ 1.7, -1.4],
    [ 0.0,  1.7],
])
X, y = make_blobs(n_samples=120, centers=centres, cluster_std=1.05,
                  random_state=7)
# Map data x range roughly (-4, 4) into panel x range
# Plot x: lpx + 0.7 ... lpx + lpw - 0.7
# Plot y: lpy + 0.7 ... lpy + lph - 1.1
plot_x0 = lpx + 0.7
plot_x1 = lpx + lpw - 0.7
plot_y0 = lpy + 0.7
plot_y1 = lpy + lph - 1.1

x_min, x_max = -4.5, 4.5
y_min, y_max = -3.7, 3.7

def to_panel(px, py):
    fx = (px - x_min) / (x_max - x_min)
    fy = (py - y_min) / (y_max - y_min)
    return plot_x0 + fx * (plot_x1 - plot_x0), \
           plot_y0 + fy * (plot_y1 - plot_y0)

# Plot training points
for cls in range(3):
    pts = X[y == cls]
    for px, py in pts:
        cx, cy = to_panel(px, py)
        ax.scatter([cx], [cy], s=28, c=CLASS_FILLS[cls],
                   edgecolors=CLASS_COLOURS[cls], linewidths=0.9, zorder=2)

# Pick a query point near the boundary between classes 1 and 2
query = np.array([0.55, 0.20])
qcx, qcy = to_panel(*query)

# Find k nearest training points to the query
k = 5
dists = np.linalg.norm(X - query, axis=1)
nn_idx = np.argpartition(dists, k)[:k]
nn_pts = X[nn_idx]
nn_y = y[nn_idx]
nn_d = dists[nn_idx]
radius = nn_d.max()

# Draw the radius circle in panel coords
# Radius in data units → average panel scale
data_to_panel_x = (plot_x1 - plot_x0) / (x_max - x_min)
data_to_panel_y = (plot_y1 - plot_y0) / (y_max - y_min)
panel_radius = radius * (data_to_panel_x + data_to_panel_y) / 2

# Highlight nearest neighbours (re-plot on top)
for (px, py), cls in zip(nn_pts, nn_y):
    cx, cy = to_panel(px, py)
    ax.scatter([cx], [cy], s=70, c=CLASS_FILLS[cls],
               edgecolors=HIGHLIGHT_COLOR, linewidths=1.6, zorder=3)
    # Connecting line from query to neighbour
    ax.plot([qcx, cx], [qcy, cy], color=HIGHLIGHT_COLOR,
            linewidth=0.6, alpha=0.5, zorder=2)

# Radius circle around query
ax.add_patch(Circle((qcx, qcy), panel_radius,
                    facecolor='none', edgecolor=RADIUS_COLOR,
                    linewidth=1.4, linestyle='--', zorder=2.5, alpha=0.7))

# Query point (star)
ax.scatter([qcx], [qcy], s=260, c=QUERY_COLOR, marker='*',
           edgecolors='white', linewidths=1.8, zorder=4)
ax.text(qcx + 0.20, qcy + 0.15, 'query',
        fontsize=10, fontweight='bold', ha='left', va='bottom',
        color=QUERY_COLOR, fontfamily='sans-serif', zorder=4)

# k = 5 label on the radius
ax.text(qcx + panel_radius * 0.7, qcy - panel_radius - 0.18,
        'k = 5 nearest', fontsize=9, fontstyle='italic',
        ha='center', va='top',
        color=RADIUS_COLOR, fontfamily='sans-serif', zorder=4)

# Class swatches (legend)
legend_x = lpx + 0.6
legend_y = lpy + 0.45
for i, name in enumerate(['class A', 'class B', 'class C']):
    sx = legend_x + i * 1.7
    ax.scatter([sx], [legend_y], s=42, c=CLASS_FILLS[i],
               edgecolors=CLASS_COLOURS[i], linewidths=1.2, zorder=3)
    ax.text(sx + 0.18, legend_y, name,
            fontsize=9, ha='left', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=3)


# --- RIGHT PANEL: vote tally --------------------------------------------
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4, 'Vote of the 5',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(rpx + rpw/2, rpy + rph - 0.85,
        'Count labels of the k nearest, take argmax.',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Tally each class
counts = np.bincount(nn_y, minlength=3)
class_names = ['class A', 'class B', 'class C']
max_count = counts.max()
winner = int(np.argmax(counts))

# Bars
bar_left = rpx + 1.5
bar_max_w = rpw - 2.4
row_top = rpy + rph - 1.6
row_step = 0.85
bar_h = 0.45

for i, (name, cnt) in enumerate(zip(class_names, counts)):
    by = row_top - i * row_step
    bw = (cnt / max(max_count, 1)) * bar_max_w
    ax.text(bar_left - 0.15, by, name,
            fontsize=10, ha='right', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=2)
    ax.add_patch(FancyBboxPatch((bar_left, by - bar_h/2),
                                max(bw, 0.05), bar_h,
                                boxstyle='round,pad=0.02,rounding_size=0.06',
                                facecolor=CLASS_FILLS[i],
                                edgecolor=CLASS_COLOURS[i],
                                linewidth=1.2, zorder=2))
    ax.text(bar_left + bw + 0.18, by, f'{cnt}',
            fontsize=10, fontweight='bold', ha='left', va='center',
            color=CLASS_COLOURS[i], fontfamily='monospace', zorder=2)

# Winner annotation
ax.text(rpx + rpw/2, rpy + 1.2,
        f'argmax → {class_names[winner]}',
        fontsize=11, fontweight='bold', ha='center', va='center',
        color=CLASS_COLOURS[winner], fontfamily='sans-serif')

ax.text(rpx + rpw/2, rpy + 0.55,
        'Prediction = majority of the\nk = 5 nearest training labels',
        fontsize=9, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3, 'Algorithms in Python  |  Supervised Learning Part 4',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/01-supervised-learning/04-k-nearest-neighbours/header_knn.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
