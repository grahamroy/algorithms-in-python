"""Generate the header image for the Hierarchical Clustering article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CLUSTER_BORDER = ['#3B82F6', '#DC2626', '#16A34A', '#F59E0B']
CLUSTER_FILL = ['#dbeafe', '#fee2e2', '#dcfce7', '#fef3c7']
DENDRO_COLOR = '#1F2937'
CUT_LINE_COLOR = '#7C3AED'  # purple


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 'Hierarchical Clustering: A Tree of Every Possible Clustering',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Build the dendrogram once, cut horizontally at any height, read off your clusters.',
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
# Generate a small synthetic dataset (12 points in 4 obvious groups)
# ========================================================================
rng = np.random.default_rng(7)
group_centres = np.array([
    [-2.5,  2.0],
    [ 2.5,  2.0],
    [-2.5, -2.0],
    [ 2.5, -2.0],
])
points = []
group_labels = []
for k, c in enumerate(group_centres):
    for _ in range(3):
        points.append(c + 0.45 * rng.standard_normal(2))
        group_labels.append(k)
points = np.array(points)
group_labels = np.array(group_labels)


# ========================================================================
# LEFT PANEL: scatter, points colored by their natural group
# ========================================================================
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4,
        '12 points in 4 natural groups',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

x_min, x_max = -4.0, 4.0
y_min, y_max = -3.5, 3.5

plot_x0 = lpx + 0.55
plot_x1 = lpx + lpw - 0.55
plot_y0 = lpy + 0.55
plot_y1 = lpy + lph - 1.0

def to_panel(p):
    fx = (p[0] - x_min) / (x_max - x_min)
    fy = (p[1] - y_min) / (y_max - y_min)
    return plot_x0 + fx * (plot_x1 - plot_x0), \
           plot_y0 + fy * (plot_y1 - plot_y0)

for i, (p, g) in enumerate(zip(points, group_labels)):
    cx, cy = to_panel(p)
    ax.scatter([cx], [cy], s=120, c=CLUSTER_FILL[g],
               edgecolors=CLUSTER_BORDER[g],
               linewidths=1.5, zorder=2)
    ax.text(cx, cy, str(i + 1),
            fontsize=8, fontweight='bold',
            ha='center', va='center',
            color=CLUSTER_BORDER[g], fontfamily='monospace', zorder=3)

ax.text(lpx + lpw/2, lpy + 0.3,
        'Each point gets a label only after the dendrogram is cut.',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# ========================================================================
# RIGHT PANEL: dendrogram with horizontal cut lines
# ========================================================================
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4,
        'Dendrogram: every merge, every height',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Compute linkage
Z = linkage(points, method='ward')

# Dendrogram coordinates (use the scipy helper but extract its drawing
# coordinates so we can re-plot inside our panel)
dendro_top = rpy + rph - 1.1
dendro_bottom = rpy + 1.1
dendro_left = rpx + 0.6
dendro_right = rpx + rpw - 0.6

# Use a temporary axes to extract the dendrogram drawing
import matplotlib
fig_tmp, ax_tmp = plt.subplots()
ddata = dendrogram(Z, no_plot=True)
plt.close(fig_tmp)

# ddata has 'icoord' (x in [5, 10*n-5] increments of 5) and 'dcoord'
# (y = merge distances). Map both into panel coordinates.
icoord = np.array(ddata['icoord'])
dcoord = np.array(ddata['dcoord'])

ix_min = icoord.min()
ix_max = icoord.max()
dy_min = 0.0
dy_max = dcoord.max() * 1.1  # extra headroom for cut labels

def x_to_panel(ix):
    f = (ix - ix_min) / (ix_max - ix_min)
    return dendro_left + f * (dendro_right - dendro_left)

def y_to_panel(d):
    f = (d - dy_min) / (dy_max - dy_min)
    return dendro_bottom + f * (dendro_top - dendro_bottom)

# Plot each U-shaped link
for xs, ys in zip(icoord, dcoord):
    px = [x_to_panel(x) for x in xs]
    py = [y_to_panel(y) for y in ys]
    ax.plot(px, py, color=DENDRO_COLOR, linewidth=1.4, zorder=2)

# Leaf labels on the bottom: original point indices in dendrogram order
leaf_order = ddata['leaves']
for i, leaf in enumerate(leaf_order):
    # Each leaf is at icoord position 5 + 10*i
    ix = 5.0 + 10.0 * i
    px = x_to_panel(ix)
    py = dendro_bottom - 0.20
    ax.text(px, py, str(leaf + 1),
            fontsize=8, ha='center', va='top',
            color=CLUSTER_BORDER[group_labels[leaf]],
            fontfamily='monospace', fontweight='bold')

# Two horizontal cut lines: K = 4 (low) and K = 2 (high)
# Choose heights between merge distances
sorted_dists = np.sort(dcoord[:, 1])  # the top of each U
# K = 4 cut: just above the 8th-smallest merge so 4 lineages remain
cut_K4 = (sorted_dists[-4] + sorted_dists[-5]) / 2
# K = 2 cut: just above the 10th-smallest merge so 2 lineages remain
cut_K2 = (sorted_dists[-2] + sorted_dists[-3]) / 2

cut_y_K4 = y_to_panel(cut_K4)
cut_y_K2 = y_to_panel(cut_K2)

ax.plot([dendro_left, dendro_right],
        [cut_y_K4, cut_y_K4],
        color=CUT_LINE_COLOR, linewidth=1.4, linestyle='--', zorder=3)
ax.text(dendro_left + 0.15, cut_y_K4 + 0.13, 'cut → K = 4',
        fontsize=9, ha='left', va='bottom',
        color=CUT_LINE_COLOR, fontfamily='sans-serif',
        fontweight='bold')

ax.plot([dendro_left, dendro_right],
        [cut_y_K2, cut_y_K2],
        color=CUT_LINE_COLOR, linewidth=1.4, linestyle='--', zorder=3)
ax.text(dendro_left + 0.15, cut_y_K2 + 0.13, 'cut → K = 2',
        fontsize=9, ha='left', va='bottom',
        color=CUT_LINE_COLOR, fontfamily='sans-serif',
        fontweight='bold')

ax.text(rpx + rpw/2, rpy + 0.3,
        "Cut lower for fine clusters, higher for broad ones.",
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Unsupervised Learning Part 2',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '03-unsupervised-learning/'
       '02-hierarchical-clustering/header_hierarchical.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
