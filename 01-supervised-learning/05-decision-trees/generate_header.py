"""Generate the header image for the Decision Trees article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CLASS_BORDER = ['#3B82F6', '#DC2626']  # blue, red
CLASS_FILL = ['#dbeafe', '#fee2e2']
REGION_FILL = ['#eff6ff', '#fef2f2']  # very pale blue / red

NODE_FILL = '#F1F5F9'
NODE_BORDER = '#1F2937'
LEAF_FILL_A = CLASS_FILL[0]
LEAF_BORDER_A = CLASS_BORDER[0]
LEAF_FILL_B = CLASS_FILL[1]
LEAF_BORDER_B = CLASS_BORDER[1]
ARROW_COLOR = '#475569'

# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 'Decision Trees: A Flowchart Learned from Data',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Greedy axis-aligned splits build a staircase boundary that fits non-linear data.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# --- Two side-by-side panels --------------------------------------------
LEFT_PANEL = (0.4, 0.9, 8.2, 6.0)
RIGHT_PANEL = (9.0, 0.9, 6.6, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle='round,pad=0.02,rounding_size=0.15',
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)


# ========================================================================
# LEFT PANEL: moons + staircase decision boundary
# ========================================================================
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4,
        'Two moons + staircase boundary',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Generate two-moons data
X, y = make_moons(n_samples=200, noise=0.22, random_state=7)

# Fit a small tree (depth=4) so the boundary is visible but not too busy
tree = DecisionTreeClassifier(max_depth=4, random_state=7)
tree.fit(X, y)

# Plot region: panel-local coords
plot_x0 = lpx + 0.55
plot_x1 = lpx + lpw - 0.55
plot_y0 = lpy + 0.55
plot_y1 = lpy + lph - 1.0

x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4

def to_panel(px, py):
    fx = (px - x_min) / (x_max - x_min)
    fy = (py - y_min) / (y_max - y_min)
    return plot_x0 + fx * (plot_x1 - plot_x0), \
           plot_y0 + fy * (plot_y1 - plot_y0)

# Draw the decision boundary as filled regions
n_grid = 250
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, n_grid),
    np.linspace(y_min, y_max, n_grid),
)
Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Convert grid corners to panel coords
extent = (
    to_panel(x_min, 0)[0],
    to_panel(x_max, 0)[0],
    to_panel(0, y_min)[1],
    to_panel(0, y_max)[1],
)
from matplotlib.colors import ListedColormap
ax.imshow(Z, extent=extent, origin='lower',
          cmap=ListedColormap(REGION_FILL),
          alpha=0.7, aspect='auto', zorder=1)

# Draw training points
for cls in (0, 1):
    pts = X[y == cls]
    panel_pts = np.array([to_panel(*p) for p in pts])
    ax.scatter(panel_pts[:, 0], panel_pts[:, 1],
               s=24, c=CLASS_FILL[cls],
               edgecolors=CLASS_BORDER[cls],
               linewidths=0.9, zorder=2)

# Legend
legend_x = lpx + 0.6
legend_y = lpy + 0.3
for i, name in enumerate(['class A', 'class B']):
    sx = legend_x + i * 1.6
    ax.scatter([sx], [legend_y], s=42, c=CLASS_FILL[i],
               edgecolors=CLASS_BORDER[i], linewidths=1.2, zorder=3)
    ax.text(sx + 0.18, legend_y, name,
            fontsize=9, ha='left', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif')

# Note about the boundary
ax.text(lpx + lpw - 0.6, lpy + 0.3,
        'depth-4 tree, 11 leaves',
        fontsize=9, fontstyle='italic',
        ha='right', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# ========================================================================
# RIGHT PANEL: tiny tree diagram
# ========================================================================
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4,
        'Each internal node asks one question',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Layout for a depth-3 tree drawn as boxes
# Three levels: root, level 1, level 2 (4 leaves)
node_w = 1.95
node_h = 0.55
leaf_w = 1.05
leaf_h = 0.55

cx = rpx + rpw / 2
top_y = rpy + rph - 1.4
mid_y = top_y - 1.6
bot_y = mid_y - 1.6

# Root (depth 0)
ax.add_patch(FancyBboxPatch((cx - node_w/2, top_y - node_h/2),
                            node_w, node_h,
                            boxstyle='round,pad=0.02,rounding_size=0.10',
                            facecolor=NODE_FILL, edgecolor=NODE_BORDER,
                            linewidth=1.4, zorder=2))
ax.text(cx, top_y, 'x₁ ≤ 0.40',
        fontsize=10, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='monospace', zorder=3)

# Level 1: two internal nodes
spread1 = 2.4
left1_x = cx - spread1
right1_x = cx + spread1

for nx, label in [(left1_x, 'x₂ ≤ 0.10'),
                  (right1_x, 'x₂ ≤ 0.55')]:
    ax.add_patch(FancyBboxPatch((nx - node_w/2, mid_y - node_h/2),
                                node_w, node_h,
                                boxstyle='round,pad=0.02,rounding_size=0.10',
                                facecolor=NODE_FILL,
                                edgecolor=NODE_BORDER,
                                linewidth=1.4, zorder=2))
    ax.text(nx, mid_y, label,
            fontsize=10, fontweight='bold',
            ha='center', va='center',
            color=TEXT_COLOR, fontfamily='monospace', zorder=3)

# Level 2: four leaves
spread2 = 1.2
leaf_specs = [
    (left1_x - spread2, 'A', LEAF_FILL_A, LEAF_BORDER_A),
    (left1_x + spread2, 'B', LEAF_FILL_B, LEAF_BORDER_B),
    (right1_x - spread2, 'A', LEAF_FILL_A, LEAF_BORDER_A),
    (right1_x + spread2, 'B', LEAF_FILL_B, LEAF_BORDER_B),
]
for nx, lab, fill, border in leaf_specs:
    ax.add_patch(FancyBboxPatch((nx - leaf_w/2, bot_y - leaf_h/2),
                                leaf_w, leaf_h,
                                boxstyle='round,pad=0.02,rounding_size=0.12',
                                facecolor=fill, edgecolor=border,
                                linewidth=1.6, zorder=2))
    ax.text(nx, bot_y, lab,
            fontsize=11, fontweight='bold',
            ha='center', va='center',
            color=border, fontfamily='sans-serif', zorder=3)

# Arrows: root -> level 1
def arrow(x0, y0, x1, y1, label=None, label_offset_x=-0.12,
          label_offset_y=0.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.1))
    if label is not None:
        mx = (x0 + x1) / 2 + label_offset_x
        my = (y0 + y1) / 2 + label_offset_y
        ax.text(mx, my, label,
                fontsize=8, fontstyle='italic',
                ha='center', va='center',
                color=SUBTLE_TEXT, fontfamily='sans-serif')

arrow(cx - 0.35, top_y - node_h/2,
      left1_x + 0.35, mid_y + node_h/2,
      label='yes', label_offset_x=-0.18, label_offset_y=0.0)
arrow(cx + 0.35, top_y - node_h/2,
      right1_x - 0.35, mid_y + node_h/2,
      label='no', label_offset_x=0.18, label_offset_y=0.0)

# Arrows: level 1 -> leaves
arrow(left1_x - 0.30, mid_y - node_h/2,
      left1_x - spread2 + 0.30, bot_y + leaf_h/2,
      label='yes', label_offset_x=-0.18)
arrow(left1_x + 0.30, mid_y - node_h/2,
      left1_x + spread2 - 0.30, bot_y + leaf_h/2,
      label='no', label_offset_x=0.18)
arrow(right1_x - 0.30, mid_y - node_h/2,
      right1_x - spread2 + 0.30, bot_y + leaf_h/2,
      label='yes', label_offset_x=-0.18)
arrow(right1_x + 0.30, mid_y - node_h/2,
      right1_x + spread2 - 0.30, bot_y + leaf_h/2,
      label='no', label_offset_x=0.18)

# Caption under the tree
ax.text(rpx + rpw/2, rpy + 0.55,
        'Walk down to a leaf; the leaf is your prediction.',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3, 'Algorithms in Python  |  Supervised Learning Part 5',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/01-supervised-learning/05-decision-trees/header_decision_tree.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
