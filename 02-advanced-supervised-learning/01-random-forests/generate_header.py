"""Generate the header image for the Random Forests article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CLASS_BORDER = ['#3B82F6', '#DC2626']  # blue, red
CLASS_FILL = ['#dbeafe', '#fee2e2']
REGION_FILL = ['#eff6ff', '#fef2f2']   # pale blue / red


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 'Random Forests: Average Away the Variance',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Bootstrap each tree, subsample features at every split, vote the predictions.',
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
X, y = make_moons(n_samples=300, noise=0.25, random_state=7)


def project(panel, data_x, data_y, x_min, x_max, y_min, y_max,
            x_pad=0.55, y_top_pad=1.0, y_bottom_pad=0.55):
    px, py, pw, ph = panel
    plot_x0 = px + x_pad
    plot_x1 = px + pw - x_pad
    plot_y0 = py + y_bottom_pad
    plot_y1 = py + ph - y_top_pad
    fx = (data_x - x_min) / (x_max - x_min)
    fy = (data_y - y_min) / (y_max - y_min)
    return plot_x0 + fx * (plot_x1 - plot_x0), \
           plot_y0 + fy * (plot_y1 - plot_y0), \
           (plot_x0, plot_x1, plot_y0, plot_y1)


def plot_classifier(panel, clf, title):
    px, py, pw, ph = panel
    ax.text(px + pw/2, py + ph - 0.4, title,
            fontsize=13, fontweight='bold', ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif')

    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4

    # Decision boundary on a grid
    n_grid = 250
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(y_min, y_max, n_grid),
    )
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    _, _, (plot_x0, plot_x1, plot_y0, plot_y1) = project(
        panel, X[0, 0], X[0, 1], x_min, x_max, y_min, y_max,
    )
    from matplotlib.colors import ListedColormap
    ax.imshow(Z, extent=(plot_x0, plot_x1, plot_y0, plot_y1),
              origin='lower', cmap=ListedColormap(REGION_FILL),
              alpha=0.7, aspect='auto', zorder=1)

    # Training points
    for cls in (0, 1):
        pts = X[y == cls]
        coords = np.array([
            project(panel, p[0], p[1], x_min, x_max, y_min, y_max)[:2]
            for p in pts
        ])
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=22, c=CLASS_FILL[cls],
                   edgecolors=CLASS_BORDER[cls],
                   linewidths=0.9, zorder=2)


# ========================================================================
# LEFT: single decision tree
# ========================================================================
tree = DecisionTreeClassifier(max_depth=6, random_state=7)
tree.fit(X, y)
plot_classifier(LEFT_PANEL, tree, 'One greedy tree')
lpx, lpy, lpw, lph = LEFT_PANEL
ax.text(lpx + lpw/2, lpy + 0.35,
        'jagged staircase boundary; high variance',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# ========================================================================
# RIGHT: random forest of 200 trees
# ========================================================================
forest = RandomForestClassifier(n_estimators=200, max_depth=6,
                                max_features='sqrt', bootstrap=True,
                                random_state=7)
forest.fit(X, y)
plot_classifier(RIGHT_PANEL, forest,
                'A forest of 200 of them, averaged')
rpx, rpy, rpw, rph = RIGHT_PANEL
ax.text(rpx + rpw/2, rpy + 0.35,
        'smoother boundary; the average is stable',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Advanced Supervised Learning Part 1',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '02-advanced-supervised-learning/'
       '01-random-forests/header_random_forest.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
