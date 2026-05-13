"""Generate the header image for the Gradient Boosting article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

LOSS_COLOR = '#7C3AED'   # purple
TREE_COLOR = '#16A34A'   # green
RESID_BAR_POS = '#3B82F6'   # blue
RESID_BAR_NEG = '#DC2626'   # red


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 'Gradient Boosting: Sequential Error Correction',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Each new tree fits the residuals of the ensemble so far. Loss drops with every round.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# --- Two side-by-side panels --------------------------------------------
LEFT_PANEL = (0.4, 0.9, 7.6, 6.0)
RIGHT_PANEL = (8.0, 0.9, 7.6, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle='round,pad=0.02,rounding_size=0.15',
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)


# ========================================================================
# LEFT PANEL: residuals shrinking across boosting rounds
# ========================================================================
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4,
        "Residuals shrink as trees are added",
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Simulate residuals at three stages
rng = np.random.default_rng(7)
n_points = 18
x_grid = np.linspace(0, 1, n_points)
true_resid = 0.6 * np.sin(2 * np.pi * x_grid) + 0.15 * rng.standard_normal(n_points)

# After m trees the residuals shrink by a factor (illustrative)
stages = [('m = 1',  true_resid * 1.00),
          ('m = 5',  true_resid * 0.50),
          ('m = 20', true_resid * 0.15)]

plot_left = lpx + 1.1
plot_right = lpx + lpw - 0.6
plot_top = lpy + lph - 1.1
plot_bottom = lpy + 0.9

# Three stacked horizontal strips
strip_h = (plot_top - plot_bottom) / len(stages)
for s_idx, (label, resids) in enumerate(stages):
    y_centre = plot_top - (s_idx + 0.5) * strip_h
    # Label on the left
    ax.text(plot_left - 0.4, y_centre, label,
            fontsize=10, fontweight='bold',
            ha='right', va='center',
            color=TEXT_COLOR, fontfamily='monospace')
    # Centre baseline
    ax.plot([plot_left, plot_right], [y_centre, y_centre],
            color='#CBD5E1', linewidth=1.0, zorder=1)
    # Bars per point
    bar_w = (plot_right - plot_left) / (n_points * 1.2)
    max_height = strip_h * 0.4
    for i, r in enumerate(resids):
        bx = plot_left + (i + 0.5) * (plot_right - plot_left) / n_points
        # scale residual magnitude into a bar height
        bh = max(min(r / 0.8, 1.0), -1.0) * max_height
        color = RESID_BAR_POS if r >= 0 else RESID_BAR_NEG
        ax.add_patch(plt.Rectangle((bx - bar_w / 2, y_centre),
                                   bar_w, bh,
                                   facecolor=color,
                                   edgecolor=color,
                                   linewidth=0,
                                   zorder=2))

# Caption
ax.text(lpx + lpw/2, lpy + 0.4,
        'Each round, a new tree fits these residuals.',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontstyle='italic',
        fontfamily='sans-serif')


# ========================================================================
# RIGHT PANEL: training-loss curve over boosting rounds
# ========================================================================
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4,
        'Training loss drops monotonically',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

plot_left = rpx + 1.1
plot_right = rpx + rpw - 0.6
plot_top = rpy + rph - 1.2
plot_bottom = rpy + 1.4

# Synthetic loss curves: train monotonic, test U-shape
m_axis = np.arange(1, 201)
train_loss = 0.69 * np.exp(-0.04 * m_axis) + 0.02
# Test loss: drops then rises a little (overfitting)
test_loss = 0.69 * np.exp(-0.05 * m_axis) + 0.20 \
            + 0.00018 * (m_axis - 30) * (m_axis > 30)

def to_plot(mx, my, ymax, ymin):
    fx = (mx - 1) / (200 - 1)
    fy = (my - ymin) / (ymax - ymin)
    return plot_left + fx * (plot_right - plot_left), \
           plot_bottom + fy * (plot_top - plot_bottom)

ymax = max(train_loss.max(), test_loss.max()) * 1.05
ymin = 0.0

# Plot frame
ax.plot([plot_left, plot_right], [plot_bottom, plot_bottom],
        color='#94A3B8', linewidth=1.0, zorder=1)
ax.plot([plot_left, plot_left], [plot_bottom, plot_top],
        color='#94A3B8', linewidth=1.0, zorder=1)

# Axis labels
ax.text(plot_left + (plot_right - plot_left) / 2,
        plot_bottom - 0.45, 'boosting rounds (M)',
        fontsize=9, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')
ax.text(plot_left - 0.55,
        plot_bottom + (plot_top - plot_bottom) / 2,
        'loss', fontsize=9, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif', rotation=90)

# Plot the curves
train_pts = np.array([to_plot(m, l, ymax, ymin)
                      for m, l in zip(m_axis, train_loss)])
test_pts = np.array([to_plot(m, l, ymax, ymin)
                     for m, l in zip(m_axis, test_loss)])
ax.plot(train_pts[:, 0], train_pts[:, 1],
        color=TREE_COLOR, linewidth=2.0, zorder=2,
        label='train')
ax.plot(test_pts[:, 0], test_pts[:, 1],
        color=LOSS_COLOR, linewidth=2.0, zorder=2,
        label='test')

# Sweet-spot marker on test curve
sweet_m = 30
sx, sy = to_plot(sweet_m, test_loss[sweet_m - 1], ymax, ymin)
ax.scatter([sx], [sy], s=70, c='white',
           edgecolors=LOSS_COLOR, linewidths=2.0, zorder=3)
ax.annotate('early-stopping\nsweet spot',
            xy=(sx, sy), xytext=(sx + 1.0, sy + 0.5),
            fontsize=9, fontstyle='italic',
            color=LOSS_COLOR, fontfamily='sans-serif',
            arrowprops=dict(arrowstyle='->',
                            color=LOSS_COLOR, lw=1.0))

# Legend
ax.text(plot_right - 1.2, plot_top - 0.15, 'train',
        fontsize=10, ha='left', va='center',
        color=TREE_COLOR, fontweight='bold',
        fontfamily='sans-serif')
ax.plot([plot_right - 1.4, plot_right - 1.25],
        [plot_top - 0.15, plot_top - 0.15],
        color=TREE_COLOR, linewidth=2.0)
ax.text(plot_right - 1.2, plot_top - 0.50, 'test',
        fontsize=10, ha='left', va='center',
        color=LOSS_COLOR, fontweight='bold',
        fontfamily='sans-serif')
ax.plot([plot_right - 1.4, plot_right - 1.25],
        [plot_top - 0.50, plot_top - 0.50],
        color=LOSS_COLOR, linewidth=2.0)

# Caption
ax.text(rpx + rpw/2, rpy + 0.55,
        'Train loss falls forever; test loss has a sweet spot.',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontstyle='italic',
        fontfamily='sans-serif')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Advanced Supervised Learning Part 2',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '02-advanced-supervised-learning/'
       '02-gradient-boosting/header_gradient_boosting.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
