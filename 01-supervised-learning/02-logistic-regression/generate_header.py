import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Colours ──
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

SIGMOID_COLOR = '#3b82f6'
THRESHOLD_COLOR = '#94A3B8'
CLASS0_COLOR = '#3b82f6'
CLASS1_COLOR = '#DC2626'
BOUNDARY_COLOR = '#1F2937'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.45, 'Logistic Regression: A Curve That Outputs Probabilities',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ── Subtitle ──
ax.text(8, 7.95,
        'Linear combination, then sigmoid --- the same shape sits at the output of every neural classifier',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# Two side-by-side panels
# ═══════════════════════════════════════════════════════════
LEFT_PANEL = (0.4, 0.9, 7.2, 6.0)
RIGHT_PANEL = (8.4, 0.9, 7.2, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)

# ═══════════════════════════════════════════════════════════
# LEFT PANEL --- The sigmoid curve
# ═══════════════════════════════════════════════════════════
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4, 'The sigmoid function',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(lpx + lpw/2, lpy + lph - 0.85,
        'σ(z) = 1 / (1 + e⁻ᶻ)  — squashes any real number into (0, 1)',
        fontsize=9.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Plot box geometry
plot_left = lpx + 0.95
plot_right = lpx + lpw - 0.5
plot_bottom = lpy + 1.1
plot_top = lpy + lph - 1.4

# Map data coords to plot coords
def to_plot(z, p):
    z_min, z_max = -6, 6
    px = plot_left + (z - z_min) / (z_max - z_min) * (plot_right - plot_left)
    py = plot_bottom + p * (plot_top - plot_bottom)
    return px, py

# Axes (gentle grey)
ax.plot([plot_left, plot_right], [plot_bottom, plot_bottom],
        color='#CBD5E1', linewidth=1.0, zorder=1)
ax.plot([plot_left, plot_right],
        [plot_bottom + (plot_top - plot_bottom),
         plot_bottom + (plot_top - plot_bottom)],
        color='#CBD5E1', linewidth=0.5, linestyle=':', zorder=1)

# z = 0 vertical guide
zero_x, _ = to_plot(0, 0)
ax.plot([zero_x, zero_x], [plot_bottom, plot_top],
        color=THRESHOLD_COLOR, linewidth=0.8, linestyle='--', zorder=2)

# 0.5 threshold
_, half_y = to_plot(0, 0.5)
ax.plot([plot_left, plot_right], [half_y, half_y],
        color=THRESHOLD_COLOR, linewidth=0.8, linestyle='--', zorder=2)

# Sigmoid curve
zs = np.linspace(-6, 6, 200)
ps = 1.0 / (1.0 + np.exp(-zs))
xs = [to_plot(z, p)[0] for z, p in zip(zs, ps)]
ys = [to_plot(z, p)[1] for z, p in zip(zs, ps)]
ax.plot(xs, ys, color=SIGMOID_COLOR, linewidth=2.6, zorder=4)

# Y-axis labels
ax.text(plot_left - 0.1, plot_bottom, '0', ha='right', va='center',
        fontsize=9, color=SUBTLE_TEXT, fontfamily='monospace')
ax.text(plot_left - 0.1, half_y, '0.5', ha='right', va='center',
        fontsize=9, color=SUBTLE_TEXT, fontfamily='monospace')
ax.text(plot_left - 0.1, plot_top, '1', ha='right', va='center',
        fontsize=9, color=SUBTLE_TEXT, fontfamily='monospace')

# X-axis labels
for z_val in [-6, 0, 6]:
    px_lab, _ = to_plot(z_val, 0)
    ax.text(px_lab, plot_bottom - 0.18, str(z_val),
            ha='center', va='top', fontsize=9, color=SUBTLE_TEXT,
            fontfamily='monospace')
ax.text((plot_left + plot_right) / 2, plot_bottom - 0.55, 'z = θᵀx',
        ha='center', va='center', fontsize=10, fontstyle='italic',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Annotate the asymptotes
ax.text(plot_right - 0.2, plot_top - 0.25, 'asymptote → 1',
        ha='right', va='center', fontsize=8.5, color=SIGMOID_COLOR,
        fontstyle='italic', fontfamily='sans-serif')
ax.text(plot_left + 0.2, plot_bottom + 0.25, 'asymptote → 0',
        ha='left', va='center', fontsize=8.5, color=SIGMOID_COLOR,
        fontstyle='italic', fontfamily='sans-serif')

# Caption
ax.text(lpx + lpw/2, lpy + 0.45,
        'Steepest near z = 0, plateaus at the extremes',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# RIGHT PANEL --- Two-class decision boundary
# ═══════════════════════════════════════════════════════════
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4, 'Decision boundary in feature space',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(rpx + rpw/2, rpy + rph - 0.85,
        'σ(b₀ + b₁x₁ + b₂x₂) = 0.5  is a hyperplane in feature space',
        fontsize=9.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Plot box
rb_left = rpx + 0.7
rb_right = rpx + rpw - 0.5
rb_bottom = rpy + 1.1
rb_top = rpy + rph - 1.4

# Background
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle((rb_left, rb_bottom),
                       rb_right - rb_left, rb_top - rb_bottom,
                       facecolor='white', edgecolor='#E2E8F0',
                       linewidth=1.0, zorder=1))

# Generate two-class data centered at (-1, +1) and (+1, -1)
rng = np.random.default_rng(42)
n_per = 30
class0 = rng.normal(loc=[-1.0, 1.0], scale=0.7, size=(n_per, 2))
class1 = rng.normal(loc=[1.0, -1.0], scale=0.7, size=(n_per, 2))

# Map (data x, data y) into plot coords; data range is roughly -3..+3
def to_rb(dx, dy):
    px = rb_left + (dx + 3) / 6 * (rb_right - rb_left)
    py = rb_bottom + (dy + 3) / 6 * (rb_top - rb_bottom)
    return px, py

for x, y in class0:
    px, py = to_rb(x, y)
    ax.scatter([px], [py], s=22, color=CLASS0_COLOR, alpha=0.75,
               edgecolors='white', linewidths=0.5, zorder=4)

for x, y in class1:
    px, py = to_rb(x, y)
    ax.scatter([px], [py], s=22, color=CLASS1_COLOR, alpha=0.75,
               edgecolors='white', linewidths=0.5, zorder=4)

# Decision boundary: y = x (since centres are at -1,1 and 1,-1)
boundary_xs = np.linspace(-2.8, 2.8, 50)
boundary_ys = boundary_xs  # the line y = x perfectly separates these two clusters
plot_xs = []
plot_ys = []
for dx, dy in zip(boundary_xs, boundary_ys):
    px, py = to_rb(dx, dy)
    plot_xs.append(px)
    plot_ys.append(py)
ax.plot(plot_xs, plot_ys, color=BOUNDARY_COLOR, linewidth=2.2, zorder=5)

# Legend
ax.scatter([], [], s=30, color=CLASS0_COLOR, alpha=0.8, label='class 0')
ax.scatter([], [], s=30, color=CLASS1_COLOR, alpha=0.8, label='class 1')
ax.plot([], [], color=BOUNDARY_COLOR, linewidth=2.2, label='σ = 0.5')
legend = ax.legend(loc='upper right',
                   bbox_to_anchor=(rb_right / 16, (rb_top - 0.05) / 9),
                   frameon=True, fontsize=8.5)
legend.get_frame().set_facecolor(PANEL_BG)
legend.get_frame().set_edgecolor(PANEL_EDGE)

# Caption
ax.text(rpx + rpw/2, rpy + 0.45,
        'On either side of the line, σ goes to 0 or 1 — confidence rises with distance',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
ax.text(8, 0.3, 'Algorithms in Python  |  Supervised Learning Part 2',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/01-supervised-learning/02-logistic-regression/header_logistic.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
