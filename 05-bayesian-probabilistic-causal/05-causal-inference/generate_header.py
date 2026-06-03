"""Generate the header image for the Causal Inference article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

X_COLOR = '#3B82F6'   # treatment (blue)
Y_COLOR = '#16A34A'   # outcome (green)
Z_COLOR = '#DC2626'   # confounder (red)
NEUTRAL = '#1F2937'

fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45,
        'Causal Inference: From "X correlates with Y" to "X causes Y"',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Confounders make naive estimates wrong. Adjustment methods recover the true effect.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

LEFT_PANEL = (0.4, 0.9, 7.6, 6.0)
RIGHT_PANEL = (8.0, 0.9, 7.6, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    ax.add_patch(FancyBboxPatch((px, py), pw, ph,
                                boxstyle='round,pad=0.02,rounding_size=0.15',
                                facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                                linewidth=1.2, zorder=0))


# ========================================================================
# LEFT PANEL: confounder DAG
# ========================================================================
lpx, lpy, lpw, lph = LEFT_PANEL
ax.text(lpx + lpw/2, lpy + lph - 0.4,
        'A confounder Z corrupts the X→Y association',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Node positions
node_r = 0.55
zx, zy = lpx + lpw / 2, lpy + lph - 1.7
xx, xy = lpx + lpw / 2 - 2.1, lpy + 2.0
yx, yy = lpx + lpw / 2 + 2.1, lpy + 2.0

def draw_node(cx, cy, label, color, fill_color):
    ax.add_patch(Circle((cx, cy), node_r, facecolor=fill_color,
                         edgecolor=color, linewidth=2.0, zorder=2))
    ax.text(cx, cy, label, fontsize=14, fontweight='bold',
            ha='center', va='center', color=color,
            fontfamily='monospace', zorder=3)

draw_node(zx, zy, 'Z', Z_COLOR, '#fee2e2')
draw_node(xx, xy, 'X', X_COLOR, '#dbeafe')
draw_node(yx, yy, 'Y', Y_COLOR, '#dcfce7')

# Labels next to nodes
ax.text(zx, zy + 0.9, 'confounder', fontsize=10, ha='center',
        color=Z_COLOR, fontstyle='italic')
ax.text(xx, xy - 0.95, 'treatment', fontsize=10, ha='center',
        color=X_COLOR, fontstyle='italic')
ax.text(yx, yy - 0.95, 'outcome', fontsize=10, ha='center',
        color=Y_COLOR, fontstyle='italic')

# Arrows: Z → X, Z → Y, X → Y (the actual causal effect, smaller)
def arrow(x0, y0, x1, y1, color, lw=2.0, style='-|>', rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                  arrowstyle=style, color=color,
                                  mutation_scale=22, lw=lw,
                                  shrinkA=node_r * 22, shrinkB=node_r * 22,
                                  connectionstyle=f'arc3,rad={rad}',
                                  zorder=4))

arrow(zx, zy, xx, xy, Z_COLOR, lw=1.8, rad=-0.1)
arrow(zx, zy, yx, yy, Z_COLOR, lw=1.8, rad=0.1)
arrow(xx, xy, yx, yy, NEUTRAL, lw=2.2)

# Annotations on the arrows
ax.text(zx - 1.6, (zy + xy) / 2 + 0.1, 'confounds',
        fontsize=9, fontstyle='italic',
        color=Z_COLOR, ha='right', va='center')
ax.text(zx + 1.6, (zy + yy) / 2 + 0.1, 'confounds',
        fontsize=9, fontstyle='italic',
        color=Z_COLOR, ha='left', va='center')
ax.text((xx + yx) / 2, xy - 0.45, 'true causal effect',
        fontsize=10, fontweight='bold', fontstyle='italic',
        color=NEUTRAL, ha='center', va='center')


# ========================================================================
# RIGHT PANEL: results bar chart
# ========================================================================
rpx, rpy, rpw, rph = RIGHT_PANEL
ax.text(rpx + rpw/2, rpy + rph - 0.4,
        'Estimated treatment effect (true = 2.0)',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

methods = ['Naive\n(diff in means)', 'Regression\nadjustment', 'Inverse\npropensity\nweighting']
estimates = [4.08, 1.93, 2.14]
colors = ['#DC2626', '#16A34A', '#16A34A']
fills  = ['#fee2e2', '#dcfce7', '#dcfce7']

plot_x0 = rpx + 1.0
plot_x1 = rpx + rpw - 0.6
plot_y0 = rpy + 1.6
plot_y1 = rpy + rph - 1.2

import numpy as np
n_bars = len(methods)
bar_w = (plot_x1 - plot_x0) / (n_bars + 0.5)
ymax = 4.5

def to_y(v):
    return plot_y0 + (v / ymax) * (plot_y1 - plot_y0)

# Reference line at true value
true_y = to_y(2.0)
ax.plot([plot_x0, plot_x1], [true_y, true_y],
        color='#1F2937', linewidth=1.5, linestyle='--', zorder=2)
ax.text(plot_x1 - 0.05, true_y + 0.12, 'true ATE = 2.0',
        fontsize=10, fontweight='bold', fontstyle='italic',
        ha='right', va='bottom', color='#1F2937')

# Bars
for i, (m, est, c, f) in enumerate(zip(methods, estimates, colors, fills)):
    bx = plot_x0 + (i + 0.5) * bar_w
    bh = to_y(est) - plot_y0
    ax.add_patch(plt.Rectangle((bx - bar_w/2 + 0.1, plot_y0),
                                bar_w - 0.2, bh,
                                facecolor=f, edgecolor=c,
                                linewidth=1.8, zorder=3))
    ax.text(bx, to_y(est) + 0.18, f'{est:.2f}',
            fontsize=11, fontweight='bold',
            ha='center', va='bottom',
            color=c, fontfamily='monospace')
    ax.text(bx, plot_y0 - 0.6, m,
            fontsize=9, ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif')

# Axis: x-axis line, y-axis line
ax.plot([plot_x0, plot_x1], [plot_y0, plot_y0],
        color='#94A3B8', linewidth=1.0)


ax.text(8, 0.3,
        'Algorithms in Python  |  Bayesian, Probabilistic & Causal Methods Part 5',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '05-bayesian-probabilistic-causal/'
       '05-causal-inference/header_causal.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
