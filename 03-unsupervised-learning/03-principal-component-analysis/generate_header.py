"""Generate the header image for the PCA article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

# Ten distinguishable digit colours
DIGIT_COLORS = ['#3B82F6', '#DC2626', '#16A34A', '#F59E0B',
                '#7C3AED', '#0891B2', '#DB2777', '#65A30D',
                '#EA580C', '#475569']

VAR_BAR_FILL = '#dbeafe'
VAR_BAR_BORDER = '#3B82F6'
CUM_LINE_COLOR = '#7C3AED'


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 'PCA: Rotate Into the Coordinates Your Data Already Has',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        '64 pixel features compressed to 2 principal components — and the digit classes show up.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# --- Two side-by-side panels --------------------------------------------
LEFT_PANEL = (0.4, 0.9, 8.0, 6.0)
RIGHT_PANEL = (8.6, 0.9, 7.0, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle='round,pad=0.02,rounding_size=0.15',
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)


# ========================================================================
# LEFT PANEL: digits projected onto first 2 PCs, coloured by digit class
# ========================================================================
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4,
        'Digits (64 features) projected onto PC1 vs PC2',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

digits = load_digits()
X = digits.data
y = digits.target
proj = PCA(n_components=2).fit_transform(X)

x_min, x_max = proj[:, 0].min() - 1, proj[:, 0].max() + 1
y_min, y_max = proj[:, 1].min() - 1, proj[:, 1].max() + 1

plot_x0 = lpx + 0.55
plot_x1 = lpx + lpw - 0.55
plot_y0 = lpy + 0.85
plot_y1 = lpy + lph - 1.0

def to_panel(p):
    fx = (p[0] - x_min) / (x_max - x_min)
    fy = (p[1] - y_min) / (y_max - y_min)
    return plot_x0 + fx * (plot_x1 - plot_x0), \
           plot_y0 + fy * (plot_y1 - plot_y0)

# Plot points coloured by digit class
for cls in range(10):
    pts = proj[y == cls]
    coords = np.array([to_panel(p) for p in pts])
    ax.scatter(coords[:, 0], coords[:, 1],
               s=8, c=DIGIT_COLORS[cls],
               edgecolors='none', alpha=0.6, zorder=2)

# Tiny legend on the bottom
legend_y = lpy + 0.35
legend_x_start = lpx + 0.4
spacing = (lpw - 0.8) / 10
for cls in range(10):
    cx = legend_x_start + cls * spacing + spacing * 0.2
    ax.scatter([cx], [legend_y], s=18, c=DIGIT_COLORS[cls],
               edgecolors='none', zorder=3)
    ax.text(cx + 0.12, legend_y, f"{cls}",
            fontsize=8, ha='left', va='center',
            color=TEXT_COLOR, fontfamily='monospace', zorder=3)


# ========================================================================
# RIGHT PANEL: explained-variance bars + cumulative line
# ========================================================================
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4,
        'How much variance does each PC carry?',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Fit PCA with all components for the variance plot
pca_full = PCA(n_components=64).fit(X)
ratios = pca_full.explained_variance_ratio_[:20]
cum = np.cumsum(ratios)

bar_left = rpx + 0.85
bar_right = rpx + rpw - 0.4
bar_top = rpy + rph - 1.2
bar_bottom = rpy + 1.4

n_bars = len(ratios)
bar_w = (bar_right - bar_left) / n_bars * 0.7
bar_gap = (bar_right - bar_left) / n_bars

ymax = 1.0  # normalised to 1.0 (cumulative scale)

def y_to_panel(y_val):
    return bar_bottom + (y_val / ymax) * (bar_top - bar_bottom)

# Plot ratio bars (scaled by ratio's own max for visibility)
ratio_max = ratios.max()

for i, r in enumerate(ratios):
    x = bar_left + i * bar_gap + (bar_gap - bar_w) / 2
    h = (r / ratio_max) * ((bar_top - bar_bottom) * 0.45)
    ax.add_patch(plt.Rectangle((x, bar_bottom), bar_w, h,
                                facecolor=VAR_BAR_FILL,
                                edgecolor=VAR_BAR_BORDER,
                                linewidth=1.0, zorder=2))

# Overlay cumulative line
xs_line = [bar_left + i * bar_gap + bar_gap / 2 for i in range(n_bars)]
ys_line = [y_to_panel(c) for c in cum]
ax.plot(xs_line, ys_line, color=CUM_LINE_COLOR,
        linewidth=2.0, marker='o', markersize=4, zorder=3)

# 95% reference line
y95 = y_to_panel(0.95)
ax.plot([bar_left, bar_right], [y95, y95],
        color='#94A3B8', linewidth=0.8, linestyle='--', zorder=1)
ax.text(bar_right - 0.05, y95 + 0.1, '95%',
        fontsize=8, ha='right', va='bottom',
        color='#475569', fontfamily='sans-serif', fontstyle='italic')

# Axes labels
for k in (1, 5, 10, 15, 20):
    cx = bar_left + (k - 1) * bar_gap + bar_gap / 2
    ax.text(cx, bar_bottom - 0.18, f"{k}",
            fontsize=8, ha='center', va='top',
            color=SUBTLE_TEXT, fontfamily='monospace')
ax.text((bar_left + bar_right) / 2, bar_bottom - 0.55,
        'Component index',
        fontsize=9, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Legend
legend_y = rpy + rph - 1.0
ax.add_patch(plt.Rectangle((rpx + 0.6, legend_y - 0.06),
                            0.18, 0.12,
                            facecolor=VAR_BAR_FILL,
                            edgecolor=VAR_BAR_BORDER,
                            linewidth=1.0))
ax.text(rpx + 0.85, legend_y, 'per-PC ratio',
        fontsize=9, ha='left', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.plot([rpx + 2.4, rpx + 2.65], [legend_y, legend_y],
        color=CUM_LINE_COLOR, linewidth=2.0, marker='o',
        markersize=4)
ax.text(rpx + 2.75, legend_y, 'cumulative',
        fontsize=9, ha='left', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Unsupervised Learning Part 3',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '03-unsupervised-learning/'
       '03-principal-component-analysis/header_pca.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
