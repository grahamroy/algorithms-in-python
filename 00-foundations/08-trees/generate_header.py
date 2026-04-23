import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

# ── Colours ──
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

# Decision-tree node colours (greens for leaves, blues for internal)
INTERNAL_FILL = '#dbeafe'
INTERNAL_BORDER = '#3b82f6'
LEAF_SAFE_FILL = '#dcfce7'
LEAF_SAFE_BORDER = '#16a34a'
LEAF_RISK_FILL = '#fee2e2'
LEAF_RISK_BORDER = '#ef4444'

# KD-tree colours
SPLIT_LINE = '#475569'
POINT_COLOR = '#8b5cf6'
QUERY_COLOR = '#F59E0B'

ARROW_COLOR = '#475569'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.45, 'Trees: Hierarchical Structure for Decisions and Search',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ── Subtitle ──
ax.text(8, 7.95,
        'From decision trees to KD-trees --- one shape, many specialisations',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# Panel geometry
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
# LEFT PANEL --- Decision tree
# ═══════════════════════════════════════════════════════════
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4, 'Decision tree (depth 3)',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Tree layout: 3 levels, but only some children expanded
cx = lpx + lpw / 2
y_root = lpy + lph - 1.4
y_l2 = y_root - 1.4
y_l3 = y_l2 - 1.4

dx_l2 = 1.7
dx_l3 = 0.85

node_w = 1.7
node_h = 0.7

def draw_box(x, y, text, is_leaf=False, is_safe=False, is_risk=False):
    if is_leaf:
        if is_safe:
            fill = LEAF_SAFE_FILL
            border = LEAF_SAFE_BORDER
        else:
            fill = LEAF_RISK_FILL
            border = LEAF_RISK_BORDER
    else:
        fill = INTERNAL_FILL
        border = INTERNAL_BORDER
    box = FancyBboxPatch((x - node_w/2, y - node_h/2), node_w, node_h,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         facecolor=fill, edgecolor=border,
                         linewidth=1.5, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text,
            fontsize=10, fontweight='bold', ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=4)
    return (x, y)

def draw_edge(p1, p2, label_yes=None, label_no=None):
    line = Line2D([p1[0], p2[0]], [p1[1] - node_h/2, p2[1] + node_h/2],
                  color='#94A3B8', linewidth=1.4, zorder=1)
    ax.add_line(line)

# Layer 1 — root
root_pos = draw_box(cx, y_root, 'income > 42k?')

# Layer 2
l_pos = draw_box(cx - dx_l2, y_l2, 'predict\nRISKY',
                 is_leaf=True, is_risk=True)
r_pos = draw_box(cx + dx_l2, y_l2, 'prior_default?')

draw_edge(root_pos, l_pos)
draw_edge(root_pos, r_pos)

# Layer 3 (only under right child)
ll_pos = draw_box(cx + dx_l2 - dx_l3, y_l3, 'predict\nRISKY',
                  is_leaf=True, is_risk=True)
lr_pos = draw_box(cx + dx_l2 + dx_l3, y_l3, 'predict\nSAFE',
                  is_leaf=True, is_safe=True)

draw_edge(r_pos, ll_pos)
draw_edge(r_pos, lr_pos)

# Edge labels
ax.text(cx - dx_l2/2 - 0.25, (y_root + y_l2)/2 + 0.05, 'no',
        fontsize=8.5, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif', style='italic')
ax.text(cx + dx_l2/2 + 0.25, (y_root + y_l2)/2 + 0.05, 'yes',
        fontsize=8.5, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif', style='italic')
ax.text(cx + dx_l2 - dx_l3/2 - 0.18, (y_l2 + y_l3)/2 + 0.05, 'yes',
        fontsize=8.5, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif', style='italic')
ax.text(cx + dx_l2 + dx_l3/2 + 0.18, (y_l2 + y_l3)/2 + 0.05, 'no',
        fontsize=8.5, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif', style='italic')

# Caption
ax.text(lpx + lpw/2, lpy + 0.55,
        'Each split reduces Gini impurity --- learned from data',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# RIGHT PANEL --- KD-tree spatial partition
# ═══════════════════════════════════════════════════════════
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4, 'KD-tree (2D nearest-neighbour)',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Plot box: roughly square inside the right panel
box_pad = 0.6
plot_left = rpx + 0.7
plot_right = rpx + rpw - 0.7
plot_bottom = rpy + 1.2
plot_top = rpy + rph - 1.0

# Outer rectangle
outer = Rectangle((plot_left, plot_bottom),
                  plot_right - plot_left,
                  plot_top - plot_bottom,
                  facecolor='white', edgecolor='#CBD5E1',
                  linewidth=1.4, zorder=1)
ax.add_patch(outer)

# Recursive partition (alternating x and y splits)
# Level 0: vertical split (on x)
x_split_0 = plot_left + (plot_right - plot_left) * 0.55
ax.add_line(Line2D([x_split_0, x_split_0], [plot_bottom, plot_top],
                   color=SPLIT_LINE, linewidth=1.6, zorder=2))

# Level 1: horizontal splits in each half (on y)
y_split_left = plot_bottom + (plot_top - plot_bottom) * 0.45
y_split_right = plot_bottom + (plot_top - plot_bottom) * 0.62
ax.add_line(Line2D([plot_left, x_split_0], [y_split_left, y_split_left],
                   color=SPLIT_LINE, linewidth=1.3, zorder=2,
                   alpha=0.85))
ax.add_line(Line2D([x_split_0, plot_right], [y_split_right, y_split_right],
                   color=SPLIT_LINE, linewidth=1.3, zorder=2,
                   alpha=0.85))

# Level 2: a couple of further vertical splits in some quadrants
x_split_tl = plot_left + (x_split_0 - plot_left) * 0.55
ax.add_line(Line2D([x_split_tl, x_split_tl], [y_split_left, plot_top],
                   color=SPLIT_LINE, linewidth=1.0, zorder=2,
                   alpha=0.7))
x_split_br = x_split_0 + (plot_right - x_split_0) * 0.5
ax.add_line(Line2D([x_split_br, x_split_br], [plot_bottom, y_split_right],
                   color=SPLIT_LINE, linewidth=1.0, zorder=2,
                   alpha=0.7))

# Scatter points (deterministic seed)
rng = np.random.default_rng(7)
N_POINTS = 28
pts_x = rng.uniform(plot_left + 0.15, plot_right - 0.15, N_POINTS)
pts_y = rng.uniform(plot_bottom + 0.15, plot_top - 0.15, N_POINTS)
ax.scatter(pts_x, pts_y, s=22, color=POINT_COLOR,
           edgecolors='white', linewidth=0.8, zorder=4)

# Query point + nearest neighbour highlight
q_x = plot_left + (plot_right - plot_left) * 0.78
q_y = plot_bottom + (plot_top - plot_bottom) * 0.78

# Find nearest by brute force just for the picture
dists = (pts_x - q_x) ** 2 + (pts_y - q_y) ** 2
nn_idx = int(np.argmin(dists))
nn_x, nn_y = pts_x[nn_idx], pts_y[nn_idx]

# Connector
ax.add_line(Line2D([q_x, nn_x], [q_y, nn_y],
                   color=QUERY_COLOR, linewidth=1.4, zorder=4,
                   linestyle=(0, (3, 2))))

# Highlight the nearest
ax.scatter([nn_x], [nn_y], s=80, facecolor='none',
           edgecolors=QUERY_COLOR, linewidth=1.8, zorder=5)

# Query marker (star)
ax.scatter([q_x], [q_y], s=120, color=QUERY_COLOR,
           marker='*', edgecolors=TEXT_COLOR, linewidth=0.6,
           zorder=6)
ax.text(q_x + 0.18, q_y + 0.1, 'query',
        fontsize=9, fontweight='bold', ha='left', va='bottom',
        color=QUERY_COLOR, fontfamily='sans-serif')

# Caption
ax.text(rpx + rpw/2, rpy + 0.55,
        'Recursive median splits prune 99% of distance checks',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
ax.text(8, 0.3, 'Algorithms in Python  |  Foundations Part 8',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/00-foundations/08-trees/header_trees.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
