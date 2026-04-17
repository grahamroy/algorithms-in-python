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

# Primary accent (FIFO items) — blues
FIFO_FILL = '#dbeafe'
FIFO_BORDER = '#3b82f6'

# Secondary accent (heap nodes) — purples
HEAP_FILL = '#ede9fe'
HEAP_BORDER = '#8b5cf6'

# Highlight / urgent priority — amber
HIGHLIGHT_FILL = '#FEF3C7'
HIGHLIGHT_BORDER = '#F59E0B'

ARROW_COLOR = '#475569'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.45, 'Queues: FIFO Ordering and Priority',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ── Subtitle ──
ax.text(8, 7.95, 'Two queue shapes: a line with two ends, and a heap',
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
# LEFT PANEL — FIFO queue (deque)
# ═══════════════════════════════════════════════════════════
lpx, lpy, lpw, lph = LEFT_PANEL

# Panel title
ax.text(lpx + lpw/2, lpy + lph - 0.4, 'FIFO queue (deque)',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Row of 5 slots
values = ['42', '17', '93', '8', '64']
n = len(values)
slot_w, slot_h = 0.85, 0.85
slot_gap = 0.25
row_w = n * slot_w + (n - 1) * slot_gap
row_x0 = lpx + (lpw - row_w) / 2
row_cy = lpy + lph / 2 - 0.1

slot_centers = []
for i, v in enumerate(values):
    sx = row_x0 + i * (slot_w + slot_gap)
    rect = FancyBboxPatch((sx, row_cy - slot_h/2), slot_w, slot_h,
                          boxstyle="round,pad=0.02,rounding_size=0.12",
                          facecolor=FIFO_FILL, edgecolor=FIFO_BORDER,
                          linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(sx + slot_w/2, row_cy, v,
            fontsize=11, fontweight='bold', ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif')
    slot_centers.append((sx + slot_w/2, row_cy))

# Head marker (above the first slot)
head_cx, head_cy = slot_centers[0]
ax.text(head_cx, row_cy + slot_h/2 + 0.3, 'head',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')
# Tail marker
tail_cx, tail_cy = slot_centers[-1]
ax.text(tail_cx, row_cy + slot_h/2 + 0.3, 'tail',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# popleft arrow: leaving from the head to the left
left_edge_x = row_x0
pop_arrow = FancyArrowPatch((left_edge_x - 0.05, row_cy),
                            (left_edge_x - 1.15, row_cy),
                            arrowstyle='-|>', mutation_scale=14,
                            color=ARROW_COLOR, linewidth=1.5, zorder=1)
ax.add_patch(pop_arrow)
ax.text(left_edge_x - 0.6, row_cy + 0.45, 'popleft',
        fontsize=10, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(left_edge_x - 0.6, row_cy - 0.45, 'O(1)',
        fontsize=9, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='monospace')

# append arrow: entering from the right into the tail
right_edge_x = row_x0 + row_w
app_arrow = FancyArrowPatch((right_edge_x + 1.15, row_cy),
                            (right_edge_x + 0.05, row_cy),
                            arrowstyle='-|>', mutation_scale=14,
                            color=ARROW_COLOR, linewidth=1.5, zorder=1)
ax.add_patch(app_arrow)
ax.text(right_edge_x + 0.6, row_cy + 0.45, 'append',
        fontsize=10, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(right_edge_x + 0.6, row_cy - 0.45, 'O(1)',
        fontsize=9, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='monospace')

# Caption below panel
ax.text(lpx + lpw/2, lpy + 0.55, 'Doubly linked blocks — both ends in constant time',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# RIGHT PANEL — Priority queue (heap)
# ═══════════════════════════════════════════════════════════
rpx, rpy, rpw, rph = RIGHT_PANEL

# Panel title
ax.text(rpx + rpw/2, rpy + rph - 0.4, 'Priority queue (heap)',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Tree geometry: root at top, 3 levels (1, 2, 4 nodes)
# Values: root=1, children=3,5, grandchildren=4,7,6,8
# Verify min-heap: 3<=4,7 ✓; 5<=6,8 ✓; root 1<=3,5 ✓
tree_vals = {
    'root': 1,
    'L': 3, 'R': 5,
    'LL': 4, 'LR': 7, 'RL': 6, 'RR': 8,
}

node_r = 0.38
# Horizontal centre of the panel
cx = rpx + rpw / 2
# Y positions for three tree levels
y_root = rpy + rph - 1.35
y_l2 = y_root - 1.35
y_l3 = y_l2 - 1.35

# X offsets
dx_l2 = 1.55
dx_l3 = 0.78

positions = {
    'root': (cx, y_root),
    'L':    (cx - dx_l2, y_l2),
    'R':    (cx + dx_l2, y_l2),
    'LL':   (cx - dx_l2 - dx_l3, y_l3),
    'LR':   (cx - dx_l2 + dx_l3, y_l3),
    'RL':   (cx + dx_l2 - dx_l3, y_l3),
    'RR':   (cx + dx_l2 + dx_l3, y_l3),
}

# Draw edges first so circles sit on top
edges = [('root', 'L'), ('root', 'R'),
         ('L', 'LL'), ('L', 'LR'),
         ('R', 'RL'), ('R', 'RR')]
for a, b in edges:
    ax_, ay_ = positions[a]
    bx_, by_ = positions[b]
    line = Line2D([ax_, bx_], [ay_, by_],
                  color=PANEL_EDGE, linewidth=1.6, zorder=1)
    # Use a darker edge color for visibility
    line.set_color('#CBD5E1')
    ax.add_line(line)

# Draw nodes
for key, (nx, ny) in positions.items():
    is_root = (key == 'root')
    fill = HIGHLIGHT_FILL if is_root else HEAP_FILL
    border = HIGHLIGHT_BORDER if is_root else HEAP_BORDER
    lw = 1.9 if is_root else 1.4
    circ = Circle((nx, ny), node_r,
                  facecolor=fill, edgecolor=border,
                  linewidth=lw, zorder=3)
    ax.add_patch(circ)
    ax.text(nx, ny, str(tree_vals[key]),
            fontsize=11, fontweight='bold', ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=4)

# heappop label pointing away from the root
root_x, root_y = positions['root']
pop_x = root_x + 1.6
pop_y = root_y + 0.55
heap_arrow = FancyArrowPatch((root_x + node_r * 0.85, root_y + node_r * 0.55),
                             (pop_x - 0.05, pop_y),
                             arrowstyle='-|>', mutation_scale=12,
                             color=HIGHLIGHT_BORDER, linewidth=1.6, zorder=2)
ax.add_patch(heap_arrow)
ax.text(pop_x + 0.05, pop_y, 'heappop \u2192 1',
        fontsize=10, fontweight='bold', ha='left', va='center',
        color=HIGHLIGHT_BORDER, fontfamily='sans-serif')

# Small invariant note under the tree
ax.text(rpx + rpw/2, rpy + 1.0, 'parent \u2264 children (min-heap)',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Caption below panel
ax.text(rpx + rpw/2, rpy + 0.55, 'Binary heap on an array — O(log n) per op',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
ax.text(8, 0.3, 'Algorithms in Python  |  Foundations Part 7',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/00-foundations/07-queues/header_queues.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
