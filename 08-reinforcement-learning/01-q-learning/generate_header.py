"""Generate the header image for the Q-Learning article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CLIFF_FILL = '#fee2e2'
CLIFF_BORDER = '#DC2626'
PATH_FILL = '#dcfce7'
PATH_BORDER = '#16A34A'
START_FILL = '#dbeafe'
START_BORDER = '#3B82F6'
GOAL_FILL = '#fef3c7'
GOAL_BORDER = '#F59E0B'
ARROW_COLOR = '#16A34A'

fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45, 'Q-Learning: Learning to Act From Consequences Alone',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'No model, no map — just nudge each Q(state, action) toward the Bellman target until the optimal policy emerges.',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===== LEFT PANEL: the cliff grid with learned path =====
LEFT = (0.4, 0.9, 9.4, 6.0)
ax.add_patch(FancyBboxPatch((LEFT[0], LEFT[1]), LEFT[2], LEFT[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))
ax.text(LEFT[0] + LEFT[2]/2, LEFT[1] + LEFT[3] - 0.4,
        'Cliff Walking: the learned greedy path',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

ROWS, COLS = 4, 12
cell = 0.62
grid_w = COLS * cell
grid_h = ROWS * cell
gx = LEFT[0] + (LEFT[2] - grid_w) / 2
gy = LEFT[1] + 1.3

def cell_xy(r, c):
    # r=0 top row; draw with row 0 at top
    return gx + c * cell, gy + (ROWS - 1 - r) * cell

# The greedy path: up from start, across row 2, down to goal
path_cells = set()
path_cells.add((2, 0))
for c in range(COLS):
    path_cells.add((2, c))

for r in range(ROWS):
    for c in range(COLS):
        x, y = cell_xy(r, c)
        if r == 3 and 1 <= c <= 10:
            fc, ec = CLIFF_FILL, CLIFF_BORDER
        elif (r, c) == (3, 0):
            fc, ec = START_FILL, START_BORDER
        elif (r, c) == (3, 11):
            fc, ec = GOAL_FILL, GOAL_BORDER
        elif (r, c) in path_cells:
            fc, ec = PATH_FILL, PATH_BORDER
        else:
            fc, ec = 'white', '#CBD5E1'
        ax.add_patch(Rectangle((x, y), cell, cell, facecolor=fc,
                               edgecolor=ec, linewidth=1.2, zorder=2))

# Labels
sx, sy = cell_xy(3, 0)
ax.text(sx + cell/2, sy + cell/2, 'S', fontsize=11, fontweight='bold',
        ha='center', va='center', color=START_BORDER, zorder=3)
gx2, gy2 = cell_xy(3, 11)
ax.text(gx2 + cell/2, gy2 + cell/2, 'G', fontsize=11, fontweight='bold',
        ha='center', va='center', color=GOAL_BORDER, zorder=3)
for c in range(1, 11):
    cx, cy = cell_xy(3, c)
    ax.text(cx + cell/2, cy + cell/2, '#', fontsize=10, fontweight='bold',
            ha='center', va='center', color=CLIFF_BORDER, zorder=3)

# Arrows along the path: up, across, down
ux, uy = cell_xy(3, 0)
ax.annotate('', xy=(ux + cell/2, uy + cell + 0.05),
            xytext=(ux + cell/2, uy + cell/2),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2.0),
            zorder=4)
for c in range(COLS - 1):
    x, y = cell_xy(2, c)
    ax.annotate('', xy=(x + cell + 0.05, y + cell/2),
                xytext=(x + cell/2, y + cell/2),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.6),
                zorder=4)
dx, dy = cell_xy(2, 11)
ax.annotate('', xy=(dx + cell/2, dy - 0.05),
            xytext=(dx + cell/2, dy + cell/2),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=2.0),
            zorder=4)

ax.text(LEFT[0] + LEFT[2]/2, LEFT[1] + 0.45,
        'Optimal route hugs the cliff edge — 13 steps, return -13.',
        fontsize=9.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===== RIGHT PANEL: the update rule =====
RIGHT = (10.2, 0.9, 5.4, 6.0)
ax.add_patch(FancyBboxPatch((RIGHT[0], RIGHT[1]), RIGHT[2], RIGHT[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))
ax.text(RIGHT[0] + RIGHT[2]/2, RIGHT[1] + RIGHT[3] - 0.4,
        'The TD update',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

cx = RIGHT[0] + RIGHT[2]/2
ax.text(cx, RIGHT[1] + RIGHT[3] - 1.5,
        'Q(s,a)  ←  Q(s,a)',
        fontsize=12, ha='center', va='center',
        color=TEXT_COLOR, fontfamily='monospace')
ax.text(cx, RIGHT[1] + RIGHT[3] - 2.2,
        '+ α · [ TD error ]',
        fontsize=12, ha='center', va='center',
        color='#7C3AED', fontfamily='monospace')

# TD error decomposition box
ax.add_patch(FancyBboxPatch((RIGHT[0] + 0.4, RIGHT[1] + 1.6),
                            RIGHT[2] - 0.8, 1.9,
                            boxstyle='round,pad=0.02,rounding_size=0.1',
                            facecolor='white', edgecolor='#7C3AED',
                            linewidth=1.4, zorder=2))
ax.text(cx, RIGHT[1] + 3.1, 'TD error =',
        fontsize=10.5, fontweight='bold', ha='center', va='center',
        color='#7C3AED', fontfamily='monospace', zorder=3)
ax.text(cx, RIGHT[1] + 2.55, 'r + γ·max Q(s′,a′)',
        fontsize=10.5, ha='center', va='center',
        color=PATH_BORDER, fontfamily='monospace', zorder=3)
ax.text(cx, RIGHT[1] + 2.05, '−  Q(s,a)',
        fontsize=10.5, ha='center', va='center',
        color=CLIFF_BORDER, fontfamily='monospace', zorder=3)

ax.text(cx, RIGHT[1] + 1.0,
        'off-policy: the max learns the\noptimal policy while exploring',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

ax.text(8, 0.3,
        'Algorithms in Python  |  Reinforcement Learning Part 1',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '08-reinforcement-learning/01-q-learning/header_q_learning.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
