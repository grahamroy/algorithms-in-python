"""Generate the header image for the SARSA article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CLIFF_FILL = '#fee2e2'
CLIFF_BORDER = '#DC2626'
SARSA_FILL = '#dcfce7'
SARSA_BORDER = '#16A34A'
Q_FILL = '#fed7aa'
Q_BORDER = '#EA580C'
START_FILL = '#dbeafe'
START_BORDER = '#3B82F6'
GOAL_FILL = '#fef3c7'
GOAL_BORDER = '#F59E0B'

fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45, 'SARSA: One Term Changes the Update — and the Policy',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'On-policy learning bootstraps from the action actually taken, so it routes around the cliff its own exploration would fall off.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===== Top strip: the two update rules side by side =====
def update_box(x0, w, title, formula, formula_color, border):
    ax.add_patch(FancyBboxPatch((x0, 6.0), w, 1.3,
                                boxstyle='round,pad=0.02,rounding_size=0.1',
                                facecolor='white', edgecolor=border,
                                linewidth=1.8, zorder=2))
    ax.text(x0 + w/2, 6.95, title, fontsize=11, fontweight='bold',
            ha='center', va='center', color=border,
            fontfamily='sans-serif', zorder=3)
    ax.text(x0 + w/2, 6.4, formula, fontsize=11, ha='center', va='center',
            color=formula_color, fontfamily='monospace', zorder=3)

update_box(1.0, 6.6, 'Q-Learning  (off-policy)',
           'r + γ · max  Q(s′, a′)', Q_BORDER, Q_BORDER)
update_box(8.4, 6.6, 'SARSA  (on-policy)',
           'r + γ · Q(s′, a′)', SARSA_BORDER, SARSA_BORDER)
ax.text(8, 5.55, "the only difference: max over next actions  vs  the action actually taken",
        fontsize=9.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===== Two cliff grids side by side =====
ROWS, COLS = 4, 12
cell = 0.42

def draw_grid(ox, oy, path_cells, path_fill, path_border, label, ret):
    for r in range(ROWS):
        for c in range(COLS):
            x = ox + c * cell
            y = oy + (ROWS - 1 - r) * cell
            if r == 3 and 1 <= c <= 10:
                fc, ec = CLIFF_FILL, CLIFF_BORDER
            elif (r, c) == (3, 0):
                fc, ec = START_FILL, START_BORDER
            elif (r, c) == (3, 11):
                fc, ec = GOAL_FILL, GOAL_BORDER
            elif (r, c) in path_cells:
                fc, ec = path_fill, path_border
            else:
                fc, ec = 'white', '#D1D5DB'
            ax.add_patch(Rectangle((x, y), cell, cell, facecolor=fc,
                                   edgecolor=ec, linewidth=0.8, zorder=2))
    # S / G labels
    sx = ox
    sy = oy
    ax.text(sx + cell/2, sy + cell/2, 'S', fontsize=8, fontweight='bold',
            ha='center', va='center', color=START_BORDER, zorder=3)
    ax.text(ox + 11*cell + cell/2, sy + cell/2, 'G', fontsize=8,
            fontweight='bold', ha='center', va='center',
            color=GOAL_BORDER, zorder=3)
    ax.text(ox + COLS*cell/2, oy + ROWS*cell + 0.45, label,
            fontsize=12, fontweight='bold', ha='center', va='center',
            color=path_border, fontfamily='sans-serif')
    ax.text(ox + COLS*cell/2, oy - 0.4, ret,
            fontsize=10, ha='center', va='center',
            color=TEXT_COLOR, fontfamily='monospace')

# SARSA path: up to row 0, across, down (rows 0,1,2,3 col patterns)
sarsa_path = set()
for c in range(COLS): sarsa_path.add((0, c))
sarsa_path.update({(2, 0), (1, 0), (1, 11), (2, 11)})
# Q-learning path: up to row 2, across, down
q_path = set()
for c in range(COLS): q_path.add((2, c))

gy = 1.7
draw_grid(1.4, gy, q_path, Q_FILL, Q_BORDER,
          'Q-Learning: edge path',
          'online return  -41.8')
draw_grid(9.2, gy, sarsa_path, SARSA_FILL, SARSA_BORDER,
          'SARSA: safe path',
          'online return  -25.1  (better)')

ax.text(8, 0.3,
        'Algorithms in Python  |  Reinforcement Learning Part 2',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '08-reinforcement-learning/02-sarsa/header_sarsa.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
