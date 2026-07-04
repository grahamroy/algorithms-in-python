"""Generate the header image for the MCTS article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PHASE_COLORS = ['#3B82F6', '#7C3AED', '#EA580C', '#16A34A']
WIN_COLOR = '#16A34A'
OTHER_COLOR = '#94A3B8'
BADGE_BG = '#F8FAFC'

# verified from mcts.py ("Take the win" position, seed 1, 400 sims)
MOVES = ['2', '5', '8', '6', '7']
VISITS = [323, 29, 17, 16, 15]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Monte Carlo Tree Search: The Algorithm That Plans Instead of Learns',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        'No training, no network — build a tree of random playouts at decision '
        'time and act on the most-visited move.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the four-phase cycle =====================
ax.text(3.9, 7.15, 'One simulation = four phases', fontsize=11.5,
        fontweight='bold', ha='center', va='center', color=TEXT_COLOR)

def phase_box(x, y, num, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, y), 3.1, 1.05,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.8))
    ax.text(x + 0.33, y + 0.52, num, fontsize=13, fontweight='bold',
            ha='center', va='center', color=col)
    ax.text(x + 0.62, y + 0.68, title, fontsize=10, fontweight='bold',
            ha='left', va='center', color=col)
    ax.text(x + 0.62, y + 0.3, sub, fontsize=7.6, ha='left', va='center',
            color=SUBTLE_TEXT)

# 2x2 loop: 1 top-left, 2 top-right, 3 bottom-right, 4 bottom-left
phase_box(0.5, 5.55, '1', 'Selection', 'walk down by UCT', PHASE_COLORS[0])
phase_box(4.3, 5.55, '2', 'Expansion', 'add one new node', PHASE_COLORS[1])
phase_box(4.3, 3.65, '3', 'Simulation', 'random playout to the end', PHASE_COLORS[2])
phase_box(0.5, 3.65, '4', 'Backprop', 'update N and W up the path', PHASE_COLORS[3])

arrows = [((3.65, 6.07), (4.25, 6.07)),      # 1 -> 2
          ((5.85, 5.5), (5.85, 4.78)),       # 2 -> 3
          ((4.25, 4.17), (3.65, 4.17)),      # 3 -> 4
          ((2.05, 4.78), (2.05, 5.5))]       # 4 -> 1 (loop)
for (x0, y0), (x1, y1) in arrows:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                 mutation_scale=13, color='#94A3B8', linewidth=1.6))

# UCT formula
ax.add_patch(FancyBboxPatch((0.5, 2.1), 6.9, 1.0,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=PHASE_COLORS[0], linewidth=1.5))
ax.text(3.95, 2.78, 'UCT: exploit + explore', fontsize=9,
        fontweight='bold', ha='center', va='center', color=PHASE_COLORS[0])
ax.text(3.95, 2.4,
        r'$\mathrm{UCT}(i) = W_i/N_i\ +\ c\,\sqrt{\ln N_p\,/\,N_i}$',
        fontsize=12, ha='center', va='center', color=TEXT_COLOR)

ax.text(3.95, 1.55, 'repeat for the budget, then play the most-visited move',
        fontsize=8.6, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: visit concentration =====================
ax2 = fig.add_axes([0.60, 0.155, 0.36, 0.56])
colors = [WIN_COLOR] + [OTHER_COLOR] * 4
bars = ax2.bar(range(len(MOVES)), VISITS, color=colors, width=0.62)
ax2.text(0, VISITS[0] + 8, '323', ha='center', va='bottom', fontsize=10,
         color=WIN_COLOR, fontweight='bold')
ax2.annotate('the winning square\ngets 323 of 400 sims', xy=(0.28, 235),
             xytext=(1.1, 165), fontsize=9, color=WIN_COLOR,
             fontweight='bold', va='center',
             arrowprops=dict(arrowstyle='->', color=WIN_COLOR, lw=1.2))
ax2.set_xticks(range(len(MOVES)))
ax2.set_xticklabels([f'sq {m}' for m in MOVES], fontsize=9, color=TEXT_COLOR)
ax2.set_ylabel('root visits (of 400)', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('"Take the win": where the thinking went',
              fontsize=11.5, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.set_ylim(0, 400)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    ax2.spines[sp].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8)

# mini board inset (X X . / O O . / . . .)
axb = fig.add_axes([0.86, 0.42, 0.10, 0.19])
axb.set_xlim(0, 3); axb.set_ylim(0, 3); axb.axis('off')
for i in range(4):
    axb.plot([i, i], [0, 3], color='#CBD5E1', lw=1)
    axb.plot([0, 3], [i, i], color='#CBD5E1', lw=1)
cells = {(0, 2): 'X', (1, 2): 'X', (0, 1): 'O', (1, 1): 'O'}
for (cx, cy), s in cells.items():
    axb.text(cx + 0.5, cy + 0.5, s, ha='center', va='center', fontsize=13,
             fontweight='bold',
             color=TEXT_COLOR if s == 'X' else '#DC2626')
axb.text(2.5, 2.5, '?', ha='center', va='center', fontsize=13,
         fontweight='bold', color=WIN_COLOR)

ax.text(8, 0.18, 'Algorithms in Python  |  Advanced Reinforcement Learning Part 6',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '09-advanced-reinforcement-learning/06-monte-carlo-tree-search/'
       'header_mcts.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
