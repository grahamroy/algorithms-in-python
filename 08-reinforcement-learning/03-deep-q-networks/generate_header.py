"""Generate the header image for the Deep Q-Networks article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_EDGE = '#E2E8F0'
NET_COLOR = '#3B82F6'
WITH_COLOR = '#16A34A'
WITHOUT_COLOR = '#EA580C'
BADGE_BG = '#F8FAFC'

# Verified against dqn.py (25-episode block means)
WITH = [21.8, 22.7, 24.2, 35.2, 45.0, 64.9, 144.8, 187.7, 168.7, 262.7]
WITHOUT = [21.8, 22.7, 24.1, 19.6, 19.1, 19.4, 46.5, 39.2, 62.2, 87.8]
EP_MID = [13, 38, 63, 88, 113, 138, 163, 188, 213, 238]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# ---- Title ----
ax.text(8, 8.5, 'Deep Q-Networks: When the Q-Table Won’t Fit',
        fontsize=19, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.0,
        'Replace the table with a neural network so the state can be continuous '
        '— then add the two tricks that make it stable.',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the pipeline =====================
# State box
ax.add_patch(FancyBboxPatch((0.5, 5.55), 2.5, 1.15,
             boxstyle='round,pad=0.05,rounding_size=0.12',
             facecolor=BADGE_BG, edgecolor=PANEL_EDGE, linewidth=1.5))
ax.text(1.75, 6.32, 'state  s', fontsize=11, fontweight='bold',
        ha='center', va='center', color=TEXT_COLOR, fontfamily='monospace')
ax.text(1.75, 5.86, '(x, ẋ, θ, θ̇)', fontsize=10,
        ha='center', va='center', color=SUBTLE_TEXT, fontfamily='monospace')

# Network nodes: columns at x = 4.0, 5.1, 6.2, 7.3
cols = [(4.0, 4), (5.1, 3), (6.2, 3), (7.3, 2)]
node_pos = []
for cx, n in cols:
    ys = np.linspace(6.55, 5.55, n)
    node_pos.append([(cx, y) for y in ys])
# edges
for li in range(len(node_pos) - 1):
    for (x1, y1) in node_pos[li]:
        for (x2, y2) in node_pos[li + 1]:
            ax.plot([x1, x2], [y1, y2], color='#CBD5E1',
                    linewidth=0.4, zorder=1)
for li, layer in enumerate(node_pos):
    fc = NET_COLOR if 0 < li < len(node_pos) - 1 else '#93C5FD'
    for (x, y) in layer:
        ax.add_patch(Circle((x, y), 0.13, facecolor=fc,
                            edgecolor='white', linewidth=0.8, zorder=2))
ax.text(5.65, 6.95, 'Q(s, a; θ)    4 → 64 → 64 → 2',
        fontsize=10, ha='center', va='center', color=NET_COLOR,
        fontweight='bold', fontfamily='monospace')

# Q-value outputs (compact, kept clear of the chart on the right)
ax.text(7.62, 6.28, 'Q(left)', fontsize=8, ha='left', va='center',
        color=TEXT_COLOR, fontfamily='monospace')
ax.text(7.62, 5.80, 'Q(right)', fontsize=8, ha='left', va='center',
        color=TEXT_COLOR, fontfamily='monospace')

# arrows
for x0, x1 in [(3.05, 3.75), (7.45, 7.58)]:
    ax.add_patch(FancyArrowPatch((x0, 6.12), (x1, 6.12),
                 arrowstyle='-|>', mutation_scale=12,
                 color='#94A3B8', linewidth=1.4))

# CartPole icon (cart + pole) bottom-left
cart_x, cart_y = 2.4, 3.3
ax.add_patch(Rectangle((cart_x - 0.6, cart_y), 1.2, 0.45,
             facecolor=TEXT_COLOR, edgecolor='none'))
ax.add_patch(Circle((cart_x - 0.35, cart_y), 0.12, facecolor='#475569'))
ax.add_patch(Circle((cart_x + 0.35, cart_y), 0.12, facecolor='#475569'))
ax.plot([cart_x, cart_x + 0.55], [cart_y + 0.45, cart_y + 1.55],
        color=WITH_COLOR, linewidth=4, solid_capstyle='round')
ax.add_patch(Circle((cart_x + 0.55, cart_y + 1.55), 0.13,
            facecolor=WITH_COLOR, edgecolor='none'))
ax.text(2.4, 2.55, 'CartPole', fontsize=10, fontweight='bold',
        ha='center', va='center', color=TEXT_COLOR)
ax.text(2.4, 2.18, '4 continuous dims — no table can fit',
        fontsize=8.5, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# Two trick badges
def badge(x, num, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, 0.55), 3.3, 1.15,
                 boxstyle='round,pad=0.05,rounding_size=0.12',
                 facecolor=BADGE_BG, edgecolor=col, linewidth=1.6))
    ax.add_patch(Circle((x + 0.42, 1.13), 0.26, facecolor=col,
                edgecolor='none'))
    ax.text(x + 0.42, 1.13, num, fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(x + 0.82, 1.36, title, fontsize=10.5, fontweight='bold',
            ha='left', va='center', color=TEXT_COLOR)
    ax.text(x + 0.82, 0.95, sub, fontsize=8.3, ha='left', va='center',
            color=SUBTLE_TEXT)

badge(0.5, '1', 'Experience replay', 'sample random past transitions',
      NET_COLOR)
badge(4.1, '2', 'Target network', 'bootstrap from a frozen copy',
      WITH_COLOR)

# ===================== RIGHT: the learning curve =====================
ax2 = fig.add_axes([0.605, 0.13, 0.355, 0.60])
ax2.plot(EP_MID, WITH, '-o', color=WITH_COLOR, linewidth=2.4,
         markersize=4.5, label='with target network')
ax2.plot(EP_MID, WITHOUT, '-o', color=WITHOUT_COLOR, linewidth=2.4,
         markersize=4.5, label='without target network')
ax2.axhline(500, color='#CBD5E1', linewidth=1.0, linestyle='--')
ax2.text(238, 500, ' max 500', fontsize=7.5, color='#94A3B8',
         va='bottom', ha='right')
ax2.annotate('215', xy=(238, 262.7), xytext=(195, 300),
             fontsize=10, color=WITH_COLOR, fontweight='bold')
ax2.annotate('75', xy=(238, 87.8), xytext=(205, 130),
             fontsize=10, color=WITHOUT_COLOR, fontweight='bold')
ax2.set_xlabel('training episodes', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_ylabel('mean return', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('The target network is the difference',
              fontsize=11.5, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.legend(fontsize=8.5, loc='upper left', frameon=False)
ax2.set_ylim(0, 540)
ax2.set_xlim(0, 250)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#CBD5E1')
ax2.spines['bottom'].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Reinforcement Learning Part 3',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '08-reinforcement-learning/03-deep-q-networks/header_dqn.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
