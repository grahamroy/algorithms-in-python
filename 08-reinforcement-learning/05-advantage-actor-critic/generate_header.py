"""Generate the header image for the Advantage Actor-Critic article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ACTOR_COLOR = '#7C3AED'     # policy (purple, as in Part 4)
CRITIC_COLOR = '#3B82F6'    # value  (blue, as in Part 3)
ADV_COLOR = '#16A34A'
REIN_COLOR = '#EA580C'
BADGE_BG = '#F8FAFC'

A2C = [150, 288, 418, 496, 496, 500, 499, 356, 500, 500]
REIN = [182, 427, 402, 449, 391, 117, 86, 148, 119, 80]
EP_MID = [25, 75, 125, 175, 225, 275, 325, 375, 425, 475]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Advantage Actor-Critic: Where the Two Branches Meet',
        fontsize=18.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.02,
        'An actor (policy) proposes; a critic (value) judges. The advantage '
        'A = G − V(s) trains the actor.',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: actor-critic architecture =====================
def box(x, y, w, h, title, sub, col, title_size=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.8))
    ax.text(x + w/2, y + h*0.62, title, fontsize=title_size, fontweight='bold',
            ha='center', va='center', color=col, fontfamily='monospace')
    if sub:
        ax.text(x + w/2, y + h*0.26, sub, fontsize=8, ha='center',
                va='center', color=SUBTLE_TEXT)

# state
box(0.6, 4.25, 1.7, 1.0, 'state s', '', TEXT_COLOR, title_size=11)
# actor (top) and critic (bottom)
box(3.5, 5.55, 2.9, 1.05, 'ACTOR', 'π(a|s) — what to do', ACTOR_COLOR)
box(3.5, 2.95, 2.9, 1.05, 'CRITIC', 'V(s) — how good a state is', CRITIC_COLOR)

# arrows state -> actor, state -> critic
ax.add_patch(FancyArrowPatch((2.35, 4.95), (3.4, 6.0), arrowstyle='-|>',
             mutation_scale=12, color='#94A3B8', linewidth=1.4))
ax.add_patch(FancyArrowPatch((2.35, 4.55), (3.4, 3.5), arrowstyle='-|>',
             mutation_scale=12, color='#94A3B8', linewidth=1.4))

# actor -> action
ax.text(7.55, 6.07, 'sample\naction', fontsize=8.5, ha='center', va='center',
        color=ACTOR_COLOR)
ax.add_patch(FancyArrowPatch((6.5, 6.07), (7.0, 6.07), arrowstyle='-|>',
             mutation_scale=11, color=ACTOR_COLOR, linewidth=1.3))

# advantage box (combines return G and critic V)
box(2.6, 1.15, 4.2, 1.05, 'advantage  A = G − V(s)',
    'better than expected from this state', ADV_COLOR, title_size=11.5)

# critic V(s) -> advantage
ax.add_patch(FancyArrowPatch((4.95, 2.9), (4.95, 2.25), arrowstyle='-|>',
             mutation_scale=11, color=CRITIC_COLOR, linewidth=1.3))
ax.text(5.35, 2.57, 'V(s)', fontsize=7.5, ha='left', va='center',
        color=CRITIC_COLOR, fontfamily='monospace')
# advantage -> actor (trains)
ax.add_patch(FancyArrowPatch((2.6, 1.95), (1.4, 1.95), arrowstyle='-|>',
             mutation_scale=11, color=ADV_COLOR, linewidth=1.4))
ax.add_patch(FancyArrowPatch((1.0, 2.2), (3.45, 5.7),
             connectionstyle='arc3,rad=-0.25', arrowstyle='-|>',
             mutation_scale=12, color=ADV_COLOR, linewidth=1.4,
             linestyle='--'))
ax.text(0.75, 4.0, 'trains the\nactor', fontsize=8, ha='center', va='center',
        color=ADV_COLOR, fontstyle='italic')

# ===================== RIGHT: A2C vs REINFORCE =====================
ax2 = fig.add_axes([0.605, 0.13, 0.355, 0.58])
ax2.plot(EP_MID, A2C, '-o', color=ADV_COLOR, linewidth=2.4, markersize=4,
         label='A2C (critic baseline)')
ax2.plot(EP_MID, REIN, '-o', color=REIN_COLOR, linewidth=2.4, markersize=4,
         label='REINFORCE (constant baseline)')
ax2.axhline(500, color='#CBD5E1', linewidth=1.0, linestyle='--')
ax2.text(475, 500, ' max 500', fontsize=7.5, color='#94A3B8',
         va='bottom', ha='right')
ax2.annotate('solves', xy=(425, 500), xytext=(355, 430),
             fontsize=9.5, color=ADV_COLOR, fontweight='bold')
ax2.annotate('collapses', xy=(425, 119), xytext=(300, 180),
             fontsize=9.5, color=REIN_COLOR, fontweight='bold')
ax2.set_xlabel('training episodes', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_ylabel('mean return', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('The critic makes learning reliable',
              fontsize=11.5, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.legend(fontsize=8, loc='lower left', frameon=False)
ax2.set_ylim(0, 540)
ax2.set_xlim(0, 500)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#CBD5E1')
ax2.spines['bottom'].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Reinforcement Learning Part 5',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '08-reinforcement-learning/05-advantage-actor-critic/header_a2c.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
