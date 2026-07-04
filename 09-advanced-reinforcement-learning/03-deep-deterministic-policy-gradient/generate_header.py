"""Generate the header image for the DDPG article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ACTOR_COLOR = '#7C3AED'
CRITIC_COLOR = '#3B82F6'
GRAD_COLOR = '#16A34A'
CURVE_COLOR = '#EA580C'
BADGE_BG = '#F8FAFC'

ANGLE = [180, 168, 147, 133, 159, 138, 91, 113, 162, 76, 36, 9, 3, 7, 9, 10, 10,
         9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 10, 10, 10, 10, 10, 9,
         10, 10, 10, 10]
STEP = list(range(0, 200, 5))

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Deep Deterministic Policy Gradient: DQN for Continuous Control',
        fontsize=17.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        "Replace DQN's impossible argmax with an actor that outputs the action, "
        'trained by ascending the critic.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: DDPG architecture =====================
def box(x, y, w, h, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.8))
    ax.text(x + w/2, y + h*0.62, title, fontsize=10.5, fontweight='bold',
            ha='center', va='center', color=col, fontfamily='monospace')
    ax.text(x + w/2, y + h*0.24, sub, fontsize=7.6, ha='center', va='center',
            color=SUBTLE_TEXT)

box(0.5, 5.15, 1.5, 0.95, 'state s', '', TEXT_COLOR)
box(2.75, 5.15, 2.35, 0.95, 'ACTOR  μ(s)', 'continuous torque', ACTOR_COLOR)
box(5.85, 5.15, 2.35, 0.95, 'CRITIC  Q(s,a)', 'scores the action', CRITIC_COLOR)

ax.add_patch(FancyArrowPatch((2.05, 5.62), (2.7, 5.62), arrowstyle='-|>',
             mutation_scale=12, color='#94A3B8', linewidth=1.4))
ax.add_patch(FancyArrowPatch((5.15, 5.62), (5.8, 5.62), arrowstyle='-|>',
             mutation_scale=12, color=ACTOR_COLOR, linewidth=1.5))
ax.text(5.47, 5.86, 'a', fontsize=9, ha='center', va='center',
        color=ACTOR_COLOR, fontfamily='monospace', fontweight='bold')

# dQ/da gradient back to the actor (the deterministic policy gradient)
ax.add_patch(FancyArrowPatch((7.0, 6.1), (3.9, 6.1),
             connectionstyle='arc3,rad=-0.32', arrowstyle='-|>',
             mutation_scale=13, color=GRAD_COLOR, linewidth=1.8,
             linestyle='--'))
ax.text(5.45, 7.15, 'ascend Q:  dQ/da → dμ/dθ', fontsize=9,
        ha='center', va='center', color=GRAD_COLOR, fontweight='bold')

# DQN inheritance badges
def badge(x, num, title):
    ax.add_patch(FancyBboxPatch((x, 3.35), 3.55, 0.95,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=BADGE_BG, edgecolor=CRITIC_COLOR, linewidth=1.4))
    ax.add_patch(Circle((x + 0.4, 3.82), 0.24, facecolor=CRITIC_COLOR))
    ax.text(x + 0.4, 3.82, num, fontsize=11, fontweight='bold', ha='center',
            va='center', color='white')
    ax.text(x + 0.78, 3.82, title, fontsize=9.2, ha='left', va='center',
            color=TEXT_COLOR, fontweight='bold')

badge(0.5, '1', 'replay buffer — off-policy')
badge(4.35, '2', 'target nets — soft update')
ax.text(4.35, 2.75, 'inherited from DQN, now driving an actor for continuous actions',
        fontsize=8.4, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: the swing-up =====================
ax2 = fig.add_axes([0.6, 0.13, 0.36, 0.6])
ax2.plot(STEP, ANGLE, color=CURVE_COLOR, linewidth=2.4)
ax2.axhline(0, color='#CBD5E1', linewidth=1.0, linestyle='--')
ax2.text(195, 4, 'upright', fontsize=7.5, color='#94A3B8', va='bottom', ha='right')
ax2.annotate('hanging (180°)', xy=(0, 180), xytext=(18, 176),
             fontsize=9, color=CURVE_COLOR, va='center')
ax2.annotate('pumps to\nbuild energy', xy=(40, 159), xytext=(52, 150),
             fontsize=8.5, color=SUBTLE_TEXT, va='center')
ax2.annotate('caught &\nbalanced (~10°)', xy=(120, 10), xytext=(120, 62),
             fontsize=8.5, color=GRAD_COLOR, fontweight='bold', va='center',
             ha='center', arrowprops=dict(arrowstyle='->', color=GRAD_COLOR, lw=1.1))
ax2.set_xlabel('step in the episode', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_ylabel('angle from upright (deg)', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('Swing-up: hanging → upright',
              fontsize=11.5, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.set_ylim(-8, 195)
ax2.set_xlim(0, 200)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    ax2.spines[sp].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Advanced Reinforcement Learning Part 3',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '09-advanced-reinforcement-learning/03-deep-deterministic-policy-gradient/'
       'header_ddpg.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
