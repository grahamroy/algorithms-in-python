"""Generate the header image for the Policy Gradient Methods article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_EDGE = '#E2E8F0'
MUTED = '#94A3B8'
POLICY_COLOR = '#7C3AED'
WITH_COLOR = '#16A34A'
WITHOUT_COLOR = '#EA580C'
BADGE_BG = '#F8FAFC'

WITH = [246.5, 379.4, 481.7, 468.5, 469.7, 477.3, 489.2, 495.4, 499.1, 487.5]
WITHOUT = [48.4, 28.9, 47.7, 38.3, 18.9, 9.6, 9.5, 9.5, 9.5, 9.4]
EP_MID = [40, 120, 200, 280, 360, 440, 520, 600, 680, 760]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Policy Gradient Methods: Learn the Policy, Not the Values',
        fontsize=18.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.02,
        'Optimise π(a | s) directly — push up the probability of actions that led to high return.',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ============ LEFT: value-based vs policy-based contrast ============
def lane(y, title, col, boxes, arrow_label):
    ax.text(0.6, y + 0.92, title, fontsize=10.5, fontweight='bold',
            ha='left', va='center', color=col)
    x = 0.6
    for i, (w, label, sub) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x, y), w, 0.72,
                     boxstyle='round,pad=0.04,rounding_size=0.08',
                     facecolor='white', edgecolor=col, linewidth=1.5))
        ax.text(x + w/2, y + 0.43, label, fontsize=8.8, ha='center',
                va='center', color=TEXT_COLOR, fontfamily='monospace',
                fontweight='bold')
        ax.text(x + w/2, y + 0.13, sub, fontsize=7, ha='center',
                va='center', color=SUBTLE_TEXT)
        x += w
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.04, y + 0.36),
                         (x + 0.42, y + 0.36), arrowstyle='-|>',
                         mutation_scale=10, color=col, linewidth=1.2))
            x += 0.46

# value-based lane (muted -- the previous approach)
lane(6.2, 'Value-based  (Q-Learning, DQN)', MUTED,
     [(1.6, 'state s', ''), (1.9, 'Q(s, a)', 'values'),
      (1.7, 'argmax', 'one action')], '')
ax.text(7.95, 6.56, 'deterministic', fontsize=8, ha='left',
        va='center', color=MUTED, fontstyle='italic')

# policy-based lane (highlighted -- this article)
lane(4.5, 'Policy-based  (REINFORCE)  — this article', POLICY_COLOR,
     [(1.6, 'state s', ''), (1.9, 'π(a | s)', 'softmax'),
      (1.7, 'sample', 'a distribution')], '')
ax.text(7.95, 4.86, 'stochastic', fontsize=8, ha='left',
        va='center', color=POLICY_COLOR, fontstyle='italic')

# policy gradient equation box
ax.add_patch(FancyBboxPatch((0.6, 2.55), 8.2, 1.15,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=POLICY_COLOR, linewidth=1.6))
ax.text(4.7, 3.28, 'the policy gradient', fontsize=9,
        ha='center', va='center', color=POLICY_COLOR, fontweight='bold')
ax.text(4.7, 2.92, r'$\nabla_\theta J = \mathbb{E}\,[\ \sum_t\ '
        r'\nabla_\theta \log \pi(a_t|s_t)\ \cdot\ (G_t - b)\ ]$',
        fontsize=14, ha='center', va='center', color=TEXT_COLOR)

# baseline badge
ax.add_patch(FancyBboxPatch((0.6, 1.0), 8.2, 1.2,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=WITH_COLOR, linewidth=1.6))
ax.text(0.95, 1.78, 'the one trick: a baseline  b', fontsize=10,
        fontweight='bold', ha='left', va='center', color=WITH_COLOR)
ax.text(0.95, 1.36,
        '(Gₜ − b) = advantage: "better or worse than average", not a raw return',
        fontsize=8.6, ha='left', va='center', color=SUBTLE_TEXT)

# ============ RIGHT: the learning curve ============
ax2 = fig.add_axes([0.605, 0.13, 0.355, 0.58])
ax2.plot(EP_MID, WITH, '-o', color=WITH_COLOR, linewidth=2.4,
         markersize=4, label='with baseline')
ax2.plot(EP_MID, WITHOUT, '-o', color=WITHOUT_COLOR, linewidth=2.4,
         markersize=4, label='without baseline')
ax2.axhline(500, color='#CBD5E1', linewidth=1.0, linestyle='--')
ax2.text(760, 500, ' max 500', fontsize=7.5, color='#94A3B8',
         va='bottom', ha='right')
ax2.annotate('490', xy=(760, 487.5), xytext=(610, 430),
             fontsize=11, color=WITH_COLOR, fontweight='bold')
ax2.annotate('9', xy=(760, 9.4), xytext=(700, 70),
             fontsize=11, color=WITHOUT_COLOR, fontweight='bold')
ax2.set_xlabel('training episodes', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_ylabel('mean return', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('The baseline makes it work',
              fontsize=11.5, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.legend(fontsize=8.5, loc='center right', frameon=False)
ax2.set_ylim(0, 540)
ax2.set_xlim(0, 800)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#CBD5E1')
ax2.spines['bottom'].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Reinforcement Learning Part 4',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '08-reinforcement-learning/04-policy-gradient-methods/header_policy_gradient.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
