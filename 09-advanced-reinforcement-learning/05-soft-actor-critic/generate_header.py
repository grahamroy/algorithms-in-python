"""Generate the header image for the SAC article."""

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
ENT_COLOR = '#16A34A'
ALPHA_COLOR = '#EA580C'
BADGE_BG = '#F8FAFC'

# verified from sac.py (seed 0): checkpoints at episodes 0(init),10..60
EPS = [0, 10, 20, 30, 40, 50, 60]
ALPHA = [0.200, 0.061, 0.068, 0.053, 0.044, 0.038, 0.030]
SIGMA = [1.0, 0.577, 0.385, 0.267, 0.287, 0.352, 0.352]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Soft Actor-Critic: Exploration Becomes Part of the Objective',
        fontsize=17.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        'A stochastic policy paid for its entropy — with a temperature that '
        'tunes itself to a target.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: objective + actor =====================
# objective box
ax.add_patch(FancyBboxPatch((0.6, 6.1), 6.9, 1.15,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=ENT_COLOR, linewidth=1.6))
ax.text(4.05, 6.98, 'the maximum-entropy objective', fontsize=9,
        fontweight='bold', ha='center', va='center', color=ENT_COLOR)
ax.text(4.05, 6.52,
        r'$J \; = \; \mathbb{E}\left[\ \sum_t \; r_t \; + \; '
        r'\alpha\, H(\,\pi(\cdot|s_t)\,)\ \right]$',
        fontsize=13.5, ha='center', va='center', color=TEXT_COLOR)

# actor pipeline
def box(x, y, w, h, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.8))
    ax.text(x + w/2, y + h*0.63, title, fontsize=10, fontweight='bold',
            ha='center', va='center', color=col, fontfamily='monospace')
    ax.text(x + w/2, y + h*0.25, sub, fontsize=7.5, ha='center', va='center',
            color=SUBTLE_TEXT)

box(0.6, 4.35, 1.5, 1.0, 'state s', '', TEXT_COLOR)
box(2.85, 4.35, 2.2, 1.0, 'ACTOR', 'outputs μ(s), σ(s)', ACTOR_COLOR)
box(5.8, 4.35, 2.75, 1.0, 'a = 2·tanh(μ+σε)', 'reparameterised sample',
    ACTOR_COLOR)
ax.add_patch(FancyArrowPatch((2.15, 4.85), (2.8, 4.85), arrowstyle='-|>',
             mutation_scale=12, color='#94A3B8', linewidth=1.4))
ax.add_patch(FancyArrowPatch((5.1, 4.85), (5.75, 4.85), arrowstyle='-|>',
             mutation_scale=12, color=ACTOR_COLOR, linewidth=1.5))

# badges
def badge(x, y, w, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.95,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=BADGE_BG, edgecolor=col, linewidth=1.4))
    ax.text(x + 0.25, y + 0.62, title, fontsize=9.3, fontweight='bold',
            ha='left', va='center', color=col)
    ax.text(x + 0.25, y + 0.27, sub, fontsize=7.8, ha='left', va='center',
            color=SUBTLE_TEXT)

badge(0.6, 2.9, 3.8, 'twin critics + min (from TD3)',
      'soft target:  r + γ(min Q′ − α log π)', CRITIC_COLOR)
badge(4.75, 2.9, 3.8, 'auto-tuned temperature α',
      'entropy above target → α falls', ALPHA_COLOR)

ax.text(4.05, 2.3,
        'gradients flow through the sample:  dQ/da · da/dμ  (no score function)',
        fontsize=8.6, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: the dial turns itself =====================
ax2 = fig.add_axes([0.605, 0.145, 0.355, 0.58])
ax2.plot(EPS, SIGMA, '-o', color=ENT_COLOR, linewidth=2.4, markersize=4.5,
         label='policy spread σ')
ax2.plot(EPS, ALPHA, '-o', color=ALPHA_COLOR, linewidth=2.4, markersize=4.5,
         label='temperature α')
ax2.annotate('(init)', xy=(0, 1.0), xytext=(3, 1.0), fontsize=8,
             color=SUBTLE_TEXT, va='center')
ax2.annotate('broad early,\ncommitted late', xy=(30, 0.267),
             xytext=(24, 0.62), fontsize=9, color=ENT_COLOR,
             fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color=ENT_COLOR, lw=1.1))
ax2.annotate('α turns itself\ndown ~7x', xy=(59, 0.045), xytext=(43, 0.155),
             fontsize=9, color=ALPHA_COLOR, fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color=ALPHA_COLOR, lw=1.1))
ax2.set_xlabel('training episode', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_ylabel('value', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('No noise schedule anywhere — the dial turns itself',
              fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.legend(fontsize=8.5, loc='upper right', frameon=False)
ax2.set_ylim(0, 1.1)
ax2.set_xlim(-2, 63)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    ax2.spines[sp].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Advanced Reinforcement Learning Part 5',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '09-advanced-reinforcement-learning/05-soft-actor-critic/header_sac.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
