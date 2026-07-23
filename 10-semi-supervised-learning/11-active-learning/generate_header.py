"""Generate the header image for the Active Learning article.

Right panels show REAL results from active_learning.py: the 120-draw
label lottery histogram and the four strategies' mean race curves.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import active_learning as AL

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'
BASE_C = '#94A3B8'

# ---- reproduce the script's results --------------------------------------
rng = np.random.default_rng(AL.RNG_SEED)
X, y = AL.make_moons(250, noise=0.15, rng=rng)
X_test, y_test = AL.make_moons(250, noise=0.15, rng=rng)
ALL = np.concatenate([X, X_test])
P, rho = AL.build_graph(ALL)
rho_pool = rho[:AL.N_POOL]

lottery = []
for i in range(AL.LOTTERY):
    d = np.random.default_rng(1000 + i)
    li = np.concatenate([d.choice(np.where(y == 0)[0], 4, replace=False),
                         d.choice(np.where(y == 1)[0], 4, replace=False)])
    F = AL.propagate(P, li, y[li])
    lottery.append(float((F.argmax(1)[AL.N_POOL:] == y_test).mean()))
lottery = np.array(lottery)
print(f'lottery: min {lottery.min():.1%} median {np.median(lottery):.1%}')
assert abs(lottery.min() - 0.722) < 1e-9

STRATS = ['random', 'uncertainty', 'info-density', 'farthest-first']
curves = {}
for s in STRATS:
    cs = [AL.run_strategy(s, seed, X, y, P, rho_pool, y_test)[0]
          for seed in range(AL.SEEDS)]
    curves[s] = np.array(cs).mean(axis=0)
print({s: f'{c[-1]:.1%}' for s, c in curves.items()})
assert abs(curves['info-density'][-1] - 0.986) < 5e-4

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Active Learning: Ask Where Confusion Matters',
        fontsize=19, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Ten methods took their eight labels as luck. The finale chooses '
        'them — and buys a floor, not a ceiling.',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the loop =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

steps = [
    (6.45, TEXT_COLOR, 'train on the labels you have',
     "Part 5's label propagation, unchanged"),
    (5.05, PURPLE, 'the model NOMINATES a point',
     'the strategy: whose label do I want next?'),
    (3.65, RED, 'pay the oracle', 'a human answers; the budget shrinks'),
]
for y0, col, title, sub in steps:
    box(0.8, y0, 4.7, 1.0, col, face=BADGE_BG if col != TEXT_COLOR else 'white')
    ax.text(3.15, y0 + 0.66, title, fontsize=9.6, fontweight='bold',
            ha='center', va='center', color=col)
    ax.text(3.15, 0.30 + y0, sub, fontsize=7.9, ha='center', va='center',
            color=SUBTLE_TEXT)
for y1, y2 in ((6.4, 6.1), (5.0, 4.7)):
    ax.add_patch(FancyArrowPatch((3.15, y1), (3.15, y2), arrowstyle='-|>',
                 mutation_scale=12, color=TEXT_COLOR, linewidth=1.6))
# loop-back arrow
ax.add_patch(FancyArrowPatch((0.75, 4.15), (0.75, 6.95),
             arrowstyle='-|>', mutation_scale=12, color=GREEN,
             linewidth=1.7, connectionstyle='arc3,rad=0.5'))
ax.text(0.22, 5.55, 'repeat', fontsize=8.5, ha='center', va='center',
        color=GREEN, rotation=90, fontweight='bold')

box(0.8, 1.1, 4.7, 2.0, GREEN, lw=1.6)
ax.text(3.15, 2.78, 'the nomination strategies', fontsize=9.4,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.15, 1.95,
        'uncertainty: smallest belief margin\n'
        'info-density: uncertain AND representative\n'
        'farthest-first: pure coverage, no opinion\n'
        'random: the control to beat',
        fontsize=8.0, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the lottery =====================
axl = fig.add_axes([0.475, 0.50, 0.50, 0.285])
bins = np.arange(0.70, 1.001, 0.01)
cnt, edges = np.histogram(lottery, bins=bins)
cols = [RED if e < 0.90 else BLUE for e in edges[:-1]]
axl.bar(edges[:-1], cnt, width=0.0092, align='edge', color=cols, alpha=0.85)
axl.axvline(np.median(lottery), color=TEXT_COLOR, linewidth=1.4,
            linestyle='--')
axl.text(np.median(lottery) - 0.007, cnt.max() * 0.78,
         f'median {np.median(lottery):.1%}', fontsize=8, ha='right',
         va='top', color=TEXT_COLOR)
axl.annotate(f'{(lottery < 0.90).sum()} of {AL.LOTTERY} tickets\n'
             f'below 90% — min {lottery.min():.1%}',
             xy=(0.80, 1.5), xytext=(0.735, cnt.max() * 0.55),
             fontsize=8.3, color=RED, fontweight='bold', ha='left',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
axl.set_xlim(0.70, 1.0)
axl.set_xticks([0.7, 0.8, 0.9, 1.0])
axl.set_xticklabels(['70%', '80%', '90%', '100%'], fontsize=8)
axl.set_ylabel('draws', fontsize=8.5, color=SUBTLE_TEXT)
axl.set_title('the label lottery: 120 random 8-label draws, one learner',
              fontsize=10, fontweight='bold', color=TEXT_COLOR, pad=5,
              loc='left')
for sp in ('top', 'right'):
    axl.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axl.spines[sp].set_color('#CBD5E1')
axl.tick_params(colors=SUBTLE_TEXT, labelsize=8)

# ===================== RIGHT BOTTOM: the race =====================
axr = fig.add_axes([0.475, 0.085, 0.50, 0.285])
xs = np.arange(2, 9)
style = {'random': (BASE_C, '--', 'o'),
         'uncertainty': (ORANGE, '-', 's'),
         'info-density': (GREEN, '-', 'o'),
         'farthest-first': (PURPLE, ':', '^')}
for s in STRATS:
    col, ls, mk = style[s]
    axr.plot(xs, 100 * curves[s], ls, marker=mk, color=col, linewidth=2.0,
             markersize=5, label=s)
axr.annotate('~98.5% by SIX labels', xy=(5.92, 98.2), xytext=(5.7, 89.0),
             fontsize=8.3, color=GREEN, fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2))
axr.annotate('the cold-start:\nuncertainty trails random',
             xy=(3.95, 92.0), xytext=(2.05, 76.0), fontsize=8,
             color=ORANGE, va='bottom',
             arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.1))
axr.set_xlim(1.8, 8.4)
axr.set_ylim(74, 101)
axr.set_xticks(xs)
axr.set_xlabel('labels bought', fontsize=8.5, color=SUBTLE_TEXT)
axr.set_ylabel('test accuracy (%)', fontsize=8.5, color=SUBTLE_TEXT)
axr.set_title('the race: mean over 12 starts — choosing beats drawing',
              fontsize=10, fontweight='bold', color=TEXT_COLOR, pad=5,
              loc='left')
axr.legend(fontsize=7.6, loc='lower right', frameon=False, ncol=2)
for sp in ('top', 'right'):
    axr.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axr.spines[sp].set_color('#CBD5E1')
axr.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 11',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/11-active-learning/header_active.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
