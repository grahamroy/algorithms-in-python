"""Generate the header image for the Offline RL article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
DOOM_COLOR = '#DC2626'
FIX_COLOR = '#16A34A'
DATA_COLOR = '#94A3B8'
BC_COLOR = '#3B82F6'
BADGE_BG = '#F8FAFC'

# verified from offline_rl.py (seed 0)
LABELS = ['naive\noffline TD3', 'the dataset\nitself', 'BC only\n(clone)',
          'TD3+BC']
VALS = [-1473.0, -643.0, -573.5, -358.2]
COLORS = [DOOM_COLOR, DATA_COLOR, BC_COLOR, FIX_COLOR]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, "Offline RL: Learning When You Can't Try Things",
        fontsize=18.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        'A fixed dataset, no environment — naive off-policy learning collapses; '
        'one anchoring term fixes it.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the doom loop + the fix =====================
ax.text(3.95, 7.2, 'Extrapolation error: the doom loop', fontsize=11.5,
        fontweight='bold', ha='center', va='center', color=DOOM_COLOR)

def step_box(x, y, num, text):
    ax.add_patch(FancyBboxPatch((x, y), 3.35, 0.82,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=DOOM_COLOR, linewidth=1.5))
    ax.text(x + 0.28, y + 0.41, num, fontsize=12, fontweight='bold',
            ha='center', va='center', color=DOOM_COLOR)
    ax.text(x + 0.55, y + 0.41, text, fontsize=8.4, ha='left', va='center',
            color=TEXT_COLOR)

step_box(0.45, 5.95, '1', 'critic asked about actions\nnot in the data')
step_box(4.15, 5.95, '2', 'the network guesses --\nsometimes far too high')
step_box(4.15, 4.65, '3', 'no new experience ever\ncorrects the guess')
step_box(0.45, 4.65, '4', 'the actor optimises toward\nthe phantom values')

arrows = [((3.85, 6.36), (4.1, 6.36)),
          ((5.85, 5.9), (5.85, 5.55)),
          ((4.1, 5.06), (3.85, 5.06)),
          ((2.1, 5.55), (2.1, 5.9))]
for (x0, y0), (x1, y1) in arrows:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                 mutation_scale=12, color=DOOM_COLOR, linewidth=1.5))

# the fix
ax.add_patch(FancyBboxPatch((0.45, 2.55), 7.05, 1.35,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=FIX_COLOR, linewidth=1.7))
ax.text(3.98, 3.6, 'the one-line fix (TD3+BC): anchor to the data',
        fontsize=9.5, fontweight='bold', ha='center', va='center',
        color=FIX_COLOR)
ax.text(3.98, 3.08,
        r'$\mathrm{actor\ loss} \; = \; -\,\lambda\, Q(s, \pi(s)) \;\; + \;\;'
        r' (\,\pi(s) - a_{data}\,)^2$',
        fontsize=12.5, ha='center', va='center', color=TEXT_COLOR)
ax.text(5.62, 2.78, 'the anchor', fontsize=8.5, ha='center', va='center',
        color=FIX_COLOR, fontstyle='italic')

ax.text(3.98, 2.05,
        'improve with Q only where the dataset can vouch for it',
        fontsize=8.8, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: the returns staircase =====================
ax2 = fig.add_axes([0.575, 0.145, 0.385, 0.58])
xs = np.arange(len(LABELS))
bars = ax2.bar(xs, VALS, color=COLORS, width=0.62)
ax2.axhline(0, color=TEXT_COLOR, linewidth=1.1)
ax2.axhline(-1300, color=DOOM_COLOR, linewidth=1.0, linestyle='--', alpha=0.6)
ax2.text(3.35, -1290, 'random policy', fontsize=7.8, color=DOOM_COLOR,
         ha='right', va='bottom', fontstyle='italic')
for x, v in zip(xs, VALS):
    ax2.text(x, v - 45, f'{v:.0f}', ha='center', va='top', fontsize=9.5,
             fontweight='bold',
             color=COLORS[int(x)] if v != VALS[1] else SUBTLE_TEXT)
ax2.annotate('worse than\nrandom!', xy=(0, -1420), xytext=(0.75, -1180),
             fontsize=9, color=DOOM_COLOR, fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color=DOOM_COLOR, lw=1.1))
ax2.annotate('beats the data\nit learned from', xy=(3.24, -375),
             xytext=(2.95, -730), fontsize=9, color=FIX_COLOR,
             fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color=FIX_COLOR, lw=1.1))
ax2.set_xlim(-0.55, 3.55)
ax2.set_xticks(xs)
ax2.set_xticklabels(LABELS, fontsize=8.6, color=TEXT_COLOR)
ax2.set_ylabel('return (0 is best)', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('Same fixed dataset, four outcomes',
              fontsize=11.5, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.set_ylim(-1600, 60)
for sp in ('top', 'right', 'bottom'):
    ax2.spines[sp].set_visible(False)
ax2.spines['left'].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8, bottom=False)

ax.text(8, 0.18,
        'Algorithms in Python  |  Advanced Reinforcement Learning Part 7 — '
        'the final article',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '09-advanced-reinforcement-learning/07-offline-reinforcement-learning/'
       'header_offline_rl.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
