"""Generate the header image for the TD3 article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
CRITIC_COLOR = '#3B82F6'
MIN_COLOR = '#7C3AED'
BAD_COLOR = '#DC2626'
GOOD_COLOR = '#16A34A'
ACTUAL_COLOR = '#94A3B8'
BADGE_BG = '#F8FAFC'

# verified from td3.py (seed 0, 60 episodes)
SINGLE_Q, SINGLE_RET = 5.0, -4.6
TWIN_Q, TWIN_RET = -8.4, -4.9

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Twin Delayed DDPG: Fixing the Value That Lied',
        fontsize=18.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        "DDPG's critic overestimates and the actor exploits it. Take the MIN of "
        'two critics and the value stays honest.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: twin-critic min mechanism =====================
ax.text(4.0, 7.05, 'Clipped double-Q target', fontsize=11.5, fontweight='bold',
        ha='center', va='center', color=TEXT_COLOR)

def cbox(x, y, label):
    ax.add_patch(FancyBboxPatch((x, y), 2.0, 0.85,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=CRITIC_COLOR, linewidth=1.7))
    ax.text(x + 1.0, y + 0.42, label, fontsize=10.5, fontweight='bold',
            ha='center', va='center', color=CRITIC_COLOR, fontfamily='monospace')

cbox(0.7, 5.9, "Q1'(s',ã')")
cbox(0.7, 4.75, "Q2'(s',ã')")

# min node
ax.add_patch(Circle((4.2, 5.75), 0.5, facecolor=MIN_COLOR, edgecolor='none'))
ax.text(4.2, 5.75, 'min', fontsize=12, fontweight='bold', ha='center',
        va='center', color='white')
ax.add_patch(FancyArrowPatch((2.7, 6.32), (3.75, 5.95), arrowstyle='-|>',
             mutation_scale=12, color=CRITIC_COLOR, linewidth=1.5))
ax.add_patch(FancyArrowPatch((2.7, 5.17), (3.75, 5.55), arrowstyle='-|>',
             mutation_scale=12, color=CRITIC_COLOR, linewidth=1.5))
ax.add_patch(FancyArrowPatch((4.7, 5.75), (5.35, 5.75), arrowstyle='-|>',
             mutation_scale=13, color=MIN_COLOR, linewidth=1.8))
ax.text(5.55, 5.75, r'$y = r + \gamma\,\min(Q_1'', Q_2'')$',
        fontsize=11, ha='left', va='center', color=TEXT_COLOR)

# three fixes
def badge(y, num, title):
    ax.add_patch(FancyBboxPatch((0.7, y), 6.6, 0.72,
                 boxstyle='round,pad=0.03,rounding_size=0.08',
                 facecolor=BADGE_BG, edgecolor=MIN_COLOR, linewidth=1.3))
    ax.add_patch(Circle((1.1, y + 0.36), 0.22, facecolor=MIN_COLOR))
    ax.text(1.1, y + 0.36, num, fontsize=10.5, fontweight='bold', ha='center',
            va='center', color='white')
    ax.text(1.5, y + 0.36, title, fontsize=9.6, ha='left', va='center',
            color=TEXT_COLOR, fontweight='bold')

badge(3.35, '1', 'twin critics + min  — kills overestimation')
badge(2.4, '2', 'delayed policy updates — let the value settle')
badge(1.45, '3', 'target smoothing — no sharp fake peaks')

# ===================== RIGHT: overestimation bar chart =====================
ax2 = fig.add_axes([0.585, 0.15, 0.37, 0.56])
groups = ['Single critic\n(DDPG-style)', 'TD3\n(twin min)']
xg = [0, 1.4]
w = 0.42
# predicted Q bars
ax2.bar(xg[0] - w/2, SINGLE_Q, w, color=BAD_COLOR, label='predicted Q')
ax2.bar(xg[1] - w/2, TWIN_Q, w, color=GOOD_COLOR)
# actual return bars
ax2.bar(xg[0] + w/2, SINGLE_RET, w, color=ACTUAL_COLOR, label='actual return')
ax2.bar(xg[1] + w/2, TWIN_RET, w, color=ACTUAL_COLOR)

ax2.axhline(0, color=TEXT_COLOR, linewidth=1.2)
ax2.text(1.7, 0.5, 'true value ≤ 0\n(all rewards ≤ 0)', fontsize=8,
         color=SUBTLE_TEXT, ha='right', va='bottom', fontstyle='italic')
ax2.annotate('impossible:\nQ > 0', xy=(xg[0] - w/2, SINGLE_Q),
             xytext=(xg[0] - 0.75, 8.5), fontsize=8.5, color=BAD_COLOR,
             fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color=BAD_COLOR, lw=1.1))
ax2.text(xg[0] - w/2, SINGLE_Q + 0.4, f'+{SINGLE_Q:.0f}', ha='center',
         va='bottom', fontsize=9, color=BAD_COLOR, fontweight='bold')
ax2.text(xg[1] - w/2, TWIN_Q - 0.4, f'{TWIN_Q:.0f}', ha='center',
         va='top', fontsize=9, color=GOOD_COLOR, fontweight='bold')

ax2.set_xticks(xg)
ax2.set_xticklabels(groups, fontsize=9, color=TEXT_COLOR)
ax2.set_ylabel('value', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('Predicted Q vs. the true return',
              fontsize=11.5, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.legend(fontsize=8, loc='lower right', frameon=False)
ax2.set_ylim(-13, 11)
ax2.set_xlim(-0.7, 2.0)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)
ax2.spines['left'].set_color('#CBD5E1')
ax2.spines['bottom'].set_visible(False)
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8, bottom=False)

ax.text(8, 0.2, 'Algorithms in Python  |  Advanced Reinforcement Learning Part 4',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '09-advanced-reinforcement-learning/04-twin-delayed-ddpg/header_td3.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
