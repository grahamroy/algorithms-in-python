"""Generate the header image for the Co-Training article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
A_COLOR = '#EA580C'
B_COLOR = '#2563EB'
POOL_COLOR = '#16A34A'
VETO_COLOR = '#DC2626'
BADGE_BG = '#F8FAFC'

# verified from co_training.py (DEMO 3, seeds 0-4)
SEEDS = [0, 1, 2, 3, 4]
SELF = [93.4, 99.8, 99.6, 94.2, 92.6]
CO = [99.8, 99.0, 98.8, 99.4, 96.0]
VETOES = [51, 4, 7, 39, 18]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, "Co-Training: Two Models That Grade Each Other's Homework",
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        'Two classifiers on two independent views label examples for each '
        'other — and veto each other\'s mistakes.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the two-judge loop =====================
def box(x, y, w, h, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.8))
    ax.text(x + w/2, y + h*0.62, title, fontsize=10.5, fontweight='bold',
            ha='center', va='center', color=col)
    ax.text(x + w/2, y + h*0.25, sub, fontsize=7.8, ha='center', va='center',
            color=SUBTLE_TEXT)

box(0.6, 6.0, 2.9, 1.05, 'JUDGE A', 'k-NN on view A', A_COLOR)
box(4.35, 6.0, 2.9, 1.05, 'JUDGE B', 'k-NN on view B', B_COLOR)

# veto node
ax.add_patch(Circle((3.95, 4.55), 0.52, facecolor='white',
                    edgecolor=VETO_COLOR, linewidth=2.0))
ax.text(3.95, 4.62, 'agree?', fontsize=9, fontweight='bold', ha='center',
        va='center', color=VETO_COLOR)
ax.text(3.95, 4.33, 'veto if not', fontsize=6.8, ha='center', va='center',
        color=VETO_COLOR)

# nominations down into veto node
ax.add_patch(FancyArrowPatch((2.05, 5.95), (3.55, 4.85), arrowstyle='-|>',
             mutation_scale=12, color=A_COLOR, linewidth=1.6))
ax.add_patch(FancyArrowPatch((5.8, 5.95), (4.35, 4.85), arrowstyle='-|>',
             mutation_scale=12, color=B_COLOR, linewidth=1.6))
ax.text(2.2, 5.45, 'nominates\nconfident points', fontsize=7.6, ha='center',
        va='center', color=A_COLOR)
ax.text(5.75, 5.45, 'nominates\nconfident points', fontsize=7.6, ha='center',
        va='center', color=B_COLOR)

# vetoed out
ax.add_patch(FancyArrowPatch((4.45, 4.4), (5.45, 3.95), arrowstyle='-|>',
             mutation_scale=11, color=VETO_COLOR, linewidth=1.4,
             linestyle='--'))
ax.text(6.15, 3.9, 'x vetoed\n(disagreement =\nerror caught)', fontsize=7.6,
        ha='center', va='center', color=VETO_COLOR)

# pool
box(2.35, 2.6, 3.2, 1.0, 'SHARED POOL', 'labels for BOTH judges', POOL_COLOR)
ax.add_patch(FancyArrowPatch((3.95, 4.0), (3.95, 3.65), arrowstyle='-|>',
             mutation_scale=12, color=POOL_COLOR, linewidth=1.6))

# retrain arrows back up
ax.add_patch(FancyArrowPatch((2.6, 3.62), (1.65, 5.95),
             connectionstyle='arc3,rad=0.25', arrowstyle='-|>',
             mutation_scale=11, color='#94A3B8', linewidth=1.3,
             linestyle=':'))
ax.add_patch(FancyArrowPatch((5.35, 3.62), (6.2, 5.95),
             connectionstyle='arc3,rad=-0.25', arrowstyle='-|>',
             mutation_scale=11, color='#94A3B8', linewidth=1.3,
             linestyle=':'))
ax.text(0.95, 4.6, 'retrain', fontsize=7.5, ha='center', va='center',
        color=SUBTLE_TEXT, rotation=72)
ax.text(6.9, 4.6, 'retrain', fontsize=7.5, ha='center', va='center',
        color=SUBTLE_TEXT, rotation=-72)

ax.text(3.95, 1.85,
        'each judge labels examples FOR the other --\na mistake must fool two'
        ' independent views',
        fontsize=8.8, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: reliability across seeds =====================
ax2 = fig.add_axes([0.585, 0.185, 0.385, 0.55])
x = np.arange(5)
ax2.plot(x, SELF, 'o--', color='#94A3B8', linewidth=1.8, markersize=8,
         label='self-training (one judge)')
ax2.plot(x, CO, 'o-', color=POOL_COLOR, linewidth=2.4, markersize=8,
         label='co-training (two judges)')
for xi, (s, c, v) in enumerate(zip(SELF, CO, VETOES)):
    ax2.annotate(f'{v} vetoes', xy=(xi, c), xytext=(xi, 100.6), fontsize=7.5,
                 color=VETO_COLOR, ha='center')
ax2.axhspan(96, 100, color=POOL_COLOR, alpha=0.06)
ax2.text(2.2, 96.15, 'co-training never leaves this band', fontsize=8,
         color=POOL_COLOR, ha='center', va='bottom', fontstyle='italic')
ax2.annotate('one judge is fine\nuntil it is not', xy=(4, 92.6),
             xytext=(2.7, 92.9), fontsize=8.5, color='#6B7280',
             ha='center', arrowprops=dict(arrowstyle='->', color='#94A3B8',
                                          lw=1.1))
ax2.set_xticks(x)
ax2.set_xticklabels([f'dataset {i}' for i in x], fontsize=8.5,
                    color=TEXT_COLOR)
ax2.set_ylabel('test accuracy (%)', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('Five datasets, same budget: two judges = insurance',
              fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=14)
ax2.legend(fontsize=8.5, loc='lower left', frameon=False)
ax2.set_ylim(91.5, 101.4)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    ax2.spines[sp].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 2',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/02-co-training/header_co_training.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
