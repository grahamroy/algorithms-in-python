"""Generate the header image for the Tri-Training article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
M_COLORS = ['#EA580C', '#2563EB', '#7C3AED']
GATE_C = '#DC2626'
OK_C = '#16A34A'
BASE_C = '#94A3B8'
BADGE_BG = '#F8FAFC'

# verified from tri_training.py (DEMO 2)
SELF = [85.4, 59.6, 51.6, 87.0, 85.2]
BASE = [89.6, 83.6, 67.4, 77.2, 82.0]
TRI = [96.0, 74.6, 80.6, 94.4, 85.4]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Tri-Training: Two Teachers, One Student, One Gate',
        fontsize=18.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        "Co-training's error check without the views: three bootstrap "
        'classifiers where any two agreeing teach the third — if the gate '
        'allows it.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the classroom =====================
def clf_box(x, y0, label, col):
    ax.add_patch(FancyBboxPatch((x, y0), 1.95, 0.85,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.9))
    ax.text(x + 0.975, y0 + 0.55, label, fontsize=10, fontweight='bold',
            ha='center', va='center', color=col)
    ax.text(x + 0.975, y0 + 0.22, '1-NN on a bootstrap', fontsize=7,
            ha='center', va='center', color=SUBTLE_TEXT)

clf_box(0.7, 6.2, 'TEACHER h1', M_COLORS[0])
clf_box(3.05, 6.2, 'TEACHER h2', M_COLORS[1])
clf_box(1.85, 2.3, 'STUDENT h3', M_COLORS[2])

# agreement node
ax.add_patch(Circle((2.85, 5.35), 0.42, facecolor='white',
                    edgecolor=TEXT_COLOR, linewidth=1.6))
ax.text(2.85, 5.35, 'agree?', fontsize=8, fontweight='bold', ha='center',
        va='center', color=TEXT_COLOR)
ax.add_patch(FancyArrowPatch((1.7, 6.15), (2.55, 5.65), arrowstyle='-|>',
             mutation_scale=11, color=M_COLORS[0], linewidth=1.5))
ax.add_patch(FancyArrowPatch((4.0, 6.15), (3.15, 5.65), arrowstyle='-|>',
             mutation_scale=11, color=M_COLORS[1], linewidth=1.5))

# gate
ax.add_patch(FancyBboxPatch((1.5, 3.75), 2.7, 0.95,
             boxstyle='round,pad=0.04,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=GATE_C, linewidth=1.9))
ax.text(2.85, 4.45, 'THE GATE', fontsize=9.5, fontweight='bold',
        ha='center', va='center', color=GATE_C)
ax.text(2.85, 4.05, 'admit only if  e · |batch|  shrinks\n(subsample to force it)',
        fontsize=7.4, ha='center', va='center', color=TEXT_COLOR)
ax.add_patch(FancyArrowPatch((2.85, 4.9), (2.85, 4.73), arrowstyle='-|>',
             mutation_scale=11, color=TEXT_COLOR, linewidth=1.5))
ax.add_patch(FancyArrowPatch((2.85, 3.7), (2.85, 3.2), arrowstyle='-|>',
             mutation_scale=12, color=OK_C, linewidth=1.7))
ax.text(3.15, 3.45, 'pseudo-labels', fontsize=7.5, ha='left', va='center',
        color=OK_C)

ax.text(5.6, 5.0, 'roles rotate:\nevery classifier is\nthe student in turn',
        fontsize=8.5, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')
ax.text(5.6, 3.0, 'predict by\nmajority vote', fontsize=8.5, ha='center',
        va='center', color=SUBTLE_TEXT, fontstyle='italic')

ax.add_patch(FancyBboxPatch((0.7, 1.0), 6.6, 0.85,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=M_COLORS[1], linewidth=1.4))
ax.text(4.0, 1.62, 'no views needed', fontsize=9, fontweight='bold',
        ha='center', va='center', color=M_COLORS[1])
ax.text(4.0, 1.26, 'diversity comes from bootstraps + an unstable learner',
        fontsize=8.3, ha='center', va='center', color=SUBTLE_TEXT)

# ===================== RIGHT: the scoreboard =====================
ax2 = fig.add_axes([0.585, 0.16, 0.385, 0.55])
x = np.arange(5)
ax2.plot(x, SELF, 'o--', color=BASE_C, linewidth=1.6, markersize=7,
         label='self-training (Part 1)')
ax2.plot(x, BASE, 's--', color=M_COLORS[1], linewidth=1.6, markersize=6,
         alpha=0.7, label='bagged baseline (labels only)')
ax2.plot(x, TRI, 'o-', color=OK_C, linewidth=2.4, markersize=8,
         label='tri-training')
ax2.annotate('+17 over its\nbaseline', xy=(3, 94.4), xytext=(2.15, 97.5),
             fontsize=8.5, color=OK_C, fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color=OK_C, lw=1.1))
ax2.annotate('the honest loss:\nthe 8-point error\nestimate misfires',
             xy=(1, 74.6), xytext=(0.8, 60),
             fontsize=8, color=GATE_C, ha='center',
             arrowprops=dict(arrowstyle='->', color=GATE_C, lw=1.1))
ax2.set_xticks(x)
ax2.set_xticklabels([f'draw {i}' for i in x], fontsize=8.5, color=TEXT_COLOR)
ax2.set_ylabel('test accuracy (%)', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title("Part 1's five hard draws: means 73.8 / 80.0 / 86.2",
              fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.legend(fontsize=7.8, loc='lower right', frameon=False)
ax2.set_ylim(48, 101)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    ax2.spines[sp].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 7',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/07-tri-training/header_tri_training.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
