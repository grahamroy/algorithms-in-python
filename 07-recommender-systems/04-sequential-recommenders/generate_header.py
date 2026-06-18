"""Generate the header image for the Sequential Recommenders article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

ITEM_FILL = '#dbeafe'
ITEM_BORDER = '#3B82F6'
PRED_FILL = '#dcfce7'
PRED_BORDER = '#16A34A'
ATT_COLOR = '#7C3AED'
ARROW_COLOR = '#475569'

fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45, 'Sequential Recommenders: Next-Item Prediction as Language Modelling',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Causal self-attention over the ordered history predicts the next item — the same machinery as a GPT decoder.',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# Row of item tokens (history) along the bottom
labels = ['i₁', 'i₂', 'i₃', 'i₄', 'i₅']
n = len(labels)
box_w = 1.4
box_h = 0.9
gap = 0.9
start_x = PANEL[0] + 1.2
row_y = PANEL[1] + 1.2

centres = []
for i, lab in enumerate(labels):
    cx = start_x + i * (box_w + gap)
    ax.add_patch(FancyBboxPatch((cx, row_y), box_w, box_h,
                                boxstyle='round,pad=0.02,rounding_size=0.1',
                                facecolor=ITEM_FILL, edgecolor=ITEM_BORDER,
                                linewidth=1.6, zorder=2))
    ax.text(cx + box_w/2, row_y + box_h/2, lab,
            fontsize=13, fontweight='bold', ha='center', va='center',
            color=TEXT_COLOR, fontfamily='monospace', zorder=3)
    centres.append(cx + box_w/2)

ax.text(start_x - 0.6, row_y + box_h/2, 'history',
        fontsize=10, fontstyle='italic', ha='right', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Representation node above the last item
rep_y = row_y + 2.4
rep_x = centres[-1]
ax.add_patch(plt.Circle((rep_x, rep_y), 0.55, facecolor='white',
                        edgecolor=ATT_COLOR, linewidth=2.0, zorder=4))
ax.text(rep_x, rep_y, 'r₅', fontsize=12, fontweight='bold',
        ha='center', va='center', color=ATT_COLOR,
        fontfamily='monospace', zorder=5)

# Causal attention arrows: r5 attends to items 1..5 (curved, weighted)
weights = [0.10, 0.12, 0.20, 0.25, 0.33]
for cx, w in zip(centres, weights):
    ax.add_patch(FancyArrowPatch((cx, row_y + box_h),
                                 (rep_x, rep_y - 0.5),
                                 arrowstyle='-', color=ATT_COLOR,
                                 lw=0.5 + w * 8, alpha=0.25 + w,
                                 connectionstyle='arc3,rad=-0.2',
                                 zorder=1))
ax.text((start_x + rep_x) / 2 - 1.0, rep_y + 0.2,
        'causal self-attention\n(weights past items)',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=ATT_COLOR, fontfamily='sans-serif')

# Prediction box above-right of the representation (kept inside panel)
pred_w = 3.6
pred_x = rep_x - pred_w / 2
pred_y = rep_y + 1.05
ax.add_patch(FancyBboxPatch((pred_x, pred_y - box_h/2), pred_w, box_h,
                            boxstyle='round,pad=0.02,rounding_size=0.1',
                            facecolor=PRED_FILL, edgecolor=PRED_BORDER,
                            linewidth=1.6, zorder=2))
ax.text(rep_x, pred_y, 'predict i₆ = argmax r₅ · vⱼ',
        fontsize=10, fontweight='bold', ha='center', va='center',
        color=PRED_BORDER, fontfamily='monospace', zorder=3)
ax.annotate('', xy=(rep_x, pred_y - box_h/2 - 0.02),
            xytext=(rep_x, rep_y + 0.55),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.6))

ax.text(PANEL[0] + PANEL[2]/2, PANEL[1] + 0.5,
        'Each position attends only to earlier items (causal mask); the last position predicts what comes next.',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

ax.text(8, 0.3,
        'Algorithms in Python  |  Recommender Systems Part 4',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '07-recommender-systems/04-sequential-recommenders/header_sasrec.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
