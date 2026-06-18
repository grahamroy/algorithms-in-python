"""Generate the header image for the Two-Tower Retrieval article."""

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

USER_FILL = '#dbeafe'
USER_BORDER = '#3B82F6'
ITEM_FILL = '#dcfce7'
ITEM_BORDER = '#16A34A'
INDEX_FILL = '#fef3c7'
INDEX_BORDER = '#F59E0B'
DOT_COLOR = '#7C3AED'
ARROW_COLOR = '#475569'

fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45, 'Two-Tower Retrieval: Deep Embeddings, Dot-Product Scoring',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Two deep towers meet only at a dot product — so item vectors index offline and retrieve by ANN in sub-linear time.',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# ===== LEFT: the two towers meeting at a dot product =====
# User tower (left), item tower (right of centre-left), dot in middle
def tower(cx, base_y, fill, border, label, layers):
    w = 2.0
    h = 0.6
    gap = 0.45
    boxes = []
    for i, lab in enumerate(layers):
        by = base_y + i * (h + gap)
        # taper: higher layers slightly narrower
        ww = w - i * 0.25
        ax.add_patch(FancyBboxPatch((cx - ww/2, by), ww, h,
                                    boxstyle='round,pad=0.02,rounding_size=0.08',
                                    facecolor=fill, edgecolor=border,
                                    linewidth=1.5, zorder=2))
        ax.text(cx, by + h/2, lab, fontsize=9, ha='center', va='center',
                color=TEXT_COLOR, fontfamily='monospace', zorder=3)
        boxes.append((cx, by, ww, h))
    # arrows between layers
    for i in range(len(layers) - 1):
        y0 = base_y + i * (h + gap) + h
        y1 = base_y + (i + 1) * (h + gap)
        ax.annotate('', xy=(cx, y1), xytext=(cx, y0),
                    arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.0))
    ax.text(cx, base_y - 0.4, label, fontsize=11, fontweight='bold',
            ha='center', va='center', color=border, fontfamily='sans-serif')
    return boxes

base = PANEL[1] + 1.4
user_x = PANEL[0] + 2.0
item_x = PANEL[0] + 5.4
user_boxes = tower(user_x, base, USER_FILL, USER_BORDER, 'User tower',
                   ['user id +\ncontext', 'dense + ReLU', 'u  (vector)'])
item_boxes = tower(item_x, base, ITEM_FILL, ITEM_BORDER, 'Item tower',
                   ['item id +\nfeatures', 'dense + ReLU', 'v  (vector)'])

# Dot product node between the tops of the towers
top_u = (user_x, base + 2 * (0.6 + 0.45) + 0.3)
top_v = (item_x, base + 2 * (0.6 + 0.45) + 0.3)
dot_x = (user_x + item_x) / 2
dot_y = top_u[1] + 1.0
ax.add_patch(plt.Circle((dot_x, dot_y), 0.45, facecolor='white',
                        edgecolor=DOT_COLOR, linewidth=2.0, zorder=3))
ax.text(dot_x, dot_y, 'u · v', fontsize=11, fontweight='bold',
        ha='center', va='center', color=DOT_COLOR,
        fontfamily='monospace', zorder=4)
ax.annotate('', xy=(dot_x - 0.35, dot_y - 0.25),
            xytext=(user_x, top_u[1] - 0.1),
            arrowprops=dict(arrowstyle='->', color=DOT_COLOR, lw=1.4))
ax.annotate('', xy=(dot_x + 0.35, dot_y - 0.25),
            xytext=(item_x, top_v[1] - 0.1),
            arrowprops=dict(arrowstyle='->', color=DOT_COLOR, lw=1.4))
ax.text(dot_x, dot_y + 0.75, 'score', fontsize=9, fontstyle='italic',
        ha='center', va='center', color=DOT_COLOR)

# ===== RIGHT: offline indexing + serving =====
serve_x = PANEL[0] + 9.0
serve_w = 6.0
ax.add_patch(FancyBboxPatch((serve_x, PANEL[1] + 0.5), serve_w,
                            PANEL[3] - 1.0,
                            boxstyle='round,pad=0.02,rounding_size=0.1',
                            facecolor='white', edgecolor=PANEL_EDGE,
                            linewidth=1.0, zorder=1))

ax.text(serve_x + serve_w/2, PANEL[1] + PANEL[3] - 0.9,
        'Why the dot product matters',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Offline box
ax.add_patch(FancyBboxPatch((serve_x + 0.4, PANEL[1] + 3.0),
                            serve_w - 0.8, 1.3,
                            boxstyle='round,pad=0.02,rounding_size=0.08',
                            facecolor=INDEX_FILL, edgecolor=INDEX_BORDER,
                            linewidth=1.5, zorder=2))
ax.text(serve_x + serve_w/2, PANEL[1] + 3.95, 'OFFLINE (one-off)',
        fontsize=9.5, fontweight='bold', ha='center', va='center',
        color=INDEX_BORDER, fontfamily='sans-serif')
ax.text(serve_x + serve_w/2, PANEL[1] + 3.4,
        'item tower → all m item vectors → ANN index (HNSW)',
        fontsize=9, ha='center', va='center',
        color=TEXT_COLOR, fontfamily='monospace')

# Online box
ax.add_patch(FancyBboxPatch((serve_x + 0.4, PANEL[1] + 1.2),
                            serve_w - 0.8, 1.3,
                            boxstyle='round,pad=0.02,rounding_size=0.08',
                            facecolor=USER_FILL, edgecolor=USER_BORDER,
                            linewidth=1.5, zorder=2))
ax.text(serve_x + serve_w/2, PANEL[1] + 2.15, 'PER REQUEST',
        fontsize=9.5, fontweight='bold', ha='center', va='center',
        color=USER_BORDER, fontfamily='sans-serif')
ax.text(serve_x + serve_w/2, PANEL[1] + 1.6,
        'user tower → u → ANN lookup → top-k   (O(log m))',
        fontsize=9, ha='center', va='center',
        color=TEXT_COLOR, fontfamily='monospace')

ax.text(8, 0.3,
        'Algorithms in Python  |  Recommender Systems Part 3',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '07-recommender-systems/03-two-tower-retrieval/header_two_tower.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
