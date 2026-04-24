"""Generate a Big-O summary table image for the knowledge graphs article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Colours (match generate_header.py palette) ──
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
HEADER_BG = '#1F2937'
HEADER_TEXT = '#FFFFFF'
ROW_EVEN = '#F8FAFC'
ROW_ODD = '#FFFFFF'
BORDER = '#E2E8F0'
LATENCY_COLOR = '#059669'   # green
SCALE_COLOR = '#3b82f6'     # blue
NOTE_COLOR = '#6B7280'

# ── Table data ──
HEADERS = ['Operation', 'Cost', 'Memory', 'Notes']
ROWS = [
    ['Triple lookup (indexed)',         'O(log n + k)',           'O(n)',          'k = matching triples; needs SPO / POS indexes'],
    ['k-hop pattern query',             'O(p_1 \u00b7 p_2 \u00b7 ...)', 'O(n)',     'cost = product of per-hop selectivities'],
    ['Full triple scan',                'O(n)',                   'O(n)',          'fallback when no usable index exists'],
    ['Embedding training (TransE)',     'O(E \u00b7 n \u00b7 d)', 'O(V \u00b7 d)', 'E = epochs, d = embedding dim'],
    ['Embedding query (h + r \u2192 t)','O(V \u00b7 d)',          'O(V \u00b7 d)', 'rank tail by L2 distance over all entities'],
    ['Reasoner-derived inference',      'O(rules \u00b7 n)',      'O(n)',          'OWL-style; often too expensive at scale'],
]

# ── Layout parameters ──
COL_WIDTHS = [5.5, 3.4, 2.4, 7.0]
TOTAL_W = sum(COL_WIDTHS)
HEADER_H = 0.9
ROW_H = 0.8
N_ROWS = len(ROWS)
TABLE_W = TOTAL_W
TABLE_H = HEADER_H + N_ROWS * ROW_H

FIG_W_PX, FIG_H_PX = 1600, 750
DPI = 150
fig, ax = plt.subplots(figsize=(FIG_W_PX/DPI, FIG_H_PX/DPI), dpi=DPI)
ax.set_xlim(0, TABLE_W + 2.0)
ax.set_ylim(0, TABLE_H + 2.0)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title above the table ──
title_y = TABLE_H + 1.3
ax.text(TABLE_W / 2 + 1.0, title_y,
        'Knowledge Graph Complexity --- Triples and Embeddings',
        fontsize=16, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Table origin
X0 = 1.0
Y0 = 0.8

def col_x(col_index):
    return X0 + sum(COL_WIDTHS[:col_index])

# ── Header row ──
header_y = Y0 + N_ROWS * ROW_H
ax.add_patch(Rectangle((X0, header_y), TOTAL_W, HEADER_H,
                       facecolor=HEADER_BG, edgecolor=HEADER_BG, linewidth=0, zorder=1))
for i, h in enumerate(HEADERS):
    x = col_x(i) + COL_WIDTHS[i] / 2
    ax.text(x, header_y + HEADER_H / 2, h,
            fontsize=13, fontweight='bold', ha='center', va='center',
            color=HEADER_TEXT, fontfamily='sans-serif', zorder=2)

# ── Data rows ──
for r, row in enumerate(ROWS):
    row_y = Y0 + (N_ROWS - 1 - r) * ROW_H
    bg = ROW_EVEN if r % 2 == 0 else ROW_ODD
    ax.add_patch(Rectangle((X0, row_y), TOTAL_W, ROW_H,
                           facecolor=bg, edgecolor=BORDER, linewidth=0.8, zorder=1))
    # Operation (left-aligned, bold)
    ax.text(col_x(0) + 0.25, row_y + ROW_H / 2, row[0],
            fontsize=11, fontweight='bold', ha='left', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=2)
    # Cost (green, monospace)
    ax.text(col_x(1) + COL_WIDTHS[1] / 2, row_y + ROW_H / 2, row[1],
            fontsize=11, ha='center', va='center',
            color=LATENCY_COLOR, fontfamily='DejaVu Sans Mono', zorder=2)
    # Memory (blue, monospace)
    ax.text(col_x(2) + COL_WIDTHS[2] / 2, row_y + ROW_H / 2, row[2],
            fontsize=11, ha='center', va='center',
            color=SCALE_COLOR, fontfamily='DejaVu Sans Mono', zorder=2)
    # Notes
    ax.text(col_x(3) + 0.25, row_y + ROW_H / 2, row[3],
            fontsize=10.5, ha='left', va='center',
            color=NOTE_COLOR, fontfamily='sans-serif', fontstyle='italic',
            zorder=2)

# ── Outer border ──
ax.add_patch(Rectangle((X0, Y0), TOTAL_W, TABLE_H,
                       facecolor='none', edgecolor=BORDER, linewidth=1.2, zorder=3))
for i in range(1, len(COL_WIDTHS)):
    x = col_x(i)
    ax.plot([x, x], [Y0, header_y + HEADER_H], color=BORDER, linewidth=0.8, zorder=3)

# ── Caption below the table ──
caption_y = Y0 - 0.45
ax.text(TABLE_W / 2 + 1.0, caption_y,
        'n = triples in store, V = entities, d = embedding dim.  Per-hop selectivity p depends on indexes and the query plan.',
        fontsize=10, ha='center', va='center',
        color='#6B7280', fontfamily='sans-serif', fontstyle='italic')

plt.tight_layout()
out_path = 'D:/Projects/Medium/algorithms-in-python/00-foundations/09-knowledge-graphs/bigo_table.png'
plt.savefig(out_path, dpi=DPI, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none', pad_inches=0.2)
plt.close(fig)
print(f'Wrote {out_path}')
