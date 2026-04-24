"""Generate a Big-O / memory summary table image for the PDS article."""

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
TIME_COLOR = '#059669'    # green
MEM_COLOR = '#3b82f6'     # blue
ERR_COLOR = '#DC2626'     # red
NOTE_COLOR = '#6B7280'

# ── Table data ──
HEADERS = ['Structure', 'Insert / Query', 'Memory', 'Error type', 'Typical scale']
ROWS = [
    ['Bloom filter',
     'O(k)',
     'O(m) bits',
     'one-sided FP',
     '~10 bits/item @ 1% FPR'],
    ['Counting Bloom filter',
     'O(k)',
     'O(m \u00b7 c) bits',
     'one-sided FP',
     'supports deletions; c-bit counters per slot'],
    ['Count-Min Sketch',
     'O(d)',
     'O(d \u00b7 w)',
     'one-sided over-count',
     '~50 KB streams unbounded vocab'],
    ['HyperLogLog',
     'O(1)',
     'O(m) bytes',
     '\u00b1 1.04/\u221am',
     '~12 KB \u2192 1% error on 10\u2079 items'],
    ['MinHash',
     'O(k)',
     'O(k) per set',
     '\u00b1 1/\u221ak',
     'k = 128 signatures \u2192 ~9% error on Jaccard'],
]

# ── Layout parameters ──
COL_WIDTHS = [4.2, 2.6, 2.4, 3.4, 5.4]
TOTAL_W = sum(COL_WIDTHS)
HEADER_H = 0.9
ROW_H = 0.8
N_ROWS = len(ROWS)
TABLE_W = TOTAL_W
TABLE_H = HEADER_H + N_ROWS * ROW_H

FIG_W_PX, FIG_H_PX = 1700, 700
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
        'Probabilistic Data Structure Complexity',
        fontsize=16, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

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
            fontsize=12.5, fontweight='bold', ha='center', va='center',
            color=HEADER_TEXT, fontfamily='sans-serif', zorder=2)

# ── Data rows ──
for r, row in enumerate(ROWS):
    row_y = Y0 + (N_ROWS - 1 - r) * ROW_H
    bg = ROW_EVEN if r % 2 == 0 else ROW_ODD
    ax.add_patch(Rectangle((X0, row_y), TOTAL_W, ROW_H,
                           facecolor=bg, edgecolor=BORDER, linewidth=0.8, zorder=1))
    # Structure (left-aligned, bold)
    ax.text(col_x(0) + 0.25, row_y + ROW_H / 2, row[0],
            fontsize=11.5, fontweight='bold', ha='left', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=2)
    # Insert/Query (green, mono)
    ax.text(col_x(1) + COL_WIDTHS[1] / 2, row_y + ROW_H / 2, row[1],
            fontsize=11, ha='center', va='center',
            color=TIME_COLOR, fontfamily='DejaVu Sans Mono', zorder=2)
    # Memory (blue, mono)
    ax.text(col_x(2) + COL_WIDTHS[2] / 2, row_y + ROW_H / 2, row[2],
            fontsize=11, ha='center', va='center',
            color=MEM_COLOR, fontfamily='DejaVu Sans Mono', zorder=2)
    # Error type (red, italic)
    ax.text(col_x(3) + COL_WIDTHS[3] / 2, row_y + ROW_H / 2, row[3],
            fontsize=10.5, ha='center', va='center', fontstyle='italic',
            color=ERR_COLOR, fontfamily='sans-serif', zorder=2)
    # Notes
    ax.text(col_x(4) + 0.25, row_y + ROW_H / 2, row[4],
            fontsize=10.5, ha='left', va='center',
            color=NOTE_COLOR, fontfamily='sans-serif', fontstyle='italic',
            zorder=2)

# ── Outer border ──
ax.add_patch(Rectangle((X0, Y0), TOTAL_W, TABLE_H,
                       facecolor='none', edgecolor=BORDER, linewidth=1.2, zorder=3))
for i in range(1, len(COL_WIDTHS)):
    x = col_x(i)
    ax.plot([x, x], [Y0, header_y + HEADER_H], color=BORDER, linewidth=0.8, zorder=3)

# ── Caption ──
caption_y = Y0 - 0.45
ax.text(TABLE_W / 2 + 1.0, caption_y,
        'k = hashes (Bloom) or signatures (MinHash); m = bits/registers; d, w = sketch dimensions; n = items in stream.',
        fontsize=10, ha='center', va='center',
        color='#6B7280', fontfamily='sans-serif', fontstyle='italic')

plt.tight_layout()
out_path = 'D:/Projects/Medium/algorithms-in-python/00-foundations/10-probabilistic-data-structures/bigo_table.png'
plt.savefig(out_path, dpi=DPI, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none', pad_inches=0.2)
plt.close(fig)
print(f'Wrote {out_path}')
