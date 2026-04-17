"""Generate a Big-O summary table image for the hash tables article."""

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
AVG_COLOR = '#059669'   # green for good case
WORST_COLOR = '#DC2626' # red for worst case

# ── Table data ──
HEADERS = ['Operation', 'Average', 'Worst case']
ROWS = [
    ['Lookup',      'O(1)',             'O(n)'],
    ['Insert',      'O(1) amortised',   'O(n)'],
    ['Delete',      'O(1)',             'O(n)'],
    ['Iterate all', 'O(n + m)',         'O(n + m)'],
]

# ── Layout parameters ──
COL_WIDTHS = [3.5, 4.5, 4.0]  # relative widths
TOTAL_W = sum(COL_WIDTHS)
HEADER_H = 0.9
ROW_H = 0.8
N_ROWS = len(ROWS)
TABLE_W = TOTAL_W
TABLE_H = HEADER_H + N_ROWS * ROW_H

# Figure: widescreen so Medium renders it at a comfortable size
FIG_W_PX, FIG_H_PX = 1200, 600
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
ax.text(TABLE_W / 2 + 1.0, title_y, 'Hash Table Complexity',
        fontsize=16, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Table origin (bottom-left); shift right to centre horizontally with some padding
X0 = 1.0
Y0 = 0.8  # bottom margin

def col_x(col_index):
    """Left x of column `col_index`."""
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
    # r=0 is the top data row, draw from top down
    row_y = Y0 + (N_ROWS - 1 - r) * ROW_H
    bg = ROW_EVEN if r % 2 == 0 else ROW_ODD
    ax.add_patch(Rectangle((X0, row_y), TOTAL_W, ROW_H,
                           facecolor=bg, edgecolor=BORDER, linewidth=0.8, zorder=1))
    # Operation (left-aligned, bold)
    ax.text(col_x(0) + 0.25, row_y + ROW_H / 2, row[0],
            fontsize=12, fontweight='bold', ha='left', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=2)
    # Average (green, monospace)
    ax.text(col_x(1) + COL_WIDTHS[1] / 2, row_y + ROW_H / 2, row[1],
            fontsize=12, ha='center', va='center',
            color=AVG_COLOR, fontfamily='DejaVu Sans Mono', zorder=2)
    # Worst case (red, monospace)
    ax.text(col_x(2) + COL_WIDTHS[2] / 2, row_y + ROW_H / 2, row[2],
            fontsize=12, ha='center', va='center',
            color=WORST_COLOR, fontfamily='DejaVu Sans Mono', zorder=2)

# ── Outer border ──
ax.add_patch(Rectangle((X0, Y0), TOTAL_W, TABLE_H,
                       facecolor='none', edgecolor=BORDER, linewidth=1.2, zorder=3))
# Column separators (for visual clarity)
for i in range(1, len(COL_WIDTHS)):
    x = col_x(i)
    ax.plot([x, x], [Y0, header_y + HEADER_H], color=BORDER, linewidth=0.8, zorder=3)

# ── Caption below the table ──
caption_y = Y0 - 0.45
ax.text(TABLE_W / 2 + 1.0, caption_y,
        'n = entries, m = slots.  Average assumes a uniform hash function.',
        fontsize=10, ha='center', va='center',
        color='#6B7280', fontfamily='sans-serif', fontstyle='italic')

plt.tight_layout()
out_path = 'bigo_table.png'
plt.savefig(out_path, dpi=DPI, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none', pad_inches=0.2)
plt.close(fig)
print(f'Wrote {out_path}')
