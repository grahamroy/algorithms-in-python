"""Generate a Big-O / memory summary table image for the sparse matrices article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Colours ──
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
HEADER_BG = '#1F2937'
HEADER_TEXT = '#FFFFFF'
ROW_EVEN = '#F8FAFC'
ROW_ODD = '#FFFFFF'
BORDER = '#E2E8F0'
GOOD_COLOR = '#059669'    # green --- good (fast)
BAD_COLOR = '#DC2626'     # red --- bad (slow)
MEM_COLOR = '#3b82f6'     # blue --- memory
NOTE_COLOR = '#6B7280'

# ── Table data ──
HEADERS = ['Format', 'Construction', 'Random write', 'SpMV / row slice', 'Memory', 'Best for']
ROWS = [
    ['COO', 'O(1) per nnz',     'O(1) append',  'O(nnz) full scan',  'O(nnz)',     'building from triples'],
    ['CSR', 'O(nnz) from COO',  'O(nnz_in_row)', 'O(nnz)',           'O(nnz + m)', 'compute / SpMV / row slice'],
    ['CSC', 'O(nnz) from COO',  'O(nnz_in_col)', 'O(nnz) col-walk',  'O(nnz + n)', 'column slice / solvers'],
    ['LIL', 'O(nnz) per row',   'O(log nnz)',    'slow',             'O(nnz + m)', 'incremental random-access fill'],
    ['DOK', 'O(1) per cell',    'O(1)',          'slow',             'O(nnz)',     'streaming / unknown shape'],
    ['BSR', 'O(nnz) from COO',  'O(block) write','O(nnz) blocked',   'O(nnz / b)', 'block-structured matrices'],
]

# ── Layout parameters ──
COL_WIDTHS = [1.7, 3.0, 2.7, 3.4, 2.4, 4.4]
TOTAL_W = sum(COL_WIDTHS)
HEADER_H = 0.9
ROW_H = 0.8
N_ROWS = len(ROWS)
TABLE_W = TOTAL_W
TABLE_H = HEADER_H + N_ROWS * ROW_H

FIG_W_PX, FIG_H_PX = 1700, 800
DPI = 150
fig, ax = plt.subplots(figsize=(FIG_W_PX/DPI, FIG_H_PX/DPI), dpi=DPI)
ax.set_xlim(0, TABLE_W + 2.0)
ax.set_ylim(0, TABLE_H + 2.0)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
title_y = TABLE_H + 1.3
ax.text(TABLE_W / 2 + 1.0, title_y,
        'Sparse Matrix Formats --- Trade-offs at a Glance',
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

# Slow words to colour red, fast to colour green
def colour_for(text):
    txt = text.lower()
    if 'slow' in txt:
        return BAD_COLOR
    if 'full scan' in txt or 'col-walk' in txt:
        return BAD_COLOR
    return GOOD_COLOR

# ── Data rows ──
for r, row in enumerate(ROWS):
    row_y = Y0 + (N_ROWS - 1 - r) * ROW_H
    bg = ROW_EVEN if r % 2 == 0 else ROW_ODD
    ax.add_patch(Rectangle((X0, row_y), TOTAL_W, ROW_H,
                           facecolor=bg, edgecolor=BORDER, linewidth=0.8, zorder=1))
    # Format (left-aligned, bold)
    ax.text(col_x(0) + 0.25, row_y + ROW_H / 2, row[0],
            fontsize=12, fontweight='bold', ha='left', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=2)
    # Construction (mono)
    ax.text(col_x(1) + COL_WIDTHS[1] / 2, row_y + ROW_H / 2, row[1],
            fontsize=10.5, ha='center', va='center',
            color=GOOD_COLOR if 'O(1)' in row[1] else NOTE_COLOR,
            fontfamily='DejaVu Sans Mono', zorder=2)
    # Random write
    ax.text(col_x(2) + COL_WIDTHS[2] / 2, row_y + ROW_H / 2, row[2],
            fontsize=10.5, ha='center', va='center',
            color=GOOD_COLOR if 'O(1)' in row[2] else NOTE_COLOR,
            fontfamily='DejaVu Sans Mono', zorder=2)
    # SpMV / row slice
    ax.text(col_x(3) + COL_WIDTHS[3] / 2, row_y + ROW_H / 2, row[3],
            fontsize=10.5, ha='center', va='center',
            color=colour_for(row[3]),
            fontfamily='DejaVu Sans Mono', zorder=2)
    # Memory (blue, mono)
    ax.text(col_x(4) + COL_WIDTHS[4] / 2, row_y + ROW_H / 2, row[4],
            fontsize=10.5, ha='center', va='center',
            color=MEM_COLOR, fontfamily='DejaVu Sans Mono', zorder=2)
    # Notes (italic, subtle)
    ax.text(col_x(5) + 0.25, row_y + ROW_H / 2, row[5],
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
        'nnz = number of non-zero entries; m = rows, n = cols, b = block size.  '
        'Build in COO/LIL/DOK, compute in CSR/CSC.',
        fontsize=10, ha='center', va='center',
        color='#6B7280', fontfamily='sans-serif', fontstyle='italic')

plt.tight_layout()
out_path = 'D:/Projects/Medium/algorithms-in-python/00-foundations/11-sparse-matrices/bigo_table.png'
plt.savefig(out_path, dpi=DPI, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none', pad_inches=0.2)
plt.close(fig)
print(f'Wrote {out_path}')
