import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

# ── Colours ──
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

# Dense matrix colours
ZERO_FILL = '#E2E8F0'
ZERO_TEXT = '#9CA3AF'
NONZERO_FILL = '#3b82f6'
NONZERO_TEXT = '#FFFFFF'

# CSR array colours
INDPTR_FILL = '#dbeafe'
INDPTR_BORDER = '#3b82f6'
INDICES_FILL = '#dcfce7'
INDICES_BORDER = '#16a34a'
DATA_FILL = '#FEF3C7'
DATA_BORDER = '#F59E0B'

ARROW_COLOR = '#475569'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.45, 'Sparse Matrices: When Most of Your Data is Zero',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ── Subtitle ──
ax.text(8, 7.95,
        'Three formats --- COO, CSR, CSC --- and one underlying idea: store only what is non-zero',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# Two panels side by side: dense matrix view + CSR layout
# ═══════════════════════════════════════════════════════════
LEFT_PANEL = (0.4, 0.9, 7.0, 6.0)
RIGHT_PANEL = (8.6, 0.9, 7.0, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)

# ═══════════════════════════════════════════════════════════
# LEFT PANEL --- Dense matrix view of the example matrix
# ═══════════════════════════════════════════════════════════
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4, 'Logical view',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(lpx + lpw/2, lpy + lph - 0.85, '4 x 5 matrix, 6 non-zeros (30% density)',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# 4 x 5 grid showing the matrix
matrix_data = [
    [5, 0, 0, 7, 0],
    [0, 0, 9, 0, 0],
    [0, 2, 0, 0, 8],
    [1, 0, 0, 0, 0],
]
n_rows = 4
n_cols = 5
cell_w = 0.85
cell_h = 0.85
grid_total_w = n_cols * cell_w
grid_total_h = n_rows * cell_h
grid_x0 = lpx + (lpw - grid_total_w) / 2
grid_y0 = lpy + (lph - grid_total_h) / 2 - 0.2

# Column labels
for c in range(n_cols):
    x = grid_x0 + c * cell_w + cell_w / 2
    ax.text(x, grid_y0 + grid_total_h + 0.15, f'col {c}',
            fontsize=8, ha='center', va='bottom',
            color=SUBTLE_TEXT, fontfamily='monospace')

# Row labels
for r in range(n_rows):
    y = grid_y0 + (n_rows - 1 - r) * cell_h + cell_h / 2
    ax.text(grid_x0 - 0.15, y, f'row {r}',
            fontsize=8, ha='right', va='center',
            color=SUBTLE_TEXT, fontfamily='monospace')

for r in range(n_rows):
    for c in range(n_cols):
        v = matrix_data[r][c]
        is_nz = (v != 0)
        fill = NONZERO_FILL if is_nz else ZERO_FILL
        text_color = NONZERO_TEXT if is_nz else ZERO_TEXT
        x = grid_x0 + c * cell_w
        y = grid_y0 + (n_rows - 1 - r) * cell_h
        ax.add_patch(Rectangle((x, y), cell_w, cell_h,
                               facecolor=fill, edgecolor='white',
                               linewidth=1.5, zorder=2))
        ax.text(x + cell_w/2, y + cell_h/2, str(v),
                fontsize=12, fontweight='bold', ha='center', va='center',
                color=text_color, fontfamily='monospace', zorder=3)

# Caption
ax.text(lpx + lpw/2, lpy + 0.55,
        'Dense storage: m x n cells, even the zeros',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# RIGHT PANEL --- CSR layout: indptr, indices, data
# ═══════════════════════════════════════════════════════════
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4, 'CSR storage',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(rpx + rpw/2, rpy + rph - 0.85, 'Three arrays --- only the non-zeros',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Three rows of arrays: indptr (5 elts), indices (6 elts), data (6 elts)
arr_label_x = rpx + 0.7

def draw_array(label, values, colors, y, n_elements, full_width=4.6):
    """Draw a labelled array of cells centered horizontally inside the right panel."""
    cell_w = full_width / n_elements
    cell_h = 0.55
    fill, border = colors
    # Centre the array in the panel area to the right of the label
    arr_x0 = rpx + 1.85
    arr_y = y - cell_h / 2
    for i, v in enumerate(values):
        x = arr_x0 + i * cell_w
        ax.add_patch(Rectangle((x, arr_y), cell_w, cell_h,
                               facecolor=fill, edgecolor=border,
                               linewidth=1.3, zorder=2))
        ax.text(x + cell_w / 2, arr_y + cell_h / 2, str(v),
                fontsize=10, fontweight='bold', ha='center', va='center',
                color=TEXT_COLOR, fontfamily='monospace', zorder=3)
    # Label on the left
    ax.text(arr_x0 - 0.18, arr_y + cell_h / 2, label,
            fontsize=10.5, fontweight='bold', ha='right', va='center',
            color=border, fontfamily='monospace', zorder=3)
    return arr_x0, cell_w

# indptr layer (pushed higher to leave room for explanation below)
y_indptr = rpy + rph - 1.7
indptr_vals = [0, 2, 3, 5, 6]
draw_array('indptr ', indptr_vals,
           (INDPTR_FILL, INDPTR_BORDER),
           y_indptr, n_elements=len(indptr_vals))

# indices layer
y_indices = y_indptr - 0.95
indices_vals = [0, 3, 2, 1, 4, 0]
draw_array('indices', indices_vals,
           (INDICES_FILL, INDICES_BORDER),
           y_indices, n_elements=len(indices_vals))

# data layer
y_data = y_indices - 0.95
data_vals = [5, 7, 9, 2, 8, 1]
draw_array('data   ', data_vals,
           (DATA_FILL, DATA_BORDER),
           y_data, n_elements=len(data_vals))

# Annotate "row 2 lives in indices[3:5]" -- placed clearly between data and caption
explain_y = y_data - 0.85
ax.text(rpx + rpw / 2, explain_y,
        'row 2 \u2192 indices[indptr[2] : indptr[3]] = indices[3:5] = [1, 4]',
        fontsize=9.5, ha='center', va='center',
        color=TEXT_COLOR, fontfamily='monospace')
ax.text(rpx + rpw / 2, explain_y - 0.40,
        '\u2192 columns 1 and 4, with values 2 and 8',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Caption
ax.text(rpx + rpw/2, rpy + 0.55,
        'Sparse storage: O(nnz + m), fast row slicing, fast SpMV',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
ax.text(8, 0.3, 'Algorithms in Python  |  Foundations Part 11',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/00-foundations/11-sparse-matrices/header_sparse.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
