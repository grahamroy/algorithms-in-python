"""Generate the header image for the Matrix Factorisation article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
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
RATING_FILL = '#fef3c7'
RATING_BORDER = '#F59E0B'
NEUTRAL = '#94A3B8'


fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45,
        'Matrix Factorisation: R ≈ U · Vᵀ',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'A sparse rating matrix decomposes into dense user and item factor matrices. Dot products predict the missing entries.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# Generate a small ratings matrix illustration (4 users x 6 items)
n_users, n_items, k = 4, 6, 3
rng = np.random.default_rng(7)
U_true = rng.normal(0, 1, (n_users, k))
V_true = rng.normal(0, 1, (n_items, k))
R_full = U_true @ V_true.T
# Mask half
mask = rng.uniform(size=(n_users, n_items)) < 0.5

# Cell sizes
cell = 0.55

# Layout positions (left to right):
# R (sparse) = U (dense) x V^T (dense)
# R: n_users x n_items
# U: n_users x k
# V^T: k x n_items

# Centre everything vertically
y_top = PANEL[1] + PANEL[3] - 1.6

# R matrix position
R_w = n_items * cell
R_h = n_users * cell
R_x = PANEL[0] + 0.8
R_y = y_top - R_h

# Draw R cells
def draw_matrix(x0, y0, n_rows, n_cols, values, mask=None,
                fill=RATING_FILL, border=RATING_BORDER,
                show_values=True):
    for r in range(n_rows):
        for c in range(n_cols):
            cx = x0 + c * cell
            cy = y0 + (n_rows - 1 - r) * cell
            if mask is not None and not mask[r, c]:
                # Empty cell
                ax.add_patch(Rectangle((cx, cy), cell, cell,
                                        facecolor='white',
                                        edgecolor=NEUTRAL,
                                        linewidth=0.6, zorder=2))
                ax.text(cx + cell/2, cy + cell/2, '?',
                        fontsize=9, ha='center', va='center',
                        color=NEUTRAL, fontfamily='monospace')
            else:
                ax.add_patch(Rectangle((cx, cy), cell, cell,
                                        facecolor=fill,
                                        edgecolor=border,
                                        linewidth=1.0, zorder=2))
                if show_values:
                    val = values[r, c]
                    ax.text(cx + cell/2, cy + cell/2,
                            f'{val:+.1f}',
                            fontsize=8, ha='center', va='center',
                            color=TEXT_COLOR,
                            fontfamily='monospace')

# Draw R (sparse — half cells are '?')
draw_matrix(R_x, R_y, n_users, n_items, R_full, mask=mask)
ax.text(R_x + R_w/2, R_y + R_h + 0.3, 'R  (sparse ratings)',
        fontsize=11, fontweight='bold', ha='center', va='center',
        color=RATING_BORDER, fontfamily='sans-serif')
ax.text(R_x - 0.1, R_y - 0.25, f'{n_users} users  ×  {n_items} items',
        fontsize=9, ha='left', va='center', fontstyle='italic',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Equals sign
eq_x = R_x + R_w + 0.6
ax.text(eq_x, R_y + R_h/2, '≈',
        fontsize=28, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR)

# U matrix (n_users x k)
U_w = k * cell
U_h = n_users * cell
U_x = eq_x + 0.6
U_y = R_y
draw_matrix(U_x, U_y, n_users, k, U_true,
            fill=USER_FILL, border=USER_BORDER)
ax.text(U_x + U_w/2, U_y + U_h + 0.3, 'U  (user factors)',
        fontsize=11, fontweight='bold', ha='center', va='center',
        color=USER_BORDER, fontfamily='sans-serif')
ax.text(U_x + U_w/2, U_y - 0.25, f'{n_users} × k',
        fontsize=9, ha='center', va='center', fontstyle='italic',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Times sign
times_x = U_x + U_w + 0.4
ax.text(times_x, R_y + R_h/2, '·',
        fontsize=36, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR)

# V^T matrix (k x n_items)
VT_w = n_items * cell
VT_h = k * cell
VT_x = times_x + 0.5
VT_y = R_y + (R_h - VT_h) / 2

draw_matrix(VT_x, VT_y, k, n_items, V_true.T,
            fill=ITEM_FILL, border=ITEM_BORDER)
ax.text(VT_x + VT_w/2, VT_y + VT_h + 0.3, 'Vᵀ  (item factors)',
        fontsize=11, fontweight='bold', ha='center', va='center',
        color=ITEM_BORDER, fontfamily='sans-serif')
ax.text(VT_x + VT_w/2, VT_y - 0.25, f'k × {n_items}',
        fontsize=9, ha='center', va='center', fontstyle='italic',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Caption underneath
ax.text(PANEL[0] + PANEL[2]/2, PANEL[1] + 0.5,
        'Predicted rating r̂ᵢⱼ = uᵢ · vⱼ.  Train U and V to fit observed ratings; use them to fill in the ?s.',
        fontsize=11, ha='center', va='center', fontstyle='italic',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

ax.text(8, 0.3,
        'Algorithms in Python  |  Recommender Systems Part 1',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '07-recommender-systems/'
       '01-matrix-factorisation/header_mf.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
