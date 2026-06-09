"""Generate a results comparison table image for the MF article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
HEADER_BG = '#1F2937'
HEADER_TEXT = '#FFFFFF'
ROW_EVEN = '#F8FAFC'
ROW_ODD = '#FFFFFF'
BORDER = '#E2E8F0'
NOTE_COLOR = '#6B7280'

HEADERS = ['Method', 'k', 'iters', 'train RMSE', 'test RMSE']
ROWS = [
    ['Baseline (global mean)', '—',  '—',  '2.245', '2.246'],
    ['ALS',                    '5',  '20', '0.281', '0.369'],
    ['SGD (FunkSVD)',          '5',  '200','0.347', '0.448'],
]

COL_WIDTHS = [5.4, 1.4, 1.8, 2.8, 2.8]
TOTAL_W = sum(COL_WIDTHS)
HEADER_H = 0.9
ROW_H = 0.8
N_ROWS = len(ROWS)
TABLE_H = HEADER_H + N_ROWS * ROW_H

fig, ax = plt.subplots(figsize=(1600/150, 500/150), dpi=150)
ax.set_xlim(0, TOTAL_W + 2.0)
ax.set_ylim(0, TABLE_H + 2.0)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

X0, Y0 = 1.0, 0.8
def col_x(i): return X0 + sum(COL_WIDTHS[:i])

ax.text(TOTAL_W / 2 + 1.0, TABLE_H + 1.3,
        'Matrix factorisation vs baseline on synthetic ratings',
        fontsize=15, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

header_y = Y0 + N_ROWS * ROW_H
ax.add_patch(Rectangle((X0, header_y), TOTAL_W, HEADER_H,
                       facecolor=HEADER_BG, edgecolor=HEADER_BG,
                       linewidth=0, zorder=1))
for i, h in enumerate(HEADERS):
    ax.text(col_x(i) + COL_WIDTHS[i] / 2, header_y + HEADER_H / 2, h,
            fontsize=12, fontweight='bold', ha='center', va='center',
            color=HEADER_TEXT, fontfamily='sans-serif', zorder=2)

for r, row in enumerate(ROWS):
    row_y = Y0 + (N_ROWS - 1 - r) * ROW_H
    bg = ROW_EVEN if r % 2 == 0 else ROW_ODD
    ax.add_patch(Rectangle((X0, row_y), TOTAL_W, ROW_H,
                           facecolor=bg, edgecolor=BORDER,
                           linewidth=0.8, zorder=1))
    # Method (left-aligned, bold)
    ax.text(col_x(0) + 0.25, row_y + ROW_H / 2, row[0],
            fontsize=11.5, fontweight='bold', ha='left', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=2)
    # k, iters (centered, mono)
    for i in (1, 2):
        ax.text(col_x(i) + COL_WIDTHS[i] / 2, row_y + ROW_H / 2,
                row[i], fontsize=11, ha='center', va='center',
                color=TEXT_COLOR, fontfamily='DejaVu Sans Mono',
                zorder=2)
    # train RMSE (mono, slightly muted)
    ax.text(col_x(3) + COL_WIDTHS[3] / 2, row_y + ROW_H / 2,
            row[3], fontsize=11, ha='center', va='center',
            color=NOTE_COLOR, fontfamily='DejaVu Sans Mono',
            zorder=2)
    # test RMSE (mono, highlighted green for ALS/SGD; same for baseline)
    color = '#059669' if r > 0 else TEXT_COLOR
    fw = 'bold' if r > 0 else 'normal'
    ax.text(col_x(4) + COL_WIDTHS[4] / 2, row_y + ROW_H / 2,
            row[4], fontsize=11, ha='center', va='center',
            color=color, fontweight=fw,
            fontfamily='DejaVu Sans Mono', zorder=2)

# Outer border + column dividers
ax.add_patch(Rectangle((X0, Y0), TOTAL_W, TABLE_H,
                       facecolor='none', edgecolor=BORDER,
                       linewidth=1.2, zorder=3))
for i in range(1, len(COL_WIDTHS)):
    ax.plot([col_x(i), col_x(i)],
            [Y0, header_y + HEADER_H],
            color=BORDER, linewidth=0.8, zorder=3)

# Caption
ax.text(TOTAL_W / 2 + 1.0, Y0 - 0.45,
        'Synthetic data: 200 users × 100 items, true latent rank 5, '
        '~40% observed density, 80/20 train/test split.',
        fontsize=9.5, ha='center', va='center',
        color=NOTE_COLOR, fontfamily='sans-serif', fontstyle='italic')

plt.tight_layout()
out = ('D:/Projects/Medium/algorithms-in-python/'
       '07-recommender-systems/'
       '01-matrix-factorisation/results_table.png')
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none', pad_inches=0.2)
plt.close(fig)
print(f'Wrote {out}')
