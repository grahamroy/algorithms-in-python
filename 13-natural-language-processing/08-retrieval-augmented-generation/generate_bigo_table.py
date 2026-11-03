"""Generate a Big-O / cost summary table for the RAG article."""

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
TIME_COLOR = '#059669'
MEM_COLOR = '#3b82f6'
NOTE_COLOR = '#6B7280'

HEADERS = ['Operation', 'Cost', 'Memory', 'Notes']
ROWS = [
    ['Index the store',
     'O(N \u00b7 m \u00b7 L \u00b7 d)',
     'O(N \u00b7 d)',
     'once, offline; N vectors is the whole index'],
    ['Retrieve (bi-encoder)',
     '1 forward + O(N \u00b7 d)',
     'O(N \u00b7 d)',
     'one pass, then N dot products'],
    ['Retrieve (lexical)',
     'O(|q| \u00b7 postings)',
     'O(vocabulary)',
     'cheaper still, and it cannot break a tie'],
    ['Re-rank top k',
     'O(k \u00b7 L(n\u00b7d + d\u00b2))',
     'O(n\u00b2)',
     'a cross-encoder, k forwards, Part 6\'s row'],
    ['Generate, no document',
     'O(L \u00b7 q(q\u00b7d + d\u00b2))',
     'O(q \u00b7 d)',
     'q = question length'],
    ['Generate, with document',
     'O(L(p+q)((p+q)d + d\u00b2))',
     'O((p+q) \u00b7 d)',
     'p = passage; 500 before 20 is 676x attention'],
    ['Update one fact',
     'O(1) store edit',
     'O(1)',
     'the pitch: no retraining to change a fact'],
    ['Measure grounding',
     'O(T) generations',
     'O(T)',
     'but needs a KNOWN prior to contradict'],
]

COL_WIDTHS = [6.0, 6.4, 3.8, 9.2]
TOTAL_W = sum(COL_WIDTHS)
HEADER_H = 0.9
ROW_H = 0.8
N_ROWS = len(ROWS)
TABLE_H = HEADER_H + N_ROWS * ROW_H

fig, ax = plt.subplots(figsize=(1820 / 150, 900 / 150), dpi=150)
ax.set_xlim(0, TOTAL_W + 2.0)
ax.set_ylim(0, TABLE_H + 2.0)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

X0, Y0 = 1.0, 0.8


def col_x(i):
    return X0 + sum(COL_WIDTHS[:i])


ax.text(TOTAL_W / 2 + 1.0, TABLE_H + 1.3,
        'Retrieval-Augmented Generation Complexity',
        fontsize=16, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

header_y = Y0 + N_ROWS * ROW_H
ax.add_patch(Rectangle((X0, header_y), TOTAL_W, HEADER_H,
                       facecolor=HEADER_BG, edgecolor=HEADER_BG,
                       linewidth=0, zorder=1))
for i, h in enumerate(HEADERS):
    ax.text(col_x(i) + COL_WIDTHS[i] / 2, header_y + HEADER_H / 2, h,
            fontsize=13, fontweight='bold', ha='center', va='center',
            color=HEADER_TEXT, fontfamily='sans-serif', zorder=2)

for r, row in enumerate(ROWS):
    row_y = Y0 + (N_ROWS - 1 - r) * ROW_H
    bg = ROW_EVEN if r % 2 == 0 else ROW_ODD
    ax.add_patch(Rectangle((X0, row_y), TOTAL_W, ROW_H,
                           facecolor=bg, edgecolor=BORDER,
                           linewidth=0.8, zorder=1))
    ax.text(col_x(0) + 0.25, row_y + ROW_H / 2, row[0],
            fontsize=11.5, fontweight='bold', ha='left', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=2)
    ax.text(col_x(1) + COL_WIDTHS[1] / 2, row_y + ROW_H / 2, row[1],
            fontsize=11, ha='center', va='center',
            color=TIME_COLOR, fontfamily='DejaVu Sans Mono', zorder=2)
    ax.text(col_x(2) + COL_WIDTHS[2] / 2, row_y + ROW_H / 2, row[2],
            fontsize=11, ha='center', va='center',
            color=MEM_COLOR, fontfamily='DejaVu Sans Mono', zorder=2)
    ax.text(col_x(3) + 0.25, row_y + ROW_H / 2, row[3],
            fontsize=10.5, ha='left', va='center',
            color=NOTE_COLOR, fontfamily='sans-serif',
            fontstyle='italic', zorder=2)

ax.add_patch(Rectangle((X0, Y0), TOTAL_W, TABLE_H,
                       facecolor='none', edgecolor=BORDER,
                       linewidth=1.2, zorder=3))
for i in range(1, len(COL_WIDTHS)):
    ax.plot([col_x(i), col_x(i)], [Y0, header_y + HEADER_H],
            color=BORDER, linewidth=0.8, zorder=3)
for after in (4, 2):
    ry = Y0 + (N_ROWS - after) * ROW_H
    ax.plot([X0, X0 + TOTAL_W], [ry, ry], color='#94A3B8',
            linewidth=1.4, zorder=5)

ax.text(TOTAL_W / 2 + 1.0, Y0 - 0.45,
        'N = documents in the store, m = document length, '
        'p = passage, q = question, d = model width, L = layers, '
        'k = shortlist, T = test questions.',
        fontsize=10, ha='center', va='center',
        color='#6B7280', fontfamily='sans-serif', fontstyle='italic')

plt.tight_layout()
out = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/08-retrieval-augmented-generation/'
       'bigo_table.png')
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none', pad_inches=0.2)
plt.close(fig)
print(f'Wrote {out}')
