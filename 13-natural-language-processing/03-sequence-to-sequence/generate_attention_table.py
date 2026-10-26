"""Generate an attention-receipt image for the Seq2Seq article.

Shows the attention heatmap as a clean table with coloured cells,
matching the series' visual style.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import seq2seq as S2S

# ---- retrain with frozen seeds ------------------------------------------
rng = np.random.default_rng(42)
short_train, long_train = S2S.make_pairs(1500, 750, rng)
short_test, long_test = S2S.make_pairs(
    400, 200, np.random.default_rng(7))
all_pairs = short_train + long_train + short_test + long_test
vocab, w2i = S2S.build_vocab(all_pairs)
V = len(vocab)

m_at = S2S.Seq2Seq(V, 32, 48, 32, True, np.random.default_rng(0))
S2S.train_model(m_at, short_train + long_train, w2i, 30,
                np.random.default_rng(1), lr=2e-3)
acc_s, ex_s = S2S.evaluate(m_at, short_test, w2i, vocab)
assert f'{acc_s:.3f}' == '1.000', f'attn short {acc_s}'

src0, tgt0, pred0, alphas0 = ex_s[0]
src_words = ['<BOS>'] + src0 + ['<EOS>']
out_words = pred0
A = np.array(alphas0[:len(out_words)])

# ---- colour palette -----------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
HEADER_BG = '#1F2937'
HEADER_TEXT = '#FFFFFF'
BORDER = '#E2E8F0'
SUBTLE = '#6B7280'

# ---- layout -------------------------------------------------------------
n_out = len(out_words)
n_src = len(src_words)

COL_W_LABEL = 2.4       # left column for output words
COL_W_CELL = 1.5        # each source-word column
ROW_H = 0.75
HEADER_H = 1.2          # taller header for rotated text

TOTAL_W = COL_W_LABEL + n_src * COL_W_CELL
TABLE_H = HEADER_H + n_out * ROW_H

fig, ax = plt.subplots(
    figsize=((TOTAL_W + 3.0) * 150/150, (TABLE_H + 3.0) * 150/150),
    dpi=150)
ax.set_xlim(0, TOTAL_W + 3.0)
ax.set_ylim(0, TABLE_H + 3.0)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

X0, Y0 = 1.2, 0.8

# Title
ax.text((TOTAL_W + 3.0) / 2, TABLE_H + 2.2,
        'Attention Receipt: Where the Decoder Looks',
        fontsize=15, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Header row
header_y = Y0 + n_out * ROW_H
ax.add_patch(Rectangle((X0, header_y), TOTAL_W, HEADER_H,
                       facecolor=HEADER_BG, edgecolor=HEADER_BG,
                       linewidth=0, zorder=1))

# "output \" label in header
ax.text(X0 + COL_W_LABEL / 2, header_y + HEADER_H / 2,
        'output \u2193', fontsize=10, fontweight='bold',
        ha='center', va='center', color=HEADER_TEXT,
        fontfamily='sans-serif', zorder=2)

# Source-word column headers (rotated)
for j, sw in enumerate(src_words):
    cx = X0 + COL_W_LABEL + j * COL_W_CELL + COL_W_CELL / 2
    ax.text(cx, header_y + HEADER_H / 2, sw,
            fontsize=10, fontweight='bold', ha='center', va='center',
            color=HEADER_TEXT, fontfamily='DejaVu Sans Mono',
            rotation=45, rotation_mode='anchor', zorder=2)

# Data rows
cmap = plt.cm.Blues
for r in range(n_out):
    row_y = Y0 + (n_out - 1 - r) * ROW_H
    # Row background
    bg = '#F8FAFC' if r % 2 == 0 else '#FFFFFF'
    ax.add_patch(Rectangle((X0, row_y), TOTAL_W, ROW_H,
                           facecolor=bg, edgecolor=BORDER,
                           linewidth=0.6, zorder=1))
    # Output word label
    ax.text(X0 + COL_W_LABEL - 0.2, row_y + ROW_H / 2,
            out_words[r], fontsize=10.5, fontweight='bold',
            ha='right', va='center', color=TEXT_COLOR,
            fontfamily='DejaVu Sans Mono', zorder=2)

    # Attention cells
    for j in range(n_src):
        v = A[r, j]
        cx = X0 + COL_W_LABEL + j * COL_W_CELL
        # Cell background colour by intensity
        cell_color = cmap(v * 0.85)  # cap so max isn't pure black
        ax.add_patch(Rectangle((cx + 0.05, row_y + 0.05),
                               COL_W_CELL - 0.1, ROW_H - 0.1,
                               facecolor=cell_color, edgecolor='none',
                               linewidth=0, zorder=2,
                               alpha=0.9))
        # Value text
        if v > 0.02:
            tc = '#FFFFFF' if v > 0.45 else TEXT_COLOR
            ax.text(cx + COL_W_CELL / 2, row_y + ROW_H / 2,
                    f'{v:.2f}', fontsize=9, ha='center', va='center',
                    color=tc, fontfamily='DejaVu Sans Mono', zorder=3)

# Outer border
ax.add_patch(Rectangle((X0, Y0), TOTAL_W, TABLE_H,
                       facecolor='none', edgecolor=BORDER,
                       linewidth=1.2, zorder=4))

# Column separators
for j in range(1, n_src + 1):
    lx = X0 + COL_W_LABEL + (j - 1) * COL_W_CELL
    ax.plot([lx, lx], [Y0, header_y + HEADER_H],
            color=BORDER, linewidth=0.5, zorder=4)
# Label/data separator
ax.plot([X0 + COL_W_LABEL, X0 + COL_W_LABEL],
        [Y0, header_y + HEADER_H],
        color=BORDER, linewidth=0.8, zorder=4)

# Caption
ax.text((TOTAL_W + 3.0) / 2, Y0 - 0.35,
        'Each row is one decoding step; each column is one '
        'encoder position. The verb (navigate) attends 1.00 '
        'to its source position.',
        fontsize=9, ha='center', va='center',
        color=SUBTLE, fontfamily='sans-serif', fontstyle='italic')

plt.tight_layout()
out = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/03-sequence-to-sequence/'
       'attention_table.png')
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none', pad_inches=0.2)
plt.close(fig)
print(f'Wrote {out}')
