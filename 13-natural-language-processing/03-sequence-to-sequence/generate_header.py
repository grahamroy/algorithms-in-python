"""Generate the header image for the Seq2Seq article.

Right panels are REAL: attention heatmap and accuracy comparison
retrained live with frozen seeds.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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

# No-attention model
m_no = S2S.Seq2Seq(V, 32, 48, 32, False, np.random.default_rng(0))
S2S.train_model(m_no, short_train + long_train, w2i, 30,
                np.random.default_rng(1), lr=2e-3)
acc_no_s, _ = S2S.evaluate(m_no, short_test, w2i, vocab)
acc_no_l, _ = S2S.evaluate(m_no, long_test, w2i, vocab)

# Attention model
m_at = S2S.Seq2Seq(V, 32, 48, 32, True, np.random.default_rng(0))
S2S.train_model(m_at, short_train + long_train, w2i, 30,
                np.random.default_rng(1), lr=2e-3)
acc_at_s, ex_s = S2S.evaluate(m_at, short_test, w2i, vocab)
acc_at_l, _ = S2S.evaluate(m_at, long_test, w2i, vocab)

# Assert frozen accuracies
assert f'{acc_no_s:.3f}' == '0.045', f'no-attn short {acc_no_s}'
assert f'{acc_no_l:.3f}' == '0.000', f'no-attn long {acc_no_l}'
assert f'{acc_at_s:.3f}' == '1.000', f'attn short {acc_at_s}'
assert f'{acc_at_l:.3f}' == '0.995', f'attn long {acc_at_l}'

# Get attention weights for heatmap
src0, tgt0, pred0, alphas0 = ex_s[0]
src_ids = S2S.enc(src0, w2i)
src_words = ['<B>'] + src0 + ['<E>']
out_words = pred0

# ---- colour palette (matches series) ------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- figure -------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Sequence-to-Sequence: The Encoder Reads, '
        'the Decoder Writes',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Bahdanau attention breaks the bottleneck \u2014 '
        'from 4.5% to 100% on short sequences, '
        '0% to 99.5% on long.',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

# Box 1: The bottleneck
box(0.7, 6.0, 5.3, 1.6, RED)
ax.text(3.35, 7.32, 'The bottleneck problem', fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 6.88,
        "encoder compresses input into ONE vector\n"
        "decoder must reconstruct from that alone",
        fontsize=8.6, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.4,
        "short: 4.5% accuracy  |  long: 0.0%",
        fontsize=8.4, ha='center', va='center', color=RED,
        fontweight='bold')

# Box 2: Attention
box(0.7, 4.35, 5.3, 1.35, GREEN, face=BADGE_BG)
ax.text(3.35, 5.45, 'Bahdanau attention', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 4.95,
        "decoder peeks at every encoder state\n"
        "short: 100%  |  long: 99.5%",
        fontsize=8.4, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

# Box 3: The architecture
box(0.7, 2.95, 5.3, 1.1, BLUE, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.82, 'Encoder-decoder architecture',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=BLUE)
ax.text(3.35, 3.35,
        "Elman RNN, d_emb=32, d_hid=48\n"
        "teacher forcing + greedy decode",
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# Box 4: The lineage
box(0.7, 1.35, 5.3, 1.25, PURPLE, lw=1.6)
ax.text(3.35, 2.35, 'Two halves, two legacies', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 1.82,
        'BERT = the encoder (reads)\n'
        'GPT  = the decoder (writes)',
        fontsize=8.6, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

# ===================== RIGHT TOP: attention heatmap =======================
axh = fig.add_axes([0.44, 0.42, 0.52, 0.32])
A = np.array(alphas0[:len(out_words)])
im = axh.imshow(A, aspect='auto', cmap='Blues', vmin=0, vmax=1)
axh.set_xticks(range(len(src_words)))
axh.set_xticklabels(src_words, fontsize=7.2, rotation=45,
                    ha='right', fontfamily='DejaVu Sans Mono',
                    color=TEXT_COLOR)
axh.set_yticks(range(len(out_words)))
axh.set_yticklabels(out_words, fontsize=7.5,
                    fontfamily='DejaVu Sans Mono', color=TEXT_COLOR)
axh.xaxis.set_ticks_position('top')
axh.xaxis.set_label_position('top')
# Add weight annotations
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        v = A[i, j]
        if v > 0.03:
            c = 'white' if v > 0.5 else TEXT_COLOR
            axh.text(j, i, f'{v:.2f}', ha='center', va='center',
                     fontsize=6.2, color=c)
axh.set_title('attention receipt: where the decoder looks',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR,
              pad=8)
for sp in axh.spines.values():
    sp.set_color('#CBD5E1')

# ===================== RIGHT BOTTOM: accuracy comparison ==================
axb = fig.add_axes([0.44, 0.06, 0.52, 0.30])
axb.set_xlim(0, 10)
axb.set_ylim(0, 6)
axb.axis('off')

axb.text(5, 5.5, 'the bottleneck receipt', fontsize=9.5,
         fontweight='bold', ha='center', va='center', color=TEXT_COLOR)

# Header row
axb.text(1.5, 4.5, '', fontsize=9, ha='left', va='center',
         color=SUBTLE_TEXT)
axb.text(5.5, 4.5, 'short', fontsize=9, fontweight='bold',
         ha='center', va='center', color=SUBTLE_TEXT,
         fontfamily='DejaVu Sans Mono')
axb.text(8.0, 4.5, 'long', fontsize=9, fontweight='bold',
         ha='center', va='center', color=SUBTLE_TEXT,
         fontfamily='DejaVu Sans Mono')

# No attention row
axb.text(1.5, 3.3, 'no attention', fontsize=9.5, ha='left',
         va='center', color=RED, fontfamily='DejaVu Sans Mono')
axb.text(5.5, 3.3, f'{acc_no_s:.1%}', fontsize=10, ha='center',
         va='center', color=RED, fontfamily='DejaVu Sans Mono',
         fontweight='bold')
axb.text(8.0, 3.3, f'{acc_no_l:.1%}', fontsize=10, ha='center',
         va='center', color=RED, fontfamily='DejaVu Sans Mono',
         fontweight='bold')

# Attention row
axb.text(1.5, 2.1, 'attention', fontsize=9.5, ha='left',
         va='center', color=GREEN, fontfamily='DejaVu Sans Mono',
         fontweight='bold')
axb.text(5.5, 2.1, f'{acc_at_s:.1%}', fontsize=10, ha='center',
         va='center', color=GREEN, fontfamily='DejaVu Sans Mono',
         fontweight='bold')
axb.text(8.0, 2.1, f'{acc_at_l:.1%}', fontsize=10, ha='center',
         va='center', color=GREEN, fontfamily='DejaVu Sans Mono',
         fontweight='bold')

axb.text(5, 0.7,
         'SVO \u2192 SOV translation on Part 1\u2019s '
         'micro-language  |  107 tokens  |  30 epochs',
         fontsize=7.5, ha='center', va='center', color=SUBTLE_TEXT)

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 3',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/03-sequence-to-sequence/'
       'header_seq2seq.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
