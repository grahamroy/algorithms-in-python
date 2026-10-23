"""Generate the header image for the BPE tokenisation article.

Right panels are REAL: the first 10 BPE merge steps shown as a
character-to-token progression, and the compression curve plotted
from actual merge counts -- retrained live with the frozen seeds.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from collections import Counter

import bpe_tokenisation as BT

# ---- retrain with frozen seeds -------------------------------------------
rng = np.random.default_rng(0)
sents = BT.gen_corpus(30000, rng)
all_words_flat = [w for s in sents for w in s]
word_freqs = Counter(all_words_flat)

merges_200, splits_200 = BT.learn_bpe(word_freqs, 200)

# Assert frozen numbers
tpw = BT.tokens_per_word(lambda w: BT.apply_bpe(w, merges_200),
                         list(word_freqs.keys()))
assert f'{tpw:.2f}' == '1.61', f'tok/word {tpw:.2f} != 1.61'

# compression curve data
merge_counts = [0, 25, 50, 100, 150, 200, 300]
curve_vocab = []
curve_tpw = []
unique_words = list(word_freqs.keys())
for nm in merge_counts:
    mi, si = BT.learn_bpe(word_freqs, nm)
    vi = set(tok for syms in si for tok in syms)
    base = set()
    for w in word_freqs:
        for ch in w + "_":
            base.add(ch)
    vi |= base
    curve_vocab.append(len(vi))
    curve_tpw.append(BT.tokens_per_word(
        lambda w, m=mi: BT.apply_bpe(w, m), unique_words))

# ---- colour palette (matches series) -------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Tokenisation: Who Decides the Pieces?',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'BPE / WordPiece / Unigram \u2014 three algorithms that let '
        'the corpus choose its own vocabulary.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.0, 5.3, 1.6, BLUE)
ax.text(3.35, 7.32, 'BPE: greedy merging', fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=BLUE)
ax.text(3.35, 6.88,
        "merge the most frequent pair, repeat\n"
        "200 merges -> 141 tokens, 1.61 tok/word",
        fontsize=8.6, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.4,
        "the humble compression algorithm that\n"
        "decides what every LLM can see",
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.35, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.45, 'WordPiece: MI scoring', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.95,
        "same loop, scores by mutual information\n"
        "only 35/200 merges match BPE",
        fontsize=8.4, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.95, 5.3, 1.1, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.82, 'Unigram LM: prune by likelihood',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=RED)
ax.text(3.35, 3.35,
        "start big, drop least-painful tokens\n"
        "Viterbi finds the optimal split",
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.35, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.35, 'OOV: no word is unsayable', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 1.82,
        'princesses -> prince | s | s | es_\n'
        'roosters   -> roost  | e | rs_',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

# ===================== RIGHT TOP: merge progression =======================
axm = fig.add_axes([0.44, 0.50, 0.34, 0.30])
# Show how "the" gets built across merges
merge_labels = []
for i, (a, b) in enumerate(merges_200[:10]):
    merge_labels.append(f'{i+1}: {a}+{b}\u2192{a+b}')

y_pos = np.arange(10)
colors = [BLUE if i < 3 else ORANGE if i < 7 else PURPLE
          for i in range(10)]
axm.barh(y_pos, [1]*10, color=colors, alpha=0.25, height=0.7)
for i, label in enumerate(merge_labels):
    axm.text(0.05, i, label, fontsize=7.6, va='center',
             color=TEXT_COLOR, fontfamily='DejaVu Sans Mono')
axm.set_yticks([])
axm.set_xticks([])
axm.invert_yaxis()
axm.set_xlim(0, 1)
for sp in axm.spines.values():
    sp.set_color('#CBD5E1')
axm.set_title('first 10 BPE merges: frequency speaks',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=5)

# ===================== RIGHT BOTTOM: compression curve ====================
axc = fig.add_axes([0.44, 0.10, 0.23, 0.30])
axc.plot(curve_vocab, curve_tpw, 'o-', color=BLUE, linewidth=1.8,
         markersize=4.5, zorder=3)
for i, nm in enumerate(merge_counts):
    if nm in (0, 100, 200, 300):
        axc.annotate(f'{nm}', (curve_vocab[i], curve_tpw[i]),
                     fontsize=6.8, color=SUBTLE_TEXT,
                     xytext=(3, 5), textcoords='offset points')
axc.set_xlabel('vocabulary size', fontsize=7.8, color=SUBTLE_TEXT)
axc.set_ylabel('tokens per word', fontsize=7.8, color=SUBTLE_TEXT)
for sp in ('top', 'right'):
    axc.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axc.spines[sp].set_color('#CBD5E1')
axc.tick_params(colors=SUBTLE_TEXT, labelsize=7)
axc.set_title('the compression curve', fontsize=9.0,
              fontweight='bold', color=TEXT_COLOR, pad=5)

# ===================== RIGHT BOTTOM RIGHT: who uses what ==================
axt = fig.add_axes([0.72, 0.10, 0.24, 0.30])
axt.axis('off')
axt.text(0.02, 0.95,
         'who uses what:\n\n'
         '  GPT-2/3/4    BPE\n'
         '  BERT         WordPiece\n'
         '  T5 / Llama   Unigram\n\n'
         'the compression knob:\n'
         '  more merges =\n'
         '    fewer tokens/word\n'
         '    bigger embedding\n'
         '    cheaper attention',
         fontsize=7.8, va='top', ha='left', color=TEXT_COLOR,
         fontfamily='DejaVu Sans Mono',
         transform=axt.transAxes)
for sp in axt.spines.values():
    sp.set_color('#CBD5E1')
axt.set_title('the landscape', fontsize=9.0,
              fontweight='bold', color=TEXT_COLOR, pad=5)

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 2',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/02-bpe-tokenisation/'
       'header_bpe_tokenisation.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
