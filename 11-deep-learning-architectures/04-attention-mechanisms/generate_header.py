"""Generate the header image for the Attention article.

Right panels are REAL: the trained model's attention weights over
positions (the receipt), and the measured flat accuracies against
Part 3's published cliff.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import attention as A

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- train the T=40 recall model for real --------------------------------
X, y = A.make_recall(A.N_SEQ, 40, np.random.default_rng(140))
Xt, yt = A.make_recall(A.N_SEQ, 40, np.random.default_rng(240))
net = A.fit(A.Attention(A.VOCAB, A.VOCAB, seed=0), X, y)
acc = A.accuracy(net, Xt, yt)
Wmap = net.attention_map(Xt)
print(f'T=40 acc {acc:.1%}, weight on pos 0 {Wmap[:, 0].mean():.3f}')
assert acc == 1.0 and abs(Wmap[:, 0].mean() - 0.992) < 5e-4

# Part 3's published cliff (rnn.py DEMO 2) and this part's measured sweep
RNN_D = [20, 30, 40]
RNN_A = [100.0, 100.0, 24.0]
ATT_D = [40, 80, 160]
ATT_A = [100.0, 100.0, 100.0]

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Attention Mechanisms: Distance Stops Existing',
        fontsize=19, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'A soft dictionary lookup puts every position one hop from every '
        "other — Part 3's cliff cannot be built, and the model shows receipts.",
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 5.95, 5.3, 1.7, BLUE)
ax.text(3.35, 7.35, 'a soft dictionary lookup', fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=BLUE)
ax.text(3.35, 6.72, 'q : what am I looking for?\n'
                    'K : what does each position offer?\n'
                    'V : what does each position say?\n'
                    'answer = softmax(q·K/√dk) · V',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 4.35, 5.3, 1.2, GREEN, lw=1.7)
ax.text(3.35, 5.3, 'one hop, any distance', fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 4.82, 'gradient reach at t=1 vs t=40: 2.2x\n'
                    "(the RNN's: ~300x, downhill)",
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 2.75, 5.3, 1.2, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.7, 'but attention sees a SET', fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 3.22, "remove positional encodings and 'first'\n"
                    'becomes inexpressible: recall 100% -> 34%',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.2, PURPLE, face=BADGE_BG, lw=1.6)
ax.text(3.35, 2.1, 'the bill', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=PURPLE)
ax.text(3.35, 1.62, 'every position its own query = O(T²):\n'
                    'double the sequence, quadruple the cost',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the receipt =====================
axm = fig.add_axes([0.47, 0.545, 0.50, 0.235])
axm.imshow(Wmap[:24], aspect='auto', cmap='Purples', vmin=0,
           vmax=Wmap[:24].max())
axm.set_xlabel('position in the sequence (0 = the answer)', fontsize=8.3,
               color=SUBTLE_TEXT)
axm.set_ylabel('test sequences', fontsize=8.3, color=SUBTLE_TEXT)
axm.set_yticks([])
axm.set_xticks([0, 10, 20, 30, 40])
axm.tick_params(colors=SUBTLE_TEXT, labelsize=7.5)
for sp in axm.spines.values():
    sp.set_color('#CBD5E1')
axm.set_title('the receipt: where 24 trained lookups landed '
              f'(99.2% of weight on position 0)',
              fontsize=9.4, fontweight='bold', color=TEXT_COLOR, pad=5)

# ===================== RIGHT BOTTOM: cliff vs flat =====================
axc = fig.add_axes([0.47, 0.115, 0.50, 0.30])
axc.plot(RNN_D, RNN_A, 'o--', color=ORANGE, linewidth=2.0, markersize=7,
         label='vanilla RNN (Part 3)')
axc.plot(ATT_D, ATT_A, 's-', color=BLUE, linewidth=2.4, markersize=7,
         label='one attention head')
axc.axhline(25, color='#CBD5E1', linewidth=1, linestyle=':')
axc.text(21, 28, 'chance', fontsize=7.4, color=SUBTLE_TEXT)
axc.annotate("Part 3's cliff", xy=(40, 24), xytext=(52, 38), fontsize=8.4,
             color=ORANGE, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.1))
axc.annotate('flat to 4x that distance', xy=(160, 100), xytext=(103, 82),
             fontsize=8.4, color=BLUE, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.1))
axc.set_xlabel('distance T between answer and question', fontsize=8.5,
               color=SUBTLE_TEXT)
axc.set_ylabel('test accuracy (%)', fontsize=8.5, color=SUBTLE_TEXT)
axc.set_xscale('log')
axc.set_xticks([20, 30, 40, 80, 160])
axc.set_xticklabels(['20', '30', '40', '80', '160'])
axc.set_ylim(15, 108)
axc.legend(fontsize=8, loc='center right', frameon=False)
axc.set_title('recall vs distance: the cliff, unbuilt (real runs)',
              fontsize=9.5, fontweight='bold', color=TEXT_COLOR, pad=5)
for sp in ('top', 'right'):
    axc.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axc.spines[sp].set_color('#CBD5E1')
axc.tick_params(colors=SUBTLE_TEXT, labelsize=8)
axc.minorticks_off()

ax.text(8, 0.18, 'Algorithms in Python  |  Deep Learning Architectures Part 4',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '11-deep-learning-architectures/04-attention-mechanisms/'
       'header_attention.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
