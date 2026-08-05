"""Generate the header image for the Transformers article.

The attention heatmaps are REAL: the reversal model is retrained live
(same seeds as transformer.py DEMO 2) and both heads' maps rendered.
LM loss marks are transformer.py's frozen DEMO 3 output.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import transformer as TF

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- retrain the DEMO 2 reversal model (real) ----------------------------
ids, tg = TF.make_rev(256, np.random.default_rng(0))
ids_t, tg_t = TF.make_rev(512, np.random.default_rng(1))
net = TF.Transformer(TF.VOC_REV, d=32, heads=2, layers=1, seed=0)
for _ in range(1000):
    net.step(ids, tg)
logits, _ = net.forward(ids_t)
pred = logits.argmax(-1)
em = float((pred[:, TF.K:] == tg_t[:, TF.K:]).all(axis=1).mean())
print(f'reversal exact-match {em:.1%}')
assert abs(em - 505 / 512) < 1e-12
_, (caches, _, _) = net.forward(ids_t[:64])
Wl = caches[0][2][4]
maps = [Wl[:, hh, TF.K:2 * TF.K, 0:TF.K].mean(0) for hh in range(2)]

# frozen from transformer.py DEMO 3
LM_STEPS = [0, 100, 500, 1500]
LM_LOSS = [3.87, 2.29, 0.21, 0.12]

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Transformers: The Section, Assembled',
        fontsize=19, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        "Attention, MLPs, residuals, and one causal triangle — a machine "
        'that learns algorithms you can read and languages you can sample.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the assembly =====================
stack = [
    ('next-token head', SUBTLE_TEXT, ''),
    ('+  residual', GREEN, ''),
    ('MLP block', ORANGE, 'Part 1 (most params)'),
    ('LayerNorm', SUBTLE_TEXT, ''),
    ('+  residual', GREEN, '0.195 vs 0.739'),
    ('causal multi-head self-attention', BLUE, 'Part 4'),
    ('LayerNorm', SUBTLE_TEXT, ''),
    ('embeddings + positions', PURPLE, 'Part 4'),
]
y = 1.05
heights = [0.62, 0.5, 0.62, 0.5, 0.5, 0.72, 0.5, 0.62]
ys = []
for (label, col, note), h in zip(stack[::-1], heights[::-1]):
    ax.add_patch(FancyBboxPatch((0.9, y), 4.5, h,
                 boxstyle='round,pad=0.03,rounding_size=0.08',
                 facecolor='white' if col != GREEN else BADGE_BG,
                 edgecolor=col, linewidth=1.7))
    ax.text(3.15, y + h / 2, label, fontsize=8.6, fontweight='bold',
            ha='center', va='center', color=col)
    if note:
        ax.text(5.55, y + h / 2, note, fontsize=7.6, ha='left',
                va='center', color=SUBTLE_TEXT, fontstyle='italic')
    ys.append(y + h)
    y += h + 0.135
ax.text(3.15, y + 0.02, 'x L layers', fontsize=8.4, ha='center',
        va='bottom', color=SUBTLE_TEXT, fontstyle='italic')

# ===================== RIGHT TOP: the receipt =====================
ax.text(11.35, 7.35, 'the receipt: reversal — one layer, two heads, '
        '98.6% exact (real weights)', fontsize=9.0,
        fontweight='bold', ha='center', va='center', color=TEXT_COLOR)
for i, (M, tag) in enumerate(zip(maps, ['head 0 — assists',
                                        'head 1 — THE MIRROR'])):
    axh = fig.add_axes([0.505 + i * 0.155, 0.485, 0.135, 0.24])
    axh.imshow(M, cmap='Purples', vmin=0, vmax=M.max())
    axh.set_xticks([0, 7]); axh.set_yticks([0, 7])
    axh.set_xticklabels(['in 0', 'in 7'], fontsize=7)
    axh.set_yticklabels(['out 0', 'out 7'] if i == 0 else ['', ''],
                        fontsize=7)
    axh.tick_params(colors=SUBTLE_TEXT, length=2)
    for sp in axh.spines.values():
        sp.set_color('#CBD5E1')
    axh.set_title(tag, fontsize=8.2, color=TEXT_COLOR, pad=4,
                  fontweight='bold' if i == 1 else 'normal')
ax.text(14.55, 5.85, 'output i attends\nto input 8−1−i:\nan anti-diagonal\n'
        'nobody programmed', fontsize=8.4, ha='center', va='center',
        color=PURPLE, fontstyle='italic')

# ===================== RIGHT BOTTOM: the writer =====================
axl = fig.add_axes([0.475, 0.115, 0.23, 0.27])
axl.plot(LM_STEPS, LM_LOSS, 'o-', color=BLUE, linewidth=2.2, markersize=6)
axl.axhline(3.37, color='#CBD5E1', linewidth=1, linestyle=':')
axl.text(600, 3.5, 'uniform (3.37)', fontsize=7.2, color=SUBTLE_TEXT)
for s, l in zip(LM_STEPS, LM_LOSS):
    axl.annotate(f'{l:.2f}', xy=(s, l), xytext=(0, 7),
                 textcoords='offset points', fontsize=7.4,
                 ha='center', color=BLUE)
axl.set_xlabel('training step', fontsize=8.3, color=SUBTLE_TEXT)
axl.set_ylabel('loss (nats/char)', fontsize=8.3, color=SUBTLE_TEXT)
axl.set_ylim(-0.1, 4.3)
axl.set_title('the language model learns', fontsize=9.2,
              fontweight='bold', color=TEXT_COLOR, pad=5)
for sp in ('top', 'right'):
    axl.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axl.spines[sp].set_color('#CBD5E1')
axl.tick_params(colors=SUBTLE_TEXT, labelsize=7.5)

ax.add_patch(FancyBboxPatch((11.45, 1.0), 4.35, 2.45,
             boxstyle='round,pad=0.06,rounding_size=0.12',
             facecolor=BADGE_BG, edgecolor=BLUE, linewidth=1.6))
ax.text(13.62, 3.18, 'and then it writes', fontsize=9.2,
        fontweight='bold', ha='center', va='center', color=BLUE)
ax.text(13.62, 2.15,
        'prompt: "the gradient "\n\n'
        '"...flows through the addition\n'
        'untouched. the attention map is\n'
        'a receipt you can read: the\n'
        'modell shows where it looked."',
        fontsize=8.0, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(13.62, 1.2, '66k parameters, trained on this section itself',
        fontsize=7.4, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

ax.text(8, 0.18, 'Algorithms in Python  |  Deep Learning Architectures Part 5',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '11-deep-learning-architectures/05-transformers/'
       'header_transformer.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
