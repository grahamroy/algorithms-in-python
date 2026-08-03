"""Generate the header image for the CNN article.

Right panels are REAL: sample images from the stroke generator, and the
two strongest learned kernels from the trained conv net in cnn.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

import cnn as C

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- real data + trained net ---------------------------------------------
rng = np.random.default_rng(C.RNG_SEED)
X, y = C.make_lines(200, rng)
X_test, y_test = C.make_lines(200, rng)
conv = C.fit(C.CNN(seed=0), X, y)
acc = C.accuracy(conv, X_test, y_test)
print(f'conv test acc {acc:.1%}')
assert abs(acc - 599 / 600) < 1e-12

samples = [C.draw_line(cls, np.random.default_rng(5 + cls)) for cls in range(3)]
norms = np.linalg.norm(conv.K, axis=0)
top2 = np.argsort(norms)[::-1][:2]
kernels = [conv.K[:, f].reshape(3, 3) for f in top2]

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Convolutional Neural Networks: Learned Here, Known Everywhere',
        fontsize=17.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Nine shared weights slide across 144 positions — locality and '
        'weight sharing, the bet that beats 12,803 parameters with 947.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the two promises =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.3, 5.3, 1.25, BLUE)
ax.text(3.35, 7.3, 'PROMISE 1: locality', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=BLUE)
ax.text(3.35, 6.82, 'each output looks at a 3x3 window —\n'
                    'edges and strokes are local events',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 4.6, 5.3, 1.4, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.72, 'PROMISE 2: weight sharing', fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 5.15, 'ONE kernel slides across the image:\n'
                    '9 weights serve all 144 positions —\n'
                    'what is learned here is known everywhere',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 3.15, 5.3, 1.05, GREEN, lw=1.6)
ax.text(3.35, 3.95, 'then max-pool 2x2', fontsize=9.4, fontweight='bold',
        ha='center', va='center', color=GREEN)
ax.text(3.35, 3.52, '"did the feature occur anywhere nearby?" —\n'
                    'keep the what, discard the exactly-where',
        fontsize=8.0, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.35, 5.3, 1.4, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 2.48, 'the shuffle test (measured)', fontsize=9.4,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 1.92, 'permute all pixels identically:\n'
                    'MLP 92.2% -> 91.0% (never saw geometry)\n'
                    'CNN 99.8% -> 62.0% (its bet made false)',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the stage =====================
titles = ['horizontal', 'vertical', 'diagonal']
ax.text(11.15, 7.42, 'the stage: one 7-pixel stroke, random position, '
        'equal brightness', fontsize=9.3, fontweight='bold',
        ha='center', va='center', color=TEXT_COLOR)
for i, (img, t) in enumerate(zip(samples, titles)):
    axp = fig.add_axes([0.475 + i * 0.115, 0.56, 0.10, 0.178])
    axp.imshow(img, cmap='gray_r', vmin=-0.3, vmax=1.2)
    axp.set_xticks([]); axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_color('#CBD5E1')
    axp.set_title(t, fontsize=8.2, color=SUBTLE_TEXT, pad=3)

# accuracy badge next to samples
box(13.35, 5.15, 2.35, 1.5, GREEN, face=BADGE_BG, lw=1.6)
ax.text(14.52, 6.28, 'conv net', fontsize=8.6, fontweight='bold',
        ha='center', va='center', color=GREEN)
ax.text(14.52, 5.86, f'{acc:.1%}', fontsize=15, fontweight='bold',
        ha='center', va='center', color=GREEN)
ax.text(14.52, 5.45, '947 params', fontsize=8.2, ha='center',
        va='center', color=SUBTLE_TEXT)

# ===================== RIGHT BOTTOM: the kernels =====================
ax.text(11.15, 4.35, 'what the nine weights became (real learned kernels)',
        fontsize=9.3, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR)
klabels = ['vertical-edge detector', 'diagonal detector']
for i, (K, f, lab) in enumerate(zip(kernels, top2, klabels)):
    axk = fig.add_axes([0.50 + i * 0.21, 0.085, 0.155, 0.276])
    vmax = np.abs(K).max()
    axk.imshow(K, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    for rr in range(3):
        for cc in range(3):
            axk.text(cc, rr, f'{K[rr, cc]:+.2f}', ha='center', va='center',
                     fontsize=8.4, fontweight='bold',
                     color='white' if abs(K[rr, cc]) > 0.55 * vmax else TEXT_COLOR)
    axk.set_xticks([]); axk.set_yticks([])
    for sp in axk.spines.values():
        sp.set_color('#CBD5E1')
    axk.set_title(f'filter {f} — {lab}', fontsize=8.4, color=SUBTLE_TEXT,
                  pad=4)
ax.text(14.05, 1.7, 'edge detectors\nnobody asked\nfor — V1 got\nthere first',
        fontsize=8.6, ha='left', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

ax.text(8, 0.18, 'Algorithms in Python  |  Deep Learning Architectures Part 2',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '11-deep-learning-architectures/02-convolutional-neural-networks/'
       'header_cnn.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
