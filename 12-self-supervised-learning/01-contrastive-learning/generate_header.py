"""Generate the header image for the contrastive-learning article.

Right panels are REAL: the shape-world stage, and PCA projections of
the actual pretrained embeddings (full menu vs no-shift), retrained
live with the same seeds as contrastive.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import contrastive as C

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'
CLS_COLS = [ORANGE, BLUE, GREEN]

# ---- rebuild the stage and the two encoders ------------------------------
rng = np.random.default_rng(0)
imgs, y, factors = C.make_instances(C.N_UNLAB, rng)
te_imgs, te_y, te_f = C.make_instances(C.N_TEST,
                                       np.random.default_rng(50))
Xtr = C.augment(imgs, np.random.default_rng(54))
Xte = C.augment(te_imgs, np.random.default_rng(55))

enc = C.Encoder(seed=0)
C.pretrain(enc, imgs, np.random.default_rng(1))
ca_full = C.probe_acc(enc.embed(Xtr), y, enc.embed(Xte), te_y)
print(f'full-menu probe {ca_full:.3f}')
assert f'{ca_full:.1%}' == '99.7%'

enc_ns = C.Encoder(seed=0)
C.pretrain(enc_ns, imgs, np.random.default_rng(1),
           augs=dict(shift=False))
ca_ns = C.probe_acc(enc_ns.embed(Xtr), y, enc_ns.embed(Xte), te_y)
print(f'no-shift probe {ca_ns:.3f}')
assert f'{ca_ns:.1%}' == '51.7%'

Xte_pos = C.augment(te_imgs, np.random.default_rng(53), shift=False)


def pca2(Z):
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ Vt[:2].T


P_full = pca2(enc.embed(Xte_pos))
P_ns = pca2(enc_ns.embed(Xte_pos))

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Contrastive Learning: The Augmentation Is the Supervision',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Two views of the same thing should agree \u2014 no labels '
        'anywhere, and the augmentation menu decides what the encoder '
        'may ignore.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.1, 5.3, 1.55, BLUE)
ax.text(3.35, 7.38, 'the rule (NT-Xent)', fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=BLUE)
ax.text(3.35, 6.98, '-log exp(zi.zj/t) / sum_k exp(zi.zk/t)',
        fontsize=8.8, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.5, 'softmax over the batch: which of these\n'
                   '2N views is your sibling?',
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.45, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.55, 'the promise: fifteen labels', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.98, 'pretrain + probe, 15 labels : 95.1%\n'
                    'supervised,     15 labels   : 84.9%\n'
                    'ceiling,     4,000 labels   : 99.7%',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.75, 5.3, 1.25, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.75, 'the menu decides', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 3.25, 'drop the shift aug: class probe 99.7% -> 51.7%\n'
                    'yet NN purity stays 98% -- invariance died,\n'
                    'not information',
        fontsize=7.9, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.15, 'the crowd holds the space open', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 1.65, 'delete the negatives: embedding spread\n'
                    'collapses 158x -- every image, one point',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the embedding =====================
axg = fig.add_axes([0.455, 0.475, 0.30, 0.335])
for c in range(3):
    m = te_y == c
    axg.scatter(P_full[m, 0], P_full[m, 1], s=6, color=CLS_COLS[c],
                alpha=0.65, linewidths=0)
axg.set_xticks([]); axg.set_yticks([])
for sp in axg.spines.values():
    sp.set_color('#CBD5E1')
axg.set_title('the learned space, coloured by class (real runs)',
              fontsize=9.3, fontweight='bold', color=TEXT_COLOR, pad=5)
ax.text(12.35, 6.0, 'PCA of the 32-d embedding\nof 1,000 test images.\n'
        'colour = shape class --\nnever shown in training.\n\n'
        'three clean islands: the\nencoder invented the\nclasses because '
        'the menu\nsaid position, brightness\nand noise do not matter.',
        fontsize=8.4, ha='left', va='center', color=SUBTLE_TEXT)

# ============ RIGHT BOTTOM LEFT: the stage ============
axs = fig.add_axes([0.455, 0.095, 0.245, 0.30])
grid = np.zeros((3 * (C.SIDE + 4) + 2, 3 * (C.SIDE + 4) + 2))
sel = np.random.default_rng(7).choice(len(te_imgs), 9, False)
for k, ix in enumerate(sel):
    r_, c_ = divmod(k, 3)
    y0 = r_ * (C.SIDE + 4) + 2
    x0 = c_ * (C.SIDE + 4) + 2
    grid[y0:y0 + C.SIDE, x0:x0 + C.SIDE] = te_imgs[ix]
axs.imshow(grid, cmap='gray_r', vmin=0, vmax=1.4)
axs.set_xticks([]); axs.set_yticks([])
for sp in axs.spines.values():
    sp.set_color('#CBD5E1')
axs.set_title('the stage: shape world, no labels',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=5)

# ============ RIGHT BOTTOM RIGHT: the no-shift embedding ============
axn = fig.add_axes([0.725, 0.095, 0.235, 0.30])
for c in range(3):
    m = te_y == c
    axn.scatter(P_ns[m, 0], P_ns[m, 1], s=6, color=CLS_COLS[c],
                alpha=0.65, linewidths=0)
axn.set_xticks([]); axn.set_yticks([])
for sp in axn.spines.values():
    sp.set_color('#CBD5E1')
axn.set_title('no shift aug: classes entangled',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=5)

ax.text(8, 0.18,
        'Algorithms in Python  |  Self-Supervised Learning Part 1',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '12-self-supervised-learning/01-contrastive-learning/'
       'header_contrastive.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
