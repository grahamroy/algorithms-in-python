"""Generate the header image for the masked-autoencoding article.

Right panels are REAL: the reconstructions come from the actual
DEMO-1 MAE retrained live (same seeds), and the two charts plot the
measured probe/fine-tune numbers, re-measured in this script and
asserted against the frozen run.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import mae as M

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- rebuild everything with the frozen seeds ----------------------------
rng = np.random.default_rng(0)
imgs, y, factors = M.make_instances(M.N_UNLAB, rng)
te_imgs, te_y, te_f = M.make_instances(M.N_TEST,
                                       np.random.default_rng(50))
Xtr = M.augment(imgs, np.random.default_rng(54))
Xte = M.augment(te_imgs, np.random.default_rng(55))
r = np.random.default_rng(2)
few = np.concatenate([r.choice(np.where(y == c)[0], 5, False)
                      for c in range(3)])
Xtr_flat = imgs.reshape(len(imgs), -1)

m = M.MAE(seed=0)
r1 = np.random.default_rng(1)
for _ in range(3000):
    idx = r1.integers(0, len(imgs), M.BATCH)
    m.step(Xtr_flat[idx], r1)
enc = M.Encoder(seed=0)
M.pretrain_contrastive(enc, imgs, np.random.default_rng(1))

p_mae = M.probe_acc(m.embed(Xtr)[few], y[few], m.embed(Xte), te_y)
p_con = M.probe_acc(enc.embed(Xtr)[few], y[few], enc.embed(Xte), te_y)
ft_mae = M.finetune_mae(m, imgs, y, few, Xte, te_y)
ft_con = M.finetune_contrastive(enc, imgs, y, few, Xte, te_y)
sup = M.supervised_scratch(imgs, y, few, Xte, te_y)
print(p_mae, p_con, ft_mae, ft_con, sup)
assert f'{p_mae:.1%}' == '49.2%' and f'{p_con:.1%}' == '95.1%'
assert f'{ft_mae:.1%}' == '87.5%' and f'{ft_con:.1%}' == '91.1%'
assert f'{sup:.1%}' == '84.9%'

Xc = M.augment(te_imgs, np.random.default_rng(53), shift=False)
Xs = M.augment(te_imgs.copy(), np.random.default_rng(53), shift=True)
ratios = {}
for name, embed in (("contrastive", enc.embed), ("MAE", m.embed)):
    Zc, Zs = embed(Xc), embed(Xs)
    d_pair = float(np.sqrt(((Zc - Zs) ** 2).sum(1)).mean())
    d_all = float(np.sqrt(((Zc[:200, None, :] - Zc[None, :200, :]) ** 2)
                          .sum(-1)).mean())
    ratios[name] = d_pair / d_all
print(ratios)
assert f"{ratios['contrastive']:.2f}" == '0.35'
assert f"{ratios['MAE']:.2f}" == '0.83'

# reconstructions for the strip
r9 = np.random.default_rng(9)
sel = np.random.default_rng(11).choice(len(te_imgs), 3, False)
X6 = te_imgs[sel].reshape(3, -1)
inp, pixhide = M.mask_batch(X6, r9)
H, _ = m.enc.forward(inp)
R, _ = m.dec.forward(H)
masked_vis = np.where(pixhide, np.nan, X6)

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Masked Autoencoding: Learning by Filling In the Blanks',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Hide three quarters of the input and predict it \u2014 weak '
        'at the probe, transformed by fine-tuning, because the '
        'reconstructor keeps everything.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.1, 5.3, 1.55, BLUE)
ax.text(3.35, 7.38, 'the pretext', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=BLUE)
ax.text(3.35, 6.98, 'blank 12/16 patches -> regress the pixels\n'
                    'loss scored ONLY on what was hidden',
        fontsize=8.6, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.5, "BERT's recipe for text, MAE's for images --\n"
                   'the supervision is redundancy itself',
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.45, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.55, 'the probes say failure...', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.98, 'linear probe, 15 labels:\n'
                    'MAE 49.2%   vs   contrastive 95.1%\n'
                    'not linearly organised by class',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.75, 5.3, 1.25, GREEN, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.75, '...until you fine-tune', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 3.25, 'MAE 49.2% -> 87.5% (past scratch, 84.9%)\n'
                    'contrastive 95.1% -> 91.1% -- the flip',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.25, RED, lw=1.6)
ax.text(3.35, 2.15, 'the reconstructor keeps everything',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=RED)
ax.text(3.35, 1.65, 'a shift moves its embedding 0.83x an image-\n'
                    'to-image distance (contrastive: 0.35x)',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ============ RIGHT TOP: the reconstructions ============
titles = ['original', 'masked 75%', 'reconstructed']
for col, (data, cmap) in enumerate((
        (X6, 'gray_r'), (masked_vis, 'gray_r'), (R, 'gray_r'))):
    for row in range(3):
        axi = fig.add_axes([0.47 + col * 0.093, 0.72 - row * 0.118,
                            0.085, 0.11])
        img = data[row].reshape(M.SIDE, M.SIDE)
        axi.imshow(img, cmap=cmap, vmin=-0.1, vmax=1.4)
        axi.set_xticks([]); axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_color('#CBD5E1')
        if row == 0:
            axi.set_title(titles[col], fontsize=8.6, fontweight='bold',
                          color=TEXT_COLOR, pad=4)
ax.text(12.6, 5.9, 'real reconstructions:\nthe grey blocks are the\n'
        '12 hidden patches; the\nmodel redraws the shape\nfrom the '
        'quarter it saw.\n\nto fill a blank you must\nknow what the '
        'thing is --\nthat necessity is the\nwhole training signal.',
        fontsize=8.4, ha='left', va='center', color=SUBTLE_TEXT)

# ============ RIGHT BOTTOM LEFT: the flip ============
axb = fig.add_axes([0.455, 0.095, 0.245, 0.30])
xpos = np.arange(3)
frozen = [p_mae * 100, p_con * 100, np.nan]
tuned = [ft_mae * 100, ft_con * 100, sup * 100]
axb.bar(xpos - 0.18, frozen, 0.34, color='#94A3B8',
        label='frozen probe')
axb.bar(xpos + 0.18, tuned, 0.34, color=GREEN, label='fine-tuned')
axb.set_xticks(xpos)
axb.set_xticklabels(['MAE', 'contrastive', 'scratch'], fontsize=8)
axb.set_ylim(0, 105)
axb.set_ylabel('accuracy (%)', fontsize=8, color=SUBTLE_TEXT)
axb.legend(fontsize=7.2, frameon=False, loc='upper left')
for sp in ('top', 'right'):
    axb.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axb.spines[sp].set_color('#CBD5E1')
axb.tick_params(colors=SUBTLE_TEXT, labelsize=7.5)
axb.set_title('the fine-tune flip (15 labels)', fontsize=9.0,
              fontweight='bold', color=TEXT_COLOR, pad=5)

# ============ RIGHT BOTTOM RIGHT: shift sensitivity ============
axs2 = fig.add_axes([0.755, 0.095, 0.205, 0.30])
axs2.barh([1, 0], [ratios['contrastive'], ratios['MAE']],
          0.55, color=[BLUE, RED])
axs2.set_yticks([1, 0])
axs2.set_yticklabels(['contrastive', 'MAE'], fontsize=8)
axs2.set_xlim(0, 1.0)
axs2.axvline(1.0, color='#CBD5E1', linewidth=1, linestyle=':')
axs2.set_xlabel('embedding shift / typical distance', fontsize=7.6,
                color=SUBTLE_TEXT)
for sp in ('top', 'right'):
    axs2.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axs2.spines[sp].set_color('#CBD5E1')
axs2.tick_params(colors=SUBTLE_TEXT, labelsize=7.5)
axs2.set_title('what a shift does', fontsize=9.0,
               fontweight='bold', color=TEXT_COLOR, pad=5)

ax.text(8, 0.18,
        'Algorithms in Python  |  Self-Supervised Learning Part 2',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '12-self-supervised-learning/02-masked-autoencoding/'
       'header_mae.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
