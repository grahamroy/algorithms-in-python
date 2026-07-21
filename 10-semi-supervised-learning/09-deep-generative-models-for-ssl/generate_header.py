"""Generate the header image for the Deep Generative Models for SSL article.

Right panels show REAL fits from deep_generative.py on draw 0: the Gaussian
mixture's 2-sigma ellipses vs the decoded curves of the two VAEs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
import numpy as np

import deep_generative as dg

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- reproduce the script's data and draw-0 fits -------------------------
rng = np.random.default_rng(dg.RNG_SEED)
X, y = dg.make_moons(250, noise=0.15, rng=rng)
X_test, y_test = dg.make_moons(250, noise=0.15, rng=rng)
d_rng = np.random.default_rng(0)
li = np.concatenate([d_rng.choice(np.where(y == 0)[0], 4, replace=False),
                     d_rng.choice(np.where(y == 1)[0], 4, replace=False)])

mus, Ss, pis = dg.gmm_ssl(X, li, y[li])
acc_gmm = float((dg.gmm_logjoint(X_test, mus, Ss, pis).argmax(axis=1)
                 == y_test).mean())
vaes = dg.movae_ssl(X, li, y[li], seed=0)
acc_vae = float((dg.movae_logjoint(vaes, X_test, seed=0).argmax(axis=1)
                 == y_test).mean())
print(f'draw 0: GMM {acc_gmm:.1%}, MoVAE {acc_vae:.1%}')
assert abs(acc_gmm - 0.838) < 1e-9 and abs(acc_vae - 0.976) < 1e-9

zs = np.linspace(-2.2, 2.2, 200)[:, None]
curves = [v.dec.fwd(zs)[0] for v in vaes]

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Deep Generative Models: The Story Learns to Bend',
        fontsize=18.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        "Part 4's EM bargain with a neural decoder for a story — the "
        'Gaussian ceiling breaks: 81.1% becomes 98.0% on eight labels.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the story upgrade =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.45, 4.9, 1.05, TEXT_COLOR)
ax.text(3.15, 7.22, 'the generative bargain (Part 4)', fontsize=9.5,
        fontweight='bold', ha='center', va='center', color=TEXT_COLOR)
ax.text(3.15, 6.82, 'every class is a story p(x | class) —\n'
                    'unlabelled points had to be made by something',
        fontsize=7.9, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.85, 4.9, 1.15, RED, face=BADGE_BG, lw=1.7)
ax.text(3.15, 5.72, 'the old story: a Gaussian per class', fontsize=9.3,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.15, 5.28, 'one ellipse must drape one curved moon —\n'
                    'capped at 81.1%, and EM cannot fix geometry',
        fontsize=7.9, ha='center', va='center', color=TEXT_COLOR)

ax.add_patch(FancyArrowPatch((3.15, 4.8), (3.15, 4.35), arrowstyle='-|>',
             mutation_scale=13, color=PURPLE, linewidth=1.8))
ax.text(3.55, 4.58, 'upgrade the story', fontsize=8.3, ha='left',
        va='center', color=PURPLE, fontstyle='italic', fontweight='bold')

box(0.7, 3.05, 4.9, 1.3, PURPLE, face=BADGE_BG)
ax.text(3.15, 4.05, 'the new story: a tiny VAE per class', fontsize=9.3,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.15, 3.52, 'z ~ N(0,1)  →  decoder bends the latent line\n'
                    'into a curve — a moon IS a curve + noise',
        fontsize=7.9, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 4.9, 1.5, GREEN, lw=1.6)
ax.text(3.15, 2.35, 'same EM loop, same clamp', fontsize=9.0,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.15, 1.78,
        'E-step: responsibilities from each ELBO, 8 labels fixed\n'
        'M-step: responsibility-weighted gradient steps\n'
        'warm-start each story on its own 4 labels',
        fontsize=7.7, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT: real fits =====================
panels = [
    ('the Gaussian story', f'{acc_gmm:.1%} — the tails claim the other horn',
     0.435),
    ('the VAE story', f'{acc_vae:.1%} — the decoded curves lie along the moons',
     0.055),
]
for pi_, (title, sub, y0) in enumerate(panels):
    axp = fig.add_axes([0.455, y0 + 0.055, 0.52, 0.30])
    axp.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=6, color=ORANGE,
                alpha=0.5, linewidths=0)
    axp.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=6, color=BLUE,
                alpha=0.5, linewidths=0)
    if pi_ == 0:
        for c, col in ((0, ORANGE), (1, BLUE)):
            vals, vecs = np.linalg.eigh(Ss[c])
            ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
            for k in (1, 2):
                axp.add_patch(Ellipse(mus[c], 2*k*np.sqrt(vals[-1]),
                              2*k*np.sqrt(vals[0]), angle=ang,
                              facecolor='none', edgecolor=col,
                              linewidth=2.2 if k == 2 else 1.2,
                              alpha=0.9 if k == 2 else 0.5))
            axp.plot(*mus[c], 'x', color=col, markersize=8, markeredgewidth=2)
    else:
        for c, col in ((0, ORANGE), (1, BLUE)):
            axp.plot(curves[c][:, 0], curves[c][:, 1], color=col,
                     linewidth=2.6, alpha=0.95)
    axp.scatter(X[li][:, 0], X[li][:, 1], s=130, marker='*',
                c=[ORANGE if c == 0 else BLUE for c in y[li]],
                edgecolors='black', linewidths=0.9, zorder=5)
    axp.set_xlim(-1.8, 2.8)
    axp.set_ylim(-1.25, 1.75)
    axp.set_xticks([]); axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_color('#CBD5E1')
    axp.text(0.015, 0.97, title, transform=axp.transAxes, fontsize=9.5,
             fontweight='bold', ha='left', va='top', color=TEXT_COLOR)
    axp.text(0.015, 0.845, sub, transform=axp.transAxes, fontsize=8.2,
             ha='left', va='top', color=SUBTLE_TEXT, fontstyle='italic')

ax.text(11.45, 7.62, 'both stories fitted for real on draw 0 — '
        'ellipses at 1σ and 2σ; curves = the decoders, z swept ±2.2',
        fontsize=8.4, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 9',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/09-deep-generative-models-for-ssl/'
       'header_deep_generative.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
