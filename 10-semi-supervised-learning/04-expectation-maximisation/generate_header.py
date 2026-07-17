"""Generate the header image for the EM article.

The right panels replay the REAL histories from em.py: likelihood and
accuracy per iteration, in the model-right and model-wrong worlds.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import em as EM


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
E_COLOR = '#2563EB'
M_COLOR = '#7C3AED'
OK_COLOR = '#16A34A'
BAD_COLOR = '#DC2626'
LL_COLOR = '#94A3B8'
BADGE_BG = '#F8FAFC'

# ---------------- reproduce the exact runs ----------------
rng = np.random.default_rng(EM.RNG_SEED)
X, y = EM.make_blobs(500, rng)
Xt, yt = EM.make_blobs(500, rng)
lab, unl = EM.split_labels(X, y, rng)
_, hist_ok = EM.semisup_em(X[lab], y[lab], X[unl], X_test=Xt, y_test=yt)

r3 = np.random.default_rng(0)
X3, y3 = EM.make_interleaved(500, r3)
Xt3, yt3 = EM.make_interleaved(500, r3)
lab3, unl3 = EM.split_labels(X3, y3, r3)
_, hist_bad = EM.semisup_em(X3[lab3], y3[lab3], X3[unl3],
                            X_test=Xt3, y_test=yt3, iters=40)

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Expectation-Maximisation: Missing Labels Are Just Missing Data',
        fontsize=17.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        'Posit a story of how the data was made, then alternate: soft-guess '
        'the missing labels, refit the story. Likelihood only ever rises.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the E/M loop =====================
def box(x, y0, w, h, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.9))
    ax.text(x + w/2, y0 + h*0.66, title, fontsize=11, fontweight='bold',
            ha='center', va='center', color=col)
    ax.text(x + w/2, y0 + h*0.28, sub, fontsize=8, ha='center', va='center',
            color=SUBTLE_TEXT)

box(0.7, 5.5, 3.1, 1.3, 'E-STEP', 'soft responsibilities\nP(class | x)', E_COLOR)
box(4.4, 5.5, 3.1, 1.3, 'M-STEP', 'weighted refit of the\nstory\'s parameters',
    M_COLOR)

ax.add_patch(FancyArrowPatch((3.85, 6.5), (4.35, 6.5), arrowstyle='-|>',
             mutation_scale=13, color='#94A3B8', linewidth=1.7))
ax.add_patch(FancyArrowPatch((4.35, 5.8), (3.85, 5.8), arrowstyle='-|>',
             mutation_scale=13, color='#94A3B8', linewidth=1.7))

ax.add_patch(FancyBboxPatch((0.7, 3.7), 6.8, 1.0,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=OK_COLOR, linewidth=1.6))
ax.text(4.1, 4.42, 'the guarantee (Dempster, Laird & Rubin, 1977)',
        fontsize=9, fontweight='bold', ha='center', va='center',
        color=OK_COLOR)
ax.text(4.1, 4.0, 'each iteration can only INCREASE the data\'s log-likelihood',
        fontsize=9.5, ha='center', va='center', color=TEXT_COLOR)

ax.add_patch(FancyBboxPatch((0.7, 2.25), 6.8, 0.95,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=E_COLOR, linewidth=1.4))
ax.text(4.1, 2.93, 'the Part 1 connection', fontsize=9, fontweight='bold',
        ha='center', va='center', color=E_COLOR)
ax.text(4.1, 2.53,
        'self-training is EM with HARD assignments -- EM never commits',
        fontsize=9, ha='center', va='center', color=SUBTLE_TEXT)

ax.text(4.1, 1.55, 'a 58%-A point counts as 0.58 of a point for A -- '
        'forever revisable',
        fontsize=8.6, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: the two worlds =====================
def panel(rect, hist, title, acc_color, note, note_xy, legend_loc):
    axp = fig.add_axes(rect)
    its = [h[0] for h in hist]
    lls = np.array([h[1] for h in hist])
    accs = np.array([h[2] for h in hist]) * 100
    a0, a1 = accs.min(), accs.max()
    pad = max((a1 - a0) * 0.25, 1.0)
    lo, hi = a0 - pad, a1 + pad
    ll_scaled = lo + (lls - lls.min()) / (lls.max() - lls.min()) * (hi - lo)
    axp.plot(its, ll_scaled, '--', color=LL_COLOR, linewidth=1.8,
             label='log-likelihood (scaled)')
    axp.plot(its, accs, '-', color=acc_color, linewidth=2.4,
             label='test accuracy')
    axp.set_ylim(lo - pad*0.3, hi + pad*0.5)
    axp.set_title(title, fontsize=10.5, fontweight='bold', color=TEXT_COLOR,
                  pad=6)
    axp.set_xlabel('EM iteration', fontsize=8.5, color=SUBTLE_TEXT)
    axp.set_ylabel('accuracy (%)', fontsize=8.5, color=SUBTLE_TEXT)
    axp.legend(fontsize=7, loc=legend_loc, frameon=False)
    axp.text(*note_xy, note, fontsize=8.5, fontweight='bold',
             color=acc_color, ha='center',
             transform=axp.transAxes)
    for sp in ('top', 'right'):
        axp.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        axp.spines[sp].set_color('#CBD5E1')
    axp.tick_params(colors=SUBTLE_TEXT, labelsize=7.5)
    return axp

panel([0.575, 0.16, 0.185, 0.52], hist_ok,
      'the model is right', OK_COLOR,
      'both rise:\n81.8% -> 88.4%', (0.6, 0.3), 'lower right')
panel([0.795, 0.16, 0.185, 0.52], hist_bad,
      'the model is wrong', BAD_COLOR,
      'likelihood rises,\naccuracy falls:\n56.2% -> 52.4%', (0.62, 0.6),
      'lower left')
ax.text(12.4, 6.85, 'the same guarantee, two worlds -- real runs from em.py',
        fontsize=9, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 4',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/04-expectation-maximisation/header_em.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
