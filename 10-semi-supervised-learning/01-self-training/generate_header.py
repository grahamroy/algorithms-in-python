"""Generate the header image for the Self-Training article.

The right panel is the REAL experiment: the exact dataset, seeds, and
pseudo-label rounds from self_training.py, re-run here deterministically.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import self_training as ST


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
LOOP_COLOR = '#3B82F6'
TAU_COLOR = '#7C3AED'
C0 = '#EA580C'          # class 0 (top moon)
C1 = '#2563EB'          # class 1 (bottom moon)
WRONG_C = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---------------- re-run the exact experiment, tracking rounds ----------------
rng = np.random.default_rng(ST.RNG_SEED)
X, y = ST.make_moons(250, noise=ST.NOISE, rng=rng)
lab_idx = ST.spread_seeds(X, y, 4)
unl = np.ones(len(X), dtype=bool)
unl[lab_idx] = False
X_lab, y_lab = X[lab_idx], y[lab_idx]
pool, pool_true = X[unl], y[unl]

XL, YL = X_lab.copy(), y_lab.copy()
P, PT = pool.copy(), pool_true.copy()
pts, pt_lab, pt_round, pt_wrong = [], [], [], []
for rnd in range(1, 31):
    model = ST.KNN().fit(XL, YL)
    if len(P) == 0:
        break
    proba = model.predict_proba(P)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    take = conf >= ST.TAU
    if not take.any():
        break
    pts.append(P[take])
    pt_lab.append(pred[take])
    pt_round.append(np.full(int(take.sum()), rnd))
    pt_wrong.append(pred[take] != PT[take])
    XL = np.concatenate([XL, P[take]])
    YL = np.concatenate([YL, pred[take]])
    P, PT = P[~take], PT[~take]
pts = np.concatenate(pts)
pt_lab = np.concatenate(pt_lab)
pt_round = np.concatenate(pt_round)
pt_wrong = np.concatenate(pt_wrong)
n_rounds = pt_round.max()

# ---------------- figure ----------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Self-Training: Teaching a Model With Its Own Best Guesses',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        'Train on a few labels, trust the confident predictions, retrain — '
        'and confidence spreads through the clusters.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the loop =====================
ax.text(3.75, 7.1, 'The wrapper (around ANY classifier)', fontsize=11.5,
        fontweight='bold', ha='center', va='center', color=TEXT_COLOR)

def step_box(x, y, num, text, col=LOOP_COLOR):
    ax.add_patch(FancyBboxPatch((x, y), 3.05, 0.85,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.6))
    ax.text(x + 0.3, y + 0.42, num, fontsize=12.5, fontweight='bold',
            ha='center', va='center', color=col)
    ax.text(x + 0.55, y + 0.42, text, fontsize=8.6, ha='left', va='center',
            color=TEXT_COLOR)

step_box(0.45, 5.85, '1', 'train on the labelled set')
step_box(4.0, 5.85, '2', 'predict the unlabelled pool')
step_box(4.0, 4.5, '3', 'keep predictions with\nconfidence >= τ', TAU_COLOR)
step_box(0.45, 4.5, '4', 'add as pseudo-labels,\nretrain')

arrows = [((3.55, 6.27), (3.95, 6.27)),
          ((5.55, 5.8), (5.55, 5.42)),
          ((3.95, 4.92), (3.55, 4.92)),
          ((1.95, 5.42), (1.95, 5.8))]
for (x0, y0), (x1, y1) in arrows:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                 mutation_scale=12, color='#94A3B8', linewidth=1.5))

ax.add_patch(FancyBboxPatch((0.45, 2.85), 6.6, 1.05,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=TAU_COLOR, linewidth=1.5))
ax.text(3.75, 3.62, 'the guard rail: the confidence threshold τ',
        fontsize=9.5, fontweight='bold', ha='center', va='center',
        color=TAU_COLOR)
ax.text(3.75, 3.18,
        'every pseudo-label is a bet; wrong bets are retrained on as truth\n'
        'and compound (confirmation bias)',
        fontsize=8.4, ha='center', va='center', color=SUBTLE_TEXT)

ax.text(3.75, 2.2,
        'here: k-NN base, pseudo-label only on unanimous neighbour votes',
        fontsize=8.6, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: the real frontier =====================
ax2 = fig.add_axes([0.575, 0.115, 0.4, 0.62])
# remaining unlabelled (never promoted)
ax2.scatter(P[:, 0], P[:, 1], s=14, color='#D1D5DB', zorder=1,
            label='never labelled')
# pseudo-labels shaded by round (early = dark, late = light)
for r in range(1, n_rounds + 1):
    m = pt_round == r
    alpha = max(0.25, 1.0 - 0.16 * (r - 1))
    for cls, col in ((0, C0), (1, C1)):
        mm = m & (pt_lab == cls) & ~pt_wrong
        ax2.scatter(pts[mm, 0], pts[mm, 1], s=16, color=col, alpha=alpha,
                    zorder=2, linewidths=0)
# the wrong pseudo-labels
w = pt_wrong
ax2.scatter(pts[w, 0], pts[w, 1], s=46, marker='x', color=WRONG_C,
            linewidths=1.6, zorder=4, label='wrong pseudo-label (11)')
# the 8 seeds
for cls, col in ((0, C0), (1, C1)):
    m = y_lab == cls
    ax2.scatter(X_lab[m, 0], X_lab[m, 1], s=210, marker='*', color=col,
                edgecolor='black', linewidths=1.1, zorder=5)
ax2.scatter([], [], s=210, marker='*', color='white', edgecolor='black',
            linewidths=1.1, label='the 8 real labels')

ax2.set_title('The real run: 8 labels → 489 pseudo-labels',
              fontsize=11.5, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.text(0.985, 0.97, 'darker = earlier round', transform=ax2.transAxes,
         fontsize=8, color=SUBTLE_TEXT, ha='right', va='top',
         fontstyle='italic')
ax2.legend(fontsize=8, loc='lower left', frameon=False)
ax2.set_xticks([]); ax2.set_yticks([])
for sp in ax2.spines.values():
    sp.set_color('#E2E8F0')

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 1',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/01-self-training/header_self_training.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}  (rounds tracked: {n_rounds}, '
      f'wrong: {int(pt_wrong.sum())})')
