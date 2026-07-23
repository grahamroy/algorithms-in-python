"""Generate the header image for the Transductive SVM article.

Right panels show REAL streets from tsvm.py on draw 0: the supervised
SVM's margin band (|f| < 1) runs through the moons; the chosen TSVM
restart's band sits in the gap.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import tsvm as T

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'
STREET = '#94A3B8'

# ---- reproduce the script's data and draw-0 models -----------------------
rng = np.random.default_rng(T.RNG_SEED)
X, y = T.make_moons(250, noise=0.15, rng=rng)
X_test, y_test = T.make_moons(250, noise=0.15, rng=rng)
Y, Y_test = 2 * y - 1, 2 * y_test - 1
d_rng = np.random.default_rng(0)
li = np.concatenate([d_rng.choice(np.where(y == 0)[0], 4, replace=False),
                     d_rng.choice(np.where(y == 1)[0], 4, replace=False)])

phi0 = T.RFF(seed=0)
w_s, b_s = T.train_svm(phi0(X)[li], Y[li], phi0(X), cu_max=0.0, anneal=False)
acc_s = float(((phi0(X_test) @ w_s + b_s > 0) == (Y_test > 0)).mean())
in_s = float((np.abs(phi0(X) @ w_s + b_s) < 1).mean())

best = None
for ps in range(T.RESTARTS):
    phi = T.RFF(seed=ps)
    P = phi(X)
    w, b = T.train_svm(P[li], Y[li], P, balance=float(Y[li].mean()))
    J = T.objective(w, b, P[li], Y[li], P)
    if best is None or J < best[0]:
        best = (J, ps, w, b)
J_t, ps_t, w_t, b_t = best
phi_t = T.RFF(seed=ps_t)
acc_t = float(((phi_t(X_test) @ w_t + b_t > 0) == (Y_test > 0)).mean())
in_t = float((np.abs(phi_t(X) @ w_t + b_t) < 1).mean())
print(f'draw 0: supervised {acc_s:.1%} ({in_s:.1%} inside), '
      f'tsvm {acc_t:.1%} ({in_t:.1%} inside, phi {ps_t})')
assert abs(acc_s - 0.738) < 1e-9 and abs(acc_t - 0.986) < 1e-9

gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 260),
                     np.linspace(-1.1, 1.6, 180))
G = np.stack([gx.ravel(), gy.ravel()], axis=1)
F_s = (phi0(G) @ w_s + b_s).reshape(gx.shape)
F_t = (phi_t(G) @ w_t + b_t).reshape(gx.shape)
# only show f where it means something: near the data
d2 = ((G[:, None, :] - X[None, ::2, :]) ** 2).sum(-1).min(1)
far = (np.sqrt(d2) > 0.32).reshape(gx.shape)
F_s = np.where(far, np.nan, F_s)
F_t = np.where(far, np.nan, F_t)

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Transductive SVM: The Widest Empty Street',
        fontsize=19, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Punish any unlabelled point standing inside the margin — the '
        'street relocates to the gap, because only there does it fit.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the objective =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.55, 4.9, 0.95, TEXT_COLOR)
ax.text(3.15, 7.28, 'labelled points: the hinge', fontsize=9.3,
        fontweight='bold', ha='center', va='center', color=TEXT_COLOR)
ax.text(3.15, 6.9, 'stay on YOUR side of the street, outside it',
        fontsize=8.0, ha='center', va='center', color=SUBTLE_TEXT)

# hat-loss mini plot
axh = fig.add_axes([0.075, 0.44, 0.24, 0.20])
m = np.linspace(-2.4, 2.4, 200)
axh.plot(m, np.maximum(0, 1 - np.abs(m)), color=PURPLE, linewidth=2.4)
axh.axvspan(-1, 1, color=STREET, alpha=0.18)
axh.set_ylim(-0.12, 1.25)
axh.set_xticks([-1, 0, 1])
axh.set_xticklabels(['-1', '0', '+1'], fontsize=7.5)
axh.set_yticks([])
for sp in ('top', 'right', 'left'):
    axh.spines[sp].set_visible(False)
axh.spines['bottom'].set_color('#CBD5E1')
axh.tick_params(colors=SUBTLE_TEXT, length=2)
axh.text(0, 1.13, 'the hat loss  max(0, 1 - |f|)', fontsize=8.8,
         fontweight='bold', ha='center', va='center', color=PURPLE)
axh.text(0, -0.02, 'inside the street', fontsize=7.2, ha='center',
         va='bottom', color=SUBTLE_TEXT)
ax.text(3.15, 3.38, 'unlabelled points: clear the street — either side.\n'
                    'two downhill directions  =  NON-CONVEX',
        fontsize=8.0, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 2.15, 4.9, 0.95, RED, face=BADGE_BG, lw=1.7)
ax.text(3.15, 2.87, 'the balance pin', fontsize=9.3, fontweight='bold',
        ha='center', va='center', color=RED)
ax.text(3.15, 2.48, 'bias set so the pool splits like the labels —\n'
                    'else the cheapest empty street is around the town',
        fontsize=7.8, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 0.95, 4.9, 0.95, GREEN, lw=1.6)
ax.text(3.15, 1.67, 'restarts, keep the lowest objective', fontsize=9.3,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.15, 1.28, 'emptier streets score lower — no labels needed '
                    'to choose', fontsize=7.8, ha='center', va='center',
        color=TEXT_COLOR)

# ===================== RIGHT: real streets =====================
panels = [
    ('the supervised street',
     f'{acc_s:.1%} — {in_s:.0%} of the pool stands inside it', F_s, 0.435),
    ('the transductive street',
     f'{acc_t:.1%} — occupancy {in_t:.1%}, the street found the gap',
     F_t, 0.055),
]
for title, sub, F, y0 in panels:
    axp = fig.add_axes([0.455, y0 + 0.055, 0.52, 0.30])
    axp.contourf(gx, gy, np.abs(F), levels=[0, 1], colors=[STREET],
                 alpha=0.28)
    axp.contour(gx, gy, F, levels=[0], colors=[TEXT_COLOR], linewidths=1.9)
    axp.contour(gx, gy, F, levels=[-1, 1], colors=[STREET],
                linewidths=1.0, linestyles='--')
    axp.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=6, color=ORANGE,
                alpha=0.55, linewidths=0)
    axp.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=6, color=BLUE,
                alpha=0.55, linewidths=0)
    axp.scatter(X[li][:, 0], X[li][:, 1], s=130, marker='*',
                c=[ORANGE if c == 0 else BLUE for c in y[li]],
                edgecolors='black', linewidths=0.9, zorder=5)
    axp.set_xlim(-1.6, 2.6)
    axp.set_ylim(-1.1, 1.6)
    axp.set_xticks([]); axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_color('#CBD5E1')
    bb = dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5)
    axp.text(0.015, 0.97, title, transform=axp.transAxes, fontsize=9.5,
             fontweight='bold', ha='left', va='top', color=TEXT_COLOR,
             bbox=bb)
    axp.text(0.015, 0.845, sub, transform=axp.transAxes, fontsize=8.2,
             ha='left', va='top', color=SUBTLE_TEXT, fontstyle='italic',
             bbox=bb)

ax.text(11.45, 7.62, 'real fits, draw 0 — shaded band = the street '
        '(|f| < 1), solid line = the boundary, stars = the 8 labels',
        fontsize=8.4, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 10',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/10-transductive-svm/header_tsvm.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
