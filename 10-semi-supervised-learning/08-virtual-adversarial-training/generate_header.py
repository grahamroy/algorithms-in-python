"""Generate the header image for the VAT article.

Right panel shows REAL probability fields: the supervised and VAT networks
from vat.py, trained on draw 0, evaluated on a grid.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import vat

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- reproduce vat.py's data and draw-0 models --------------------------
rng = np.random.default_rng(vat.RNG_SEED)
X, y = vat.make_moons(250, noise=0.15, rng=rng)
X_test, y_test = vat.make_moons(250, noise=0.15, rng=rng)
d_rng = np.random.default_rng(0)
li = np.concatenate([d_rng.choice(np.where(y == 0)[0], 4, replace=False),
                     d_rng.choice(np.where(y == 1)[0], 4, replace=False)])

sup = vat.train(X[li], y[li], X, use_vat=False, seed=0)
adv = vat.train(X[li], y[li], X, use_vat=True, seed=0)
acc_sup = vat.accuracy(sup, X_test, y_test)
acc_adv = vat.accuracy(adv, X_test, y_test)
print(f'draw 0: supervised {acc_sup:.1%}, VAT {acc_adv:.1%}')
assert abs(acc_sup - 0.742) < 1e-9 and abs(acc_adv - 0.908) < 1e-9

gx, gy = np.meshgrid(np.linspace(-1.7, 2.7, 260),
                     np.linspace(-1.2, 1.7, 180))
G = np.stack([gx.ravel(), gy.ravel()], axis=1)
P_sup = sup.probs(G)[:, 1].reshape(gx.shape)
P_adv = adv.probs(G)[:, 1].reshape(gx.shape)

cmap = LinearSegmentedColormap.from_list(
    'moons', ['#F5C6A5', '#FDF0E6', '#FFFFFF', '#E4EDFB', '#A8C4EE'])

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Virtual Adversarial Training: The Boundary May Not Live Here',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'For every point — labelled or not — find the nudge that most changes '
        'the prediction, and penalise it: smoothness written into the loss.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the attack loop =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.9, 6.35, 4.6, 0.95, TEXT_COLOR)
ax.text(3.2, 7.05, 'any point x  +  current belief p(y|x)', fontsize=9.5,
        fontweight='bold', ha='center', va='center', color=TEXT_COLOR)
ax.text(3.2, 6.66, 'no label needed — the model attacks its own prediction',
        fontsize=7.6, ha='center', va='center', color=SUBTLE_TEXT)

box(0.9, 4.1, 4.6, 1.75, PURPLE, face=BADGE_BG)
ax.text(3.2, 5.55, 'POWER ITERATION  (one step)', fontsize=9.5,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.2, 4.92,
        '1.  probe with a tiny random nudge  ξd\n'
        '2.  backprop the KL change to the INPUT\n'
        '3.  normalise  →  worst direction, length ε',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)
ax.text(5.85, 4.98, 'two extra\npasses', fontsize=7.8, ha='left',
        va='center', color=SUBTLE_TEXT, fontstyle='italic')

box(0.9, 2.7, 4.6, 0.95, RED)
ax.text(3.2, 3.4, 'penalise the damage', fontsize=9.5, fontweight='bold',
        ha='center', va='center', color=RED)
ax.text(3.2, 3.0, 'loss  +=  α · KL( p(x)  ‖  p(x + r_adv) )',
        fontsize=8.6, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

for y1, y2 in ((6.3, 5.9), (4.05, 3.7)):
    ax.add_patch(FancyArrowPatch((3.2, y1), (3.2, y2), arrowstyle='-|>',
                 mutation_scale=12, color=TEXT_COLOR, linewidth=1.6))
ax.text(3.45, 3.87, 'r_adv', fontsize=8.4, ha='left', va='center',
        color=PURPLE, fontfamily='DejaVu Sans Mono')

box(0.9, 1.15, 4.6, 1.05, GREEN, face=BADGE_BG, lw=1.6)
ax.text(3.2, 1.93, 'the geometric consequence', fontsize=8.8,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.2, 1.5, 'a boundary near any data point is expensive —\n'
                  'training evicts it into the low-density gap',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)
ax.add_patch(FancyArrowPatch((3.2, 2.65), (3.2, 2.25), arrowstyle='-|>',
             mutation_scale=12, color=GREEN, linewidth=1.6))

# ===================== RIGHT: real probability fields =====================
panels = [
    ('supervised only: 8 labels', f'{acc_sup:.1%} — boundary through the moons',
     P_sup, [0.435, 0.435]),
    (f'+ VAT penalty on all 500 points', f'{acc_adv:.1%} — evicted into the gap',
     P_adv, [0.435, 0.055]),
]
for title, sub, P, (x0, y0) in panels:
    axp = fig.add_axes([x0 + 0.02, y0 + 0.055, 0.52, 0.30])
    axp.contourf(gx, gy, P, levels=np.linspace(0, 1, 21), cmap=cmap)
    axp.contour(gx, gy, P, levels=[0.5], colors=[TEXT_COLOR],
                linewidths=1.8)
    axp.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=6, color=ORANGE,
                alpha=0.55, linewidths=0)
    axp.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=6, color=BLUE,
                alpha=0.55, linewidths=0)
    axp.scatter(X[li][:, 0], X[li][:, 1], s=130, marker='*',
                c=[ORANGE if c == 0 else BLUE for c in y[li]],
                edgecolors='black', linewidths=0.9, zorder=5)
    axp.set_xlim(-1.7, 2.7)
    axp.set_ylim(-1.2, 1.7)
    axp.set_xticks([]); axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_color('#CBD5E1')
    axp.text(0.015, 0.97, title, transform=axp.transAxes, fontsize=9.5,
             fontweight='bold', ha='left', va='top', color=TEXT_COLOR)
    axp.text(0.015, 0.845, sub, transform=axp.transAxes, fontsize=8.2,
             ha='left', va='top', color=SUBTLE_TEXT, fontstyle='italic')

ax.text(11.6, 7.62, 'same network, same 8 stars — one new loss term  (draw 0, real runs)',
        fontsize=8.6, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 8',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/08-virtual-adversarial-training/'
       'header_vat.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
