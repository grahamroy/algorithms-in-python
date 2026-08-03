"""Generate the header image for the MLP article.

Right panels show REAL decision surfaces from mlp.py: the linear model
(57.2%) and the two-storey [16,16] network (94.8%, seed 0) on the
section's new stage, two spirals.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import mlp as M

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- reproduce the script's data and two models ---------------------------
rng = np.random.default_rng(M.RNG_SEED)
X, y = M.make_spirals(250, M.NOISE, rng)
X_test, y_test = M.make_spirals(250, M.NOISE, rng)

lin = M.train([], X, y)
deep = M.train([16, 16], X, y, seed=0)
acc_lin = M.accuracy(lin, X_test, y_test)
acc_deep = M.accuracy(deep, X_test, y_test)
print(f'linear {acc_lin:.1%}, [16,16] {acc_deep:.1%}')
assert abs(acc_lin - 0.572) < 1e-9 and abs(acc_deep - 0.948) < 1e-9

lim = 1.25
gx, gy = np.meshgrid(np.linspace(-lim, lim, 240), np.linspace(-lim, lim, 240))
G = np.stack([gx.ravel(), gy.ravel()], axis=1)
P_lin = lin.probs(G)[:, 1].reshape(gx.shape)
P_deep = deep.probs(G)[:, 1].reshape(gx.shape)

cmap = LinearSegmentedColormap.from_list(
    'spiral', ['#F5C6A5', '#FDF0E6', '#FFFFFF', '#E4EDFB', '#A8C4EE'])

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Multi-Layer Perceptron: Width Memorises, Depth Composes',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        "Deep learning's atom, taken apart: one crease per hidden unit — "
        'and two small storeys beat one wide wall on every seed.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the three ideas =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.3, 5.2, 1.25, TEXT_COLOR)
ax.text(3.3, 7.28, 'a layer:  a = act( xW + b )', fontsize=10.5,
        fontweight='bold', ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.3, 6.78, 'affine map + pointwise nonlinearity —\n'
                   'without act(), a stack of layers folds flat into one',
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.55, 5.2, 1.35, PURPLE, face=BADGE_BG)
ax.text(3.3, 5.62, 'backprop:  the chain rule + bookkeeping',
        fontsize=9.8, fontweight='bold', ha='center', va='center',
        color=PURPLE)
ax.text(3.3, 5.06, 'forward: keep every activation\n'
                   'backward: start from p − onehot, multiply\n'
                   'local slopes layer by layer, in reverse',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 2.95, 5.2, 1.2, GREEN, lw=1.7)
ax.text(3.3, 3.9, 'the audit', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=GREEN)
ax.text(3.3, 3.4, 'every gradient vs central differences:\n'
                  'max disagreement 1.25e-10 across 49 parameters',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.0, 5.2, 1.55, RED, face=BADGE_BG, lw=1.7)
ax.text(3.3, 2.28, 'two ways a network dies', fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.3, 1.62, 'zero init: gradients exactly 0.0 forever (stillborn)\n'
                   'six sigmoid storeys: gradient shrinks ~25,000x\n'
                   'by the input layer (vanishing) — ReLU is the fix',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT: the two surfaces =====================
panels = [
    ('no hidden layer — a linear model',
     f'{acc_lin:.1%} — a straight line laid across a spiral',
     P_lin, 0.435),
    ('[16, 16] — two storeys',
     f'{acc_deep:.1%} — the spiral, composed from creases',
     P_deep, 0.055),
]
for title, sub, Pgrid, y0 in panels:
    axp = fig.add_axes([0.47, y0 + 0.055, 0.355, 0.30])
    axp.contourf(gx, gy, Pgrid, levels=np.linspace(0, 1, 21), cmap=cmap)
    axp.contour(gx, gy, Pgrid, levels=[0.5], colors=[TEXT_COLOR],
                linewidths=1.8)
    axp.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=5, color=ORANGE,
                alpha=0.6, linewidths=0)
    axp.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=5, color=BLUE,
                alpha=0.6, linewidths=0)
    axp.set_xlim(-lim, lim)
    axp.set_ylim(-lim, lim)
    axp.set_xticks([]); axp.set_yticks([])
    axp.set_aspect('equal')
    for sp in axp.spines.values():
        sp.set_color('#CBD5E1')
    bb = dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5)
    axp.text(0.02, 0.975, title, transform=axp.transAxes, fontsize=9.3,
             fontweight='bold', ha='left', va='top', color=TEXT_COLOR,
             bbox=bb)
    axp.text(0.02, 0.845, sub, transform=axp.transAxes, fontsize=7.9,
             ha='left', va='top', color=SUBTLE_TEXT, fontstyle='italic',
             bbox=bb)

# ladder strip to the right of panels
axl = fig.add_axes([0.855, 0.13, 0.125, 0.60])
arch = ['linear', '[2]', '[4]', '[8]', '[16]', '[64]', '[16,16]']
accs = [57.2, 55.2, 55.4, 57.6, 67.8, 84.2, 94.8]
cols = [SUBTLE_TEXT] * 6 + [GREEN]
ypos = np.arange(len(arch))
axl.barh(ypos, accs, color=cols, alpha=0.85, height=0.62)
for yp, a in zip(ypos, accs):
    axl.text(a - 2, yp, f'{a:.0f}%', fontsize=7.4, ha='right',
             va='center', color='white', fontweight='bold')
axl.set_yticks(ypos)
axl.set_yticklabels(arch, fontsize=7.6, color=TEXT_COLOR)
axl.invert_yaxis()
axl.set_xlim(0, 100)
axl.set_xticks([])
axl.set_title('the ladder\n(test acc)', fontsize=8.6, color=TEXT_COLOR,
              fontweight='bold', pad=4)
for sp in axl.spines.values():
    sp.set_visible(False)
axl.tick_params(left=False)

ax.text(8, 0.18, 'Algorithms in Python  |  Deep Learning Architectures Part 1',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '11-deep-learning-architectures/01-multi-layer-perceptron/'
       'header_mlp.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
