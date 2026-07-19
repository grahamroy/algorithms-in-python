"""Generate the header image for the Graph-Based Methods article.

Right panel: the REAL Fiedler vector of the actual 1,000-node graph,
painted onto the points -- classes found with zero labels.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import graph_methods as GM


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
HARD_C = '#EA580C'
SOFT_C = '#2563EB'
SPEC_C = '#7C3AED'
METER_C = '#16A34A'
BADGE_BG = '#F8FAFC'

# ---------------- reproduce the exact graph and Fiedler vector --------------
rng = np.random.default_rng(GM.RNG_SEED)
X, y = GM.make_moons(250, noise=0.15, rng=rng)
X_test, y_test = GM.make_moons(250, noise=0.15, rng=rng)
ALL = np.concatenate([X, X_test])
yALL = np.concatenate([y, y_test])
n = len(ALL)
W = GM.build_W(ALL)
d = W.sum(axis=1)
Di = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
Ln = np.eye(n) - Di @ W @ Di
evals, evecs = np.linalg.eigh(Ln)
fiedler = evecs[:, 1]
pred = (fiedler > 0).astype(int)
acc = max(float((pred == yALL).mean()), float(((1 - pred) == yALL).mean()))

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Graph-Based Methods: One Matrix Under All of It',
        fontsize=18.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        'The Laplacian scores any labelling\'s roughness — and the whole '
        'family just minimises it with different loyalty to the labels.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the family tree =====================
ax.add_patch(FancyBboxPatch((0.7, 6.1), 6.6, 1.25,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=METER_C, linewidth=1.7))
ax.text(4.0, 7.02, 'the smoothness meter', fontsize=9, fontweight='bold',
        ha='center', va='center', color=METER_C)
ax.text(4.0, 6.55,
        r'$f^{T} L f \;=\; \frac{1}{2}\sum_{ij} w_{ij}\,(f_i - f_j)^2$',
        fontsize=13, ha='center', va='center', color=TEXT_COLOR)

ax.text(4.0, 5.55, 'minimise it, with three levels of loyalty to the labels:',
        fontsize=9, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

def branch(x, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, 3.6), 2.05, 1.5,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.7))
    ax.text(x + 1.02, 4.75, title, fontsize=9.3, fontweight='bold',
            ha='center', va='center', color=col)
    ax.text(x + 1.02, 4.12, sub, fontsize=7.6, ha='center', va='center',
            color=SUBTLE_TEXT)
    ax.add_patch(FancyArrowPatch((4.0, 5.3), (x + 1.02, 5.15),
                 arrowstyle='-|>', mutation_scale=10, color='#94A3B8',
                 linewidth=1.2))

branch(0.7, 'labels clamped', 'harmonic propagation\n(Part 5)\nμ → ∞', HARD_C)
branch(2.98, 'labels can bend', 'label spreading\n(soft clamp)\nμ finite', SOFT_C)
branch(5.26, 'no labels at all', 'the spectrum:\nFiedler vector\nμ = 0', SPEC_C)

ax.add_patch(FancyBboxPatch((0.7, 2.15), 6.6, 0.95,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=SOFT_C, linewidth=1.4))
ax.text(4.0, 2.83, 'bonus: the audit', fontsize=9, fontweight='bold',
        ha='center', va='center', color=SOFT_C)
ax.text(4.0, 2.43,
        'leave one label out, ask the graph what it expected --\n'
        'poisoned labels rank straight to the top',
        fontsize=8.3, ha='center', va='center', color=SUBTLE_TEXT)

ax.text(4.0, 1.5, 'flip 2 points of 1,000 and the meter nearly doubles '
        '(57.9 → 99.8)',
        fontsize=8.6, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: the Fiedler painting =====================
ax2 = fig.add_axes([0.575, 0.13, 0.4, 0.62])
lim = np.abs(fiedler).max()
sc = ax2.scatter(ALL[:, 0], ALL[:, 1], c=fiedler, cmap='coolwarm',
                 vmin=-lim, vmax=lim, s=13, linewidths=0)
cb = fig.colorbar(sc, ax=ax2, fraction=0.035, pad=0.02)
cb.set_label('Fiedler vector value', fontsize=8, color=SUBTLE_TEXT)
cb.set_ticks([])
cb.outline.set_edgecolor('#E2E8F0')
ax2.set_title(f'The Fiedler vector: {acc:.1%} of the classes, zero labels',
              fontsize=10.5, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.text(0.015, 0.03, 'computed from the graph alone\n(2nd eigenvector of '
         'the Laplacian)',
         transform=ax2.transAxes, fontsize=8, color=SUBTLE_TEXT,
         ha='left', va='bottom', fontstyle='italic')
ax2.set_xticks([]); ax2.set_yticks([])
ax2.margins(0.06)
for sp in ax2.spines.values():
    sp.set_color('#E2E8F0')

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 6',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/06-graph-based-methods/'
       'header_graph_methods.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}  (fiedler acc {acc:.3f})')
