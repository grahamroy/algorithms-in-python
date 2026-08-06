"""Generate the header image for the GNN article.

Right panels are REAL: the actual SBM graph from gnn.py (nodes, edges,
labelled stars) and the over-smoothing curves recomputed live.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection
import numpy as np

import gnn as G

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'
COM_COLS = [ORANGE, BLUE, GREEN]

# ---- rebuild the exact stage ---------------------------------------------
rng = np.random.default_rng(0)
A, X, y, lab = G.make_sbm(rng)
Ah = G.norm_adj(A)

# layout: communities at triangle poles + jitter
lay_rng = np.random.default_rng(11)
poles = np.array([[0, 1.0], [-0.95, -0.55], [0.95, -0.55]])
pos = poles[y] + 0.34 * lay_rng.standard_normal((len(y), 2))

# over-smoothing curves, recomputed live
H = X.copy()
probe_L, probe_acc = [], []
for L in range(0, 33):
    if L in (0, 1, 2, 4, 8, 16, 32):
        m = G.fit(G.MLPBase(G.D_FEAT, 16, G.K_COM, seed=0), H, y, lab)
        probe_L.append(L)
        probe_acc.append(100 * G.accuracy(m, H, y, lab))
    H = Ah @ H
depths = [2, 4, 8, 16]
plain, resid = [], []
for L in depths:
    plain.append(100 * G.accuracy(
        G.fit(G.GCN(Ah, G.D_FEAT, 16, G.K_COM, layers=L, seed=0),
              X, y, lab), X, y, lab))
    resid.append(100 * G.accuracy(
        G.fit(G.GCN(Ah, G.D_FEAT, 16, G.K_COM, layers=L, seed=0,
                    use_resid=True), X, y, lab), X, y, lab))
print('probe:', [f'{a:.0f}' for a in probe_acc])
print('plain:', plain, 'resid:', resid)
assert abs(plain[2] - 100 * 1 / 3) < 1.0 and resid[2] > 85

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Graph Neural Networks: The Graph Decides Who May Look at Whom',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Message passing on nodes and edges — worth +27 points when the '
        'structure is right, poison when wrong, self-erasing when overdone.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.1, 5.3, 1.55, BLUE)
ax.text(3.35, 7.38, 'message passing', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=BLUE)
ax.text(3.35, 6.98, "H' = relu( A_hat H W )", fontsize=9.5, ha='center',
        va='center', color=TEXT_COLOR, fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.5, 'every node averages its neighbours, then\n'
                   'thinks -- one layer per hop of structure',
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.45, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.55, 'the family tree, in one breath', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.98, 'grid graph      ->  a CNN (Part 2)\n'
                    'complete graph  ->  a transformer (Part 4)\n'
                    'any graph       ->  message passing',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.75, 5.3, 1.25, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.75, 'the depth lesson, inverted', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 3.25, 'averaging is a random walk, and walks forget:\n'
                    '8 plain layers = chance. residuals: 87% at 16.',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.15, 'an honest defeat', fontsize=9.6, fontweight='bold',
        ha='center', va='center', color=GREEN)
ax.text(3.35, 1.65, 'GAT (learned edge attention) loses to the\n'
                    "GCN's fixed average at every label budget here",
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the graph =====================
axg = fig.add_axes([0.455, 0.475, 0.30, 0.335])
segs, cols = [], []
ii, jj = np.where(np.triu(A, 1) > 0)
for i, j in zip(ii, jj):
    segs.append([pos[i], pos[j]])
    cols.append('#DC2626' if y[i] != y[j] else '#94A3B8')
lc = LineCollection(segs, colors=cols, linewidths=0.35,
                    alpha=0.25, zorder=1)
axg.add_collection(lc)
for c in range(3):
    m = (y == c) & ~lab
    axg.scatter(pos[m, 0], pos[m, 1], s=10, color=COM_COLS[c],
                alpha=0.85, linewidths=0, zorder=2)
axg.scatter(pos[lab, 0], pos[lab, 1], s=120, marker='*',
            c=[COM_COLS[c] for c in y[lab]], edgecolors='black',
            linewidths=0.8, zorder=3)
axg.set_xlim(-1.75, 1.75)
axg.set_ylim(-1.45, 1.75)
axg.set_xticks([]); axg.set_yticks([])
for sp in axg.spines.values():
    sp.set_color('#CBD5E1')
axg.set_title('the stage: 360 nodes, 2,144 edges, 15 stars',
              fontsize=9.3, fontweight='bold', color=TEXT_COLOR, pad=5)
n_intra = int((y[ii] == y[jj]).sum())
n_inter = int((y[ii] != y[jj]).sum())
ax.text(12.35, 6.0, f'grey = within a community\n({n_intra} edges);'
        f' red = across\n({n_inter}). the noise is real --\nthat is what'
        ' deep layers\nwill blend together.\n\nfeatures alone: 58.3%.\n'
        'with the edges: 85.8%.\nshuffled edges: 42.3%.',
        fontsize=8.4, ha='left', va='center', color=SUBTLE_TEXT)

# ===================== RIGHT BOTTOM: over-smoothing =====================
axc = fig.add_axes([0.455, 0.095, 0.505, 0.30])
axc.plot(probe_L, probe_acc, 'o-', color=SUBTLE_TEXT, linewidth=1.8,
         markersize=5, label='raw operator: probe on $\\hat{A}^L X$')
axc.plot(depths, plain, 's--', color=RED, linewidth=2.0, markersize=6,
         label='trained plain GCN')
axc.plot(depths, resid, '^-', color=GREEN, linewidth=2.0, markersize=6,
         label='trained GCN + residuals')
axc.axhline(100 / 3, color='#CBD5E1', linewidth=1, linestyle=':')
axc.text(24.5, 30.5, 'chance', fontsize=7.2, color=SUBTLE_TEXT, va='top')
axc.annotate('depth erases', xy=(8, plain[2]), xytext=(10.5, 48),
             fontsize=8.4, color=RED, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.1))
axc.annotate('residuals remember', xy=(16, resid[3]), xytext=(6.5, 64),
             fontsize=8.4, color=GREEN, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.1))
axc.set_xscale('symlog', base=2, linthresh=1)
axc.set_xticks([0, 1, 2, 4, 8, 16, 32])
axc.set_xticklabels(['0', '1', '2', '4', '8', '16', '32'])
axc.minorticks_off()
axc.set_xlabel('hops / layers', fontsize=8.5, color=SUBTLE_TEXT)
axc.set_ylabel('accuracy (%)', fontsize=8.5, color=SUBTLE_TEXT)
axc.set_ylim(25, 100)
axc.legend(fontsize=7.6, loc='lower left', frameon=False)
axc.set_title('over-smoothing, measured (real runs)', fontsize=9.4,
              fontweight='bold', color=TEXT_COLOR, pad=5)
for sp in ('top', 'right'):
    axc.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axc.spines[sp].set_color('#CBD5E1')
axc.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Deep Learning Architectures Part 6',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '11-deep-learning-architectures/06-graph-neural-networks/'
       'header_gnn.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
