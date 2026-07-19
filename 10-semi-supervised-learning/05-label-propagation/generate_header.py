"""Generate the header image for the Label Propagation article.

The right panel is the REAL flood: every point of the actual graph coloured
by the iteration at which label mass first reached it.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import label_propagation as LP


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
FLOW_COLOR = '#2563EB'
CLAMP_COLOR = '#EA580C'
HARM_COLOR = '#16A34A'
BADGE_BG = '#F8FAFC'

# ---------------- reproduce the exact DEMO 1 run, tracking first-reach ------
rng = np.random.default_rng(LP.RNG_SEED)
X, y = LP.make_moons(250, noise=0.15, rng=rng)
X_test, y_test = LP.make_moons(250, noise=0.15, rng=rng)
ALL = np.concatenate([X, X_test])
d_rng = np.random.default_rng(0)
li = np.concatenate([d_rng.choice(np.where(y == 0)[0], 4, replace=False),
                     d_rng.choice(np.where(y == 1)[0], 4, replace=False)])
P = LP.build_transition(ALL)

n = len(ALL)
F = np.zeros((n, 2))
F[li] = np.eye(2)[y[li]]
first_reach = np.full(n, np.inf)
first_reach[li] = 0
for it in range(1, 26):
    F = P @ F
    F[li] = np.eye(2)[y[li]]
    newly = (F.sum(axis=1) > 1e-12) & np.isinf(first_reach)
    first_reach[newly] = it
first_reach[np.isinf(first_reach)] = 25

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Label Propagation: Let the Labels Flow',
        fontsize=19, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        'Build a similarity graph over every point, pour the known labels in, '
        'and let them diffuse — the gap has no edges to cross.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the recipe =====================
def step_box(x, y0, num, text, col):
    ax.add_patch(FancyBboxPatch((x, y0), 6.6, 0.82,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.6))
    ax.text(x + 0.32, y0 + 0.41, num, fontsize=12.5, fontweight='bold',
            ha='center', va='center', color=col)
    ax.text(x + 0.62, y0 + 0.41, text, fontsize=9, ha='left', va='center',
            color=TEXT_COLOR)

step_box(0.55, 6.35, '1',
         'connect every point to its k nearest neighbours\n'
         '(Gaussian edge weights -- closer pulls harder)', FLOW_COLOR)
step_box(0.55, 5.25, '2',
         'flow:  F <- P F   (each point averages its neighbours)',
         FLOW_COLOR)
step_box(0.55, 4.15, '3',
         'clamp: the labelled points never change', CLAMP_COLOR)

ax.add_patch(FancyArrowPatch((2.2, 4.1), (2.2, 3.6), arrowstyle='-|>',
             mutation_scale=12, color='#94A3B8', linewidth=1.5))
ax.text(2.45, 3.85, 'repeat to equilibrium', fontsize=8, ha='left',
        va='center', color=SUBTLE_TEXT, fontstyle='italic')

ax.add_patch(FancyBboxPatch((0.55, 2.35), 6.6, 1.15,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=HARM_COLOR, linewidth=1.6))
ax.text(3.85, 3.22, 'the equilibrium is harmonic (Zhu & Ghahramani, 2002)',
        fontsize=9, fontweight='bold', ha='center', va='center',
        color=HARM_COLOR)
ax.text(3.85, 2.72,
        'every point = the weighted average of its neighbours\n'
        '= where a random walk from it first hits a label',
        fontsize=8.8, ha='center', va='center', color=TEXT_COLOR)

ax.text(3.85, 1.7,
        'no parameters, no training -- one linear solve gives the answer',
        fontsize=8.8, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: the real flood map =====================
ax2 = fig.add_axes([0.575, 0.13, 0.4, 0.62])
sc = ax2.scatter(ALL[:, 0], ALL[:, 1], c=first_reach, cmap='viridis_r',
                 s=13, linewidths=0)
star_colors = ['#EA580C' if c == 0 else '#2563EB' for c in y[li]]
ax2.scatter(ALL[li, 0], ALL[li, 1], s=230, marker='*', c=star_colors,
            edgecolor='black', linewidths=1.1, zorder=5)
cb = fig.colorbar(sc, ax=ax2, fraction=0.035, pad=0.02)
cb.set_label('iteration first reached', fontsize=8, color=SUBTLE_TEXT)
cb.ax.tick_params(labelsize=7, colors=SUBTLE_TEXT)
cb.outline.set_edgecolor('#E2E8F0')
ax2.set_title('The real flood: 8 labels reach all 1,000 points in 20 hops',
              fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.text(0.015, 0.97, 'stars = the 8 labels\nbrighter = reached earlier',
         transform=ax2.transAxes, fontsize=8, color=SUBTLE_TEXT,
         ha='left', va='top', fontstyle='italic')
ax2.margins(0.07)
ax2.set_xticks([]); ax2.set_yticks([])
for sp in ax2.spines.values():
    sp.set_color('#E2E8F0')

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 5',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/05-label-propagation/header_label_prop.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
