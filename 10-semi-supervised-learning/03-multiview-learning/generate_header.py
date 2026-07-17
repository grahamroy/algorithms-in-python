"""Generate the header image for the Multiview Learning article.

The right panels are the REAL experiment: the same test points from
multiview.py, plotted in PCA space and in CCA space.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import multiview as MV


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
A_COLOR = '#EA580C'
B_COLOR = '#2563EB'
SHARED_C = '#16A34A'
BADGE_BG = '#F8FAFC'
C0, C1 = '#EA580C', '#2563EB'

# ---------------- reproduce the exact seed-0 experiment ----------------
A, B, y, z, tr, te, lab = MV.experiment(MV.RNG_SEED)
P = MV.pca(A[tr], 2)
Wa, Wb, S = MV.cca(A[tr], B[tr], 2)
Xp = A[te] @ P          # PCA space (view A)
Xc = A[te] @ Wa         # CCA space (view A)
yt = y[te]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Multiview Learning: What Two Views Agree On Is Probably Real',
        fontsize=17.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        'Nuisance is view-specific; signal is shared. CCA finds the shared '
        'directions — from unlabelled pairs alone.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT: the agreement principle =====================
def box(x, y0, w, h, title, sub, col):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor='white', edgecolor=col, linewidth=1.8))
    ax.text(x + w/2, y0 + h*0.62, title, fontsize=10.5, fontweight='bold',
            ha='center', va='center', color=col)
    ax.text(x + w/2, y0 + h*0.25, sub, fontsize=7.8, ha='center', va='center',
            color=SUBTLE_TEXT)

box(0.6, 5.7, 2.75, 1.15, 'VIEW A', 'signal + loud\nnuisance A', A_COLOR)
box(4.15, 5.7, 2.75, 1.15, 'VIEW B', 'signal + loud\nnuisance B', B_COLOR)
box(2.05, 3.4, 3.5, 1.05, 'SHARED DIRECTIONS',
    'what correlates across views', SHARED_C)

ax.add_patch(FancyArrowPatch((1.95, 5.65), (3.1, 4.5), arrowstyle='-|>',
             mutation_scale=12, color=A_COLOR, linewidth=1.6))
ax.add_patch(FancyArrowPatch((5.55, 5.65), (4.5, 4.5), arrowstyle='-|>',
             mutation_scale=12, color=B_COLOR, linewidth=1.6))
ax.text(1.85, 5.0, 'project\nw_A', fontsize=7.8, ha='center', va='center',
        color=A_COLOR)
ax.text(5.7, 5.0, 'project\nw_B', fontsize=7.8, ha='center', va='center',
        color=B_COLOR)

ax.add_patch(FancyBboxPatch((0.6, 1.9), 6.3, 1.0,
             boxstyle='round,pad=0.05,rounding_size=0.1',
             facecolor=BADGE_BG, edgecolor=SHARED_C, linewidth=1.5))
ax.text(3.75, 2.62, 'CCA (Hotelling, 1936)', fontsize=9,
        fontweight='bold', ha='center', va='center', color=SHARED_C)
ax.text(3.75, 2.22,
        r'$\max_{w_A, w_B}\ \ \mathrm{corr}(\,A\,w_A,\ B\,w_B\,)$',
        fontsize=12, ha='center', va='center', color=TEXT_COLOR)

ax.text(3.75, 1.3, 'trained on 600 unlabelled pairs -- zero labels used',
        fontsize=8.8, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: the twin scatters (real data) =============
def scatterpanel(rect, X, title, sub, subcol):
    axp = fig.add_axes(rect)
    for cls, col in ((0, C0), (1, C1)):
        m = yt == cls
        axp.scatter(X[m, 0], X[m, 1], s=10, color=col, alpha=0.75,
                    linewidths=0)
    axp.set_title(title, fontsize=10, fontweight='bold', color=TEXT_COLOR,
                  pad=5)
    axp.text(0.5, -0.09, sub, transform=axp.transAxes, fontsize=9,
             fontweight='bold', color=subcol, ha='center', va='top')
    axp.set_xticks([]); axp.set_yticks([])
    for sp in axp.spines.values():
        sp.set_color('#E2E8F0')
    return axp

scatterpanel([0.575, 0.17, 0.19, 0.52], Xp,
             'PCA: "what is loud?"', '10 labels here: 53.5%',
             '#DC2626')
scatterpanel([0.79, 0.17, 0.19, 0.52], Xc,
             'CCA: "what is shared?"', '10 labels here: 92.5%',
             SHARED_C)
ax.text(12.35, 7.1, 'the same 400 test points, coloured by their true class',
        fontsize=9, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

ax.text(8, 0.18, 'Algorithms in Python  |  Semi-Supervised Learning Part 3',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '10-semi-supervised-learning/03-multiview-learning/header_multiview.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}  (corrs {np.round(S,2)})')
