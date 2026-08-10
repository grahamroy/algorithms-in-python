"""Generate the header image for the normalising-flows article.

Right panels are REAL: the spiral flow and ring flow from flows.py
are retrained live (same seeds); the ring panel shows the model's
own exact log-density -- the filament bridges are the topology tax.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

import flows as FL

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- retrain the DEMO-1 spiral flow --------------------------------------
f = FL.Flow(n_layers=8, hidden=64, seed=0)
rng = np.random.default_rng(1)
for _ in range(8000):
    f.step(FL.spiral_data(128, rng))
Xs = f.sample(2000, np.random.default_rng(99))
q, c = FL.spiral_metrics(Xs)
print(f'flow q {q} ({round(q*2000)}/2000)  cov {c}')
assert f'{q:.1%}' == '96.7%' and f'{c:.1%}' == '98.3%'

g = np.linspace(-2.5, 2.5, 500)
GX, GY = np.meshgrid(g, g)
P = np.stack([GX.ravel(), GY.ravel()], 1)
integral = np.exp(f.logp(P)).sum() * (g[1] - g[0]) ** 2
print(f'integral {integral:.4f}')
assert f'{integral:.4f}' == '1.0000'

Xreal = FL.spiral_data(2000, np.random.default_rng(97))

# ---- retrain the DEMO-2 ring flow, seed 0 --------------------------------
fr = FL.Flow(n_layers=12, hidden=64, seed=0, lr=3e-4)
rng = np.random.default_rng(10)
for _ in range(16000):
    fr.step(FL.make_ring(256, rng))
Xr = fr.sample(2000, np.random.default_rng(99))
m, hq, _ = FL.ring_metrics(Xr)
print(f'ring: {m}/8  {hq:.0%}')
assert m == 8 and f'{hq:.0%}' == '45%'

# the model's own exact density on a grid
gr = np.linspace(-1.6, 1.6, 300)
RX, RY = np.meshgrid(gr, gr)
PR = np.stack([RX.ravel(), RY.ravel()], 1)
LP = fr.logp(PR).reshape(300, 300)

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Normalising Flows: The Model That Prices Every Point in '
        'the Plane',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Invertible steps, exact likelihood \u2014 a density that '
        'integrates to 1.0000, and a map that can bend but never tear.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.1, 5.3, 1.55, BLUE)
ax.text(3.35, 7.38, 'the exact number', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=BLUE)
ax.text(3.35, 6.98, 'log p(x) = log N(f(x)) + log|det df/dx|',
        fontsize=8.8, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.5, 'invertible couplings only -- the training\n'
                   'loss IS the likelihood, no bound, no game',
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.45, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.55, 'the receipts', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.98, 'integral of p over the plane : 1.0000\n'
                    'round trip x -> z -> x       : 3.9e-15\n'
                    'log p two units off-manifold : -147701',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.75, 5.3, 1.25, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.75, 'the topology tax', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 3.25, 'a bijection cannot tear noise into islands:\n'
                    'the gap between modes costs 1.11; truth 9.5e-13',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.15, 'the quartet, closed', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 1.65, 'four architectures, four homes for difficulty:\n'
                    'opponent  |  encoder  |  schedule  |  Jacobian',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the samples =====================
axg = fig.add_axes([0.455, 0.475, 0.30, 0.335])
axg.scatter(Xreal[:, 0], Xreal[:, 1], s=5, color='#64748B',
            alpha=0.7, linewidths=0, zorder=1)
axg.scatter(Xs[:, 0], Xs[:, 1], s=2.5, color=ORANGE,
            alpha=0.5, linewidths=0, zorder=2)
axg.set_xlim(-1.25, 1.25)
axg.set_ylim(-1.25, 1.25)
axg.set_aspect('equal')
axg.set_xticks([]); axg.set_yticks([])
for sp in axg.spines.values():
    sp.set_color('#CBD5E1')
axg.set_title('2,000 draws, one inverse pass each (real runs)',
              fontsize=9.3, fontweight='bold', color=TEXT_COLOR, pad=5)
ax.text(12.35, 6.0, 'grey = 2,000 real points.\norange = 2,000 samples --\n'
        'noise through the inverted\ncouplings, one pass.\n\n'
        'quality 96.7%, coverage\n98.3% -- and unlike every\n'
        'other row in the quartet,\nthis model can price any\n'
        'point exactly.',
        fontsize=8.4, ha='left', va='center', color=SUBTLE_TEXT)

# ========== RIGHT BOTTOM LEFT: ring samples (the filaments) ==========
axr = fig.add_axes([0.455, 0.095, 0.245, 0.30])
for cx, cy in FL.ring_centers():
    axr.add_patch(Circle((cx, cy), 3 * FL.RING_SD, facecolor='none',
                         edgecolor='#94A3B8', linewidth=1.0,
                         linestyle=':', zorder=2))
axr.scatter(Xr[:, 0], Xr[:, 1], s=4, color=RED, alpha=0.45,
            linewidths=0, zorder=1)
axr.set_xlim(-1.55, 1.55)
axr.set_ylim(-1.45, 1.65)
axr.set_aspect('equal')
axr.set_xticks([]); axr.set_yticks([])
for sp in axr.spines.values():
    sp.set_color('#CBD5E1')
axr.set_title(f'ring samples: {m}/8 modes, filaments',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=5)

# ========== RIGHT BOTTOM RIGHT: the model's own density ==========
axd = fig.add_axes([0.725, 0.095, 0.235, 0.30])
axd.imshow(np.exp(np.clip(LP, -30, 10)), origin='lower',
           extent=[-1.6, 1.6, -1.6, 1.6], cmap='inferno', aspect='equal',
           vmax=np.exp(0.5))
axd.set_xticks([]); axd.set_yticks([])
for sp in axd.spines.values():
    sp.set_color('#CBD5E1')
axd.set_title('its own exact density: the tax, visible',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=5)

ax.text(8, 0.18,
        'Algorithms in Python  |  Deep Learning Architectures Part 10',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '11-deep-learning-architectures/10-normalising-flows/'
       'header_flows.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
