"""Generate the header image for the VAE article.

Right panels are REAL: the spiral samples are the actual DEMO-1 VAE
from vae.py retrained live (same seeds), the ring panel is the
demo's seed-0 run, and the KL ledger is the DEMO-3 16-dim model's.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

import vae as V

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- retrain the exact DEMO-1 VAE ----------------------------------------
v = V.VAE(seed=0)
rng = np.random.default_rng(1)
for _ in range(6000):
    v.step(V.spiral_data(128, rng), rng)
Xfake = v.sample(2000, np.random.default_rng(99))
q, c = V.spiral_metrics(Xfake)
Xreal = V.spiral_data(2000, np.random.default_rng(97))
print(f'VAE quality {q} ({round(q*2000)}/2000) coverage {c}')
assert f'{q:.1%}' == '89.5%' and f'{c:.1%}' == '98.3%'

# ---- the DEMO-1 ring run, seed 0 -----------------------------------------
vr = V.VAE(seed=0)
r2 = np.random.default_rng(10)
for _ in range(4000):
    vr.step(V.make_ring(64, r2), r2)
Xring = vr.sample(2000, np.random.default_rng(99))
m, hq, _ = V.ring_metrics(Xring)
print(f'ring: modes {m}/8 on-mode {hq:.0%}')
assert m == 8 and f'{hq:.0%}' == '39%'

# ---- the DEMO-3 dimension ledger -----------------------------------------
v16 = V.VAE(seed=0, d_z=16)
rng = np.random.default_rng(1)
for _ in range(6000):
    v16.step(V.spiral_data(128, rng), rng)
Xev = V.spiral_data(2000, np.random.default_rng(96))
kd = np.sort(v16.perdim_kl(Xev))[::-1]
alive = int((kd > 0.1).sum())
print('per-dim KL:', [f'{x:.2f}' for x in kd])
assert alive == 2 and f'{kd[0]:.2f}' == '1.67' and f'{kd[1]:.2f}' == '1.52'

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Variational Autoencoders: The Loss That Counts Its Own '
        'Dimensions',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'An honest loss \u2014 reconstruction plus a compression bill '
        '\u2014 that descends where the GAN orbited, covers every mode, '
        'and audits its own latent space.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.1, 5.3, 1.55, BLUE)
ax.text(3.35, 7.38, 'the honest loss', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=BLUE)
ax.text(3.35, 6.98, 'recon/(2*sig^2)  +  beta * KL(q(z|x) || N(0,I))',
        fontsize=8.6, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.5, 'the KL prices every nat the latent carries;\n'
                   'z = mu + sigma*eps lets the gradient through',
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.45, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.55, 'the trade vs Part 7, measured', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.98, 'GAN sharper : quality 98.7% vs 89.5%\n'
                    'VAE covers  : arc 98.3% vs 81.7%,\n'
                    'ring modes 8/8 every seed (GAN kept 3/8)',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.75, 5.3, 1.25, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.75, 'posterior collapse', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 3.25, 'beta 64: KL 0.001 -- the latent goes silent,\n'
                    'every z decodes to the mean. coverage 11.7%.',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.15, 'the dimension ledger', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 1.65, '16 latent dims offered for 2-d data:\n'
                    'the ELBO pays for exactly 2, returns 14 at 0.00',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the samples =====================
axg = fig.add_axes([0.455, 0.475, 0.30, 0.335])
axg.scatter(Xreal[:, 0], Xreal[:, 1], s=5, color='#64748B',
            alpha=0.7, linewidths=0, zorder=1)
axg.scatter(Xfake[:, 0], Xfake[:, 1], s=2.5, color=ORANGE,
            alpha=0.4, linewidths=0, zorder=2)
axg.set_xlim(-1.25, 1.25)
axg.set_ylim(-1.25, 1.25)
axg.set_aspect('equal')
axg.set_xticks([]); axg.set_yticks([])
for sp in axg.spines.values():
    sp.set_color('#CBD5E1')
axg.set_title('2,000 VAE draws from the prior (real runs)',
              fontsize=9.3, fontweight='bold', color=TEXT_COLOR, pad=5)
ax.text(12.35, 6.0, 'grey = 2,000 real points.\norange = 2,000 decoded\n'
        'prior samples.\n\nquality 89.5%, coverage\n98.3% -- softer than\n'
        'the GAN, but the faint\ntails are all there, and\nthe eval ELBO '
        'fell at\nevery checkpoint.',
        fontsize=8.4, ha='left', va='center', color=SUBTLE_TEXT)

# ===================== RIGHT BOTTOM LEFT: the ring =====================
axr = fig.add_axes([0.455, 0.095, 0.245, 0.30])
for cx, cy in V.ring_centers():
    axr.add_patch(Circle((cx, cy), 3 * V.RING_SD, facecolor='none',
                         edgecolor='#94A3B8', linewidth=1.0,
                         linestyle=':', zorder=1))
axr.scatter(Xring[:, 0], Xring[:, 1], s=4, color=BLUE, alpha=0.45,
            linewidths=0, zorder=2)
axr.set_xlim(-1.55, 1.55)
axr.set_ylim(-1.45, 1.65)
axr.set_aspect('equal')
axr.set_xticks([]); axr.set_yticks([])
for sp in axr.spines.values():
    sp.set_color('#CBD5E1')
axr.set_title(f'mode-covering: {m}/8, blur between',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=5)

# ===================== RIGHT BOTTOM RIGHT: the ledger =====================
axl = fig.add_axes([0.725, 0.095, 0.235, 0.30])
cols = [GREEN if x > 0.1 else '#CBD5E1' for x in kd]
axl.bar(np.arange(16), kd, color=cols, width=0.75)
axl.set_ylim(0, 1.9)
axl.set_xticks([0, 5, 10, 15])
axl.set_xticklabels(['1', '6', '11', '16'], fontsize=7.5)
axl.set_xlabel('latent dimension (sorted)', fontsize=8,
               color=SUBTLE_TEXT)
axl.set_ylabel('KL (nats)', fontsize=8, color=SUBTLE_TEXT)
axl.text(8.5, 1.0, 'paid: 2\nreturned: 14', fontsize=8.4,
         color=TEXT_COLOR, ha='center')
for sp in ('top', 'right'):
    axl.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axl.spines[sp].set_color('#CBD5E1')
axl.tick_params(colors=SUBTLE_TEXT, labelsize=7.5)
axl.set_title('the ledger: 16 dims, 2 in use',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=5)

ax.text(8, 0.18,
        'Algorithms in Python  |  Deep Learning Architectures Part 8',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '11-deep-learning-architectures/08-variational-autoencoders/'
       'header_vae.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
