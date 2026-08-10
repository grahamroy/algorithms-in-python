"""Generate the header image for the diffusion article.

Right panels are REAL: the DEMO-1 denoiser from diffusion.py is
retrained live (same seeds) and the trajectory snapshots and final
samples come from its actual sampling walk.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import diffusion as DF

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- retrain the exact DEMO-1 denoiser -----------------------------------
d = DF.Diffusion(seed=0)
rng = np.random.default_rng(1)
for _ in range(16000):
    d.step(DF.spiral_data(128, rng), rng)

X100 = d.sample(2000, np.random.default_rng(99))
q100, c100 = DF.spiral_metrics(X100)
X20 = d.sample(2000, np.random.default_rng(99), n_steps=20)
q20, c20 = DF.spiral_metrics(X20)
print(f'100-pass q {q100} ({round(q100*2000)}/2000)  20-pass q {q20} '
      f'({round(q20*2000)}/2000) cov {c20}')
assert f'{q100:.1%}' == '94.5%' and f'{q20:.1%}' == '98.7%'
assert f'{c20:.1%}' == '100.0%'

_, snaps = d.sample(2000, np.random.default_rng(99),
                    record_at=[99, 50, 25, 0])
meds = {}
for t in (99, 50, 25, 0):
    dd = np.sqrt(((snaps[t][:, None, :] - DF.REF[None, :, :]) ** 2)
                 .sum(-1)).min(1)
    meds[t] = float(np.median(dd))
print('medians:', {t: f'{m:.3f}' for t, m in meds.items()})
assert f'{meds[99]:.3f}' == '0.259' and f'{meds[25]:.3f}' == '0.085'
assert f'{meds[0]:.3f}' == '0.047'

Xreal = DF.spiral_data(2000, np.random.default_rng(97))

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Diffusion Models: A Thousand Tiny Problems Instead of One '
        'Impossible One',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'A fixed noising encoder, one denoiser trained by plain '
        'regression \u2014 sharp as the GAN, covering like the VAE, '
        'paid for in passes.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.1, 5.3, 1.55, BLUE)
ax.text(3.35, 7.38, 'the fixed encoder', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=BLUE)
ax.text(3.35, 6.98, 'x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps',
        fontsize=9.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.5, 'no parameters -- the label eps comes free.\n'
                   'the only network is a denoiser, a regression',
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.45, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.55, 'the triptych, settled', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.98, 'quality : 98.7% = the GAN, exactly\n'
                    'coverage: 100.0% vs the GAN\'s 81.7%\n'
                    'sharp AND covering -- at 20 passes/sample',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.75, 5.3, 1.25, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.75, 'the bill of passes', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 3.25, 'quality peaks at 10-20 strided steps (DDIM),\n'
                    'then the cliff: 2 passes score below pure noise',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.15, 'what compute buys', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 1.65, 'ring: 26% -> 74% on-mode with 3x budget,\n'
                    '8/8 modes every seed, no game, no collapse',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the samples =====================
axg = fig.add_axes([0.455, 0.475, 0.30, 0.335])
axg.scatter(Xreal[:, 0], Xreal[:, 1], s=5, color='#64748B',
            alpha=0.7, linewidths=0, zorder=1)
axg.scatter(X20[:, 0], X20[:, 1], s=2.5, color=ORANGE,
            alpha=0.5, linewidths=0, zorder=2)
axg.set_xlim(-1.25, 1.25)
axg.set_ylim(-1.25, 1.25)
axg.set_aspect('equal')
axg.set_xticks([]); axg.set_yticks([])
for sp in axg.spines.values():
    sp.set_color('#CBD5E1')
axg.set_title('2,000 draws, 20 strided passes (real runs)',
              fontsize=9.3, fontweight='bold', color=TEXT_COLOR, pad=5)
ax.text(12.35, 6.0, 'grey = 2,000 real points.\norange = 2,000 samples,\n'
        'noise walked backwards\ndown the staircase.\n\n'
        'quality 98.7% -- equal to\nthe GAN -- at coverage\n'
        '100.0%. the price: twenty\npasses per sample.',
        fontsize=8.4, ha='left', va='center', color=SUBTLE_TEXT)

# ============ RIGHT BOTTOM: the denoising walk, four stages ============
for k, t in enumerate((99, 50, 25, 0)):
    axs = fig.add_axes([0.455 + k * 0.13, 0.095, 0.12, 0.30])
    X = snaps[t]
    Xp = X[:1200]
    axs.scatter(Xp[:, 0], Xp[:, 1], s=1.2, color=ORANGE, alpha=0.75,
                linewidths=0)
    axs.set_xlim(-1.5, 1.5)
    axs.set_ylim(-1.5, 1.5)
    axs.set_aspect('equal')
    axs.set_xticks([]); axs.set_yticks([])
    for sp in axs.spines.values():
        sp.set_color('#CBD5E1')
    axs.set_title(f't={t}\nmed {meds[t]:.3f}', fontsize=7.8,
                  color=TEXT_COLOR, pad=3)
ax.text(11.15, 0.55,
        'the walk: half of it changes nothing, then the shape snaps in '
        '-- coarse first, fine last',
        fontsize=8.6, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR)

ax.text(8, 0.18,
        'Algorithms in Python  |  Deep Learning Architectures Part 9',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '11-deep-learning-architectures/09-diffusion-models/'
       'header_diffusion.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
