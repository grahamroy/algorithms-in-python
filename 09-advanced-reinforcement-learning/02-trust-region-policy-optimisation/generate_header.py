"""Generate the header image for the TRPO article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
NAT_COLOR = '#7C3AED'     # natural gradient
PLAIN_COLOR = '#94A3B8'   # plain gradient
REGION_FILL = '#dbeafe'
REGION_EDGE = '#3B82F6'
CURVE_COLOR = '#16A34A'

STEPS = [2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480,
         22528, 24576, 26624, 28672, 30720, 32768, 34816, 36864, 38912, 40960,
         43008, 45056, 47104, 49152, 51200, 53248, 55296, 57344, 59392, 61440,
         63488, 65536, 67584, 69632, 71680, 73728, 75776, 77824, 79872, 81920,
         83968, 86016, 88064, 90112, 92160, 94208, 96256, 98304, 100352, 102400]
RET = [35, 32, 76, 99, 138, 169, 188, 230, 290, 345, 403, 439, 467, 479, 477,
       481, 481, 479, 479, 498, 498, 498, 500, 481, 481, 476, 476, 476, 495,
       495, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
       500, 500, 500, 500, 500, 500, 500]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Trust Region Policy Optimisation: The Exact Method PPO Approximates',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Maximise the surrogate subject to a hard KL constraint — the natural gradient '
        'takes the biggest safe step.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')
ax.text(8, 7.42,
        r'$\max_{\theta}\ \ L(\theta)\quad\mathrm{subject\ to}\quad '
        r'\mathrm{KL}(\pi_{old},\ \pi_\theta)\ \leq\ \delta$',
        fontsize=13, ha='center', va='center', color=TEXT_COLOR)

# ===================== LEFT: trust-region diagram =====================
ax.text(3.9, 6.5, 'One exact trust-region step', fontsize=11.5,
        fontweight='bold', ha='center', va='center', color=TEXT_COLOR)

cx, cy, R = 3.9, 3.55, 1.7
ax.add_patch(Circle((cx, cy), R, facecolor=REGION_FILL, edgecolor=REGION_EDGE,
                    linewidth=1.6, alpha=0.55, zorder=1))
ax.text(cx, cy + R - 0.32, 'KL ≤ δ', fontsize=10, ha='center', va='center',
        color=REGION_EDGE, fontstyle='italic', zorder=4)

# old policy at centre
ax.add_patch(Circle((cx, cy), 0.1, facecolor=TEXT_COLOR, zorder=5))
ax.text(cx - 0.15, cy - 0.42, r'$\pi_{old}$', fontsize=11, ha='center',
        va='center', color=TEXT_COLOR, zorder=5)

# plain gradient: overshoots the region
gx, gy = cx + 1.9, cy + 2.55
ax.add_patch(FancyArrowPatch((cx, cy), (gx, gy), arrowstyle='-|>',
             mutation_scale=14, color=PLAIN_COLOR, linewidth=2.0,
             linestyle='--', zorder=3))
ax.text(gx + 0.15, gy + 0.05, 'plain ∇L\novershoots', fontsize=8.5, ha='left',
        va='center', color=PLAIN_COLOR)

# natural gradient: lands on the trust-region boundary
nx, ny = cx + 1.67, cy + 0.32
ax.add_patch(FancyArrowPatch((cx, cy), (nx, ny), arrowstyle='-|>',
             mutation_scale=15, color=NAT_COLOR, linewidth=2.6, zorder=4))
ax.add_patch(Circle((nx, ny), 0.1, facecolor=NAT_COLOR, zorder=5))
ax.text(nx + 0.18, ny - 0.02, r'natural $F^{-1}g$', fontsize=9.5, ha='left',
        va='center', color=NAT_COLOR, fontweight='bold')
ax.text(nx + 0.2, ny - 0.42, 'lands on the edge', fontsize=8,
        ha='left', va='center', color=SUBTLE_TEXT, fontstyle='italic')

ax.text(3.9, 1.15,
        'conjugate gradient + Fisher-vector products  →  natural gradient',
        fontsize=8.6, ha='center', va='center', color=SUBTLE_TEXT,
        fontstyle='italic')

# ===================== RIGHT: learning curve =====================
ax2 = fig.add_axes([0.575, 0.13, 0.38, 0.56])
ax2.plot(STEPS, RET, color=CURVE_COLOR, linewidth=2.6)
ax2.axhline(500, color='#CBD5E1', linewidth=1.0, linestyle='--')
ax2.text(102400, 500, ' max 500', fontsize=7.5, color='#94A3B8',
         va='bottom', ha='right')
ax2.annotate('monotonic, collapse-proof', xy=(70000, 500), xytext=(34000, 360),
             fontsize=9.5, color=CURVE_COLOR, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=CURVE_COLOR, lw=1.1))
# KL guarantee badge
ax2.text(52000, 70, 'every update:  KL ≤ δ\n(max 0.0094, δ = 0.01)',
         fontsize=9, ha='center', va='center', color=TEXT_COLOR,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8FAFC',
                   edgecolor=REGION_EDGE, linewidth=1.3))
ax2.set_xlabel('environment steps', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_ylabel('mean return', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('A smooth, stable climb to 500',
              fontsize=11.5, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.set_ylim(0, 540)
ax2.set_xlim(0, 103000)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    ax2.spines[sp].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Advanced Reinforcement Learning Part 2',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '09-advanced-reinforcement-learning/02-trust-region-policy-optimisation/'
       'header_trpo.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
