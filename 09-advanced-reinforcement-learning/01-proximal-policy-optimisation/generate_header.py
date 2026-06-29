"""Generate the header image for the PPO article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
POS_COLOR = '#16A34A'    # A > 0
NEG_COLOR = '#EA580C'    # A < 0
CLIP_COLOR = '#16A34A'
NOCLIP_COLOR = '#EA580C'
BAND = '#E2E8F0'

STEPS = [1024*(i+1) for i in range(40)]
CLIP = [33, 29, 58, 78, 106, 124, 159, 191, 217, 238, 285, 319, 351, 378, 402,
        425, 446, 458, 467, 467, 475, 476, 460, 460, 419, 419, 419, 419, 419,
        430, 431, 460, 460, 479, 500, 500, 483, 483, 483, 483]
NOCLIP = [33, 77, 113, 161, 207, 202, 213, 210, 238, 271, 309, 343, 369, 407,
          22, 70, 118, 151, 208, 246, 108, 78, 38, 61, 95, 108, 81, 42, 54, 86,
          118, 149, 99, 125, 148, 182, 229, 260, 238, 258]

fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.5, 'Proximal Policy Optimisation: The Clip That Tamed Policy Gradients',
        fontsize=17.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.03,
        'Clip the objective so the new policy stays near the old — capping the step '
        'and making each rollout safe to reuse.',
        fontsize=10.3, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')
ax.text(8, 7.4,
        r'$L = \min\,(\ r\,A,\ \ \mathrm{clip}(r,\ 1-\epsilon,\ 1+\epsilon)\,A\ )$'
        r'      $r = \pi_{new}/\pi_{old}$',
        fontsize=13, ha='center', va='center', color=TEXT_COLOR)

# ===================== LEFT: the clip diagram =====================
axc = fig.add_axes([0.055, 0.13, 0.37, 0.56])
eps = 0.2
r = np.linspace(0.5, 1.5, 400)
# A > 0:  L = min(r, clip(r,1-eps,1+eps))
Lpos = np.minimum(r, np.clip(r, 1 - eps, 1 + eps))
# A < 0:  L = max(-r, -clip(r,...)) = -max(r, clip(r,...))
Lneg = -np.maximum(r, np.clip(r, 1 - eps, 1 + eps))

axc.axvspan(1 - eps, 1 + eps, color=BAND, alpha=0.7, zorder=0)
axc.axvline(1.0, color='#CBD5E1', linewidth=1.0, linestyle='--', zorder=1)
axc.plot(r, Lpos, color=POS_COLOR, linewidth=2.6, label='advantage A > 0', zorder=3)
axc.plot(r, Lneg, color=NEG_COLOR, linewidth=2.6, label='advantage A < 0', zorder=3)

axc.annotate('clipped:\nno gradient', xy=(1.38, 1.2), xytext=(1.21, 0.45),
             fontsize=8.5, color=POS_COLOR, ha='left',
             arrowprops=dict(arrowstyle='->', color=POS_COLOR, lw=1.1))
axc.annotate('clipped:\nno gradient', xy=(0.62, -0.8), xytext=(0.52, -0.35),
             fontsize=8.5, color=NEG_COLOR, ha='left',
             arrowprops=dict(arrowstyle='->', color=NEG_COLOR, lw=1.1))
axc.text(1.0, 1.62, 'trust band\n1±ε', fontsize=8.5, ha='center', va='center',
         color=SUBTLE_TEXT, fontstyle='italic')

axc.set_xlabel('ratio  r = π_new / π_old', fontsize=9.5, color=SUBTLE_TEXT)
axc.set_ylabel('clipped objective  L', fontsize=9.5, color=SUBTLE_TEXT)
axc.set_title('No incentive to move r outside the band',
              fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=8)
axc.legend(fontsize=8, loc='lower right', frameon=False)
axc.set_xlim(0.5, 1.5)
axc.set_ylim(-1.7, 1.9)
for sp in ('top', 'right'):
    axc.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axc.spines[sp].set_color('#CBD5E1')
axc.tick_params(colors=SUBTLE_TEXT, labelsize=8)

# ===================== RIGHT: the ablation curve =====================
ax2 = fig.add_axes([0.605, 0.13, 0.355, 0.56])
ax2.plot(STEPS, CLIP, color=CLIP_COLOR, linewidth=2.4, label='with clip (PPO)')
ax2.plot(STEPS, NOCLIP, color=NOCLIP_COLOR, linewidth=2.4,
         label='without clip')
ax2.axhline(500, color='#CBD5E1', linewidth=1.0, linestyle='--')
ax2.text(40960, 500, ' max 500', fontsize=7.5, color='#94A3B8',
         va='bottom', ha='right')
ax2.annotate('stable → 483', xy=(38000, 483), xytext=(20000, 430),
             fontsize=9.5, color=CLIP_COLOR, fontweight='bold')
ax2.annotate('collapses', xy=(15360, 22), xytext=(17000, 70),
             fontsize=9.5, color=NOCLIP_COLOR, fontweight='bold')
ax2.set_xlabel('environment steps', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_ylabel('mean return', fontsize=9.5, color=SUBTLE_TEXT)
ax2.set_title('10 epochs of reuse: clip vs no clip',
              fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=8)
ax2.legend(fontsize=8, loc='center right', frameon=False)
ax2.set_ylim(0, 540)
ax2.set_xlim(0, 41000)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    ax2.spines[sp].set_color('#CBD5E1')
ax2.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Advanced Reinforcement Learning Part 1',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '09-advanced-reinforcement-learning/01-proximal-policy-optimisation/'
       'header_ppo.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
