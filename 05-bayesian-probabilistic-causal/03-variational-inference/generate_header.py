"""Generate the header image for the Variational Inference article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

TRUE_COLOR = '#1F2937'
Q_FIT_COLOR = '#3B82F6'
Q_FIT_FILL = '#dbeafe'


fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45,
        'Variational Inference: Posterior Approximation as Optimisation',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Find the closest q(θ) in a tractable family to the true posterior p(θ | data). Optimise the ELBO.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

LEFT_PANEL = (0.4, 0.9, 7.6, 6.0)
RIGHT_PANEL = (8.0, 0.9, 7.6, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    ax.add_patch(FancyBboxPatch((px, py), pw, ph,
                                boxstyle='round,pad=0.02,rounding_size=0.15',
                                facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                                linewidth=1.2, zorder=0))


# ========================================================================
# LEFT PANEL: ELBO ascent curve
# ========================================================================
lpx, lpy, lpw, lph = LEFT_PANEL
ax.text(lpx + lpw/2, lpy + lph - 0.4,
        'ELBO ascent over iterations',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Synthetic ELBO curve
iters = np.arange(1, 51)
elbo = -200 + 180 * (1 - np.exp(-iters / 8))

plot_x0, plot_x1 = lpx + 1.0, lpx + lpw - 0.5
plot_y0, plot_y1 = lpy + 1.2, lpy + lph - 1.2

ax.plot([plot_x0, plot_x1], [plot_y0, plot_y0],
        color='#94A3B8', linewidth=1.0)
ax.plot([plot_x0, plot_x0], [plot_y0, plot_y1],
        color='#94A3B8', linewidth=1.0)

xs_p = plot_x0 + (iters - 1) / (50 - 1) * (plot_x1 - plot_x0)
e_min, e_max = elbo.min() - 5, elbo.max() + 5
ys_p = plot_y0 + (elbo - e_min) / (e_max - e_min) * (plot_y1 - plot_y0)
ax.plot(xs_p, ys_p, color=Q_FIT_COLOR, linewidth=2.5, zorder=3)
ax.fill_between(xs_p, plot_y0, ys_p, color=Q_FIT_FILL,
                alpha=0.4, zorder=2)

# Axis labels
ax.text((plot_x0 + plot_x1) / 2, plot_y0 - 0.35,
        'iteration', fontsize=10, fontstyle='italic',
        ha='center', va='center', color=SUBTLE_TEXT)
ax.text(plot_x0 - 0.55, (plot_y0 + plot_y1) / 2,
        'ELBO', fontsize=10, fontstyle='italic',
        ha='center', va='center', color=SUBTLE_TEXT,
        rotation=90)

ax.text(lpx + lpw/2, lpy + 0.45,
        'Maximising the ELBO = minimising KL(q ‖ p).',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# ========================================================================
# RIGHT PANEL: true vs approximate posterior contours
# ========================================================================
rpx, rpy, rpw, rph = RIGHT_PANEL
ax.text(rpx + rpw/2, rpy + rph - 0.4,
        'q(θ) approximates p(θ | data)',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Centre of contours
cx = rpx + rpw / 2
cy = rpy + rph / 2 - 0.2

# True posterior: elongated diagonal Gaussian
true_cov = np.array([[2.0, 1.2], [1.2, 1.5]])
vals, vecs = np.linalg.eigh(true_cov)
order = vals.argsort()[::-1]
vals = vals[order]
vecs = vecs[:, order]
angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
for sigma in [1, 2, 3]:
    ax.add_patch(Ellipse((cx, cy),
                          2 * sigma * np.sqrt(vals[0]),
                          2 * sigma * np.sqrt(vals[1]),
                          angle=angle,
                          facecolor='none', edgecolor=TRUE_COLOR,
                          linewidth=1.2, linestyle='--',
                          alpha=0.7, zorder=2))

# Mean-field q: axis-aligned ellipse (same marginals as projection of true)
q_var_x = true_cov[0, 0]
q_var_y = true_cov[1, 1]
for sigma in [1, 2, 3]:
    ax.add_patch(Ellipse((cx, cy),
                          2 * sigma * np.sqrt(q_var_x) * 0.7,
                          2 * sigma * np.sqrt(q_var_y) * 0.7,
                          angle=0,
                          facecolor=Q_FIT_FILL, edgecolor=Q_FIT_COLOR,
                          linewidth=1.4, alpha=0.4 if sigma == 1 else 0.0,
                          zorder=1))
    if sigma == 1:
        continue
    ax.add_patch(Ellipse((cx, cy),
                          2 * sigma * np.sqrt(q_var_x) * 0.7,
                          2 * sigma * np.sqrt(q_var_y) * 0.7,
                          angle=0,
                          facecolor='none', edgecolor=Q_FIT_COLOR,
                          linewidth=1.4, zorder=3))

# Legend
lx = rpx + 0.6
ly = rpy + 0.45
ax.plot([lx, lx + 0.5], [ly, ly], color=TRUE_COLOR,
        linewidth=1.5, linestyle='--')
ax.text(lx + 0.62, ly,
        'true posterior p(θ | data)',
        fontsize=9, ha='left', va='center', color=TEXT_COLOR)
ax.plot([lx + 4.0, lx + 4.5], [ly, ly], color=Q_FIT_COLOR,
        linewidth=2.0)
ax.text(lx + 4.62, ly,
        'mean-field q(θ)', fontsize=9,
        ha='left', va='center', color=TEXT_COLOR)


ax.text(8, 0.3,
        'Algorithms in Python  |  Bayesian, Probabilistic & Causal Methods Part 3',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '05-bayesian-probabilistic-causal/'
       '03-variational-inference/header_vi.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
