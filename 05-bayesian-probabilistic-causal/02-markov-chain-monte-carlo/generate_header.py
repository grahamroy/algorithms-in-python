"""Generate the header image for the MCMC article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CHAIN_COLORS = ['#3B82F6', '#DC2626', '#16A34A', '#F59E0B']


def log_target(theta):
    x, y = theta[..., 0], theta[..., 1]
    return -(1 - x) ** 2 / 0.5 - (y - x ** 2) ** 2 / 0.5


def mh(init, n_samples, sigma, seed):
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_samples, 2))
    theta = np.array(init, dtype=float)
    lp = log_target(theta)
    for t in range(n_samples):
        prop = theta + rng.normal(0, sigma, size=2)
        lp_new = log_target(prop)
        if np.log(rng.random()) < lp_new - lp:
            theta = prop
            lp = lp_new
        samples[t] = theta
    return samples


fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45,
        'MCMC: Sample from Posteriors that Have No Closed Form',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        '4 chains × 10000 Metropolis-Hastings iterations exploring a 2-D banana-shaped posterior.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# Plot region
plot_x0 = PANEL[0] + 0.8
plot_x1 = PANEL[0] + PANEL[2] - 0.8
plot_y0 = PANEL[1] + 0.7
plot_y1 = PANEL[1] + PANEL[3] - 0.9

x_min, x_max = -2.5, 3.5
y_min, y_max = -2, 9


def to_panel(p):
    fx = (p[0] - x_min) / (x_max - x_min)
    fy = (p[1] - y_min) / (y_max - y_min)
    return plot_x0 + fx * (plot_x1 - plot_x0), \
           plot_y0 + fy * (plot_y1 - plot_y0)


# Background: contour-like density via fine grid
xs = np.linspace(x_min, x_max, 200)
ys = np.linspace(y_min, y_max, 200)
XX, YY = np.meshgrid(xs, ys)
grid = np.stack([XX, YY], axis=-1)
lp = log_target(grid)
density = np.exp(lp - lp.max())

# Map grid to panel coordinates and imshow
extent = (
    to_panel((x_min, 0))[0], to_panel((x_max, 0))[0],
    to_panel((0, y_min))[1], to_panel((0, y_max))[1],
)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "post", ['#FFFFFF', '#F1F5F9', '#CBD5E1'],
)
ax.imshow(density, extent=extent, origin='lower',
          cmap=cmap, alpha=0.9, aspect='auto', zorder=1)

# Run 4 chains and plot their first 2000 iterations
inits = [(0, 0), (-1, 1), (2, 2), (-2, 4)]
for i, init in enumerate(inits):
    samples = mh(init, n_samples=2000, sigma=0.5, seed=7 + i)
    pts = np.array([to_panel(p) for p in samples])
    ax.scatter(pts[:, 0], pts[:, 1],
               s=4, c=CHAIN_COLORS[i],
               alpha=0.5, edgecolors='none', zorder=2)
    # Highlight start point
    s0 = to_panel(init)
    ax.scatter([s0[0]], [s0[1]],
               s=90, c=CHAIN_COLORS[i],
               edgecolors='white', linewidths=1.5,
               marker='o', zorder=4)

# Legend
lx = PANEL[0] + 0.6
ly = PANEL[1] + 0.45
for i in range(4):
    sx = lx + i * 2.0
    ax.scatter([sx], [ly], s=40, c=CHAIN_COLORS[i],
               alpha=0.7, edgecolors='none')
    ax.text(sx + 0.15, ly, f'chain {i + 1}',
            fontsize=10, ha='left', va='center',
            color=CHAIN_COLORS[i], fontweight='bold')

# Axis hints
ax.text((plot_x0 + plot_x1) / 2,
        plot_y0 - 0.35, 'θ₁',
        fontsize=11, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')
ax.text(plot_x0 - 0.4,
        (plot_y0 + plot_y1) / 2, 'θ₂',
        fontsize=11, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, rotation=90, fontfamily='sans-serif')

ax.text(8, 0.3,
        'Algorithms in Python  |  Bayesian, Probabilistic & Causal Methods Part 2',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '05-bayesian-probabilistic-causal/'
       '02-markov-chain-monte-carlo/header_mcmc.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
