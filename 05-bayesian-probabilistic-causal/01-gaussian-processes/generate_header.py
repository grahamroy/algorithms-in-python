"""Generate the header image for the Gaussian Processes article."""

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

TRUE_COLOR = '#1F2937'
MEAN_COLOR = '#3B82F6'
BAND_COLOR = '#dbeafe'
SAMPLE_COLOR = '#DC2626'


def true_f(x):
    return np.sin(x) * x / 2


def rbf(A, B, length_scale=1.0, sigma_f=1.0):
    sq = (A ** 2).sum(axis=1)[:, None] + \
         (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
    return (sigma_f ** 2) * np.exp(-sq / (2 * length_scale ** 2))


fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45,
        'Gaussian Processes: A Posterior Distribution Over Functions',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Predictive mean + uncertainty band. Wide where data is sparse, tight where it is dense.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# Generate training data + fit GP
rng = np.random.default_rng(7)
X_train = rng.uniform(-5, 5, size=(30, 1))
y_train = true_f(X_train.ravel()) + rng.normal(0, 0.2, 30)

# Make test grid that includes a sparse region in the middle for emphasis
X_test = np.linspace(-6, 6, 200).reshape(-1, 1)

# Compute GP posterior
K = rbf(X_train, X_train)
K_inv = np.linalg.inv(K + 0.04 * np.eye(30))
K_star = rbf(X_test, X_train)
mu = K_star @ K_inv @ y_train
K_starstar = rbf(X_test, X_test)
cov = K_starstar - K_star @ K_inv @ K_star.T
std = np.sqrt(np.maximum(np.diag(cov), 0.0))

# Plot region
plot_x0, plot_x1 = PANEL[0] + 0.8, PANEL[0] + PANEL[2] - 0.8
plot_y0, plot_y1 = PANEL[1] + 0.7, PANEL[1] + PANEL[3] - 0.9

x_data_min, x_data_max = -6, 6
y_data_min, y_data_max = -4, 4

def to_panel_x(x):
    return plot_x0 + (x - x_data_min) / (x_data_max - x_data_min) * \
           (plot_x1 - plot_x0)
def to_panel_y(y):
    return plot_y0 + (y - y_data_min) / (y_data_max - y_data_min) * \
           (plot_y1 - plot_y0)

xs_p = np.array([to_panel_x(x) for x in X_test.ravel()])
mu_p = np.array([to_panel_y(m) for m in mu])
upper_p = np.array([to_panel_y(m + 1.96 * s) for m, s in zip(mu, std)])
lower_p = np.array([to_panel_y(m - 1.96 * s) for m, s in zip(mu, std)])

# Uncertainty band
ax.fill_between(xs_p, lower_p, upper_p, color=BAND_COLOR,
                alpha=0.9, zorder=1, label='95% credible interval')

# Predictive mean
ax.plot(xs_p, mu_p, color=MEAN_COLOR, linewidth=2.5,
        zorder=3, label='predictive mean')

# True function
true_y_p = np.array([to_panel_y(y) for y in true_f(X_test.ravel())])
ax.plot(xs_p, true_y_p, color=TRUE_COLOR, linewidth=1.5,
        linestyle='--', zorder=2, label='true f(x)')

# Training points
xt_p = np.array([to_panel_x(x) for x in X_train.ravel()])
yt_p = np.array([to_panel_y(y) for y in y_train])
ax.scatter(xt_p, yt_p, s=42, c=SAMPLE_COLOR,
           edgecolors='white', linewidths=1.0,
           zorder=4, label='30 noisy observations')

# Axis labels
ax.text((plot_x0 + plot_x1) / 2, plot_y0 - 0.32,
        'x',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')
ax.text(plot_x0 - 0.4,
        (plot_y0 + plot_y1) / 2, 'f(x)',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif',
        rotation=90)

# Legend (manual)
lx = PANEL[0] + 0.6
ly = PANEL[1] + PANEL[3] - 0.5
ax.plot([lx, lx + 0.4], [ly, ly], color=TRUE_COLOR,
        linewidth=1.5, linestyle='--')
ax.text(lx + 0.5, ly, 'true f(x)', fontsize=9,
        ha='left', va='center', color=TEXT_COLOR)
ax.plot([lx + 2.0, lx + 2.4], [ly, ly], color=MEAN_COLOR,
        linewidth=2.5)
ax.text(lx + 2.5, ly, 'predictive mean', fontsize=9,
        ha='left', va='center', color=TEXT_COLOR)
ax.scatter([lx + 4.85], [ly], s=30, c=SAMPLE_COLOR,
           edgecolors='white', linewidths=0.8)
ax.text(lx + 5.0, ly, 'observations', fontsize=9,
        ha='left', va='center', color=TEXT_COLOR)
ax.add_patch(plt.Rectangle((lx + 6.5, ly - 0.07),
                            0.4, 0.14,
                            facecolor=BAND_COLOR,
                            edgecolor=MEAN_COLOR,
                            linewidth=0.6))
ax.text(lx + 7.0, ly, '95% credible interval', fontsize=9,
        ha='left', va='center', color=TEXT_COLOR)

ax.text(8, 0.3,
        'Algorithms in Python  |  Bayesian, Probabilistic & Causal Methods Part 1',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '05-bayesian-probabilistic-causal/'
       '01-gaussian-processes/header_gp.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
