"""Generate the header image for the DBSCAN article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CLUSTER_BORDER = ['#3B82F6', '#DC2626']
CLUSTER_FILL = ['#dbeafe', '#fee2e2']
NOISE_FILL = '#E5E7EB'
NOISE_BORDER = '#6B7280'


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, "DBSCAN: Density Tells You Where the Clusters Are",
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Moons + injected noise. K-Means slices through both. DBSCAN finds two clusters and flags the strays.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# --- Two side-by-side panels --------------------------------------------
LEFT_PANEL = (0.4, 0.9, 7.6, 6.0)
RIGHT_PANEL = (8.0, 0.9, 7.6, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle='round,pad=0.02,rounding_size=0.15',
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)


# ========================================================================
# Generate data: two moons + uniform random noise
# ========================================================================
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=7)
rng = np.random.default_rng(7)
x_min, x_max = X_moons[:, 0].min() - 0.3, X_moons[:, 0].max() + 0.3
y_min, y_max = X_moons[:, 1].min() - 0.3, X_moons[:, 1].max() + 0.3
noise = rng.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(20, 2))
X = np.vstack([X_moons, noise])

km_labels = KMeans(n_clusters=2, n_init=10,
                   random_state=7).fit_predict(X)
db_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)


def plot_clusters(panel, labels, title, subtitle, allow_noise=True):
    px, py, pw, ph = panel
    ax.text(px + pw/2, py + ph - 0.4, title,
            fontsize=13, fontweight='bold', ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif')

    plot_x0 = px + 0.55
    plot_x1 = px + pw - 0.55
    plot_y0 = py + 1.0
    plot_y1 = py + ph - 1.0

    def to_panel(p):
        fx = (p[0] - x_min) / (x_max - x_min)
        fy = (p[1] - y_min) / (y_max - y_min)
        return plot_x0 + fx * (plot_x1 - plot_x0), \
               plot_y0 + fy * (plot_y1 - plot_y0)

    # Noise points first (smaller, grey)
    if allow_noise:
        noise_mask = labels == -1
        for p in X[noise_mask]:
            cx, cy = to_panel(p)
            ax.scatter([cx], [cy], s=44, marker='x',
                       c=NOISE_BORDER, linewidths=1.5, zorder=3)

    # Clustered points
    cluster_ids = sorted(set(int(l) for l in labels if l >= 0))
    for i, cls in enumerate(cluster_ids):
        col = i % len(CLUSTER_BORDER)
        pts = X[labels == cls]
        coords = np.array([to_panel(p) for p in pts])
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=26, c=CLUSTER_FILL[col],
                   edgecolors=CLUSTER_BORDER[col],
                   linewidths=1.0, zorder=2)

    ax.text(px + pw/2, py + 0.45, subtitle,
            fontsize=10, fontstyle='italic',
            ha='center', va='center',
            color=SUBTLE_TEXT, fontfamily='sans-serif')


plot_clusters(LEFT_PANEL, km_labels,
              'K-Means: ARI = 0.205, 0 noise points',
              'Straight bisector. Every point forced into a cluster.',
              allow_noise=False)
plot_clusters(RIGHT_PANEL, db_labels,
              'DBSCAN: ARI = 0.961, 14 noise points (X)',
              'Each moon is its own cluster. Strays flagged as noise.')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Advanced Unsupervised Learning Part 1',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '04-advanced-unsupervised-learning/'
       '01-dbscan/header_dbscan.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
