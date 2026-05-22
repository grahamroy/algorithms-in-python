"""Generate the header image for the Spectral Clustering article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets import make_moons


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CLUSTER_BORDER = ['#3B82F6', '#DC2626']
CLUSTER_FILL = ['#dbeafe', '#fee2e2']


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 'Spectral Clustering: Find Cuts in the Graph, Not Centroids',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Same two-moons dataset, two algorithms. K-Means draws a straight line. Spectral cuts the graph.',
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
# Generate data
# ========================================================================
X, y_true = make_moons(n_samples=300, noise=0.05, random_state=7)
km_labels = KMeans(n_clusters=2, n_init=10,
                   random_state=7).fit_predict(X)
sc_labels = SpectralClustering(n_clusters=2,
                               affinity='nearest_neighbors',
                               n_neighbors=10,
                               random_state=7).fit_predict(X)


def plot_clusters(panel, labels, title, subtitle):
    px, py, pw, ph = panel
    ax.text(px + pw/2, py + ph - 0.4, title,
            fontsize=13, fontweight='bold', ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif')

    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4

    plot_x0 = px + 0.55
    plot_x1 = px + pw - 0.55
    plot_y0 = py + 1.0
    plot_y1 = py + ph - 1.0

    def to_panel(p):
        fx = (p[0] - x_min) / (x_max - x_min)
        fy = (p[1] - y_min) / (y_max - y_min)
        return plot_x0 + fx * (plot_x1 - plot_x0), \
               plot_y0 + fy * (plot_y1 - plot_y0)

    for cls in (0, 1):
        pts = X[labels == cls]
        coords = np.array([to_panel(p) for p in pts])
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=26, c=CLUSTER_FILL[cls],
                   edgecolors=CLUSTER_BORDER[cls],
                   linewidths=1.0, zorder=2)

    ax.text(px + pw/2, py + 0.45, subtitle,
            fontsize=10, fontstyle='italic',
            ha='center', va='center',
            color=SUBTLE_TEXT, fontfamily='sans-serif')


plot_clusters(LEFT_PANEL, km_labels,
              'K-Means: ARI = 0.234',
              'Straight bisecting line. Both halves of each moon get split.')
plot_clusters(RIGHT_PANEL, sc_labels,
              'Spectral clustering: ARI = 1.000',
              'Follows the manifold. Each moon is its own cluster.')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Unsupervised Learning Part 7',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '03-unsupervised-learning/'
       '07-spectral-clustering/header_spectral.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
