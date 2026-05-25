"""Generate the header image for the GMM article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

CLUSTER_BORDER = ['#3B82F6', '#DC2626', '#16A34A']
CLUSTER_FILL = ['#dbeafe', '#fee2e2', '#dcfce7']


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 'Gaussian Mixture Models: Elliptical Clusters with Probabilities',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'K-Means draws spherical boundaries. GMM fits the actual ellipses with EM.',
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
# Generate same anisotropic data the script uses
# ========================================================================
def make_dataset(seed=7):
    centres = np.array([[-1.5, 0.0], [1.5, 0.0], [0.0, 2.5]])
    X, y = make_blobs(n_samples=600, centers=centres,
                      cluster_std=0.8, random_state=seed)
    transforms = [
        np.array([[3.0,  0.0], [0.0, 0.2]]),
        np.array([[0.2,  0.0], [0.0, 3.0]]),
        np.array([[2.0, -1.5], [1.5, 0.2]]),
    ]
    for k in range(3):
        mask = y == k
        Xk = X[mask] - centres[k]
        X[mask] = Xk @ transforms[k] + centres[k]
    return X, y


X, y_true = make_dataset()
km = KMeans(n_clusters=3, n_init=10, random_state=7).fit(X)
gmm = GaussianMixture(n_components=3, covariance_type='full',
                      init_params='kmeans', random_state=7).fit(X)
gmm_labels = gmm.predict(X)


def plot_panel(panel, labels, title, subtitle, draw_ellipses=False,
               ellipse_params=None):
    px, py, pw, ph = panel
    ax.text(px + pw/2, py + ph - 0.4, title,
            fontsize=13, fontweight='bold', ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif')

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    plot_x0 = px + 0.55
    plot_x1 = px + pw - 0.55
    plot_y0 = py + 1.0
    plot_y1 = py + ph - 1.0

    sx = (plot_x1 - plot_x0) / (x_max - x_min)
    sy = (plot_y1 - plot_y0) / (y_max - y_min)

    def to_panel(p):
        return (plot_x0 + (p[0] - x_min) * sx,
                plot_y0 + (p[1] - y_min) * sy)

    for cls in range(3):
        pts = X[labels == cls]
        coords = np.array([to_panel(p) for p in pts])
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=20, c=CLUSTER_FILL[cls],
                   edgecolors=CLUSTER_BORDER[cls],
                   linewidths=0.8, alpha=0.85, zorder=2)

    if draw_ellipses and ellipse_params is not None:
        for k, (mean, cov) in enumerate(ellipse_params):
            # Eigendecomposition to get axes
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]
            # Convert eigenvectors angle to panel space — careful
            # because sx != sy. Approximate by drawing in data
            # space first, then scaling. Use 2-sigma contour.
            width_data = 2 * 2 * np.sqrt(vals[0])
            height_data = 2 * 2 * np.sqrt(vals[1])
            # Angle in data coordinates
            angle_data = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            cx, cy = to_panel(mean)
            # Scale axes to panel space (with isotropic-ish data
            # the sx ~ sy assumption holds reasonably well)
            ell = Ellipse((cx, cy),
                          width_data * sx,
                          height_data * sy,
                          angle=angle_data,
                          edgecolor=CLUSTER_BORDER[k],
                          facecolor='none',
                          linewidth=1.8, linestyle='--',
                          alpha=0.9, zorder=4)
            ax.add_patch(ell)

    ax.text(px + pw/2, py + 0.45, subtitle,
            fontsize=10, fontstyle='italic',
            ha='center', va='center',
            color=SUBTLE_TEXT, fontfamily='sans-serif')


plot_panel(LEFT_PANEL, km.labels_,
           'K-Means: ARI = 0.334',
           'Spherical assumption + perpendicular bisectors = wrong cuts.',
           draw_ellipses=False)

ellipse_params = list(zip(gmm.means_, gmm.covariances_))
plot_panel(RIGHT_PANEL, gmm_labels,
           'GMM: ARI = 0.842, 2-σ ellipses shown',
           'Multivariate Gaussians fit the actual cluster shapes.',
           draw_ellipses=True, ellipse_params=ellipse_params)


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Advanced Unsupervised Learning Part 2',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '04-advanced-unsupervised-learning/'
       '02-gaussian-mixture-models/header_gmm.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
