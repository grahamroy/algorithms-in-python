"""Generate the header image for the t-SNE article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

DIGIT_COLORS = ['#3B82F6', '#DC2626', '#16A34A', '#F59E0B',
                '#7C3AED', '#0891B2', '#DB2777', '#65A30D',
                '#EA580C', '#475569']


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 't-SNE: Local Neighbourhoods at the Cost of Global Geometry',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Same 64-D digits dataset, projected to 2D two different ways. t-SNE shows the clusters PCA cannot.',
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
# Shared data
# ========================================================================
digits = load_digits()
X = digits.data
y = digits.target

pca_proj = PCA(n_components=2).fit_transform(X)
tsne_proj = TSNE(n_components=2, perplexity=30,
                 max_iter=1000, init='pca',
                 random_state=7).fit_transform(X)


def plot_proj(panel, proj, title):
    px, py, pw, ph = panel
    ax.text(px + pw/2, py + ph - 0.4, title,
            fontsize=13, fontweight='bold', ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif')

    x_min, x_max = proj[:, 0].min() - 1, proj[:, 0].max() + 1
    y_min, y_max = proj[:, 1].min() - 1, proj[:, 1].max() + 1

    plot_x0 = px + 0.55
    plot_x1 = px + pw - 0.55
    plot_y0 = py + 0.85
    plot_y1 = py + ph - 1.0

    def to_panel(p):
        fx = (p[0] - x_min) / (x_max - x_min)
        fy = (p[1] - y_min) / (y_max - y_min)
        return plot_x0 + fx * (plot_x1 - plot_x0), \
               plot_y0 + fy * (plot_y1 - plot_y0)

    for cls in range(10):
        pts = proj[y == cls]
        coords = np.array([to_panel(p) for p in pts])
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=8, c=DIGIT_COLORS[cls],
                   edgecolors='none', alpha=0.7, zorder=2)

    # Legend on the bottom
    legend_y = py + 0.35
    legend_x_start = px + 0.4
    spacing = (pw - 0.8) / 10
    for cls in range(10):
        cx = legend_x_start + cls * spacing + spacing * 0.2
        ax.scatter([cx], [legend_y], s=18, c=DIGIT_COLORS[cls],
                   edgecolors='none', zorder=3)
        ax.text(cx + 0.12, legend_y, f"{cls}",
                fontsize=8, ha='left', va='center',
                color=TEXT_COLOR, fontfamily='monospace', zorder=3)


plot_proj(LEFT_PANEL, pca_proj,
          'PCA (linear): clusters smeared together')
plot_proj(RIGHT_PANEL, tsne_proj,
          't-SNE (non-linear): clusters separate cleanly')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Unsupervised Learning Part 4',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '03-unsupervised-learning/'
       '04-t-sne/header_tsne.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
