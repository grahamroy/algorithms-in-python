import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

# ── Colours ──
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

# Per-family palette
HNSW_NODE = '#3b82f6'
HNSW_EDGE = '#94A3B8'

IVF_CENTROID = '#F59E0B'
IVF_POINT = '#FCD34D'
IVF_REGION = '#FEF3C7'

PQ_FILL = '#ede9fe'
PQ_BORDER = '#8b5cf6'
PQ_CODE = '#7c3aed'

LSH_FILL = '#dcfce7'
LSH_BORDER = '#16a34a'

QUERY_COLOR = '#DC2626'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.45,
        'Vector Indexes (ANN): Four Ways to Search Embedding Space',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ── Subtitle ──
ax.text(8, 7.95,
        'HNSW walks a graph, IVF partitions space, PQ compresses vectors, LSH hashes them',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# Four panels in a 2x2 grid
# ═══════════════════════════════════════════════════════════
PANEL_W = 7.4
PANEL_H = 3.05
GAP = 0.25
TOTAL_W = 2 * PANEL_W + GAP
START_X = (16 - TOTAL_W) / 2

PANEL_TOP_Y = 4.05
PANEL_BOTTOM_Y = 0.85

panels = [
    (START_X, PANEL_TOP_Y),                 # top-left: HNSW
    (START_X + PANEL_W + GAP, PANEL_TOP_Y), # top-right: IVF
    (START_X, PANEL_BOTTOM_Y),              # bottom-left: PQ
    (START_X + PANEL_W + GAP, PANEL_BOTTOM_Y),  # bottom-right: LSH
]

for (px, py) in panels:
    panel = FancyBboxPatch((px, py), PANEL_W, PANEL_H,
                           boxstyle="round,pad=0.02,rounding_size=0.12",
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)


# ═══════════════════════════════════════════════════════════
# PANEL 1 (top-left): HNSW --- multi-layer graph
# ═══════════════════════════════════════════════════════════
px, py = panels[0]
ax.text(px + PANEL_W/2, py + PANEL_H - 0.3, 'HNSW',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(px + PANEL_W/2, py + PANEL_H - 0.6,
        'multi-layer graph, greedy walk',
        fontsize=8.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Three layers, each with nodes and a few edges
layer_ys = [py + 1.95, py + 1.40, py + 0.85]
layer_n = [3, 5, 9]
layer_xs = []
# Push the nodes to the right of the layer labels
node_left = px + 1.15
for ly, ny in zip(layer_ys, layer_n):
    xs = np.linspace(node_left, px + PANEL_W - 0.85, ny)
    layer_xs.append(xs)
    # Edges within the layer (a few)
    for i in range(ny - 1):
        ax.plot([xs[i], xs[i+1]], [ly, ly],
                color=HNSW_EDGE, linewidth=0.9, zorder=2)
    # A non-adjacent shortcut for layers 1 & 2
    if ny >= 4:
        ax.plot([xs[0], xs[ny-1]], [ly, ly],
                color=HNSW_EDGE, linewidth=0.7, alpha=0.6, zorder=2)
    # Nodes
    for x in xs:
        ax.add_patch(Circle((x, ly), 0.10,
                            facecolor=HNSW_NODE, edgecolor='white',
                            linewidth=0.8, zorder=3))

# Vertical connections between adjacent layers (just a couple to hint at it)
ax.plot([layer_xs[0][1], layer_xs[1][2]],
        [layer_ys[0], layer_ys[1]],
        color=HNSW_EDGE, linewidth=0.7, linestyle=':', zorder=2)
ax.plot([layer_xs[1][2], layer_xs[2][4]],
        [layer_ys[1], layer_ys[2]],
        color=HNSW_EDGE, linewidth=0.7, linestyle=':', zorder=2)

# Layer labels (right-aligned just left of the first node)
for ly, label in zip(layer_ys, ['L2', 'L1', 'L0']):
    ax.text(node_left - 0.18, ly, label,
            fontsize=8, ha='right', va='center',
            color=SUBTLE_TEXT, fontfamily='monospace')

# Search annotation
ax.text(px + PANEL_W/2, py + 0.35,
        'query enters at top, descends greedily to nearest at layer 0',
        fontsize=8, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# PANEL 2 (top-right): IVF --- partitioned space
# ═══════════════════════════════════════════════════════════
px, py = panels[1]
ax.text(px + PANEL_W/2, py + PANEL_H - 0.3, 'IVF',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(px + PANEL_W/2, py + PANEL_H - 0.6,
        'k-means buckets, search top nprobe',
        fontsize=8.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Plot box for the embedding space
plot_left = px + 0.5
plot_right = px + PANEL_W - 0.5
plot_bottom = py + 0.5
plot_top = py + PANEL_H - 0.85

ax.add_patch(Rectangle((plot_left, plot_bottom),
                       plot_right - plot_left,
                       plot_top - plot_bottom,
                       facecolor='white', edgecolor='#E2E8F0',
                       linewidth=1.0, zorder=1))

# Centroid positions (5 clusters)
rng = np.random.default_rng(11)
centres = np.array([
    [plot_left + 1.0, plot_bottom + 1.5],
    [plot_left + 3.2, plot_bottom + 1.7],
    [plot_left + 5.0, plot_bottom + 1.4],
    [plot_left + 1.8, plot_bottom + 0.6],
    [plot_left + 4.5, plot_bottom + 0.5],
])

# Tinted bucket regions (loose circles)
for cx, cy in centres:
    ax.add_patch(Circle((cx, cy), 0.65,
                        facecolor=IVF_REGION, edgecolor='none',
                        alpha=0.6, zorder=1.5))

# Scatter points around each centre
for cx, cy in centres:
    pts = rng.normal(loc=[cx, cy], scale=0.18, size=(8, 2))
    pts = np.clip(pts,
                  [plot_left + 0.1, plot_bottom + 0.1],
                  [plot_right - 0.1, plot_top - 0.1])
    ax.scatter(pts[:, 0], pts[:, 1], s=12, color=IVF_POINT,
               edgecolors='#D97706', linewidths=0.5, zorder=2)

# Centroid markers
for cx, cy in centres:
    ax.add_patch(Circle((cx, cy), 0.13,
                        facecolor=IVF_CENTROID, edgecolor='white',
                        linewidth=1.2, zorder=4))

# Query point
qx, qy = plot_left + 3.4, plot_top - 0.35
ax.scatter([qx], [qy], s=120, color=QUERY_COLOR, marker='*',
           edgecolors=TEXT_COLOR, linewidth=0.6, zorder=6)
ax.text(qx + 0.18, qy, 'query',
        fontsize=8, fontweight='bold', ha='left', va='center',
        color=QUERY_COLOR, fontfamily='sans-serif', zorder=7)

# Highlight the nearest centroid (centre 1)
nearest_cx, nearest_cy = centres[1]
ax.add_patch(Circle((nearest_cx, nearest_cy), 0.78,
                    facecolor='none', edgecolor=QUERY_COLOR,
                    linewidth=1.4, linestyle=(0, (3, 2)), zorder=5))

# ═══════════════════════════════════════════════════════════
# PANEL 3 (bottom-left): PQ --- subvector compression
# ═══════════════════════════════════════════════════════════
px, py = panels[2]
ax.text(px + PANEL_W/2, py + PANEL_H - 0.3, 'PQ',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(px + PANEL_W/2, py + PANEL_H - 0.6,
        'split into m subvectors, code each as 1 byte',
        fontsize=8.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Original vector: a long row of cells
orig_y = py + 1.65
n_orig = 16
orig_w = 0.30
orig_h = 0.30
orig_total = n_orig * orig_w
orig_x0 = px + (PANEL_W - orig_total) / 2
ax.text(orig_x0 - 0.18, orig_y + orig_h / 2, 'vector ',
        fontsize=8.5, ha='right', va='center',
        color=TEXT_COLOR, fontfamily='monospace')
for i in range(n_orig):
    ax.add_patch(Rectangle((orig_x0 + i * orig_w, orig_y),
                           orig_w, orig_h,
                           facecolor='white', edgecolor=PQ_BORDER,
                           linewidth=0.8, zorder=2))

# Subvector groupings (m = 4 here for visual clarity)
m = 4
sub_w = orig_w * (n_orig // m)
sub_y = orig_y - 0.05
for j in range(m):
    sx = orig_x0 + j * sub_w
    ax.add_patch(Rectangle((sx, orig_y - 0.02), sub_w, orig_h + 0.04,
                           facecolor='none',
                           edgecolor=PQ_BORDER, linewidth=1.5,
                           zorder=3))

# Compressed code row
code_y = orig_y - 1.2
code_total = m * 0.45
code_x0 = px + (PANEL_W - code_total) / 2
ax.text(code_x0 - 0.18, code_y + 0.18, 'codes  ',
        fontsize=8.5, ha='right', va='center',
        color=TEXT_COLOR, fontfamily='monospace')
example_codes = [37, 192, 8, 101]
for j in range(m):
    cx = code_x0 + j * 0.45
    ax.add_patch(Rectangle((cx, code_y), 0.42, 0.36,
                           facecolor=PQ_FILL, edgecolor=PQ_BORDER,
                           linewidth=1.2, zorder=2))
    ax.text(cx + 0.21, code_y + 0.18, str(example_codes[j]),
            fontsize=9, fontweight='bold', ha='center', va='center',
            color=PQ_CODE, fontfamily='monospace', zorder=3)

# Arrow from each subvector group down to its code
for j in range(m):
    sx_centre = orig_x0 + j * sub_w + sub_w / 2
    cx_centre = code_x0 + j * 0.45 + 0.21
    ax.annotate('',
                xy=(cx_centre, code_y + 0.36),
                xytext=(sx_centre, orig_y - 0.04),
                arrowprops=dict(arrowstyle='->',
                                color='#9CA3AF', lw=0.8))

# Caption
ax.text(px + PANEL_W/2, py + 0.30,
        '64x compression, distance via lookup tables',
        fontsize=8.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# PANEL 4 (bottom-right): LSH --- hash buckets
# ═══════════════════════════════════════════════════════════
px, py = panels[3]
ax.text(px + PANEL_W/2, py + PANEL_H - 0.3, 'LSH',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(px + PANEL_W/2, py + PANEL_H - 0.6,
        'similar vectors collide in the same bucket',
        fontsize=8.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Hash table: a row of buckets, each containing a few vector dots
n_buckets = 5
bucket_w = (PANEL_W - 1.0) / n_buckets
bucket_y = py + 0.85
bucket_h = 1.05
bucket_x0 = px + 0.5

# Vectors per bucket (just dots)
points_per_bucket = [3, 4, 6, 2, 3]
for j, n_pts in enumerate(points_per_bucket):
    bx = bucket_x0 + j * bucket_w
    ax.add_patch(Rectangle((bx + 0.05, bucket_y),
                           bucket_w - 0.10, bucket_h,
                           facecolor=LSH_FILL, edgecolor=LSH_BORDER,
                           linewidth=1.0, zorder=2))
    # Bucket label
    ax.text(bx + bucket_w / 2, bucket_y - 0.18, f'h={j}',
            fontsize=7.5, ha='center', va='top',
            color=SUBTLE_TEXT, fontfamily='monospace')
    # Dots inside
    rng_b = np.random.default_rng(20 + j)
    for _ in range(n_pts):
        dx = rng_b.uniform(bx + 0.20, bx + bucket_w - 0.20)
        dy = rng_b.uniform(bucket_y + 0.20, bucket_y + bucket_h - 0.20)
        ax.add_patch(Circle((dx, dy), 0.07,
                            facecolor=LSH_BORDER, edgecolor='white',
                            linewidth=0.6, zorder=3))

# Query lands in bucket 2
target_bucket = 2
tb_x = bucket_x0 + target_bucket * bucket_w + bucket_w / 2
tb_y_top = bucket_y + bucket_h
ax.scatter([tb_x], [tb_y_top + 0.45], s=110,
           color=QUERY_COLOR, marker='*',
           edgecolors=TEXT_COLOR, linewidth=0.6, zorder=6)
ax.annotate('',
            xy=(tb_x, tb_y_top + 0.05),
            xytext=(tb_x, tb_y_top + 0.32),
            arrowprops=dict(arrowstyle='->', color=QUERY_COLOR, lw=1.2))
ax.text(tb_x + 0.30, tb_y_top + 0.45, 'query  hash',
        fontsize=7.5, ha='left', va='center',
        color=QUERY_COLOR, fontfamily='monospace', zorder=6)

# Caption
ax.text(px + PANEL_W/2, py + 0.30,
        'covered in Part 10 -- still strong for set similarity (MinHash)',
        fontsize=8.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
ax.text(8, 0.3, 'Algorithms in Python  |  Foundations Part 12 (final)',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/00-foundations/12-vector-indexes/header_ann.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
