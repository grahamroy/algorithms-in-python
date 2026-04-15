import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np

# ── Colours ──
NODE_FILL = '#dbeafe'
NODE_BORDER = '#1e40af'
EDGE_COLOR = '#475569'
HIGHLIGHT_RED = '#ef4444'
HIGHLIGHT_ORANGE = '#f97316'
HIGHLIGHT_YELLOW = '#fbbf24'
TEXT_COLOR = '#333333'
LABEL_COLOR = '#555555'
BG_COLOR = '#ffffff'
MATRIX_FILL_ON = '#1e40af'
MATRIX_FILL_OFF = '#f1f5f9'
MATRIX_BORDER = '#cbd5e1'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.4, 'Graphs: Nodes and Edges',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ── Subtitle ──
ax.text(8, 7.75, 'The data structure behind social networks, knowledge bases, and GNNs',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=LABEL_COLOR, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# LEFT PANEL — Undirected Graph
# ═══════════════════════════════════════════════════════════

# Panel label
ax.text(4, 6.85, 'Undirected Graph',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=NODE_BORDER, fontfamily='sans-serif')

# Node positions — hand-placed for a clean readable graph (7 nodes)
# Layout: node 0 at center-left, with a small cluster spreading outward
node_positions = {
    0: (2.2, 5.4),
    1: (4.0, 6.1),
    2: (5.8, 5.3),
    3: (3.6, 4.2),
    4: (5.6, 3.4),
    5: (1.8, 3.1),
    6: (3.4, 2.3),
}

# Edges — undirected graph structure
edges = [
    (0, 1),
    (0, 3),
    (0, 5),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 4),
    (3, 6),
    (5, 6),
    (4, 6),
]

# BFS traversal colouring starting from node 0
# Distance 0: node 0 (red)
# Distance 1: nodes 1, 3, 5 (orange)
# Distance 2: nodes 2, 4, 6 (yellow)
bfs_colors = {
    0: HIGHLIGHT_RED,
    1: HIGHLIGHT_ORANGE,
    3: HIGHLIGHT_ORANGE,
    5: HIGHLIGHT_ORANGE,
    2: HIGHLIGHT_YELLOW,
    4: HIGHLIGHT_YELLOW,
    6: HIGHLIGHT_YELLOW,
}

# Draw edges first so they sit behind nodes
for u, v in edges:
    x1, y1 = node_positions[u]
    x2, y2 = node_positions[v]
    ax.plot([x1, x2], [y1, y2],
            color=EDGE_COLOR, linewidth=1.8, zorder=1, alpha=0.85)

# Draw nodes
node_radius = 0.32
for node_id, (x, y) in node_positions.items():
    fill = bfs_colors.get(node_id, NODE_FILL)
    border = NODE_BORDER
    # Use a lighter border for highlighted nodes to contrast with fill
    if node_id in bfs_colors:
        border = '#7f1d1d' if node_id == 0 else ('#9a3412' if bfs_colors[node_id] == HIGHLIGHT_ORANGE else '#92400e')
    circle = Circle((x, y), node_radius,
                    facecolor=fill, edgecolor=border,
                    linewidth=2.2, zorder=2)
    ax.add_patch(circle)
    # Node label — white text on coloured nodes, dark text on default nodes
    label_color = 'white' if node_id in bfs_colors else TEXT_COLOR
    ax.text(x, y, str(node_id),
            fontsize=11, fontweight='bold', ha='center', va='center',
            color=label_color, zorder=3)

# BFS legend beneath the graph
legend_y = 1.45
legend_items = [
    (HIGHLIGHT_RED, 'start'),
    (HIGHLIGHT_ORANGE, '1 hop'),
    (HIGHLIGHT_YELLOW, '2 hops'),
]
legend_start_x = 1.6
legend_spacing = 1.55
for i, (color, label) in enumerate(legend_items):
    cx = legend_start_x + i * legend_spacing
    c = Circle((cx, legend_y), 0.17,
               facecolor=color, edgecolor='#555555', linewidth=1.2)
    ax.add_patch(c)
    ax.text(cx + 0.3, legend_y, label,
            fontsize=9, ha='left', va='center', color=TEXT_COLOR)

# Small caption under BFS legend
ax.text(4, 0.85, 'BFS traversal from node 0',
        fontsize=8, fontstyle='italic', ha='center', va='center',
        color=LABEL_COLOR)

# ═══════════════════════════════════════════════════════════
# DIVIDER
# ═══════════════════════════════════════════════════════════
ax.plot([8, 8], [1.1, 6.95], color='#e2e8f0', linewidth=1.2, zorder=0)

# ═══════════════════════════════════════════════════════════
# RIGHT PANEL — Adjacency Matrix
# ═══════════════════════════════════════════════════════════

# Panel label
ax.text(12, 6.85, 'Adjacency Matrix',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=NODE_BORDER, fontfamily='sans-serif')

# Build adjacency matrix from edges
n = 7
adj = np.zeros((n, n), dtype=int)
for u, v in edges:
    adj[u, v] = 1
    adj[v, u] = 1

# Matrix drawing parameters
cell_size = 0.55
matrix_origin_x = 10.05  # top-left of the grid (x of first cell)
matrix_origin_y = 6.25   # top of first row
# Reserve space on left/top for row/column labels

# Column headers (node indices across the top)
for j in range(n):
    cx = matrix_origin_x + j * cell_size + cell_size / 2
    cy = matrix_origin_y + 0.3
    ax.text(cx, cy, str(j),
            fontsize=9, fontweight='bold', ha='center', va='center',
            color=NODE_BORDER)

# Row headers (node indices down the left)
for i in range(n):
    rx = matrix_origin_x - 0.3
    ry = matrix_origin_y - i * cell_size - cell_size / 2
    ax.text(rx, ry, str(i),
            fontsize=9, fontweight='bold', ha='center', va='center',
            color=NODE_BORDER)

# Draw matrix cells
for i in range(n):
    for j in range(n):
        x = matrix_origin_x + j * cell_size
        y = matrix_origin_y - (i + 1) * cell_size
        val = adj[i, j]
        fill = MATRIX_FILL_ON if val == 1 else MATRIX_FILL_OFF
        rect = Rectangle((x, y), cell_size, cell_size,
                         facecolor=fill, edgecolor=MATRIX_BORDER,
                         linewidth=0.8)
        ax.add_patch(rect)
        txt_color = 'white' if val == 1 else '#94a3b8'
        ax.text(x + cell_size / 2, y + cell_size / 2, str(val),
                fontsize=8, fontweight='bold', ha='center', va='center',
                color=txt_color)

# Axis labels for the matrix
matrix_left = matrix_origin_x
matrix_right = matrix_origin_x + n * cell_size
matrix_top = matrix_origin_y
matrix_bottom = matrix_origin_y - n * cell_size

# "to" label across the top
ax.text((matrix_left + matrix_right) / 2, matrix_top + 0.65, 'to',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=LABEL_COLOR)
# "from" label down the left, rotated
ax.text(matrix_left - 0.75, (matrix_top + matrix_bottom) / 2, 'from',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=LABEL_COLOR, rotation=90)

# Small caption under the matrix
ax.text(12, 0.85, 'Symmetric matrix: A[i][j] = 1 iff edge (i, j) exists',
        fontsize=8, fontstyle='italic', ha='center', va='center',
        color=LABEL_COLOR)

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
ax.text(8, 0.3, 'Algorithms in Python  |  Foundations Part 5',
        fontsize=8, ha='center', va='center', color='#aaaaaa',
        fontfamily='sans-serif')

plt.tight_layout(pad=0.3)
out = 'D:/Projects/Medium/algorithms-in-python/00-foundations/05-graphs/header_graphs.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR, bbox_inches='tight', pad_inches=0.2)
plt.close()
print(f'Saved to {out}')
