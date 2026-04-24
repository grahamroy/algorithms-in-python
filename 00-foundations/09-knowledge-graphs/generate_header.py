import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

# ── Colours ──
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

# Symbolic / triple colours (blue family)
ENTITY_FILL = '#dbeafe'
ENTITY_BORDER = '#3b82f6'
RELATION_COLOR = '#1F2937'

# Embedding / vector colours (purple family)
HEAD_COLOR = '#8b5cf6'
TAIL_COLOR = '#16a34a'
RELATION_VEC = '#F59E0B'
PREDICTED = '#DC2626'

ARROW_COLOR = '#475569'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.45, 'Knowledge Graphs: Symbols Meet Embeddings',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ── Subtitle ──
ax.text(8, 7.95,
        'Triples for reasoning, vectors for similarity --- and RAG uses both',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# Panel geometry
# ═══════════════════════════════════════════════════════════
LEFT_PANEL = (0.4, 0.9, 7.2, 6.0)
RIGHT_PANEL = (8.4, 0.9, 7.2, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)

# ═══════════════════════════════════════════════════════════
# LEFT PANEL --- Symbolic graph of triples
# ═══════════════════════════════════════════════════════════
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4, 'Symbolic: typed triples',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Layout entities as nodes around a triangle / fan
cx_l = lpx + lpw / 2
cy_l = lpy + lph / 2 - 0.2

# Place 5 entities in a sensible layout
entity_layout = {
    'Marie Curie':  (cx_l - 2.0, cy_l + 0.7),
    'Warsaw':       (cx_l + 1.6, cy_l + 1.5),
    'Poland':       (cx_l + 1.9, cy_l - 1.4),
    'Polonium':     (cx_l - 2.4, cy_l - 1.5),
    'Nobel Prize':  (cx_l + 0.0, cy_l + 2.1),
}

# Box geometry per entity
node_w = 1.6
node_h = 0.55

def draw_entity(name, pos):
    x, y = pos
    box = FancyBboxPatch((x - node_w/2, y - node_h/2), node_w, node_h,
                         boxstyle="round,pad=0.02,rounding_size=0.10",
                         facecolor=ENTITY_FILL, edgecolor=ENTITY_BORDER,
                         linewidth=1.4, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, name,
            fontsize=9.5, fontweight='bold', ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=4)

for name, pos in entity_layout.items():
    draw_entity(name, pos)

# Draw labelled edges (triples)
edges = [
    ('Marie Curie', 'Warsaw',      'born_in'),
    ('Marie Curie', 'Polonium',    'discovered'),
    ('Marie Curie', 'Nobel Prize', 'won'),
    ('Warsaw',      'Poland',      'located_in'),
    ('Polonium',    'Poland',      'named_after'),
]

def edge_endpoints(p1, p2, r1=node_w/2, r2=node_w/2):
    """Return start/end points pulled in from box edges along the connector line."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    dist = math_safe_dist(dx, dy)
    if dist == 0:
        return (x1, y1), (x2, y2)
    # Pull in by approximately half the box width along the line direction
    # (a rough approximation; close enough for a header image)
    pad = 0.55
    s = ((x1 + (pad / dist) * dx), (y1 + (pad / dist) * dy))
    e = ((x2 - (pad / dist) * dx), (y2 - (pad / dist) * dy))
    return s, e


def math_safe_dist(dx, dy):
    return (dx * dx + dy * dy) ** 0.5


for (a, b, label) in edges:
    pa = entity_layout[a]
    pb = entity_layout[b]
    (sx, sy), (ex, ey) = edge_endpoints(pa, pb)
    arrow = FancyArrowPatch((sx, sy), (ex, ey),
                            arrowstyle='-|>', mutation_scale=12,
                            color=ARROW_COLOR, linewidth=1.3,
                            zorder=2)
    ax.add_patch(arrow)
    # Label at midpoint
    mx, my = (sx + ex) / 2, (sy + ey) / 2
    ax.text(mx, my + 0.05, label,
            fontsize=8, fontstyle='italic', ha='center', va='center',
            color=RELATION_COLOR, fontfamily='sans-serif',
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor=PANEL_BG, edgecolor='none', alpha=0.85),
            zorder=3.5)

# Caption
ax.text(lpx + lpw/2, lpy + 0.55,
        'Every edge is a typed relation --- queryable with SPARQL',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# RIGHT PANEL --- TransE embedding: h + r ≈ t
# ═══════════════════════════════════════════════════════════
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4, 'Neural: h + r ≈ t  (TransE)',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# 2D vector space inside the right panel
plot_left = rpx + 0.7
plot_right = rpx + rpw - 0.7
plot_bottom = rpy + 1.2
plot_top = rpy + rph - 1.0

# Plot box (faint)
from matplotlib.patches import Rectangle
outer = Rectangle((plot_left, plot_bottom),
                  plot_right - plot_left,
                  plot_top - plot_bottom,
                  facecolor='white', edgecolor='#E2E8F0',
                  linewidth=1.0, zorder=1)
ax.add_patch(outer)

# Origin (centre of plot box)
origin_x = (plot_left + plot_right) / 2
origin_y = (plot_bottom + plot_top) / 2

# Choose three vectors so that h + r lands very close to t
# h points up-left, r points down-right, t = h + r is mid-right
h_end = (origin_x - 1.8, origin_y + 1.4)
r_offset = (2.6, -0.5)
t_end = (h_end[0] + r_offset[0], h_end[1] + r_offset[1])  # exact

# A few extra entity points scattered around the space
rng = np.random.default_rng(11)
extras = []
for _ in range(8):
    ex = rng.uniform(plot_left + 0.4, plot_right - 0.4)
    ey = rng.uniform(plot_bottom + 0.4, plot_top - 0.4)
    # Avoid placing too close to t_end
    while abs(ex - t_end[0]) < 0.6 and abs(ey - t_end[1]) < 0.6:
        ex = rng.uniform(plot_left + 0.4, plot_right - 0.4)
        ey = rng.uniform(plot_bottom + 0.4, plot_top - 0.4)
    extras.append((ex, ey))

# Plot the extras as small grey points
for (ex, ey) in extras:
    ax.add_patch(Circle((ex, ey), 0.10, facecolor='#CBD5E1',
                        edgecolor='#94A3B8', linewidth=0.8, zorder=3))

# Vector arrows (no inline label -- we place text manually for control)
def vec_arrow(start, end, color, lw=2.0):
    arrow = FancyArrowPatch(start, end,
                            arrowstyle='-|>', mutation_scale=14,
                            color=color, linewidth=lw, zorder=4)
    ax.add_patch(arrow)

# Origin -> h (head vector, pointing up-left)
vec_arrow((origin_x, origin_y), h_end, HEAD_COLOR, lw=2.2)
# h -> h+r (the relation step)
vec_arrow(h_end, t_end, RELATION_VEC, lw=2.2)

# Labels placed manually so they do not collide with the t marker
# h label: just LEFT of h_end (the vector tip), since h points up-left
ax.text(h_end[0] - 0.30, h_end[1] - 0.05, 'h (Marie Curie)',
        fontsize=10, fontweight='bold', ha='right', va='center',
        color=HEAD_COLOR, fontfamily='sans-serif',
        bbox=dict(boxstyle="round,pad=0.18",
                  facecolor=PANEL_BG, edgecolor='none', alpha=0.9),
        zorder=5)

# r label: above the relation arrow midpoint, well clear of the t marker
r_mid_x = (h_end[0] + t_end[0]) / 2
r_mid_y = (h_end[1] + t_end[1]) / 2
ax.text(r_mid_x - 0.20, r_mid_y + 0.45, 'r (born_in)',
        fontsize=10, fontweight='bold', ha='center', va='center',
        color=RELATION_VEC, fontfamily='sans-serif',
        bbox=dict(boxstyle="round,pad=0.18",
                  facecolor=PANEL_BG, edgecolor='none', alpha=0.9),
        zorder=5)

# Highlight the predicted tail position (h + r)
ax.add_patch(Circle(t_end, 0.20, facecolor=PREDICTED,
                    edgecolor='white', linewidth=1.4, zorder=6))
ax.add_patch(Circle(t_end, 0.32, facecolor='none',
                    edgecolor=PREDICTED, linewidth=1.4,
                    linestyle=(0, (3, 2)), zorder=6))
ax.text(t_end[0] + 0.40, t_end[1] - 0.05, 't (Warsaw)',
        fontsize=10, fontweight='bold', ha='left', va='center',
        color=TAIL_COLOR, fontfamily='sans-serif',
        bbox=dict(boxstyle="round,pad=0.18",
                  facecolor=PANEL_BG, edgecolor='none', alpha=0.9),
        zorder=7)

# Origin marker
ax.add_patch(Circle((origin_x, origin_y), 0.06, facecolor=TEXT_COLOR,
                    edgecolor='none', zorder=5))

# Caption
ax.text(rpx + rpw/2, rpy + 0.55,
        'Embed each entity and relation as a vector --- geometry mirrors logic',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
ax.text(8, 0.3, 'Algorithms in Python  |  Foundations Part 9',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/00-foundations/09-knowledge-graphs/header_knowledge_graphs.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
