import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Colors
ARRAY_FILL = '#e3f2fd'
ARRAY_BORDER = '#1565c0'
LL_FILL = '#e0f2f1'
LL_BORDER = '#00695c'
ARROW_COLOR = '#ff9800'
TEXT_COLOR = '#333333'
LABEL_COLOR = '#555555'
BG_COLOR = '#ffffff'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.5, "Linked Lists \u2014 When Arrays Aren't Enough",
        fontsize=16, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ══════════════════════════════════════════════
# TOP HALF — Array
# ══════════════════════════════════════════════
array_label_y = 7.4
array_y = 6.2
box_w = 1.6
box_h = 1.0
values = [10, 20, 30, 40, 50]
start_x = 8 - (5 * box_w) / 2  # center 5 boxes

ax.text(8, array_label_y, 'Array: Contiguous Memory',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=ARRAY_BORDER, fontfamily='sans-serif')

for i, val in enumerate(values):
    x = start_x + i * box_w
    rect = FancyBboxPatch((x, array_y), box_w, box_h,
                           boxstyle="round,pad=0.05",
                           facecolor=ARRAY_FILL, edgecolor=ARRAY_BORDER, linewidth=2)
    ax.add_patch(rect)
    # value
    ax.text(x + box_w/2, array_y + box_h/2 + 0.1, str(val),
            fontsize=13, fontweight='bold', ha='center', va='center', color=TEXT_COLOR)
    # index
    ax.text(x + box_w/2, array_y - 0.25, f'[{i}]',
            fontsize=9, ha='center', va='center', color=LABEL_COLOR)

# ══════════════════════════════════════════════
# MIDDLE — Trade-off labels
# ══════════════════════════════════════════════
mid_y = 5.15

# Left trade-offs
ax.text(2.5, mid_y + 0.35, 'Insert at head', fontsize=9, fontweight='bold',
        ha='center', va='center', color=TEXT_COLOR)
ax.text(2.5, mid_y - 0.05, 'Array  O(n)', fontsize=9, ha='center', va='center',
        color=ARRAY_BORDER)
ax.text(2.5, mid_y - 0.4, 'Linked List  O(1)', fontsize=9, ha='center', va='center',
        color=LL_BORDER)

# Right trade-offs
ax.text(13.5, mid_y + 0.35, 'Random access', fontsize=9, fontweight='bold',
        ha='center', va='center', color=TEXT_COLOR)
ax.text(13.5, mid_y - 0.05, 'Array  O(1)', fontsize=9, ha='center', va='center',
        color=ARRAY_BORDER)
ax.text(13.5, mid_y - 0.4, 'Linked List  O(n)', fontsize=9, ha='center', va='center',
        color=LL_BORDER)

# VS divider
ax.text(8, mid_y, 'vs', fontsize=14, fontweight='bold', fontstyle='italic',
        ha='center', va='center', color='#999999')

# ══════════════════════════════════════════════
# BOTTOM HALF — Linked List
# ══════════════════════════════════════════════
ll_label_y = 4.15
node_y = 2.4
node_w = 1.8
node_h = 1.0
gap = 0.7  # space between nodes for arrows
total_w = 5 * node_w + 4 * gap
ll_start_x = 8 - total_w / 2

ax.text(8, ll_label_y, 'Linked List: Nodes + Pointers',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=LL_BORDER, fontfamily='sans-serif')

for i, val in enumerate(values):
    x = ll_start_x + i * (node_w + gap)
    # Node box
    rect = FancyBboxPatch((x, node_y), node_w, node_h,
                           boxstyle="round,pad=0.1",
                           facecolor=LL_FILL, edgecolor=LL_BORDER, linewidth=2)
    ax.add_patch(rect)

    # Data label
    ax.text(x + node_w * 0.35, node_y + node_h/2 + 0.12, str(val),
            fontsize=12, fontweight='bold', ha='center', va='center', color=TEXT_COLOR)
    ax.text(x + node_w * 0.35, node_y + node_h/2 - 0.18, 'data',
            fontsize=7, ha='center', va='center', color=LABEL_COLOR)

    # Divider line inside node
    div_x = x + node_w * 0.65
    ax.plot([div_x, div_x], [node_y + 0.12, node_y + node_h - 0.12],
            color=LL_BORDER, linewidth=1, alpha=0.5)

    # Pointer section label
    if i < len(values) - 1:
        ax.text(x + node_w * 0.83, node_y + node_h/2, '->',
                fontsize=10, fontweight='bold', ha='center', va='center',
                color=ARROW_COLOR)
    else:
        ax.text(x + node_w * 0.83, node_y + node_h/2, '/',
                fontsize=12, fontweight='bold', ha='center', va='center',
                color='#c62828')

    # Arrow to next node
    if i < len(values) - 1:
        arrow_start_x = x + node_w + 0.02
        arrow_end_x = ll_start_x + (i + 1) * (node_w + gap) - 0.02
        arrow_y = node_y + node_h / 2
        ax.annotate('',
                    xy=(arrow_end_x, arrow_y),
                    xytext=(arrow_start_x, arrow_y),
                    arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                    lw=2.5, mutation_scale=18))

# None label after last node
last_x = ll_start_x + 4 * (node_w + gap) + node_w + 0.15
ax.text(last_x + 0.4, node_y + node_h/2, 'None',
        fontsize=10, fontstyle='italic', fontweight='bold',
        ha='left', va='center', color='#c62828')

# ── Subtitle ──
ax.text(8, 1.3, 'Understanding when and why to use linked lists over arrays',
        fontsize=10, ha='center', va='center', color=LABEL_COLOR,
        fontstyle='italic', fontfamily='sans-serif')

# ── Footer ──
ax.text(8, 0.5, 'Algorithms in Python  |  Foundations Series',
        fontsize=8, ha='center', va='center', color='#aaaaaa',
        fontfamily='sans-serif')

plt.tight_layout(pad=0.3)
out = 'D:/Projects/Medium/algorithms-in-python/00-foundations/04-linked-lists/header_linked_lists.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR, bbox_inches='tight', pad_inches=0.2)
plt.close()
print(f'Saved to {out}')
