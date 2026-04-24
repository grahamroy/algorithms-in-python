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

# Bloom: blue
BLOOM_FILL = '#dbeafe'
BLOOM_BORDER = '#3b82f6'
BLOOM_SET_FILL = '#3b82f6'  # set bits

# CMS: amber/gold
CMS_FILL = '#FEF3C7'
CMS_BORDER = '#F59E0B'

# HLL: purple
HLL_FILL = '#ede9fe'
HLL_BORDER = '#8b5cf6'

ARROW_COLOR = '#475569'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.45, 'Probabilistic Data Structures: Sublinear Memory',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ── Subtitle ──
ax.text(8, 7.95,
        'Bloom for membership, Count-Min for counts, HyperLogLog for cardinality',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# Three side-by-side panels for the three structures
# ═══════════════════════════════════════════════════════════
PANEL_WIDTH = 5.0
PANEL_HEIGHT = 6.1
GAP = 0.25
TOTAL_WIDTH = 3 * PANEL_WIDTH + 2 * GAP
START_X = (16 - TOTAL_WIDTH) / 2
PANEL_Y = 0.85

panel_x = []
for i in range(3):
    px = START_X + i * (PANEL_WIDTH + GAP)
    panel_x.append(px)
    panel = FancyBboxPatch((px, PANEL_Y), PANEL_WIDTH, PANEL_HEIGHT,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)

# ═══════════════════════════════════════════════════════════
# PANEL 1 --- Bloom filter
# ═══════════════════════════════════════════════════════════
p1x = panel_x[0]
ax.text(p1x + PANEL_WIDTH/2, PANEL_Y + PANEL_HEIGHT - 0.4,
        'Bloom filter',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(p1x + PANEL_WIDTH/2, PANEL_Y + PANEL_HEIGHT - 0.85,
        '"have I seen this?"',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Bit array: 12 cells in a row
n_bits = 12
cell_w = 0.32
cell_h = 0.4
bits_total_w = n_bits * cell_w
bits_x0 = p1x + (PANEL_WIDTH - bits_total_w) / 2
bits_y = PANEL_Y + PANEL_HEIGHT / 2 - 0.2

set_bits = {1, 4, 8}  # bits set after one item is inserted
for i in range(n_bits):
    is_set = i in set_bits
    fill = BLOOM_SET_FILL if is_set else 'white'
    border = BLOOM_BORDER
    rect = Rectangle((bits_x0 + i * cell_w, bits_y), cell_w, cell_h,
                     facecolor=fill, edgecolor=border,
                     linewidth=1.0, zorder=2)
    ax.add_patch(rect)
    label = '1' if is_set else '0'
    color = 'white' if is_set else SUBTLE_TEXT
    ax.text(bits_x0 + i * cell_w + cell_w/2, bits_y + cell_h/2, label,
            fontsize=8.5, fontweight='bold', ha='center', va='center',
            color=color, fontfamily='monospace', zorder=3)

# Three "h_i(x)" arrows pointing into the three set bits
hash_y = bits_y + cell_h + 0.7
for i, bit_idx in enumerate(sorted(set_bits)):
    target_x = bits_x0 + bit_idx * cell_w + cell_w / 2
    target_y = bits_y + cell_h
    src_x = target_x
    src_y = hash_y
    arrow = FancyArrowPatch((src_x, src_y), (target_x, target_y),
                            arrowstyle='-|>', mutation_scale=10,
                            color=ARROW_COLOR, linewidth=1.0, zorder=2)
    ax.add_patch(arrow)
    ax.text(src_x, src_y + 0.18, f'h{i+1}(x)',
            fontsize=8, ha='center', va='center',
            color=BLOOM_BORDER, fontfamily='monospace')

# Item label
item_y = hash_y + 0.85
ax.text(p1x + PANEL_WIDTH/2, item_y, 'item x',
        fontsize=11, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Caption
ax.text(p1x + PANEL_WIDTH/2, PANEL_Y + 1.1,
        'm bits, k hashes',
        fontsize=10, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(p1x + PANEL_WIDTH/2, PANEL_Y + 0.65,
        '~10 bits/item @ 1% FPR',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# PANEL 2 --- Count-Min Sketch
# ═══════════════════════════════════════════════════════════
p2x = panel_x[1]
ax.text(p2x + PANEL_WIDTH/2, PANEL_Y + PANEL_HEIGHT - 0.4,
        'Count-Min Sketch',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(p2x + PANEL_WIDTH/2, PANEL_Y + PANEL_HEIGHT - 0.85,
        '"how often have I seen this?"',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# A 4 x 8 grid of counter cells; highlight one cell per row (the hashed positions)
cms_rows = 4
cms_cols = 8
cms_cell_w = 0.42
cms_cell_h = 0.42
cms_total_w = cms_cols * cms_cell_w
cms_total_h = cms_rows * cms_cell_h
cms_x0 = p2x + (PANEL_WIDTH - cms_total_w) / 2
cms_y0 = PANEL_Y + PANEL_HEIGHT / 2 - cms_total_h / 2 - 0.2

# Random-looking hashed positions per row
hashed_cols = [3, 6, 1, 5]
counters = [
    [12,  7, 18,  9, 15,  3, 11,  6],
    [ 8, 14,  4, 11,  2, 17,  9,  5],
    [ 5,  3, 22,  7, 13,  6, 10,  1],
    [11,  4,  9,  2,  8, 16,  3, 12],
]

for r in range(cms_rows):
    for c in range(cms_cols):
        is_hashed = (c == hashed_cols[r])
        fill = CMS_FILL if is_hashed else 'white'
        border = CMS_BORDER if is_hashed else PANEL_EDGE
        lw = 1.4 if is_hashed else 0.8
        x = cms_x0 + c * cms_cell_w
        y = cms_y0 + (cms_rows - 1 - r) * cms_cell_h  # row 0 at top
        ax.add_patch(Rectangle((x, y), cms_cell_w, cms_cell_h,
                               facecolor=fill, edgecolor=border,
                               linewidth=lw, zorder=2))
        # Show counter value only in the hashed cell of each row
        if is_hashed:
            ax.text(x + cms_cell_w/2, y + cms_cell_h/2, str(counters[r][c]),
                    fontsize=8, fontweight='bold', ha='center', va='center',
                    color=TEXT_COLOR, fontfamily='monospace', zorder=3)

# Annotation pointing to the min: (3, 22, 8, 9) -> min = 3 (row 2)
min_row = min(range(cms_rows), key=lambda r: counters[r][hashed_cols[r]])
min_col = hashed_cols[min_row]
min_val = counters[min_row][min_col]
min_x = cms_x0 + min_col * cms_cell_w + cms_cell_w / 2
min_y = cms_y0 + (cms_rows - 1 - min_row) * cms_cell_h - 0.05
ax.annotate(f'min = {min_val}',
            xy=(min_x, min_y),
            xytext=(min_x + 0.6, min_y - 0.55),
            fontsize=9, fontweight='bold', color=CMS_BORDER,
            fontfamily='sans-serif',
            arrowprops=dict(arrowstyle='->', color=CMS_BORDER, lw=1.0))

# Caption
ax.text(p2x + PANEL_WIDTH/2, PANEL_Y + 1.1,
        'd × w counters, take min',
        fontsize=10, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(p2x + PANEL_WIDTH/2, PANEL_Y + 0.65,
        '~50 KB streams unbounded vocab',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# PANEL 3 --- HyperLogLog
# ═══════════════════════════════════════════════════════════
p3x = panel_x[2]
ax.text(p3x + PANEL_WIDTH/2, PANEL_Y + PANEL_HEIGHT - 0.4,
        'HyperLogLog',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(p3x + PANEL_WIDTH/2, PANEL_Y + PANEL_HEIGHT - 0.85,
        '"how many distinct items?"',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# 4x4 grid of "registers" each showing a small bar inside (visualising the
# leading-zero count stored in that register)
hll_rows = 4
hll_cols = 4
reg_w = 0.7
reg_h = 0.7
hll_total_w = hll_cols * reg_w
hll_total_h = hll_rows * reg_h
hll_x0 = p3x + (PANEL_WIDTH - hll_total_w) / 2
hll_y0 = PANEL_Y + PANEL_HEIGHT / 2 - hll_total_h / 2 - 0.2

# Each register stores a leading-zero count (0..7 ish for a small demo)
register_values = [
    [3, 5, 2, 4],
    [1, 7, 3, 2],
    [4, 2, 6, 3],
    [5, 1, 4, 2],
]

for r in range(hll_rows):
    for c in range(hll_cols):
        x = hll_x0 + c * reg_w
        y = hll_y0 + (hll_rows - 1 - r) * reg_h
        ax.add_patch(Rectangle((x, y), reg_w, reg_h,
                               facecolor=HLL_FILL, edgecolor=HLL_BORDER,
                               linewidth=1.0, zorder=2))
        # Render a small "level bar" inside each register
        v = register_values[r][c]
        max_v = 8
        bar_h = (v / max_v) * (reg_h * 0.7)
        bar_w = reg_w * 0.5
        ax.add_patch(Rectangle((x + (reg_w - bar_w)/2, y + reg_h * 0.15),
                               bar_w, bar_h,
                               facecolor=HLL_BORDER, edgecolor='none',
                               alpha=0.85, zorder=3))
        ax.text(x + reg_w / 2, y + reg_h * 0.93, str(v),
                fontsize=7.5, ha='center', va='top',
                color=TEXT_COLOR, fontfamily='monospace', zorder=3)

# Caption
ax.text(p3x + PANEL_WIDTH/2, PANEL_Y + 1.1,
        'm registers of leading-zeros',
        fontsize=10, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(p3x + PANEL_WIDTH/2, PANEL_Y + 0.65,
        '~12 KB → 1% error on 10⁹ items',
        fontsize=9, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
ax.text(8, 0.3, 'Algorithms in Python  |  Foundations Part 10',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/00-foundations/10-probabilistic-data-structures/header_pds.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
