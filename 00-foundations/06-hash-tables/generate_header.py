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

SLOT_FILL_ON = '#dbeafe'
SLOT_BORDER_ON = '#3b82f6'
SLOT_FILL_OFF = '#F3F4F6'
SLOT_BORDER_OFF = '#D1D5DB'

KEY_FILL = '#FEF3C7'
KEY_BORDER = '#F59E0B'

HASH_FILL = '#ede9fe'
HASH_BORDER = '#8b5cf6'

ARROW_COLOR = '#475569'
ACCENT_RED = '#EF4444'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.45, 'Hash Tables: O(1) Lookup',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ── Subtitle ──
ax.text(8, 7.95, 'Keys mapped to indices by a hash function',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# Panel geometry
# ═══════════════════════════════════════════════════════════
# Each panel ~7.2 wide × 6 tall, side-by-side
LEFT_PANEL = (0.4, 0.9, 7.2, 6.0)   # x, y, w, h
RIGHT_PANEL = (8.4, 0.9, 7.2, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle="round,pad=0.02,rounding_size=0.15",
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)

# ═══════════════════════════════════════════════════════════
# LEFT PANEL — Keys → Indices
# ═══════════════════════════════════════════════════════════
lpx, lpy, lpw, lph = LEFT_PANEL

# Panel title
ax.text(lpx + lpw/2, lpy + lph - 0.4, 'Keys \u2192 Indices',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Keys on the left
keys = ['alice', 'bob', 'carol', 'dave', 'eve']
# Target slots (indices 0-7), chosen so bob & eve collide on slot 3
key_slots = {
    'alice': 1,
    'bob':   3,
    'carol': 5,
    'dave':  7,
    'eve':   3,   # collides with bob
}

# Key box geometry (left column)
key_w, key_h = 1.15, 0.55
key_x = lpx + 0.45
key_top = lpy + lph - 1.25
key_gap = 0.85
key_positions = {}
for i, k in enumerate(keys):
    cy = key_top - i * key_gap
    rect = FancyBboxPatch((key_x, cy - key_h/2), key_w, key_h,
                          boxstyle="round,pad=0.02,rounding_size=0.12",
                          facecolor=KEY_FILL, edgecolor=KEY_BORDER,
                          linewidth=1.4, zorder=2)
    ax.add_patch(rect)
    ax.text(key_x + key_w/2, cy, k,
            fontsize=10, ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif')
    key_positions[k] = (key_x + key_w, cy)

# Hash function box (middle)
hash_w, hash_h = 1.25, 0.75
hash_cx = lpx + lpw/2 - 0.2
hash_cy = lpy + lph/2 - 0.1
hash_box = FancyBboxPatch((hash_cx - hash_w/2, hash_cy - hash_h/2), hash_w, hash_h,
                          boxstyle="round,pad=0.02,rounding_size=0.14",
                          facecolor=HASH_FILL, edgecolor=HASH_BORDER,
                          linewidth=1.6, zorder=3)
ax.add_patch(hash_box)
ax.text(hash_cx, hash_cy, 'hash()',
        fontsize=11, fontweight='bold', ha='center', va='center',
        color=HASH_BORDER, fontfamily='sans-serif')

# Array of 8 slots (right column)
n_slots = 8
slot_w, slot_h = 0.7, 0.5
slot_x = lpx + lpw - 0.45 - slot_w
slot_top = lpy + lph - 1.0
slot_gap = 0.60
slot_positions = {}
for i in range(n_slots):
    cy = slot_top - i * slot_gap
    # All slots on left panel shown as grid skeleton; filled colour where a key lands
    used = i in set(key_slots.values())
    fill = SLOT_FILL_ON if used else SLOT_FILL_OFF
    border = SLOT_BORDER_ON if used else SLOT_BORDER_OFF
    rect = Rectangle((slot_x, cy - slot_h/2), slot_w, slot_h,
                     facecolor=fill, edgecolor=border,
                     linewidth=1.4, zorder=2)
    ax.add_patch(rect)
    # Index label to the right of the slot
    ax.text(slot_x + slot_w + 0.22, cy, str(i),
            fontsize=9, ha='left', va='center',
            color=SUBTLE_TEXT, fontfamily='sans-serif')
    slot_positions[i] = (slot_x, cy)

# Arrows: key → hash box (left segment)
for k in keys:
    kx, ky = key_positions[k]
    # target is the left edge of the hash box, aimed at centre
    hx_in = hash_cx - hash_w/2
    arrow = FancyArrowPatch((kx + 0.05, ky), (hx_in - 0.02, hash_cy),
                            arrowstyle='-|>', mutation_scale=10,
                            color=ARROW_COLOR, linewidth=1.1,
                            connectionstyle="arc3,rad=0.02", zorder=1)
    ax.add_patch(arrow)

# Arrows: hash box → slot (right segment)
# Highlight collision arrows for bob and eve in red
hx_out = hash_cx + hash_w/2
for k in keys:
    target_i = key_slots[k]
    sx, sy = slot_positions[target_i]
    is_collision = k in ('bob', 'eve')
    col = ACCENT_RED if is_collision else ARROW_COLOR
    lw = 1.6 if is_collision else 1.1
    # curve the two collision arrows so they clearly converge on slot 3
    if k == 'bob':
        rad = -0.15
    elif k == 'eve':
        rad = 0.18
    else:
        rad = 0.02
    arrow = FancyArrowPatch((hx_out + 0.02, hash_cy), (sx - 0.02, sy),
                            arrowstyle='-|>', mutation_scale=10,
                            color=col, linewidth=lw,
                            connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(arrow)

# Small collision marker near slot 3
sx3, sy3 = slot_positions[3]
ax.text(sx3 - 0.18, sy3 + 0.5, 'collision',
        fontsize=8, fontstyle='italic', ha='right', va='center',
        color=ACCENT_RED, fontfamily='sans-serif')

# Caption below panel
ax.text(lpx + lpw/2, lpy + 0.35, 'hash(key) % size',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# RIGHT PANEL — Separate Chaining
# ═══════════════════════════════════════════════════════════
rpx, rpy, rpw, rph = RIGHT_PANEL

# Panel title
ax.text(rpx + rpw/2, rpy + rph - 0.4, 'Separate Chaining',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Array of 8 slots (left side of right panel)
r_slot_w, r_slot_h = 0.8, 0.55
r_slot_x = rpx + 0.55
r_slot_top = rpy + rph - 1.0
r_slot_gap = 0.65
r_slot_positions = {}
for i in range(n_slots):
    cy = r_slot_top - i * r_slot_gap
    used = i in set(key_slots.values())
    fill = SLOT_FILL_ON if used else SLOT_FILL_OFF
    border = SLOT_BORDER_ON if used else SLOT_BORDER_OFF
    rect = Rectangle((r_slot_x, cy - r_slot_h/2), r_slot_w, r_slot_h,
                     facecolor=fill, edgecolor=border,
                     linewidth=1.4, zorder=2)
    ax.add_patch(rect)
    ax.text(r_slot_x - 0.22, cy, str(i),
            fontsize=9, ha='right', va='center',
            color=SUBTLE_TEXT, fontfamily='sans-serif')
    r_slot_positions[i] = (r_slot_x, cy)

# Single entries in non-collision slots (show the key name inside the slot)
single_entries = {
    1: 'alice',
    5: 'carol',
    7: 'dave',
}
for i, name in single_entries.items():
    sx, sy = r_slot_positions[i]
    ax.text(sx + r_slot_w/2, sy, name,
            fontsize=9, ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif')

# Chain hanging off slot 3 — two rounded boxes connected by an arrow
chain_box_w, chain_box_h = 1.0, 0.5
chain_gap = 0.35
sx3r, sy3r = r_slot_positions[3]
chain_start_x = sx3r + r_slot_w + 0.5
chain_y = sy3r

# Connector arrow from slot 3 to first chain box
conn = FancyArrowPatch((sx3r + r_slot_w + 0.02, sy3r),
                       (chain_start_x - 0.02, chain_y),
                       arrowstyle='-|>', mutation_scale=10,
                       color=ARROW_COLOR, linewidth=1.3, zorder=1)
ax.add_patch(conn)

# First chain node: bob
cb1_x = chain_start_x
cb1 = FancyBboxPatch((cb1_x, chain_y - chain_box_h/2), chain_box_w, chain_box_h,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     facecolor=KEY_FILL, edgecolor=KEY_BORDER,
                     linewidth=1.4, zorder=2)
ax.add_patch(cb1)
ax.text(cb1_x + chain_box_w/2, chain_y, 'bob',
        fontsize=10, ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Arrow between chain nodes
cb2_x = cb1_x + chain_box_w + chain_gap
mid_arrow = FancyArrowPatch((cb1_x + chain_box_w + 0.02, chain_y),
                            (cb2_x - 0.02, chain_y),
                            arrowstyle='-|>', mutation_scale=10,
                            color=ARROW_COLOR, linewidth=1.3, zorder=1)
ax.add_patch(mid_arrow)

# Second chain node: eve
cb2 = FancyBboxPatch((cb2_x, chain_y - chain_box_h/2), chain_box_w, chain_box_h,
                     boxstyle="round,pad=0.02,rounding_size=0.10",
                     facecolor=KEY_FILL, edgecolor=KEY_BORDER,
                     linewidth=1.4, zorder=2)
ax.add_patch(cb2)
ax.text(cb2_x + chain_box_w/2, chain_y, 'eve',
        fontsize=10, ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Highlight the chain slot border in red to flag the collision
sx3r_rect = Rectangle((sx3r, sy3r - r_slot_h/2), r_slot_w, r_slot_h,
                      facecolor='none', edgecolor=ACCENT_RED,
                      linewidth=1.8, zorder=3)
ax.add_patch(sx3r_rect)

# Small annotation above the chain
ax.text(cb1_x + chain_box_w + chain_gap/2, chain_y + 0.55, 'chain',
        fontsize=8, fontstyle='italic', ha='center', va='center',
        color=ACCENT_RED, fontfamily='sans-serif')

# Caption below panel
ax.text(rpx + rpw/2, rpy + 0.35, 'Collisions live in a list per slot',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
ax.text(8, 0.3, 'Algorithms in Python  |  Foundations Part 6',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/00-foundations/06-hash-tables/header_hashtables.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
