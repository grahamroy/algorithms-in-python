"""Generate the header image for the Association Rule Mining article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

TX_BG = '#dbeafe'
TX_BORDER = '#3B82F6'
RULE_BG = '#fef3c7'
RULE_BORDER = '#F59E0B'
LIFT_HIGH = '#16A34A'
LIFT_LOW = '#9CA3AF'


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45,
        'Association Rule Mining: From Baskets to Buy-Together Rules',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Apriori counts frequent itemsets, then turns them into rules sorted by lift.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# --- Two side-by-side panels --------------------------------------------
LEFT_PANEL = (0.4, 0.9, 7.2, 6.0)
RIGHT_PANEL = (8.4, 0.9, 7.2, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle='round,pad=0.02,rounding_size=0.15',
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)


# ========================================================================
# LEFT PANEL: example transactions
# ========================================================================
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4,
        'Transactions: sets of items per basket',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

transactions = [
    ('T1', 'bread, milk, butter, jam'),
    ('T2', 'beer, chips, soda'),
    ('T3', 'bread, eggs, milk'),
    ('T4', 'beer, chips, soda, bread'),
    ('T5', 'bread, butter, jam, eggs'),
    ('T6', 'milk, yogurt'),
    ('T7', 'beer, chips, soda, cheese'),
    ('T8', 'bread, butter, milk'),
]

row_h = 0.55
top_y = lpy + lph - 1.1
left_x = lpx + 0.5
width = lpw - 1.0
for i, (label, items) in enumerate(transactions):
    y = top_y - i * row_h
    ax.add_patch(Rectangle((left_x, y - row_h/2 + 0.05),
                            width, row_h - 0.1,
                            facecolor=TX_BG, edgecolor=TX_BORDER,
                            linewidth=1.0, zorder=2))
    ax.text(left_x + 0.2, y, label,
            fontsize=10, fontweight='bold', ha='left', va='center',
            color=TX_BORDER, fontfamily='monospace')
    ax.text(left_x + 0.9, y, '  ' + items,
            fontsize=10, ha='left', va='center',
            color=TEXT_COLOR, fontfamily='monospace')

ax.text(lpx + lpw/2, lpy + 0.3,
        '50 transactions over 10 items in the companion script.',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# ========================================================================
# RIGHT PANEL: example rules, sorted by lift
# ========================================================================
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4,
        'Rules: sorted by lift',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(rpx + rpw/2, rpy + rph - 0.85,
        'lift > 1 = items co-occur more than chance',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

rules = [
    ('{beer, bread}', '{butter, soda}', 3.03),
    ('{chips, soda}', '{beer}',         2.78),
    ('{butter, soda}', '{beer}',        2.53),
    ('{jam}',         '{bread, butter}', 2.05),
    ('{butter}',      '{bread}',         1.30),
]

top_y = rpy + rph - 1.7
row_h = 0.75
left_x = rpx + 0.5
for i, (a, b, lift) in enumerate(rules):
    y = top_y - i * row_h
    # Antecedent box
    ax.add_patch(Rectangle((left_x, y - row_h/2 + 0.1),
                            2.4, row_h - 0.2,
                            facecolor=RULE_BG, edgecolor=RULE_BORDER,
                            linewidth=1.0, zorder=2))
    ax.text(left_x + 1.2, y, a,
            fontsize=9, ha='center', va='center',
            color=TEXT_COLOR, fontfamily='monospace')
    # Arrow
    arr = FancyArrowPatch((left_x + 2.5, y), (left_x + 3.05, y),
                          arrowstyle='->', color=TEXT_COLOR, lw=1.2)
    ax.add_patch(arr)
    # Consequent box
    ax.add_patch(Rectangle((left_x + 3.15, y - row_h/2 + 0.1),
                            2.4, row_h - 0.2,
                            facecolor=RULE_BG, edgecolor=RULE_BORDER,
                            linewidth=1.0, zorder=2))
    ax.text(left_x + 4.35, y, b,
            fontsize=9, ha='center', va='center',
            color=TEXT_COLOR, fontfamily='monospace')
    # Lift score
    lift_color = LIFT_HIGH if lift >= 2.0 else LIFT_LOW
    ax.text(left_x + 6.0, y, f'lift {lift:.2f}',
            fontsize=10, fontweight='bold', ha='left', va='center',
            color=lift_color, fontfamily='monospace')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Unsupervised Learning Part 8',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '03-unsupervised-learning/'
       '08-association-rule-mining/header_apriori.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
