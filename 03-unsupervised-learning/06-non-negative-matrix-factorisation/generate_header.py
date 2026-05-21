"""Generate the header image for the NMF article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


# --- Colours -------------------------------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

X_FILL = '#dbeafe'
X_BORDER = '#3B82F6'
W_FILL = '#fef3c7'
W_BORDER = '#F59E0B'
H_FILL = '#dcfce7'
H_BORDER = '#16A34A'


# --- Figure --------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# --- Title and subtitle --------------------------------------------------
ax.text(8, 8.45, 'NMF: Decompose Non-Negative Data Into Non-Negative Parts',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'X ≈ W · H with W ≥ 0 and H ≥ 0. The constraint forces an additive, parts-based decomposition.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# --- Two side-by-side panels --------------------------------------------
LEFT_PANEL = (0.4, 0.9, 8.0, 6.0)
RIGHT_PANEL = (8.6, 0.9, 7.0, 6.0)

for (px, py, pw, ph) in [LEFT_PANEL, RIGHT_PANEL]:
    panel = FancyBboxPatch((px, py), pw, ph,
                           boxstyle='round,pad=0.02,rounding_size=0.15',
                           facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                           linewidth=1.2, zorder=0)
    ax.add_patch(panel)


# ========================================================================
# LEFT PANEL: the matrix equation X ≈ W · H visualised
# ========================================================================
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4,
        'The decomposition',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Geometry: X (tall thin), ≈ , W (tall narrow), · , H (short wide)
centre_y = lpy + lph / 2 - 0.4

# X
x_box_w = 1.5
x_box_h = 2.6
x_x = lpx + 0.7
x_y = centre_y - x_box_h / 2
ax.add_patch(Rectangle((x_x, x_y), x_box_w, x_box_h,
                       facecolor=X_FILL, edgecolor=X_BORDER,
                       linewidth=1.6, zorder=2))
ax.text(x_x + x_box_w/2, x_y + x_box_h/2, 'X',
        fontsize=22, fontweight='bold', ha='center', va='center',
        color=X_BORDER, fontfamily='serif', zorder=3)
ax.text(x_x + x_box_w/2, x_y - 0.25,
        f'n × d', fontsize=10, ha='center', va='top',
        color=SUBTLE_TEXT, fontfamily='monospace')
ax.text(x_x + x_box_w/2, x_y + x_box_h + 0.15,
        'data', fontsize=10, ha='center', va='bottom',
        color=X_BORDER, fontfamily='sans-serif')

# ≈
ax.text(x_x + x_box_w + 0.5, centre_y, '≈',
        fontsize=24, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='serif')

# W
w_box_w = 0.85
w_box_h = 2.6
w_x = x_x + x_box_w + 1.0
w_y = centre_y - w_box_h / 2
ax.add_patch(Rectangle((w_x, w_y), w_box_w, w_box_h,
                       facecolor=W_FILL, edgecolor=W_BORDER,
                       linewidth=1.6, zorder=2))
ax.text(w_x + w_box_w/2, w_y + w_box_h/2, 'W',
        fontsize=22, fontweight='bold', ha='center', va='center',
        color=W_BORDER, fontfamily='serif', zorder=3)
ax.text(w_x + w_box_w/2, w_y - 0.25,
        f'n × k', fontsize=10, ha='center', va='top',
        color=SUBTLE_TEXT, fontfamily='monospace')
ax.text(w_x + w_box_w/2, w_y + w_box_h + 0.15,
        'encodings', fontsize=10, ha='center', va='bottom',
        color=W_BORDER, fontfamily='sans-serif')

# ·
ax.text(w_x + w_box_w + 0.4, centre_y, '·',
        fontsize=30, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='serif')

# H
h_box_w = 2.4
h_box_h = 0.85
h_x = w_x + w_box_w + 0.85
h_y = centre_y - h_box_h / 2
ax.add_patch(Rectangle((h_x, h_y), h_box_w, h_box_h,
                       facecolor=H_FILL, edgecolor=H_BORDER,
                       linewidth=1.6, zorder=2))
ax.text(h_x + h_box_w/2, h_y + h_box_h/2, 'H',
        fontsize=22, fontweight='bold', ha='center', va='center',
        color=H_BORDER, fontfamily='serif', zorder=3)
ax.text(h_x + h_box_w/2, h_y - 0.25,
        f'k × d', fontsize=10, ha='center', va='top',
        color=SUBTLE_TEXT, fontfamily='monospace')
ax.text(h_x + h_box_w/2, h_y + h_box_h + 0.15,
        'components', fontsize=10, ha='center', va='bottom',
        color=H_BORDER, fontfamily='sans-serif')

# Constraints reminder under the equation
ax.text(lpx + lpw/2, lpy + 0.6,
        'All entries of W and H are constrained to be ≥ 0',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# ========================================================================
# RIGHT PANEL: example topics discovered by NMF
# ========================================================================
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4,
        'Each row of H is an interpretable "part"',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(rpx + rpw/2, rpy + rph - 0.85,
        'On 20-newsgroups TF-IDF, the rows of H look like topics',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

topics = [
    ('Topic 0', '#3B82F6',
     'people · think · religion · say · know'),
    ('Topic 1', '#F59E0B',
     'car · cars · engine · dealer · price'),
    ('Topic 2', '#16A34A',
     'god · jesus · faith · christ · bible'),
    ('Topic 3', '#DC2626',
     'team · game · runs · hit · pitching'),
]

top_y = rpy + rph - 1.6
row_h = 0.95
for i, (name, colour, words) in enumerate(topics):
    y = top_y - i * row_h
    # Label box
    ax.add_patch(Rectangle((rpx + 0.5, y - row_h/2 + 0.1),
                            1.5, row_h - 0.2,
                            facecolor=colour + '22',
                            edgecolor=colour, linewidth=1.4))
    ax.text(rpx + 1.25, y, name,
            fontsize=10, fontweight='bold', ha='center', va='center',
            color=colour, fontfamily='sans-serif')
    # Words
    ax.text(rpx + 2.3, y, words,
            fontsize=10, ha='left', va='center',
            color=TEXT_COLOR, fontfamily='monospace')

ax.text(rpx + rpw/2, rpy + 0.4,
        'Add up these "parts" with non-negative weights → reconstruct a document.',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


# --- Footer --------------------------------------------------------------
ax.text(8, 0.3,
        'Algorithms in Python  |  Unsupervised Learning Part 6',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '03-unsupervised-learning/'
       '06-non-negative-matrix-factorisation/header_nmf.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
