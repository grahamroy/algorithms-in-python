"""Generate the header image for the Neural Collaborative Filtering
article --- the NeuMF architecture: two embedding towers (a GMF
element-wise-product tower and a concatenate-then-MLP tower) fused
into a single predicted interaction probability."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

USER_FILL = '#dbeafe'
USER_BORDER = '#3B82F6'
ITEM_FILL = '#dcfce7'
ITEM_BORDER = '#16A34A'
GMF_FILL = '#fef3c7'
GMF_BORDER = '#F59E0B'
MLP_FILL = '#ede9fe'
MLP_BORDER = '#8B5CF6'
OUT_FILL = '#fee2e2'
OUT_BORDER = '#EF4444'
ARROW = '#94A3B8'


fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.5, 'Neural Collaborative Filtering: a learned interaction function',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.0,
        'Replace the fixed dot product u·v with a trained network. NeuMF fuses a GMF tower and an MLP tower.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

ax.add_patch(FancyBboxPatch((0.4, 0.7), 15.2, 6.7,
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))


def box(cx, cy, w, h, text, fill, border, fs=11, fw='bold', tcolor=None):
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                boxstyle='round,pad=0.02,rounding_size=0.08',
                                facecolor=fill, edgecolor=border,
                                linewidth=1.5, zorder=3))
    ax.text(cx, cy, text, fontsize=fs, fontweight=fw, ha='center',
            va='center', color=tcolor or TEXT_COLOR,
            fontfamily='sans-serif', zorder=4)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle='-|>', mutation_scale=12,
                                 color=ARROW, linewidth=1.4, zorder=2))


# Column x-centres: GMF tower on the left, MLP tower on the right.
GX_U, GX_I = 3.4, 5.6      # GMF user / item embeddings
MX_U, MX_I = 9.2, 11.4     # MLP user / item embeddings

# Row 1 (bottom): inputs
y_in = 1.4
box(4.5, y_in, 2.2, 0.7, 'User  i', USER_FILL, USER_BORDER)
box(10.3, y_in, 2.2, 0.7, 'Item  j', ITEM_FILL, ITEM_BORDER)

# Row 2: embeddings
y_emb = 2.7
box(GX_U, y_emb, 1.9, 0.7, 'pᵢᴳ', USER_FILL, USER_BORDER, fs=12)
box(GX_I, y_emb, 1.9, 0.7, 'qⱼᴳ', ITEM_FILL, ITEM_BORDER, fs=12)
box(MX_U, y_emb, 1.9, 0.7, 'pᵢᴹ', USER_FILL, USER_BORDER, fs=12)
box(MX_I, y_emb, 1.9, 0.7, 'qⱼᴹ', ITEM_FILL, ITEM_BORDER, fs=12)
ax.text((GX_U + GX_I) / 2, y_emb + 0.65, 'GMF embeddings',
        fontsize=9, ha='center', color=GMF_BORDER, fontstyle='italic')
ax.text((MX_U + MX_I) / 2, y_emb + 0.65, 'MLP embeddings',
        fontsize=9, ha='center', color=MLP_BORDER, fontstyle='italic')

arrow(4.5, y_in + 0.4, GX_U, y_emb - 0.4)
arrow(4.5, y_in + 0.4, MX_U, y_emb - 0.4)
arrow(10.3, y_in + 0.4, GX_I, y_emb - 0.4)
arrow(10.3, y_in + 0.4, MX_I, y_emb - 0.4)

# Row 3: tower operations
y_op = 4.0
box((GX_U + GX_I) / 2, y_op, 2.6, 0.75,
    'element-wise  ⊙', GMF_FILL, GMF_BORDER, fs=11)
box((MX_U + MX_I) / 2, y_op, 2.6, 0.75,
    'concatenate', MLP_FILL, MLP_BORDER, fs=11)
arrow(GX_U, y_emb + 0.4, (GX_U + GX_I) / 2 - 0.4, y_op - 0.4)
arrow(GX_I, y_emb + 0.4, (GX_U + GX_I) / 2 + 0.4, y_op - 0.4)
arrow(MX_U, y_emb + 0.4, (MX_U + MX_I) / 2 - 0.4, y_op - 0.4)
arrow(MX_I, y_emb + 0.4, (MX_U + MX_I) / 2 + 0.4, y_op - 0.4)

# Row 4: MLP hidden layers (GMF tower passes straight up)
y_h = 5.25
box((GX_U + GX_I) / 2, y_h, 2.6, 0.7, 'φ_GMF', GMF_FILL, GMF_BORDER, fs=11)
box((MX_U + MX_I) / 2, y_h, 2.6, 0.7,
    'ReLU MLP layers', MLP_FILL, MLP_BORDER, fs=10.5)
arrow((GX_U + GX_I) / 2, y_op + 0.38, (GX_U + GX_I) / 2, y_h - 0.35)
arrow((MX_U + MX_I) / 2, y_op + 0.38, (MX_U + MX_I) / 2, y_h - 0.35)

# Row 5: fusion + output
y_fuse = 6.5
box(8, y_fuse, 4.4, 0.8,
    'fuse  →  σ( hᵀ[ φ_GMF ; φ_MLP ] )', OUT_FILL, OUT_BORDER,
    fs=11.5, tcolor=OUT_BORDER)
arrow((GX_U + GX_I) / 2, y_h + 0.36, 8 - 1.4, y_fuse - 0.42)
arrow((MX_U + MX_I) / 2, y_h + 0.36, 8 + 1.4, y_fuse - 0.42)

ax.text(8, y_fuse + 0.62, 'ŷᵢⱼ  =  P(user i interacts with item j)',
        fontsize=11, ha='center', va='center', fontstyle='italic',
        color=TEXT_COLOR, fontfamily='sans-serif')

ax.text(8, 0.3,
        'Algorithms in Python  |  Recommender Systems Part 2',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '07-recommender-systems/'
       '02-neural-collaborative-filtering/header_ncf.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
