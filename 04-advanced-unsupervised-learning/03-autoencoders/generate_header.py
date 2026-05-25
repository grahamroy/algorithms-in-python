"""Generate the header image for the Autoencoders article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

LAYER_FILL = '#dbeafe'
LAYER_BORDER = '#3B82F6'
BOTTLENECK_FILL = '#fef3c7'
BOTTLENECK_BORDER = '#F59E0B'
ARROW_COLOR = '#475569'

fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

ax.text(8, 8.45, 'Autoencoders: Compress and Reconstruct',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Encoder squeezes input through a bottleneck. Decoder rebuilds it. Loss is reconstruction error.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Single wide panel
PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# Layer architecture: 64 → 32 → 8 → 32 → 64
sizes = [64, 32, 8, 32, 64]
labels = ['input\n(64)', 'hidden\n(32)', 'code\n(8)',
          'hidden\n(32)', 'output\n(64)']
n_layers = len(sizes)

# Horizontal spacing
x_centres = np.linspace(PANEL[0] + 1.6, PANEL[0] + PANEL[2] - 1.6,
                        n_layers)
y_centre = PANEL[1] + PANEL[3] / 2 - 0.2

# Each layer drawn as a column of small boxes
max_size = max(sizes)
column_h = 3.8

for i, (size, label, cx) in enumerate(zip(sizes, labels, x_centres)):
    h = column_h * (size / max_size)
    w = 0.7
    is_bottleneck = (size == min(sizes))
    fc = BOTTLENECK_FILL if is_bottleneck else LAYER_FILL
    ec = BOTTLENECK_BORDER if is_bottleneck else LAYER_BORDER
    ax.add_patch(Rectangle((cx - w/2, y_centre - h/2),
                            w, h,
                            facecolor=fc, edgecolor=ec,
                            linewidth=1.6, zorder=2))
    ax.text(cx, y_centre + h/2 + 0.35, label,
            fontsize=10, fontweight='bold',
            ha='center', va='bottom',
            color=TEXT_COLOR, fontfamily='monospace')

# Arrows between layers
for i in range(n_layers - 1):
    x0 = x_centres[i] + 0.4
    x1 = x_centres[i + 1] - 0.4
    ax.annotate('', xy=(x1, y_centre), xytext=(x0, y_centre),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR,
                                lw=1.3))

# Section labels
encoder_x = (x_centres[0] + x_centres[2]) / 2
decoder_x = (x_centres[2] + x_centres[4]) / 2
ax.text(encoder_x, PANEL[1] + 0.65, 'Encoder',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=LAYER_BORDER, fontfamily='sans-serif')
ax.text(decoder_x, PANEL[1] + 0.65, 'Decoder',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=LAYER_BORDER, fontfamily='sans-serif')

# Reconstruction-loss curve
ax.annotate('', xy=(x_centres[0], PANEL[1] + 0.3),
            xytext=(x_centres[-1], PANEL[1] + 0.3),
            arrowprops=dict(arrowstyle='->',
                            color='#7C3AED', lw=1.0,
                            connectionstyle='arc3,rad=-0.25'))
ax.text((x_centres[0] + x_centres[-1]) / 2,
        PANEL[1] + 1.6,
        'minimise ‖x − x̂‖²',
        fontsize=11, fontweight='bold', ha='center', va='center',
        color='#7C3AED', fontfamily='monospace')

# Bottleneck callout
ax.text(x_centres[2], y_centre - column_h / 2 - 0.6,
        'bottleneck',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=BOTTLENECK_BORDER, fontfamily='sans-serif')

ax.text(8, 0.3,
        'Algorithms in Python  |  Advanced Unsupervised Learning Part 3',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '04-advanced-unsupervised-learning/'
       '03-autoencoders/header_autoencoder.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
