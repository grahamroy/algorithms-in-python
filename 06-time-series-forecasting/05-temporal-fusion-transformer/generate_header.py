"""Generate the header image for the TFT article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

VSN_COLOR = '#F59E0B'
LSTM_COLOR = '#3B82F6'
ATTN_COLOR = '#16A34A'
OUTPUT_COLOR = '#7C3AED'
ARROW_COLOR = '#475569'

fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45,
        'Temporal Fusion Transformer: Deep Learning Forecasting at Scale',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'One global model across many related series. LSTM + attention + variable selection + quantile output.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# Block diagram of TFT
box_y = PANEL[1] + PANEL[3] / 2 - 0.4
box_h = 2.4
box_w = 2.4
gap = 0.5

# Inputs on the left
labels_left = [
    ('static covariates', '(store, category)'),
    ('past observed', '(sales, weather)'),
    ('known future', '(promotions, holidays)'),
]
input_y_top = box_y + box_h / 2 + 0.4
for i, (lbl, sublbl) in enumerate(labels_left):
    y = input_y_top - i * 1.0
    ax.text(PANEL[0] + 0.6, y, lbl,
            fontsize=10, fontweight='bold', ha='left', va='center',
            color=TEXT_COLOR)
    ax.text(PANEL[0] + 0.6, y - 0.32, sublbl,
            fontsize=8.5, fontstyle='italic', ha='left', va='center',
            color=SUBTLE_TEXT)

# 4 main blocks
blocks = [
    (3.4, 'Variable\nSelection', 'per-feature gating\n+ importance', VSN_COLOR),
    (6.4, 'LSTM\nEncoder-Decoder', 'local temporal\nstructure', LSTM_COLOR),
    (9.4, 'Multi-Head\nSelf-Attention', 'long-range patterns\n+ interpretability', ATTN_COLOR),
    (12.4, 'Quantile\nOutput', '10/50/90 percentile\nforecasts', OUTPUT_COLOR),
]

for (bx, title, sub, color) in blocks:
    fill = {VSN_COLOR: '#fef3c7', LSTM_COLOR: '#dbeafe',
            ATTN_COLOR: '#dcfce7', OUTPUT_COLOR: '#ede9fe'}[color]
    ax.add_patch(FancyBboxPatch((bx, box_y - box_h/2), box_w, box_h,
                                boxstyle='round,pad=0.02,rounding_size=0.12',
                                facecolor=fill, edgecolor=color,
                                linewidth=2.0, zorder=2))
    ax.text(bx + box_w/2, box_y + 0.3, title,
            fontsize=11, fontweight='bold',
            ha='center', va='center',
            color=color, fontfamily='sans-serif', zorder=3)
    ax.text(bx + box_w/2, box_y - 0.5, sub,
            fontsize=9, fontstyle='italic',
            ha='center', va='center',
            color=TEXT_COLOR, fontfamily='sans-serif', zorder=3)

# Arrows between blocks
for i in range(len(blocks) - 1):
    x0 = blocks[i][0] + box_w
    x1 = blocks[i + 1][0]
    ax.add_patch(FancyArrowPatch((x0 + 0.05, box_y),
                                  (x1 - 0.05, box_y),
                                  arrowstyle='-|>', color=ARROW_COLOR,
                                  mutation_scale=20, lw=1.8, zorder=3))

# Inputs converge into VSN
for i in range(3):
    y = input_y_top - i * 1.0 - 0.05
    ax.add_patch(FancyArrowPatch((PANEL[0] + 2.7, y),
                                  (blocks[0][0] + 0.05, box_y),
                                  arrowstyle='-|>', color=ARROW_COLOR,
                                  mutation_scale=14, lw=1.0,
                                  alpha=0.6,
                                  connectionstyle='arc3,rad=0.2',
                                  zorder=3))

# Output arrow off the right
ax.add_patch(FancyArrowPatch((blocks[-1][0] + box_w + 0.05, box_y),
                              (PANEL[0] + PANEL[2] - 0.2, box_y),
                              arrowstyle='-|>', color=OUTPUT_COLOR,
                              mutation_scale=20, lw=2.0, zorder=3))
ax.text(PANEL[0] + PANEL[2] - 0.15, box_y + 0.4,
        'forecasts\n+ intervals',
        fontsize=10, fontweight='bold',
        ha='right', va='center',
        color=OUTPUT_COLOR, fontfamily='sans-serif')

# Caption
ax.text(8, PANEL[1] + 0.4,
        'Trained globally across n_series simultaneously — cross-series sharing is where TFT beats classical methods.',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

ax.text(8, 0.3,
        'Algorithms in Python  |  Time Series & Forecasting Part 5',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '06-time-series-forecasting/'
       '05-temporal-fusion-transformer/header_tft.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
