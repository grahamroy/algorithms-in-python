"""Generate the header image for the LDA article."""

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

TOPIC_COLORS = ['#F59E0B', '#3B82F6', '#16A34A']  # cooking, ml, sports
TOPIC_FILLS = ['#fef3c7', '#dbeafe', '#dcfce7']
DOC_BORDER = '#9CA3AF'
DOC_FILL = '#F1F5F9'

fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45, 'LDA: Documents Are Mixtures of Topics, Topics Are Distributions Over Words',
        fontsize=16, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'Generative model: sample a topic mixture per doc, then per word sample a topic, then a word from that topic.',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# Three topics on the right
topic_names = ['cooking', 'ml', 'sports']
topic_x = PANEL[0] + PANEL[2] - 3.5
topic_top = PANEL[1] + PANEL[3] - 1.2
topic_h = 1.0

# Per-topic top words
topic_words = [
    'recipe, flour, oven,\nbutter, bake, egg',
    'model, train, neural,\nloss, layer, weight',
    'goal, match, team,\nscore, league, ball'
]

ax.text(topic_x + 1.3, topic_top + 0.4, 'Topics (φ)',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

for k in range(3):
    ty = topic_top - k * (topic_h + 0.5)
    ax.add_patch(Rectangle((topic_x, ty - topic_h/2),
                            2.6, topic_h,
                            facecolor=TOPIC_FILLS[k],
                            edgecolor=TOPIC_COLORS[k],
                            linewidth=1.8, zorder=2))
    ax.text(topic_x + 0.15, ty + 0.25, f"topic {k}: {topic_names[k]}",
            fontsize=10, fontweight='bold',
            ha='left', va='center',
            color=TOPIC_COLORS[k], fontfamily='sans-serif')
    ax.text(topic_x + 0.15, ty - 0.18, topic_words[k],
            fontsize=8, ha='left', va='center',
            color=TEXT_COLOR, fontfamily='monospace')

# Documents on the left, each shown as a mixture (colored bar)
doc_x = PANEL[0] + 0.6
doc_top = topic_top
doc_w = 4.0
doc_h = 0.55
doc_gap = 0.25

ax.text(doc_x + doc_w/2, doc_top + 0.4, 'Document mixtures (θ)',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# Mock 6 documents with different mixtures
mixtures = [
    [0.95, 0.02, 0.03],
    [0.94, 0.03, 0.03],
    [0.03, 0.94, 0.03],
    [0.03, 0.94, 0.03],
    [0.03, 0.03, 0.94],
    [0.03, 0.03, 0.94],
]
for i, mix in enumerate(mixtures):
    dy = doc_top - i * (doc_h + doc_gap)
    # Background
    ax.add_patch(Rectangle((doc_x, dy - doc_h/2), doc_w, doc_h,
                            facecolor='white', edgecolor=DOC_BORDER,
                            linewidth=1.0, zorder=2))
    # Colored segments by mixture
    xstart = doc_x
    for k, frac in enumerate(mix):
        w = frac * doc_w
        ax.add_patch(Rectangle((xstart, dy - doc_h/2), w, doc_h,
                                facecolor=TOPIC_FILLS[k],
                                edgecolor='none', alpha=0.95, zorder=3))
        xstart += w
    # Doc label
    ax.text(doc_x - 0.15, dy, f"doc {i}",
            fontsize=9, ha='right', va='center',
            color=TEXT_COLOR, fontfamily='monospace')

# Arrows from documents to topics (illustrate the generative process)
arrow_x_start = doc_x + doc_w + 0.2
arrow_x_end = topic_x - 0.2
mid_x = (arrow_x_start + arrow_x_end) / 2
mid_y = doc_top - 1.8
ax.annotate('', xy=(arrow_x_end, mid_y),
            xytext=(arrow_x_start, mid_y),
            arrowprops=dict(arrowstyle='->', color='#475569',
                            lw=1.5))
ax.text(mid_x, mid_y + 0.3,
        'generative\nstory',
        fontsize=10, fontstyle='italic',
        ha='center', va='center',
        color='#475569', fontfamily='sans-serif')

ax.text(8, 0.3,
        'Algorithms in Python  |  Advanced Unsupervised Learning Part 5',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '04-advanced-unsupervised-learning/'
       '05-latent-dirichlet-allocation/header_lda.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
