import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# ── Colours ──
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

SPAM_FILL = '#fee2e2'
SPAM_BORDER = '#DC2626'
HAM_FILL = '#dcfce7'
HAM_BORDER = '#16a34a'
ARROW_COLOR = '#475569'
NEUTRAL_FILL = '#dbeafe'
NEUTRAL_BORDER = '#3b82f6'

fig, ax = plt.subplots(1, 1, figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ── Title ──
ax.text(8, 8.45, 'Naive Bayes: Counting Plus Bayes\u2019 Rule',
        fontsize=18, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')

# ── Subtitle ──
ax.text(8, 7.95,
        'A clearly false assumption, a single pass through the data, a baseline that won\u2019t go away',
        fontsize=11, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# Two side-by-side panels
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
# LEFT PANEL --- The probabilistic decision
# ═══════════════════════════════════════════════════════════
lpx, lpy, lpw, lph = LEFT_PANEL

ax.text(lpx + lpw/2, lpy + lph - 0.4, 'Bayes\u2019 rule on a message',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(lpx + lpw/2, lpy + lph - 0.85,
        '\u0177 = argmax\u2096  log P(y) + \u03a3\u1d62 log P(x\u1d62 | y)',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='monospace')

# Show a message at top, with tokens flowing down to two class scores
msg_text = '"free entry to win cash now"'
msg_y = lpy + lph - 1.55
msg_w = 4.6
msg_h = 0.55
msg_x = lpx + (lpw - msg_w) / 2
ax.add_patch(FancyBboxPatch((msg_x, msg_y - msg_h/2), msg_w, msg_h,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            facecolor=NEUTRAL_FILL, edgecolor=NEUTRAL_BORDER,
                            linewidth=1.4, zorder=2))
ax.text(lpx + lpw/2, msg_y, msg_text,
        fontsize=11, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='monospace', zorder=3)

# Tokens
tokens = ['free', 'entry', 'win', 'cash', 'now']
n_tok = len(tokens)
tok_y = lpy + 3.05
tok_w = 0.85
tok_gap = 0.05
total_tok_w = n_tok * tok_w + (n_tok - 1) * tok_gap
tok_x0 = lpx + (lpw - total_tok_w) / 2
for i, t in enumerate(tokens):
    tx = tok_x0 + i * (tok_w + tok_gap)
    ax.add_patch(Rectangle((tx, tok_y - 0.2), tok_w, 0.4,
                           facecolor='white', edgecolor=NEUTRAL_BORDER,
                           linewidth=1.0, zorder=2))
    ax.text(tx + tok_w/2, tok_y, t,
            fontsize=9, ha='center', va='center',
            color=TEXT_COLOR, fontfamily='monospace', zorder=3)

# Arrow from message down to tokens
mid_x = lpx + lpw/2
ax.annotate('', xy=(mid_x, tok_y + 0.25),
            xytext=(mid_x, msg_y - msg_h/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.2))
ax.text(mid_x + 0.18, (tok_y + 0.25 + msg_y - msg_h/2 - 0.05) / 2,
        'tokenise',
        fontsize=8, fontstyle='italic', ha='left', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# Two class score boxes at the bottom
score_y = lpy + 1.25
score_w = 2.7
score_h = 0.95
spam_x = lpx + 0.6
ham_x = lpx + lpw - 0.6 - score_w

for (sx, label, fill, border, score) in [
    (spam_x, 'spam', SPAM_FILL, SPAM_BORDER, '-7.84'),
    (ham_x, 'ham',  HAM_FILL,  HAM_BORDER,  '-12.41'),
]:
    ax.add_patch(FancyBboxPatch((sx, score_y - score_h/2), score_w, score_h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fill, edgecolor=border,
                                linewidth=1.6, zorder=2))
    ax.text(sx + score_w/2, score_y + 0.18, label,
            fontsize=11, fontweight='bold', ha='center', va='center',
            color=border, fontfamily='sans-serif', zorder=3)
    ax.text(sx + score_w/2, score_y - 0.18, f'log score = {score}',
            fontsize=9, ha='center', va='center',
            color=TEXT_COLOR, fontfamily='monospace', zorder=3)

# Arrow from tokens to each score; highlight the winner
for sx in [spam_x, ham_x]:
    ax.annotate('', xy=(sx + score_w/2, score_y + score_h/2 + 0.05),
                xytext=(mid_x, tok_y - 0.25),
                arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=1.0,
                                alpha=0.6))

# Winner annotation
ax.text(lpx + lpw/2, lpy + 0.5,
        'argmax \u2192 spam (higher log score)',
        fontsize=10, fontweight='bold', ha='center', va='center',
        color=SPAM_BORDER, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# RIGHT PANEL --- Per-class log-likelihood bars
# ═══════════════════════════════════════════════════════════
rpx, rpy, rpw, rph = RIGHT_PANEL

ax.text(rpx + rpw/2, rpy + rph - 0.4, 'Per-token evidence',
        fontsize=13, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(rpx + rpw/2, rpy + rph - 0.85,
        'log P(token | spam) - log P(token | ham)',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='monospace')

# Pick four spam-leaning tokens and four ham-leaning tokens
items = [
    ('free',      +1.92),
    ('urgent',    +1.36),
    ('click',     +1.36),
    ('cash',      +1.07),
    ('thanks',    -1.13),
    ('home',      -1.13),
    ('the',       -1.64),
    ('at',        -1.41),
]
items.sort(key=lambda kv: -kv[1])  # most spam at top

# Plot box
plot_left = rpx + 1.6
plot_right = rpx + rpw - 0.6
plot_centre = (plot_left + plot_right) / 2
plot_top = rpy + rph - 1.4
plot_bottom = rpy + 1.0
n_items = len(items)
bar_h = 0.40
row_step = (plot_top - plot_bottom) / (n_items + 1)

# Centre vertical line
ax.plot([plot_centre, plot_centre], [plot_bottom, plot_top],
        color='#CBD5E1', linewidth=1.0, zorder=1)

# Max abs ratio sets the scale
max_abs = max(abs(r) for _, r in items)
half_width = (plot_right - plot_left) / 2 - 0.4

for i, (tok, ratio) in enumerate(items):
    y = plot_top - (i + 1) * row_step
    bar_w = (abs(ratio) / max_abs) * half_width
    if ratio >= 0:
        x0 = plot_centre
        fill = SPAM_FILL
        border = SPAM_BORDER
    else:
        x0 = plot_centre - bar_w
        fill = HAM_FILL
        border = HAM_BORDER
    ax.add_patch(Rectangle((x0, y - bar_h/2), bar_w, bar_h,
                           facecolor=fill, edgecolor=border,
                           linewidth=1.2, zorder=2))
    # Token label on the LEFT margin (consistent for every row)
    ax.text(plot_left - 0.15, y, tok,
            fontsize=10, ha='right', va='center',
            color=TEXT_COLOR, fontfamily='monospace', zorder=3)
    # Numeric value at the end of the bar
    if ratio >= 0:
        ax.text(x0 + bar_w + 0.08, y, f'{ratio:+.2f}',
                fontsize=9, ha='left', va='center',
                color=SPAM_BORDER, fontfamily='monospace', zorder=3)
    else:
        ax.text(x0 - 0.08, y, f'{ratio:+.2f}',
                fontsize=9, ha='right', va='center',
                color=HAM_BORDER, fontfamily='monospace', zorder=3)

# Side labels at the top of the chart area
ax.text(plot_centre - 0.15, plot_top + 0.1, '\u2190 ham',
        fontsize=9, fontweight='bold', ha='right', va='center',
        color=HAM_BORDER, fontfamily='sans-serif')
ax.text(plot_centre + 0.15, plot_top + 0.1, 'spam \u2192',
        fontsize=9, fontweight='bold', ha='left', va='center',
        color=SPAM_BORDER, fontfamily='sans-serif')

# Caption
ax.text(rpx + rpw/2, rpy + 0.5,
        'The classifier is just the sum of these per-token contributions',
        fontsize=10, ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
ax.text(8, 0.3, 'Algorithms in Python  |  Supervised Learning Part 3',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = 'D:/Projects/Medium/algorithms-in-python/01-supervised-learning/03-naive-bayes/header_naive_bayes.png'
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
