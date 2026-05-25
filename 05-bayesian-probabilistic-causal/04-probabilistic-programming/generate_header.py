"""Generate the header image for the Probabilistic Programming article."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np


BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
PANEL_BG = '#F8FAFC'
PANEL_EDGE = '#E2E8F0'

MODEL_FILL = '#dbeafe'
MODEL_BORDER = '#3B82F6'
ENGINE_FILL = '#fef3c7'
ENGINE_BORDER = '#F59E0B'
POST_FILL = '#dcfce7'
POST_BORDER = '#16A34A'
ARROW_COLOR = '#475569'


fig, ax = plt.subplots(figsize=(1600/150, 900/150), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG_COLOR)

ax.text(8, 8.45,
        'Probabilistic Programming: Declare the Model, Let the Engine Pick the Sampler',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 7.95,
        'PyMC, Stan, NumPyro, Pyro — write priors + likelihood + data; the PPL handles compilation, NUTS, and diagnostics.',
        fontsize=10.5, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

PANEL = (0.4, 0.9, 15.2, 6.0)
ax.add_patch(FancyBboxPatch((PANEL[0], PANEL[1]), PANEL[2], PANEL[3],
                            boxstyle='round,pad=0.02,rounding_size=0.15',
                            facecolor=PANEL_BG, edgecolor=PANEL_EDGE,
                            linewidth=1.2, zorder=0))

# Three boxes: Model → Engine → Posterior
box_y = PANEL[1] + PANEL[3] / 2
box_h = 3.4

# Model definition box
mx = PANEL[0] + 0.8
mw = 5.0
ax.add_patch(FancyBboxPatch((mx, box_y - box_h/2), mw, box_h,
                            boxstyle='round,pad=0.02,rounding_size=0.12',
                            facecolor=MODEL_FILL, edgecolor=MODEL_BORDER,
                            linewidth=2.0, zorder=2))
ax.text(mx + mw/2, box_y + box_h/2 - 0.35,
        'Model (PyMC)',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=MODEL_BORDER, fontfamily='sans-serif')

model_code = """with pm.Model() as m:
    a = pm.Normal("a", 0, 10)
    b = pm.Normal("b", 0, 10)
    s = pm.HalfNormal("s", 1)
    pm.Normal(
        "y", mu=a + b*x,
        sigma=s, observed=y)"""
ax.text(mx + 0.2, box_y - 0.2, model_code,
        fontsize=9.5, ha='left', va='center',
        color=TEXT_COLOR, fontfamily='monospace')

# Engine box
ex = PANEL[0] + 6.7
ew = 2.8
ax.add_patch(FancyBboxPatch((ex, box_y - box_h/2), ew, box_h,
                            boxstyle='round,pad=0.02,rounding_size=0.12',
                            facecolor=ENGINE_FILL, edgecolor=ENGINE_BORDER,
                            linewidth=2.0, zorder=2))
ax.text(ex + ew/2, box_y + box_h/2 - 0.35,
        'Engine',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=ENGINE_BORDER, fontfamily='sans-serif')
engine_lines = [
    'compile graph',
    'auto-diff',
    'pick sampler (NUTS)',
    '4 chains in parallel',
    'tune + adapt',
    'sample',
]
for i, line in enumerate(engine_lines):
    ax.text(ex + ew/2, box_y + 0.7 - i * 0.35, line,
            fontsize=9.5, ha='center', va='center',
            color=TEXT_COLOR, fontfamily='monospace')

# Posterior box
px = PANEL[0] + 10.4
pw = 4.0
ax.add_patch(FancyBboxPatch((px, box_y - box_h/2), pw, box_h,
                            boxstyle='round,pad=0.02,rounding_size=0.12',
                            facecolor=POST_FILL, edgecolor=POST_BORDER,
                            linewidth=2.0, zorder=2))
ax.text(px + pw/2, box_y + box_h/2 - 0.35,
        'Posterior',
        fontsize=12, fontweight='bold', ha='center', va='center',
        color=POST_BORDER, fontfamily='sans-serif')

post_lines = [
    'a ~ N(1.97, 0.035)',
    'b ~ N(0.50, 0.020)',
    's ~ N(0.49, 0.025)',
    '',
    'R-hat = 1.00',
    'ESS  ≈ 5000',
]
for i, line in enumerate(post_lines):
    ax.text(px + 0.2, box_y + 0.7 - i * 0.35, line,
            fontsize=9.5, ha='left', va='center',
            color=TEXT_COLOR, fontfamily='monospace')

# Arrows
for x0, x1 in [(mx + mw, ex), (ex + ew, px)]:
    ax.annotate('', xy=(x1 - 0.05, box_y),
                xytext=(x0 + 0.05, box_y),
                arrowprops=dict(arrowstyle='->',
                                color=ARROW_COLOR, lw=1.8))

ax.text(8, 0.3,
        'Algorithms in Python  |  Bayesian, Probabilistic & Causal Methods Part 4',
        fontsize=8, ha='center', va='center',
        color='#aaaaaa', fontfamily='sans-serif')

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
out = ('D:/Projects/Medium/algorithms-in-python/'
       '05-bayesian-probabilistic-causal/'
       '04-probabilistic-programming/header_ppl.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
