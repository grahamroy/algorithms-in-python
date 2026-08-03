"""Generate the header image for the RNN/LSTM article.

Right panels: the measured distance cliff (accuracies from rnn.py's
frozen DEMO 1/2 output) and the full gradient-reach curves at
initialisation, recomputed live from the same seeded models.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import rnn as R

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# verified from rnn.py output (DEMO 1 control + DEMO 2 sweep)
DISTS = [20, 30, 40]
RNN_ACC = [100.0, 100.0, 24.0]
LSTM_ACC = [100.0, 98.8, 100.0]
CONTROL = 99.5                       # T=40, answer last (distance 0)

# live: gradient reach at init on the T=40 recall task (no training)
X, y = R.make_recall(R.N_SEQ, 40, np.random.default_rng(140))
_, nr = R.RNN(R.VOCAB, R.HIDDEN, R.VOCAB, seed=0).grads(X, y, want_reach=True)
_, nl = R.LSTM(R.VOCAB, R.HIDDEN, R.VOCAB, seed=0).grads(X, y,
                                                         want_reach=True)
print(f'reach at t=1: RNN {nr[0]:.1e}, LSTM {nl[0]:.1e}')

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Recurrent Neural Networks: The Wall Is Distance, Not Length',
        fontsize=17.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'One cell shared across time — perfect recall through 30 steps of '
        'noise, chance at 40, and the gated highway that survives the trip.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.15, 5.3, 1.45, BLUE)
ax.text(3.35, 7.32, 'weight sharing, pointed at time', fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=BLUE)
ax.text(3.35, 6.78, 'h_t = tanh( x_t Wx + h_(t-1) Wh + b )\n'
                    'same cell every step: 796 parameters,\n'
                    'any sequence length',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 4.45, 5.3, 1.3, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 5.5, "Part 1's villain, applied T times", fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 4.98, 'every step of DISTANCE multiplies the error\n'
                    'by the same squashing Jacobian — 40 steps,\n'
                    '~300x attenuation before training starts',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 2.6, 5.3, 1.45, GREEN)
ax.text(3.35, 3.78, 'the LSTM: an additive highway', fontsize=9.8,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 3.2, 'c_t = f * c_(t-1) + i * g\n'
                   'backwards, the cell path multiplies by f —\n'
                   'a learned number near 1, not a matrix',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 1.05, 5.3, 1.15, PURPLE, face=BADGE_BG, lw=1.6)
ax.text(3.35, 1.95, 'the honest boundary', fontsize=9.4,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 1.5, '40-bit parity defeats BOTH cells (~50%) —\n'
                   'gates cure vanishing, not every long horizon',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the cliff =====================
axc = fig.add_axes([0.47, 0.525, 0.50, 0.27])
axc.plot(DISTS, RNN_ACC, 'o-', color=ORANGE, linewidth=2.2, markersize=7,
         label='vanilla RNN')
axc.plot(DISTS, LSTM_ACC, 's-', color=BLUE, linewidth=2.2, markersize=6,
         label='LSTM')
axc.scatter([40], [CONTROL], marker='*', s=150, color=GREEN, zorder=5)
axc.annotate('T=40 control: answer at the\nEND (distance 0) — RNN 99.5%',
             xy=(39.6, CONTROL), xytext=(24.8, 86), fontsize=7.8,
             color=GREEN, ha='center',
             arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.0))
axc.annotate('the cliff:\n100% at 30,\nchance at 40', xy=(40, 24),
             xytext=(33.8, 45), fontsize=8.2, color=RED, ha='center',
             fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.1))
axc.axhline(25, color='#CBD5E1', linewidth=1, linestyle=':')
axc.text(20.2, 27, 'chance (4 classes)', fontsize=7.2, color=SUBTLE_TEXT)
axc.set_xticks(DISTS)
axc.set_xlabel('distance T between answer and question', fontsize=8.5,
               color=SUBTLE_TEXT)
axc.set_ylabel('test accuracy (%)', fontsize=8.5, color=SUBTLE_TEXT)
axc.set_ylim(15, 106)
axc.legend(fontsize=8, loc='center left', frameon=False)
axc.set_title('recall through T steps of noise (real runs)',
              fontsize=9.5, fontweight='bold', color=TEXT_COLOR, pad=5)
for sp in ('top', 'right'):
    axc.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axc.spines[sp].set_color('#CBD5E1')
axc.tick_params(colors=SUBTLE_TEXT, labelsize=8)

# ===================== RIGHT BOTTOM: gradient reach =====================
axr = fig.add_axes([0.47, 0.115, 0.50, 0.29])
ts = np.arange(1, len(nr) + 1)
axr.semilogy(ts, nr, color=ORANGE, linewidth=2.2, label='vanilla RNN')
axr.semilogy(ts, nl, color=BLUE, linewidth=2.2, label='LSTM')
axr.annotate('arrives ~10x stronger,\nwith UNTRAINED gates',
             xy=(1, nl[0]), xytext=(6.4, 4.5e-3), fontsize=7.9, color=BLUE,
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))
axr.annotate('~300x attenuation', xy=(1, nr[0]), xytext=(7.5, 2.4e-5),
             fontsize=7.9, color=ORANGE,
             arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.0))
axr.set_xlabel('timestep t the error signal has reached (T = 40, at init)',
               fontsize=8.5, color=SUBTLE_TEXT)
axr.set_ylabel('gradient norm', fontsize=8.5, color=SUBTLE_TEXT)
axr.legend(fontsize=8, loc='lower right', frameon=False)
axr.set_title('the mechanism, before any training (live measurement)',
              fontsize=9.5, fontweight='bold', color=TEXT_COLOR, pad=5)
for sp in ('top', 'right'):
    axr.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axr.spines[sp].set_color('#CBD5E1')
axr.tick_params(colors=SUBTLE_TEXT, labelsize=8)

ax.text(8, 0.18, 'Algorithms in Python  |  Deep Learning Architectures Part 3',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '11-deep-learning-architectures/03-recurrent-neural-networks/'
       'header_rnn.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
