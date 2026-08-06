"""Generate the header image for the GAN article.

Right panels are REAL: the spiral forgery is the actual DEMO-1
generator from gan.py retrained live (same seeds), and the two ring
panels are the healthy and collapsed runs from DEMO 3.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

import gan as G

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- retrain the exact DEMO-1 forger -------------------------------------
gan1 = G.GAN(seed=0)
rng = np.random.default_rng(1)
for _ in range(6000):
    Xr = G.spiral_data(128, rng)
    Z = rng.standard_normal((128, gan1.d_z))
    Xf, _ = gan1.G.forward(Z)
    gan1.d_step(Xr, Xf)
    gan1.g_step(rng.standard_normal((128, gan1.d_z)))
Xfake = gan1.sample(2000, np.random.default_rng(99))
q, c = G.spiral_metrics(Xfake)
acc = G.d_accuracy(gan1, G.spiral_data, 1000, np.random.default_rng(98))
Xreal = G.spiral_data(2000, np.random.default_rng(97))
qr, cr = G.spiral_metrics(Xreal)
print(f'fake quality {q} ({round(q*2000)}/2000) coverage {c}')
print(f'real quality {qr} ({round(qr*2000)}/2000) coverage {cr}')
print(f'detective acc {acc} ({round(acc*2000)}/2000)')
assert round(q * 2000) == 1973 and round(qr * 2000) == 1990
assert round(acc * 2000) == 1008
assert f'{q:.1%}' == '98.7%' and f'{qr:.1%}' == '99.5%'
assert f'{c:.1%}' == '81.7%' and f'{cr:.1%}' == '93.3%'

# ---- retrain the DEMO-3 healthy and collapsed runs -----------------------
ring_runs = {}
for lr_d in (1e-3, 1e-4):
    g = G.GAN(seed=0, lr_d=lr_d, lr_g=1e-3)
    r = np.random.default_rng(10)
    for _ in range(4000):
        Xr = G.make_ring(64, r)
        Z = r.standard_normal((64, g.d_z))
        Xf, _ = g.G.forward(Z)
        g.d_step(Xr, Xf)
        g.g_step(r.standard_normal((64, g.d_z)))
    Xs = g.sample(2000, np.random.default_rng(99))
    m, hq, counts = G.ring_metrics(Xs)
    ring_runs[lr_d] = (Xs, m, hq)
    print(f'lr_d {lr_d:g}: modes {m}/8  on-mode {hq:.0%}')
assert ring_runs[1e-3][1] == 8 and ring_runs[1e-4][1] == 3
assert f'{ring_runs[1e-3][2]:.0%}' == '81%'
assert f'{ring_runs[1e-4][2]:.0%}' == '7%'

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Generative Adversarial Networks: The Loss Function Is '
        'Another Network',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'A forger and a detective, each the other\u2019s loss \u2014 '
        'victory is a coin-flip detective, and every failure is a '
        'lost race.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.1, 5.3, 1.55, BLUE)
ax.text(3.35, 7.38, 'the game', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=BLUE)
ax.text(3.35, 6.98, 'min_G max_D  E[log D(x)] + E[log(1-D(G(z)))]',
        fontsize=8.6, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.5, "the forger's gradient arrives through the\n"
                   "detective -- no loss is ever written down",
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.45, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.55, 'victory is a coin flip', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.98, 'forgery quality  98.7%  (real data: 99.5%)\n'
                    'detective ends at 50.4% -- guessing\n'
                    'the score orbits; it never descends',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.75, 5.3, 1.25, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.75, 'the frozen forger', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 3.25, 'confident detective + minimax loss: gradient\n'
                    '1e-38, quality 0.0%. non-saturating: 93.1%.',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.15, 'mode collapse is a race', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 1.65, 'slow the detective 30x: modes 8/8 -> 0/8.\n'
                    'the forger flees around the ring, never covers it',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the forgery =====================
axg = fig.add_axes([0.455, 0.475, 0.30, 0.335])
axg.scatter(Xreal[:, 0], Xreal[:, 1], s=4, color='#94A3B8',
            alpha=0.55, linewidths=0, zorder=1)
axg.scatter(Xfake[:, 0], Xfake[:, 1], s=4, color=ORANGE,
            alpha=0.6, linewidths=0, zorder=2)
axg.set_xlim(-1.25, 1.25)
axg.set_ylim(-1.25, 1.25)
axg.set_aspect('equal')
axg.set_xticks([]); axg.set_yticks([])
for sp in axg.spines.values():
    sp.set_color('#CBD5E1')
axg.set_title('the forgery: 2,000 draws from noise (real runs)',
              fontsize=9.3, fontweight='bold', color=TEXT_COLOR, pad=5)
ax.text(12.35, 6.0, 'grey = 2,000 real points.\norange = 2,000 forgeries,\n'
        'each one noise pushed\nthrough the trained forger.\n\n'
        'quality 98.7% against the\nreal data\u2019s 99.5% anchor;\n'
        'the detective is down to\n50.4% -- a coin flip.',
        fontsize=8.4, ha='left', va='center', color=SUBTLE_TEXT)

# ===================== RIGHT BOTTOM: the race =====================
for k, (lr_d, x0) in enumerate(((1e-3, 0.455), (1e-4, 0.715))):
    Xs, m, hq = ring_runs[lr_d]
    axr = fig.add_axes([x0, 0.095, 0.245, 0.30])
    C = G.ring_centers()
    for cx, cy in C:
        axr.add_patch(Circle((cx, cy), 3 * G.RING_SD, facecolor='none',
                             edgecolor='#94A3B8', linewidth=1.0,
                             linestyle=':', zorder=1))
    col = GREEN if lr_d == 1e-3 else RED
    axr.scatter(Xs[:, 0], Xs[:, 1], s=4, color=col, alpha=0.45,
                linewidths=0, zorder=2)
    axr.set_xlim(-1.55, 1.55)
    axr.set_ylim(-1.45, 1.65)
    axr.set_aspect('equal')
    axr.set_xticks([]); axr.set_yticks([])
    for sp in axr.spines.values():
        sp.set_color('#CBD5E1')
    ttl = (f'detective lr 1e-3: {m}/8 modes' if lr_d == 1e-3
           else f'detective 10x slower: {m}/8 -- collapse')
    axr.set_title(ttl, fontsize=9.0, fontweight='bold',
                  color=TEXT_COLOR, pad=5)

ax.text(8, 0.18,
        'Algorithms in Python  |  Deep Learning Architectures Part 7',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '11-deep-learning-architectures/07-generative-adversarial-networks/'
       'header_gan.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
