"""Generate the header image for the DINO article.

Right panels are REAL: PCA of the actual DINO teacher embedding, the
2x2 recipes plotted in the entropy plane, and the teacher/student
probe trace -- all retrained live with the frozen seeds.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import dino as D

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'
CLS_COLS = [ORANGE, BLUE, GREEN]

# ---- rebuild stage + main DINO run with the frozen seeds -----------------
rng = np.random.default_rng(0)
imgs, y, factors = D.make_instances(D.N_UNLAB, rng)
te_imgs, te_y, te_f = D.make_instances(D.N_TEST,
                                       np.random.default_rng(50))
Xtr = D.augment(imgs, np.random.default_rng(54))
Xte = D.augment(te_imgs, np.random.default_rng(55))
r = np.random.default_rng(2)
few = np.concatenate([r.choice(np.where(y == c)[0], 5, False)
                      for c in range(3)])

d = D.DINO(seed=0)
r1 = np.random.default_rng(1)
trace = []
for s in range(D.STEPS):
    idx = r1.integers(0, len(imgs), D.BATCH)
    Xa = D.augment(imgs[idx], r1)
    Xb = D.augment(imgs[idx], r1)
    d.step(Xa, Xb)
    if (s + 1) in (100, 500, 1000, 3000):
        at = D.probe_acc(d.embed(Xtr), y, d.embed(Xte), te_y)
        as_ = D.probe_acc(d.embed(Xtr, teacher=False), y,
                          d.embed(Xte, teacher=False), te_y)
        trace.append((s + 1, at, as_))
a15 = D.probe_acc(d.embed(Xtr)[few], y[few], d.embed(Xte), te_y)
per, om = D.collapse_stats(d, Xte[:500])
print(a15, trace[-1], per, om)
assert f'{a15:.1%}' == '94.5%' and f'{trace[-1][1]:.1%}' == '99.9%'
assert f'{per:.2f}' == '0.14' and f'{om:.2f}' == '2.09'
assert f'{trace[0][1]:.1%}' == '54.7%' and f'{trace[0][2]:.1%}' == '59.2%'

pts = {'centre + sharpen': (per, om, GREEN)}
for center, sharpen, name, col in (
        (True, False, 'centre only', BLUE),
        (False, True, 'sharpen only', ORANGE),
        (False, False, 'neither', RED)):
    d2 = D.train_dino(imgs, center=center, sharpen=sharpen)
    p2, o2 = D.collapse_stats(d2, Xte[:500])
    pts[name] = (p2, o2, col)
print({k: (round(v[0], 2), round(v[1], 2)) for k, v in pts.items()})

Zt = d.embed(Xte)


def pca2(Z):
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ Vt[:2].T


P2 = pca2(Zt)

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Self-Distillation (DINO): Chasing Your Own Slower Shadow',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Agreement with no negatives: an EMA teacher, a running mean '
        'and a temperature \u2014 and the collapse Part 1 measured '
        'never comes.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.1, 5.3, 1.55, BLUE)
ax.text(3.35, 7.38, 'the recipe', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=BLUE)
ax.text(3.35, 6.98, 'teacher(view A) -> centred, sharpened targets\n'
                    'student(view B) -> cross-entropy toward them',
        fontsize=8.4, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.5, 'the teacher is never trained: it is an EMA\n'
                   "of the student -- the student's own past",
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.45, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.55, 'no negatives, no collapse', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.98, 'DINO        : 94.5% / 99.9%  spread 0.152\n'
                    'contrastive : 95.1% / 99.7%  spread 0.174\n'
                    'naive pull  : 34.0% (chance) spread 0.001',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.75, 5.3, 1.25, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.75, 'the tightrope', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 3.25, 'centring alone -> uniform collapse (H = log K)\n'
                    'sharpening alone -> one-hot collapse (H = 0)',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.15, "the teacher's job is stability", fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 1.65, 'momentum 0 breaks (67.3%); 0.9-0.99 works.\n'
                    'the teacher trails early -- it is not better,'
                    ' it is slow',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the embedding =====================
axg = fig.add_axes([0.455, 0.475, 0.30, 0.335])
for c in range(3):
    m = te_y == c
    axg.scatter(P2[m, 0], P2[m, 1], s=6, color=CLS_COLS[c],
                alpha=0.65, linewidths=0)
axg.set_xticks([]); axg.set_yticks([])
for sp in axg.spines.values():
    sp.set_color('#CBD5E1')
axg.set_title('the teacher embedding, coloured by class (real runs)',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=5)
ax.text(12.35, 6.0, 'PCA of the DINO teacher\nembedding of 1,000 test\n'
        'images. colour = class,\nnever shown in training.\n\n'
        'three islands again --\nbuilt with no negatives,\n'
        'no contrast, no crowd:\njust a slow shadow and\ntwo softmax '
        'tweaks.',
        fontsize=8.4, ha='left', va='center', color=SUBTLE_TEXT)

# ============ RIGHT BOTTOM LEFT: the entropy plane ============
axe = fig.add_axes([0.455, 0.095, 0.245, 0.30])
logk = np.log(D.K_PROTO)
for name, (p_, o_, col) in pts.items():
    axe.scatter([p_], [o_], s=70, color=col, zorder=3,
                edgecolors='black', linewidths=0.6)
axe.annotate('healthy', pts['centre + sharpen'][:2],
             xytext=(0.5, 1.55), fontsize=7.8, color=GREEN,
             fontweight='bold')
axe.annotate('uniform\ncollapse', (2.85, 3.15), fontsize=7.4,
             color=RED, ha='right', va='center')
axe.annotate('one-hot\ncollapse', (0.45, 0.35), fontsize=7.4,
             color=ORANGE, ha='center')
axe.axvline(logk, color='#CBD5E1', linewidth=0.8, linestyle=':')
axe.axhline(logk, color='#CBD5E1', linewidth=0.8, linestyle=':')
axe.set_xlim(-0.2, 3.9)
axe.set_ylim(-0.2, 3.9)
axe.set_xlabel('H(per sample)', fontsize=8, color=SUBTLE_TEXT)
axe.set_ylabel('H(mean prediction)', fontsize=8, color=SUBTLE_TEXT)
for sp in ('top', 'right'):
    axe.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axe.spines[sp].set_color('#CBD5E1')
axe.tick_params(colors=SUBTLE_TEXT, labelsize=7)
axe.set_title('the entropy plane: 4 recipes', fontsize=9.0,
              fontweight='bold', color=TEXT_COLOR, pad=5)

# ============ RIGHT BOTTOM RIGHT: the trace ============
axt = fig.add_axes([0.755, 0.095, 0.205, 0.30])
steps = [t[0] for t in trace]
axt.plot(steps, [100 * t[1] for t in trace], 'o-', color=PURPLE,
         linewidth=1.8, markersize=4, label='teacher')
axt.plot(steps, [100 * t[2] for t in trace], 's--', color=ORANGE,
         linewidth=1.8, markersize=4, label='student')
axt.set_xscale('log')
axt.set_xticks([100, 500, 1000, 3000])
axt.set_xticklabels(['100', '500', '1k', '3k'], fontsize=7)
axt.minorticks_off()
axt.set_ylim(50, 103)
axt.set_ylabel('probe acc (%)', fontsize=7.6, color=SUBTLE_TEXT)
axt.set_xlabel('step', fontsize=7.6, color=SUBTLE_TEXT)
axt.legend(fontsize=7.2, frameon=False, loc='lower right')
for sp in ('top', 'right'):
    axt.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axt.spines[sp].set_color('#CBD5E1')
axt.tick_params(colors=SUBTLE_TEXT, labelsize=7)
axt.set_title('the teacher trails, then ties', fontsize=9.0,
              fontweight='bold', color=TEXT_COLOR, pad=5)

ax.text(8, 0.18,
        'Algorithms in Python  |  Self-Supervised Learning Part 3',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '12-self-supervised-learning/03-self-distillation/'
       'header_dino.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
