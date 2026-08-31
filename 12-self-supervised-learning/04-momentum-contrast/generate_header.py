"""Generate the header image for the MoCo article.

Right panels are REAL: the consistency-law curve, the closing family
bars, and the flat queue audit -- all retrained live with the frozen
seeds and asserted against the frozen run's numbers.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import moco as M

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- rebuild everything with the frozen seeds ----------------------------
rng = np.random.default_rng(0)
imgs, y, factors = M.make_instances(M.N_UNLAB, rng)
te_imgs, te_y, te_f = M.make_instances(M.N_TEST,
                                       np.random.default_rng(50))
Xtr = M.augment(imgs, np.random.default_rng(54))
Xte = M.augment(te_imgs, np.random.default_rng(55))
r = np.random.default_rng(2)
few = np.concatenate([r.choice(np.where(y == c)[0], 5, False)
                      for c in range(3)])


def probes(model):
    a15 = M.probe_acc(model.embed(Xtr)[few], y[few],
                      model.embed(Xte), te_y)
    aall = M.probe_acc(model.embed(Xtr), y, model.embed(Xte), te_y)
    return a15, aall


# momentum curve
moms = (0.0, 0.9, 0.99, 0.999)
mom_acc = []
for m_key in moms:
    mo = M.train_moco(imgs, m_key=m_key)
    mom_acc.append(probes(mo)[1])
print('momentum', [f'{a:.1%}' for a in mom_acc])
assert f'{mom_acc[0]:.1%}' == '32.9%' and f'{mom_acc[3]:.1%}' == '95.4%'

# queue audit
qs = (16, 256, 4096)
q_acc = []
for qk in qs:
    mo = M.train_moco(imgs, queue_k=qk)
    q_acc.append(probes(mo)[1])
print('queue', [f'{a:.1%}' for a in q_acc])
assert f'{q_acc[0]:.1%}' == '95.3%' and f'{q_acc[2]:.1%}' == '95.8%'

# family
mo32 = M.train_moco(imgs, steps=16000, batch=32)
enc = M.train_ntxent(imgs)
Xtr_flat = imgs.reshape(len(imgs), -1)
mae = M.MAE(seed=0)
rm = np.random.default_rng(1)
for _ in range(3000):
    idx = rm.integers(0, len(imgs), M.BATCH)
    mae.step(Xtr_flat[idx], rm)
di = M.DINO(seed=0)
rd = np.random.default_rng(1)
for _ in range(3000):
    idx = rd.integers(0, len(imgs), M.BATCH)
    Xa = M.augment(imgs[idx], rd)
    Xb = M.augment(imgs[idx], rd)
    di.step(Xa, Xb)
fam = {}
for name, model in (("SimCLR", enc), ("MAE", mae), ("DINO", di),
                    ("MoCo", mo32)):
    fam[name] = probes(model)
print(fam)
assert f'{fam["MoCo"][0]:.1%}' == '96.0%'
assert f'{fam["SimCLR"][0]:.1%}' == '95.1%'
assert f'{fam["MAE"][0]:.1%}' == '49.2%'
assert f'{fam["DINO"][0]:.1%}' == '94.5%'

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Momentum Contrast (MoCo): A Crowd You Keep in a Ledger',
        fontsize=17, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Negatives from a queue, kept comparable by a slow twin \u2014 '
        'and the section\u2019s four yardsticks close on one table.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.1, 5.3, 1.55, BLUE)
ax.text(3.35, 7.38, 'the ledger', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=BLUE)
ax.text(3.35, 6.98, 'q  vs  [ k+ | last 1,024 banked keys ]\n'
                    'push new keys, pop the oldest (FIFO)',
        fontsize=8.6, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.5, 'keys written by a slow EMA twin --\n'
                   'gradients flow through the query side only',
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.45, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.55, 'crowd, decoupled from batch', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.98, 'batch 32 + queue 1024 : 96.0% / 99.6%\n'
                    'batch  8 + queue 1024 : 96.0% / 99.7%\n'
                    '(SimCLR batch 256     : 95.1% / 99.7%)',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.75, 5.3, 1.25, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.75, 'the consistency law', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 3.25, 'keys from the live encoder: 32.9% -- chance.\n'
                    'from a 0.999-momentum twin: 95.4%.',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.15, 'the honest audit', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 1.65, 'queue 16 = queue 4096 on this stage --\n'
                    'MoCo solves a SCALE problem. measure first.',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ============ RIGHT TOP: the consistency law ============
axm = fig.add_axes([0.455, 0.475, 0.30, 0.335])
xpos = np.arange(4)
axm.plot(xpos, [100 * a for a in mom_acc], 'o-', color=RED,
         linewidth=2.2, markersize=6)
axm.axhline(33.3, color='#CBD5E1', linewidth=1, linestyle=':')
axm.text(0.05, 35, 'chance', fontsize=7.4, color=SUBTLE_TEXT)
axm.set_xticks(xpos)
axm.set_xticklabels(['0', '0.9', '0.99', '0.999'], fontsize=8)
axm.set_ylim(25, 103)
axm.set_ylabel('probe accuracy (%)', fontsize=8.2, color=SUBTLE_TEXT)
for sp in ('top', 'right'):
    axm.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axm.spines[sp].set_color('#CBD5E1')
axm.tick_params(colors=SUBTLE_TEXT, labelsize=8)
axm.set_title('the consistency law: probe vs key momentum',
              fontsize=9.3, fontweight='bold', color=TEXT_COLOR, pad=5)
ax.text(12.35, 6.0, 'same queue, same budget --\nonly the momentum of\n'
        'the key encoder moves.\n\nat 0 the ledger is written\nby a '
        'drifting coordinate\nsystem: chance. slow the\ntwin and the '
        'entries stay\ncomparable: 95.4%.\nconsistency is the\ninvention.',
        fontsize=8.4, ha='left', va='center', color=SUBTLE_TEXT)

# ============ RIGHT BOTTOM LEFT: the family ============
axf = fig.add_axes([0.455, 0.095, 0.245, 0.30])
names = ['SimCLR', 'MAE', 'DINO', 'MoCo']
cols = [BLUE, ORANGE, PURPLE, GREEN]
x4 = np.arange(4)
axf.bar(x4 - 0.18, [100 * fam[n][0] for n in names], 0.34,
        color='#94A3B8', label='15 labels')
axf.bar(x4 + 0.18, [100 * fam[n][1] for n in names], 0.34,
        color=cols, label='all labels')
axf.set_xticks(x4)
axf.set_xticklabels(names, fontsize=7.6)
axf.set_ylim(0, 105)
axf.set_ylabel('probe acc (%)', fontsize=7.6, color=SUBTLE_TEXT)
axf.legend(fontsize=6.8, frameon=False, loc='center',
           bbox_to_anchor=(0.5, 0.86))
for sp in ('top', 'right'):
    axf.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axf.spines[sp].set_color('#CBD5E1')
axf.tick_params(colors=SUBTLE_TEXT, labelsize=7.5)
axf.set_title('the family, closed (4 live retrains)', fontsize=9.0,
              fontweight='bold', color=TEXT_COLOR, pad=5)

# ============ RIGHT BOTTOM RIGHT: the audit ============
axq = fig.add_axes([0.755, 0.095, 0.205, 0.30])
axq.plot([0, 1, 2], [100 * a for a in q_acc], 'o-', color=GREEN,
         linewidth=2.0, markersize=5, label='MoCo, queue size')
axq.axhline(99.9, color=ORANGE, linewidth=1.4, linestyle='--',
            label='NT-Xent b8, long')
axq.set_xticks([0, 1, 2])
axq.set_xticklabels(['16', '256', '4096'], fontsize=7.6)
axq.set_ylim(90, 102)
axq.set_xlabel('queue size', fontsize=7.6, color=SUBTLE_TEXT)
axq.set_ylabel('probe-all (%)', fontsize=7.6, color=SUBTLE_TEXT)
axq.legend(fontsize=6.6, frameon=False, loc='lower right')
for sp in ('top', 'right'):
    axq.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axq.spines[sp].set_color('#CBD5E1')
axq.tick_params(colors=SUBTLE_TEXT, labelsize=7.5)
axq.set_title('the audit: flat', fontsize=9.0,
              fontweight='bold', color=TEXT_COLOR, pad=5)

ax.text(8, 0.18,
        'Algorithms in Python  |  Self-Supervised Learning Part 4',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '12-self-supervised-learning/04-momentum-contrast/'
       'header_moco.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
