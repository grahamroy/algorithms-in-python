"""Generate the header image for the BERT-style Encoders article.

Right panels are REAL: both encoders are pre-trained live with frozen
seeds, the fill-the-gap accuracies are measured on the 1,000-sentence
held-out set, and the label-efficiency curve is measured by actually
fine-tuning. Every number is asserted against the frozen run.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import bert_encoder as B

# ---- retrain with frozen seeds ------------------------------------------
rng = np.random.default_rng(42)
corpus = B.gen_corpus(8000, rng)
ids, keep = B.encode([s["words"] for s in corpus])
test = B.gen_corpus(1000, np.random.default_rng(7))
test_accept = B.make_acceptability(600, np.random.default_rng(13))

bi = B.Encoder(d=64, heads=4, layers=2, causal=False, seed=0, lr=3e-3)
B.pretrain(bi, ids, keep, B.STEPS, np.random.default_rng(1))
ca = B.Encoder(d=64, heads=4, layers=2, causal=True, seed=0, lr=3e-3)
B.pretrain(ca, ids, keep, B.STEPS, np.random.default_rng(1))

bs = B.score_subject(bi, test)
cs = B.score_subject(ca, test)
bv = B.score_verb(bi, test)
cv = B.score_verb(ca, test)
bp = B.score_poss(bi, test)
cp = B.score_poss(ca, test)

SLOTS = [("subject\nnumber", cs[0], bs[0]),
         ("subject\ngender", cs[3], bs[3]),
         ("subject\ntopic", cs[1], bs[1]),
         ("verb\nnumber", cv[0], bv[0]),
         ("possessive\nexact", cp[0], bp[0])]

assert f'{bs[0]:.3f}' == '0.998', f'bidir subj number {bs[0]}'
assert f'{cs[0]:.3f}' == '0.667', f'causal subj number {cs[0]}'
assert f'{bs[3]:.3f}' == '1.000', f'bidir subj gender {bs[3]}'
assert f'{cs[3]:.3f}' == '0.206', f'causal subj gender {cs[3]}'
assert f'{bs[1]:.3f}' == '1.000', f'bidir subj topic {bs[1]}'
assert f'{cs[1]:.3f}' == '0.687', f'causal subj topic {cs[1]}'
assert f'{bv[0]:.3f}' == f'{cv[0]:.3f}' == '1.000', 'verb number'
assert f'{bp[0]:.3f}' == f'{cp[0]:.3f}' == '1.000', 'possessive'

LAB = [20, 50, 100, 400]
SCRATCH, PROBE, FT = [], [], []
for n in LAB:
    train = B.make_acceptability(n, np.random.default_rng(1000 + n))
    sc = B.Encoder(d=64, heads=4, layers=2, seed=5, lr=1e-3)
    SCRATCH.append(B.finetune(sc, train, test_accept, 400,
                              np.random.default_rng(11)))
    pr = bi.clone()
    pr.lr = 1e-2
    PROBE.append(B.finetune(pr, train, test_accept, 400,
                            np.random.default_rng(11), only={"Wc", "bc"}))
    ft = bi.clone()
    ft.lr = 1e-3
    FT.append(B.finetune(ft, train, test_accept, 400,
                         np.random.default_rng(11)))

assert [f'{v:.3f}' for v in SCRATCH] == \
    ['0.508', '0.565', '0.557', '0.872'], SCRATCH
assert [f'{v:.3f}' for v in PROBE] == \
    ['0.567', '0.617', '0.563', '0.568'], PROBE
assert [f'{v:.3f}' for v in FT] == \
    ['0.548', '0.672', '0.800', '0.955'], FT

s_pll = B.pseudo_log_likelihood(bi, [w for w, _ in test_accept])
thr = B.pll_threshold(bi, B.make_acceptability(
    20, np.random.default_rng(1020)))
labels = np.array([y for _, y in test_accept])
ZERO = float(((s_pll >= thr) == (labels == 1)).mean())
assert f'{ZERO:.3f}' == '0.905', f'zero-shot {ZERO}'

# ---- colour palette (matches series) ------------------------------------
BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'

# ---- figure -------------------------------------------------------------
fig = plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'BERT-style Encoders: Reading in Both Directions',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Hide a word and predict it \u2014 the masked subject\u2019s '
        f'number goes from {cs[0]:.1%} (the base rate) to {bs[0]:.1%}.',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))


# Box 1: the pretext
box(0.7, 6.05, 5.3, 1.45, ORANGE)
ax.text(3.35, 7.22, 'The pretext: masked language modelling',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=ORANGE)
ax.text(3.35, 6.62,
        'the [MASK] rule their crown\n'
        '15% chosen | 80% [MASK], 10% random, 10% kept',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

# Box 2: where the evidence lies
box(0.7, 4.35, 5.3, 1.45, GREEN, face=BADGE_BG)
ax.text(3.35, 5.52, 'Bidirectionality is not free generality',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=GREEN)
ax.text(3.35, 4.92,
        'evidence to the LEFT:  causal ties, 100%\n'
        f'evidence to the RIGHT: causal {cs[3]:.1%}, bidir {bs[3]:.0%}',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

# Box 3: the architecture
box(0.7, 2.95, 5.3, 1.1, BLUE, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.82, 'The encoder half, alone',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=BLUE)
ax.text(3.35, 3.35,
        f'2 layers, 4 heads, d=64, {bi.n_params() // 1000}k parameters\n'
        'no causal mask, no decoder, a [CLS] in front',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# Box 4: transfer
box(0.7, 1.35, 5.3, 1.25, PURPLE, lw=1.6)
ax.text(3.35, 2.35, 'Pre-train once, then spend labels',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=PURPLE)
ax.text(3.35, 1.82,
        f'100 labels: {SCRATCH[2]:.1%} scratch -> {FT[2]:.1%} fine-tuned\n'
        f'20 labels + no new parameters: {ZERO:.1%}',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

# ---------------- RIGHT TOP: fill-the-gap receipt ----------------
axh = fig.add_axes([0.45, 0.44, 0.50, 0.30])
y = np.arange(len(SLOTS))
h = 0.36
ca_v = [s[1] for s in SLOTS]
bi_v = [s[2] for s in SLOTS]
axh.barh(y + h / 2, ca_v, h, color=RED, label='causal', zorder=3)
axh.barh(y - h / 2, bi_v, h, color=GREEN, label='bidirectional', zorder=3)
axh.set_yticks(y)
axh.set_yticklabels([s[0] for s in SLOTS], fontsize=7.4,
                    color=TEXT_COLOR)
axh.invert_yaxis()
axh.set_xlim(0, 1.18)
axh.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axh.set_xticklabels(['0', '25%', '50%', '75%', '100%'], fontsize=7.2,
                    color=SUBTLE_TEXT)
for yy, v in zip(y + h / 2, ca_v):
    axh.text(v + 0.015, yy, f'{v:.1%}', va='center', fontsize=6.8,
             color=RED)
for yy, v in zip(y - h / 2, bi_v):
    axh.text(v + 0.015, yy, f'{v:.1%}', va='center', fontsize=6.8,
             color=GREEN)
axh.legend(fontsize=7.2, loc='lower right', frameon=False,
           bbox_to_anchor=(1.01, 1.0), ncol=2, borderpad=0.0)
axh.set_title('fill the gap: accuracy by what is hidden',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=6,
              loc='left')
axh.grid(axis='x', color='#E2E8F0', linewidth=0.6, zorder=0)
for sp in ('top', 'right'):
    axh.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axh.spines[sp].set_color('#CBD5E1')

# ---------------- RIGHT BOTTOM: label efficiency ----------------
axb = fig.add_axes([0.45, 0.085, 0.50, 0.25])
x = np.arange(len(LAB))
axb.plot(x, FT, 'o-', color=GREEN, lw=2, ms=5, zorder=4)
axb.plot(x, SCRATCH, 's-', color=RED, lw=2, ms=4.5, zorder=4)
axb.plot(x, PROBE, '^-', color=SUBTLE_TEXT, lw=1.6, ms=4.5, zorder=4)
axb.axhline(ZERO, color=PURPLE, ls='--', lw=1.5, zorder=3)
axb.text(-0.08, ZERO + 0.028, f'no fine-tuning at all: {ZERO:.1%}',
         fontsize=7.0, color=PURPLE, ha='left')
axb.axhline(0.5, color='#CBD5E1', ls=':', lw=1.2, zorder=2)
for yv, lb, cl in ((FT[-1], 'fine-tuned', GREEN),
                   (SCRATCH[-1], 'from scratch', RED),
                   (PROBE[-1], 'frozen probe', SUBTLE_TEXT),
                   (0.5, 'chance', '#9CA3AF')):
    axb.text(3.08, yv, lb, fontsize=7.2, color=cl, va='center',
             ha='left', fontweight='bold' if cl != '#9CA3AF' else None)
axb.set_xlim(-0.12, 4.15)
axb.set_xticks(x)
axb.set_xticklabels([str(n) for n in LAB], fontsize=7.4,
                    color=SUBTLE_TEXT)
axb.set_yticks([0.5, 0.75, 1.0])
axb.set_yticklabels(['50%', '75%', '100%'], fontsize=7.2,
                    color=SUBTLE_TEXT)
axb.set_ylim(0.42, 1.06)
axb.set_xlabel('labelled sentences', fontsize=7.6, color=SUBTLE_TEXT)
axb.set_title('is this sentence grammatical? accuracy vs labels',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=6,
              loc='left')
axb.grid(axis='y', color='#E2E8F0', linewidth=0.6, zorder=0)
for sp in ('top', 'right'):
    axb.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axb.spines[sp].set_color('#CBD5E1')

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 4',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/04-bert-style-encoders/'
       'header_bert_encoder.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
