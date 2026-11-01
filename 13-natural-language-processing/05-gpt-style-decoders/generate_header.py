"""Generate the header image for the GPT-style Decoders article.

Right panels are REAL: the decoder and the encoder are pre-trained live
with gpt_decoder.py's frozen seeds, the sample-efficiency curve is the
measured probe history, and the temperature frontier is measured by
actually sampling. Every number is asserted against the frozen run.
Runs in about three minutes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import gpt_decoder as G

# ---- retrain with frozen seeds -----------------------------------------
print('training the decoder and the encoder (about two and a half '
      'minutes)...')
corpus, ids, keep, test, accept, train20 = G.build_stage()
(lm, h_lm, p_lm, n_lm), (mlm, h_ml, p_ml, n_ml) = G.train_pair(
    ids, keep, accept, train20, test)

TOK, SENT, OTOK, OSENT = G.lm_eval(lm, test)
STEPS_G = [s for s, _ in p_lm]
LM_SENT = [v[0] for _, v in p_lm]
ML_SENT = [v[0] for _, v in p_ml]
RATIO = n_lm / n_ml

_, _, sel_lm = G.lm_targets(ids, keep)
LM_TGT = float(sel_lm.sum(1).mean())
ML_TGT = float(np.mean([
    G.mlm_targets(ids, keep, np.random.default_rng(100 + d))[2]
    .sum(1).mean() for d in range(20)]))

TEMP = []
for t in (0.0, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0):
    q = G.sample_quality(lm, np.random.default_rng(G.SEED_SAMPLE),
                         n=G.N_SAMPLES, temp=t)
    TEMP.append((t, q['grammatical'], q['unique']))

M = G.model_on_language(lm)
BASE = G.singular_base_rate(test)
bs = G.score_subject(lm, test)
a20, _ = G.score_task(G.lm_logprob, lm,
                      G.make_acceptability(20, np.random.default_rng(1020)),
                      accept)
f_ml400 = G.finetune(mlm, 'first',
                     G.make_acceptability(400, np.random.default_rng(1400)),
                     accept, causal=False)

# ---- assert against the frozen run of gpt_decoder.py -------------------
assert f'{TOK:.4f}' == '1.4706', TOK
assert f'{OTOK:.4f}' == '1.4190', OTOK
assert f'{LM_TGT:.3f}' == '6.491', LM_TGT
assert f'{ML_TGT:.3f}' == '1.235', ML_TGT
assert f'{RATIO:.2f}' == '5.25', RATIO
assert len(G.SENT_P) == 31740, len(G.SENT_P)
assert f'{G.language_entropy():.4f}' == '9.2538', G.language_entropy()
assert f'{M["mass"]:.4f}' == '0.9770', M['mass']
assert f'{bs[0]:.4f}' == f'{BASE:.4f}' == '0.6670', (bs[0], BASE)
assert f'{a20:.4f}' == '0.9900', a20
assert f'{f_ml400:.4f}' == '0.9833', f_ml400
assert [f'{g:.4f}' for _, g, _ in TEMP] == [
    '1.0000', '1.0000', '0.9990', '0.9780', '0.9450', '0.8170',
    '0.4830'], TEMP
assert [f'{u:.4f}' for _, _, u in TEMP] == [
    '0.0010', '0.7110', '0.8330', '0.8770', '0.9060', '0.9520',
    '0.9860'], TEMP
assert [f'{v:.4f}' for v in LM_SENT[:3]] == [
    '0.5050', '0.7267', '0.9517'], LM_SENT[:3]
assert [f'{v:.4f}' for v in ML_SENT[:3]] == [
    '0.5267', '0.5267', '0.5233'], ML_SENT[:3]
print('all header numbers match the frozen run')

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
AMBER = '#B45309'
BADGE_BG = '#F8FAFC'

# ---- figure -------------------------------------------------------------
fig = plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'GPT-style Decoders: Writing One Token at a Time',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Put the causal mask back and every position becomes a lesson '
        f'\u2014 {RATIO:.2f}x the signal, and a loss '
        f'{100 * (TOK / OTOK - 1):.1f}% above the language\u2019s own '
        'entropy.',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')


def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))


# Box 1: the objective
box(0.7, 6.05, 5.3, 1.45, ORANGE)
ax.text(3.35, 7.22, 'The objective: shift the targets left',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=ORANGE)
ax.text(3.35, 6.62,
        '[BOS] the kings rule their crown [EOS]\n'
        f'every position a target: {LM_TGT:.2f} vs masked {ML_TGT:.2f}',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

# Box 2: the computed floor
box(0.7, 4.35, 5.3, 1.45, GREEN, face=BADGE_BG)
ax.text(3.35, 5.52, 'A loss with a floor we can compute',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=GREEN)
ax.text(3.35, 4.92,
        f'{len(G.SENT_P):,} sentences enumerated, '
        f'H = {G.language_entropy():.4f} nats\n'
        f'model {TOK:.4f} vs irreducible {OTOK:.4f} nats/token',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

# Box 3: the architecture
box(0.7, 2.95, 5.3, 1.1, BLUE, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.82, 'The decoder half, alone',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=BLUE)
ax.text(3.35, 3.35,
        f'2 layers, 4 heads, d=64, {lm.n_params() // 1000}k parameters\n'
        'the causal mask is back, and [EOS] is a target',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# Box 4: scoring and the price
box(0.7, 1.35, 5.3, 1.25, PURPLE, lw=1.6)
ax.text(3.35, 2.35, 'One pass to score, nothing to the right',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=PURPLE)
ax.text(3.35, 1.82,
        f'20 labels, no head: {a20:.1%} vs 400 fine-tuned: '
        f'{f_ml400:.1%}\n'
        f'subject number {bs[0]:.1%} = the base rate exactly',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

# ---------------- RIGHT TOP: sample efficiency ----------------
axh = fig.add_axes([0.45, 0.44, 0.50, 0.30])
axh.plot(STEPS_G, LM_SENT, 'o-', color=BLUE, lw=2, ms=4.5,
         label='decoder (next token)', zorder=4)
axh.plot(STEPS_G, ML_SENT, 's-', color=AMBER, lw=2, ms=4.5,
         label='encoder (masked)', zorder=4)
axh.axhline(0.5, color='#CBD5E1', ls=':', lw=1.2, zorder=2)
axh.text(1450, 0.452, 'chance', fontsize=7.0, color='#9CA3AF',
         ha='right')
axh.axvline(50, color=BLUE, ls='--', lw=1.1, zorder=2)
axh.axvline(300, color=AMBER, ls='--', lw=1.1, zorder=2)
axh.text(80, 0.462, '95% at step 50 vs 300', fontsize=7.0,
         color=SUBTLE_TEXT, ha='center')
axh.set_xscale('log')
axh.set_xticks([10, 25, 50, 100, 300, 800, 1500])
axh.set_xticklabels(['10', '25', '50', '100', '300', '800', '1500'],
                    fontsize=7.2, color=SUBTLE_TEXT)
axh.set_yticks([0.5, 0.75, 1.0])
axh.set_yticklabels(['50%', '75%', '100%'], fontsize=7.2,
                    color=SUBTLE_TEXT)
axh.set_ylim(0.44, 1.05)
axh.set_xlabel('pre-training step', fontsize=7.6, color=SUBTLE_TEXT)
axh.legend(fontsize=7.2, loc='center right', frameon=False)
axh.set_title('is this sentence grammatical? no labels, one threshold',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=6,
              loc='left')
axh.grid(color='#E2E8F0', linewidth=0.6, zorder=0)
for sp in ('top', 'right'):
    axh.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axh.spines[sp].set_color('#CBD5E1')

# ---------------- RIGHT BOTTOM: the temperature frontier ----------------
axb = fig.add_axes([0.45, 0.135, 0.50, 0.215])
tx = np.arange(len(TEMP))
axb.plot(tx, [g for _, g, _ in TEMP], 'o-', color=GREEN, lw=2, ms=5,
         zorder=4, label='in the language')
axb.plot(tx, [u for _, _, u in TEMP], 's-', color=RED, lw=2, ms=4.5,
         zorder=4, label='distinct')
axb.legend(fontsize=7.2, loc='lower right', frameon=False)
axb.annotate('greedy: all\nthe same sentence',
             xy=(0.04, TEMP[0][2]), xytext=(1.05, 0.10),
             textcoords='data', fontsize=6.8, color=RED,
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.0))
axb.set_xticks(tx)
axb.set_xticklabels(['greedy' if t == 0 else f'{t:.1f}'
                     for t, _, _ in TEMP], fontsize=7.2,
                    color=SUBTLE_TEXT)
axb.set_yticks([0, 0.5, 1.0])
axb.set_yticklabels(['0', '50%', '100%'], fontsize=7.2,
                    color=SUBTLE_TEXT)
axb.set_ylim(-0.06, 1.1)
axb.set_xlabel('sampling temperature', fontsize=7.6, color=SUBTLE_TEXT)
axb.set_title('quality and diversity pull against each other',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=6,
              loc='left')
axb.grid(axis='y', color='#E2E8F0', linewidth=0.6, zorder=0)
for sp in ('top', 'right'):
    axb.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axb.spines[sp].set_color('#CBD5E1')

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 5',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/05-gpt-style-decoders/'
       'header_gpt_decoder.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
