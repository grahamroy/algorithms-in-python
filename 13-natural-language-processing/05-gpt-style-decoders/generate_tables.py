"""Generate every data-table image for the GPT-style Decoders article.

Medium has no table support and collapses the whitespace in wide code
blocks, so each measured table ships as an image. The decoder and the
encoder are pre-trained ONCE here with the frozen seeds from
gpt_decoder.py, and every cell is measured live and asserted against
the frozen run of that script.

Writes: floor_curve.png, floor_table.png, efficiency_curve.png,
        receipt_table.png, labels_table.png, temperature_table.png,
        distribution_table.png
Runs in about four minutes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import gpt_decoder as G

OUT = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/05-gpt-style-decoders/')

# ---- palette (matches the series) ---------------------------------------
BG = '#FFFFFF'
TEXT = '#1F2937'
HEADER_BG = '#1F2937'
HEADER_TEXT = '#FFFFFF'
BORDER = '#E2E8F0'
SUBTLE = '#6B7280'
GREEN = '#15803D'
RED = '#B91C1C'
BLUE = '#2563EB'
PURPLE = '#7C3AED'
AMBER = '#B45309'
NEUTRAL = '#9CA3AF'
GOOD_BG, MID_BG, BAD_BG = '#DCFCE7', '#FEF3C7', '#FEE2E2'
MONO = 'DejaVu Sans Mono'


def cell(text, color=TEXT, bg=None, mono=True, bold=False, size=12,
         align='center'):
    return dict(text=text, color=color, bg=bg, mono=mono, bold=bold,
                size=size, align=align)


def label(text, size=12):
    return cell(text, mono=False, bold=True, size=size, align='left')


def note(text, size=11):
    return cell(text, color=SUBTLE, mono=False, size=size, align='left')


def render_table(path, title, subtitle, headers, widths, rows,
                 captions=(), fig_px=(1700, 900), row_h=0.85,
                 header_h=0.95, rules=()):
    """One table renderer for all of them. rows = list of cell lists."""
    total_w = sum(widths)
    n = len(rows)
    table_h = header_h + n * row_h
    fig, ax = plt.subplots(figsize=(fig_px[0] / 150, fig_px[1] / 150),
                           dpi=150)
    ax.set_xlim(0, total_w + 2.0)
    ax.set_ylim(0, table_h + 2.4 + 0.55 * len(captions))
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(BG)
    x0, y0 = 1.0, 1.0 + 0.55 * len(captions)

    def col_x(i):
        return x0 + sum(widths[:i])

    ax.text(total_w / 2 + 1.0, table_h + y0 + 1.05, title, fontsize=16,
            fontweight='bold', ha='center', va='center', color=TEXT)
    if subtitle:
        ax.text(total_w / 2 + 1.0, table_h + y0 + 0.45, subtitle,
                fontsize=10.5, ha='center', va='center', color=SUBTLE,
                fontstyle='italic')

    header_y = y0 + n * row_h
    ax.add_patch(Rectangle((x0, header_y), total_w, header_h,
                           facecolor=HEADER_BG, edgecolor=HEADER_BG,
                           zorder=1))
    for i, h in enumerate(headers):
        ax.text(col_x(i) + widths[i] / 2, header_y + header_h / 2, h,
                fontsize=11.8, fontweight='bold', ha='center',
                va='center', color=HEADER_TEXT, zorder=2,
                linespacing=1.5)

    for r, cells in enumerate(rows):
        ry = y0 + (n - 1 - r) * row_h
        ax.add_patch(Rectangle((x0, ry), total_w, row_h,
                               facecolor='#F8FAFC' if r % 2 == 0
                               else '#FFFFFF', edgecolor=BORDER,
                               linewidth=0.8, zorder=1))
        for i, c in enumerate(cells):
            if c is None:
                continue
            if c['bg']:
                ax.add_patch(Rectangle((col_x(i) + 0.18, ry + 0.11),
                                       widths[i] - 0.36, row_h - 0.22,
                                       facecolor=c['bg'],
                                       edgecolor='none', zorder=2))
            if c['align'] == 'left':
                tx, ha = col_x(i) + 0.3, 'left'
            else:
                tx, ha = col_x(i) + widths[i] / 2, 'center'
            ax.text(tx, ry + row_h / 2, c['text'], fontsize=c['size'],
                    fontweight='bold' if c['bold'] else None, ha=ha,
                    va='center', color=c['color'], zorder=3,
                    fontfamily=MONO if c['mono'] else 'sans-serif')

    ax.add_patch(Rectangle((x0, y0), total_w, table_h, facecolor='none',
                           edgecolor=BORDER, linewidth=1.2, zorder=4))
    for i in range(1, len(widths)):
        ax.plot([col_x(i), col_x(i)], [y0, header_y + header_h],
                color=BORDER, linewidth=0.8, zorder=4)
    for after in rules:
        ry = y0 + (n - after) * row_h
        ax.plot([x0, x0 + total_w], [ry, ry], color='#94A3B8',
                linewidth=1.4, zorder=5)

    for k, (txt, col) in enumerate(captions):
        ax.text(total_w / 2 + 1.0, y0 - 0.5 - 0.55 * k, txt,
                fontsize=10.5, ha='center', va='center', color=col,
                fontstyle='italic')

    plt.tight_layout()
    plt.savefig(OUT + path, dpi=150, bbox_inches='tight',
                facecolor=BG, edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    print(f'Wrote {OUT + path}')


def style_axes(ax):
    ax.grid(color='#E2E8F0', linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=9.5, colors=SUBTLE)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#CBD5E1')


# =======================================================================
# Train the pair once, with gpt_decoder.py's frozen seeds.
# =======================================================================

print('training the decoder and the encoder (about two and a half '
      'minutes)...')
corpus, ids, keep, test, accept, train20 = G.build_stage()
(lm, h_lm, p_lm, n_lm), (mlm, h_ml, p_ml, n_ml) = G.train_pair(
    ids, keep, accept, train20, test)

TOK, SENT, OTOK, OSENT = G.lm_eval(lm, test)
assert f'{TOK:.4f}' == '1.4706', TOK
assert f'{OTOK:.4f}' == '1.4190', OTOK
assert f'{SENT:.4f}' == '9.5722', SENT
assert f'{OSENT:.4f}' == '9.2364', OSENT
assert f'{n_lm / n_ml:.2f}' == '5.25', n_lm / n_ml

STEPS_G = [s for s, _ in p_lm]
LM_NLL = [v[2] for _, v in p_lm]
LM_SENT = [v[0] for _, v in p_lm]
ML_SENT = [v[0] for _, v in p_ml]
LM_VERB = [v[1] for _, v in p_lm]
ML_VERB = [v[1] for _, v in p_ml]
assert STEPS_G == list(G.GRID), STEPS_G
assert [f'{v:.4f}' for v in LM_SENT] == [
    '0.5050', '0.7267', '0.9517', '0.9533', '0.9633', '0.9750',
    '0.9717', '0.9950', '0.9767', '0.9800', '0.9700', '0.9883',
    '0.9900'], LM_SENT
assert [f'{v:.4f}' for v in ML_SENT] == [
    '0.5267', '0.5267', '0.5233', '0.5700', '0.7867', '0.9333',
    '0.9483', '0.9633', '0.9617', '0.9200', '0.9083', '0.8817',
    '0.9233'], ML_SENT
assert [f'{v:.4f}' for v in LM_VERB] == [
    '0.6900', '0.6920', '0.7380', '0.8340', '1.0000', '1.0000',
    '1.0000', '1.0000', '1.0000', '1.0000', '1.0000', '1.0000',
    '1.0000'], LM_VERB
assert [f'{v:.4f}' for v in ML_VERB] == [
    '0.6900', '0.6900', '0.6900', '0.8620', '1.0000', '1.0000',
    '0.9980', '1.0000', '1.0000', '1.0000', '1.0000', '1.0000',
    '1.0000'], ML_VERB
assert f'{LM_NLL[-1]:.4f}' == f'{TOK:.4f}', (LM_NLL[-1], TOK)
assert [f'{v:.4f}' for v in LM_NLL] == [
    '3.0190', '2.5814', '2.1676', '1.9400', '1.7292', '1.5766',
    '1.5673', '1.5335', '1.5026', '1.4828', '1.4758', '1.4631',
    '1.4706'], LM_NLL


# =======================================================================
# 1. Held-out loss against the computed floor.
# =======================================================================

fig, ax = plt.subplots(figsize=(1500 / 150, 820 / 150), dpi=150)
fig.patch.set_facecolor(BG)
ax.plot(STEPS_G, LM_NLL, 'o-', color=BLUE, lw=2.2, ms=5,
        label='decoder, held-out', zorder=4)
ax.axhline(OTOK, color=GREEN, ls='--', lw=1.8, zorder=3)
ax.text(11, OTOK - 0.17,
        f"the language's own entropy = {OTOK:.4f} nats/token",
        fontsize=10, color=GREEN, fontweight='bold')
ax.axhline(np.log(G.V), color=NEUTRAL, ls=':', lw=1.4, zorder=2)
ax.text(11, np.log(G.V) + 0.08,
        f'uniform guess over {G.V} tokens = {np.log(G.V):.2f} nats',
        fontsize=9.5, color=SUBTLE)
ax.annotate(f'{TOK:.4f}\n({TOK - OTOK:+.4f})',
            xy=(STEPS_G[-1], LM_NLL[-1]),
            xytext=(-14, 34), textcoords='offset points',
            fontsize=10, color=BLUE, fontweight='bold', ha='center')
ax.set_xscale('log')
TICKS = [10, 25, 50, 100, 200, 400, 800, 1500]
ax.set_xticks(TICKS)
ax.set_xticklabels([str(t) for t in TICKS])
ax.set_xlabel('pre-training step', fontsize=10.5, color=SUBTLE)
ax.set_ylabel('held-out loss (nats per token)', fontsize=10.5,
              color=SUBTLE)
ax.set_title('A Loss With a Known Floor', fontsize=15,
             fontweight='bold', color=TEXT, pad=12)
ax.legend(fontsize=10, frameon=False, loc='upper right')
ax.set_ylim(1.15, 5.05)
style_axes(ax)
fig.text(0.5, 0.058,
         'The micro-language is small enough to enumerate, so the floor '
         'is computed rather than estimated.',
         fontsize=9.5, ha='center', color=SUBTLE, fontstyle='italic')
fig.text(0.5, 0.014,
         f'Perplexity {np.exp(TOK):.3f} against an irreducible '
         f'{np.exp(OTOK):.3f}. No masked model gets this yardstick.',
         fontsize=9.5, ha='center', color=SUBTLE, fontstyle='italic')
plt.tight_layout(rect=[0, 0.095, 1, 1])
plt.savefig(OUT + 'floor_curve.png', dpi=150, facecolor=BG)
plt.close(fig)
print(f'Wrote {OUT}floor_curve.png')


# =======================================================================
# 2. The same comparison, slot by slot.
# =======================================================================

SLOT_ROWS = G.slot_nll(lm, test)
assert [f'{SLOT_ROWS[k][0]:.4f}' for k in
        ('the', 'adj', 'subj', 'verb', 'poss', 'obj', 'eos')] == [
    '0.0001', '3.2713', '4.0075', '1.4334', '0.0013', '2.4647',
    '0.0002'], SLOT_ROWS
assert [f'{SLOT_ROWS[k][1]:.4f}' for k in
        ('the', 'adj', 'subj', 'verb', 'poss', 'obj', 'eos')] == [
    '0.0000', '3.1781', '3.9688', '1.3863', '0.0000', '2.2637',
    '0.0000'], SLOT_ROWS
assert f'{SLOT_ROWS["verb"][1]:.4f}' == f'{np.log(4):.4f}'

WHAT = {
    'the': 'nothing \u2014 every sentence opens with it',
    'adj': 'an adjective, or straight to the subject',
    'subj': 'one of 12 nouns, and its number',
    'verb': 'one of 4 verbs; the number is fixed',
    'poss': 'nothing \u2014 the subject already decided',
    'obj': 'the partner noun, or any noun',
    'eos': 'nothing \u2014 the template decided the length',
}
rows = []
for k in ('the', 'adj', 'subj', 'verb', 'poss', 'obj', 'eos'):
    m, o, n = SLOT_ROWS[k]
    gap = m - o
    bg = GOOD_BG if gap < 0.05 else (MID_BG if gap < 0.15 else BAD_BG)
    rows.append([
        label(k if k != 'eos' else '[EOS]'),
        note(WHAT[k]),
        cell(f'{m:.4f}'),
        cell(f'{o:.4f}', color=GREEN, bold=True),
        cell(f'{gap:+.4f}', bg=bg,
             color=GREEN if gap < 0.05 else TEXT),
    ])
render_table(
    'floor_table.png',
    'Where the Decoder Loses Its Nats',
    f'Next-token loss on {len(test):,} held-out sentences, slot by '
    f'slot, against the exact conditional entropy of each slot',
    ['slot', 'what has to be guessed', 'model\n(nats)',
     'floor\n(nats)', 'gap'],
    [2.6, 9.4, 3.0, 3.0, 2.6], rows,
    captions=[
        ('Three slots are deterministic and the model drives them to '
         'zero. The verb is a uniform choice of four, and ln 4 = '
         f'{np.log(4):.4f}.', SUBTLE),
        ('The object is the widest gap: half the time it is the '
         "subject's signature partner, which is Part 1's structure "
         'and the hardest thing here to learn.', SUBTLE)],
    fig_px=(1760, 880))


# =======================================================================
# 3. Sample efficiency on two yardsticks.
# =======================================================================

fig, axes = plt.subplots(1, 2, figsize=(1680 / 150, 760 / 150), dpi=150)
fig.patch.set_facecolor(BG)
for ax, (a, b, ttl, base, blab) in zip(axes, [
        (LM_SENT, ML_SENT, 'Whole-sentence acceptability', 0.5,
         'chance = 0.50'),
        (LM_VERB, ML_VERB, 'Verb-number agreement', 0.69,
         'majority guess = 0.69')]):
    ax.plot(STEPS_G, a, 'o-', color=BLUE, lw=2.2, ms=5,
            label='decoder (next token)', zorder=4)
    ax.plot(STEPS_G, b, 's-', color=AMBER, lw=2.2, ms=5,
            label='encoder (masked)', zorder=4)
    ax.axhline(base, color=NEUTRAL, ls=':', lw=1.4, zorder=2)
    ax.text(11, base - 0.042, blab, fontsize=9, color=SUBTLE)
    ax.set_xscale('log')
    ax.set_xticks([10, 25, 50, 100, 200, 400, 800, 1500])
    ax.set_xticklabels(['10', '25', '50', '100', '200', '400', '800',
                        '1500'])
    ax.set_xlabel('pre-training step', fontsize=10.5, color=SUBTLE)
    ax.set_title(ttl, fontsize=12.5, fontweight='bold', color=TEXT,
                 pad=8)
    ax.set_ylim(0.44, 1.035)
    ax.legend(fontsize=9.5, frameon=False, loc='lower right')
    style_axes(ax)
axes[0].set_ylabel('accuracy', fontsize=10.5, color=SUBTLE)
fig.suptitle('Same Trunk, Same Budget, 5.25x the Targets', fontsize=15,
             fontweight='bold', color=TEXT, y=0.985)
fig.text(0.5, 0.055,
         'The flood does not buy the agreement RULE sooner \u2014 both '
         'reach a perfect verb score at step 100, and the encoder is '
         'briefly ahead.',
         fontsize=9.5, ha='center', color=SUBTLE, fontstyle='italic')
fig.text(0.5, 0.012,
         'It buys a usable sentence score far sooner: the decoder '
         'passes 95% at step 50, the encoder not until step 300.',
         fontsize=9.5, ha='center', color=SUBTLE, fontstyle='italic')
plt.tight_layout(rect=[0, 0.09, 1, 0.95])
plt.savefig(OUT + 'efficiency_curve.png', dpi=150, facecolor=BG)
plt.close(fig)
print(f'Wrote {OUT}efficiency_curve.png')


# =======================================================================
# 4. The receipt: what reading one way costs.
# =======================================================================

BASE = G.singular_base_rate(test)
bs = G.score_subject(lm, test)
bv = G.score_verb(lm, test)
bp = G.score_poss(lm, test)
es = G.score_subject(mlm, test, predict=G.mlm_slot_prediction)
ev = G.score_verb(mlm, test, predict=G.mlm_slot_prediction)
ep = G.score_poss(mlm, test, predict=G.mlm_slot_prediction)
assert f'{BASE:.4f}' == '0.6670', BASE
assert f'{bs[0]:.4f}' == '0.6670', bs[0]
assert f'{bs[3]:.4f}' == '0.0704', bs[3]
assert f'{bs[1]:.4f}' == '0.6650', bs[1]
assert f'{bv[0]:.4f}' == '1.0000', bv[0]
assert f'{bp[0]:.4f}' == '1.0000', bp[0]
assert f'{es[0]:.4f}' == f'{es[3]:.4f}' == f'{es[1]:.4f}' == '1.0000', es
assert f'{ev[0]:.4f}' == f'{ep[0]:.4f}' == '1.0000', (ev, ep)

RECEIPT = [
    ('subject number', 'right', bs[0], es[0],
     f'base rate for "singular" is {BASE:.4f}'),
    ('subject gender', 'right', bs[3], es[3],
     'nothing to the left carries it'),
    ('subject topic', 'right', bs[1], es[1],
     'ceiling is 0.5 + 0.5/3 = 0.6667'),
    ('verb number', 'left', bv[0], ev[0],
     'the subject is already in the prefix'),
    ('possessive exact', 'left', bp[0], ep[0],
     'the subject is already in the prefix'),
]
rows = []
for name, side, dec, enc, why in RECEIPT:
    rows.append([
        label(name),
        cell(side, mono=False, bold=True,
             color=RED if side == 'right' else GREEN),
        cell(f'{dec:.4f}',
             bg=BAD_BG if dec < 0.99 else GOOD_BG),
        cell(f'{enc:.4f}', bg=GOOD_BG if enc > 0.99 else BAD_BG),
        note(why),
    ])
render_table(
    'receipt_table.png',
    'The Bill for Reading One Way',
    f'Predicting one slot on {len(test):,} held-out sentences. The '
    f'decoder reads the prefix; the encoder sees the whole sentence '
    f'with that slot blanked.',
    ['slot / property', 'evidence\nlies', 'decoder\n(causal)',
     'encoder\n(bidirectional)', 'why'],
    [4.4, 2.6, 3.2, 3.6, 8.0], rows, rules=(3,),
    captions=[
        ('The causal model is not looking in the wrong place. For the '
         'top three slots there is no place to look, so it returns the '
         'base rate exactly.', SUBTLE),
        ('Where the evidence is already in the prefix, the two tie at '
         '1.0000. Bidirectionality is a located advantage, not a '
         'general one.', SUBTLE)],
    fig_px=(1780, 760))


# =======================================================================
# 5. Five routes to the same labelled task.
# =======================================================================

LAB_ROWS = []
for n in G.LABELS:
    tr = G.make_acceptability(n, np.random.default_rng(1000 + n))
    a_lm, _ = G.score_task(G.lm_logprob, lm, tr, accept)
    a_ml, _ = G.score_task(G.mlm_pseudo_logprob, mlm, tr, accept)
    f_lm = G.finetune(lm, 'last', tr, accept)
    f_ml = G.finetune(mlm, 'first', tr, accept, causal=False)
    f_sc = G.finetune(None, 'last', tr, accept)
    LAB_ROWS.append((n, a_lm, f_lm, a_ml, f_ml, f_sc))
assert [f'{v:.4f}' for r in LAB_ROWS for v in r[1:]] == [
    '0.9900', '0.4900', '0.9233', '0.5167', '0.4867',
    '0.9833', '0.5667', '0.9550', '0.7083', '0.5350',
    '0.9800', '0.6267', '0.9550', '0.8200', '0.5200',
    '0.9883', '0.8967', '0.9517', '0.9833', '0.8150'], LAB_ROWS

rows = []
for n, a_lm, f_lm, a_ml, f_ml, f_sc in LAB_ROWS:
    vals = [a_lm, f_lm, a_ml, f_ml, f_sc]
    best = max(vals)
    rows.append([label(str(n))] + [
        cell(f'{v:.4f}', bold=(v == best),
             bg=GOOD_BG if v == best else (
                 BAD_BG if v < 0.6 else (MID_BG if v < 0.9 else None)))
        for v in vals])
render_table(
    'labels_table.png',
    'Five Routes to the Same Task',
    'Agreement acceptability on 600 held-out sentences, chance 0.50. '
    'The oracle scores 1.0000, so the ceiling is 1.',
    ['labels', 'decoder\nscore, 1 pass', 'decoder\nfine-tuned',
     'encoder\nPLL, 6 passes', 'encoder\nfine-tuned',
     'from scratch\nfine-tuned'],
    [2.4, 4.0, 3.4, 4.0, 3.4, 3.6], rows,
    captions=[
        ('Fine-tuned, the ENCODER wins at every label count \u2014 '
         'which is the history. BERT owned the fine-tuning era.',
         SUBTLE),
        ('Scored with no new parameters, the DECODER wins: twenty '
         'labels spent on one threshold beat four hundred spent '
         'fine-tuning the encoder.', SUBTLE)],
    fig_px=(1760, 700))


# =======================================================================
# 6. The temperature frontier.
# =======================================================================

TEMP_ROWS = []
for t in G.TEMPS:
    q = G.sample_quality(lm, np.random.default_rng(G.SEED_SAMPLE),
                         n=G.N_SAMPLES, temp=t)
    bad = sorted(((v, k) for k, v in q['reasons'].items() if k != 'ok'),
                 reverse=True)
    TEMP_ROWS.append((t, q['grammatical'], q['unique'],
                      bad[0] if bad else (0, '\u2014')))
assert [f'{g:.4f}' for _, g, _, _ in TEMP_ROWS] == [
    '1.0000', '1.0000', '0.9990', '0.9780', '0.9450', '0.8170',
    '0.4830'], TEMP_ROWS
assert [f'{u:.4f}' for _, _, u, _ in TEMP_ROWS] == [
    '0.0010', '0.7110', '0.8330', '0.8770', '0.9060', '0.9520',
    '0.9860'], TEMP_ROWS

rows = []
for t, gram, uniq, (bn, bk) in TEMP_ROWS:
    rows.append([
        label('greedy' if t == 0 else f'{t:.1f}'),
        cell(f'{gram:.4f}',
             bg=GOOD_BG if gram > 0.97 else (
                 MID_BG if gram > 0.8 else BAD_BG)),
        cell(f'{uniq:.4f}',
             bg=BAD_BG if uniq < 0.5 else (
                 MID_BG if uniq < 0.8 else GOOD_BG)),
        note('\u2014' if bn == 0 else f'{bk} ({bn}/{G.N_SAMPLES})'),
    ])
render_table(
    'temperature_table.png',
    'Quality and Diversity Pull Against Each Other',
    f'{G.N_SAMPLES:,} sentences sampled at each temperature. The '
    f'oracle decides whether each one is in the language, and which '
    f'rule broke if not.',
    ['temperature', 'in the language', 'distinct',
     'commonest failure'],
    [3.4, 4.2, 3.4, 7.4], rows,
    captions=[
        ('Greedy decoding is not careful, it is stuck: every sample is '
         'grammatical and they are all the same sentence.', SUBTLE),
        ('Nothing in the loss curve says where the useful band is. '
         'Here it is 0.5 to 1.0, and only sampling finds it.', SUBTLE)],
    fig_px=(1740, 840))


# =======================================================================
# 7. The distribution, computed rather than sampled.
# =======================================================================

M = G.model_on_language(lm)
_, WANT = G.exact_marginals()
assert f'{M["mass"]:.4f}' == '0.9770', M['mass']
assert f'{M["kl"]:.4f}' == '0.2987', M['kl']
assert f'{M["tv"]:.4f}' == '0.2765', M['tv']
assert [f'{v:.4f}' for v in M['marginals']] == [
    '0.4582', '0.3243', '0.2247', '0.4785', '0.2595',
    '0.3679'], M['marginals']
assert [f'{v:.4f}' for v in WANT] == [
    '0.5000', '0.3208', '0.2292', '0.4792', '0.2832',
    '0.3333'], WANT

rows = []
for nm, a, b in zip(M['names'], M['marginals'], WANT):
    d = a - b
    bg = GOOD_BG if abs(d) < 0.01 else (
        MID_BG if abs(d) < 0.03 else BAD_BG)
    rows.append([label(nm), cell(f'{a:.4f}'),
                 cell(f'{b:.4f}', color=GREEN, bold=True),
                 cell(f'{d:+.4f}', bg=bg)])
render_table(
    'distribution_table.png',
    'Did It Learn the Distribution, or Just the Grammar?',
    f'The model assigns a probability to each of the '
    f'{len(G.SENT_P):,} sentences in the language, so these are '
    f'summed exactly \u2014 no sampling error at all.',
    ['statistic', 'model', 'language', 'difference'],
    [6.4, 3.4, 3.4, 3.4], rows,
    captions=[
        (f'Probability mass on the language {M["mass"]:.4f} (sampling '
         f'at temperature 1.0 said {TEMP_ROWS[3][1]:.4f}). '
         f'KL(language || model) = {M["kl"]:.4f} nats.', SUBTLE),
        ('The two widest misses are the adjective coin flip and the '
         "object's partner rate \u2014 exactly the two slots with the "
         'widest loss gap.', SUBTLE)],
    fig_px=(1700, 760))

print()
print('all seven images written, every cell asserted against the '
      'frozen run')
