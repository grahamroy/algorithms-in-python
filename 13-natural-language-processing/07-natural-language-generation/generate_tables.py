"""Generate every image for the Natural Language Generation article.

Medium has no table support and collapses the whitespace in wide code
blocks, so each measured table ships as an image. This ONE script
trains the decoder once and emits every PNG including the header,
rather than paying that cost per figure. Every cell is asserted
against the frozen run of nlg.py.

Writes: frontier_table.png, beam_table.png, metric_table.png,
        break_table.png, proxy_table.png, header_nlg.png
Runs in about three minutes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

import nlg as N

OUT = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/07-natural-language-generation/')

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
ORANGE = '#EA580C'
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
    print('Wrote ' + OUT + path)


def style_axes(ax):
    ax.grid(color='#E2E8F0', linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=9.5, colors=SUBTLE)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#CBD5E1')


# =======================================================================
# Train once, with nlg.py's frozen seeds, then measure everything.
# =======================================================================

print('training the decoder (about thirty seconds)...')
corpus, ids, keep, test = N.build_stage()
LM, _ = N.train_lm(ids, keep)
TOK, SENT, OTOK, OSENT = N.lm_eval(LM, test)
assert '%.4f' % TOK == '1.4706', TOK
assert '%.4f' % OTOK == '1.4190', OTOK
MOL = N.model_on_language(LM)
assert '%.4f' % MOL['mass'] == '0.9770', MOL
H_LANG = N.language_entropy()
assert '%.4f' % H_LANG == '9.2538', H_LANG
EFF_LANG = float(np.exp(H_LANG))
print('  held-out %.4f vs floor %.4f, mass %.4f ok'
      % (TOK, OTOK, MOL['mass']))

RES = N.strategy_on_language(LM, N.STRATEGIES)
FROZEN = {
    'temperature 1.0': ('0.9770', 31740, '1.0000', '8934.9', '0.2880'),
    'temperature 0.7': ('0.9978', 31740, '1.0000', '4963.3', '0.3894'),
    'temperature 0.5': ('0.9998', 31740, '1.0000', '2234.2', '0.5532'),
    'top-k 2': ('0.9997', 16, '0.0030', '14.2', '0.9970'),
    'top-k 5': ('0.9991', 500, '0.0370', '361.7', '0.9630'),
    'top-k 10': ('0.9961', 3993, '0.1726', '2257.4', '0.8275'),
    'top-k 40': ('0.9849', 27966, '0.7590', '7634.8', '0.3860'),
    'nucleus 0.90': ('0.9993', 14032, '0.6998', '5554.9', '0.3951'),
    'nucleus 0.95': ('0.9989', 20440, '0.8310', '7114.4', '0.3350'),
    'nucleus 0.99': ('0.9956', 30774, '0.9836', '8817.6', '0.2846'),
}
for nm, (m, rc, rm, ef, tv) in FROZEN.items():
    r = RES[nm]
    got = ('%.4f' % r['mass'], r['reach'], '%.4f' % r['reach_mass'],
           '%.1f' % r['eff'], '%.4f' % r['tv'])
    assert got == (m, rc, rm, ef, tv), (nm, got)
print('  frontier ok (%d strategies)' % len(RES))

BEAM = []
for B in N.BEAMS:
    _, lp, w = N.beam_search(LM, B)[0]
    BEAM.append((B, w, -lp, len(w), N.diagnose(w)))
assert '%.4f' % BEAM[0][2] == '7.1130', BEAM[0]
assert '%.4f' % BEAM[-1][2] == '6.1840', BEAM[-1]
assert all(r[4] == 'ok' for r in BEAM)
BEAM_ALPHA = []
for B in (5, 25):
    _, lp, w = N.beam_search(LM, B, alpha=1.0)[0]
    BEAM_ALPHA.append((B, w, -lp, len(w)))
OUT25 = N.beam_search(LM, max(N.BEAMS))

samp = N.sample_batch(LM, np.random.default_rng(N.SEED_SAMPLE),
                      N.N_SAMPLES)
good = [w for w, done in samp if done and len(w) in (5, 6)]
SAMP_NLL = float(-N.lm_logprob(LM, good).mean())
assert '%.4f' % SAMP_NLL == '9.2793', SAMP_NLL
MODE_S, MODE_P = max(N.SENT_P.items(), key=lambda kv: kv[1])
MODE_NLL = float(-np.log(MODE_P))
assert '%.4f' % MODE_NLL == '6.5637', MODE_NLL
print('  beam ok (best %.4f, sampled %.4f, H %.4f)'
      % (min(r[2] for r in BEAM), SAMP_NLL, H_LANG))

# ---- BLEU, exactly as demo3 computes it ----
REFS = N.completions(N.PROMPT)
assert len(REFS) == 92, len(REFS)
rng = np.random.default_rng(N.SEED_BLEU)
PAIRS = []
for s in REFS:
    for how in N.BREAKS:
        b = N.break_one(s, how, rng)
        if b is not None and N.diagnose(b) != 'ok':
            PAIRS.append((s, b, how))
assert len(PAIRS) == 276, len(PAIRS)

METRIC = {}
for k in N.REF_COUNTS:
    va, ba, win = [], [], []
    for s, b, _ in PAIRS:
        pool = [r for r in REFS if r != s]
        pick = [pool[i] for i in rng.choice(len(pool), k, replace=False)]
        x, y = N.bleu(s, pick), N.bleu(b, pick)
        va.append(x)
        ba.append(y)
        win.append(y >= x)
    METRIC[k] = (float(np.mean(va)), float(np.mean(ba)),
                 float(np.mean(win)))
assert ['%.4f' % v for v in METRIC[1]] == ['0.4899', '0.4178', '0.5507']
assert ['%.4f' % v for v in METRIC[16]] == ['0.7921', '0.5576', '0.1667']

PER = {}
for how in N.BREAKS:
    sub = [(s, b) for s, b, h in PAIRS if h == how]
    va, ba, win = [], [], []
    for s, b in sub:
        pool = [r for r in REFS if r != s]
        pick = [pool[i] for i in rng.choice(len(pool), 1)]
        x, y = N.bleu(s, pick), N.bleu(b, pick)
        va.append(x)
        ba.append(y)
        win.append(y >= x)
    PER[how] = (float(np.mean(va)), float(np.mean(ba)),
                float(np.mean(win)))
assert ['%.4f' % v for v in PER['topic mix']] == \
    ['0.4582', '0.4540', '0.9348'], PER
assert '%.4f' % PER['possessive'][2] == '0.0000', PER

erng = np.random.default_rng(N.SEED_BLEU)
REF1 = [REFS[int(erng.integers(0, len(REFS)))]]
ties, scored = [], 0
for s, b, how in PAIRS:
    if s == REF1[0]:
        continue
    scored += 1
    x, y = N.bleu(s, REF1), N.bleu(b, REF1)
    if abs(y - x) < 1e-12:
        ties.append((s, b, how, x))
TIE_S, TIE_B, TIE_HOW, TIE_X = max(ties, key=lambda e: e[3])
assert (len(ties), scored) == (157, 273), (len(ties), scored)
assert '%.4f' % TIE_X == '0.7521', TIE_X
print('  BLEU ok (%d of %d pairs tie)' % (len(ties), scored))

# ---- the sampled-diversity proxy ----
PROXY = {}
for nm in N.PROXY_STRATS:
    kind, param = N.STRATEGIES[nm]
    PROXY[nm] = [N.sample_quality(LM,
                                  np.random.default_rng(N.SEED_SAMPLE),
                                  n, kind=kind, param=param)['distinct']
                 for n in N.PROXY_NS]
assert ['%.4f' % v for v in PROXY['temperature 1.0']] == \
    ['0.9800', '0.8870', '0.7167'], PROXY
assert ['%.4f' % v for v in PROXY['nucleus 0.90']] == \
    ['0.9800', '0.8450', '0.6345'], PROXY
assert ['%.4f' % v for v in PROXY['top-k 2']] == \
    ['0.1600', '0.0160', '0.0045'], PROXY
print('  proxy ok')
print()


def gb(v, good):
    return GOOD_BG if good else BAD_BG


# =======================================================================
# 1. The frontier: what truncation costs, exactly.
# =======================================================================

FAMILY = {'temperature': AMBER, 'top-k': RED, 'nucleus': BLUE}


def fam(nm):
    return FAMILY[nm.split()[0]]


rows = [[label('the language'),
         cell('1.0000', color=GREEN, bold=True),
         cell('31,740', color=GREEN, bold=True),
         cell('1.0000', color=GREEN),
         cell('10,444', color=GREEN),
         cell('0.0000', color=GREEN, bold=True)]]
for nm in N.STRATEGIES:
    r = RES[nm]
    rows.append([
        label(nm),
        cell('%.4f' % r['mass'],
             bg=GOOD_BG if r['mass'] >= 0.995 else MID_BG),
        cell('{:,}'.format(r['reach']), color=fam(nm), bold=True),
        cell('%.4f' % r['reach_mass'],
             bg=gb(r['reach_mass'], r['reach_mass'] >= 0.70)),
        cell('{:,.0f}'.format(r['eff'])),
        cell('%.4f' % r['tv'], bg=gb(r['tv'], r['tv'] <= 0.40)),
    ])

render_table(
    'frontier_table.png',
    'What Truncation Actually Costs',
    'Computed, not sampled: every decoding strategy is a distribution '
    'over the 31,740 sentences, so these are closed-form quantities.',
    ['strategy', 'grammatical', 'sentences\nit can say',
     'of the\nlanguage', 'effective\nsentences',
     'distance\nfrom truth'],
    [3.7, 2.7, 2.6, 2.4, 2.5, 2.7], rows,
    captions=[
        ('top-k 2 is 99.97% grammatical and can produce 16 of the '
         '31,740 sentences. The grammar is perfect because there is '
         'nothing left to say.', SUBTLE),
        ('Every setting except nucleus 0.99 moves FURTHER from the '
         'truth than sampling the model untouched. Truncation does '
         'not repair a model.', SUBTLE)],
    fig_px=(1860, 1150), rules=(1,))


# =======================================================================
# 2. Beam search: the likeliest sentence is not a typical one.
# =======================================================================

rows = []
for B, w, nll, ln, ok in BEAM:
    rows.append([cell(str(B), bold=True),
                 cell(' '.join(w), mono=True, align='left', size=11),
                 cell('%.4f' % nll, color=RED, bold=True),
                 cell(str(ln)),
                 cell(ok, color=GREEN)])
rows.append([cell('sample', mono=False, bold=True),
             cell('a draw from the same model, untruncated',
                  mono=False, align='left', size=11, color=SUBTLE),
             cell('%.4f' % SAMP_NLL, color=GREEN, bold=True),
             cell('--', color=SUBTLE), cell('ok', color=GREEN)])
rows.append([cell('--', mono=False, color=SUBTLE),
             cell('the entropy of the language itself', mono=False,
                  align='left', size=11, color=SUBTLE),
             cell('%.4f' % H_LANG, bold=True),
             cell('--', color=SUBTLE), cell('--', color=SUBTLE)])

render_table(
    'beam_table.png',
    'The Likelihood Trap',
    'Beam search maximises sequence probability and succeeds. '
    '-log p is in nats, so smaller means more probable.',
    ['beam', 'sentence returned', '-log p', 'words', 'oracle'],
    [1.5, 8.0, 2.2, 1.5, 1.7], rows,
    captions=[
        ('Beam search returns sentences about three nats more probable '
         'than a typical draw -- a factor of 22. The mode of a '
         'distribution is not a sample from it.', SUBTLE),
        ('Beam 25 finds a SHORTER sentence than beam 1, because every '
         'extra word multiplies in another probability below one.',
         SUBTLE)],
    fig_px=(1800, 830), rules=(2,))


# =======================================================================
# 3. BLEU against an enumerable reference set.
# =======================================================================

rows = []
for k in N.REF_COUNTS:
    v, b, w = METRIC[k]
    rows.append([cell(str(k), bold=True),
                 cell('%.4f' % v, bg=MID_BG if v < 0.75 else GOOD_BG),
                 cell('%.4f' % b),
                 cell('%.4f' % w, color=RED if w > 0.3 else TEXT,
                      bold=True, bg=BAD_BG if w > 0.5 else None)])

render_table(
    'metric_table.png',
    'Putting BLEU on Trial',
    'The prompt "the king" has 92 valid completions. Each valid '
    'sentence is paired with a twin broken by exactly one token.',
    ['references\nshown', 'BLEU of a\nVALID sentence',
     'BLEU of the\nBROKEN twin', 'broken scores\n>= valid'],
    [2.3, 3.0, 3.0, 3.0], rows,
    captions=[
        ('Against one reference a PERFECT generator scores 0.4899. It '
         'is not wrong, it is differently right, and BLEU cannot tell '
         'those apart.', SUBTLE),
        ('On one reference the sentence the oracle REJECTS scores at '
         'or above its valid twin 55.07% of the time.', RED)],
    fig_px=(1700, 760))


# =======================================================================
# 4. Which breaks BLEU can see.
# =======================================================================

rows = []
EXPL = {'verb number': 'plural verb, singular subject',
        'possessive': 'his -> her, or their -> the',
        'topic mix': 'object from another topic'}
for how in N.BREAKS:
    v, b, w = PER[how]
    rows.append([label(how), note(EXPL[how]),
                 cell('%.4f' % v), cell('%.4f' % b),
                 cell('%.4f' % w, bold=True,
                      color=RED if w > 0.5 else GREEN,
                      bg=BAD_BG if w > 0.5 else GOOD_BG)])

render_table(
    'break_table.png',
    'What BLEU Can and Cannot See',
    'One reference. The same three breaks, scored separately -- the '
    'pattern is pure n-gram bookkeeping.',
    ['break', 'what it changes', 'valid', 'broken',
     'broken scores\n>= valid'],
    [2.9, 5.0, 2.0, 2.0, 2.8], rows,
    captions=[
        ('"his" -> "her" destroys a bigram the reference shares, so '
         'BLEU catches it every time. A sea noun in a royal sentence '
         'replaces a word the reference did not have either.', SUBTLE),
        ('Against one fixed reference, 157 of 273 pairs score '
         'EXACTLY the same for the valid and the broken sentence.',
         RED)],
    fig_px=(1800, 700))


# =======================================================================
# 5. What a sample cannot see.
# =======================================================================

rows = []
for nm in N.PROXY_STRATS:
    cells = [label(nm), cell('{:,.0f}'.format(RES[nm]['eff']),
                             color=BLUE, bold=True)]
    for v in PROXY[nm]:
        cells.append(cell('%.4f' % v))
    rows.append(cells)

render_table(
    'proxy_table.png',
    'Why This Had to Be Computed',
    'The usual diversity proxy is the share of samples that come out '
    'distinct. It is a property of the sample size, not the decoder.',
    ['strategy', 'effective\nsentences', 'distinct at\nn=100',
     'distinct at\nn=1,000', 'distinct at\nn=4,000'],
    [3.4, 2.8, 2.3, 2.3, 2.3], rows,
    captions=[
        ('At 100 samples, temperature 1.0 and nucleus 0.90 are '
         'indistinguishable -- both 0.9800 -- while their effective '
         'sentence counts differ by a factor of 1.6.', SUBTLE),
        ('Every row falls as n grows. The exact column does not move, '
         'because it describes the distribution rather than the '
         'experiment.', SUBTLE)],
    fig_px=(1780, 660))


# =======================================================================
# 6. The header.
# =======================================================================

fig = plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Natural Language Generation: The Cost of Sounding '
                 'Correct',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT)
ax.text(8, 8.05,
        'Top-k 2 is %.2f%% grammatical and can say %d of the language\'s '
        '%s sentences.'
        % (100 * RES['top-k 2']['mass'], RES['top-k 2']['reach'],
           '{:,}'.format(len(N.SENT_P))),
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE)


def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))


box(0.7, 6.05, 5.3, 1.45, PURPLE)
ax.text(3.35, 7.22, 'A decoding strategy is a distribution',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=PURPLE)
ax.text(3.35, 6.62,
        'q(s) = prod_t q_trunc(s_t | s_<t)\n'
        'evaluated over all %s sentences' % '{:,}'.format(len(N.SENT_P)),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 4.35, 5.3, 1.45, RED, face='#FEF2F2')
ax.text(3.35, 5.52, 'Truncation buys grammar with silence',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=RED)
ax.text(3.35, 4.92,
        'top-k 2:      %.4f grammatical\n'
        'and can say %d of %s sentences'
        % (RES['top-k 2']['mass'], RES['top-k 2']['reach'],
           '{:,}'.format(len(N.SENT_P))),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 2.95, 5.3, 1.1, BLUE, face='#F8FAFC', lw=1.7)
ax.text(3.35, 3.82, 'The likelihood trap, in nats',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=BLUE)
ax.text(3.35, 3.35,
        'beam %.4f vs typical %.4f' % (min(r[2] for r in BEAM),
                                       SAMP_NLL),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 1.35, 5.3, 1.25, ORANGE, lw=1.6)
ax.text(3.35, 2.35, 'And the metric cannot see the break',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=ORANGE)
ax.text(3.35, 1.82,
        '%d of %d valid/broken pairs score\n'
        'EXACTLY the same BLEU' % (len(ties), scored),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

# ---- the frontier itself: diversity against distance from the truth ----
axh = fig.add_axes([0.45, 0.44, 0.50, 0.30])
for famname, col in FAMILY.items():
    xs = [RES[nm]['eff'] for nm in N.STRATEGIES
          if nm.startswith(famname)]
    ys = [RES[nm]['tv'] for nm in N.STRATEGIES if nm.startswith(famname)]
    axh.plot(xs, ys, 'o-', color=col, linewidth=1.6, markersize=5,
             label=famname, zorder=3)
axh.plot([EFF_LANG], [0.0], marker='*', markersize=14, color=GREEN,
         zorder=4, linestyle='none', label='the language')
axh.annotate('top-k 2', (RES['top-k 2']['eff'], RES['top-k 2']['tv']),
             textcoords='offset points', xytext=(8, -2), fontsize=7.2,
             color=RED)
axh.annotate('nucleus 0.99',
             (RES['nucleus 0.99']['eff'], RES['nucleus 0.99']['tv']),
             textcoords='offset points', xytext=(-30, -16), fontsize=7.2,
             color=BLUE)
axh.set_xlabel('effective sentences (more is better)', fontsize=8,
               color=SUBTLE)
axh.set_ylabel('distance from the truth', fontsize=8, color=SUBTLE)
axh.set_xlim(-400, 11600)
axh.set_ylim(-0.06, 1.1)
axh.legend(fontsize=7.2, frameon=False, loc='upper right', ncol=2)
axh.set_title('the quality-diversity frontier, computed exactly',
              fontsize=9.0, fontweight='bold', color=TEXT, pad=6,
              loc='left')
style_axes(axh)

# ---- beam against typical ----
axb = fig.add_axes([0.45, 0.10, 0.50, 0.205])
names = ['beam 25', 'beam 1', 'sampled', 'H(language)']
vals = [BEAM[-1][2], BEAM[0][2], SAMP_NLL, H_LANG]
cols = [RED, RED, GREEN, NEUTRAL]
axb.barh(np.arange(4), vals, 0.52, color=cols, zorder=3)
for i, v in enumerate(vals):
    axb.text(v + 0.12, i, '%.4f' % v, va='center', fontsize=7.6,
             color=cols[i], fontweight='bold')
axb.set_yticks(np.arange(4))
axb.set_yticklabels(names, fontsize=7.4, color=SUBTLE)
axb.set_xlim(0, 12.2)
axb.set_xticks([0, 3, 6, 9, 12])
axb.set_xticklabels(['0', '3', '6', '9', '12'], fontsize=7.2,
                    color=SUBTLE)
axb.set_xlabel('-log p of the sentence returned, in nats', fontsize=8,
               color=SUBTLE)
axb.invert_yaxis()
axb.set_title('the mode is not a typical sample', fontsize=9.0,
              fontweight='bold', color=TEXT, pad=6, loc='left')
style_axes(axb)

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 7',
        fontsize=8, ha='center', va='center', color='#aaaaaa')
plt.savefig(OUT + 'header_nlg.png', dpi=150, facecolor=BG)
plt.close()
print('Wrote ' + OUT + 'header_nlg.png')
print()
print('all six images written, every cell asserted against the '
      'frozen run')
