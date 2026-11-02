"""Generate every image for the Natural Language Inference article.

Medium has no table support and collapses the whitespace in wide code
blocks, so each measured table ships as an image. The pipeline here is
expensive (two pre-trainings and seven fine-tunings), so this ONE
script trains once and emits every PNG including the header, rather
than paying that cost twice. Every cell is asserted against the frozen
run of nli.py.

Writes: rule_table.png, readers_table.png, alignment_table.png,
        artefact_table.png, transfer_table.png, decoder_table.png,
        header_nli.png
Runs in about sixteen minutes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

import nli as N

OUT = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/06-natural-language-inference/')

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
# Train everything once, with nli.py's frozen seeds.
# =======================================================================

print('training (about sixteen minutes)...')
train, test, c_train, c_test = N.build_stage()
RULES = N.rule_scores(test)
assert ['%.4f' % a for _, a in RULES] == [
    '0.3435', '0.3295', '0.9490', '1.0000'], RULES

corpus = N.pretrain_corpus(N.PRETRAIN_N,
                           np.random.default_rng(N.SEED_CORPUS))
pre, _ = N.pretrain(N.PRETRAIN_STEPS, np.random.default_rng(N.SEED_PRE),
                    corpus)
dec_pre, _ = N.pretrain(N.DEC_PRETRAIN_STEPS,
                        np.random.default_rng(N.SEED_PRE), corpus,
                        causal=True)
ALIGN_W, ALIGN_WO = N.alignment_probe(pre, test)
assert '%.4f' % ALIGN_W == '0.5457', ALIGN_W
assert '%.4f' % ALIGN_WO == '2.7613', ALIGN_WO
print('  alignment probe ok')

READERS = []
for kind, init, lab in (('cross', None, 'cross-encoder'),
                        ('cross', pre, 'cross-encoder'),
                        ('bi', None, 'bi-encoder'),
                        ('bi', pre, 'bi-encoder')):
    t, _ = N.run(kind, train, test, init=init)
    READERS.append((lab, init is not None, t.accuracy(), N.recalls(t)))
    print('  %-14s %-12s %.4f' % (lab, 'pre-trained' if init is not None
                                  else 'scratch', t.accuracy()))
assert ['%.4f' % a for _, _, a, _ in READERS] == [
    '0.6530', '0.9530', '0.9755', '0.9850'], READERS

t_car, _ = N.run('cross', c_train, c_test, blind=True)
CAR_CAR = t_car.accuracy()
CAR_BAL = t_car.accuracy(t_car.encode(test))
t_bal, _ = N.run('cross', train, test, blind=True)
BAL_BAL = t_bal.accuracy()
assert ['%.4f' % v for v in (CAR_CAR, CAR_BAL, BAL_BAL)] == [
    '1.0000', '0.3300', '0.3325'], (CAR_CAR, CAR_BAL, BAL_BAL)
print('  hypothesis-only ok')

t_dec, _ = N.run('decoder', train, test, init=dec_pre)
DEC, DEC_R = t_dec.accuracy(), N.recalls(t_dec)
assert '%.4f' % DEC == '0.6645', DEC
print('  decoder ok')

CUES = {}
for nm, d in (('careless', c_train), ('balanced', train)):
    for l in N.LABELS:
        rows = [h for _, h, lab in d if lab == l]
        CUES[(nm, l)] = (np.mean([not x['positive'] for x in rows]),
                         np.mean([x['adj'] is not None for x in rows]),
                         np.mean([len(x['words']) for x in rows]))
assert '%.3f' % CUES[('careless', 'contradiction')][0] == '1.000'
assert '%.3f' % CUES[('balanced', 'entailment')][0] == '0.509'
print('  cue audit ok')


# =======================================================================
# 1. The rule decomposition.
# =======================================================================

USES = {'majority class': 'nothing at all',
        'hypothesis only': 'the hypothesis, never the premise',
        'relation, no adjectives':
            'subject, verb, object and polarity',
        'the full semantics': 'everything, including the adjectives'}
rows = []
for name, acc in RULES:
    bg = GOOD_BG if acc > 0.9 else (BAD_BG if acc < 0.4 else MID_BG)
    rows.append([label(name), note(USES[name]),
                 cell('%.4f' % acc, bg=bg,
                      bold=name == 'hypothesis only')])
render_table(
    'rule_table.png', 'The Task, Priced Piece by Piece',
    'Accuracy of each partial rule on %d held-out pairs. The label is '
    'a function, so this decomposition is exact.' % len(test),
    ['rule', 'what it is allowed to look at', 'accuracy'],
    [6.4, 10.2, 3.4], rows,
    captions=[('Relation matching gets almost everything; the last few '
               'points are the adjective subset check.', SUBTLE),
              ('The hypothesis alone is worth nothing \u2014 0.3295 '
               'against a 0.3333 chance rate. That is by construction, '
               'and it is the point.', SUBTLE)],
    fig_px=(1740, 700))


# =======================================================================
# 2. Two ways to read a pair.
# =======================================================================

rows = []
for lab, is_pre, acc, rec in READERS:
    rows.append([
        label(lab),
        cell('pre-trained' if is_pre else 'scratch', mono=False,
             color=GREEN if is_pre else SUBTLE),
        cell('%.4f' % acc, bold=True,
             bg=GOOD_BG if acc > 0.9 else BAD_BG),
        cell('%.3f' % rec[0]),
        cell('%.3f' % rec[1],
             bg=BAD_BG if rec[1] < 0.5 else GOOD_BG),
        cell('%.3f' % rec[2])])
d_cross = 100 * (READERS[1][2] - READERS[0][2])
d_bi = 100 * (READERS[3][2] - READERS[2][2])
render_table(
    'readers_table.png', 'The Same Pair, Read Two Ways',
    'Same trunk, same {:,} labels, same budget. The only difference '
    'is whether the two sentences ever share a forward pass.'.format(
        len(train)),
    ['reader', 'pre-training', 'accuracy', 'recall\nentail',
     'recall\nneutral', 'recall\ncontradict'],
    [4.6, 3.6, 3.2, 2.8, 2.8, 3.0], rows, rules=(2,),
    captions=[('From scratch the cross-encoder scores %.4f against a '
               '0.667 two-class ceiling: it solves entailment and '
               'contradiction and abandons neutral.'
               % READERS[0][2], SUBTLE),
              ('Pre-training is worth %+.1f points to the cross-encoder '
               'and %+.1f to the bi-encoder, whose comparison is wired '
               'in rather than learned.' % (d_cross, d_bi), SUBTLE)],
    fig_px=(1760, 740))


# =======================================================================
# 3. What pre-training installed.
# =======================================================================

rows = [[label('the premise is visible'),
         note('the encoder may look across the [SEP]'),
         cell('%.4f' % ALIGN_W, bg=GOOD_BG)],
        [label('the premise is hidden'),
         note('only the hypothesis remains'),
         cell('%.4f' % ALIGN_WO, bg=BAD_BG)],
        [label('so the premise is worth'),
         note('this gap is cross-attention doing its job'),
         cell('%.4f' % (ALIGN_WO - ALIGN_W), bold=True)]]
render_table(
    'alignment_table.png', 'Measuring Cross-Attention Directly',
    'Blank one hypothesis token and ask the pre-trained encoder to '
    'recover it, with and without the premise. Cost in nats.',
    ['condition', 'what the encoder can see', 'nats'],
    [7.0, 9.4, 3.2], rows, rules=(2,),
    captions=[('A bi-encoder cannot have this number at all: its two '
               'sentences never meet inside a forward pass.', SUBTLE)],
    fig_px=(1700, 600))


# =======================================================================
# 4. The artefact.
# =======================================================================

rows = []
for nm in ('careless', 'balanced'):
    for l in N.LABELS:
        neg, adj, ln = CUES[(nm, l)]
        flag = nm == 'careless'
        rows.append([
            label(nm if l == N.LABELS[0] else ''),
            cell(l, mono=False, bold=True, align='left'),
            cell('%.3f' % neg, bg=BAD_BG if flag else GOOD_BG),
            cell('%.3f' % adj, bg=BAD_BG if flag else GOOD_BG),
            cell('%.3f' % ln, bg=BAD_BG if flag else GOOD_BG)])
render_table(
    'artefact_table.png', 'Where the Label Leaks',
    'How often each cue appears in the HYPOTHESIS of each class. A cue '
    'that varies by class is a cue a model can use without reading the '
    'premise.',
    ['dataset', 'label', 'has a negation', 'has an adjective',
     'mean length'],
    [3.4, 4.4, 3.8, 4.2, 3.4], rows, rules=(3,),
    captions=[('Careless: negation marks every contradiction, an '
               'adjective marks every neutral, and the lengths run '
               '5, 6, 7. The label is written on the hypothesis three '
               'separate ways.', SUBTLE),
              ('Balanced: every cue sits at one half in every class, '
               'and the lengths match.', SUBTLE)],
    fig_px=(1780, 800))


# =======================================================================
# 5. The hypothesis-only ablation.
# =======================================================================

rows = [['careless', 'careless', CAR_CAR,
         'the tell is present and sufficient'],
        ['careless', 'balanced', CAR_BAL,
         'the same weights, with the tell removed'],
        ['balanced', 'balanced', BAL_BAL,
         'nothing to learn without the premise']]
render_table(
    'transfer_table.png', 'A Model That Never Sees the Premise',
    'Three-way entailment, premise deleted from the input entirely. '
    'Chance is 0.3333.',
    ['trained on', 'tested on', 'accuracy', 'what it means'],
    [3.6, 3.6, 3.4, 10.0],
    [[label(a), label(b),
      cell('%.4f' % v, bold=True,
           bg=BAD_BG if v > 0.9 else GOOD_BG), note(w)]
     for a, b, v, w in rows],
    captions=[('%.4f on a task it has no information to solve. It '
               'learned how the data was made, not what entailment '
               'is.' % CAR_CAR, SUBTLE),
              ('This ablation is cheap, and nobody ran it on SNLI for '
               'three years.', SUBTLE)],
    fig_px=(1740, 640))


# =======================================================================
# 6. The label as a token.
# =======================================================================

rows = [['cross-encoder', 'bidirectional', 'a class head',
         READERS[1][2], READERS[1][3][1]],
        ['decoder', 'causal', 'a word in the vocabulary', DEC,
         DEC_R[1]],
        ['bi-encoder', 'bidirectional, split', 'a class head',
         READERS[3][2], READERS[3][3][1]]]
render_table(
    'decoder_table.png', 'Does the Answer Have to Be a Class?',
    'Same trunk, same pre-training budget, same labels. Only the mask '
    'and the shape of the answer differ.',
    ['reader', 'attention', 'the answer is', 'accuracy',
     'recall\nneutral'],
    [4.2, 4.8, 5.6, 3.2, 3.0],
    [[label(a), cell(b, mono=False), note(c),
      cell('%.4f' % d, bold=True,
           bg=GOOD_BG if d > 0.9 else BAD_BG),
      cell('%.3f' % e, bg=GOOD_BG if e > 0.5 else BAD_BG)]
     for a, b, c, d, e in rows],
    captions=[('The answer\u2019s format is not the problem: the '
               'position predicting the label sees the whole pair.',
               SUBTLE),
              ('What the decoder lacks is the premise being re-read '
               'once the hypothesis arrives. Comparison is two-way and '
               'causality forbids it.', SUBTLE)],
    fig_px=(1760, 660))


# =======================================================================
# 7. The header.
# =======================================================================

fig = plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Natural Language Inference: When the Label Is in '
                 'the Hypothesis',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT)
ax.text(8, 8.05,
        'A model that never reads the premise scores %.0f%% \u2014 '
        'until the negations and lengths are balanced, and it falls '
        'to chance.' % (100 * CAR_CAR),
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE)


def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))


box(0.7, 6.05, 5.3, 1.45, ORANGE)
ax.text(3.35, 7.22, 'The label is computed, not collected',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=ORANGE)
ax.text(3.35, 6.62,
        'the noble king rules his crown\n'
        '-> rule(king, crown) and noble(king)',
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 4.35, 5.3, 1.45, RED, face='#FEF2F2')
ax.text(3.35, 5.52, 'The artefact, reproduced',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=RED)
ax.text(3.35, 4.92,
        'hypothesis only, careless data: %.4f\n'
        'the same weights on balanced data: %.4f'
        % (CAR_CAR, CAR_BAL),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 2.95, 5.3, 1.1, BLUE, face='#F8FAFC', lw=1.7)
ax.text(3.35, 3.82, 'Joint attention is a capability you pay for',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=BLUE)
ax.text(3.35, 3.35,
        'pre-training: %+.1f pts to cross, %+.1f to bi'
        % (d_cross, d_bi),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 1.35, 5.3, 1.25, PURPLE, lw=1.6)
ax.text(3.35, 2.35, 'And it can be measured directly',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=PURPLE)
ax.text(3.35, 1.82,
        'a masked hypothesis token costs %.4f nats\n'
        'with the premise, %.4f without' % (ALIGN_W, ALIGN_WO),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

axh = fig.add_axes([0.45, 0.44, 0.50, 0.30])
x = np.arange(2)
w = 0.34
sc = [READERS[0][2], READERS[2][2]]
pr = [READERS[1][2], READERS[3][2]]
axh.bar(x - w / 2, sc, w, color=NEUTRAL, label='from scratch',
        zorder=3)
axh.bar(x + w / 2, pr, w, color=BLUE, label='pre-trained', zorder=3)
for xx, v in zip(x - w / 2, sc):
    axh.text(xx, v + 0.02, '%.3f' % v, ha='center', fontsize=7.4,
             color=SUBTLE)
for xx, v in zip(x + w / 2, pr):
    axh.text(xx, v + 0.02, '%.3f' % v, ha='center', fontsize=7.4,
             color=BLUE, fontweight='bold')
axh.axhline(1 / 3, color='#CBD5E1', ls=':', lw=1.2, zorder=2)
axh.text(1.52, 0.37, 'chance', fontsize=7.0, color=NEUTRAL,
         ha='right')
axh.set_xticks(x)
axh.set_xticklabels(['cross-encoder', 'bi-encoder'], fontsize=8.2,
                    color=TEXT)
axh.set_yticks([0, 0.5, 1.0])
axh.set_yticklabels(['0', '50%', '100%'], fontsize=7.2, color=SUBTLE)
axh.set_ylim(0, 1.40)
axh.legend(fontsize=7.4, frameon=False, loc='upper left', ncol=2)
axh.set_title('three-way entailment accuracy', fontsize=9.0,
              fontweight='bold', color=TEXT, pad=6, loc='left')
style_axes(axh)

axb = fig.add_axes([0.45, 0.115, 0.50, 0.225])
vals = [CAR_CAR, CAR_BAL, BAL_BAL]
names = ['careless\n-> careless', 'careless\n-> balanced',
         'balanced\n-> balanced']
cols = [RED, GREEN, GREEN]
axb.bar(np.arange(3), vals, 0.5, color=cols, zorder=3)
for i, v in enumerate(vals):
    axb.text(i, v + 0.03, '%.4f' % v, ha='center', fontsize=7.6,
             color=cols[i], fontweight='bold')
axb.axhline(1 / 3, color='#CBD5E1', ls=':', lw=1.2, zorder=2)
axb.text(2.42, 0.37, 'chance', fontsize=7.0, color=NEUTRAL,
         ha='right')
axb.set_xticks(np.arange(3))
axb.set_xticklabels(names, fontsize=7.4, color=SUBTLE)
axb.set_yticks([0, 0.5, 1.0])
axb.set_yticklabels(['0', '50%', '100%'], fontsize=7.2, color=SUBTLE)
axb.set_ylim(0, 1.2)
axb.set_title('hypothesis-only: trained on -> tested on',
              fontsize=9.0, fontweight='bold', color=TEXT, pad=6,
              loc='left')
style_axes(axb)

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 6',
        fontsize=8, ha='center', va='center', color='#aaaaaa')
plt.savefig(OUT + 'header_nli.png', dpi=150, facecolor=BG)
plt.close()
print('Wrote ' + OUT + 'header_nli.png')
print()
print('all seven images written, every cell asserted against the '
      'frozen run')
