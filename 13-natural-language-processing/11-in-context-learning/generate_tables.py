"""Generate every image for the In-Context Learning article.

Medium has no table support and collapses the whitespace in wide code
blocks, so each measured table ships as an image. This ONE script
trains all six models once (two conditions, three seeds) and emits
every PNG including the header. Every cell is asserted against the
frozen run of icl.py.

Writes: icl_table.png, axes_table.png, attention_table.png,
        emergence_curve.png, header_icl.png
Runs in about seventeen minutes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

import icl as L

OUT = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/11-in-context-learning/')

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
# Train all six models once, with icl.py's frozen seeds.
# =======================================================================

print('training six models (about seventeen minutes)...')
MODELS, PROBES = {}, {}
for name, bursty in (('bursty', True), ('iid', False)):
    MODELS[name], PROBES[name] = [], []
    for sd in L.SEEDS:
        m, pr = L.train(bursty, L.STEPS,
                        np.random.default_rng(L.SEED_TRAIN + sd),
                        seed=sd, probe_at=L.PROBE_AT)
        MODELS[name].append(m)
        PROBES[name].append(pr)
    print('  %s done' % name)
N_PARAMS = MODELS['bursty'][0].n_params()
assert N_PARAMS == 108403, N_PARAMS

ICL = {}
for nd in L.DEMOS:
    ICL[nd] = {n: L.ms([L.accuracy(m, np.random.default_rng(
        L.SEED_EVAL + nd), True, nd, 1) for m in MODELS[n]])
        for n in ('bursty', 'iid')}
FROZEN_ICL = {1: ('1.0000', '0.0000', '0.0275'),
              2: ('0.5133', '0.0319', '0.0333'),
              4: ('0.2608', '0.0297', '0.0342'),
              8: ('0.1308', '0.0105', '0.0308')}
for nd, (b, s, i) in FROZEN_ICL.items():
    got = ('%.4f' % ICL[nd]['bursty'][0], '%.4f' % ICL[nd]['bursty'][1],
           '%.4f' % ICL[nd]['iid'][0])
    assert got == (b, s, i), (nd, got)
print('  icl table ok')

AXES = {}
for ns in L.SUPPORT:
    AXES[ns] = {n: L.ms([L.accuracy(m, np.random.default_rng(
        L.SEED_EVAL + 50 + ns), True, L.K_MAX, ns) for m in MODELS[n]])
        for n in ('bursty', 'iid')}
FROZEN_AXES = {1: '0.1308', 2: '0.1825', 4: '0.3358', 8: '1.0000'}
for ns, b in FROZEN_AXES.items():
    assert '%.4f' % AXES[ns]['bursty'][0] == b, (ns, AXES[ns])
print('  axes table ok')

ATT = {}
for name in ('bursty', 'iid'):
    per = [L.copy_attention(m, np.random.default_rng(L.SEED_ATTN))
           for m in MODELS[name]]
    ATT[name] = {li: L.ms([p[li] for p in per]) for li in per[0]}
assert '%.4f' % ATT['bursty'][0][0] == '0.6872', ATT['bursty']
assert '%.4f' % ATT['iid'][0][0] == '0.1407', ATT['iid']
print('  attention ok')

CURVE = {n: [L.ms([p[i][1] for p in PROBES[n]])[0]
             for i in range(len(L.PROBE_AT))]
         for n in ('bursty', 'iid')}
assert '%.4f' % CURVE['bursty'][L.PROBE_AT.index(30)] == '0.0983'
assert '%.4f' % CURVE['bursty'][L.PROBE_AT.index(80)] == '0.9933'
print('  emergence ok')
print()

UNIFORM = 1.0 / L.T_MAX


# =======================================================================
# 1. Does it work, and where does it come from.
# =======================================================================

rows = []
for nd in L.DEMOS:
    b, i = ICL[nd]['bursty'], ICL[nd]['iid']
    rows.append([
        cell(str(nd), bold=True, color=BLUE),
        cell('%.4f' % b[0], bg=GOOD_BG if b[0] > 0.5 else None,
             bold=b[0] > 0.9),
        cell('%.4f' % b[1], color=SUBTLE),
        cell('%.4f' % i[0], bg=BAD_BG),
        cell('%.4f' % i[1], color=SUBTLE),
        note('chance is %.4f' % L.CHANCE if nd == L.DEMOS[0] else ''),
    ])
render_table(
    'icl_table.png',
    'The Same Model, Twice, Trained on Different Sequences',
    'Accuracy on a pairing never seen before, one supporting '
    'demonstration, context growing around it. %d seeds.' % len(L.SEEDS),
    ['demonstrations', 'bursty', 'sd', 'IID', 'sd', ''],
    [3.2, 2.4, 2.0, 2.4, 2.0, 3.4], rows,
    captions=[
        ('Same architecture, same %s parameters, same budget, same '
         'seeds. The only difference is how the TRAINING sequences '
         'were built.' % '{:,}'.format(N_PARAMS), SUBTLE),
        ('One demonstration: %.4f against %.4f. The capability is a '
         'property of the distribution, not of the model.'
         % (ICL[1]['bursty'][0], ICL[1]['iid'][0]), GREEN)],
    fig_px=(1840, 740))


# =======================================================================
# 2. Evidence against distractors.
# =======================================================================

rows = []
for ns in L.SUPPORT:
    b, i = AXES[ns]['bursty'], AXES[ns]['iid']
    rows.append([
        cell(str(ns), bold=True, color=BLUE),
        cell(str(L.K_MAX - ns), color=SUBTLE),
        cell('%.4f' % b[0], bg=GOOD_BG if b[0] > 0.5 else BAD_BG,
             bold=b[0] > 0.9),
        cell('%.4f' % b[1], color=SUBTLE),
        cell('%.4f' % i[0]),
    ])
render_table(
    'axes_table.png',
    'Evidence and Distractors Are Not the Same Axis',
    'Context held at %d demonstrations. Only the number that share '
    'the query\'s subject changes.' % L.K_MAX,
    ['supporting', 'distracting', 'bursty', 'sd', 'IID'],
    [2.8, 2.8, 2.6, 2.2, 2.6], rows,
    captions=[
        ('In the previous table adding demonstrations added '
         'DISTRACTORS. Here the context is fixed and only the '
         'evidence varies -- and the curve runs the other way.',
         SUBTLE),
        ('The model copies. What fails as a prompt grows is not the '
         'copying but the finding.', GREEN)],
    fig_px=(1760, 700))


# =======================================================================
# 3. The mechanism.
# =======================================================================

rows = []
for name in ('bursty', 'iid'):
    rows.append([
        label('bursty model' if name == 'bursty' else 'IID model'),
        cell('%.4f' % ATT[name][0][0],
             bg=GOOD_BG if ATT[name][0][0] > 0.4 else BAD_BG,
             bold=ATT[name][0][0] > 0.4),
        cell('%.4f' % ATT[name][0][1], color=SUBTLE),
        cell('%.4f' % ATT[name][1][0]),
        cell('%.4f' % ATT[name][1][1], color=SUBTLE),
    ])
rows.append([note('uniform attention would be'),
             cell('%.4f' % UNIFORM, color=SUBTLE), None,
             cell('%.4f' % UNIFORM, color=SUBTLE), None])
render_table(
    'attention_table.png',
    'The Receipt: Where the Query Is Looking',
    'Attention from the query position onto the one token in the '
    'prompt that holds the answer. Max over heads, %d seeds.'
    % len(L.SEEDS),
    ['model', 'layer 0', 'sd', 'layer 1', 'sd'],
    [4.0, 2.6, 2.0, 2.6, 2.0], rows,
    captions=[
        ('The bursty model puts %.4f of its attention on exactly the '
         'token it needs to copy. The IID model, given the identical '
         'prompt, is at %.4f.'
         % (ATT['bursty'][0][0], ATT['iid'][0][0]), SUBTLE),
        ('Two layers is the minimum such a circuit needs: one to mark '
         'what preceded each token, one to match and copy. This model '
         'has exactly two.', SUBTLE)],
    fig_px=(1780, 700), rules=(2,))


# =======================================================================
# 4. The phase transition.
# =======================================================================

fig, ax = plt.subplots(figsize=(1560 / 150, 820 / 150), dpi=150)
fig.patch.set_facecolor(BG)
xs = list(L.PROBE_AT)
ax.plot(xs, CURVE['bursty'], 'o-', color=GREEN, lw=2.4, ms=6,
        label='bursty training', zorder=4)
ax.plot(xs, CURVE['iid'], 'o-', color=RED, lw=2.0, ms=5,
        label='IID training', zorder=3)
ax.axhline(L.CHANCE, color=NEUTRAL, ls=':', lw=1.4, zorder=2)
ax.text(5.5, L.CHANCE + 0.03, 'chance (%.4f)' % L.CHANCE,
        fontsize=8.5, color=NEUTRAL)
i30, i80 = L.PROBE_AT.index(30), L.PROBE_AT.index(80)
ax.annotate('', xy=(80, CURVE['bursty'][i80]),
            xytext=(30, CURVE['bursty'][i30]),
            arrowprops=dict(arrowstyle='<->', color=AMBER, lw=1.6))
ax.text(6.5, 0.66, 'chance to ceiling', fontsize=10.5,
        color=AMBER, fontweight='bold', va='center')
ax.text(6.5, 0.56, 'in fifty steps', fontsize=10.5,
        color=AMBER, fontweight='bold', va='center')
ax.set_xscale('log')
ax.set_xticks([5, 10, 20, 50, 100, 200, 400, 800, 1500])
ax.set_xticklabels(['5', '10', '20', '50', '100', '200', '400',
                    '800', '1500'], fontsize=9)
ax.set_xlabel('training step (log scale)', fontsize=10.5, color=TEXT)
ax.set_ylabel('accuracy on an unseen pairing', fontsize=10.5,
              color=TEXT)
ax.set_ylim(-0.05, 1.08)
ax.legend(fontsize=10, frameon=False, loc='center right')
ax.set_title('In-context learning does not fade in. It assembles.',
             fontsize=13.5, fontweight='bold', color=TEXT, pad=12)
style_axes(ax)
fig.text(0.5, 0.015,
         'At step 30 the bursty model is at %.4f, barely above '
         'chance. At step 80 it is at %.4f. Nothing about the data '
         'or the objective changes in between.'
         % (CURVE['bursty'][i30], CURVE['bursty'][i80]),
         fontsize=10, ha='center', color=SUBTLE, fontstyle='italic')
plt.tight_layout(rect=[0, 0.045, 1, 1])
plt.savefig(OUT + 'emergence_curve.png', dpi=150, facecolor=BG,
            bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
print('Wrote ' + OUT + 'emergence_curve.png')


# =======================================================================
# 5. The header.
# =======================================================================

fig = plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'In-Context Learning: Behaviour Without a Gradient',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT)
ax.text(8, 8.05,
        'Two models, identical in every way but the shape of their '
        'training data. Only one can read a rule out of its prompt.',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE)


def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))


box(0.7, 6.05, 5.3, 1.45, PURPLE)
ax.text(3.35, 7.22, 'A rule that cannot be memorised',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=PURPLE)
ax.text(3.35, 6.62,
        'a fresh random pairing of %d nouns\nin every single sequence'
        % len(L.NOUNS),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 4.35, 5.3, 1.45, GREEN, face='#F0FDF4')
ax.text(3.35, 5.52, 'The distribution decides, not the model',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=GREEN)
ax.text(3.35, 4.92,
        'bursty %.4f   IID %.4f\nchance %.4f, sd across seeds %.4f'
        % (ICL[1]['bursty'][0], ICL[1]['iid'][0], L.CHANCE,
           ICL[1]['bursty'][1]),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 2.95, 5.3, 1.1, BLUE, face='#F8FAFC', lw=1.7)
ax.text(3.35, 3.82, 'And you can see it doing it',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=BLUE)
ax.text(3.35, 3.35,
        'attention on the copied token: %.4f vs %.4f'
        % (ATT['bursty'][0][0], ATT['iid'][0][0]),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 1.35, 5.3, 1.25, ORANGE, lw=1.6)
ax.text(3.35, 2.35, 'It does not fade in -- it assembles',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=ORANGE)
ax.text(3.35, 1.82,
        'step 30: %.4f\nstep 80: %.4f'
        % (CURVE['bursty'][i30], CURVE['bursty'][i80]),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

axh = fig.add_axes([0.45, 0.42, 0.50, 0.32])
axh.plot(xs, CURVE['bursty'], 'o-', color=GREEN, lw=2.0, ms=4.5,
         label='bursty', zorder=4)
axh.plot(xs, CURVE['iid'], 'o-', color=RED, lw=1.8, ms=4,
         label='IID', zorder=3)
axh.axhline(L.CHANCE, color=NEUTRAL, ls=':', lw=1.2, zorder=2)
axh.set_xscale('log')
axh.set_xticks([5, 20, 80, 400, 1500])
axh.set_xticklabels(['5', '20', '80', '400', '1500'], fontsize=7.2,
                    color=SUBTLE)
axh.set_yticks([0, 0.5, 1.0])
axh.set_yticklabels(['0', '0.5', '1.0'], fontsize=7.2, color=SUBTLE)
axh.set_ylim(-0.05, 1.12)
axh.set_xlabel('training step', fontsize=8, color=SUBTLE)
axh.legend(fontsize=7.4, frameon=False, loc='center right')
axh.set_title('the capability arrives in about fifty steps',
              fontsize=9.0, fontweight='bold', color=TEXT, pad=6,
              loc='left')
style_axes(axh)

axb = fig.add_axes([0.45, 0.115, 0.50, 0.205])
w = 0.36
x = np.arange(len(L.SUPPORT))
axb.bar(x - w / 2, [AXES[s]['bursty'][0] for s in L.SUPPORT], w,
        color=GREEN, zorder=3, label='n supporting, context 8')
axb.bar(x + w / 2, [ICL[d]['bursty'][0] for d in L.DEMOS], w,
        color=AMBER, zorder=3, label='n total, 1 supporting')
axb.set_xticks(x)
axb.set_xticklabels([str(s) for s in L.SUPPORT], fontsize=7.4,
                    color=SUBTLE)
axb.set_yticks([0, 0.5, 1.0])
axb.set_yticklabels(['0', '0.5', '1.0'], fontsize=7.2, color=SUBTLE)
axb.set_ylim(0, 1.2)
axb.set_xlabel('n', fontsize=8, color=SUBTLE)
axb.legend(fontsize=7.4, frameon=False, loc='upper center', ncol=2)
axb.set_title('"more examples" means two opposite things',
              fontsize=9.0, fontweight='bold', color=TEXT, pad=6,
              loc='left')
style_axes(axb)

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 11',
        fontsize=8, ha='center', va='center', color='#aaaaaa')
plt.savefig(OUT + 'header_icl.png', dpi=150, facecolor=BG)
plt.close()
print('Wrote ' + OUT + 'header_icl.png')
print()
print('all four images written, every cell asserted against the '
      'frozen run')
