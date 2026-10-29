"""Generate every data-table image for the BERT-style Encoders article.

Medium has no table support and collapses the whitespace in wide code
blocks, so each measured table ships as an image. The three encoders
are pre-trained once here with frozen seeds and every cell is measured
live and asserted against the frozen run of bert_encoder.py.

Writes: loss_curve.png, receipt_table.png, attention_table.png,
        labels_table.png, pll_table.png, recipe_table.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import bert_encoder as B

OUT = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/04-bert-style-encoders/')

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


# =======================================================================
# Train the three encoders once.
# =======================================================================

rng = np.random.default_rng(42)
corpus = B.gen_corpus(8000, rng)
ids, keep = B.encode([s["words"] for s in corpus])
test = B.gen_corpus(1000, np.random.default_rng(7))
test_accept = B.make_acceptability(600, np.random.default_rng(13))

print('pre-training bidirectional...')
bi = B.Encoder(d=64, heads=4, layers=2, causal=False, seed=0, lr=3e-3)
h_bi = B.pretrain(bi, ids, keep, B.STEPS, np.random.default_rng(1),
                  log_every=250)
print('pre-training causal...')
ca = B.Encoder(d=64, heads=4, layers=2, causal=True, seed=0, lr=3e-3)
h_ca = B.pretrain(ca, ids, keep, B.STEPS, np.random.default_rng(1),
                  log_every=250)
print('pre-training 100% [MASK]...')
mk = B.Encoder(d=64, heads=4, layers=2, causal=False, seed=0, lr=3e-3)
h_mk = B.pretrain(mk, ids, keep, B.STEPS, np.random.default_rng(1),
                  recipe='mask', log_every=250)

assert [f'{l:.2f}' for _, l in h_bi] == \
    ['3.72', '1.62', '1.31', '1.35', '1.34', '1.09', '1.27'], h_bi
assert [f'{l:.2f}' for _, l in h_ca] == \
    ['3.88', '2.12', '1.78', '1.73', '1.94', '1.59', '1.83'], h_ca
assert [f'{l:.2f}' for _, l in h_mk] == \
    ['3.74', '1.60', '1.29', '1.51', '0.81', '1.02', '0.97'], h_mk

# =======================================================================
# 1. Pre-training loss curve.
# =======================================================================

fig, ax = plt.subplots(figsize=(1500 / 150, 820 / 150), dpi=150)
fig.patch.set_facecolor(BG)
steps = [s for s, _ in h_bi]
for hist, col, name, mark in ((h_bi, GREEN, 'bidirectional', 'o'),
                              (h_mk, PURPLE, '100% [MASK]', 'D'),
                              (h_ca, RED, 'causal', 's')):
    ax.plot(steps, [l for _, l in hist], mark + '-', color=col, lw=2,
            ms=5, label=name, zorder=4)
ax.axhline(np.log(B.V), color=NEUTRAL, ls=':', lw=1.4, zorder=2)
ax.text(40, np.log(B.V) + 0.08,
        f'uniform guess over {B.V} tokens = {np.log(B.V):.2f} nats',
        fontsize=9.5, color=SUBTLE)
ax.set_xlabel('pre-training step', fontsize=10.5, color=SUBTLE)
ax.set_ylabel('masked-LM loss (nats)', fontsize=10.5, color=SUBTLE)
ax.set_title('Masked-LM Loss During Pre-training', fontsize=15,
             fontweight='bold', color=TEXT, pad=12)
ax.legend(fontsize=10, frameon=False, loc='upper right')
ax.grid(color='#E2E8F0', linewidth=0.6, zorder=0)
ax.tick_params(labelsize=9.5, colors=SUBTLE)
ax.set_ylim(0.5, 5.05)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    ax.spines[sp].set_color('#CBD5E1')
fig.text(0.5, 0.058,
         'Per-batch losses, so they wobble. The floor is not zero: a '
         'masked subject is genuinely ambiguous.',
         fontsize=9.5, ha='center', color=SUBTLE, fontstyle='italic')
fig.text(0.5, 0.014,
         '100% [MASK] reaches the lowest loss of the three and '
         'transfers the worst — pre-training loss is a progress '
         'bar, not a target.',
         fontsize=9.5, ha='center', color=SUBTLE, fontstyle='italic')
plt.tight_layout(rect=[0, 0.095, 1, 1])
plt.savefig(OUT + 'loss_curve.png', dpi=150, facecolor=BG)
plt.close(fig)
print(f'Wrote {OUT}loss_curve.png')

# =======================================================================
# 2. Fill-the-gap receipt.
# =======================================================================

bs, cs = B.score_subject(bi, test), B.score_subject(ca, test)
bv, cv = B.score_verb(bi, test), B.score_verb(ca, test)
bp, cp = B.score_poss(bi, test), B.score_poss(ca, test)
BASE = 1 - sum(s["plural"] for s in test) / len(test)

RECEIPT = [('subject: number', 'right', cs[0], bs[0]),
           ('subject: gender', 'right', cs[3], bs[3]),
           ('subject: topic', 'both', cs[1], bs[1]),
           ('subject: exact word', 'both', cs[2], bs[2]),
           ('verb: number', 'left', cv[0], bv[0]),
           ('verb: topic', 'left', cv[1], bv[1]),
           ('possessive: exact', 'left', cp[0], bp[0])]
assert [f'{v:.3f}' for _, _, c, b in RECEIPT for v in (c, b)] == \
    ['0.667', '0.998', '0.206', '1.000', '0.687', '1.000', '0.027',
     '0.455', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000']
assert f'{BASE:.3f}' == '0.667', BASE

EV_COL = {'right': RED, 'left': BLUE, 'both': PURPLE}


def tint(v):
    if v >= 0.95:
        return GOOD_BG, GREEN
    if v >= 0.60:
        return MID_BG, AMBER
    return BAD_BG, RED


rows = []
for name, ev, c, b in RECEIPT:
    cbg, cfg = tint(c)
    bbg, bfg = tint(b)
    rows.append([
        label(name),
        cell('to the ' + ev if ev != 'both' else 'both sides',
             color=EV_COL[ev], size=11),
        cell(f'{c:.1%}', color=cfg, bg=cbg, bold=True),
        cell(f'{b:.1%}', color=bfg, bg=bbg, bold=True)])

render_table(
    'receipt_table.png', 'Fill the Gap: What Bidirectionality Buys',
    'One slot masked at a time in 1,000 held-out sentences.',
    ['What is hidden', 'Evidence lies', 'Causal', 'Bidirectional'],
    [6.0, 4.2, 3.6, 4.4], rows,
    captions=[(f'Guessing the majority class scores {BASE:.1%} on '
               'number: the causal model manages exactly that, and no '
               'more.', SUBTLE)],
    fig_px=(1700, 950))

# =======================================================================
# 3. Attention receipt.
# =======================================================================

sent = next(s for s in test if s["plural"] and len(s["words"]) == 6)
att_bi, words_bi, pos = B.attention_from_mask(bi, sent, 'subj')
att_ca, _, _ = B.attention_from_mask(ca, sent, 'subj')
words = list(words_bi)
words[pos] = '[MASK]'
COLS = [('bidir L0', att_bi[0]), ('bidir L1', att_bi[1]),
        ('causal L0', att_ca[0]), ('causal L1', att_ca[1])]
assert [f'{v:.2f}' for _, a in COLS for v in a] == \
    ['0.00', '0.00', '0.25', '0.00', '0.24', '0.25', '0.25',
     '0.00', '0.50', '0.00', '0.00', '0.00', '0.50', '0.00',
     '0.00', '0.00', '0.75', '0.25', '0.00', '0.00', '0.00',
     '0.16', '0.05', '0.05', '0.74', '0.00', '0.00', '0.00'], \
    [f'{v:.2f}' for _, a in COLS for v in a]

blues = plt.cm.Blues
rows = []
for j, w in enumerate(words):
    cells = [label(w, size=11.5)]
    for _, a in COLS:
        v = float(a[j])
        cells.append(cell(f'{v:.2f}',
                          color='#FFFFFF' if v > 0.45 else TEXT,
                          bg=blues(v * 0.85), bold=v > 0.2))
    rows.append(cells)

render_table(
    'attention_table.png', 'Where the Masked Position Looks',
    '"the distant [MASK] navigate their island" \u2014 mean attention '
    'out of the gap, averaged over heads.',
    ['Position', 'bidir L0', 'bidir L1', 'causal L0', 'causal L1'],
    [4.6, 3.4, 3.4, 3.4, 3.4], rows,
    captions=[('Layer 1 of the bidirectional model puts half its '
               'weight on "their", the token that carries the number.',
               SUBTLE),
              ('The causal rows are zero below the gap because the '
               'mask forbids it \u2014 there is no place to look.',
               SUBTLE)],
    fig_px=(1500, 1000), row_h=0.8)

# =======================================================================
# 4. Label efficiency.
# =======================================================================

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
                            np.random.default_rng(11),
                            only={'Wc', 'bc'}))
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

rows = []
for i, n in enumerate(LAB):
    rows.append([
        cell(str(n), bold=True, size=12.5),
        cell(f'{SCRATCH[i]:.1%}', color=TEXT),
        cell(f'{PROBE[i]:.1%}', color=TEXT),
        cell(f'{FT[i]:.1%}', color=GREEN, bg=GOOD_BG, bold=True,
             size=12.5)])

render_table(
    'labels_table.png', 'Pre-train Once, Then Spend Labels',
    'Is this sentence grammatical? Accuracy on 600 held-out '
    'sentences. Chance is 50%.',
    ['Labelled\nsentences', 'From scratch', 'Frozen probe',
     'Fine-tuned'],
    [4.6, 4.4, 4.4, 4.4], rows,
    captions=[('The frozen probe never passes 62%: masked language '
               'modelling never gives [CLS] a job to do.', SUBTLE),
              ('Same architecture in every column \u2014 only the '
               'starting weights differ.', SUBTLE)],
    fig_px=(1500, 800), header_h=1.3, row_h=0.9)

# =======================================================================
# 5. Zero-shot scoring.
# =======================================================================

cal = B.make_acceptability(20, np.random.default_rng(1020))
labels_t = np.array([y for _, y in test_accept])
words_t = [w for w, _ in test_accept]
PLL = []
for tag, m in (('bidirectional', bi), ('causal', ca)):
    s = B.pseudo_log_likelihood(m, words_t)
    thr = B.pll_threshold(m, cal)
    acc = float(((s >= thr) == (labels_t == 1)).mean())
    best = max(float(((s >= t) == (labels_t == 1)).mean())
               for t in np.sort(s))
    PLL.append((tag, float(s[labels_t == 1].mean()),
                float(s[labels_t == 0].mean()), acc, best))

assert [f'{v:.2f}' if abs(v) > 1 else f'{v:.3f}'
        for _, a, b, c, d in PLL for v in (a, b, c, d)] == \
    ['-6.41', '-16.20', '0.905', '0.940',
     '-9.77', '-17.31', '0.965', '0.977']

rows = [[label(tag, size=12.5), cell(f'{a:.2f}'), cell(f'{b:.2f}'),
         cell(f'{acc:.1%}', color=GREEN, bg=GOOD_BG, bold=True,
              size=12.5), cell(f'{best:.1%}', color=SUBTLE)]
        for tag, a, b, acc, best in PLL]

render_table(
    'pll_table.png', 'Scoring a Sentence With No Fine-Tuning At All',
    'Mask each position in turn, sum log P(the word that was really '
    'there). No new parameters.',
    ['Encoder', 'Mean score:\nacceptable', 'Mean score:\ncorrupt',
     'Accuracy\n(20-label threshold)', 'Best possible\nthreshold'],
    [5.0, 4.2, 4.0, 5.6, 4.4], rows,
    captions=[('Twenty labels spent on one threshold beat twenty '
               f'labels spent fine-tuning a head ({FT[0]:.1%}).',
               PURPLE),
              ('The causal model wins here because every violation in '
               'this task is local and visible from the left.',
               SUBTLE)],
    fig_px=(1750, 760), header_h=1.35, row_h=0.95)

# =======================================================================
# 6. The corruption recipe.
# =======================================================================

e_b, t_b, x_b = B.copy_test(bi, test, np.random.default_rng(5))
e_m, t_m, x_m = B.copy_test(mk, test, np.random.default_rng(5))
ftr = {}
for n in (50, 100):
    train = B.make_acceptability(n, np.random.default_rng(1000 + n))
    for tag, model in (('b', bi), ('m', mk)):
        c = model.clone()
        c.lr = 1e-3
        ftr[(tag, n)] = B.finetune(c, train, test_accept, 400,
                                   np.random.default_rng(11))

RECIPE = [('echoes the wrong word', e_b, e_m, False),
          ('restores the right topic', t_b, t_m, True),
          ('restores the exact word', x_b, x_m, True),
          ('fine-tuned, 50 labels', ftr[('b', 50)], ftr[('m', 50)],
           True),
          ('fine-tuned, 100 labels', ftr[('b', 100)], ftr[('m', 100)],
           True)]
assert [f'{v:.3f}' for _, a, b, _ in RECIPE for v in (a, b)] == \
    ['0.000', '0.002', '0.999', '0.899', '0.440', '0.414', '0.672',
     '0.610', '0.800', '0.708']

rows = []
for name, a, b, higher in RECIPE:
    d = 100 * ((a - b) if higher else (b - a))
    if abs(d) < 1.0:
        dc = cell('no difference', color=NEUTRAL, mono=False, size=10)
    else:
        dc = cell(f'+{d:.1f}', color=GREEN, bold=True)
    rows.append([label(name), cell(f'{a:.1%}'), cell(f'{b:.1%}'), dc])

render_table(
    'recipe_table.png', 'The Corruption Recipe: 80/10/10 vs 100% [MASK]',
    'Same corpus, same seeds, same budget \u2014 only the recipe '
    'differs.',
    ['Measurement', '80/10/10', '100% [MASK]', 'Difference'],
    [7.0, 4.0, 4.4, 4.0], rows, rules=(3,),
    captions=[('Top three rows: a real but wrong word, no [MASK] '
               'anywhere. Lower is better on the first.', SUBTLE),
              ('Neither model echoes the word it was shown \u2014 the '
               'usual story for the 10% random replacement.', SUBTLE)],
    fig_px=(1700, 900))

print('all tables written')
