"""Generate every image for the Retrieval-Augmented Generation article.

Medium has no table support and collapses the whitespace in wide code
blocks, so each measured table ships as an image. The pipeline is
expensive (a pre-training, a retriever and six generator fine-tunes),
so this ONE script trains everything once and emits every PNG
including the header. Every cell is asserted against the frozen run
of rag.py.

Writes: retrieval_table.png, conditions_table.png, mix_table.png,
        grounding_curve.png, header_rag.png
Runs in about six minutes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

import rag as R

OUT = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/08-retrieval-augmented-generation/')

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
# Train everything once, with rag.py's frozen seeds.
# =======================================================================

print('training (about six minutes)...')
corpus, ids, keep, kb, train, test = R.build_stage()
assert (len(kb), len(train), len(test)) == (144, 108, 36)
lm, _ = R.pretrain_lm(ids, keep, R.PRE_STEPS,
                      np.random.default_rng(R.SEED_PRE))
enc, _ = R.train_retriever([(f["question"], f["doc"]) for f in train],
                           R.RET_STEPS, np.random.default_rng(R.SEED_RET),
                           seed=R.SEED_MODEL)
AGREE, CONTRA = R.build_eval(test)

# ---- retrieval ----
docs = [g["doc"] for g in kb]
gold = [kb.index(g) for g in test]
Zd = R.embed_all(enc, docs)
Zq = R.embed_all(enc, [g["question"] for g in test])
Slex = R.lexical_scores([g["question"] for g in test], docs)
DENSE = R.recall_at(Zq @ Zd.T, gold)
LEX = R.recall_at(Slex, gold)
BAND = float(np.mean([(Slex[i] == Slex[i, g]).sum()
                      for i, g in enumerate(gold)]))
BEATEN = float(np.mean([(Slex[i] > Slex[i, g]).sum()
                        for i, g in enumerate(gold)]))
assert ['%.4f' % LEX[1], '%.4f' % LEX[5]] == ['0.1389', '0.5556'], LEX
assert DENSE[1] == 1.0 and DENSE[5] == 1.0, DENSE
assert '%.2f' % BAND == '10.11', BAND
assert '%.2f' % BEATEN == '0.08', BEATEN
PRIOR = R.prior_argmax(lm, AGREE)
assert '%.4f' % PRIOR == '0.9722', PRIOR
print('  retrieval ok (lex %.4f, dense %.4f, band %.2f, prior %.4f)'
      % (LEX[1], DENSE[1], BAND, PRIOR))

# ---- the four conditions at the frozen training mix ----
gen, _ = R.train_generator(lm, R.gen_items(R.remix(
    train, R.TRAIN_MIX, np.random.default_rng(R.SEED_MIX + 50))),
    R.GEN_STEPS, np.random.default_rng(R.SEED_GEN))
DMAP = {f["subj"] + f["verb"]: R.distractor_for(f, kb) for f in CONTRA}
COND = [
    ("closed book (no document)", R.score(gen, AGREE, lambda f: [])),
    ("agreeing document", R.score(gen, AGREE, lambda f: f["doc"])),
    ("contradicting document", R.score(gen, CONTRA, lambda f: f["doc"])),
    ("a DIFFERENT document",
     R.score(gen, CONTRA, lambda f: DMAP[f["subj"] + f["verb"]]["doc"])),
]
MISLED = float(np.mean([x == DMAP[f["subj"] + f["verb"]]["obj"]
                        for x, f in zip(COND[3][1]["got"], CONTRA)]))
FROZEN_COND = [('0.3056', '0.3056', '0.6944'),
               ('1.0000', '1.0000', '0.0000'),
               ('0.3056', '0.4167', '0.2778'),
               ('0.0278', '0.5000', '0.4722')]
for (nm, s), want in zip(COND, FROZEN_COND):
    got = ('%.4f' % s["acc"], '%.4f' % s["prior"], '%.4f' % s["other"])
    assert got == want, (nm, got)
assert '%.4f' % MISLED == '0.1667', MISLED
print('  conditions ok (misled %.4f)' % MISLED)

# ---- the training-mix sweep ----
MIX = []
for mix in R.MIXES:
    tr = R.remix(train, mix,
                 np.random.default_rng(R.SEED_MIX + int(mix * 100)))
    g, _ = R.train_generator(lm, R.gen_items(tr), R.GEN_STEPS,
                             np.random.default_rng(R.SEED_GEN))
    cb = R.score(g, AGREE, lambda f: [])
    sa = R.score(g, AGREE, lambda f: f["doc"])
    sc = R.score(g, CONTRA, lambda f: f["doc"])
    sd = R.score(g, CONTRA,
                 lambda f: DMAP[f["subj"] + f["verb"]]["doc"])
    mis = float(np.mean([x == DMAP[f["subj"] + f["verb"]]["obj"]
                         for x, f in zip(sd["got"], CONTRA)]))
    MIX.append((mix, cb["acc"], sa["acc"], sc["acc"], sc["prior"],
                sc["other"], mis))
    print('  mix %.2f done' % mix)
FROZEN_MIX = [
    ('0.6111', '1.0000', '0.0000', '1.0000', '0.0000', '0.0000'),
    ('0.4167', '0.8889', '0.0278', '0.7222', '0.2500', '0.0278'),
    ('0.3056', '1.0000', '0.3056', '0.4167', '0.2778', '0.1667'),
    ('0.1667', '0.7778', '0.2222', '0.2500', '0.5278', '0.3611'),
    ('0.0833', '0.8056', '0.6667', '0.0278', '0.3056', '0.8056'),
]
for row, want in zip(MIX, FROZEN_MIX):
    got = tuple('%.4f' % v for v in row[1:])
    assert got == want, (row[0], got)
print('  mix sweep ok')
print()


# =======================================================================
# 1. The retrieval half.
# =======================================================================

rows = [
    [label('lexical overlap'),
     note('words the two have in common'),
     cell('%.4f' % LEX[1], bg=BAD_BG),
     cell('%.4f' % LEX[5], bg=MID_BG)],
    [label('bi-encoder'),
     note('in-batch negatives, Part 6\'s tower'),
     cell('%.4f' % DENSE[1], bg=GOOD_BG, bold=True),
     cell('%.4f' % DENSE[5], bg=GOOD_BG, bold=True)],
]
render_table(
    'retrieval_table.png',
    'Finding the Document',
    'One question against a store of %d documents. The question '
    'carries the verb\'s BASE form, the document its inflected one.'
    % len(kb),
    ['retriever', 'what it matches on', 'recall@1', 'recall@5'],
    [3.9, 6.4, 2.7, 2.7], rows,
    captions=[
        ('Overlap is not failing to FIND the document: on average only '
         '%.2f documents outscore it. It is failing to CHOOSE -- %.2f '
         'documents TIE with it.' % (BEATEN, BAND), SUBTLE),
        ('Several verb stems here (crown, plough, anchor) are also '
         'nouns, so unrelated documents join the tie. The bi-encoder '
         'has learned that "command" and "commands" are one relation.',
         SUBTLE)],
    fig_px=(1820, 700))


# =======================================================================
# 2. Reading or reciting.
# =======================================================================

rows = []
MARK = [None, 'the benchmark', 'the test', 'a wrong retrieval']
for (nm, s), tag in zip(COND, MARK):
    rows.append([
        label(nm),
        cell('%.4f' % s["acc"],
             bg=GOOD_BG if s["acc"] >= 0.9 else BAD_BG, bold=True),
        cell('%.4f' % s["prior"],
             bg=BAD_BG if s["prior"] >= 0.4 else None),
        cell('%.4f' % s["other"]),
        note(tag or '', size=10.5),
    ])
render_table(
    'conditions_table.png',
    'Is It Reading, or Reciting?',
    'The same generator and the same questions. Only the document in '
    'front of it changes. Columns are where the answer came from.',
    ['condition', 'said the\nDOCUMENT', 'said the\nROLE', 'neither', ''],
    [5.0, 2.7, 2.5, 2.2, 3.4], rows,
    captions=[
        ('Row 2 has the same number twice because on an agreeing fact '
         'the document and the prior say the same word -- that row '
         'cannot tell reading from reciting.', SUBTLE),
        ('Row 3 is the honest test: %.4f with a gold document that '
         'agrees, %.4f with one that does not.'
         % (COND[1][1]["acc"], COND[2][1]["acc"]), RED)],
    fig_px=(1860, 700), rules=(3, 1))


# =======================================================================
# 3. Where grounding comes from.
# =======================================================================

rows = []
for mix, cb, sa, sc, pr, ot, mis in MIX:
    rows.append([
        cell('%.2f' % mix, bold=True),
        cell('%.4f' % cb, bg=MID_BG if cb < 0.3 else None),
        cell('%.4f' % sa),
        cell('%.4f' % sc, bg=GOOD_BG if sc > 0.5 else BAD_BG),
        cell('%.4f' % pr, bg=BAD_BG if pr > 0.5 else None),
        cell('%.4f' % ot),
        cell('%.4f' % mis, bg=BAD_BG if mis > 0.5 else None),
    ])
render_table(
    'mix_table.png',
    'Grounding Is Learned, Not Inherited',
    'Sweeping the share of FINE-TUNING documents that contradict the '
    'prior. Everything else -- model, data, budget, seeds -- is fixed.',
    ['contradicting\nshare of training',
     'closed\nbook', 'agreeing\ndocument', 'reads the\ndocument',
     'recites\nthe role', 'neither\nof them', 'misled by a\nWRONG doc'],
    [3.6, 2.1, 2.4, 2.5, 2.3, 2.2, 2.8], rows,
    captions=[
        ('Top row: every training document agrees with the prior, '
         'which is what collected evidence looks like. It scores '
         '1.0000 on agreeing facts and 0.0000 on contradicting ones.',
         SUBTLE),
        ('The last two columns are the finding: reciting falls from '
         '1.0000 to 0.0278 while being misled rises from 0.0000 to '
         '0.8056. One behaviour, two names.', RED)],
    fig_px=(1880, 760))


# =======================================================================
# 4. The same thing as a curve.
# =======================================================================

fig, ax = plt.subplots(figsize=(1500 / 150, 800 / 150), dpi=150)
fig.patch.set_facecolor(BG)
xs = [m[0] for m in MIX]
ax.plot(xs, [m[4] for m in MIX], 'o-', color=BLUE, lw=2.2, ms=7,
        label='recites the role signature', zorder=4)
ax.plot(xs, [m[6] for m in MIX], 'o-', color=RED, lw=2.2, ms=7,
        label='misled by a WRONG document', zorder=4)
ax.plot(xs, [m[3] for m in MIX], 'o--', color=GREEN, lw=1.8, ms=6,
        label='reads the gold document', zorder=3)
ax.plot(xs, [m[1] for m in MIX], 'o:', color=NEUTRAL, lw=1.6, ms=5,
        label='closed book (no document)', zorder=2)
ax.set_xlabel('share of fine-tuning documents that contradict the prior',
              fontsize=10.5, color=TEXT)
ax.set_ylabel('rate on the held-out facts', fontsize=10.5, color=TEXT)
ax.set_xlim(-0.04, 1.04)
ax.set_ylim(-0.05, 1.08)
ax.set_xticks(xs)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.legend(fontsize=9.5, frameon=False, loc='upper right',
          bbox_to_anchor=(0.74, 0.99))
ax.set_title('Trusting documents and being fooled by them are the '
             'same behaviour',
             fontsize=13.5, fontweight='bold', color=TEXT, pad=12)
style_axes(ax)
fig.text(0.5, 0.012,
         'The blue and red lines are mirror images. There is no '
         'setting that reads the document and resists a bad one.',
         fontsize=10, ha='center', color=SUBTLE, fontstyle='italic')
plt.tight_layout(rect=[0, 0.035, 1, 1])
plt.savefig(OUT + 'grounding_curve.png', dpi=150, facecolor=BG,
            bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
print('Wrote ' + OUT + 'grounding_curve.png')


# =======================================================================
# 5. The header.
# =======================================================================

fig = plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Retrieval-Augmented Generation: Reading, or Reciting?',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT)
ax.text(8, 8.05,
        'Trained only on documents that agree with it, the generator '
        'scores %.4f on agreeing facts and %.4f on contradicting ones.'
        % (MIX[0][2], MIX[0][3]),
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE)


def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))


box(0.7, 6.05, 5.3, 1.45, PURPLE)
ax.text(3.35, 7.22, 'The prior is a known quantity',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=PURPLE)
ax.text(3.35, 6.62,
        'the pre-trained model names the role\n'
        'signature %.4f of the time' % PRIOR,
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 4.35, 5.3, 1.45, RED, face='#FEF2F2')
ax.text(3.35, 5.52, 'So the document can be made to disagree',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=RED)
ax.text(3.35, 4.92,
        'gold doc that agrees:     %.4f\n'
        'gold doc that contradicts: %.4f'
        % (COND[1][1]["acc"], COND[2][1]["acc"]),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 2.95, 5.3, 1.1, BLUE, face='#F8FAFC', lw=1.7)
ax.text(3.35, 3.82, 'Retrieval is the easy half',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=BLUE)
ax.text(3.35, 3.35,
        'bi-encoder %.4f vs lexical %.4f' % (DENSE[1], LEX[1]),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 1.35, 5.3, 1.25, ORANGE, lw=1.6)
ax.text(3.35, 2.35, 'And grounding has a price',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=ORANGE)
ax.text(3.35, 1.82,
        'misled by a wrong document:\n'
        '%.4f  ->  %.4f' % (MIX[0][6], MIX[-1][6]),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

axh = fig.add_axes([0.45, 0.42, 0.50, 0.32])
axh.plot(xs, [m[4] for m in MIX], 'o-', color=BLUE, lw=2.0, ms=5,
         label='recites the role', zorder=4)
axh.plot(xs, [m[6] for m in MIX], 'o-', color=RED, lw=2.0, ms=5,
         label='misled by a wrong doc', zorder=4)
axh.plot(xs, [m[3] for m in MIX], 'o--', color=GREEN, lw=1.5, ms=4,
         label='reads the document', zorder=3)
axh.set_xlabel('contradicting share of fine-tuning', fontsize=8,
               color=SUBTLE)
axh.set_xlim(-0.04, 1.04)
axh.set_ylim(-0.05, 1.12)
axh.set_xticks(xs)
axh.set_xticklabels(['%.2f' % x for x in xs], fontsize=7.2,
                    color=SUBTLE)
axh.set_yticks([0, 0.5, 1.0])
axh.set_yticklabels(['0', '0.5', '1.0'], fontsize=7.2, color=SUBTLE)
axh.legend(fontsize=7.2, frameon=False, loc='upper center', ncol=2)
axh.set_title('grounding and gullibility are one behaviour',
              fontsize=9.0, fontweight='bold', color=TEXT, pad=6,
              loc='left')
style_axes(axh)

axb = fig.add_axes([0.45, 0.115, 0.50, 0.205])
names = ['closed\nbook', 'agreeing\ndoc', 'contra\ndoc', 'wrong\ndoc']
vals = [COND[0][1]["acc"], COND[1][1]["acc"],
        COND[2][1]["acc"], COND[3][1]["acc"]]
cols = [NEUTRAL, GREEN, RED, RED]
axb.bar(np.arange(4), vals, 0.5, color=cols, zorder=3)
for i, v in enumerate(vals):
    axb.text(i, v + 0.04, '%.4f' % v, ha='center', fontsize=7.6,
             color=cols[i], fontweight='bold')
axb.set_xticks(np.arange(4))
axb.set_xticklabels(names, fontsize=7.4, color=SUBTLE)
axb.set_yticks([0, 0.5, 1.0])
axb.set_yticklabels(['0', '0.5', '1.0'], fontsize=7.2, color=SUBTLE)
axb.set_ylim(0, 1.25)
axb.set_title('answered with the document\'s object', fontsize=9.0,
              fontweight='bold', color=TEXT, pad=6, loc='left')
style_axes(axb)

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 8',
        fontsize=8, ha='center', va='center', color='#aaaaaa')
plt.savefig(OUT + 'header_rag.png', dpi=150, facecolor=BG)
plt.close()
print('Wrote ' + OUT + 'header_rag.png')
print()
print('all five images written, every cell asserted against the '
      'frozen run')
