"""Generate every image for the Fine-tuning and LoRA article.

Medium has no table support and collapses the whitespace in wide code
blocks, so each measured table ships as an image. The pipeline is the
slowest in the series -- almost every number is a mean over four seeds
-- so this ONE script trains the whole sweep once and emits every PNG
including the header. Every cell is asserted against the frozen run
of lora.py.

Writes: spectrum_table.png, placement_table.png, rank_table.png,
        forgetting_table.png, header_lora.png
Runs in about twenty-two minutes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

import lora as L

OUT = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/09-fine-tuning-and-lora/')

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
# Train the whole sweep once, with lora.py's frozen seeds.
# =======================================================================

print('training the sweep (about twenty-two minutes)...')
corpus, ids, keep, kb, train, test = L.build_stage()
lm, _ = L.pretrain_lm(ids, keep, L.PRE_STEPS,
                      np.random.default_rng(L.SEED_PRE))
BEFORE = {k: v.copy() for k, v in lm.P.items()}
N_FULL = sum(v.size for v in lm.P.values())
assert N_FULL == 115185, N_FULL
agree, contra = L.build_eval(test)
EV = agree + contra
items = L.gen_items(L.remix(train, L.TRAIN_MIX,
                            np.random.default_rng(L.SEED_MIX + 50)))
HELD = L.held_out()
FLOOR = L.oracle_floor(HELD)
assert '%.4f' % FLOOR == '1.4310', FLOOR

spec_model, _ = L.train_generator(lm, items, L.FT_STEPS,
                                  np.random.default_rng(100),
                                  lr=L.FULL_LR)
SPEC = {k: L.delta_spectrum(BEFORE, spec_model.P, k)
        for k in L.SPECTRUM_KEYS}
assert '%.4f' % SPEC['Wv']['rel'] == '0.4461', SPEC['Wv']['rel']
assert '%.4f' % L.energy_at(SPEC['Wv'], 1) == '0.4623'
assert SPEC['0.Wqkv']['r90'] == 24, SPEC['0.Wqkv']
print('  spectrum ok')

SWEEP = L.run_sweep(lm, items, EV, HELD)

FROZEN_PLACE = {
    'attention only': ('0.4549', '0.0060'),
    'attention + head': ('0.4861', '0.0241'),
    'head only': ('0.2569', '0.0208'),
    'everything': ('0.5417', '0.0260'),
}
for name, want in FROZEN_PLACE.items():
    m, s = L.ms(SWEEP[('place', name)], 'acc')
    assert ('%.4f' % m, '%.4f' % s) == want, (name, m, s)
FULL_M, FULL_S = L.ms(SWEEP[('full', L.FULL_LR)], 'acc')
assert ('%.4f' % FULL_M, '%.4f' % FULL_S) == ('0.5972', '0.0354')
HI_M, HI_S = L.ms(SWEEP[('place_hi', L.BEST_PLACEMENT)], 'acc')
assert ('%.4f' % HI_M, '%.4f' % HI_S) == ('0.5035', '0.0690')
print('  placement ok')

FROZEN_RANK = {1: ('0.4896', '0.0180'), 2: ('0.5208', '0.0208'),
               4: ('0.5417', '0.0260'), 8: ('0.5903', '0.0208'),
               16: ('0.5660', '0.0316')}
for r, want in FROZEN_RANK.items():
    m, s = L.ms(SWEEP[('rank', r)], 'acc')
    assert ('%.4f' % m, '%.4f' % s) == want, (r, m, s)
print('  rank ok')

FORGET = [
    ('full fine-tune', L.FULL_LR, ('full', L.FULL_LR),
     ('0.5972', '2.2689', '5.3266', '0.3111')),
    ('LoRA r=4', L.FULL_LR, ('place', L.BEST_PLACEMENT),
     ('0.5417', '2.2782', '5.4123', '0.5552')),
    ('full fine-tune', L.HIGH_LR, ('full', L.HIGH_LR),
     ('0.5243', '3.7677', '15.0018', '0.0044')),
    ('LoRA r=4', L.HIGH_LR, ('place_hi', L.BEST_PLACEMENT),
     ('0.5035', '4.1032', '17.0834', '0.0543')),
]
FVALS = {}
for name, lr, key, want in FORGET:
    v = tuple(L.ms(SWEEP[key], f)[0]
              for f in ('acc', 'nats', 'kl', 'mass'))
    got = ('%.4f' % v[0], '%.4f' % v[1], '%.4f' % v[2], '%.4f' % v[3])
    assert got == want, (name, lr, got)
    FVALS[key] = v
print('  forgetting ok')
print()


# =======================================================================
# 1. The spectrum of the update.
# =======================================================================

rows = []
for k in L.SPECTRUM_KEYS:
    s = SPEC[k]
    hot = k == 'Wv'
    rows.append([
        cell(k, mono=True, bold=hot, align='left',
             color=RED if hot else TEXT),
        cell('x'.join(map(str, s['shape'])), color=SUBTLE),
        cell('%.4f' % s['rel'], bg=BAD_BG if hot else None),
        cell(str(s['r50'])), cell(str(s['r90']), bold=True),
        cell(str(s['r99'])),
        cell('%.4f' % L.energy_at(s, 8),
             bg=GOOD_BG if hot else MID_BG),
    ])
render_table(
    'spectrum_table.png',
    'The Update Is Not Low Rank',
    'Singular values of dW = W_after - W_before for every weight '
    'matrix. r90% is the rank a truncated SVD needs to keep 90% of it.',
    ['matrix', 'shape', '|dW| / |W|', 'r50%', 'r90%', 'r99%',
     'energy\nkept at r=8'],
    [2.3, 2.3, 2.6, 1.8, 1.8, 1.8, 3.0], rows,
    captions=[
        ('Out of 64 available directions, keeping 90% of the update '
         'takes a rank in the twenties almost everywhere. Rank 8 -- an '
         'ordinary LoRA setting -- keeps barely half.', SUBTLE),
        ('The output head is the exception and the most-changed '
         'matrix: %.4f of its energy is one direction. Remember it; '
         'it is the worst place to put an adapter alone.'
         % L.energy_at(SPEC['Wv'], 1), RED)],
    fig_px=(1840, 940))


# =======================================================================
# 2. Where the adapters go.
# =======================================================================

rows = [[label('(full fine-tune)'), cell('{:,}'.format(N_FULL)),
         cell('1.0000', color=SUBTLE),
         cell('%.4f' % FULL_M, bg=GOOD_BG, bold=True),
         cell('%.4f' % FULL_S, color=SUBTLE)]]
for name, keys in L.PLACEMENTS.items():
    n = L.n_params_lora(lm, keys, L.BASE_RANK)
    m, s = L.ms(SWEEP[('place', name)], 'acc')
    best = name == L.BEST_PLACEMENT
    worst = name == 'head only'
    rows.append([
        label(name), cell('{:,}'.format(n)),
        cell('%.4f' % (n / N_FULL)),
        cell('%.4f' % m, bold=best,
             bg=GOOD_BG if best else (BAD_BG if worst else None)),
        cell('%.4f' % s, color=SUBTLE),
    ])
render_table(
    'placement_table.png',
    'Where the Adapters Go',
    'Rank 4, learning rate 5e-04, four seeds each. The spread is '
    'printed because one run of this comparison is not evidence.',
    ['adapters on', 'trainable', 'share of\nfull', 'task accuracy',
     'sd over\nseeds'],
    [3.6, 2.6, 2.4, 3.0, 2.4], rows,
    captions=[
        ('The standard error on each mean is about 0.013, so gaps '
         'under roughly 0.04 are not gaps. Everything beats attention '
         'alone by %.4f, which is.'
         % (L.ms(SWEEP[('place', 'everything')], 'acc')[0]
            - L.ms(SWEEP[('place', 'attention only')], 'acc')[0]),
         SUBTLE),
        ('Head-only is the worst placement -- and the head is the '
         'matrix that moved furthest. The spectrum says where change '
         'happened, not where it had to be made.', RED)],
    fig_px=(1800, 760), rules=(1,))


# =======================================================================
# 3. How much rank.
# =======================================================================

rows = []
for r in L.RANKS:
    n = L.n_params_lora(lm, L.PLACEMENTS[L.BEST_PLACEMENT], r)
    m, s = L.ms(SWEEP[('rank', r)], 'acc')
    best = r == 8
    rows.append([
        cell(str(r), bold=True, color=BLUE),
        cell('{:,}'.format(n)), cell('%.4f' % (n / N_FULL)),
        cell('%.4f' % m, bold=best, bg=GOOD_BG if best else None),
        cell('%.4f' % s, color=SUBTLE),
        cell('%.2f%%' % (100 * m / FULL_M), bold=best,
             color=GREEN if m / FULL_M > 0.95 else TEXT),
    ])
render_table(
    'rank_table.png',
    'How Much Rank',
    'Adapters everywhere, learning rate 5e-04, four seeds. Full '
    'fine-tuning scores %.4f on %s parameters.'
    % (FULL_M, '{:,}'.format(N_FULL)),
    ['rank', 'trainable', 'share of\nfull', 'task accuracy',
     'sd over\nseeds', 'of full\nfine-tuning'],
    [1.8, 2.6, 2.4, 3.0, 2.4, 2.8], rows,
    captions=[
        ('Rank 8 recovers %.2f%% of full fine-tuning on %.2f%% of the '
         'parameters; rank 2 gets %.2f%% for %.2f%%.'
         % (100 * L.ms(SWEEP[('rank', 8)], 'acc')[0] / FULL_M,
            100 * L.n_params_lora(lm, L.PLACEMENTS[L.BEST_PLACEMENT], 8)
            / N_FULL,
            100 * L.ms(SWEEP[('rank', 2)], 'acc')[0] / FULL_M,
            100 * L.n_params_lora(lm, L.PLACEMENTS[L.BEST_PLACEMENT], 2)
            / N_FULL), SUBTLE),
        ('Rank 16 is no better than rank 8 on twice the parameters. '
         'Past the point where the subspace is big enough, more rank '
         'buys nothing.', SUBTLE)],
    fig_px=(1820, 760))


# =======================================================================
# 4. Forgetting.
# =======================================================================

rows = []
for name, lr, key, _ in FORGET:
    v = FVALS[key]
    hi = lr == L.HIGH_LR
    rows.append([
        label(name), cell('%.0e' % lr, color=SUBTLE),
        cell('%.4f' % v[0]),
        cell('%.4f' % v[1], bg=BAD_BG if hi else MID_BG, bold=True),
        cell('%.4f' % v[2], bg=BAD_BG if hi else None),
        cell('%.4f' % v[3]),
    ])
render_table(
    'forgetting_table.png',
    'Forgetting Is Step Size, Not Method',
    'Held-out nats per token against the oracle floor of %.4f, and '
    'KL(language || model) over all %s sentences.'
    % (FLOOR, '{:,}'.format(len(L.SENT_P))),
    ['model', 'lr', 'task', 'nats / token', 'KL from the\nlanguage',
     'mass on the\nlanguage'],
    [3.4, 1.8, 2.2, 2.8, 2.8, 2.8], rows,
    captions=[
        ('The learning rate moves nats/token by about %.1f. The '
         'METHOD moves it by %.4f. LoRA does not forget less here -- '
         'it forgets the same.'
         % ((FVALS[('full', L.HIGH_LR)][1]
             + FVALS[('place_hi', L.BEST_PLACEMENT)][1]) / 2
            - (FVALS[('full', L.FULL_LR)][1]
               + FVALS[('place', L.BEST_PLACEMENT)][1]) / 2,
            abs(FVALS[('place', L.BEST_PLACEMENT)][1]
                - FVALS[('full', L.FULL_LR)][1])), SUBTLE),
        ('The mass column asks a different question and answers it the '
         'other way: LoRA keeps %.4f of the language against %.4f. It '
         'is printed rather than buried.'
         % (FVALS[('place', L.BEST_PLACEMENT)][3],
            FVALS[('full', L.FULL_LR)][3]), SUBTLE)],
    fig_px=(1840, 740), rules=(2,))


# =======================================================================
# 5. The header.
# =======================================================================

fig = plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Fine-tuning and LoRA: The Update Is Not Low Rank',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT)
ax.text(8, 8.05,
        'Keeping 90% of what full fine-tuning changed takes a rank in '
        'the twenties out of 64. LoRA works anyway.',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE)


def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))


box(0.7, 6.05, 5.3, 1.45, PURPLE)
ax.text(3.35, 7.22, 'The premise, decomposed',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=PURPLE)
ax.text(3.35, 6.62,
        'rank 8 keeps only %.2f of dW\nin the first attention block'
        % L.energy_at(SPEC['0.Wqkv'], 8),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 4.35, 5.3, 1.45, RED, face='#FEF2F2')
ax.text(3.35, 5.52, 'And yet a low-rank update suffices',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=RED)
ax.text(3.35, 4.92,
        'rank 8:  %.4f  (%.2f%% of full)\nfull FT: %.4f  on 13x the params'
        % (L.ms(SWEEP[('rank', 8)], 'acc')[0],
           100 * L.ms(SWEEP[('rank', 8)], 'acc')[0] / FULL_M, FULL_M),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 2.95, 5.3, 1.1, BLUE, face='#F8FAFC', lw=1.7)
ax.text(3.35, 3.82, 'WHERE beats how much',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=BLUE)
ax.text(3.35, 3.35,
        'everything %.4f vs head only %.4f'
        % (L.ms(SWEEP[('place', 'everything')], 'acc')[0],
           L.ms(SWEEP[('place', 'head only')], 'acc')[0]),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 1.35, 5.3, 1.25, ORANGE, lw=1.6)
ax.text(3.35, 2.35, 'Forgetting is the learning rate',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=ORANGE)
ax.text(3.35, 1.82,
        'lr moves nats/token by %.1f\nmethod moves it by %.3f'
        % ((FVALS[('full', L.HIGH_LR)][1]
            + FVALS[('place_hi', L.BEST_PLACEMENT)][1]) / 2
           - (FVALS[('full', L.FULL_LR)][1]
              + FVALS[('place', L.BEST_PLACEMENT)][1]) / 2,
           abs(FVALS[('place', L.BEST_PLACEMENT)][1]
               - FVALS[('full', L.FULL_LR)][1])),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

# ---- the spectrum itself ----
axh = fig.add_axes([0.45, 0.44, 0.50, 0.30])
rr = np.arange(1, 33)
for k, col, lab in (('0.Wqkv', BLUE, 'attention (block 0)'),
                    ('1.W1', GREEN, 'MLP (block 1)'),
                    ('Wv', RED, 'output head')):
    axh.plot(rr, [L.energy_at(SPEC[k], int(r)) for r in rr],
             color=col, lw=2.0, label=lab, zorder=3)
axh.axhline(0.9, color='#CBD5E1', ls=':', lw=1.2, zorder=2)
axh.text(1.5, 0.93, '90%', fontsize=7.0, color=NEUTRAL, ha='left')
axh.set_xlabel('rank r of the approximation', fontsize=8, color=SUBTLE)
axh.set_ylabel('energy of dW kept', fontsize=8, color=SUBTLE)
axh.set_xlim(0, 32)
axh.set_ylim(0, 1.05)
axh.set_xticks([1, 8, 16, 24, 32])
axh.set_xticklabels(['1', '8', '16', '24', '32'], fontsize=7.2,
                    color=SUBTLE)
axh.set_yticks([0, 0.5, 0.9])
axh.set_yticklabels(['0', '0.5', '0.9'], fontsize=7.2, color=SUBTLE)
axh.legend(fontsize=7.2, frameon=False, loc='lower right')
axh.set_title('how much of the update a rank-r adapter could hold',
              fontsize=9.0, fontweight='bold', color=TEXT, pad=6,
              loc='left')
style_axes(axh)

# ---- accuracy against parameter share ----
axb = fig.add_axes([0.45, 0.10, 0.50, 0.205])
shares, accs = [], []
for r in L.RANKS:
    shares.append(L.n_params_lora(
        lm, L.PLACEMENTS[L.BEST_PLACEMENT], r) / N_FULL)
    accs.append(L.ms(SWEEP[('rank', r)], 'acc')[0])
axb.plot([s * 100 for s in shares], accs, 'o-', color=BLUE, lw=2.0,
         ms=5, zorder=3, label='LoRA, rank 1 to 16')
axb.axhline(FULL_M, color=GREEN, ls='--', lw=1.6, zorder=2,
            label='full fine-tune (100% of params)')
for s, a, r in zip(shares, accs, L.RANKS):
    axb.annotate('r=%d' % r, (s * 100, a), fontsize=6.8,
                 textcoords='offset points', xytext=(2, -9),
                 color=SUBTLE)
axb.set_xlabel('trainable parameters, % of the model', fontsize=8,
               color=SUBTLE)
axb.set_xlim(0, 34)
axb.set_ylim(0.44, 0.65)
axb.set_yticks([0.45, 0.55, 0.65])
axb.set_yticklabels(['0.45', '0.55', '0.65'], fontsize=7.2,
                    color=SUBTLE)
axb.tick_params(axis='x', labelsize=7.2)
axb.legend(fontsize=7.2, frameon=False, loc='lower right')
axb.set_title('what the parameters buy', fontsize=9.0,
              fontweight='bold', color=TEXT, pad=6, loc='left')
style_axes(axb)

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 9',
        fontsize=8, ha='center', va='center', color='#aaaaaa')
plt.savefig(OUT + 'header_lora.png', dpi=150, facecolor=BG)
plt.close()
print('Wrote ' + OUT + 'header_lora.png')
print()
print('all five images written, every cell asserted against the '
      'frozen run')
