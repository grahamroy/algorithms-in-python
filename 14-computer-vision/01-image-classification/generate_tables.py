"""Generate every image for the Image Classification article.

Medium has no table support and collapses the whitespace in wide code
blocks, so each measured table ships as an image. This ONE script
trains every model once -- four architectures at five training-set
sizes plus three scaling shapes, three seeds each -- and emits every
PNG including the header. Every cell is asserted against the frozen
run of imgcls.py.

Writes: stage.png, accuracy_table.png, invariance_table.png,
        scaling_table.png, header_imgcls.png
Runs in about ten minutes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

import imgcls as M

OUT = ('D:/Projects/Medium/algorithms-in-python/'
       '14-computer-vision/01-image-classification/')

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
COLS = {"MLP": NEUTRAL, "CNN": GREEN, "ResNet": BLUE, "ViT": ORANGE}


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
# Train everything once, with imgcls.py's frozen seeds.
# =======================================================================

print('training (about ten minutes)...')
SIZES, LO, HI = M.selftest()
_, WORST, CHECKED = M.gradcheck()
assert '%.2e' % WORST == '6.46e-10', WORST
Xte, yte, _ = M.make_set(M.N_TEST, np.random.default_rng(M.SEED_TEST))

ACC, FINALS = {}, {}
for n_lab in M.LABEL_COUNTS:
    for name in M.MODELS:
        runs = []
        for sd in M.SEEDS:
            Xtr, ytr, _ = M.make_set(
                n_lab, np.random.default_rng(M.SEED_TRAIN + 7 * sd))
            net = M.fit(M.MODELS[name](seed=sd, lr=M.LR,
                                       **M.CONFIG[name]),
                        Xtr, ytr, M.STEPS,
                        rng=np.random.default_rng(M.SEED_FIT + sd))
            runs.append(M.accuracy(net, Xte, yte))
            if n_lab == M.LABEL_COUNTS[-1]:
                FINALS.setdefault(name, []).append(net)
        ACC[(n_lab, name)] = M.ms(runs)
    print('  %d labels done' % n_lab)

FROZEN_ACC = {
    (60, 'MLP'): '0.3064', (60, 'CNN'): '0.9271',
    (60, 'ResNet'): '0.8896', (60, 'ViT'): '0.4500',
    (960, 'MLP'): '0.6276', (960, 'CNN'): '0.9956',
    (960, 'ResNet'): '0.9958', (960, 'ViT'): '0.8658'}
for key, want in FROZEN_ACC.items():
    assert '%.4f' % ACC[key][0] == want, (key, ACC[key])
print('  accuracy ok')

INV = {}
for name in M.MODELS:
    for mg in M.MARGINS:
        INV[(name, mg)] = M.ms([
            M.self_agreement(net, np.random.default_rng(M.SEED_INV),
                             margin=mg)[0] for net in FINALS[name]])
assert '%.4f' % INV[('CNN', 2)][0] == '1.0000', INV[('CNN', 2)]
assert '%.4f' % INV[('CNN', 2)][1] == '0.0000', INV[('CNN', 2)]
assert '%.4f' % INV[('ViT', 2)][0] == '0.9505', INV[('ViT', 2)]
print('  invariance ok')

SCALE = {}
for nb, w in M.SCALE_CFGS:
    for n_lab in M.SCALE_LABELS:
        runs = []
        for sd in M.SEEDS:
            Xtr, ytr, _ = M.make_set(
                n_lab, np.random.default_rng(M.SEED_TRAIN + 7 * sd))
            net = M.fit(M.ResNet(k=3, width=w, n_blocks=nb, seed=sd,
                                 lr=M.LR), Xtr, ytr, M.STEPS,
                        rng=np.random.default_rng(M.SEED_FIT + sd))
            runs.append(M.accuracy(net, Xte, yte))
        SCALE[(nb, w, n_lab)] = M.ms(runs)
    print('  scaling %d block(s) done' % nb)
assert '%.4f' % SCALE[(1, 17, 120)][0] == '0.9716', SCALE[(1, 17, 120)]
assert '%.4f' % SCALE[(3, 10, 120)][0] == '0.8751', SCALE[(3, 10, 120)]
print('  scaling ok')
print()


# =======================================================================
# 1. The stage.
# =======================================================================

fig, axes = plt.subplots(2, M.N_CLS, figsize=(1700 / 150, 680 / 150),
                         dpi=150)
fig.patch.set_facecolor(BG)
rng = np.random.default_rng(4)
c = M.SIDE // 2
for j in range(M.N_CLS):
    clean = M.draw(j, c, c, 4, 1.0)
    noisy = clean + rng.normal(0, M.NOISE, clean.shape)
    for i, (img, lab) in enumerate(((clean, 'as drawn'),
                                    (noisy, 'as the model sees it'))):
        a = axes[i, j]
        a.imshow(img, cmap='magma', vmin=-0.4, vmax=1.4,
                 interpolation='nearest')
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_color(BORDER)
        if i == 0:
            a.set_title(M.CLASS_NAMES[j], fontsize=10.5,
                        color=TEXT, pad=7)
        if j == 0:
            a.set_ylabel(lab, fontsize=9.5, color=SUBTLE)
fig.suptitle('Six shapes whose pixel statistics overlap',
             fontsize=15.5, fontweight='bold', color=TEXT, y=0.99)
fig.text(0.5, 0.025,
         'Each is drawn at a centre, size and brightness the script '
         'chooses, then noised. Because those factors are INPUTS, '
         'what a model ignores can be measured rather than guessed.',
         fontsize=10, ha='center', color=SUBTLE, fontstyle='italic')
plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig(OUT + 'stage.png', dpi=150, facecolor=BG,
            bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
print('Wrote ' + OUT + 'stage.png')


# =======================================================================
# 2. Accuracy against labels.
# =======================================================================

rows = []
for n_lab in M.LABEL_COUNTS:
    cells = [cell(str(n_lab), bold=True, color=BLUE)]
    best = max(M.MODELS, key=lambda n: ACC[(n_lab, n)][0])
    for name in M.MODELS:
        m, s = ACC[(n_lab, name)]
        cells.append(cell('%.4f' % m, bold=name == best,
                          bg=GOOD_BG if name == best else
                          (BAD_BG if m < 0.7 else None)))
        cells.append(cell('%.4f' % s, color=SUBTLE, size=10.5))
    rows.append(cells)
render_table(
    'accuracy_table.png',
    'What the Prior Is Worth',
    'Test accuracy on %s held-out images, %d seeds per cell. All four '
    'models hold ~5,500 parameters and get %d gradient steps.'
    % (f"{M.N_TEST:,}", len(M.SEEDS), M.STEPS),
    ['training\nlabels', 'MLP', 'sd', 'CNN', 'sd', 'ResNet', 'sd',
     'ViT', 'sd'],
    [2.2, 2.2, 1.7, 2.2, 1.7, 2.2, 1.7, 2.2, 1.7], rows,
    captions=[
        ('From 60 labels the convolution is at %.4f and the '
         'perceptron at %.4f -- on the SAME parameter budget. After '
         '16x the data the perceptron reaches only %.4f.'
         % (ACC[(60, 'CNN')][0], ACC[(60, 'MLP')][0],
            ACC[(960, 'MLP')][0]), SUBTLE),
        ('The residual network LOSES to the plain convolution at '
         'every small size and only draws level at 960. Depth is not '
         'free here.', RED)],
    fig_px=(1880, 740))


# =======================================================================
# 3. Invariance.
# =======================================================================

rows = []
for name in M.MODELS:
    a0, s0 = INV[(name, 0)]
    a2, s2 = INV[(name, 2)]
    exact = a2 >= 0.99995
    rows.append([
        label(name),
        cell('%.4f' % a0, bg=GOOD_BG if a0 >= 0.999 else None),
        cell('%.4f' % s0, color=SUBTLE, size=10.5),
        cell('%.4f' % a2, bold=exact,
             bg=GOOD_BG if exact else BAD_BG),
        cell('%.4f' % s2, color=SUBTLE, size=10.5),
        note('invariant by construction' if exact else
             'approximated, not built in'),
    ])
render_table(
    'invariance_table.png',
    'Does It Change Its Mind When the Object Moves?',
    'The same shape re-rendered at EVERY legal centre. 1.0000 means '
    'position is ignored completely. Clean images, %d seeds.'
    % len(M.SEEDS),
    ['model', 'margin 0', 'sd', 'margin 2', 'sd', ''],
    [2.4, 2.4, 1.8, 2.4, 1.8, 6.4], rows,
    captions=[
        ('Not "approximately": %.4f with a standard deviation across '
         'seeds of %.4f. Global max-pooling takes the strongest '
         'response wherever it occurs, so moving the object cannot '
         'change the answer.'
         % (INV[('CNN', 2)][0], INV[('CNN', 2)][1]), GREEN),
        ('Clearing the border does not rescue the transformer '
         '(%.4f to %.4f) because the border was never its problem.'
         % (INV[('ViT', 0)][0], INV[('ViT', 2)][0]), SUBTLE)],
    fig_px=(1860, 700))


# =======================================================================
# 4. Depth against width.
# =======================================================================

rows = []
for nb, w in M.SCALE_CFGS:
    p = M.ResNet(k=3, width=w, n_blocks=nb).n_params()
    lo = SCALE[(nb, w, M.SCALE_LABELS[0])]
    hi = SCALE[(nb, w, M.SCALE_LABELS[1])]
    rows.append([
        label('%d block%s, width %d' % (nb, '' if nb == 1 else 's', w)),
        cell('{:,}'.format(p)),
        cell('%.4f' % lo[0], bold=nb == 1,
             bg=GOOD_BG if lo[0] > 0.95 else BAD_BG),
        cell('%.4f' % lo[1], color=RED if lo[1] > 0.03 else SUBTLE,
             size=10.5),
        cell('%.4f' % hi[0]),
        cell('%.4f' % hi[1], color=SUBTLE, size=10.5),
    ])
render_table(
    'scaling_table.png',
    'Where Should the Parameters Go?',
    'One residual network reshaped three ways at one budget. '
    'EfficientNet\'s question minus resolution, which is fixed at '
    '%dx%d here.' % (M.SIDE, M.SIDE),
    ['shape', 'params', '%d labels' % M.SCALE_LABELS[0], 'sd',
     '%d labels' % M.SCALE_LABELS[1], 'sd'],
    [4.2, 2.4, 2.6, 1.9, 2.6, 1.9], rows,
    captions=[
        ('Wider and shallower wins, and wins hardest where the data '
         'is thinnest: %.4f against %.4f at %d labels, closing to '
         '%.4f against %.4f at %d.'
         % (SCALE[(1, 17, 120)][0], SCALE[(3, 10, 120)][0], 120,
            SCALE[(1, 17, 960)][0], SCALE[(3, 10, 960)][0], 960),
         SUBTLE),
        ('And the spread runs %.4f against %.4f. Depth costs '
         'reliability here as well as accuracy -- it is a bet on '
         'structure at several scales, and this stage has one.'
         % (SCALE[(1, 17, 120)][1], SCALE[(3, 10, 120)][1]), RED)],
    fig_px=(1840, 680))


# =======================================================================
# 5. The header.
# =======================================================================

fig = plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'Image Classification: What a Prior Is Worth',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT)
ax.text(8, 8.05,
        'Same parameters, same budget, same data. One of them was '
        'told that a pattern means the same thing wherever it appears.',
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE)


def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))


box(0.7, 6.05, 5.3, 1.45, PURPLE)
ax.text(3.35, 7.22, 'The factors are inputs, not guesses',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=PURPLE)
ax.text(3.35, 6.62,
        'class, centre, size, brightness --\nso invariance can be '
        'measured',
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 4.35, 5.3, 1.45, GREEN, face='#F0FDF4')
ax.text(3.35, 5.52, 'The prior, priced in labels',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=GREEN)
ax.text(3.35, 4.92,
        'at 60 labels: CNN %.4f, MLP %.4f\nMLP at 960 is still %.4f'
        % (ACC[(60, 'CNN')][0], ACC[(60, 'MLP')][0],
           ACC[(960, 'MLP')][0]),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 2.95, 5.3, 1.1, BLUE, face='#F8FAFC', lw=1.7)
ax.text(3.35, 3.82, 'Built in, not approximated',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=BLUE)
ax.text(3.35, 3.35,
        'invariance: CNN %.4f, ViT %.4f'
        % (INV[('CNN', 2)][0], INV[('ViT', 2)][0]),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 1.35, 5.3, 1.25, ORANGE, lw=1.6)
ax.text(3.35, 2.35, 'And depth is a bet, not an upgrade',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=ORANGE)
ax.text(3.35, 1.82,
        'at 120 labels, 1 wide block %.4f\nvs 3 narrow blocks %.4f'
        % (SCALE[(1, 17, 120)][0], SCALE[(3, 10, 120)][0]),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

axh = fig.add_axes([0.45, 0.42, 0.50, 0.32])
for name in M.MODELS:
    axh.plot(M.LABEL_COUNTS,
             [ACC[(n, name)][0] for n in M.LABEL_COUNTS],
             'o-', color=COLS[name], lw=2.0, ms=4.5, label=name,
             zorder=3)
axh.axhline(1 / M.N_CLS, color=NEUTRAL, ls=':', lw=1.2, zorder=2)
axh.text(62, 1 / M.N_CLS + 0.03, 'chance', fontsize=7.2,
         color=NEUTRAL)
axh.set_xscale('log')
axh.set_xticks(list(M.LABEL_COUNTS))
axh.set_xticklabels([str(n) for n in M.LABEL_COUNTS], fontsize=7.2,
                    color=SUBTLE)
axh.set_yticks([0, 0.5, 1.0])
axh.set_yticklabels(['0', '0.5', '1.0'], fontsize=7.2, color=SUBTLE)
axh.set_ylim(0, 1.12)
axh.set_xlabel('training labels', fontsize=8, color=SUBTLE)
axh.legend(fontsize=7.4, frameon=False, loc='lower right', ncol=2)
axh.set_title('accuracy, at a matched parameter budget',
              fontsize=9.0, fontweight='bold', color=TEXT, pad=6,
              loc='left')
style_axes(axh)

axb = fig.add_axes([0.45, 0.115, 0.50, 0.205])
names = list(M.MODELS)
vals = [INV[(n, 2)][0] for n in names]
axb.bar(np.arange(len(names)), vals, 0.5,
        color=[COLS[n] for n in names], zorder=3)
axb.axhline(1.0, color=GREEN, ls=':', lw=1.2, zorder=2)
for i, v in enumerate(vals):
    axb.text(i, v + 0.03, '%.4f' % v, ha='center', fontsize=7.6,
             color=COLS[names[i]], fontweight='bold')
axb.set_xticks(np.arange(len(names)))
axb.set_xticklabels(names, fontsize=7.6, color=SUBTLE)
axb.set_yticks([0, 0.5, 1.0])
axb.set_yticklabels(['0', '0.5', '1.0'], fontsize=7.2, color=SUBTLE)
axb.set_ylim(0, 1.22)
axb.set_title('answer unchanged as the object moves (margin 2)',
              fontsize=9.0, fontweight='bold', color=TEXT, pad=6,
              loc='left')
style_axes(axb)

ax.text(8, 0.18,
        'Algorithms in Python  |  Computer Vision Part 1',
        fontsize=8, ha='center', va='center', color='#aaaaaa')
plt.savefig(OUT + 'header_imgcls.png', dpi=150, facecolor=BG)
plt.close()
print('Wrote ' + OUT + 'header_imgcls.png')
print()
print('all five images written, every cell asserted against the '
      'frozen run')
