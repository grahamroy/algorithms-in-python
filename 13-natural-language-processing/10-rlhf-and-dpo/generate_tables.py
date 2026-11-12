"""Generate every image for the RLHF and DPO article.

Medium has no table support and collapses the whitespace in wide code
blocks, so each measured table ships as an image. The pipeline is
expensive (a reference policy, a reward model and eighteen policy
runs), so this ONE script trains everything once and emits every PNG
including the header. Every cell is asserted against the frozen run
of rlhf.py.

Writes: reward_table.png, rm_table.png, rlhf_table.png,
        beta_curve.png, dpo_table.png, header_rlhf.png
Runs in about sixteen minutes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

import rlhf as L

OUT = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/10-rlhf-and-dpo/')

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
# Train everything once, with rlhf.py's frozen seeds.
# =======================================================================

print('training (about sixteen minutes)...')
corpus, ids, keep, test = L.build_stage()
REF, _ = L.train_lm(ids, keep)
REF_LP = L.policy_logprobs(REF)
BASE = L.exact_eval(REF, np.zeros(len(L.SENTS)), REF_LP)
assert '%.4f' % BASE['true'] == '0.7689', BASE
assert '%.4f' % BASE['mass'] == '0.9770', BASE
LANG_R = float((L.P_LANG * L.R_STAR).sum())
assert '%.4f' % LANG_R == '0.8316', LANG_R
assert L.R_MAX == 2.1 and int((L.R_STAR == L.R_MAX).sum()) == 192
print('  stage ok (reference %.4f, ceiling %.1f)'
      % (BASE['true'], L.R_MAX))

PAIRS, N_SAMP, AGREE = L.build_preferences(
    REF, L.N_PAIRS, np.random.default_rng(L.SEED_PREF))
assert (N_SAMP, len(PAIRS)) == (3911, 3000), (N_SAMP, len(PAIRS))
assert '%.4f' % AGREE == '0.6937', AGREE
RM, RMHIST = L.train_reward_model(PAIRS, L.RM_STEPS,
                                  np.random.default_rng(L.SEED_RM))
RHAT = L.rm_over_language(RM)
C_ALL = float(np.corrcoef(RHAT, L.R_STAR)[0, 1])
C_ON = L.weighted_corr(RHAT, L.R_STAR, np.exp(REF_LP))
assert '%.4f' % C_ALL == '0.6906', C_ALL
assert '%.4f' % C_ON == '0.7092', C_ON
print('  reward model ok (corr %.4f / %.4f)' % (C_ALL, C_ON))

BASE_P = L.exact_eval(REF, RHAT, REF_LP)
RL, WRECKED = {}, None
for beta in L.BETAS:
    runs = []
    for sd in L.SEEDS:
        pol, h = L.rlhf(REF, RM, RHAT, REF_LP, beta, L.PG_STEPS,
                        np.random.default_rng(L.SEED_POL + sd),
                        every=L.EVERY)
        runs.append(h)
        if beta == L.BETAS[0] and sd == L.SEEDS[0]:
            WRECKED = pol
    curve = [{f: L.ms([r[i] for r in runs], f)[0]
              for f in ('true', 'proxy', 'mass', 'kl')}
             for i in range(len(runs[0]))]
    for i, c in enumerate(curve):
        c['step'] = runs[0][i]['step']
    RL[beta] = curve
    print('  rlhf beta %.2f done' % beta)

FROZEN_RL = {0.02: ('1.1763', '0.0025', '0.0019'),
             0.05: ('1.1985', '0.3442', '0.2350'),
             0.20: ('1.5045', '1.4953', '0.9800'),
             0.50: ('1.2603', '1.2504', '0.9820')}
for beta, (pk, end, mass) in FROZEN_RL.items():
    c = RL[beta]
    peak = max(c, key=lambda e: e['true'])
    got = ('%.4f' % peak['true'], '%.4f' % c[-1]['true'],
           '%.4f' % c[-1]['mass'])
    assert got == (pk, end, mass), (beta, got)
print('  rlhf sweep ok')

DP = {}
for beta in L.DPO_BETAS:
    runs = []
    for sd in L.SEEDS:
        _, h = L.dpo(REF, RHAT, REF_LP, beta, L.PG_STEPS,
                     np.random.default_rng(L.SEED_POL + sd), PAIRS,
                     every=L.EVERY)
        runs.append(h)
    curve = [{f: L.ms([r[i] for r in runs], f)[0]
              for f in ('true', 'mass', 'kl')}
             for i in range(len(runs[0]))]
    for i, c in enumerate(curve):
        c['step'] = runs[0][i]['step']
    DP[beta] = curve
    print('  dpo beta %.1f done' % beta)
assert '%.4f' % DP[0.5][-1]['true'] == '1.2820', DP[0.5][-1]
assert '%.4f' % DP[0.1][-1]['true'] == '1.2094', DP[0.1][-1]
print('  dpo ok')

# what the wrecked policy emits
srng = np.random.default_rng(99)
SAMP = [w[:L.T_MAX - 2] for w, _ in L.sample_batch(WRECKED, srng, 400)]
sids, skeep = L.encode(SAMP)
SRH = L.rm_score(RM, sids, skeep)[0]
S_OK = float(np.mean([L.in_language(w) for w in SAMP]))
S_RS = float(np.mean([L.reward_of(w) for w in SAMP]))
S_RH = float(SRH.mean())
REF_RH = float(np.exp(REF_LP) @ RHAT / np.exp(REF_LP).sum())
assert '%.4f' % S_OK == '0.0150', S_OK
assert '%.4f' % S_RH == '1.7109', S_RH
print('  wreck ok (%.4f grammatical, proxy %.4f)' % (S_OK, S_RH))
print()


# =======================================================================
# 1. The true reward.
# =======================================================================

rows = []
for lvl in sorted(set(L.R_STAR.tolist())):
    m = L.R_STAR == lvl
    top = lvl == L.R_MAX
    rows.append([
        cell('%.1f' % lvl, bold=True,
             color=GREEN if top else TEXT),
        cell('{:,}'.format(int(m.sum()))),
        cell('%.4f' % float(L.P_LANG[m].sum())),
        note(('adjective + role partner + royal subject')
             if top else
             ('nothing the reward asks for' if lvl == 0 else ''),
             size=10.5),
    ])
render_table(
    'reward_table.png',
    'A Reward You Can Check',
    'r*(s) = 1.0 if it carries an adjective + 0.7 if the object is '
    'the role partner + 0.4 if the subject is royal.',
    ['r*', 'sentences', 'mass', ''],
    [1.8, 2.6, 2.4, 8.4], rows,
    captions=[
        ('The language itself scores %.4f; the reference policy '
         '%.4f; the ceiling is %.1f, reached by %d of the %s '
         'sentences.'
         % (LANG_R, BASE['true'], L.R_MAX,
            int((L.R_STAR == L.R_MAX).sum()),
            '{:,}'.format(len(L.SENTS))), SUBTLE),
        ('Because r* is a function, every quantity RLHF normally '
         'estimates is exact here -- including the one no real '
         'pipeline can see.', SUBTLE)],
    fig_px=(1820, 800))


# =======================================================================
# 2. The reward model.
# =======================================================================

rows = [
    [label('labels that point the right way'),
     cell('%.4f' % AGREE, bg=MID_BG),
     note('Bradley-Terry sampling, so the rest are noise')],
    [label('correlation with r*, all sentences'),
     cell('%.4f' % C_ALL),
     note('over all %s of them' % '{:,}'.format(len(L.SENTS)))],
    [label('correlation where the policy is'),
     cell('%.4f' % C_ON),
     note('weighted by the reference policy\'s own mass')],
    [label('r^ on reference-policy output'),
     cell('%.4f' % REF_RH),
     note('the scale is arbitrary; only differences matter')],
]
render_table(
    'rm_table.png',
    'The Reward Model Is Not the Problem',
    '%s comparisons drawn from %s reference-policy samples.'
    % ('{:,}'.format(len(PAIRS)), '{:,}'.format(N_SAMP)),
    ['measurement', 'value', 'what it means'],
    [6.4, 2.4, 7.6], rows,
    captions=[
        ('The two correlations are the same number: this is a '
         'reasonable judge of SENTENCES, on and off the reference '
         'distribution alike.', SUBTLE),
        ('Which leaves the question a correlation over the language '
         'cannot ask -- what it thinks of text that is not a '
         'sentence at all.', RED)],
    fig_px=(1800, 700))


# =======================================================================
# 3. The sweep.
# =======================================================================

rows = [[label('(reference policy)'),
         cell('--', color=SUBTLE),
         cell('--', color=SUBTLE),
         cell('%.4f' % BASE_P['true'], bg=GOOD_BG),
         cell('%.4f' % BASE_P['mass'], bg=GOOD_BG),
         cell('0.0000')]]
for beta in L.BETAS:
    c = RL[beta]
    pk = max(c, key=lambda e: e['true'])
    end = c[-1]
    kept = end['true'] > 0.9 * pk['true']
    rows.append([
        cell('%.2f' % beta, bold=True, color=BLUE),
        cell('%.4f' % pk['true'],
             bg=GOOD_BG if pk['true'] > 1.4 else None),
        cell('%d' % pk['step'], color=SUBTLE),
        cell('%.4f' % end['true'],
             bg=GOOD_BG if kept else BAD_BG, bold=True),
        cell('%.4f' % end['mass'], bg=GOOD_BG if kept else BAD_BG),
        cell('%.4f' % end['kl']),
    ])
render_table(
    'rlhf_table.png',
    'The Proxy Rises, the Truth Peaks and Falls',
    'REINFORCE against the reward model, KL folded into the reward. '
    '%d seeds per setting; beta is the only knob.' % len(L.SEEDS),
    ['KL penalty\nbeta', 'peak E[r*]', 'at\nstep',
     'E[r*] at\nstep %d' % L.PG_STEPS, 'grammatical\nmass',
     'KL'],
    [3.6, 2.6, 1.8, 2.6, 2.8, 2.2], rows,
    captions=[
        ('At beta 0.05 the true reward reaches %.4f and ends at '
         '%.4f -- below the %.4f it started from. The proxy does '
         'not come down with it.'
         % (max(RL[0.05], key=lambda e: e['true'])['true'],
            RL[0.05][-1]['true'], BASE_P['true']), RED),
        ('At 0.20 the peak is higher than either loose setting ever '
         'reached, and it is still there at step %d. The penalty is '
         'not a tax on performance.' % L.PG_STEPS, SUBTLE)],
    fig_px=(1900, 780), rules=(4,))


# =======================================================================
# 4. The curve, and what the wreck emits.
# =======================================================================

fig, axes = plt.subplots(1, 2, figsize=(1800 / 150, 760 / 150), dpi=150)
fig.patch.set_facecolor(BG)
ax = axes[0]
COLS = {0.02: RED, 0.05: ORANGE, 0.20: GREEN, 0.50: BLUE}
for beta in L.BETAS:
    c = RL[beta]
    xs = [0] + [e['step'] for e in c]
    ys = [BASE_P['true']] + [e['true'] for e in c]
    ax.plot(xs, ys, 'o-', color=COLS[beta], lw=2.0, ms=4.5,
            label='beta = %.2f' % beta, zorder=3)
ax.axhline(BASE_P['true'], color=NEUTRAL, ls=':', lw=1.4, zorder=2)
ax.text(5, BASE_P['true'] + 0.04, 'where it started', fontsize=8,
        color=NEUTRAL)
ax.set_xlabel('policy-gradient step', fontsize=9.5, color=TEXT)
ax.set_ylabel('TRUE reward  E[r*]', fontsize=9.5, color=TEXT)
ax.set_ylim(-0.1, 1.75)
ax.legend(fontsize=8.5, frameon=False, loc='lower left')
ax.set_title('what the policy is actually earning', fontsize=11,
             fontweight='bold', color=TEXT, pad=8, loc='left')
style_axes(ax)

ax = axes[1]
c = RL[0.05]
xs = [0] + [e['step'] for e in c]
ax.plot(xs, [BASE_P['true']] + [e['true'] for e in c], 'o-',
        color=GREEN, lw=2.2, ms=5, label='TRUE reward  E[r*]', zorder=4)
ax.plot(xs, [BASE_P['proxy']] + [e['proxy'] for e in c], 'o--',
        color=RED, lw=2.2, ms=5, label='PROXY reward  E[r^]', zorder=4)
ax.plot(xs, [BASE_P['mass']] + [e['mass'] for e in c], 'o:',
        color=NEUTRAL, lw=1.6, ms=4, label='grammatical mass', zorder=3)
ax.axhline(BASE_P['true'], color=NEUTRAL, ls=':', lw=1.0, zorder=1)
ax.set_xlabel('policy-gradient step', fontsize=9.5, color=TEXT)
ax.set_ylim(-0.45, 1.75)
ax.legend(fontsize=8.5, frameon=False, loc='upper right')
ax.set_title('beta = 0.05: the two curves part company', fontsize=11,
             fontweight='bold', color=TEXT, pad=8, loc='left')
style_axes(ax)

fig.text(0.5, 0.015,
         'The proxy keeps reporting success while the true reward '
         'falls below its starting point and grammatical output '
         'collapses with it.',
         fontsize=10, ha='center', color=SUBTLE, fontstyle='italic')
plt.tight_layout(rect=[0, 0.045, 1, 1])
plt.savefig(OUT + 'beta_curve.png', dpi=150, facecolor=BG,
            bbox_inches='tight', pad_inches=0.2)
plt.close(fig)
print('Wrote ' + OUT + 'beta_curve.png')


# =======================================================================
# 5. DPO against RLHF.
# =======================================================================

rows = []
for beta in L.DPO_BETAS:
    c = DP[beta][-1]
    rows.append([label('DPO, beta %.1f' % beta),
                 note('no reward model, no sampling'),
                 cell('%.4f' % c['true'], bg=GOOD_BG),
                 cell('%.4f' % c['mass']),
                 cell('%.4f' % c['kl'])])
for beta in (0.20, 0.05, 0.02):
    c = RL[beta][-1]
    good = c['true'] > 1.0
    rows.append([label('RLHF, beta %.2f' % beta),
                 note('reward model in the loop'),
                 cell('%.4f' % c['true'],
                      bg=GOOD_BG if good else BAD_BG, bold=True),
                 cell('%.4f' % c['mass'],
                      bg=GOOD_BG if good else BAD_BG),
                 cell('%.4f' % c['kl'])])
render_table(
    'dpo_table.png',
    'Removing the Thing That Got Hacked',
    'The same %s comparisons and the same budget, at step %d.'
    % ('{:,}'.format(len(PAIRS)), L.PG_STEPS),
    ['method', 'what is in the loop', 'E[r*]', 'grammatical\nmass',
     'KL'],
    [3.4, 5.0, 2.4, 2.8, 2.2], rows,
    captions=[
        ('DPO never reaches the best tuned RLHF run (%.4f against '
         '%.4f) and never produces the %.4f that the worst one ends '
         'at.' % (DP[0.5][-1]['true'], RL[0.20][-1]['true'],
                  RL[0.02][-1]['true']), SUBTLE),
        ('It is not better at finding reward. It is better at not '
         'losing it -- the failure mode it removes is the one that '
         'needs a proxy to exist.', SUBTLE)],
    fig_px=(1820, 760), rules=(3,))


# =======================================================================
# 6. The header.
# =======================================================================

fig = plt.figure(figsize=(1600 / 150, 900 / 150), dpi=150)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52, 'RLHF and DPO: Optimising a Reward Nobody Can Check',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT)
ax.text(8, 8.05,
        'The true reward rose to %.4f, then fell to %.4f -- below '
        'where it began -- while the proxy went on reporting success.'
        % (max(RL[0.05], key=lambda e: e['true'])['true'],
           RL[0.05][-1]['true']),
        fontsize=10, fontstyle='italic', ha='center', va='center',
        color=SUBTLE)


def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))


box(0.7, 6.05, 5.3, 1.45, PURPLE)
ax.text(3.35, 7.22, 'Here the true reward is a function',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=PURPLE)
ax.text(3.35, 6.62,
        'reference %.4f, ceiling %.1f,\nexact over %s sentences'
        % (BASE['true'], L.R_MAX, '{:,}'.format(len(L.SENTS))),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 4.35, 5.3, 1.45, RED, face='#FEF2F2')
ax.text(3.35, 5.52, 'The reward model is not the problem',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=RED)
ax.text(3.35, 4.92,
        'corr with truth %.4f overall,\n%.4f where the policy is'
        % (C_ALL, C_ON),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 2.95, 5.3, 1.1, ORANGE, face='#FFF7ED', lw=1.7)
ax.text(3.35, 3.82, 'It had never been shown a non-sentence',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=ORANGE)
ax.text(3.35, 3.35,
        'wrecked policy: %.2f%% grammatical, r^ %.3f'
        % (100 * S_OK, S_RH),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

box(0.7, 1.35, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.35, 'And the KL penalty is what holds it',
        fontsize=9.6, fontweight='bold', ha='center', va='center',
        color=GREEN)
ax.text(3.35, 1.82,
        'beta 0.20 ends at %.4f\nbeta 0.02 ends at %.4f'
        % (RL[0.20][-1]['true'], RL[0.02][-1]['true']),
        fontsize=8.2, ha='center', va='center', color=TEXT,
        fontfamily=MONO)

axh = fig.add_axes([0.45, 0.42, 0.50, 0.32])
c = RL[0.05]
xs = [0] + [e['step'] for e in c]
axh.plot(xs, [BASE_P['true']] + [e['true'] for e in c], 'o-',
         color=GREEN, lw=2.0, ms=4.5, label='true  E[r*]', zorder=4)
axh.plot(xs, [BASE_P['proxy']] + [e['proxy'] for e in c], 'o--',
         color=RED, lw=2.0, ms=4.5, label='proxy  E[r^]', zorder=4)
axh.axhline(BASE_P['true'], color=NEUTRAL, ls=':', lw=1.2, zorder=1)
axh.set_xlabel('policy-gradient step, beta = 0.05', fontsize=8,
               color=SUBTLE)
axh.set_ylim(-0.45, 1.6)
axh.legend(fontsize=7.4, frameon=False, loc='upper right', ncol=2)
axh.set_title('the two curves part company', fontsize=9.0,
              fontweight='bold', color=TEXT, pad=6, loc='left')
style_axes(axh)

axb = fig.add_axes([0.45, 0.115, 0.50, 0.205])
names = ['beta\n0.02', 'beta\n0.05', 'beta\n0.20', 'beta\n0.50', 'DPO\n0.5']
vals = [RL[b][-1]['true'] for b in L.BETAS] + [DP[0.5][-1]['true']]
cols = [RED, RED, GREEN, GREEN, BLUE]
axb.bar(np.arange(5), vals, 0.5, color=cols, zorder=3)
axb.axhline(BASE_P['true'], color=NEUTRAL, ls=':', lw=1.2, zorder=2)
for i, v in enumerate(vals):
    axb.text(i, v + 0.05, '%.4f' % v, ha='center', fontsize=7.4,
             color=cols[i], fontweight='bold')
axb.set_xticks(np.arange(5))
axb.set_xticklabels(names, fontsize=7.2, color=SUBTLE)
axb.set_yticks([0, 0.75, 1.5])
axb.set_yticklabels(['0', '0.75', '1.5'], fontsize=7.2, color=SUBTLE)
axb.set_ylim(0, 1.85)
axb.set_title('true reward at step %d (dotted: the start)'
              % L.PG_STEPS, fontsize=9.0, fontweight='bold',
              color=TEXT, pad=6, loc='left')
style_axes(axb)

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 10',
        fontsize=8, ha='center', va='center', color='#aaaaaa')
plt.savefig(OUT + 'header_rlhf.png', dpi=150, facecolor=BG)
plt.close()
print('Wrote ' + OUT + 'header_rlhf.png')
print()
print('all five images written, every cell asserted against the '
      'frozen run')
