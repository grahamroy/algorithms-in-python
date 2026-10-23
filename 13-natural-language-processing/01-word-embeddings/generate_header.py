"""Generate the header image for the word-embeddings article.

Right panels are REAL: PCA of the actual SGNS embedding with the
gender-pair arrows, and the PMI-vs-dot-product receipt scatter --
retrained live with the frozen seeds and asserted.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

import word_embeddings as WE

BG_COLOR = '#FFFFFF'
TEXT_COLOR = '#1F2937'
SUBTLE_TEXT = '#6B7280'
ORANGE = '#EA580C'
BLUE = '#2563EB'
GREEN = '#16A34A'
PURPLE = '#7C3AED'
RED = '#DC2626'
BADGE_BG = '#F8FAFC'
TOPIC_COLS = {"royal": PURPLE, "farm": ORANGE, "sea": BLUE}

# ---- retrain with the frozen seeds ---------------------------------------
rng = np.random.default_rng(0)
vocab = WE.build_vocab()
idx = {w: i for i, w in enumerate(vocab)}
sents = WE.gen_corpus(WE.N_SENT, rng)
cs, os_ = WE.extract_pairs(sents, idx, np.random.default_rng(1))
cnt = np.bincount(np.concatenate([cs, os_]),
                  minlength=len(vocab)).astype(float)
p_neg = cnt ** 0.75
p_neg /= p_neg.sum()
m = WE.SGNS(len(vocab), seed=0)
m.train(cs, os_, np.random.default_rng(2), p_neg)

g, p = WE.analogy_scores(m.Win, vocab, idx)
print(f'gender {g:.1%} plural {p:.1%}')
assert f'{g:.1%}' == '100.0%' and f'{p:.1%}' == '87.5%'

V = len(vocab)
C = np.zeros((V, V))
np.add.at(C, (cs, os_), 1.0)
tot = C.sum()
pw_c = C.sum(1) / tot
pw_o = C.sum(0) / tot
ii, jj = np.where(C > 3)
pmi = np.log((C[ii, jj] / tot) / (pw_c[ii] * pw_o[jj] + 1e-12) + 1e-12)
dots = np.einsum('ij,ij->i', m.Win[ii], m.Wout[jj])
r = np.corrcoef(pmi, dots)[0, 1]
A = np.vstack([pmi, np.ones_like(pmi)]).T
slope, intercept = np.linalg.lstsq(A, dots, rcond=None)[0]
print(f'r {r:.2f} slope {slope:.2f} intercept {intercept:.2f}')
assert f'{r:.2f}' == '0.90' and f'{intercept:.2f}' == '-1.76'

# noun map: x = first principal component (topic axis),
# y = the learned gender direction (mean female - mean male)
nouns = [n for t in WE.TOPICS.values() for n in t["nouns"]]
noun_topic = {}
for tname, t in WE.TOPICS.items():
    for n in t["nouns"]:
        noun_topic[n] = tname
Zall = WE.norm_rows(m.Win)
Zn = Zall[[idx[n] for n in nouns]]
gdir = (np.mean([Zall[idx[f]] for f in WE.GENDER_F], 0)
        - np.mean([Zall[idx[mm]] for mm in WE.GENDER_M], 0))
gdir /= np.linalg.norm(gdir)
Zc = Zn - Zn.mean(0)
Zres = Zc - np.outer(Zc @ gdir, gdir)
U, S, Vt = np.linalg.svd(Zres, full_matrices=False)
P2 = np.stack([Zc @ Vt[0], Zc @ gdir], 1)

# ---- figure --------------------------------------------------------------
fig = plt.figure(figsize=(1600/150, 900/150), dpi=150)
fig.patch.set_facecolor(BG_COLOR)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.52,
        'Word Embeddings: You Shall Know a Word by the Company It '
        'Keeps',
        fontsize=16.5, fontweight='bold', ha='center', va='center',
        color=TEXT_COLOR, fontfamily='sans-serif')
ax.text(8, 8.05,
        'Predict the word next door and geometry falls out \u2014 '
        'measured on a micro-language where every famous claim has a '
        'ground truth.',
        fontsize=10.2, fontstyle='italic', ha='center', va='center',
        color=SUBTLE_TEXT, fontfamily='sans-serif')

# ===================== LEFT =====================
def box(x, y0, w, h, edge, face='white', lw=1.9):
    ax.add_patch(FancyBboxPatch((x, y0), w, h,
                 boxstyle='round,pad=0.04,rounding_size=0.1',
                 facecolor=face, edgecolor=edge, linewidth=lw))

box(0.7, 6.1, 5.3, 1.55, BLUE)
ax.text(3.35, 7.38, 'the pretext', fontsize=9.8, fontweight='bold',
        ha='center', va='center', color=BLUE)
ax.text(3.35, 6.98, 'score the real neighbour high,\n'
                    'five sampled impostors low  (SGNS)',
        fontsize=8.6, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')
ax.text(3.35, 6.5, "Section 12's self-supervision, five years\n"
                   'earlier and one modality over',
        fontsize=8.2, ha='center', va='center', color=SUBTLE_TEXT)

box(0.7, 4.35, 5.3, 1.45, PURPLE, face=BADGE_BG)
ax.text(3.35, 5.55, 'the parallelogram, scored', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=PURPLE)
ax.text(3.35, 4.98, 'king:queen :: bull:?  ->  cow\n'
                    'gender analogies 100.0%\n'
                    'plural analogies  87.5%',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR,
        fontfamily='DejaVu Sans Mono')

box(0.7, 2.75, 5.3, 1.25, RED, face=BADGE_BG, lw=1.7)
ax.text(3.35, 3.75, 'the theory receipt', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=RED)
ax.text(3.35, 3.25, 'SGNS secretly factorises shifted PMI:\n'
                    'r = 0.90, intercept -1.76 (theory -1.61)',
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

box(0.7, 1.15, 5.3, 1.25, GREEN, lw=1.6)
ax.text(3.35, 2.15, 'the subword insurance', fontsize=9.6,
        fontweight='bold', ha='center', va='center', color=GREEN)
ax.text(3.35, 1.65, "fastText prices words the corpus never held:\n"
                    "'princesses' -> princess, princes, queens",
        fontsize=8.2, ha='center', va='center', color=TEXT_COLOR)

# ===================== RIGHT TOP: the map =====================
axg = fig.add_axes([0.455, 0.475, 0.30, 0.335])
for i, n in enumerate(nouns):
    axg.scatter(P2[i, 0], P2[i, 1], s=14,
                color=TOPIC_COLS[noun_topic[n]], alpha=0.8,
                linewidths=0)
lab_words = ["king", "queen", "bull", "cow", "ship", "castle"]
for n in lab_words:
    i = nouns.index(n)
    axg.annotate(n, (P2[i, 0], P2[i, 1]), fontsize=6.8,
                 color=TEXT_COLOR, xytext=(2, 2),
                 textcoords='offset points')
for (a, b) in WE.GENDER_PAIRS:
    ia, ib = nouns.index(a), nouns.index(b)
    axg.annotate('', xy=P2[ib], xytext=P2[ia],
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.2,
                                 alpha=0.8))
axg.set_xticks([]); axg.set_yticks([])
axg.set_xlabel('principal component (topic)', fontsize=7.6,
               color=SUBTLE_TEXT)
axg.set_ylabel('learned gender direction', fontsize=7.6,
               color=SUBTLE_TEXT)
for sp in axg.spines.values():
    sp.set_color('#CBD5E1')
axg.set_title('the learned map: nouns by topic, gender as arrows',
              fontsize=9.0, fontweight='bold', color=TEXT_COLOR, pad=5)
ax.text(12.35, 6.0, 'noun vectors: x = topic\naxis, y = the gender\n'
        'direction the model\nlearned. red arrows =\nmale -> female '
        'pairs.\n\nall five arrows climb the\nsame axis: one gender\n'
        'direction, shared across\nroyalty and livestock --\nlearned '
        'from his/her\nagreement alone.',
        fontsize=8.4, ha='left', va='center', color=SUBTLE_TEXT)

# ============ RIGHT BOTTOM LEFT: the receipt ============
axr = fig.add_axes([0.455, 0.095, 0.245, 0.30])
sub = np.random.default_rng(9).choice(len(pmi), 400, replace=False)
axr.scatter(pmi[sub], dots[sub], s=5, color=BLUE, alpha=0.4,
            linewidths=0)
xs = np.array([pmi.min(), pmi.max()])
axr.plot(xs, slope * xs + intercept, color=RED, linewidth=1.6)
axr.set_xlabel('PMI(word, context)', fontsize=7.8, color=SUBTLE_TEXT)
axr.set_ylabel('win . wout', fontsize=7.8, color=SUBTLE_TEXT)
for sp in ('top', 'right'):
    axr.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    axr.spines[sp].set_color('#CBD5E1')
axr.tick_params(colors=SUBTLE_TEXT, labelsize=7)
axr.set_title(f'the receipt: r = {r:.2f}', fontsize=9.0,
              fontweight='bold', color=TEXT_COLOR, pad=5)

# ============ RIGHT BOTTOM RIGHT: the language + OOV ============
axt = fig.add_axes([0.725, 0.095, 0.235, 0.30])
axt.axis('off')
axt.text(0.02, 0.97,
         'the micro-language:\n'
         '  the noble king rules\n'
         '    his throne\n'
         '  the cows graze their\n'
         '    barn\n\n'
         'never in the corpus:\n'
         '  princesses ->\n'
         '    princess, princes\n'
         '  compasses ->\n'
         '    compass, island',
         fontsize=7.6, va='top', ha='left', color=TEXT_COLOR,
         fontfamily='DejaVu Sans Mono',
         transform=axt.transAxes)
for sp in axt.spines.values():
    sp.set_color('#CBD5E1')
axt.set_title('the stage, and the insurance', fontsize=9.0,
              fontweight='bold', color=TEXT_COLOR, pad=5)

ax.text(8, 0.18,
        'Algorithms in Python  |  Natural Language Processing Part 1',
        fontsize=8, ha='center', va='center', color='#aaaaaa')

out = ('D:/Projects/Medium/algorithms-in-python/'
       '13-natural-language-processing/01-word-embeddings/'
       'header_word_embeddings.png')
plt.savefig(out, dpi=150, facecolor=BG_COLOR)
plt.close()
print(f'Saved to {out}')
