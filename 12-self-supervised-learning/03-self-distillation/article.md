# Self-Distillation (DINO) — Chasing Your Own Slower Shadow

### *Algorithms in Python --- Self-Supervised Learning, Part 3*

---

Part 1 ended on a measured cliffhanger. Keep the pull — make two
augmented views of the same image agree — and delete the crowd
of negatives pushing back, and the embedding collapses 158-fold:
every image mapped to the same point, loss perfectly minimised,
representation worthless. The obvious conclusion was that the
crowd is what holds the space open.

**DINO** (Caron et al., 2021 — *self-DIstillation with NO
labels*) pulls with no negatives anyway, and does not fall. Two
networks, same architecture. The **student** is trained; the
**teacher** is not trained at all — it is an exponential moving
average of the student's own weights. The student's own past,
distilled, made into the target it chases:

```
    teacher sees view A  ->  soft targets (sharpened, centred)
    student sees view B  ->  cross-entropy toward the targets
    (and symmetrically, B -> A;  teacher <- EMA(student))
```

Both networks end in a small head over K "prototypes" — learned
soft categories, no labels anywhere. What keeps the tightrope
walkable is a pair of opposing tweaks to the teacher's softmax.
**Centring** subtracts a running mean of teacher logits, killing
the everything-onto-one-prototype solution. **Sharpening** gives
the teacher a lower softmax temperature than the student,
killing the everything-uniform solution. A running mean and a
temperature — that is the entire safety apparatus.

The stage is shape world, and the companion script retrains Part
1's contrastive encoder *and* Part 1's pull-only collapse live,
so all three sit on one measured table. The student's backward
pass — through both softmaxes, teacher and centre held fixed —
is hand-derived and audited at startup (8.83e-06 against central
differences).

---

## A worked example: the walk, the tightrope, the pace

### Agreement without the crowd

```
     step    teacher probe   student probe
      100        54.7%           59.2%
      500        83.3%           90.1%
     1000        98.7%           99.0%
     3000        99.9%           99.9%

                            probe15   probe-all   spread
    pull-only (Part 1)      34.0%     34.0%     0.0011
    DINO (this article)     94.5%     99.9%     0.1524
    contrastive (Part 1)    95.1%     99.7%     0.1737
```

Read the bottom table's rows in order. The pull-only row is Part
1's collapse, reproduced live: probes at chance, embedding
spread three orders of magnitude below healthy. DINO is *the
same pull* — two views, agree, no negatives anywhere in the loss
— and it matches the full NT-Xent contrastive encoder on every
column: 94.5% on fifteen labels against 95.1%, 99.9% with all
labels against 99.7%, spread fully held open. The crowd of
negatives, whose absence looked fatal in Part 1, turned out to
be optional. What is *not* optional is the next demo.

### The tightrope: centring × sharpening

Two failure modes wait, one on each side. `H(per)` is the mean
per-sample entropy of the teacher's predictions; `H(mean)` is
the entropy of the average prediction; `log K = 3.47`:

```
    recipe             probe-all   H(per)   H(mean)   verdict
    centre + sharpen     99.9%      0.14     2.09    healthy
    centre only          54.7%      3.47     3.47    uniform collapse
    sharpen only         69.8%      0.00     0.51    one-hot collapse
    neither              52.0%      3.15     3.15    uniform collapse
```

Centring alone: per-sample entropy pinned at exactly `log K` —
the teacher says "anything is fine" and the student agrees to
say nothing; the safest possible agreement is total vagueness.
Sharpening alone: entropy 0.00 with the batch's mass crowded
onto one prototype — perfectly confident, perfectly empty; the
safest possible agreement is one answer for everything. The two
tweaks fail in *opposite directions*, which is exactly why the
pair works: together they pin the only useful operating point —
confident about each image (0.14 nats), diverse across images
(2.09 nats). This is the DINO paper's central ablation,
reproduced here down to the entropies, and it is the honest
answer to "where did the negatives' job go?": into a running
mean and a temperature.

### The teacher's pace

The teacher is the student's own past, averaged with momentum
`m`. How much past is the right amount?

```
    momentum    probe-all    spread
     0.000       67.3%     0.1107
     0.900       99.9%     0.1520
     0.990       99.9%     0.1524
     0.999       99.1%     0.1595
```

Momentum zero — the teacher *is* the student — and the chase
degenerates: a target that moves with every step teaches
little. Momentum 0.9 to 0.99 works; 0.999 remembers too much
for a 3,000-step budget. And one piece of folklore, checked
honestly: the DINO paper reports that the EMA teacher
*outperforms* its student during training. Here the teacher
**trails** early — 54.7% against the student's 59.2% at step
100 — and converges to a tie. At this scale the teacher's job
is not to be better; it is to be **stable**: slow enough to be
chased, which is precisely what the momentum-zero row loses.
That is the deepest reading of the whole negative-free family —
BYOL's predictor head, SimSiam's stop-gradient, DINO's EMA
teacher differ in decoration, but each one is an asymmetry that
keeps the target out of lockstep with the learner. Break the
asymmetry and you are back at Part 1's single point.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

DINO's bill is Part 2's, not Part 1's: two encoder passes per
view but **no `O(B²)` similarity matrix** — the quadratic crowd
line is simply gone, which is why negative-free methods scale
to batch sizes and devices the contrastive recipe struggled
with. The teacher costs one extra forward pass and one EMA
update (`O(P)` — cheaper than a gradient step, since nothing is
backpropagated through it), and centring is a running mean,
`O(B · K)`. The head's prototypes add a `d × K` matrix. As with
Part 1, adaptation downstream is a linear probe over frozen
features — and at real scale it is DINO's descendants, not the
contrastive originals, whose frozen features became the default:
the probe column of DEMO 1 at four thousand times the size.

---

## Where the shadows went

DINO's headline at scale was not the probe numbers — it was
what appeared unasked: the attention maps of a DINO-trained
vision transformer segment objects from their backgrounds with
no supervision whatsoever, sharp enough to use as masks. The
recipe's descendants took over the field. DINOv2 added Part 2's
masked prediction to the self-distillation loss — this
section's two philosophies, agreement and reconstruction, in
one objective — and its frozen features now sit under
depth-estimation, segmentation and retrieval systems as the
closest thing vision has to a universal backbone. BYOL and
SimSiam mapped the rest of the negative-free territory (a
predictor head and a stop-gradient are asymmetries enough);
EMA teachers now stabilise semi-supervised learning, masked
distillation (data2vec across speech, vision and text), and
reinforcement-learning targets — Section 8's DQN target
network was this same idea wearing older clothes. Wherever a
model must learn from its own outputs without spiralling, a
slower copy of itself is the standing answer.

---

## What comes next

Part 4, **Momentum Contrast (MoCo)**, closes the section by
going back to Part 1's crowd and asking the economic question:
if negatives are the signal, must they all fit in one batch?
MoCo's queue of past embeddings — served by the same EMA
encoder this article just built — bought SimCLR-sized crowds at
a fraction of the memory, and completes the section's family
picture: every method is a different answer to *where the
second view's yardstick comes from*.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**dino.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/12-self-supervised-learning/03-self-distillation/dino.py)

Run it with:

```bash
pip install numpy
python dino.py
```

It needs only `numpy` and runs in about seven minutes.
Everything is from scratch: student, EMA teacher, prototype
head, centring and sharpening, the symmetric distillation loss
with hand-derived audited gradients, live retrains of Part 1's
contrastive and pull-only baselines, and the entropy
instruments for the collapse taxonomy. The headline insight
worth pinning to the wall: **the crowd of negatives was never
the point — a slow-moving copy of yourself is yardstick
enough: DINO matches the contrastive encoder (94.5%/99.9% vs
95.1%/99.7%) with zero negatives while the naive version of
the same pull collapses to chance at spread 0.0011; the
tightrope is held by a running mean and a temperature, whose
solo failures are opposite and exact (centring alone: entropy
pinned at log K, uniform collapse; sharpening alone: entropy
0.00, one-hot collapse; together: 0.14 within, 2.09 across);
and the teacher earns its keep not by being better — it
trails its student early, 54.7% to 59.2% — but by being
stable, the asymmetry every working negative-free method
shares**.

---

*This is Part 3 of the Self-Supervised Learning track in the Algorithms in Python series. The companion script `dino.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It answers the collapse measured in [Part 1](https://medium.com/p/37f7feeb811e) and shares its stage with [Part 2](https://medium.com/p/3dd3a0792184)'s masked autoencoder; the same slow-copy trick stabilised DQN's target networks in Section 8 long before it had this name. Part 4 will close the section with Momentum Contrast (MoCo).*
