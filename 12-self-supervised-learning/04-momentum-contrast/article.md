# Momentum Contrast (MoCo) — A Crowd You Keep in a Ledger

### *Algorithms in Python --- Self-Supervised Learning, Part 4*

---

Part 1 established the economics of the crowd: in contrastive
learning the negatives *are* the signal, they all ride inside
the batch, and the similarity matrix bills you `O(B²)` for
their company. SimCLR's answer was brute force — batches in the
thousands. **MoCo** (He et al., 2020 — *Momentum Contrast*)
asks the question an accountant would: must the crowd actually
*live in the batch*?

Its answer is a queue. Keep the last `K` key embeddings in a
running ledger; each step, score the current queries against
one positive key and the entire ledger, push the new keys in,
pop the oldest out:

```
    query q  vs  [ its key k+ | the last 1,024 banked keys ]
    push this step's keys; pop the oldest    (a FIFO queue)
```

A batch of eight can face a crowd of a thousand. But the naive
version of this idea fails completely, and the reason is the
actual invention. The ledger spans thousands of steps, so its
entries were written by *old versions of the encoder* — and if
the encoder drifts quickly, today's queries are being compared
against coordinates from yesterday's map. MoCo's fix is Part
3's trick pointed backwards: encode the keys with a slow **EMA
twin** of the query encoder. DINO's teacher made the *targets*
stable; MoCo's key encoder makes the *ledger* stable.
Gradients flow through the query side only; the twin is one
weight-lerp per step; the audit at startup checks the hand
backprop through the queue softmax (4.02e-10 against central
differences).

The stage is shape world one last time — and, closing the
section, the companion script retrains all four of its methods
live for one final family table.

---

## A worked example: the ledger, the law, the audit, the family

### A crowd that does not live in the batch

```
    MoCo, batch 32, queue 1024 :  96.0% /  99.6%
    MoCo, batch  8, queue 1024 :  96.0% /  99.7%
```

(Part 1's NT-Xent at batch 256 scored 95.1% / 99.7%.) A batch
of *eight* walks through training with a crowd of a thousand at
its back, and matches the batch-256 baseline on both probes.
The crowd size and the batch size have been fully decoupled —
the whole point of the design — at the cost of a queue and a
second set of weights that no gradient ever touches.

### The consistency law

Same budget everywhere; the only knob is the key encoder's
momentum:

```
    key momentum    probe15   probe-all
       0.000        31.8%     32.9%
       0.900        49.1%     51.4%
       0.990        75.1%     89.7%
       0.999        84.4%     95.4%
```

Momentum 0 — keys written by the live network — is **chance**.
Not weak: dead. This is the measurement that explains why MoCo
is a paper and not a footnote. The queue itself is a data
structure Section 00 covered in an afternoon; a queue whose
thousand entries remain *mutually comparable* while the network
that wrote them keeps moving — that requires the slowest twin
in this section, and the dose–response is monotone all the way
up. Consistency, not capacity, is what the momentum buys. (The
MoCo paper's own ablation found exactly this cliff at scale;
here it reproduces in 256 lines of NumPy.)

### The honest audit: does the queue pay here?

```
    queue     probe15   probe-all
       16       85.6%     95.3%
      256       85.5%     96.0%
     4096       85.7%     95.8%
```

Flat. Sixteen negatives do what four thousand do, because three
shape classes saturate a crowd early. And the control
experiment seals it — plain NT-Xent at batch 8, fourteen
in-batch negatives, no queue, no twin, given the same long
budget:

```
    NT-Xent, batch 8, 32k steps:  95.9% /  99.9%
```

It matches MoCo. On shape world the queue buys *nothing* — and
that is the finding, stated honestly rather than staged: MoCo
solves a **scale** problem. Its machinery pays off where useful
crowds number in the tens of thousands and no device holds the
batch — at ImageNet resolution with ResNet encoders, which is
where its results were earned. This is the third time this
section has caught famous machinery not transferring to a small
stage (MAE's 75% mask ratio, reparameterisation's necessity,
now the queue), and the lesson compounds: **measure before you
inherit machinery.**

### The family, closed

All four methods of the section, canonical configurations from
their own articles, retrained live on the same 4,000 unlabelled
images:

```
                          probe15   probe-all   yardstick
    SimCLR / NT-Xent (P1)   95.1%     99.7%    the batch beside you
    MAE (P2)                49.2%     67.6%    the data itself
    DINO (P3)               94.5%     99.9%    a slow twin's targets
    MoCo (P4)               96.0%     99.6%    a ledger, kept by a twin
```

Every self-supervised method in this section is one answer to
the same question: when two views of the world should agree,
*where does the yardstick for "agree" come from?* SimCLR buys a
fresh crowd every batch. MoCo banks the crowd and keeps the
bank honest with a slow twin. DINO drops the crowd entirely and
chases the twin itself. MAE never compares to anyone — the
hidden three quarters of the image is yardstick enough (and its
low probe row is not a defect but a different contract: Part 2
measured it winning after fine-tuning). Four pretexts, one
16×16 stage, not a label in sight.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

MoCo's ledger line replaces SimCLR's quadratic one: scoring
against the queue costs `O(B · K · d)` — *linear* in batch,
with the crowd size `K` a free dial rather than a memory
hostage. The key encoder adds one gradient-free forward pass
and an `O(P)` weight-lerp; the queue itself is `O(K · d)`
floats of memory — a few megabytes where SimCLR's crowd wanted
a data-parallel cluster. That arithmetic is why MoCo ran on
one GPU in 2020 while SimCLR wanted a TPU pod, and why the
queue idea survived into every retrieval system that banks
embeddings. The economics of this whole section then land in
one contrast: pretraining is the expensive, label-free
purchase — by crowd, ledger, twin or blank — and every
downstream task thereafter is a probe or a light fine-tune.

---

## Where the ledgers went

MoCo v2 absorbed SimCLR's tricks (projection head, stronger
augmentations); v3 dropped the queue for ViT encoders — by
then, batches had caught up with crowds — but kept the momentum
twin, which proved the more durable half of the invention:
today EMA twins stabilise DINO and its descendants, data2vec,
semi-supervised teachers and RL targets alike. The ledger half
lives on elsewhere: banked-embedding negatives are standard in
retrieval and recommendation training (Section 07's two-tower
models mine hard negatives from exactly such stores), and
cross-batch memory is routine in metric learning. And the
section's four answers now blend freely at the frontier —
DINOv2 trains agreement *and* masking; contrastive towers sit
beside masked backbones in every foundation-model stack. The
question each method answered — where does the yardstick come
from? — has stopped being a rivalry and become a menu.

---

## What comes next

Section 12 is complete. Section 13 takes the self-supervision
story to its most consequential habitat — **Natural Language
Processing** — beginning with Part 1, Word Embeddings
(Word2Vec, GloVe, fastText): the discovery that a humble
pretext on raw text ("predict the word next door") packs
meaning into vectors so well that arithmetic on them does
analogy. It is this section's ideas, five years earlier and
one modality over — and it is where the road to modern
language models begins.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**moco.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/12-self-supervised-learning/04-momentum-contrast/moco.py)

Run it with:

```bash
pip install numpy
python moco.py
```

It needs only `numpy` and runs in about five minutes.
Everything is from scratch: the queue, the EMA key encoder,
the InfoNCE-against-the-ledger loss with hand-derived audited
gradients, and live retrains of SimCLR, MAE and DINO for the
family table. The headline insight worth pinning to the wall:
**the crowd never needed to live in the batch — a batch of
eight with a thousand-deep ledger matches batch-256 SimCLR
(96.0% vs 95.1%) — but a ledger is only worth keeping if its
entries stay comparable, and that is the actual invention:
keys from the live encoder score at chance (32.9%) while a
0.999-momentum twin restores 95.4%, a monotone consistency law;
on this small stage the honest audit shows the queue buying
nothing at all (16 negatives = 4,096; plain batch-8 NT-Xent
matches MoCo) because MoCo solves a scale problem — and the
section closes with its four methods on one table, each a
different answer to where the yardstick for agreement comes
from: the batch, the ledger, the twin, or the data itself**.

---

*This is Part 4 of the Self-Supervised Learning track in the Algorithms in Python series, closing the section begun with [contrastive learning](https://medium.com/p/37f7feeb811e), [masked autoencoding](https://medium.com/p/3dd3a0792184) and DINO — all three retrained live inside the companion script `moco.py` in the [series repository](https://github.com/grahamroy/algorithms-in-python). Section 13 opens with Word Embeddings.*
