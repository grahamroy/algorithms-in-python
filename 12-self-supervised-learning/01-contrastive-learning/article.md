# Contrastive Learning — The Augmentation Is the Supervision

### *Algorithms in Python --- Self-Supervised Learning, Part 1*

---

Section 11 built architectures that classify and architectures
that create, and every one of them sat at one of two extremes:
either it needed labels for everything, or it needed labels for
nothing because it never learned what things *were*. This
section is about the third way — models that manufacture their
own supervision out of raw, unlabelled data — and it matters
because that is where nearly all the world's data lives.
Labels are expensive; pixels are free.

**Contrastive learning** (SimCLR: Chen et al., 2020) makes the
move with a disarmingly simple rule: two *augmented views* of
the same example should agree, and views of different examples
should not. Take an image, distort it twice — shift it, jitter
its brightness, add noise — push both copies through the same
encoder onto the unit sphere, and train with a softmax whose
"correct class" is: *which of these 2N views is your sibling?*

```
L_i = -log  exp(z_i · z_j / tau) / sum_k  exp(z_i · z_k / tau)
```

That is the whole loss (NT-Xent). No labels, no reconstruction,
no generator — just agreement against a crowd. The deep content
is not the formula; it is the **augmentation menu**. Declaring
"a shifted, brightened, noised copy is still the same thing" is
a claim about what should *not matter*, and the encoder learns
to be invariant to exactly what the menu varies and to keep
exactly what it preserves. The augmentations are the
supervision.

The stage is new for this section — **shape world**: 16×16
images of three shapes (a cross, a hollow square, a diagonal X)
at random position, size and brightness, plus pixel noise.
Augmentations: random shift, brightness jitter, fresh noise.
The companion script hand-derives every gradient — through the
L2 normalisation and both InfoNCE losses — and audits them at
startup (NT-Xent 5.65e-10, the CLIP two-tower loss 4.08e-10
against central differences).

---

## A worked example: the promise, the menu, the crowd, the shot

### The promise: fifteen labels

Pretrain on 4,000 *unlabelled* shape images. Then freeze the
encoder and train a linear probe — one softmax layer — on five
labels per class. Against it: the same encoder architecture
trained supervised, with the same augmentations, on the same
fifteen labels; and the ceiling, supervised on all 4,000:

```
    contrastive + probe, 15 labels :  95.1%
    contrastive + probe, all labels:  99.7%
    supervised, 15 labels          :  84.9%
    supervised, all 4,000 labels   :  99.7%
```

(Robustness: over five different draws of the fifteen labels
the probe averages 94.4%, range 89.9%–98.0%.)

Fifteen labels plus four thousand unlabelled images land within
five points of four thousand labels; the same fifteen labels
alone stop ten points lower still. The unlabelled pile did the
work. That is the whole economic argument for this section in
four rows — and, scaled up nine orders of magnitude, the
training recipe behind every modern foundation model's encoder.

### The menu is the supervision

Withdraw one item from the augmentation menu and watch what the
encoder decides to care about. The class probe reads augmented
views; the nearest-neighbour checks read clean, unshifted views
where each image's true position is known:

```
    menu          class probe   NN same class   NN same quadrant
    full menu       99.7%        100.0%           80.2%
    no shift        51.7%         98.0%           97.1%
    noise only      56.0%         98.2%           96.8%
```

Chance is 33% for class, ~25% for quadrant. Remove the shift
augmentation and the class probe halves — but look at the
neighbour columns before concluding the encoder learned
nothing. Nearest-neighbour class purity is still 98%: instances
are still beautifully separated. The encoder has simply filed
images by *where they are* as much as by *what they are* —
neighbours now share a position quadrant 97% of the time. The
information never disappeared; the **invariance** did, and with
it the linear readability of class. Augmentation does not add
signal. It declares which differences are noise — and the
encoder believes the declaration exactly. This is why
augmentation design, not architecture, is where SimCLR's
authors spent their ablation budget, and why "what should your
model ignore?" is the first question of any self-supervised
pipeline.

### Take away the crowd and the space collapses

The NT-Xent denominator is a crowd of negatives: every other
image in the batch pushes back. Keep the pull — make positive
views agree — and delete the push, with the same budget:

```
    embedding spread, pull-only  : 0.0011
    embedding spread, contrastive: 0.1737
```

A 158× collapse. With nothing to push against, the perfect
solution is to map *every* image to the same point: loss zero,
representation worthless. The crowd is what holds the space
open — which is why SimCLR wants enormous batches, and why an
entire literature (BYOL, SimSiam, DINO — Part 3 of this
section) exists to dodge this collapse *without* paying for
the crowd.

### Mini-CLIP: two towers and a zero shot

CLIP (Radford et al., 2021) is the same contrastive rule with
one twist: the two views come from different *modalities*. Give
every image a seven-number "caption" (class one-hot, position,
size, brightness), train a second encoder on captions, and
align the towers with the symmetric InfoNCE — images classify
captions, captions classify images. Then evaluate things the
model was never trained to do:

```
    zero-shot classification, neutral prompts :  95.1%
    zero-shot classification, corner prompts  :  68.0%
    caption->image retrieval among 1,000      : top-1 5.1%,
                       top-10 39.6%, median rank 15
```

No classifier was ever trained. The "classes" are three
sentences shown to the caption tower at test time, and the
image tower agrees with the right one 95% of the time — that
is zero-shot classification, the mechanism that let CLIP match
supervised ResNets without touching ImageNet's labels. Write
the sentences badly — claim the shape sits in a corner — and
27 points vanish: prompt engineering is not a folk ritual, it
is measurable geometry in a shared embedding space. And the
same space does retrieval for free: the right image out of a
thousand lands at median rank 15. One loss, three products —
the CLIP economy.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The pretraining bill has a distinctive quadratic line: the
similarity matrix costs `O(B² · d)` per step and every entry
matters, because the negatives *are* the signal — SimCLR's
famous appetite for batch sizes in the thousands is this line,
not the encoder's. The encoder itself is ordinary Section 11
economics, and no labels appear anywhere in the bill. The
payoff shows up downstream: adapting to a new task is a linear
probe over frozen features — a softmax, seconds — and CLIP's
zero-shot classification costs one dot product per candidate
caption. Pretrain once, expensively; harvest cheaply, forever.
That amortisation, more than any single number, is why
self-supervision took over.

---

## Where the agreements went

SimCLR's recipe — two views, one encoder, a crowd of negatives
— set the template, and its descendants each attacked one line
of the bill. MoCo (Part 4) replaced the giant batch with a
memory queue and a momentum encoder; BYOL and SimSiam threw
away the negatives entirely and survived the collapse this
article measured; DINO (Part 3) did it with self-distillation
and got segmentation maps nobody asked for. CLIP scaled the
two-tower version to 400 million image–text pairs and made
"zero-shot" a product surface: classification by prompt,
retrieval by embedding — and its image tower, frozen, became
the eyes of a generation of multimodal systems, including the
text-to-image models of Section 11's Part 9 lineage. Beyond
vision, the same InfoNCE trains sentence embeddings, speech
models and recommender towers (Section 07's two-tower
retrieval is this loss wearing industrial clothes). Wherever
two things ought to agree — two crops, two languages, a query
and its document — the contrastive rule turns that agreement
into an encoder.

---

## What comes next

Part 2, **Masked Autoencoding**, builds supervision the other
way round: instead of declaring what should not matter, hide
part of the input and demand the model predict it — BERT's
recipe for text, MAE's for images. Same free data, opposite
philosophy: contrastive learning learns by *ignoring*, masked
modelling learns by *reconstructing* — and shape world will
host both.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**contrastive.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/12-self-supervised-learning/01-contrastive-learning/contrastive.py)

Run it with:

```bash
pip install numpy
python contrastive.py
```

It needs only `numpy` and runs in about five minutes.
Everything is from scratch: shape world and its augmentation
menu, the encoder with L2-normalised embeddings, NT-Xent and
the CLIP two-tower loss with hand-derived, audited gradients,
linear probes, nearest-neighbour purity, and the zero-shot
evaluation. The headline insight worth pinning to the wall:
**contrastive learning manufactures supervision out of a
statement about what should not matter — and every measurement
follows from taking that statement seriously: 4,000 unlabelled
images plus 15 labels land within five points of 4,000 labels
(95.1% vs 99.7%) while 15 labels alone manage 84.9%; delete
the shift augmentation and class readability halves (99.7% →
51.7%) even though nearest-neighbour purity stays at 98% —
the information survives, the invariance dies; delete the
negatives and the embedding collapses 158-fold to a single
point; and the same rule across two towers buys zero-shot
classification at 95.1%, loses 27 points to badly-worded
prompts, and retrieves the right image from a thousand at
median rank 15 — one agreement rule, an encoder economy**.

---

*This is Part 1 of the Self-Supervised Learning track in the Algorithms in Python series. The companion script `contrastive.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). The few-labels economics continues Section 10's semi-supervised story ([Self-Training](https://medium.com/p/eeca5accd031)), and the encoders it pretrains are Section 11's ([MLP](https://medium.com/p/9c0d36baa5bc), [CNN](https://medium.com/p/6f860b54044f)). Part 2 will look at Masked Autoencoding.*
