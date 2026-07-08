# Self-Training — Teaching a Model With Its Own Best Guesses

### *Algorithms in Python --- Semi-Supervised Learning, Part 1*

---

This article opens a new track, and the track opens with an
awkward fact about real machine learning: **labels are the
expensive part**. Data is everywhere — a hospital archives
thousands of scans, a company logs millions of documents, a
telescope photographs the sky all night. But *labelled* data
means a radiologist's diagnosis, a lawyer's tag, an astronomer's
classification: expert hours, per example. The result is the
defining situation of applied ML: a **handful of labelled
examples sitting on a mountain of unlabelled ones**.

**Semi-supervised learning (SSL)** is the study of making the
mountain useful. Its central question sounds almost
paradoxical: how can data *without answers* improve a model
that's supposed to predict answers? This track works through
the classic responses — and it starts with the simplest,
oldest, and still most widely used of them all.

**Self-training** is machine learning's version of pulling
yourself up by your bootstraps: train on the few labels you
have, then let the model label the rest of the data *itself* —
keeping only the predictions it is most confident about — and
retrain on its own guesses as if they were ground truth. Done
carefully, the model genuinely teaches itself. Done carelessly,
it confidently teaches itself nonsense. This article builds the
loop from scratch, watches it close most of the gap to a fully
labelled oracle from just **8 labels**, and then — because the
failure mode is the real lesson — watches the same loop destroy
a classifier.

---

## The algorithm: four steps and a threshold

Self-training is not a model; it is a **wrapper** around any
classifier that can report how confident it is:

1. **Train** the base classifier on the labelled set, however
   small.
2. **Predict** every point in the unlabelled pool.
3. **Promote** the confident predictions — probability at least
   some threshold `τ` — into the training set as
   **pseudo-labels**: the model's own guesses, now treated
   exactly like real labels.
4. **Retrain** and repeat, until nothing new clears the bar.

That is the whole algorithm. It dates back to the earliest days
of the field (Yarowsky's 1995 word-sense bootstrapper is the
classic citation), and its modern descendants — pseudo-labelling
in deep learning, FixMatch's confidence-thresholded consistency,
Noisy Student training — power state-of-the-art semi-supervised
image models. The loop never looks inside the classifier; any
model with `fit` and `predict_proba` drops in.

The single load-bearing component is the **confidence
threshold**. Every pseudo-label is a small bet that the model's
guess is right. The threshold decides which bets get placed.

---

## Why it works: confidence spreads like a rumour

Self-training leans on the **cluster assumption**, the quiet
axiom under most of semi-supervised learning: *points in the
same dense cluster tend to share a label*, so the decision
boundary should run through the **low-density gap between
clusters**, not through the middle of one.

If that holds, the geometry does the teaching. The few labelled
points sit inside clusters. The model's most confident
predictions are their immediate neighbours — same cluster,
surely same label. Promote those, and the *next* ring of points
becomes the confident frontier. Round by round, labels spread
outward **along** each cluster, the way a rumour spreads through
a crowd — and they stop at the gap, because points near the
boundary are near both clusters, and confidence dies there.

The unlabelled data never contributes answers. What it
contributes is **shape**: where the dense regions are, and where
they end. Self-training converts shape into labels.

---

## Why it fails: confirmation bias, compounding

Now run the same logic with one wrong bet. A pseudo-label that
is confidently *wrong* enters the training set and is retrained
on as truth. The model becomes more sure of the error — sure
enough to mislabel the point's neighbours, which recruit *their*
neighbours. Wrong labels don't wash out; they **compound**,
exactly like the right ones do. The loop has no way to tell the
two apart, because it grades its own homework.

This is **confirmation bias**, and it is the tax on every
self-training system. The threshold is the guard rail — but as
the experiment below shows, it cannot save you from a deeper
problem: labels that leave whole regions of a cluster closer to
the *wrong* frontier than to the right one.

---

## A worked example: 8 labels and two moons

The companion script stages everything on **two moons** — two
interleaved crescents, one class each, separated by a
low-density gap. It is the classic semi-supervised picture
precisely because it makes the cluster assumption visible.

The base classifier is k-nearest-neighbours (`k = 3`, confidence
= the fraction of neighbours that agree, pseudo-labels require
unanimity). kNN is chosen deliberately: its confidence is
**local by construction** — far from any label, the neighbours
disagree — which is exactly the property the loop needs. More on
that choice below.

```
DEMO 1 --- The setting: 8 labels, 492 unlabelled points
  Trained on the 8 labels only        :  89.8% test accuracy
  Oracle: all 500 true labels (bound) :  99.0% test accuracy

  The gap to close from unlabelled data alone: 9.2%
```

An annotator labelled four points spread along each moon —
eight labels, full stop. Eight labels alone reach 89.8%; five
hundred would reach 99.0%. Self-training's job is to close that
gap using points with *no labels at all*.

### The loop, round by round

```
DEMO 2 --- The loop: the pseudo-label frontier expands
    round   added   cumulative   wrong so far   test accuracy
       1       48         48            0           89.6%
       2      380        428            3           95.8%
       3       54        482           11           97.2%
       4        2        484           11           97.6%
       5        2        486           11           97.6%
       6        2        488           11           97.6%
       7        1        489           11           97.6%

  Final training set: 8 real labels + 489 pseudo-labels (11 wrong, 2.2%)
  Test accuracy: 89.8% (labels only)  ->  97.6% (self-trained)   [oracle 99.0%]
```

Watch the frontier work. The first 48 pseudo-labels arrive with
**zero mistakes** — unanimous neighbourhoods only exist deep
inside a single cluster. Those 48 extend the model's reach, and
in round 2 the frontier sweeps up 380 more points with just 3
errors, pushing accuracy from 89.8% to 95.8%. By round 7 the
loop has labelled 489 of the 492 unlabelled points itself, only
2.2% of them wrongly, and test accuracy stands at **97.6%** —
most of the way from the 8-label baseline to the fully-labelled
oracle, using nothing but the data's shape.

### The catch: the same loop, ruined by placement

Here is the part most write-ups skip. Run the *identical*
algorithm — same threshold, same unanimity rule, same budget of
8 labels — but let the labels fall at **random** instead of
spread along the moons:

```
DEMO 3 --- The catch: the same loop, with badly-placed labels
    draw    labels only    self-trained    wrong pseudo-labels
      0       80.8%          85.4%              67
      1       56.2%          59.6%             221
      2       93.2%          51.6%             227   <- went backwards
      3       77.0%          87.0%              53
      4       82.0%          85.2%              69

    mean    77.8%          73.8%
```

Draw 2 is the horror story: a baseline of 93.2% — *better* than
our spread-label baseline — is dragged down to **51.6%**, a coin
flip, by 227 confidently wrong pseudo-labels. With clumped
labels, whole stretches of a moon sit closer to the *other*
class's frontier than to their own. The first wrong labels land
there, get retrained on as truth, and recruit their neighbours —
confirmation bias, measured in a table. Averaged over the five
draws, self-training made things *worse* (77.8% → 73.8%).

Same algorithm. Same data. The difference between +7.8 points
and catastrophe was **where the 8 labels sat**. Choosing which
points deserve your labelling budget is a whole field of its own
— it closes this track as Part 11, Active Learning.

### An honest note on the base classifier

The first version of this experiment used a small neural network
as the base classifier, and it failed in an instructive way: a
network trained to convergence on 8 points is **confidently
wrong everywhere** — its softmax probabilities saturate near 1.0
across the entire plane, so the threshold admitted almost every
point, errors included, in round one. Self-training is only as
good as the confidence signal you feed it. kNN's
vote-fraction confidence is honest about locality; a deep net's
raw softmax usually is not, which is why deep pseudo-labelling
methods add calibration, augmentation-consistency, or very high
thresholds (FixMatch uses 0.95) before trusting it.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Self-training's costs are your base classifier's costs,
multiplied by the loop.

**Per round**: one base-model fit on the current training set,
plus one scoring pass over the remaining pool — for the kNN base
here, `O(N_pool · N_train · d)` per round; for a parametric
base, one `fit` plus one `predict_proba` sweep.

**Total**: `R` rounds of the above, where `R` is however long
the frontier keeps growing (7 rounds here). The wrapper adds no
model of its own and no memory beyond the pooled dataset,
`O(N · d)`.

**What you get for it**: up to `N_unlabelled` extra training
labels — here, 489 of them from 8 originals, a 60× expansion of
the training set — at zero labelling cost. That multiplier,
against the price of expert annotation, is the entire economic
argument for semi-supervised learning.

---

## When to use self-training

**Reach for it when** labels are scarce, unlabelled data is
plentiful, and the cluster assumption is plausible — your
classes form coherent groups rather than interleaving smears.
It is the first thing to try because it wraps whatever model you
already have.

**Prerequisites worth checking:**

- **A trustworthy confidence signal.** Calibrated probabilities,
  a local measure like neighbour votes, or a deep net tamed by
  a high threshold and consistency tricks.
- **Label coverage.** Your seeds should touch every major region
  of every class — DEMO 3 is what happens when they don't.
- **A high threshold.** Pseudo-label errors compound; be
  stingy. Adding fewer, cleaner labels per round is almost
  always the better trade.

**Skip it when** classes overlap heavily (there is no low-density
boundary for the frontier to respect), or when your model's
confidence is uninformative and can't be fixed.

---

## What comes next

Self-training's weakness is that it grades its own homework —
one model, confirming itself. Part 2, **Co-Training**, is the
classic remedy: train *two* models on two different *views* of
the data (say, the words of a web page and the links pointing at
it), and let each model label examples *for the other*. A
mistake must now fool two independent judges to survive, and the
disagreement between them becomes a built-in error check. It is
the first of several ways this track will turn one model's
self-confirmation into a conversation.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**self_training.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/01-self-training/self_training.py)

Run it with:

```bash
pip install numpy
python self_training.py
```

It needs only `numpy` and runs in seconds. Everything is from
scratch: the two-moons generator, the k-NN base classifier, the
self-training wrapper with its confidence threshold, and the
three experiments — the gap, the expanding frontier, and the
placement catastrophe — with pseudo-label errors audited against
the hidden ground truth. The headline insight worth pinning to
the wall: **self-training turns a model's own confident
predictions into training labels, and the cluster assumption
makes that legitimate — confidence spreads along dense clusters
and stops at the low-density gap, turning 8 labels into 489
(97.6% from an 89.8% baseline, oracle 99.0%); but the loop
grades its own homework, so a bad confidence signal or
badly-placed seeds make errors compound instead — the same code
that gained 8 points dropped a 93% baseline to 51% when the
labels were unluckily placed**.

---

*This is Part 1 of the Semi-Supervised Learning track in the Algorithms in Python series. The companion script `self_training.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). The previous track closed with [Offline Reinforcement Learning](https://medium.com/p/c7203745d39d). Part 2 will look at Co-Training, where two models on two views of the data label examples for each other.*
