# Tri-Training — Two Teachers, One Student, One Gate

### *Algorithms in Python --- Semi-Supervised Learning, Part 7*

---

Co-training (Part 2) earned this track's best safety mechanism —
two judges with different evidence, vetoing each other's
mistakes — and paid for it with a steep entry requirement: your
data must split into two views, each sufficient to classify
alone. Web pages with their inbound links qualify. Most datasets
don't.

**Tri-training** (Zhou & Li, 2005) recovers the spirit of that
check with no views at all, by changing where the diversity
comes from. Instead of two models on two feature sets, train
**three models on three bootstrap resamples** of the labelled
data — an unstable base learner turns small resampling
differences into genuinely different classifiers. Then arrange
them into a rotating classroom: for each classifier, the *other
two* act as its teachers, and every unlabelled point the
teachers **agree** on becomes a lesson for the student.

But agreement alone is not enough — that was the hard lesson of
Part 1, and this article measures it again. Tri-training's real
contribution is **the gate**: a measurable condition, checked
every round, that only lets a batch of pseudo-labels in if the
teachers' estimated error keeps a classical noise bound
shrinking. Agreement proposes; the gate disposes. The companion
script builds all of it, revisits Part 1's five catastrophic
label draws for a third time — and is honest about the one draw
where even three judges get it wrong.

---

## The algorithm

**Diversity from bootstraps.** Each of the three classifiers
starts from a bootstrap resample of the labelled set (draw `n`
examples *with replacement*). A resample of 8 points omits
roughly a third of them, so the three training sets — and, with
an unstable learner like 1-NN, the three classifiers — genuinely
differ. This is bagging's trick, redeployed: instability, which
supervised learning fights, is here the *fuel* that makes
"agreement between models" mean something.

**The teaching rule.** In each round, for classifier `i`, let
its two peers label the whole unlabelled pool. The points they
agree on form the candidate batch `L_i`, labelled with the
peers' shared prediction. A wrong lesson now requires *both*
teachers to make the same mistake.

**The gate.** Before the student accepts the batch:

1. Measure the teachers' error `e_i` — on the labelled set,
   restricted to points where they agree (the only place truth
   is known).
2. Accept only if `e_i` fell since last round **and** the
   product `e_i · |L_i|` — the expected number of wrong labels
   ingested — shrinks relative to the last accepted round. If
   the batch is too big for that, **subsample** it down until
   the bound holds.

That product rule comes straight from classical learning theory
(the Angluin–Laird noise bound): training on noisy labels is
survivable if the noise mass shrinks as the sample grows. The
gate turns that theorem into an admission policy, using only
quantities you can compute.

**Prediction** is a majority vote of the three.

---

## A worked example: the classroom in action

The stage is, deliberately, the same as Parts 1 and 5: the exact
two-moons data and the exact five random 8-label draws that
broke self-training. The base learner is 1-NN — chosen for its
instability, which is what makes the bootstraps diverse.

```
DEMO 1 --- The mechanism: two teachers, one student, one gate
    round 1: classifier 0 accepts 274 pseudo-labels  (teachers' error estimate 0.000)
    round 1: classifier 1 accepts 435 pseudo-labels  (teachers' error estimate 0.000)
    round 1: classifier 2 accepts   3 pseudo-labels  (teachers' error estimate 0.143)
    round 2: classifier 2 accepts 469 pseudo-labels  (teachers' error estimate 0.000)

  Majority vote of the three: 96.0% test accuracy
  (the same three bootstraps WITHOUT unlabelled data: 89.6%)
```

Watch classifier 2. In round 1 its teachers disagree with the
truth on 14.3% of the labelled points where they agree — so the
gate lets in a batch of just **3** (subsampled to keep
`e·|L|` bounded). One round later the teachers have improved,
the estimate drops, and the gate opens to 469. That is the
admission policy working as designed: batch size tracks
measured teacher quality. Result: 96.0%, against 89.6% for the
identical ensemble without the unlabelled pool.

### The scoreboard, third visit

```
DEMO 2 --- The scoreboard: Part 1's five hard draws, revisited
    draw   self-training (Part 1)   bagged baseline   tri-training
      0           85.4%                89.6%           96.0%
      1           59.6%                83.6%           74.6%
      2           51.6%                67.4%           80.6%
      3           87.0%                77.2%           94.4%
      4           85.2%                82.0%           85.4%
     mean          73.8%                80.0%           86.2%
```

The fair comparison is the middle column — the *same* three
bootstrap 1-NNs, trained on the labels alone — because ensembling
by itself is already worth +6 points over Part 1's lone
self-trainer. Tri-training adds another **+6 on top** from the
unlabelled pool, and four of the five draws improve, two of them
dramatically (77.2 → 94.4; 67.4 → 80.6).

And then there is draw 1, which this article refuses to hide:
tri-training *loses* to its own baseline there, 83.6 → 74.6. The
cause is visible in DEMO 1's diagnostics: the gate's error
estimate is computed on **eight labelled points**, and eight
points make a noisy voltmeter — note those round-1 estimates of
exactly 0.000, an obviously optimistic reading from a tiny
sample. Sometimes the voltmeter waves a bad batch through. Three
judges shrink the risk of confident nonsense; they do not end
it. (Part 5's label propagation, which replaces judged opinions
with geometry, scored 98.6% on this same draw — every mechanism
in this track has a domain where it is the right one.)

### The gate is the point

Same three judges, same agreement rule — but accept *every*
agreed batch, no error check, no subsampling:

```
DEMO 3 --- The gate is the point: remove it and watch
    draw   tri (gated)   tri (no gate)      pseudo-labels used
      0       96.0%        94.8%           1181 gated vs  5310 ungated
      1       74.6%        78.0%           1170 gated vs  5210 ungated
      2       80.6%        81.8%           1151 gated vs  5086 ungated
      3       94.4%        77.4%           1467 gated vs  5898 ungated
      4       85.4%        83.2%           1327 gated vs  5606 ungated
```

Ungated, each classifier swallows essentially the whole pool
every round — five thousand pseudo-labels of unchecked quality —
and the mean drops from 86.2% to 83.0%, with draw 3 collapsing
by seventeen points. The gate admits roughly a fifth as many
labels, each batch only when the measured noise bound improves.
(Honesty again: the flood gets lucky on two draws — noise
sometimes cancels — but on average, and at its worst, it costs.)
Agreement finds candidate labels; the gate decides whether they
are safe to eat.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Tri-training is three self-trainings plus bookkeeping.

**Per round**: three base-model fits and three scoring passes
over the pool (`O(3(F + U · P))`), plus the gate's error
measurement on the labelled set — negligible.

**Total**: rounds until no gate opens (a handful here — the
gate is also a natural stopping rule, something plain
self-training lacks).

**Memory**: three training sets, `O(3 · N · d)` worst case.

**What the third model buys**: teachers for every student
without needing views, a majority vote at prediction time, and —
through the gate — a *measurable* admission policy with a
classical noise bound behind it. The price is 3× the base
model's cost and the standing caveat that the gate is only as
good as the labelled set it measures on.

---

## The ensemble thread

Tri-training sits at the junction of two ideas this series keeps
meeting. From **bagging** it takes bootstrap diversity and
majority voting; from **co-training** it takes peer teaching and
agreement as evidence. Its descendants push both directions:
co-forest extends the committee to a full random forest;
multi-view co-regularisation blends it back toward Part 2; and
the modern deep-learning echoes — co-teaching for noisy labels,
consistency between differently-augmented students — reuse the
same triangle of *diverse peers checking each other's homework*
with networks in place of 1-NNs. The gate's lesson travels
furthest of all: whatever proposes your pseudo-labels, admit
them on a **measured** error budget, not on confidence alone.

---

## What comes next

Every method so far has manufactured its safety from other
models — more judges, more views, more graphs. Part 8, **Virtual
Adversarial Training (VAT)**, manufactures it from the *loss
function*: perturb each unlabelled point in the direction that
most changes the model's prediction, and penalise that change.
No teachers, no committees — just a smoothness demand enforced
adversarially, and one of the strongest ideas to survive into
the deep-learning era of semi-supervision.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**tri_training.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/07-tri-training/tri_training.py)

Run it with:

```bash
pip install numpy
python tri_training.py
```

It needs only `numpy` and runs in a couple of seconds.
Everything is from scratch: the bootstrap ensemble of 1-NN
classifiers, the rotating teacher-student rounds, the
Angluin-Laird-style gate with its subsampling rule, the majority
vote, and the three experiments on Part 1's exact data. The
headline insight worth pinning to the wall: **tri-training gets
co-training's peer-checking without views — three
bootstrap-diversified classifiers where any two agreeing teach
the third — but its real engine is the gate, which admits a
pseudo-label batch only when the teachers' measured error keeps
the expected noise mass `e·|L|` shrinking; on the five draws
that broke self-training (73.8%) it reaches 86.2% against its
own bagged baseline's 80.0%, loses one draw honestly to its
eight-point error estimate, and without the gate drops three
points while swallowing five times the pseudo-labels**.

---

*This is Part 7 of the Semi-Supervised Learning track in the Algorithms in Python series. The companion script `tri_training.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It delivers [Part 2](https://medium.com/p/6f581df978ff)'s error check without the view requirement, on [Part 1](https://medium.com/p/eeca5accd031)'s exact data. Part 8 will look at Virtual Adversarial Training, where the smoothness demand moves into the loss function itself.*
