# Active Learning — Ask Where Confusion Matters

### *Algorithms in Python --- Semi-Supervised Learning, Part 11*

---

Every article in this track has carried a silent variable. Ten
methods — self-trainers, committees, graphs, generative
stories, adversaries, margin machines — all accepted their
eight labels as *given*: a random draw's worth of luck, frozen
in Part 1 and inherited ever since. And the luck mattered more
than almost anything else. Draw 1 broke self-training at 59.6%.
Draw 0 capped VAT at 90.8% and stumped the TSVM's restarts.
The scoreboard's variance was never only in the algorithms; it
was in the ticket.

**Active learning**, the track's finale, asks the only question
left: *if you may choose what gets labelled, which eight points
buy the most?* The setting is honest about how labels actually
happen — a human **oracle** paid per answer — and the loop is
the simplest in the series:

1. Train on the labels you have.
2. Let the model **nominate** the unlabelled point whose label
   it wants next.
3. Pay the oracle. Add the answer. Repeat until the budget dies.

Everything interesting lives in step 2, and this article races
the classic nomination strategies against each other on the
track's own stage — same two moons, same 500-point pool, and
Part 5's label propagation as the learner throughout, so the
only variable in the whole experiment is *which labels*.

---

## The candidates

**Uncertainty sampling** — query where the current model is
least sure. Here, the point whose two label beliefs are closest
(the smallest margin). The oldest, most intuitive strategy, and
the one with the most famous vice.

**Information density** (Settles & Craven, 2008) — uncertainty
*times* local density: be unsure **and** be somewhere
representative. A one-line multiplication that repairs the
famous vice.

**Farthest-first** — no model opinion at all: query the point
farthest from everything labelled so far. Pure coverage; the
seed of modern coreset methods.

**Random** — the control. Any strategy that cannot beat a
lottery ticket is decoration.

---

## A worked example: the lottery, the race, the vice

### First, the lottery the track has been playing

One hundred and twenty random 8-label draws, every one solved
by the same label propagation that rescued Part 1's disasters:

```
DEMO 1 --- The label lottery: what the whole track lived with
    minimum   5th pct   median   maximum
     72.2%    80.0%    98.4%    99.2%

  Draws below 95%: 19 of 120.  Below 90%: 16.
```

The median says *solved*; the tail says *sometimes 72%*. Eight
random labels can all land where the flood starts on the wrong
side of a thin spot, and nothing in the training data warns
you. One draw in seven lands below 95%. Part 1's five draws —
97.8–98.6% under this learner — were **good tickets**: the
track, in eleven articles, never saw its own tail.

### Now choose instead of drawing

Each strategy starts from one label per class and nominates six
queries, to the same budget of eight. Twelve starting pairs,
learner never changes:

```
DEMO 2 --- The race: the same budget, chosen instead of drawn
    labels:            2       4       6       8      worst at 8
    random          78.4%  94.7%  94.6%  95.8%      88.4%
    uncertainty     78.4%  92.5%  96.3%  98.6%      96.0%
    info-density    78.4%  95.3%  98.5%  98.6%      98.4%
    farthest-first  78.4%  81.8%  93.0%  98.4%      97.4%
```

Two facts here are worth the whole track. **Information density
reaches ~98.5% by six labels** — the budget question, answered:
the right six beat eight random. And at the full budget its
*worst* starting pair (98.4%) beats random's *average* (95.8%).
Look at the lottery again: its median was already 98.4%. Active
learning's product is not a higher ceiling — it is a **higher
floor**. It doesn't make the good tickets better; it amputates
the tail where the bad tickets lived.

### Where each strategy points

```
DEMO 3 --- Where the strategies point: hermits and the square
    (mean local density of queried points; 1.0 = densest)
    uncertainty     0.26
    info-density    0.72
    farthest-first  0.18
```

Pure uncertainty **interviews hermits**. The margin is smallest
at sparse frontier points and unreached corners, so its early
queries go to the least representative citizens — and at four
labels it trails even random (92.5% vs 94.7%). That is the
classic **cold-start**: an ill-trained model's confusion is not
yet worth following. Farthest-first has the mirror vice —
coverage without judgement (0.18, the sparsest picks of all,
and the slowest start in the table). Information density asks
its questions in the village square — uncertain *and*
representative — and owns both the fastest curve and the
highest floor.

The strategy is not *ask where you are confused*. It is *ask
where confusion matters*.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The loop's computational cost is learner-dependent — here each
retrain is Part 5's closed-form solve, `O(n³)` on the graph,
and each nomination is a cheap pass over the pool. But the
honest accounting inverts the table: in active learning
**the dominant cost is the oracle**, not the CPU. Every
strategy above spends identical compute per query; they differ
only in how much *accuracy each human answer purchases*. That
is the whole economic argument: when a label costs a
radiologist's minute or a lab assay, an hour of GPU time spent
choosing the next question is a rounding error.

The practical caveats are the field's folklore: retraining per
query is wasteful at scale (batch-mode AL queries dozens at
once, trading freshness for throughput); tired oracles return
noisy labels; and a model that is badly wrong nominates badly
— the cold-start above, which warm pools or density weighting
must cover.

---

## The wider field

Everything here scales up along recognisable lines.
**Query-by-committee** replaces the margin with disagreement
among an ensemble — Part 7's tri-training instincts, repurposed
for choosing rather than teaching. **Expected error reduction**
asks the counterfactual directly (which label would shrink my
future mistakes most?) at brutal cost. **BALD** and its
Bayesian kin measure information gain under model uncertainty;
**coreset** methods industrialise farthest-first for deep
networks, where uncertainty is least trustworthy. And the
modern data-centric pipelines that label web-scale corpora run
this exact loop with humans-in-the-loop at both ends. Settles'
survey remains the field's front door.

---

## The track, closed

Eleven parts ago, this track began with a promise and a
cautionary tale: 500 points, eight labels, and a self-trainer
that could hit 93% or 52% depending on the draw. Every part
since has been a different answer to the same question — *what
must be true of the world for unlabelled data to help?*
Committees demanded independent errors. Graphs demanded that
geometry carry labels. Stories demanded that the data was made
the way the model imagines. Adversaries and margins demanded
an empty gap. Each assumption, priced and broken on the same
five draws.

The finale's answer is the quiet one underneath all of them:
part of the variance was never in the assumptions at all. It
was in eight random coin flips nobody audited. Choose the
coins, and the track's hardest lever — *which* labels — moves
from luck to design.

**What comes next**: a new section. The Deep Learning
Architectures track opens with the multi-layer perceptron —
the machine this track kept meeting in disguise (Part 8's
network, Part 9's decoders), finally taken apart on its own
terms.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**active_learning.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/11-active-learning/active_learning.py)

Run it with:

```bash
pip install numpy
python active_learning.py
```

It needs only `numpy` and runs in about a minute and a half.
Everything is from scratch: Part 5's graph and harmonic solver
rebuilt verbatim, the four nomination strategies, the
120-ticket lottery audit, and the twelve-seed race. The
headline insight worth pinning to the wall: **active learning
attacks the variable every other semi-supervised method
accepts as given — which points get labelled — and its product
is a floor, not a ceiling: the label lottery's median was
already 98.4% but its tail reached 72.2%, and choosing queries
by uncertainty-times-density lifted the worst case to 98.4%
while matching eight random labels with six chosen ones —
because the right question is never where the model is most
confused, but where its confusion matters**.

---

*This is Part 11 of the Semi-Supervised Learning track in the Algorithms in Python series — the track finale. The companion script `active_learning.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It audits the label lottery that [Part 1](https://medium.com/p/eeca5accd031) drew and the whole track inherited, using [Part 5](https://medium.com/p/56d8ae72db5a)'s label propagation as the fixed learner, and closes the question [Part 10](https://medium.com/p/14b31a5ba0fc) left open. The next section, Deep Learning Architectures, begins with the Multi-Layer Perceptron.*
