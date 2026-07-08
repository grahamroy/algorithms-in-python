# Co-Training — Two Models That Grade Each Other's Homework

### *Algorithms in Python --- Semi-Supervised Learning, Part 2*

---

Part 1 ended with self-training's structural flaw on full
display: one model, promoting its own confident guesses to
training labels, **grades its own homework**. When a confident
guess was wrong, nothing in the loop could catch it — the error
was retrained on as truth, recruited its neighbours, and
compounded until a 93% baseline collapsed to a coin flip.

**Co-training** (Blum & Mitchell, 1998) is the classic remedy,
and its idea is disarmingly social: if one judge can't be
trusted to check itself, hire a **second judge with different
information**. Suppose every example naturally splits into two
**views** — two feature sets, each sufficient to classify on its
own. Blum and Mitchell's original: classify university web
pages using *(view A)* the words on the page and *(view B)* the
words in the links pointing **at** the page. Either view works
alone; crucially, they fail in different places.

Then train one classifier per view and let them teach each
other: each nominates the unlabelled examples it is most
confident about, and those pseudo-labels go into a **shared
pool** — so each model spends its rounds labelling examples *for
the other*. A wrong pseudo-label now has to fool two judges with
different evidence. And when the judges *disagree* about an
example, the nomination is simply vetoed — a built-in error
detector that self-training does not have and cannot have.

This article builds co-training from scratch, watches the veto
mechanism catch dozens of would-be errors on the way to oracle
accuracy, pits it against self-training across five datasets —
and then breaks its famous assumption on purpose, to show
exactly where the power comes from.

---

## The algorithm

Co-training needs data whose features split into two views,
each individually predictive. Given that:

1. **Train** one classifier per view on the labelled set.
2. **Nominate**: each classifier marks the unlabelled examples
   it is confident about, with its predicted label.
3. **Veto**: if both classifiers nominate the same example with
   *different* labels, discard the nomination — the judges
   disagree, so somebody is wrong.
4. **Promote** the surviving nominations into the shared
   labelled pool — each example now carries its label into
   *both* models' next round.
5. **Retrain** both and repeat until nothing new is nominated.

The contrast with self-training is step 3 and the sharing in
step 4. In self-training, confidence is a monologue. Here it is
a conversation — and the argument is where the safety lives.

### Why the views must be different

The theory asks for two properties: each view is **sufficient**
(a good classifier exists using it alone), and the views are
**conditionally independent given the class** — knowing the
label, view A tells you nothing extra about view B. Perfect
independence is rare in practice, but the *spirit* is what
matters: the two views must make **different mistakes**. Where
view A is ambiguous, view B usually isn't; each model hands the
other exactly the labels the other could not safely have
produced itself. Two copies of the same view give you two copies
of the same blind spots — and, as the final experiment shows,
precisely nothing else.

---

## A worked example: two views, ten labels

The companion script makes the assumption literal. Every example
is a hidden class plus **two independent 2-D snapshots** — each
view is its own draw of the two-moons picture given that class.
Same label, independent everything else: conditional
independence by construction. The base classifier per view is
Part 1's k-NN (k = 3, pseudo-labels need unanimous votes), so
the only new ingredient is the second judge.

```
DEMO 1 --- The setting: 10 labels, two half-informed views
  View A alone, 10 labels        :  82.8%
  View B alone, 10 labels        :  88.2%
  Both views concatenated        :  93.6%
  Oracle (all 500 true labels)   :  99.6%
```

Ten labels, and each view alone is a mediocre classifier. Even
concatenating the views only reaches 93.6%. The oracle — all 500
true labels — sits at 99.6%.

### The loop: nominations and vetoes

```
DEMO 2 --- The loop: nominations, vetoes, and the shared pool
    round   nom A   nom B   vetoed   added   wrong so far   accuracy
       1      375     365      21      434          10           99.2%
       2       44      49      11       45          10           99.8%
       3       11       7       7        4          10           99.8%
       4        7       6       6        1          10           99.8%

  Co-training: 93.6% (10 labels) -> 99.8%   [oracle 99.6%]
  51 conflicting nominations vetoed; only 10 wrong labels slipped through.
```

Read round 1 closely, because the whole mechanism is in it. View
A nominates 375 points; view B nominates 365. They overlap and
they complement — between them, 434 distinct points enter the
pool. But **21 nominations disagreed** and were vetoed on the
spot: examples where one view's neighbourhood said class 0 and
the other's said class 1. Those are precisely the ambiguous
points self-training would have swallowed. Over the full run, 51
conflicting nominations were caught by the veto, and only 10
wrong labels slipped past both judges.

The result: **99.8% test accuracy**, statistically at the oracle
(the 0.2% "advantage" over it is one test point — sampling
noise, not magic). Ten real labels, two half-informed views, and
an argument between two classifiers did the rest.

### Two judges vs one

Is the second judge actually earning its keep, or would
self-training on *all* the features do just as well? Same k-NN,
same threshold, same label budget, five independent datasets:

```
DEMO 3 --- Two judges vs one
    seed   self-train (wrong)    co-train (wrong)    vetoes
      0      93.4%  ( 29)        99.8%  ( 10)         51
      1      99.8%  (  2)        99.0%  (  3)          4
      2      99.6%  (  3)        98.8%  (  0)          7
      3      94.2%  ( 36)        99.4%  ( 12)         39
      4      92.6%  ( 35)        96.0%  ( 11)         18
```

The pattern is about **reliability**, and it's worth stating
carefully. On its good days (seeds 1, 2), self-training is
excellent — there was nothing to catch, and co-training matches
it. But on seeds 0, 3, and 4, self-training swallowed 29–36
wrong pseudo-labels and paid for it (92–94%). Co-training never
had a bad day: its wrong-label count stayed at 0–12 on every
seed — look at the veto column tracking exactly the runs where
self-training got into trouble — and its accuracy never left the
96–99.8% band. One judge is fine until it isn't; two judges with
different evidence are consistently fine.

### Breaking the assumption on purpose

Where does the power come from — the second *model*, or the
second *view*? Replace view B with a near-copy of view A (same
coordinates plus a whisper of noise) and run everything again:

```
    seed   self-train (wrong)    co-train (wrong)    vetoes
      0      84.4%  ( 37)        94.2%  ( 20)          0
      1      83.6%  ( 58)        83.6%  ( 58)          0
      2      81.6%  ( 82)        81.4%  ( 82)          0
```

**Zero vetoes, on every seed.** Two judges looking at the same
evidence never disagree, so the error detector never fires, and
co-training collapses into self-training — on two of the three
seeds it admits *the identical wrong labels* (58 and 82). The
second model was never the point. The second, *independent*
view was the point. This is the cleanest empirical statement of
Blum and Mitchell's assumption: co-training buys you exactly as
much as your views' errors are uncorrelated, and nothing more.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Co-training costs roughly **twice self-training** — two base
models per round instead of one — plus a trivial veto check.

**Per round**: two base-model fits and two scoring passes over
the pool (`O(F + U · P)` each), then an `O(U)` comparison of the
nominations.

**Total**: `O(R · 2(F + U · P))` for `R` rounds — four here,
since the shared pool grows fast when two models feed it.

**Memory**: the dataset in both views, `O(N · (d_A + d_B))`,
plus two models. Nothing else.

**What the second model buys**: not accuracy on the good days —
self-training matched it there — but *insurance* on the bad
ones: an error check that fires exactly when one view's
confidence goes wrong, at the price of one extra classifier.

---

## When to use co-training

**Reach for it when your data has a natural view split.** The
classics: a web page's text and its inbound anchor text; a
video's audio and its frames; a product's description and its
reviews; a patient's imaging and their labs. If each view could
classify alone and their failure modes plausibly differ, you
have what the algorithm needs.

**Engineer the split when it almost exists.** Random feature
splits sometimes work in practice, but the duplicate-view
experiment is the warning: views that carry the same information
in different clothes give zero vetoes and zero benefit. The
question to ask is always *"would these two views make different
mistakes?"*

**Skip it when** the features are one indivisible view — forcing
a split of correlated features buys the costs without the
insurance. (Part 7, Tri-Training, gets a similar effect with
*no* view split at all, using three models and disagreement
alone — the idea generalises.)

---

## What comes next

Co-training treats its two views as a happy accident of the
data. Part 3, **Multiview Learning**, makes the views the
central object: instead of two models trading labels, it learns
*representations* that different views agree on — the framework
behind canonical correlation analysis and the agreement
principle that modern contrastive methods rediscovered at scale.
Same instinct — independent perspectives that must concur — one
level deeper in the stack.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**co_training.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/02-co-training/co_training.py)

Run it with:

```bash
pip install numpy
python co_training.py
```

It needs only `numpy` and runs in seconds. Everything is from
scratch: the two-view data generator (independent moons per
view), the per-view k-NN classifiers, the nomination-and-veto
loop with its shared pool, the self-training head-to-head, and
the duplicate-view knockout — with every pseudo-label audited
against the hidden truth. The headline insight worth pinning to
the wall: **co-training replaces self-training's monologue with
an argument — two classifiers on two independent views label
examples for each other, and nominations they disagree on are
vetoed, an error check that caught 51 would-be pseudo-labels on
the way from 93.6% to oracle-level 99.8%; break the independence
(duplicate the view) and the vetoes drop to zero, the check
vanishes, and co-training collapses into self-training — the
power was never the second model, it was the second view**.

---

*This is Part 2 of the Semi-Supervised Learning track in the Algorithms in Python series. The companion script `co_training.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 1](https://medium.com/p/eeca5accd031) covered Self-Training, the one-judge loop this article adds a second judge to. Part 3 will look at Multiview Learning, where the views themselves become the object of study.*
