# Transductive SVM — The Widest Empty Street

### *Algorithms in Python --- Semi-Supervised Learning, Part 10*

---

Every assumption this track has leaned on — smoothness,
clusters, the low-density gap — has an oldest, bluntest
statement, and it belongs to Vapnik: **the decision boundary is
a street, and the street should be wide and empty**. The
supervised SVM already believes this; it just can't see most of
the town. Trained on eight labelled points it finds the widest
street *between the points it can see* — and parks it, with
perfect confidence, on top of five hundred points it cannot.

The **Transductive SVM** (Vapnik, 1998; made practical by
Joachims, 1999) repairs that with one loss term and one
constraint. Every unlabelled point gets the *hat loss*
`max(0, 1 − |f(x)|)`: a penalty for standing inside the street,
whichever side of it the point calls home. No pseudo-labels, no
graph, no generative story, no adversary — the margin itself,
the same idea that powered the supervised SVM, becomes the
semi-supervised engine:

```
min   λ/2 ||w||²  +  C_l · hinge(labelled)  +  C_u · hat(unlabelled)

subject to:  the pool's predicted class balance is pinned
             to the labelled ratio
```

That constraint is not decoration. Without it the cheapest
empty street is *around* the town — shove everyone to one side
and nobody is inside. With it, the only way to empty the street
is to move the street.

The price of this elegance is printed in the loss itself: the
hat has two downhill directions per point — push it left or
push it right — so the objective is **non-convex**, the single
property that shaped every practical TSVM ever built. This
article builds one from scratch, watches it move the boundary
into the gap, and then maps the valleys it can fall into — and
the label-free trick that picks the right one.

---

## Making it practical

**The model.** An SVM needs a kernel for moons, so the script
builds an RBF kernel machine the cheap modern way: **random
Fourier features** — 200 random cosines whose inner products
approximate the Gaussian kernel — with `f(x) = w·φ(x) + b`
trained by subgradient descent (Adam). A bonus that matters
later: each restart *redraws* the random feature map, handing
the non-convex search genuinely different starting geometry.

**The balance pin.** The bias `b` is not a trained parameter.
Every step it is set analytically so the pool's mean output
equals the labelled class ratio — the constraint holds
*exactly*, and the one-class collapse is simply not in the
search space.

**The annealing.** Following Joachims, the unlabelled weight
`C_u` climbs from 1/128th of full strength to full over eight
stages: the labelled street forms first, the unlabelled
evidence bends it gradually.

**The selection.** Non-convex objectives are handled the
honest way: several restarts, keep the *lowest objective*.
Note what that selection does **not** need: labels. An emptier
street is visibly emptier — you can score it on unlabelled data
alone.

---

## A worked example: eight labels and a full street

The stage is Part 1's exact two-moons data and its five random
8-label draws.

```
DEMO 1 --- The supervised street: margin over 8 points
    draw:    73.8%   84.2%   84.2%   88.6%   87.2%    mean 83.6%
    inside:  44.0%   42.4%   29.6%   44.6%   17.4%    mean 35.6%
```

The supervised SVM is the track's strongest baseline yet
(kernel smoothness alone beats Part 8's supervised network,
83.6% vs 77.4%). But look at the second row: **a third of the
pool is standing inside a margin that is supposed to be
empty**. The labels cannot object — they are all comfortably
outside it. Only the unlabelled points know, and nothing asks
them.

### Ask them

```
DEMO 2 --- The transductive street: everyone must clear it
    draw   supervised   transductive     chosen restart
      0        73.8%        98.6%         J = 0.0474
      1        84.2%        98.4%         J = 0.0297
      2        84.2%        93.6%         J = 0.0459
      3        88.6%        98.4%         J = 0.0256
      4        87.2%        98.0%         J = 0.0269
     mean       83.6%        97.4%

    street occupancy of the chosen solutions: 2.6%  4.2%  7.0%  3.4%  4.0%
```

Mean 97.4%, every draw at 93.6% or better, and the street
occupancy collapses from a third of the pool to a few percent.
Draw 0 — the draw that capped VAT at 90.8% and the Gaussian
story at 83.8% — hits 98.6%. The boundary moved into the gap
for the most literal reason in this track: **the gap is the
only place a wide empty street fits**. That puts the TSVM in
the track's top tier — 97.4% against the VAE mixture's 98.0%
and label propagation's ~98.3% — achieved with nothing but a
margin and a constraint.

### The hat is not a bowl

```
DEMO 3 --- The hat is not a bowl: six restarts, three streets
    objective J     test accuracy
      0.0474          98.6%   <-- chosen
      0.0558          88.6%
      0.0617          98.0%
      0.0779          76.0%
      0.0899          61.2%
      0.0914          65.6%
```

Draw 0's six restarts land in three distinct basins: the true
gap (~98%), a street that shears through one moon (~76%), and
worse (~61–66%). This is the non-convexity made visible —
*trust a single run and you inherit its valley*: the six runs
average 81.3%, spanning 61.2% to 98.6%. But sort the table by
its first column: the objective ranks the basins almost
perfectly, because emptier streets genuinely score lower.
Label-free selection recovers the 98.6%.

Two more measured facts complete the honest picture. Unpin the
balance and the street's population drifts wherever its valley
leads — across the same six restarts the pool splits 38–71%
positive (accuracy 52.2–98.6%) against a true 50/50; the pin
costs one line and deletes that whole axis of failure. And
Joachims' annealing schedule, kept in the script for fidelity,
measured as *neutral* on a problem this small — restarts plus
objective selection already explore what the schedule exists
to protect. At scale, with stochastic gradients and no budget
for six restarts, it earns its keep.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

**One training run** is plain full-batch subgradient descent:
`O(N · D)` per epoch for `N` pool points and `D` random
features — every term, the balance pin included, rides the same
`P·w` product. **The full algorithm** multiplies by epochs and
restarts: `O(R · E · N · D)`. The script's 6 restarts × 6,000
epochs × 500 points finish in about half a minute in NumPy.

**Against its own history**: Joachims' original label-switching
TSVM solves a full SVM QP inside every iteration of a
combinatorial outer loop — exact but brutal; the gradient
version here (in the spirit of Chapelle & Zien's ∇TSVM) trades
exactness for a smooth ride at `O(N · D)` a step.

**The knobs**: kernel width `σ` (0.35 here — the street must be
drawable at the data's scale), `C_l` vs `C_u` (how loudly
labels outrank geometry), and the restart count `R` — the only
knob that exists purely because the hat is not a bowl.

---

## A word about the name

Vapnik's framing was philosophical: *"do not solve a more
general problem as an intermediate step"* — if you only need
labels for *these* points, ask for exactly that
(**transduction**), not for a rule over all space
(**induction**). The irony of the practical TSVM is that it
delivers both: the optimisation is transductive — those 500
specific unlabelled points shaped the street — but `w` and `b`
survive it, so a full inductive classifier walks out for free
(DEMO 2's scores are on untouched test data). Modern
literature often calls the same objective **S3VM**,
semi-supervised SVM, precisely to drop the philosophical
baggage. And the family tree runs forward from here: swap the
hat loss's hard hinge for a smooth confidence penalty on a
neural network and you are within arm's reach of entropy
minimisation and Part 8's VAT — the low-density street, one
generation later, enforced by an adversary instead of stated
as an objective.

---

## What comes next

Part 11 closes the track with **Active Learning** — the
question every part so far has carefully stepped around. All
ten methods accepted their eight labels as given, a random
draw's worth of luck; the whole scoreboard's variance came from
*which* eight. Active learning asks the only question left:
if you may choose what gets labelled, which eight points buy
the most?

---

## The complete code

The full script is on GitHub — grab it and run it:

[**tsvm.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/10-transductive-svm/tsvm.py)

Run it with:

```bash
pip install numpy
python tsvm.py
```

It needs only `numpy` and runs in about half a minute.
Everything is from scratch: the random Fourier feature kernel
machine, the hinge and hat subgradients, the exact balance pin
through the bias, the annealing schedule, and the restart
selection. The headline insight worth pinning to the wall:
**the TSVM states semi-supervised learning as pure geometry —
punish any unlabelled point standing inside the margin street
(the hat loss) while pinning the class balance, and the street
relocates to the low-density gap because that is the only
place a wide empty street fits: 83.6% becomes 97.4% on eight
labels, with a third of the pool inside the supervised street
and 3–7% inside the transductive one — and the price is
non-convexity, paid honestly with restarts whose objective
values rank their valleys almost perfectly, so the right
street can be chosen without ever touching a label**.

---

*This is Part 10 of the Semi-Supervised Learning track in the Algorithms in Python series. The companion script `tsvm.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It is evaluated on [Part 1](https://medium.com/p/eeca5accd031)'s exact label draws, states as optimisation the low-density idea that [Part 8](https://medium.com/p/5ad699fa2684)'s VAT enforced with an adversary, and sits beside [Part 9](https://medium.com/p/59e50bb71435)'s VAE mixture in the track's top tier. Part 11, the track finale, will look at Active Learning.*
