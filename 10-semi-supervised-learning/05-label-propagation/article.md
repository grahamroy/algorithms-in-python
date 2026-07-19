# Label Propagation — Let the Labels Flow

### *Algorithms in Python --- Semi-Supervised Learning, Part 5*

---

Every method in this track so far has *reasoned* its way from
few labels to many — promoting confident guesses, trading
nominations between judges, learning shared representations,
fitting generative stories. **Label propagation** (Zhu &
Ghahramani, 2002) does something almost embarrassingly more
direct: it treats labels as a **fluid** and lets them flow.

The bet is the **smoothness assumption** in its purest form:
*nearby points should share a label*. No distributions, no
parameters, no classifier — just geometry. Build a graph
connecting every point to its nearest neighbours, labelled and
unlabelled alike; pour the known labels in at their points; and
let them diffuse along the edges until the whole graph is
covered. Dense regions conduct labels; empty regions insulate.
The low-density gap between classes — the same gap every
article in this track has leaned on — stops the flow *by
construction*, because a gap is precisely where there are no
edges.

This article builds the algorithm from scratch, watches the
flood spread hop by hop over a thousand points, and then settles
a score: the five randomly-placed label draws that made Part 1's
self-training collapse — one of them to a coin flip — get
revisited, on exactly the same data. All five come back at ~98%.

---

## The recipe

Three steps, and the third has a beautiful closed form.

**1. Build the graph.** Connect each point (labelled or not) to
its `k` nearest neighbours, and weight each edge by similarity —
a Gaussian kernel `w_ij = exp(−‖x_i − x_j‖² / 2σ²)`, so closer
neighbours pull harder. Row-normalise the weights into a
transition matrix `P`: each point's row says how much it listens
to each neighbour.

**2. Let the labels flow.** Give every labelled point a one-hot
belief vector and every unlabelled point nothing. Then repeat:

```
F  ←  P · F          every point averages its neighbours' beliefs
F[labelled]  ←  known labels        (the clamp)
```

Beliefs spread outward from the labelled points like heat
through a plate whose labelled points are held at fixed
temperatures. The clamp is essential: the known labels are
boundary conditions, never overwritten.

**3. Read off the equilibrium.** The iteration converges to the
**harmonic solution**: every unlabelled point's belief is
exactly the weighted average of its neighbours'. It has a lovely
probabilistic reading — a point's score for class A is *the
probability that a random walk starting there reaches a
labelled A point before any other label* — and a closed form,

```
F_unlabelled  =  (I − P_uu)⁻¹ P_ul · Y_labelled
```

one linear solve, which is what the companion script uses for
its final answers. No learning rate, no epochs, no model: the
graph and the labels determine everything.

---

## A worked example: watching the flood

The companion script uses the exact two-moons pipeline of
Part 1 — same generator, same master seed — with 1,000 points in
the graph (the 500-point pool plus 500 held-out test points, all
unlabelled except **8**), each point tied to its 7 nearest
neighbours with `σ = 0.05`.

```
DEMO 1 --- The flood: labels spread hop by hop along the graph
    iteration   points reached (of 1,000)
         1               69
         2              161
         3              295
         5              545
        10              857
        20             1000

  Equilibrium (closed-form harmonic solution): 98.2% accuracy
```

The table is the mechanism. Eight labelled points touch 69
neighbours on the first hop, 161 by the second, and the wave
rolls down the arcs of both moons until every one of the 1,000
points has been reached by iteration 20. The flood follows the
data's shape — along the crescents, never across the gap,
because the gap has no edges to carry it. Equilibrium: **98.2%**
on the 500 held-out points, from 8 labels.

### The rescue: Part 1's catastrophe, revisited

Part 1 ended with a warning table: self-training with 8
*randomly-placed* labels averaged 73.8%, and one unlucky draw
fell from a 93.2% baseline to 51.6% — a coin flip — as the
frontier spread from badly-seeded positions and compounded its
own errors. Same data, same five draws, label propagation:

```
DEMO 2 --- The rescue: Part 1's catastrophic label draws, revisited
    draw   self-training (Part 1)   label propagation
      0           85.4%                98.2%
      1           59.6%                98.6%
      2           51.6%                98.4%
      3           87.0%                98.4%
      4           85.2%                97.8%
     mean          73.8%                98.3%
```

Every draw, including the catastrophe, lands between 97.8% and
98.6%. The reason is structural, not luck: self-training's
frontier grows outward *from wherever the seeds sit*, so bad
placement means bad geometry from round one. Label propagation's
reach is set by the **graph's** shape — a label planted anywhere
on a moon flows along that moon's edges to all of it. Placement
stops mattering because the data's connectivity, not the seeds'
positions, decides where influence travels.

### The graph is the model

That power has a precise price, and the third experiment states
it three ways. Same algorithm, same 8 labels — only the graph
construction changes:

```
DEMO 3 --- The graph IS the model: three ways to break it
    graph                                  accuracy   unreached
    sharp local graph (k=7, sigma=0.05)      98.2%         0
    blurry edges     (k=7, sigma=0.5)        86.4%         0
    gap erased  (full graph, sigma=1.5)      50.0%         0
    fragmented       (k=2, sigma=0.05)       73.2%       618
```

Blur the kernel (`σ = 0.5`) and the few edges that happen to
cross the gap carry as much weight as on-moon edges — labels
leak, 86.4%. Connect everything to everything with a huge
bandwidth and the gap ceases to exist as far as the flow is
concerned: an exact coin flip. Starve the graph (`k = 2`) and it
fragments into islands — **618 of the 1,000 points are never
reached by any label at all**. Label propagation has no opinions
of its own. It faithfully diffuses over whatever graph you hand
it, which makes graph construction — `k`, `σ`, the distance
metric — the entire modelling decision, and the first place to
look when it misbehaves.

One more honest caveat: this method is **transductive**. It
labels the points *in the graph* — there is no classifier left
over to apply to tomorrow's data. (The script's test points were
inside the graph as unlabelled nodes; that is the standard
protocol. For new points, you either re-solve or attach them to
the graph's labelled output with a nearest-neighbour rule.)

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Label propagation's costs live in the graph, not in any model.

**Building the graph**: `O(n² · d)` for pairwise distances (the
dominant cost at our scale; tree- or hash-based neighbour search
cuts it for large `n`), then `O(n · k)` edges survive.

**Iterative flow**: `O(n · k)` per sweep on the sparse graph —
each point averages `k` neighbours — for however many sweeps the
flood needs (20 reached everything here).

**Closed form**: one linear solve in the unlabelled block,
`O(n_u³)` dense — exact, and preferable at moderate scale;
conjugate-gradient solvers make it near-linear on sparse graphs.

**No training, no parameters**: like MCTS in the RL track, this
is computation at *decision time* over a structure, not learning
of weights. The flip side is transduction — the answers are for
these points, and new data means new computation.

---

## The graph family

Label propagation is the front door to a whole family that
Part 6 explores properly. **Label spreading** (Zhou et al.,
2004) uses the normalised graph Laplacian and a soft clamp —
labelled points can bend slightly, which helps with label
noise. The **harmonic function / Gaussian random field** view
(this article) connects to electrical networks: beliefs are
voltages, labelled points are batteries, edge weights are
conductances. And the modern echo: **graph neural networks** do
learned, feature-aware message passing over exactly this kind of
structure — label propagation with parameters. scikit-learn
ships `LabelPropagation` and `LabelSpreading`; what they do is
what you just built.

**Reach for it when** distances between examples are meaningful,
the classes form connected regions, and you want a strong
transductive answer with zero training. **Hesitate when** the
graph would need to span disconnected class regions (islands
starve), or features are too high-dimensional for raw distances
to mean much — fix the representation first (Part 3's lesson),
then propagate.

---

## What comes next

Part 6, **Graph-Based Methods**, widens this article into the
family view: the graph Laplacian as a smoothness penalty, label
spreading's soft clamp, and graph regularisation as a framework
where propagation, spectral clustering, and modern
message-passing all turn out to be the same idea wearing
different coats.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**label_propagation.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/05-label-propagation/label_propagation.py)

Run it with:

```bash
pip install numpy
python label_propagation.py
```

It needs only `numpy` and runs in a few seconds. Everything is
from scratch: the k-NN Gaussian graph, the clamped iterative
flow with its flood trace, the closed-form harmonic solve, the
Part 1 comparison on identical label draws, and the three graph
ablations. The headline insight worth pinning to the wall:
**label propagation treats labels as a fluid on a similarity
graph — flow to equilibrium with the known labels clamped, and
each unlabelled point ends up at the harmonic average of its
neighbours (equivalently: where a random walk from it first hits
a label); the flood follows the data's connectivity rather than
the seeds' placement, which is why the five badly-placed draws
that broke self-training (mean 73.8%, worst 51.6%) all land at
~98% on the same data — but the graph IS the model: blur it,
over-connect it, or fragment it, and the same flow fails three
different ways**.

---

*This is Part 5 of the Semi-Supervised Learning track in the Algorithms in Python series. The companion script `label_propagation.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). The rescue in DEMO 2 revisits [Part 1](https://medium.com/p/eeca5accd031)'s self-training on identical data; [Part 4](https://medium.com/p/dc0679e0bf4e) covered the generative alternative. Part 6 widens this into the graph-based family.*
