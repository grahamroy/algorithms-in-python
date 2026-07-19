# Graph-Based Methods — One Matrix Under All of It

### *Algorithms in Python --- Semi-Supervised Learning, Part 6*

---

Part 5 built one algorithm: labels flowing over a similarity
graph to a harmonic equilibrium. It worked beautifully — and it
is one member of a family. This article is the family portrait,
and the family has a single patriarch: the **graph Laplacian**,

```
L  =  D − W
```

the diagonal degree matrix minus the weight matrix. On paper it
looks like bookkeeping. Its power is one identity:

```
fᵀ L f   =   ½ · Σ_ij  w_ij (f_i − f_j)²
```

Take *any* labelling `f` of the graph's nodes and this quadratic
form returns its **roughness** — the total squared disagreement
across every edge, weighted by how strongly the endpoints are
connected. Smooth labellings (neighbours agree) score low;
jagged ones score high. One number, computable for anything you
might write on the graph.

That number turns the whole graph family into variations on a
single sentence — *make `fᵀLf` small without contradicting what
you know* — and this article demonstrates three of its faces:
the meter itself, the spectrum that finds classes with **zero**
labels, and a practical trick Part 5 couldn't do: using the
graph to **audit the labels themselves** and catch the poisoned
ones.

---

## The family, in one objective

Write the family's shared goal as

```
minimise    fᵀ L f    +    μ · (fidelity to the known labels)
```

and the members differ only in how hard they hold the labels:

- **Harmonic propagation** (Part 5) takes `μ → ∞`: the known
  labels are *clamped*, immovable boundary conditions, and the
  smoothest completion is the harmonic solution the flood
  converged to.
- **Label spreading** (Zhou et al., 2004) takes `μ` finite — a
  *soft clamp*. Known labels can bend if the graph pushes back
  hard enough. One dial, `α`, sets the balance between trusting
  the graph and trusting the annotator, and the solution is again
  one linear system (the script uses its closed form).
- **Spectral methods** take the labels away entirely: minimise
  `fᵀLf` subject only to `f` being balanced and non-trivial —
  and that constrained minimisation *is* an eigenvector problem.

Same objective, three settings of one knob: infinite trust,
finite trust, no labels at all.

---

## A worked example, in three acts

The stage is Part 5's exact graph: 1,000 two-moons points,
each tied to its 7 nearest neighbours with Gaussian weights.

### Act 1: the meter

```
DEMO 1 --- The Laplacian is a smoothness meter
    the true labelling                :     57.9
    the truth with  2 points flipped  :     99.8
    the truth with 10 points flipped  :    181.6
    a random labelling                :   3510.1
```

The true class labelling scores 57.9 — almost all its
disagreement coming from the handful of noisy points near the
gap. Flip just **two points out of a thousand** and the meter
nearly doubles: each flipped point now disagrees with its entire
neighbourhood, and the quadratic form bills every one of those
edges. A random labelling reads sixty times rougher. This
sensitivity is the family's entire engine: the truth is
*detectably* smooth, so smoothness is worth optimising.

### Act 2: the spectrum knows

Minimising `fᵀLf` with no labels at all (excluding the trivial
all-ones labelling) is, by the standard variational argument, an
eigenvector problem — and the smoothest non-trivial direction is
the second eigenvector of the (normalised) Laplacian, the
**Fiedler vector**:

```
DEMO 2 --- The spectrum knows: classes from ZERO labels
  Smallest eigenvalues: 0.0000, 0.0002, 0.0007, 0.0010
  Sign of the Fiedler vector vs the true classes: 96.7%
```

Read that carefully: the sign pattern of one eigenvector —
computed from the graph alone, before any label exists — agrees
with the true classes on **96.7%** of the points. The class
structure was *in the graph all along*; labels only tell you
which half is called what. (If this feels familiar, it should:
it is spectral clustering, from the unsupervised track,
resurfacing as the zero-label endpoint of the same smoothness
family. The tiny second eigenvalue against a visible gap to the
third is the spectrum's way of saying "this graph has exactly
two loosely-joined pieces.")

### Act 3: the audit

Soft clamping earns its keep in a way accuracy tables don't
show. Because label spreading's solution is a linear system, a
**leave-one-out** question becomes a cheap rank-one update: for
each labelled point, *remove its own clamp and ask what belief
the graph would hand it from everything else.* If the graph's
answer contradicts the given label, somebody should look at that
label again.

The experiment: 40 labelled points, of which **4 are poisoned**
(flipped) — and the annotator doesn't know which. Rank all 40 by
graph disagreement:

```
DEMO 3 --- The audit: the graph proofreads your labels
    trial   ranks of the 4 poisoned labels (of 40)
      0          [1, 2, 3, 5]
      1          [1, 2, 6, 7]
      2          [1, 2, 3, 4]
```

All twelve poisoned labels across the three trials land in the
top 7 of 40 — trial 2 is a perfect 1-2-3-4. The stragglers at
ranks 6–7 are instructive rather than embarrassing: they are
poisoned seeds that happen to sit *near each other*, each
vouching for the other's lie — exactly how label noise hides in
real datasets, where one confused annotator mislabels a whole
neighbourhood of similar examples. The practical recipe is
free: before trusting a labelled dataset, propagate, rank the
labels by disagreement with their own neighbourhood, and read
the top of the list.

One honest note from the experiments behind this article: on
this dataset the *accuracy* difference between hard and soft
clamping is small in every condition tested — the family's
practical dividends here are the spectrum and the audit, not a
robustness gap between clamps.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Everything in the family is linear algebra on one sparse matrix.

**The meter**: `fᵀLf` costs `O(edges)` — one pass over the
graph, `O(n · k)` here.

**The spectrum**: a full eigendecomposition is `O(n³)`, but only
the few smallest eigenvectors are needed, and sparse iterative
eigensolvers get them in roughly `O(edges)` per iteration —
spectral partitions of million-node graphs are routine.

**Spreading / propagation**: one sparse linear solve —
`O(n_u³)` dense, near-linear with conjugate-gradient methods on
sparse graphs.

**The audit**: with the inverse (or a factorisation) in hand,
each leave-one-out check is a rank-one update — `O(1)` extra per
labelled point. Forty audits cost essentially nothing beyond the
solve you already did.

---

## From Laplacians to message passing

The modern echo, one paragraph, because the lineage is direct.
**Graph neural networks** compute, at each layer, a learned
transformation of each node's neighbourhood average — which is
to say, they apply the same neighbourhood-smoothing operator
this family is built on, interleaved with learned feature maps.
Simplified GNN variants (SGC) are *literally* powers of the
normalised adjacency applied to features; "over-smoothing", the
famous deep-GNN failure, is this article's flow run too long,
converging toward the Fiedler-dominated equilibrium. The
Laplacian family isn't a historical curiosity under the GNN era
— it is the linear core the learned machinery wraps.

**Reach for these methods when** similarity is meaningful and
classes form connected regions: fast, parameter-light, and — as
Act 3 shows — useful even as a data-quality tool before any
model is trained. **The standing caveat is Part 5's**: the graph
is the model, and every conclusion — meter, spectrum, audit — is
only as good as the edges.

---

## What comes next

Part 7, **Tri-Training**, returns to the classifier-ensemble
thread with a twist on Part 2: it gets co-training's error-check
*without needing two views at all* — three classifiers, trained
on bootstrap samples, where any two agreeing can teach the
third. Disagreement as a resource, view requirements dropped.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**graph_methods.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/06-graph-based-methods/graph_methods.py)

Run it with:

```bash
pip install numpy
python graph_methods.py
```

It needs only `numpy` and runs in a couple of seconds.
Everything is from scratch: the k-NN Gaussian graph, the
Laplacian and its quadratic form, the normalised-Laplacian
eigendecomposition for the Fiedler split, label spreading's
closed form, and the rank-one leave-one-out audit. The headline
insight worth pinning to the wall: **the graph family is one
objective — minimise the Laplacian roughness `fᵀLf = ½Σw_ij(f_i−f_j)²`
under some loyalty to the known labels — with hard clamping
giving Part 5's propagation, soft clamping giving label
spreading, and no labels at all giving the spectrum, whose
Fiedler vector already carries 96.7% of the class structure;
and because the truth is detectably smooth (two flipped points
nearly double the meter), the graph can even proofread your
labels — ranking all twelve poisoned labels into the top 7 of
40, with the escapees hiding exactly the way real label noise
does: next to each other**.

---

*This is Part 6 of the Semi-Supervised Learning track in the Algorithms in Python series. The companion script `graph_methods.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It generalises [Part 5](https://medium.com/p/56d8ae72db5a)'s label propagation into the Laplacian family. Part 7 will look at Tri-Training, which wins co-training's error check without needing views.*
