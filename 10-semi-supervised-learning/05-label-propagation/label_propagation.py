"""
label_propagation.py --- companion code for "Label Propagation"
(Semi-Supervised Learning, Part 5).

EM (Part 4) bet on a parametric story of the data. Label propagation
(Zhu & Ghahramani, 2002) bets on GEOMETRY alone. No distributions, no
parameters -- just the SMOOTHNESS assumption in its purest form: nearby
points should share a label.

The recipe:
  1. Build a similarity graph over ALL points, labelled and unlabelled --
     here: connect each point to its k nearest neighbours, weighted by a
     Gaussian kernel exp(-d^2 / 2 sigma^2).
  2. Let the labels FLOW: every point repeatedly takes the weighted average
     of its neighbours' label beliefs, while the labelled points are
     CLAMPED to their known labels.
  3. At equilibrium, each unlabelled point's belief is a harmonic function:
     exactly the weighted average of its neighbours -- equivalently, the
     probability that a random walk from that point hits a labelled point
     of each class first.

The equilibrium also has a CLOSED FORM (a single linear solve), which the
script uses for its final answers.

Why this matters for the track: self-training (Part 1) collapsed when the 8
labels were placed badly -- its frontier spread from wherever the seeds
happened to sit. Label propagation's flood follows the DATA's shape, not the
seeds' placement: the moons have no edges across the gap, so labels cannot
leak, no matter where they start.

Demonstrates:
  1. The flood: labels physically spreading hop by hop until all 1,000
     points are reached -- and the gap never crossed.
  2. The rescue: the exact five random-label draws that broke self-training
     in Part 1, all solved at ~98%.
  3. The graph IS the model: blur the edges, erase the gap, or fragment the
     graph, and the same algorithm fails in three instructive ways.

Everything is plain NumPy. Dependencies: numpy. Runs in ~10-20 seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
SIGMA = 0.05
KNN = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Two moons -- the exact generator and pipeline of Part 1, so the comparison
# in DEMO 2 is against the very same data and label draws.
# ---------------------------------------------------------------------------

def make_moons(n_per_class, noise, rng):
    t = rng.uniform(0, np.pi, n_per_class)
    top = np.stack([np.cos(t), np.sin(t)], axis=1)
    t = rng.uniform(0, np.pi, n_per_class)
    bottom = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([top, bottom])
    X += rng.normal(0, noise, X.shape)
    y = np.concatenate([np.zeros(n_per_class, dtype=int),
                        np.ones(n_per_class, dtype=int)])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# The graph: k-nearest-neighbour edges, Gaussian weights, row-normalised.
# ---------------------------------------------------------------------------

def build_transition(X, sigma=SIGMA, knn=KNN):
    n = len(X)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    W = np.exp(-d2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    if knn:
        keep = np.zeros_like(W, dtype=bool)
        nn = np.argsort(d2 + np.eye(n) * 1e9, axis=1)[:, :knn]
        keep[np.repeat(np.arange(n), knn), nn.ravel()] = True
        keep |= keep.T                      # symmetrise
        W = W * keep
    P = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-300)
    return P


# ---------------------------------------------------------------------------
# Propagation: iterative (to watch the flood) and closed-form (the answer).
# ---------------------------------------------------------------------------

def propagate_iterative(P, lab_idx, y_lab, iters, trace_at=()):
    n = len(P)
    F = np.zeros((n, 2))
    F[lab_idx] = np.eye(2)[y_lab]
    traces = {}
    for it in range(1, iters + 1):
        F = P @ F
        F[lab_idx] = np.eye(2)[y_lab]       # clamp the known labels
        if it in trace_at:
            traces[it] = int((F.sum(axis=1) > 1e-12).sum())
    return F, traces


def propagate_closed_form(P, lab_idx, y_lab):
    """The harmonic solution: F_u = (I - P_uu)^-1 P_ul Y_l."""
    n = len(P)
    unl = np.setdiff1d(np.arange(n), lab_idx)
    Y_l = np.eye(2)[y_lab]
    F = np.zeros((n, 2))
    F[lab_idx] = Y_l
    F[unl] = np.linalg.solve(np.eye(len(unl)) - P[np.ix_(unl, unl)],
                             P[np.ix_(unl, lab_idx)] @ Y_l)
    return F


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    # Part 1's exact pipeline: same master RNG, same pool, same test set.
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_moons(250, noise=0.15, rng=rng)
    X_test, y_test = make_moons(250, noise=0.15, rng=rng)
    ALL = np.concatenate([X, X_test])
    test = np.zeros(len(ALL), dtype=bool)
    test[len(X):] = True

    def draw_labels(draw):
        d_rng = np.random.default_rng(draw)
        return np.concatenate([
            d_rng.choice(np.where(y == 0)[0], 4, replace=False),
            d_rng.choice(np.where(y == 1)[0], 4, replace=False)])

    banner("DEMO 1 --- The flood: labels spread hop by hop along the graph")
    print("  1,000 points (the 500-point pool plus 500 test points), 8 labels.")
    print(f"  Graph: each point tied to its {KNN} nearest neighbours, Gaussian")
    print(f"  weights (sigma={SIGMA}). Labels flow; labelled points stay clamped.")
    print()
    li = draw_labels(0)
    P = build_transition(ALL)
    _, traces = propagate_iterative(P, li, y[li], iters=20,
                                    trace_at=(1, 2, 3, 5, 10, 20))
    print("    iteration   points reached (of 1,000)")
    for it in sorted(traces):
        print(f"       {it:3d}            {traces[it]:5d}")
    F = propagate_closed_form(P, li, y[li])
    acc = float((F.argmax(1)[test] == y_test).mean())
    print()
    print(f"  Equilibrium (closed-form harmonic solution): {acc:.1%} accuracy")
    print("  on the 500 held-out points. The flood follows the moons' arcs --")
    print("  the gap between them has no edges, so it is never crossed.")

    banner("DEMO 2 --- The rescue: Part 1's catastrophic label draws, revisited")
    print("  In Part 1, self-training with these SAME five random 8-label draws")
    print("  averaged 73.8% -- and one draw fell from 93.2% to 51.6%. Same data,")
    print("  same draws, label propagation instead:")
    print()
    part1_selftrain = {0: 85.4, 1: 59.6, 2: 51.6, 3: 87.0, 4: 85.2}
    print("    draw   self-training (Part 1)   label propagation")
    lp_accs = []
    for draw in range(5):
        li = draw_labels(draw)
        F = propagate_closed_form(build_transition(ALL), li, y[li])
        a = float((F.argmax(1)[test] == y_test).mean())
        lp_accs.append(a)
        print(f"      {draw}          {part1_selftrain[draw]:5.1f}%"
              f"               {a:6.1%}")
    print(f"     mean          73.8%               {np.mean(lp_accs):6.1%}")
    print()
    print("  Label placement stops mattering, because the labels' reach is set")
    print("  by the graph's shape, not by where the frontier happens to start.")
    print("  (Part 1 figures are that script's published output, for reference.)")

    banner("DEMO 3 --- The graph IS the model: three ways to break it")
    print("  Same algorithm, same labels (draw 0) -- only the graph changes:")
    print()
    li = draw_labels(0)
    configs = [
        ("sharp local graph (k=7, sigma=0.05)", dict(sigma=0.05, knn=7)),
        ("blurry edges     (k=7, sigma=0.5)", dict(sigma=0.5, knn=7)),
        ("gap erased  (full graph, sigma=1.5)", dict(sigma=1.5, knn=None)),
        ("fragmented       (k=2, sigma=0.05)", dict(sigma=0.05, knn=2)),
    ]
    print("    graph                                  accuracy   unreached")
    for tag, kw in configs:
        P = build_transition(ALL, **kw)
        Fi, _ = propagate_iterative(P, li, y[li], iters=2000)
        acc = float((Fi.argmax(1)[test] == y_test).mean())
        unreached = int((Fi.sum(axis=1) <= 1e-12).sum())
        print(f"    {tag:38s}  {acc:6.1%}      {unreached:4d}")
    print()
    print("  Blur the weights and cross-gap edges count like on-moon ones (86%).")
    print("  Connect everything and the gap vanishes entirely -- a coin flip.")
    print("  Starve the graph (k=2) and it fragments: 618 points are never")
    print("  reached by any label at all. Label propagation has no opinion of")
    print("  its own; it faithfully diffuses over whatever graph you built.")


if __name__ == "__main__":
    main()
