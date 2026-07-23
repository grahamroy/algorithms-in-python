"""
active_learning.py --- companion code for "Active Learning"
(Semi-Supervised Learning, Part 11 -- the track finale).

Ten methods in this track accepted their 8 labels as GIVEN -- a random
draw's worth of luck. The whole scoreboard's variance came from WHICH
eight: draw 1 broke self-training at 59.6%, draw 0 capped VAT at 90.8%.
Active learning asks the only question left:

    if you may CHOOSE what gets labelled, which eight points buy the most?

The pool-based loop: train on what you have, let the MODEL nominate the
next point to label, pay the oracle, repeat. The classic nomination
strategies, all implemented here:

  * uncertainty sampling -- query where the current model is least sure
    (here: smallest margin between the two label beliefs);
  * information density  -- uncertainty x local density: be unsure AND be
    somewhere representative (Settles & Craven, 2008);
  * farthest-first       -- pure coverage, no model opinion: query the
    point farthest from everything labelled so far;
  * random               -- the control every strategy must beat.

The learner throughout is Part 5's label propagation (same graph, same
sigma=0.05, k=7, same closed-form harmonic solve) -- the track's rescue
method, chosen so the only variable is WHICH labels, never the learner.

Demonstrates (two moons, Part 1's exact data pipeline):
  1. The label lottery: 120 random 8-label draws through the SAME
     learner: median 98.4%, minimum 72.2%, 16 draws below 90%. Random
     labelling has a fat, silent tail.
  2. The race: from 2 labels to 8, strategies choosing every query.
     Information density reaches ~98.5% by SIX labels; at eight its
     WORST seed beats random's average. Active learning buys the floor,
     not the ceiling.
  3. Where the strategies point: uncertainty interviews hermits (mean
     queried density 0.26) and pays for it early -- the cold-start
     crossover, measured; density-weighting fixes it.

Everything is plain NumPy. Dependencies: numpy. Runs in about a minute
and a half.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
SIGMA = 0.05        # Part 5's graph, verbatim
KNN = 7
N_POOL = 500
LOTTERY = 120       # random draws audited in DEMO 1
SEEDS = 12          # repeats per strategy in DEMO 2
BUDGET = 8          # the track's label budget


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Part 1's exact data pipeline
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
# Part 5's learner: the kNN-Gaussian graph and the harmonic solution.
# ---------------------------------------------------------------------------

def build_graph(ALL):
    n = len(ALL)
    d2 = ((ALL[:, None, :] - ALL[None, :, :]) ** 2).sum(-1)
    W = np.exp(-d2 / (2 * SIGMA ** 2))
    np.fill_diagonal(W, 0)
    keep = np.zeros_like(W, dtype=bool)
    nn = np.argsort(d2 + np.eye(n) * 1e9, axis=1)[:, :KNN]
    keep[np.repeat(np.arange(n), KNN), nn.ravel()] = True
    keep |= keep.T
    W = W * keep
    P = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-300)
    rho = W.sum(axis=1)                     # local similarity mass
    return P, rho / rho.max()


def propagate(P, lab_idx, y_lab):
    """Closed-form harmonic solution, labels clamped (Part 5)."""
    n = len(P)
    unl = np.setdiff1d(np.arange(n), lab_idx)
    Y_l = np.eye(2)[y_lab]
    F = np.zeros((n, 2))
    F[lab_idx] = Y_l
    F[unl] = np.linalg.solve(np.eye(len(unl)) - P[np.ix_(unl, unl)],
                             P[np.ix_(unl, lab_idx)] @ Y_l)
    return F


# ---------------------------------------------------------------------------
# The active-learning loop: one nomination strategy, one seed pair.
# ---------------------------------------------------------------------------

def run_strategy(strategy, seed, X, y, P, rho, y_test):
    r = np.random.default_rng(5000 + seed)
    li = [int(r.choice(np.where(y == 0)[0])),
          int(r.choice(np.where(y == 1)[0]))]        # one label per class
    curve, qdens = [], []
    for step in range(BUDGET - 1):
        F = propagate(P, np.array(li), y[np.array(li)])
        curve.append(float((F.argmax(1)[N_POOL:] == y_test).mean()))
        if len(li) == BUDGET:
            break
        cand = np.setdiff1d(np.arange(N_POOL), li)
        margin = np.abs(F[cand, 0] - F[cand, 1])
        tie = 1e-9 * r.random(len(cand))             # seeded tie-break
        if strategy == "random":
            q = int(r.choice(cand))
        elif strategy == "uncertainty":
            q = int(cand[np.argmax((1 - margin) + tie)])
        elif strategy == "info-density":
            q = int(cand[np.argmax((1 - margin) * rho[cand] + tie)])
        elif strategy == "farthest-first":
            dmin = ((X[cand][:, None, :]
                     - X[np.array(li)][None, :, :]) ** 2).sum(-1).min(1)
            q = int(cand[np.argmax(dmin + tie)])
        qdens.append(float(rho[q]))
        li.append(q)
    return curve, qdens


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_moons(250, noise=0.15, rng=rng)
    X_test, y_test = make_moons(250, noise=0.15, rng=rng)
    ALL = np.concatenate([X, X_test])
    P, rho = build_graph(ALL)
    rho_pool = rho[:N_POOL]

    banner("DEMO 1 --- The label lottery: what the whole track lived with")
    print(f"  {LOTTERY} random 8-label draws (4 per class), each solved by")
    print("  Part 5's label propagation -- the track's most placement-proof")
    print("  method. Same learner every time; only the LUCK changes.")
    print()
    accs = []
    for i in range(LOTTERY):
        d = np.random.default_rng(1000 + i)
        li = np.concatenate([
            d.choice(np.where(y == 0)[0], 4, replace=False),
            d.choice(np.where(y == 1)[0], 4, replace=False)])
        F = propagate(P, li, y[li])
        accs.append(float((F.argmax(1)[N_POOL:] == y_test).mean()))
    accs = np.array(accs)
    print(f"    minimum   5th pct   median   maximum")
    print(f"     {accs.min():5.1%}    {np.percentile(accs, 5):5.1%}"
          f"    {np.median(accs):5.1%}    {accs.max():5.1%}")
    print()
    print(f"  Draws below 95%: {(accs < 0.95).sum()} of {LOTTERY}."
          f"  Below 90%: {(accs < 0.90).sum()}.")
    print()
    print("  The median says 'solved'; the tail says 'sometimes 72%'. Eight")
    print("  random labels can all land where the flood starts on the wrong")
    print("  side of a thin spot. Part 1's five draws (97.8-98.6% here) were")
    print("  GOOD tickets -- the track never saw its own tail.")

    banner("DEMO 2 --- The race: the same budget, chosen instead of drawn")
    print(f"  Start with one label per class, let each strategy nominate the")
    print(f"  next query, pay the oracle, repeat to {BUDGET} labels. Mean over")
    print(f"  {SEEDS} starting pairs; the learner never changes.")
    print()
    strategies = ["random", "uncertainty", "info-density", "farthest-first"]
    results, qd = {}, {}
    for s in strategies:
        cs, ds = [], []
        for seed in range(SEEDS):
            c, d_ = run_strategy(s, seed, X, y, P, rho_pool, y_test)
            cs.append(c)
            ds.append(d_)
        results[s] = np.array(cs)
        qd[s] = np.array(ds)
    print("    labels:            2       4       6       8      worst at 8")
    for s in strategies:
        c = results[s]
        print(f"    {s:14s}  " + "  ".join(
            f"{c[:, k].mean():5.1%}" for k in (0, 2, 4, 6))
            + f"      {c[:, 6].min():5.1%}")
    print()
    print("  Two facts worth the whole track. Information density reaches")
    print("  ~98.5% by SIX labels -- the budget question answered: the right")
    print("  six beat eight random. And at the full budget its WORST start")
    print("  (98.4%) beats random's AVERAGE (95.8%): active learning's")
    print("  product is not a higher ceiling -- the lottery's median was")
    print("  already 98.4% -- it is a HIGHER FLOOR. It amputates the tail.")

    banner("DEMO 3 --- Where the strategies point: hermits and the square")
    print("  Mean local density of the points each strategy chose to query")
    print("  (1.0 = the densest spot on the graph):")
    print()
    for s in strategies[1:]:
        print(f"    {s:14s}  {qd[s].mean():.2f}")
    print()
    print("  Pure uncertainty interviews HERMITS: margin is smallest at")
    print("  sparse frontier points and unreached corners, so it spends")
    print("  early queries on the least representative citizens -- at 4")
    print(f"  labels it trails even random"
          f" ({results['uncertainty'][:, 2].mean():.1%} vs"
          f" {results['random'][:, 2].mean():.1%}), the classic")
    print("  cold-start. Farthest-first has the mirror vice: coverage")
    print("  without judgement (0.18 -- the sparsest picks of all, and the")
    print("  slowest start in the table). Information density asks its")
    print("  questions in the village square -- uncertain AND representative")
    print("  -- and owns both the fastest curve and the highest floor. The")
    print("  strategy is not 'ask where you are confused'; it is 'ask where")
    print("  confusion MATTERS'.")


if __name__ == "__main__":
    main()
