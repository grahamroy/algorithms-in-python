"""
graph_methods.py --- companion code for "Graph-Based Methods"
(Semi-Supervised Learning, Part 6).

Part 5 built ONE graph algorithm: harmonic label propagation. This article is
the family view, and the family has a single central object -- the GRAPH
LAPLACIAN, L = D - W (degree matrix minus weight matrix). Its magic is one
identity:

    f' L f  =  1/2 * sum_ij  w_ij (f_i - f_j)^2

For any labelling f of the nodes, f' L f is its ROUGHNESS: how much connected
neighbours disagree. Smooth labellings (neighbours agree) score low; jagged
ones score high. That single number unifies the family:

  - Harmonic propagation (Part 5): minimise roughness with the known labels
    CLAMPED hard.
  - Label spreading (Zhou et al., 2004): minimise roughness PLUS a fidelity
    penalty for bending the known labels -- a soft clamp, one dial (alpha)
    between trusting the graph and trusting the labels.
  - Spectral methods: with NO labels at all, the smoothest non-trivial
    labelling is an eigenvector of L -- the Fiedler vector -- and it finds
    the classes on its own.

Demonstrates:
  1. The meter: the true labelling scores 58; flip just TWO of the 1,000
     points and the meter nearly doubles; a random labelling scores 3,510.
  2. The spectrum knows: the Fiedler vector's sign splits the moons at 96.7%
     -- using ZERO labels.
  3. The audit: poison 4 of 40 labels and ask, for each labelled point, what
     the graph would have told it (leave-one-out). The poisoned labels rank
     at the top of the disagreement list -- the graph proofreads your labels.

Everything is plain NumPy. Dependencies: numpy. Runs in ~15-30 seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
SIGMA = 0.05
KNN = 7
ALPHA = 0.9


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Two moons and the k-NN Gaussian graph (exactly as in Part 5).
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


def build_W(X, sigma=SIGMA, knn=KNN):
    n = len(X)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    W = np.exp(-d2 / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    keep = np.zeros_like(W, dtype=bool)
    nn = np.argsort(d2 + np.eye(n) * 1e9, axis=1)[:, :knn]
    keep[np.repeat(np.arange(n), knn), nn.ravel()] = True
    keep |= keep.T
    return W * keep


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_moons(250, noise=0.15, rng=rng)
    X_test, y_test = make_moons(250, noise=0.15, rng=rng)
    ALL = np.concatenate([X, X_test])
    yALL = np.concatenate([y, y_test])
    n = len(ALL)

    W = build_W(ALL)
    L = np.diag(W.sum(axis=1)) - W

    banner("DEMO 1 --- The Laplacian is a smoothness meter")
    print("  For any labelling f (+1 / -1 per point):")
    print("      f' L f  =  1/2 * sum of  w_ij * (f_i - f_j)^2")
    print("  -- the total disagreement across the graph's edges.")
    print()
    energy = lambda f: float(f @ L @ f)
    f_true = 2.0 * yALL - 1.0
    r2 = np.random.default_rng(7)
    rows = [("the true labelling", f_true)]
    for flips in (2, 10):
        f = f_true.copy()
        f[r2.choice(n, flips, replace=False)] *= -1
        rows.append((f"the truth with {flips:2d} points flipped", f))
    rows.append(("a random labelling", r2.choice([-1.0, 1.0], n)))
    for name, f in rows:
        print(f"    {name:34s}: {energy(f):8.1f}")
    print()
    print("  The truth is the smooth labelling. Flip TWO points of 1,000 and")
    print("  the meter nearly doubles; label at random and it reads 60x higher.")
    print("  Every method in this family is some way of driving this number")
    print("  down without contradicting what you know.")

    banner("DEMO 2 --- The spectrum knows: classes from ZERO labels")
    print("  Minimising f' L f over all balanced labellings (no labels held")
    print("  fixed at all) is an eigenvector problem. The smoothest non-trivial")
    print("  direction -- the FIEDLER VECTOR -- is the graph's own idea of its")
    print("  two halves:")
    print()
    d = W.sum(axis=1)
    Di = np.diag(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    Ln = np.eye(n) - Di @ W @ Di          # normalised Laplacian
    evals, evecs = np.linalg.eigh(Ln)
    print(f"  Smallest eigenvalues: "
          + ", ".join(f"{v:.4f}" for v in evals[:4]))
    fiedler = evecs[:, 1]
    pred = (fiedler > 0).astype(int)
    acc = max(float((pred == yALL).mean()), float(((1 - pred) == yALL).mean()))
    print(f"  Sign of the Fiedler vector vs the true classes: {acc:.1%}")
    print()
    print("  96.7% of the class structure was in the graph before a single")
    print("  label arrived -- this is spectral clustering, met in the")
    print("  unsupervised track, resurfacing as the zero-label end of the")
    print("  same smoothness family.")

    banner("DEMO 3 --- The audit: the graph proofreads your labels")
    print("  Label spreading (soft clamp): minimise roughness plus a penalty")
    print("  for bending the given labels. Its closed form makes a")
    print("  leave-one-out check cheap: for each labelled point, remove its")
    print("  clamp and ask what belief the graph would hand it.")
    print()
    print("  40 labels, of which 4 are POISONED (flipped). Rank all 40 by how")
    print("  strongly the graph disagrees with their given label:")
    print()
    S = Di @ W @ Di
    M = np.linalg.inv(np.eye(n) - ALPHA * S)
    print("    trial   ranks of the 4 poisoned labels (of 40)")
    for trial in range(3):
        d_rng = np.random.default_rng(trial)
        li = np.concatenate([
            d_rng.choice(np.where(y == 0)[0], 20, replace=False),
            d_rng.choice(np.where(y == 1)[0], 20, replace=False)])
        ylab = y[li].copy()
        flip_pos = d_rng.choice(40, 4, replace=False)
        ylab[flip_pos] = 1 - ylab[flip_pos]
        Y = np.zeros((n, 2))
        Y[li] = np.eye(2)[ylab]
        F = (1 - ALPHA) * (M @ Y)
        scores = np.zeros(40)
        for i, ix in enumerate(li):
            loo = F[ix] - (1 - ALPHA) * M[ix, ix] * Y[ix]   # remove own clamp
            p = loo / max(loo.sum(), 1e-12)
            scores[i] = p[1 - ylab[i]] - p[ylab[i]]
        order = np.argsort(-scores)
        ranks = sorted(int(np.where(order == f)[0][0]) + 1 for f in flip_pos)
        print(f"      {trial}          {ranks}")
    print()
    print("  All twelve poisoned labels across the three trials rank in the")
    print("  top 7 of 40; trial 2 is a perfect 1-2-3-4. The stragglers")
    print("  (ranks 6-7) are poisoned seeds sitting NEAR EACH OTHER -- ")
    print("  conspirators vouching for one another, which is exactly how label")
    print("  noise hides in real datasets too. Review the top of this list")
    print("  before you trust your labels: the graph audits for free.")


if __name__ == "__main__":
    main()
