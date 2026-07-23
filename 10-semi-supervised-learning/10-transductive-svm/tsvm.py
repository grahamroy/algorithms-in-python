"""
tsvm.py --- companion code for "Transductive SVM"
(Semi-Supervised Learning, Part 10).

The oldest idea in this track, stated as pure geometry (Vapnik, 1998):
place the WIDEST EMPTY STREET. A supervised SVM maximises the margin over
the labelled points only -- with 8 labels the street happily runs through
hundreds of unlabelled points it cannot see. The transductive SVM adds one
term and one constraint:

    min  lam/2 ||w||^2  +  C_l * hinge(labelled)  +  C_u * hat(unlabelled)

    hat(x) = max(0, 1 - |f(x)|)   -- punish any unlabelled point INSIDE
                                     the street, whichever side it is on
    balance: mean f over the pool is pinned to the labelled class ratio
             (else the street drifts to infinity and calls everyone one
             class -- the classic collapse)

The hat makes the objective NON-CONVEX (two valleys per point: push it
left or push it right), which is the whole character of the method:
  * anneal C_u from tiny to full strength, so the labelled street forms
    first and the unlabelled evidence bends it gradually;
  * run several restarts and keep the lowest OBJECTIVE -- selection needs
    no labels, because an emptier street is visibly emptier.

The model is an RBF kernel machine via random Fourier features (D=200
cosines approximate the kernel; each restart redraws the approximation,
giving the non-convex search genuinely different starting geometry).

Demonstrates (two moons, Part 1's exact data and 8-label draws):
  1. The supervised street: hinge on 8 labels only -- mean 83.6%, with
     a third of the unlabelled pool standing INSIDE the margin.
  2. The transductive street: hat + anneal + balance + restarts --
     mean 97.4%, every draw 93.6%+, street occupancy down to a few
     percent. The boundary moved into the gap because the gap is where
     an empty street fits.
  3. The hat is not a bowl: draw 0's six restarts land at 61-98%
     accuracy -- and the objective value ranks them almost perfectly,
     while unpinning the balance lets the street's split drift.

Everything is plain NumPy. Dependencies: numpy. Runs in about half a
minute.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
SIGMA = 0.35        # RBF kernel width
D_FEAT = 200        # random Fourier features
LAM = 1e-3          # ||w||^2 weight
C_L = 5.0           # labelled hinge weight
C_U = 1.0           # unlabelled hat weight (annealed up to this)
EPOCHS = 6000
STAGES = 8          # annealing: C_U * 2^(stage - STAGES + 1)
RESTARTS = 6
LR = 5e-2


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
# The kernel machine: f(x) = w . phi(x) + b, with phi a random Fourier
# approximation of the RBF kernel. Redrawing phi = a fresh restart geometry.
# ---------------------------------------------------------------------------

class RFF:
    def __init__(self, sigma=SIGMA, D=D_FEAT, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, 1.0 / sigma, (2, D))
        self.u = r.uniform(0, 2 * np.pi, D)
        self.D = D

    def __call__(self, X):
        return np.sqrt(2.0 / self.D) * np.cos(X @ self.W + self.u)


def train_svm(P_lab, y_lab, P_unl, cu_max=C_U, anneal=True, balance=0.0,
              epochs=EPOCHS, lam=LAM, C_l=C_L, lr=LR):
    """Subgradient descent (Adam) on the (T)SVM objective. cu_max = 0 gives
    the plain supervised SVM. With balance set, the bias is not a free
    parameter: each step it is pinned so the pool's mean output equals the
    labelled class ratio -- the balance constraint, enforced exactly.
    balance=None frees the bias (DEMO 3 shows why you should not)."""
    pinned = balance is not None
    w = np.zeros(P_lab.shape[1])
    b = 0.0
    m, v = np.zeros_like(w), np.zeros_like(w)
    mb = vb = 0.0
    t = 0
    mu_u = P_unl.mean(axis=0) if pinned else np.zeros(P_lab.shape[1])
    for ep in range(epochs):
        if anneal and cu_max > 0:
            stage = min(int(ep / epochs * STAGES), STAGES - 1)
            cu = cu_max * 2.0 ** (stage - STAGES + 1)
        else:
            cu = cu_max
        if pinned:
            b = balance - (P_unl @ w).mean()
        f_lab = P_lab @ w + b
        act = (y_lab * f_lab) < 1                     # violating the margin
        g = lam * w - C_l * (y_lab[act, None]
                             * (P_lab[act] - mu_u)).sum(0) / len(P_lab)
        gb = -C_l * y_lab[act].sum() / len(P_lab)
        f_u = P_unl @ w + b
        inside = np.abs(f_u) < 1                      # standing in the street
        s = np.sign(f_u[inside])
        g -= cu * (s[:, None] * (P_unl[inside] - mu_u)).sum(0) / len(P_unl)
        gb -= cu * s.sum() / len(P_unl)
        t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if not pinned:
            mb = b1 * mb + (1 - b1) * gb
            vb = b2 * vb + (1 - b2) * gb * gb
            b -= lr * (mb / (1 - b1 ** t)) / (
                np.sqrt(vb / (1 - b2 ** t)) + eps)
    if pinned:
        b = balance - (P_unl @ w).mean()
    return w, b


def objective(w, b, P_lab, y_lab, P_unl, cu=C_U):
    """The number the restarts compete on. No test labels involved."""
    hinge = np.maximum(0, 1 - y_lab * (P_lab @ w + b)).mean()
    hat = np.maximum(0, 1 - np.abs(P_unl @ w + b)).mean()
    return LAM / 2 * w @ w + C_L * hinge + cu * hat


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_moons(250, noise=0.15, rng=rng)
    X_test, y_test = make_moons(250, noise=0.15, rng=rng)
    Y = 2 * y - 1
    Y_test = 2 * y_test - 1

    def draw_labels(draw):
        d_rng = np.random.default_rng(draw)
        return np.concatenate([
            d_rng.choice(np.where(y == 0)[0], 4, replace=False),
            d_rng.choice(np.where(y == 1)[0], 4, replace=False)])

    banner("DEMO 1 --- The supervised street: margin over 8 points")
    print("  An RBF-kernel SVM (random Fourier features) trained on the 8")
    print("  labels alone. It finds the widest street between the points it")
    print("  can see -- and parks it on top of the points it cannot.")
    print()
    phi0 = RFF(seed=0)
    P0, P0_test = phi0(X), phi0(X_test)
    sup_accs, sup_ins = [], []
    for draw in range(5):
        li = draw_labels(draw)
        w, b = train_svm(P0[li], Y[li], P0, cu_max=0.0, anneal=False)
        sup_accs.append(float(((P0_test @ w + b > 0) == (Y_test > 0)).mean()))
        sup_ins.append(float((np.abs(P0 @ w + b) < 1).mean()))
    print("    draw:    " + "   ".join(f"{a:5.1%}" for a in sup_accs)
          + f"    mean {np.mean(sup_accs):.1%}")
    print("    inside:  " + "   ".join(f"{a:5.1%}" for a in sup_ins)
          + f"    mean {np.mean(sup_ins):.1%}")
    print()
    print("  ('inside' = fraction of the 500-point pool standing in the")
    print("  street, |f| < 1.) A third of the data lives inside a margin")
    print("  that is supposed to be empty. The labels cannot object -- they")
    print("  are all outside it. Only the unlabelled points know.")

    banner("DEMO 2 --- The transductive street: everyone must clear it")
    print("  Add the hat loss max(0, 1-|f|) over all 500 points, anneal its")
    print("  weight from C_u/128 up to C_u, pin the class balance through")
    print("  the bias, and run 6 restarts (each redraws the random feature")
    print("  map), keeping the lowest OBJECTIVE -- never touching a label.")
    print()
    print("    draw   supervised   transductive     chosen restart")
    t_accs, t_ins, best_models = [], [], []
    restart_log = []
    for draw in range(5):
        li = draw_labels(draw)
        best = None
        rows = []
        for ps in range(RESTARTS):
            phi = RFF(seed=ps)
            P, P_test = phi(X), phi(X_test)
            w, b = train_svm(P[li], Y[li], P, balance=float(Y[li].mean()))
            J = objective(w, b, P[li], Y[li], P)
            acc = float(((P_test @ w + b > 0) == (Y_test > 0)).mean())
            ins = float((np.abs(P @ w + b) < 1).mean())
            rows.append((ps, J, acc))
            if best is None or J < best[0]:
                best = (J, acc, ps, ins, w, b)
        restart_log.append(rows)
        t_accs.append(best[1])
        t_ins.append(best[3])
        best_models.append(best)
        print(f"      {draw}       {sup_accs[draw]:6.1%}       {best[1]:6.1%}"
              f"         J = {best[0]:.4f}")
    print(f"     mean      {np.mean(sup_accs):6.1%}       "
          f"{np.mean(t_accs):6.1%}")
    print()
    print("    street occupancy of the chosen solutions: "
          + "  ".join(f"{a:.1%}" for a in t_ins))
    print()
    print("  Every draw lands at 93.6% or better. Nothing was pseudo-")
    print("  labelled, no graph was built, no story was told: the boundary")
    print("  moved into the gap because the gap is the only place a wide")
    print("  EMPTY street fits. The margin -- the same idea that powered")
    print("  the supervised SVM -- became the semi-supervised engine.")

    banner("DEMO 3 --- The hat is not a bowl: six restarts, three streets")
    print("  The hat loss can push each point to either side: the objective")
    print("  has VALLEYS. Draw 0's six restarts, sorted by objective:")
    print()
    print("    objective J     test accuracy")
    for ps, J, acc in sorted(restart_log[0], key=lambda r: r[1]):
        mark = "   <-- chosen" if ps == best_models[0][2] else ""
        print(f"      {J:.4f}          {acc:5.1%}{mark}")
    print()
    accs0 = [acc for _, _, acc in restart_log[0]]
    print(f"  Three basins: the true gap (~98%), a street that shears one")
    print(f"  moon (~76%), a worse one (~62-66%). Trust a single run and")
    print(f"  you inherit its valley: the six runs average "
          f"{np.mean(accs0):.1%}, spanning")
    print(f"  {min(accs0):.1%} to {max(accs0):.1%}. Selection by the "
          f"objective -- free, label-less --")
    print(f"  recovers {best_models[0][1]:.1%}: emptier streets simply "
          f"score lower.")
    print()
    li = draw_labels(0)
    nb_accs, nb_fracs = [], []
    for ps in range(RESTARTS):
        phi = RFF(seed=ps)
        P, P_test = phi(X), phi(X_test)
        w, b = train_svm(P[li], Y[li], P, balance=None)
        nb_accs.append(float(((P_test @ w + b > 0) == (Y_test > 0)).mean()))
        nb_fracs.append(float(((P @ w + b) > 0).mean()))
    print(f"  Unpin the balance (bias trained freely) and the street's")
    print(f"  population drifts wherever its valley leads: across the same")
    print(f"  six restarts the pool splits "
          f"{min(nb_fracs):.0%}-{max(nb_fracs):.0%} positive (accuracy")
    print(f"  {min(nb_accs):.1%}-{max(nb_accs):.1%}). The true split is "
          f"50/50; the pin costs one")
    print(f"  line and deletes that whole axis of failure. (Joachims'")
    print(f"  annealing schedule, kept above for fidelity, measured as")
    print(f"  neutral on a problem this small -- restarts plus selection")
    print(f"  already explore what the schedule exists to protect.)")


if __name__ == "__main__":
    main()
