"""
vat.py --- companion code for "Virtual Adversarial Training (VAT)"
(Semi-Supervised Learning, Part 8).

Every method in this track so far manufactured its safety from OTHER things:
more judges, more views, a graph, a generative story. VAT (Miyato et al.,
2018) needs none of them. It writes the smoothness assumption directly into
the LOSS FUNCTION:

    For every point x -- labelled or not -- find the small perturbation
    r_adv (with ||r|| <= eps) that MOST changes the model's prediction,
    and penalise that change:

        loss = CE(labelled)  +  alpha * KL( p(y|x)  ||  p(y|x + r_adv) )

"Virtual" because the adversary attacks the model's own current prediction,
not a true label -- which is exactly why unlabelled data can join in. The
geometric consequence: a decision boundary that passes near any data point is
expensive (a tiny nudge flips predictions there), so training EVICTS the
boundary from wherever data lives -- into the low-density gap.

Finding r_adv is one step of POWER ITERATION: nudge x by a tiny random
direction, backpropagate the KL to the INPUT, and normalise -- the dominant
direction of the local curvature. Two extra passes per batch.

(At scale, the literature pairs VAT with entropy minimisation, because a big
model can satisfy smoothness by refusing to commit off the labelled set. On
this problem the cross-entropy on 8 labelled points supplies the commitment,
and pure VAT wins outright -- measured, not assumed.)

Demonstrates (two moons, Part 1's exact data and 8-label draws):
  1. The baseline: a small MLP on 8 labels alone (mean 77.4%), boundary
     through the moons, 99% confident everywhere.
  2. The scoreboard: the same MLP + the VAT penalty on all 500 points:
     mean 95.4%, every draw at 90% or better.
  3. The eps dial: attack radius too small does nothing, too large welds
     the classes together across the gap -- a clean inverted U.

Everything (the MLP with analytic input gradients, the power iteration, the
losses) is plain NumPy. Dependencies: numpy. Runs in about half a minute.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
EPS = 0.2          # attack radius
ALPHA = 2.0        # VAT weight
EPOCHS = 1000


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
# A small MLP whose backward pass also returns the gradient AT THE INPUT --
# that input gradient is what powers the adversarial direction.
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, hidden=16, lr=1e-2, seed=0):
        rng = np.random.default_rng(seed)
        self.P = [rng.standard_normal((2, hidden)) * np.sqrt(0.5),
                  np.zeros(hidden),
                  rng.standard_normal((hidden, 2)) * np.sqrt(1 / hidden),
                  np.zeros(2)]
        self.M = [np.zeros_like(p) for p in self.P]
        self.V = [np.zeros_like(p) for p in self.P]
        self.t = 0
        self.lr = lr

    def probs(self, X):
        H = np.tanh(X @ self.P[0] + self.P[1])
        L = H @ self.P[2] + self.P[3]
        Z = L - L.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def grads(self, X, dlogits):
        """Parameter grads and INPUT grads for a given dLoss/dlogits."""
        H = np.tanh(X @ self.P[0] + self.P[1])
        gW2 = H.T @ dlogits / len(X)
        gb2 = dlogits.mean(axis=0)
        dH = dlogits @ self.P[2].T * (1 - H ** 2)
        gW1 = X.T @ dH / len(X)
        gb1 = dH.mean(axis=0)
        dX = dH @ self.P[0].T
        return [gW1, gb1, gW2, gb2], dX

    def adam(self, g):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, gi) in enumerate(zip(self.P, g)):
            self.M[i] = b1 * self.M[i] + (1 - b1) * gi
            self.V[i] = b2 * self.V[i] + (1 - b2) * gi ** 2
            p -= self.lr * (self.M[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.V[i] / (1 - b2 ** self.t)) + eps)


# ---------------------------------------------------------------------------
# The virtual adversary: one step of power iteration.
# dKL(p || q)/dlogits_q = q - p, so one backward pass to the input gives the
# direction in which the prediction changes fastest.
# ---------------------------------------------------------------------------

def vat_direction(net, X, rng, eps=EPS, xi=1e-4):
    p = net.probs(X)
    d = rng.standard_normal(X.shape)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    q = net.probs(X + xi * d)
    _, dX = net.grads(X + xi * d, q - p)
    return eps * dX / (np.linalg.norm(dX, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Training: supervised CE, plus (optionally) the VAT and entropy terms.
# ---------------------------------------------------------------------------

def train(X_lab, y_lab, X_all, epochs=EPOCHS, alpha=ALPHA, eps=EPS,
          seed=0, use_vat=True):
    net = MLP(seed=seed)
    rng = np.random.default_rng(seed + 9)
    onehot = np.eye(2)[y_lab]
    for _ in range(epochs):
        g, _ = net.grads(X_lab, net.probs(X_lab) - onehot)
        if use_vat:
            r = vat_direction(net, X_all, rng, eps=eps)
            p_all = net.probs(X_all)               # the (fixed) clean belief
            q_all = net.probs(X_all + r)           # belief under attack
            gv, _ = net.grads(X_all + r, q_all - p_all)
            g = [a + alpha * b for a, b in zip(g, gv)]
        net.adam(g)
    return net


def accuracy(net, X, y):
    return float((net.probs(X).argmax(axis=1) == y).mean())


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_moons(250, noise=0.15, rng=rng)
    X_test, y_test = make_moons(250, noise=0.15, rng=rng)

    def draw_labels(draw):
        d_rng = np.random.default_rng(draw)
        return np.concatenate([
            d_rng.choice(np.where(y == 0)[0], 4, replace=False),
            d_rng.choice(np.where(y == 1)[0], 4, replace=False)])

    banner("DEMO 1 --- The baseline: a neural network and 8 labels")
    print("  A small MLP (2 -> 16 -> 2) trained on the 8 labels alone, on")
    print("  Part 1's five random draws. With nothing constraining it, the")
    print("  network parks its boundary wherever 8 points allow -- which is")
    print("  usually straight through the moons.")
    print()
    sup_accs = []
    for draw in range(5):
        li = draw_labels(draw)
        net = train(X[li], y[li], X, use_vat=False, seed=draw)
        sup_accs.append(accuracy(net, X_test, y_test))
    print("    draw:  " + "   ".join(f"{a:5.1%}" for a in sup_accs)
          + f"    mean {np.mean(sup_accs):.1%}")

    banner("DEMO 2 --- The scoreboard: add the attack, evict the boundary")
    print("  VAT adds, for EVERY point, the penalty KL(p(x) || p(x + r_adv)),")
    print("  where r_adv is the most damaging nudge of size eps -- found by one")
    print("  power-iteration step (two extra passes). No labels involved:")
    print("  the model attacks its own beliefs.")
    print()
    print("    draw   supervised only   + VAT penalty")
    v_accs = []
    for draw in range(5):
        li = draw_labels(draw)
        v = train(X[li], y[li], X, use_vat=True, seed=draw)
        va = accuracy(v, X_test, y_test)
        v_accs.append(va)
        print(f"      {draw}        {sup_accs[draw]:6.1%}          {va:6.1%}")
    print(f"     mean       {np.mean(sup_accs):6.1%}          "
          f"{np.mean(v_accs):6.1%}")
    print()
    li = draw_labels(0)
    sup0 = train(X[li], y[li], X, use_vat=False, seed=0)
    vat0 = train(X[li], y[li], X, use_vat=True, seed=0)
    print(f"  Mean prediction confidence across the pool:")
    print(f"    supervised only: {float(sup0.probs(X).max(1).mean()):.2f}"
          f"   (99% sure, boundary through the moons)")
    print(f"    with VAT       : {float(vat0.probs(X).max(1).mean()):.2f}"
          f"   (hedges exactly where the data thins)")
    print()
    print("  Every draw lands at 90% or better -- including draw 1, the one that")
    print("  broke self-training and resisted tri-training. The unlabelled points")
    print("  never contribute a label; they contribute PLACES THE BOUNDARY MAY")
    print("  NOT GO.")

    banner("DEMO 3 --- The eps dial: how hard to shove")
    print("  eps is the attack radius -- the distance within which predictions")
    print("  must not change. Five draws per setting:")
    print()
    print("    eps     per-draw accuracies              mean")
    for eps_ in (0.05, 0.2, 0.5, 1.0, 2.0):
        accs = []
        for draw in range(5):
            li = draw_labels(draw)
            m = train(X[li], y[li], X, eps=eps_, seed=draw)
            accs.append(accuracy(m, X_test, y_test))
        row = "  ".join(f"{a:4.0%}" for a in accs)
        print(f"    {eps_:<5}   {row}      {np.mean(accs):6.1%}")
    print()
    print("  Too small (0.05) and the constraint has no reach -- the boundary")
    print("  can still hide between data points. Right-sized (0.2) and the only")
    print("  place it can live is the gap: 95.4%. Too large (2.0) and the attack")
    print("  radius SPANS the gap -- smoothness now demands the same prediction")
    print("  on both moons, and the classes weld together (one draw hits 49%).")


if __name__ == "__main__":
    main()
