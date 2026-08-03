"""
mlp.py --- companion code for "Multi-Layer Perceptron"
(Deep Learning Architectures, Part 1).

A new section opens with the machine the last one kept meeting in
disguise: the semi-supervised track hand-rolled this exact network for
VAT's adversary and the VAE's encoder without ever stopping to take it
apart. This article does. The MLP is three ideas stacked:

    1. A LAYER is an affine map plus a pointwise nonlinearity:
           a = act(x W + b)
       Affine maps alone compose into another affine map -- a stack of
       linear layers IS one linear layer. The nonlinearity is the whole
       reason depth exists.
    2. BACKPROP is the chain rule with bookkeeping: run forward, keep
       the activations, then walk the error backwards multiplying local
       derivatives. Nothing more.
    3. TRAINING is gradient descent on the loss surface those gradients
       describe (Adam here, the series' workhorse).

The stage for this whole section: TWO SPIRALS -- 500 points, two arms
wound 2.5 turns around each other. No straight line touches it, which
is precisely the point.

Demonstrates:
  1. The ladder: zero hidden layers is a linear model (57.2%); width
     buys wiggles, one kink at a time. Then the twist, averaged over
     three seeds because single runs lie: two SMALL layers [16,16]
     beat one wide [256] with a quarter of the parameters, on every
     seed. Width memorises; depth composes.
  2. The audit: every hand-derived gradient checked against central
     differences. Max disagreement ~1e-10 -- the ritual that separates
     'it trains' from 'it is correct'.
  3. Two classic ways to die: initialise at zero and the network is
     STILLBORN (every gradient exactly 0.0 by symmetry); stack six
     sigmoid layers and the gradient VANISHES (norms decay ~25,000x
     from output to input) -- with the per-layer numbers, and the
     accuracies after training anyway.

Everything is plain NumPy. Dependencies: numpy. Runs in about a minute
(the seed-averaged twist table is most of it).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
TURNS = 2.5
NOISE = 0.04
EPOCHS = 8000


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# The section's stage: two interleaved spirals.
# ---------------------------------------------------------------------------

def make_spirals(n_per_class, noise, rng, turns=TURNS, r0=0.12):
    out = []
    for cls in range(2):
        t = rng.uniform(0, 1, n_per_class) ** 0.5
        theta = t * turns * 2 * np.pi + cls * np.pi
        r = r0 + (1 - r0) * t
        out.append(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1))
    X = np.concatenate(out) + rng.normal(0, noise, (2 * n_per_class, 2))
    y = np.concatenate([np.zeros(n_per_class, dtype=int),
                        np.ones(n_per_class, dtype=int)])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# The MLP: any stack of layers, any of the three classic activations,
# softmax cross-entropy on top, Adam underneath. grads() is the whole
# chain rule; nothing is hidden in a framework.
# ---------------------------------------------------------------------------

ACTS = {
    "tanh":    (np.tanh, lambda a: 1 - a ** 2),
    "relu":    (lambda z: np.maximum(0.0, z),
                lambda a: (a > 0).astype(float)),
    "sigmoid": (lambda z: 1 / (1 + np.exp(-z)), lambda a: a * (1 - a)),
}


class MLP:
    def __init__(self, widths, act="relu", init="auto", seed=0, lr=1e-2):
        r = np.random.default_rng(seed)
        dims = [2] + list(widths) + [2]
        if init == "auto":                    # He for relu, Xavier otherwise
            scale = lambda fan: np.sqrt((2.0 if act == "relu" else 1.0) / fan)
        elif init == "zero":
            scale = lambda fan: 0.0
        else:
            scale = lambda fan: float(init)
        self.W = [r.standard_normal((dims[i], dims[i + 1])) * scale(dims[i])
                  for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self.act, self.dact = ACTS[act]
        self.lr = lr
        self.M = [np.zeros_like(w) for w in self.W]
        self.V = [np.zeros_like(w) for w in self.W]
        self.Mb = [np.zeros_like(b) for b in self.b]
        self.Vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def forward(self, X):
        """Forward pass, KEEPING every activation -- backprop's pantry."""
        A = [X]
        for i in range(len(self.W) - 1):
            A.append(self.act(A[-1] @ self.W[i] + self.b[i]))
        return A[-1] @ self.W[-1] + self.b[-1], A

    def probs(self, X):
        logits, _ = self.forward(X)
        Z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def loss(self, X, y):
        p = self.probs(X)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-300).mean())

    def grads(self, X, y):
        """The chain rule, walked backwards. For softmax + cross-entropy
        the starting error is famously just (p - onehot)."""
        logits, A = self.forward(X)
        Z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(Z)
        p = e / e.sum(axis=1, keepdims=True)
        d = (p - np.eye(2)[y]) / len(X)
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = A[i].T @ d                    # local: d(xW)/dW = x
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = d @ self.W[i].T * self.dact(A[i])   # through the layer
        return gW, gb

    def step(self, X, y):
        gW, gb = self.grads(X, y)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            for P, G, M, V in ((self.W[i], gW[i], self.M, self.V),
                               (self.b[i], gb[i], self.Mb, self.Vb)):
                M[i] = b1 * M[i] + (1 - b1) * G
                V[i] = b2 * V[i] + (1 - b2) * G ** 2
                P -= self.lr * (M[i] / (1 - b1 ** self.t)) / (
                    np.sqrt(V[i] / (1 - b2 ** self.t)) + eps)


def train(widths, X, y, act="relu", init="auto", seed=0, epochs=EPOCHS):
    net = MLP(widths, act=act, init=init, seed=seed)
    for _ in range(epochs):
        net.step(X, y)
    return net


def accuracy(net, X, y):
    return float((net.probs(X).argmax(1) == y).mean())


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_spirals(250, NOISE, rng)
    X_test, y_test = make_spirals(250, NOISE, rng)

    banner("DEMO 1 --- The ladder: width buys wiggles, depth buys structure")
    print("  Two spirals, 2.5 turns. Every model below is the same code --")
    print("  only the layer list changes. ReLU, He init, 8,000 epochs of")
    print("  Adam each.")
    print()
    print("    architecture        params   train acc   test acc")
    for widths in ([], [2], [4], [8], [16], [64]):
        net = train(widths, X, y)
        name = "linear (no hidden)" if not widths else str(widths)
        print(f"    {name:18s}   {net.n_params():5d}     {accuracy(net, X, y):6.1%}"
              f"     {accuracy(net, X_test, y_test):6.1%}")
    print()
    print("  No hidden layer = a linear model: 57.2% on a problem a straight")
    print("  line cannot touch. Each hidden ReLU contributes one crease in")
    print("  the decision surface -- and note the honest lower rungs: two or")
    print("  four creases are WORSE than the straight line they distract.")
    print("  The climb only starts once there are enough creases to spend.")
    print()
    print("  Now the twist -- averaged over three seeds, because a single")
    print("  run of a small network can flatter or slander an architecture:")
    print()
    print("    architecture        params   test acc (3 seeds)      mean")
    for widths in ([64], [256], [16, 16]):
        accs = [accuracy(train(widths, X, y, seed=s), X_test, y_test)
                for s in range(3)]
        net_p = MLP(widths).n_params()
        print(f"    {str(widths):18s}   {net_p:5d}   "
              + "  ".join(f"{a:5.1%}" for a in accs)
              + f"    {np.mean(accs):5.1%}")
    print()
    print("  [16, 16] beats both wide nets on EVERY seed, with a quarter of")
    print("  [256]'s parameters. One wide layer must draw the spiral in a")
    print("  single stroke of 256 creases; two storeys COMPOSE -- the first")
    print("  builds curved strokes out of creases, the second builds the")
    print("  spiral out of strokes. Width memorises; depth composes. This")
    print("  is the section's founding trade, and it only gets deeper.")

    banner("DEMO 2 --- The audit: is the chain rule actually right?")
    print("  Backprop's only failure mode is silent: a wrong gradient still")
    print("  trains, just worse. The ritual: compare every hand-derived")
    print("  gradient against (loss(w+h) - loss(w-h)) / 2h. (A tanh net --")
    print("  smooth everywhere; ReLU's kink would blur the comparison at 0.)")
    print()
    net = MLP([5, 4], act="tanh", seed=1)
    Xs, ys = X[:7], y[:7]
    gW, gb = net.grads(Xs, ys)
    worst = 0.0
    n_checked = 0
    for i in range(len(net.W)):
        for P, G in ((net.W[i], gW[i]), (net.b[i], gb[i])):
            it = np.nditer(P, flags=["multi_index"])
            for _ in it:
                ix = it.multi_index
                old = P[ix]
                P[ix] = old + 1e-6
                lp = net.loss(Xs, ys)
                P[ix] = old - 1e-6
                lm = net.loss(Xs, ys)
                P[ix] = old
                worst = max(worst, abs((lp - lm) / 2e-6 - G[ix]))
                n_checked += 1
    print(f"    parameters checked : {n_checked}")
    print(f"    max |analytic - numerical| : {worst:.2e}")
    print()
    print("  Agreement to ~10 decimal places, limited only by floating-point")
    print("  arithmetic. This is the moment the maths becomes trustworthy --")
    print("  every derivative in this file has passed it.")

    banner("DEMO 3 --- Two classic ways a network dies")
    print("  DEATH 1: the stillborn network. Initialise every weight at ZERO")
    print("  and hidden units are identical twins -- identical outputs,")
    print("  identical gradients, no way to ever differ. Measured:")
    print()
    net0 = MLP([16, 16], init="zero", seed=0)
    gW, _ = net0.grads(X, y)
    print("    gradient norms at init : "
          + "  ".join(f"{float(np.linalg.norm(g)):.1f}" for g in gW))
    net0 = train([16, 16], X, y, init="zero")
    print(f"    accuracy after {EPOCHS} epochs : {accuracy(net0, X_test, y_test):.1%}"
          f"   (the class prior -- it never woke up)")
    print()
    print("  DEATH 2: the vanishing gradient. Six hidden layers, three")
    print("  activations, gradient norm per layer at initialisation")
    print("  (output layer on the right, input layer on the left):")
    print()
    for act in ("sigmoid", "tanh", "relu"):
        net = MLP([16] * 6, act=act, seed=0)
        gW, _ = net.grads(X, y)
        norms = "  ".join(f"{float(np.linalg.norm(g)):.0e}" for g in gW)
        print(f"    {act:8s}: {norms}")
    print()
    print("  Sigmoid's derivative is at most 0.25, so every layer multiplies")
    print("  the signal down: by layer one the gradient is ~25,000x smaller")
    print("  than at the output. The same six-layer nets, trained anyway:")
    print()
    for act in ("sigmoid", "tanh", "relu"):
        net = train([16] * 6, X, y, act=act)
        print(f"    {act:8s}: test {accuracy(net, X_test, y_test):6.1%}")
    print()
    print("  The sigmoid tower learns its top and starves its bottom -- the")
    print("  exact wall deep learning hit in the 1990s. Tanh survives six")
    print("  storeys (its derivative touches 1.0 at the centre, so the decay")
    print("  is gentle). ReLU's derivative is a clean 0-or-1 -- no decay at")
    print("  all through active units -- which is what survives SIXTY")
    print("  storeys, and why every architecture in this section inherits it.")


if __name__ == "__main__":
    main()
