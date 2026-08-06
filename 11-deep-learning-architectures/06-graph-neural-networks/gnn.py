"""
gnn.py --- companion code for "Graph Neural Networks"
(Deep Learning Architectures, Part 6).

Every architecture so far assumed the data had a SHAPE: a grid of
pixels (Part 2), a line of timesteps (Parts 3-5). Molecules, social
networks, citation webs, and road maps have neither -- they arrive as
NODES and EDGES, and the only geometry is who-is-connected-to-whom.

The graph neural network's move is MESSAGE PASSING. Each layer, every
node averages its neighbours' feature vectors and passes the result
through a small learned map -- the GCN layer (Kipf & Welling, 2017):

    H' = relu( A_hat H W ),   A_hat = D^-1/2 (A + I) D^-1/2

One layer mixes 1-hop neighbourhoods; L layers, L hops. And the family
resemblance is the section's punchline: a CNN is a GNN on the grid
graph (Part 2's kernel = messages from pixel-neighbours), and a
transformer is a GNN on the COMPLETE graph (Part 4's attention =
messages from everyone, with learned weights). The graph decides who
may look at whom.

The stage: a stochastic block model. 360 nodes in 3 hidden
communities; edges form with probability 0.06 inside a community and
0.02 across. Node features are the community signal drowned in noise
-- DELIBERATELY weak, so the edges carry what the features cannot.
15 labelled nodes. Transductive, like the whole of Section 10.

Demonstrates:
  1. What the edges know: features alone 58.3%; features + message
     passing 85.8%; the same GCN on SHUFFLED edges 42.3% -- worse than
     no edges at all. Wrong structure is not absence, it is poison.
  2. Over-smoothing -- the section's depth lesson, inverted: averaging
     is a random-walk step, and walks forget where they started. The
     raw operator peaks at 4 hops then collapses to chance; a trained
     8-layer GCN lands at 33.3% (chance) -- and residual connections
     rescue it to 87.0%. Part 5's lesson pays off a second time.
  3. Attention's cousin, honestly: a GAT (learned neighbour weights,
     Part 4's mechanism masked to the graph) LOSES to the GCN's fixed
     averaging at every label budget tried here -- attention computed
     from noisy features is noise with extra parameters. Part 2's
     rhyme: the less data, the more the architecture should know.

Everything is plain NumPy, gradients hand-derived and audited.
Dependencies: numpy. Runs in about a minute.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
K_COM = 3
N_PER = 120
P_IN = 0.06
P_OUT = 0.02
D_FEAT = 8
FEAT_NOISE = 2.5
N_LAB = 5
EPOCHS = 300


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# The stage: a stochastic block model with deliberately weak features.
# ---------------------------------------------------------------------------

def make_sbm(rng, n_lab=N_LAB):
    n = K_COM * N_PER
    y = np.repeat(np.arange(K_COM), N_PER)
    P = np.where(y[:, None] == y[None, :], P_IN, P_OUT)
    A = (rng.uniform(size=(n, n)) < P).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    cent = rng.standard_normal((K_COM, D_FEAT))
    X = cent[y] + FEAT_NOISE * rng.standard_normal((n, D_FEAT))
    lab = np.zeros(n, dtype=bool)
    for c in range(K_COM):
        lab[rng.choice(np.where(y == c)[0], n_lab, replace=False)] = True
    return A, X, y, lab


def norm_adj(A):
    Ah = A + np.eye(len(A))
    Di = 1.0 / np.sqrt(Ah.sum(1))
    return Ah * Di[:, None] * Di[None, :]


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr = params, lr
        self.M = [np.zeros_like(p) for p in params]
        self.V = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.M[i] = b1 * self.M[i] + (1 - b1) * g
            self.V[i] = b2 * self.V[i] + (1 - b2) * g * g
            p -= self.lr * (self.M[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.V[i] / (1 - b2 ** self.t)) + eps)


def softmax(L):
    Z = L - L.max(-1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# The GCN: message passing with fixed, degree-normalised weights.
# ---------------------------------------------------------------------------

class GCN:
    def __init__(self, Ah, d_in, hidden, n_cls, layers=2, seed=0, lr=1e-2,
                 use_resid=False):
        r = np.random.default_rng(seed)
        self.Ah = Ah
        self.L = layers
        self.use_resid = use_resid
        dims = [d_in] + [hidden] * (layers - 1)
        self.Ws = [r.standard_normal((dims[i], hidden))
                   * np.sqrt(2.0 / dims[i]) for i in range(layers - 1)]
        self.Wout = r.standard_normal((dims[-1], n_cls)) * np.sqrt(
            1.0 / dims[-1])
        self.params = self.Ws + [self.Wout]
        self.opt = Adam(self.params, lr)

    def forward(self, X):
        H = X
        caches = []
        for W in self.Ws:
            AH = self.Ah @ H                    # hear the neighbours
            Z = AH @ W                          # think about it
            Hn = np.maximum(0, Z)
            if self.use_resid and Hn.shape == H.shape:
                Hn = Hn + H                     # ...but keep your own signal
            caches.append((H, AH, Z))
            H = Hn
        AH = self.Ah @ H
        caches.append((H, AH, None))
        return AH @ self.Wout, caches

    def probs(self, X):
        return softmax(self.forward(X)[0])

    def loss(self, X, y, mask):
        p = self.probs(X)
        return float(-np.log(p[mask, y[mask]] + 1e-300).mean())

    def grads_(self, X, y, mask):
        logits, caches = self.forward(X)
        P = softmax(logits)
        d = P.copy()
        d[np.arange(len(y)), y] -= 1
        d = d * mask[:, None] / mask.sum()
        grads = [np.zeros_like(p) for p in self.params]
        H_last, AH_last, _ = caches[-1]
        grads[-1] = AH_last.T @ d
        dH = self.Ah.T @ (d @ self.Wout.T)
        for i in range(self.L - 2, -1, -1):
            H, AH, Z = caches[i]
            dZ = dH * (Z > 0)
            grads[i] = AH.T @ dZ
            dH_new = self.Ah.T @ (dZ @ self.Ws[i].T)
            if self.use_resid and H.shape == dH.shape:
                dH_new = dH_new + dH
            dH = dH_new
        return grads

    def step(self, X, y, mask):
        self.opt.step(self.grads_(X, y, mask))


# ---------------------------------------------------------------------------
# Features-only baseline: Part 1's MLP, blind to the edges.
# ---------------------------------------------------------------------------

class MLPBase:
    def __init__(self, d_in, hidden, n_cls, seed=0, lr=1e-2):
        r = np.random.default_rng(seed)
        self.W1 = r.standard_normal((d_in, hidden)) * np.sqrt(2.0 / d_in)
        self.W2 = r.standard_normal((hidden, n_cls)) * np.sqrt(1.0 / hidden)
        self.params = [self.W1, self.W2]
        self.opt = Adam(self.params, lr)

    def probs(self, X):
        return softmax(np.maximum(0, X @ self.W1) @ self.W2)

    def loss(self, X, y, mask):
        p = self.probs(X)
        return float(-np.log(p[mask, y[mask]] + 1e-300).mean())

    def step(self, X, y, mask):
        H = np.maximum(0, X @ self.W1)
        P = softmax(H @ self.W2)
        d = P.copy()
        d[np.arange(len(y)), y] -= 1
        d = d * mask[:, None] / mask.sum()
        dH = (d @ self.W2.T) * (H > 0)
        self.opt.step([X.T @ dH, H.T @ d])


# ---------------------------------------------------------------------------
# The GAT: Part 4's attention, masked to the graph's edges.
# ---------------------------------------------------------------------------

class GAT:
    def __init__(self, A, d_in, hidden, n_cls, seed=0, lr=1e-2):
        r = np.random.default_rng(seed)
        self.Mask = (A + np.eye(len(A))) > 0
        self.W1 = r.standard_normal((d_in, hidden)) * np.sqrt(2.0 / d_in)
        self.a1 = r.standard_normal(hidden) * 0.1
        self.a2 = r.standard_normal(hidden) * 0.1
        self.W2 = r.standard_normal((hidden, n_cls)) * np.sqrt(1.0 / hidden)
        self.params = [self.W1, self.a1, self.a2, self.W2]
        self.opt = Adam(self.params, lr)

    @staticmethod
    def lrelu(z):
        return np.where(z > 0, z, 0.2 * z)

    @staticmethod
    def dlrelu(z):
        return np.where(z > 0, 1.0, 0.2)

    def forward(self, X, cache=False):
        H1 = X @ self.W1
        Zpre = (H1 @ self.a1)[:, None] + (H1 @ self.a2)[None, :]
        E = np.where(self.Mask, self.lrelu(Zpre), -1e30)
        Al = softmax(E)                        # who to listen to -- learned
        G = np.maximum(0, Al @ H1)
        logits = G @ self.W2
        if cache:
            return logits, (H1, Zpre, Al, G)
        return logits

    def probs(self, X):
        return softmax(self.forward(X))

    def loss(self, X, y, mask):
        p = self.probs(X)
        return float(-np.log(p[mask, y[mask]] + 1e-300).mean())

    def grads_(self, X, y, mask):
        logits, (H1, Zpre, Al, G) = self.forward(X, cache=True)
        P = softmax(logits)
        d = P.copy()
        d[np.arange(len(y)), y] -= 1
        d = d * mask[:, None] / mask.sum()
        gW2 = G.T @ d
        dG = d @ self.W2.T
        dAH = dG * ((Al @ H1) > 0)
        dAl = dAH @ H1.T
        dH1 = Al.T @ dAH
        dE = Al * (dAl - (Al * dAl).sum(-1, keepdims=True))
        dZ = dE * self.dlrelu(Zpre) * self.Mask
        dH1 += np.outer(dZ.sum(1), self.a1) + np.outer(dZ.sum(0), self.a2)
        return [X.T @ dH1, H1.T @ dZ.sum(1), H1.T @ dZ.sum(0), gW2]

    def step(self, X, y, mask):
        self.opt.step(self.grads_(X, y, mask))


def fit(net, X, y, lab, epochs=EPOCHS):
    for _ in range(epochs):
        net.step(X, y, lab)
    return net


def accuracy(net, X, y, lab):
    return float((net.probs(X).argmax(1)[~lab] == y[~lab]).mean())


def gradcheck(net, X, y, lab, n_per=20, seed=5):
    g = net.grads_(X, y, lab)
    r = np.random.default_rng(seed)
    worst = 0.0
    for P_, G in zip(net.params, g):
        flat, gflat = P_.reshape(-1), G.reshape(-1)
        for ix in r.choice(len(flat), min(n_per, len(flat)), replace=False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp = net.loss(X, y, lab)
            flat[ix] = old - 1e-6
            lm = net.loss(X, y, lab)
            flat[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
    return worst


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(0)
    A, X, y, lab = make_sbm(rng)
    Ah = norm_adj(A)
    print(f"  [audit] GCN backward vs central differences: "
          f"{gradcheck(GCN(Ah, D_FEAT, 6, K_COM, seed=1), X, y, lab):.2e}")
    print(f"  [audit] GAT backward vs central differences: "
          f"{gradcheck(GAT(A, D_FEAT, 6, K_COM, seed=1), X, y, lab):.2e}")

    banner("DEMO 1 --- What the edges know")
    print(f"  {len(A)} nodes, {int(A.sum() / 2)} edges, 3 hidden communities,"
          f" {int(lab.sum())} labels.")
    print("  Features are the community signal drowned in noise -- weak on")
    print("  purpose. Three models, same budget:")
    print()
    mlp = fit(MLPBase(D_FEAT, 16, K_COM, seed=0), X, y, lab)
    gcn = fit(GCN(Ah, D_FEAT, 16, K_COM, layers=2, seed=0), X, y, lab)
    perm = np.random.default_rng(7).permutation(len(A))
    gcn_sh = fit(GCN(norm_adj(A[perm][:, perm]), D_FEAT, 16, K_COM,
                     layers=2, seed=0), X, y, lab)
    print(f"    MLP, features only          : {accuracy(mlp, X, y, lab):6.1%}")
    print(f"    GCN, features + edges       : {accuracy(gcn, X, y, lab):6.1%}")
    print(f"    GCN, edges SHUFFLED         : {accuracy(gcn_sh, X, y, lab):6.1%}")
    print()
    print("  The edges are worth 27 points: message passing turns 'my")
    print("  features are noisy' into 'my NEIGHBOURHOOD's average is not'.")
    print("  And the third row is the sharp one -- shuffled edges score")
    print("  BELOW the no-edge baseline. Wrong structure is not absence of")
    print("  signal; it is poison, faithfully propagated.")

    banner("DEMO 2 --- Over-smoothing: the depth lesson, inverted")
    print("  Message passing is a random-walk step, and walks forget where")
    print("  they started. Propagate the raw features L hops (no training),")
    print("  then fit a small classifier:")
    print()
    H = X.copy()
    probe = []
    for L in range(0, 33):
        if L in (0, 1, 2, 4, 8, 16, 32):
            m = fit(MLPBase(D_FEAT, 16, K_COM, seed=0), H, y, lab)
            probe.append((L, accuracy(m, H, y, lab)))
        H = Ah @ H
    print("    hops L : " + "   ".join(f"{l:3d}" for l, _ in probe))
    print("    acc    : " + "   ".join(f"{a:.0%}" for _, a in probe))
    print()
    print("  A clean rise -- more hops, more denoising -- then a collapse")
    print("  to chance: past the mixing time, every node's summary")
    print("  converges to the same stationary blur. Training does not")
    print("  save the plain architecture, and Part 5's fix does:")
    print()
    print("    depth    plain GCN    + residuals")
    for L in (2, 4, 8, 16):
        g = fit(GCN(Ah, D_FEAT, 16, K_COM, layers=L, seed=0), X, y, lab)
        g2 = fit(GCN(Ah, D_FEAT, 16, K_COM, layers=L, seed=0,
                     use_resid=True), X, y, lab)
        print(f"      {L:2d}       {accuracy(g, X, y, lab):6.1%}"
              f"       {accuracy(g2, X, y, lab):6.1%}")
    print()
    print("  Eight plain layers: 33.3% -- exactly chance, reached by honest")
    print("  depth. In this architecture depth does not compose; it ERASES.")
    print("  The residual stream keeps each node's own signal alive")
    print("  alongside the neighbourhood blur, and 87% survives sixteen")
    print("  layers.")

    banner("DEMO 3 --- Attention's cousin, and an honest defeat")
    print("  A GAT scores every edge from its endpoints' features --")
    print("  Part 4's attention, masked to the graph -- instead of the")
    print("  GCN's fixed degree-normalised average. More flexible,")
    print("  strictly. Measured, across label budgets:")
    print()
    print("    labels/class    GCN      GAT")
    for n_lab in (5, 10, 20):
        r2 = np.random.default_rng(0)
        A2, X2, y2, lab2 = make_sbm(r2, n_lab=n_lab)
        g = fit(GCN(norm_adj(A2), D_FEAT, 16, K_COM, layers=2, seed=0),
                X2, y2, lab2)
        t = fit(GAT(A2, D_FEAT, 16, K_COM, seed=0), X2, y2, lab2,
                epochs=2 * EPOCHS)
        print(f"        {n_lab:2d}        {accuracy(g, X2, y2, lab2):6.1%}"
              f"   {accuracy(t, X2, y2, lab2):6.1%}")
    print()
    print("  The GCN wins every row. No trick: attention weights computed")
    print("  from noise-drowned features are noise with extra parameters,")
    print("  and at these label counts the extra parameters just overfit.")
    print("  Part 2's rhyme, third verse: the less data you have, the more")
    print("  your architecture should already know. (GAT earns its keep on")
    print("  feature-rich graphs -- citation networks with thousands of")
    print("  informative dimensions -- where an edge CAN be judged.)")
    print()
    print("  And the family tree, in one breath: a CNN is a GNN on the")
    print("  grid graph. A transformer is a GNN on the COMPLETE graph.")
    print("  Message passing is the general case; the last five articles")
    print("  were special graphs all along.")


if __name__ == "__main__":
    main()
