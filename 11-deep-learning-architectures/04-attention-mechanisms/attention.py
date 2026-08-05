"""
attention.py --- companion code for "Attention Mechanisms"
(Deep Learning Architectures, Part 4).

Part 3 ended at a cliff: a recurrent network relates step 40 to step 1
by SURVIVING 39 hops of state, and the error signal arrives ~300x
weaker for the trip. Attention (Bahdanau et al., 2014) removes the
middleman. Step 40 does not wait for information to swim to it -- it
LOOKS AT step 1 directly:

    q = query (what am I looking for?)          -- from the reader
    K = keys  (what does each position offer?)  -- one per position
    V = values (what does each position say?)

    weights = softmax( q . K / sqrt(d_k) )      -- a soft lookup
    answer  = sum_t weights_t * V_t             -- a weighted average

Every position is ONE hop from every other. Distance stops existing as
a concept -- which this script measures three ways: accuracy flat in T
where the RNN fell off a cliff, gradient reach flat where the RNN's
decayed ~300x, and attention weights you can read like a receipt.

The honest price, both directions:
  * attention is a SET operation -- shuffle the sequence and nothing
    changes. Order must be smuggled back in (sinusoidal positional
    encodings here), and the script shows exactly what breaks without
    them: tasks that need "where" collapse; tasks that don't, don't.
  * this is Bahdanau's ORIGINAL setting: one query attending over a
    sequence, O(T) per query. Let every position query every other --
    self-attention, Part 5's subject -- and the bill is O(T^2), printed
    here as exact multiply counts.

The stage is Part 3's exact recall task: first symbol = the label,
then T distractors.

Demonstrates:
  1. The cliff, unbuilt: recall at distance 40 / 80 / 160 -- 100%
     everywhere (the RNN: chance at 40) -- and flat gradient reach.
  2. A set, not a sequence: without positional encodings the same
     model drops to 34.2% on recall (it cannot say which symbol was
     FIRST) yet scores 100% on a majority task, which needs no order.
  3. The receipt and the bill: the trained model puts 99.2% of its
     attention on position 0 -- inspectable evidence -- and the exact
     multiply counts that make full self-attention quadratic.

Everything is plain NumPy: embeddings, sinusoidal encodings, the
attention forward and its full hand-derived backward (softmax Jacobian
included), audited against central differences at startup.
Dependencies: numpy. Runs in about two minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
VOCAB = 4
D_MODEL = 24
D_K = 16
N_SEQ = 400
EPOCHS = 600
LR = 3e-3


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Part 3's exact stage, plus a task that needs no order at all.
# ---------------------------------------------------------------------------

def make_recall(n, T, rng):
    sym = rng.integers(0, VOCAB, (n, T + 1))
    y = sym[:, 0].copy()
    return np.eye(VOCAB)[sym], y


def make_majority(n, T1, rng):
    bits = rng.integers(0, 2, (n, T1))
    y = (bits.sum(1) > T1 // 2).astype(int)
    return np.eye(VOCAB)[bits], y


def sinusoidal_pe(T1, d):
    pos = np.arange(T1)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle))


class Adam:
    def __init__(self, params, lr):
        self.p = params
        self.lr = lr
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


def softmax_rows(S):
    Z = S - S.max(-1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------------
# One attention head, Bahdanau-style: the final position's vector is the
# query; every position offers a key and a value. Backward pass by hand,
# softmax Jacobian included.
# ---------------------------------------------------------------------------

class Attention:
    def __init__(self, n_in, n_out, d=D_MODEL, dk=D_K, seed=0, lr=LR,
                 use_pos=True):
        r = np.random.default_rng(seed)
        self.d, self.dk = d, dk
        self.use_pos = use_pos
        self.We = r.standard_normal((n_in, d)) * np.sqrt(1.0 / n_in)
        self.Wq = r.standard_normal((d, dk)) * np.sqrt(1.0 / d)
        self.Wk = r.standard_normal((d, dk)) * np.sqrt(1.0 / d)
        self.Wv = r.standard_normal((d, dk)) * np.sqrt(1.0 / d)
        self.Wo = r.standard_normal((dk, n_out)) * np.sqrt(1.0 / dk)
        self.bo = np.zeros(n_out)
        self.params = [self.We, self.Wq, self.Wk, self.Wv, self.Wo, self.bo]
        self.opt = Adam(self.params, lr)

    def n_params(self):
        return sum(p.size for p in self.params)

    def forward(self, X):
        N, T1, _ = X.shape
        E = X @ self.We
        if self.use_pos:
            E = E + sinusoidal_pe(T1, self.d)      # order, smuggled back in
        q = E[:, -1] @ self.Wq
        K = E @ self.Wk
        V = E @ self.Wv
        S = np.einsum("nk,ntk->nt", q, K) / np.sqrt(self.dk)
        W = softmax_rows(S)                        # the soft lookup
        ctx = np.einsum("nt,ntk->nk", W, V)        # the weighted answer
        logits = ctx @ self.Wo + self.bo
        return logits, (E, q, K, V, S, W, ctx)

    def probs(self, X):
        L, _ = self.forward(X)
        Z = L - L.max(1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(1, keepdims=True)

    def loss(self, X, y):
        p = self.probs(X)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-300).mean())

    def attention_map(self, X):
        _, (_, _, _, _, _, W, _) = self.forward(X)
        return W

    def grads(self, X, y, want_reach=False):
        N, T1, _ = X.shape
        logits, (E, q, K, V, S, W, ctx) = self.forward(X)
        Z = logits - logits.max(1, keepdims=True)
        e = np.exp(Z)
        p = e / e.sum(1, keepdims=True)
        d_ = (p - np.eye(logits.shape[1])[y]) / N
        gWo = ctx.T @ d_
        gbo = d_.sum(0)
        dctx = d_ @ self.Wo.T
        dW = np.einsum("nk,ntk->nt", dctx, V)
        dV = W[:, :, None] * dctx[:, None, :]
        dS = W * (dW - (W * dW).sum(-1, keepdims=True))   # softmax Jacobian
        dS /= np.sqrt(self.dk)
        dq = np.einsum("nt,ntk->nk", dS, K)
        dK = dS[:, :, None] * q[:, None, :]
        dE = dK @ self.Wk.T + dV @ self.Wv.T
        dE[:, -1] += dq @ self.Wq.T
        g = [np.einsum("ntd,nte->de", X, dE),
             E[:, -1].T @ dq,
             np.einsum("ntd,ntk->dk", E, dK),
             np.einsum("ntd,ntk->dk", E, dV),
             gWo, gbo]
        if want_reach:
            return g, [float(np.linalg.norm(dE[:, t])) for t in range(T1)]
        return g

    def step(self, X, y):
        self.opt.step(self.grads(X, y))


def fit(net, X, y, epochs=EPOCHS):
    for _ in range(epochs):
        net.step(X, y)
    return net


def accuracy(net, X, y):
    return float((net.probs(X).argmax(1) == y).mean())


def gradcheck(net, X, y):
    g = net.grads(X, y)
    worst = 0.0
    for P_, G in zip(net.params, g):
        it = np.nditer(P_, flags=["multi_index"])
        for _ in it:
            ix = it.multi_index
            old = P_[ix]
            P_[ix] = old + 1e-6
            lp = net.loss(X, y)
            P_[ix] = old - 1e-6
            lm = net.loss(X, y)
            P_[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - G[ix]))
    return worst


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    r7 = np.random.default_rng(7)
    Xs, ys = make_recall(5, 3, r7)
    print(f"  [audit] attention backward vs central differences: "
          f"{gradcheck(Attention(VOCAB, VOCAB, d=6, dk=5, seed=1), Xs, ys):.2e}")

    banner("DEMO 1 --- The cliff, unbuilt")
    print("  Part 3's recall task, Part 3's budget. The RNN was perfect at")
    print("  30 steps of distance and at chance at 40. One attention head:")
    print()
    net40 = None
    for T in (40, 80, 160):
        X, y = make_recall(N_SEQ, T, np.random.default_rng(100 + T))
        Xt, yt = make_recall(N_SEQ, T, np.random.default_rng(200 + T))
        a = fit(Attention(VOCAB, VOCAB, seed=0), X, y)
        if T == 40:
            net40, test40 = a, (Xt, yt)
        print(f"    distance T = {T:3d}:  {accuracy(a, Xt, yt):6.1%}")
    print()
    print("  Flat. Distance is not a concept this architecture has: every")
    print("  position is one softmax away from the query, whether it is 1")
    print("  step back or 160. The gradient tells the same story, at")
    print("  initialisation on T = 40 (compare the RNN's ~300x decay):")
    print()
    X, y = make_recall(N_SEQ, 40, np.random.default_rng(140))
    _, reach = Attention(VOCAB, VOCAB, seed=0).grads(X, y, want_reach=True)
    print(f"    |dL/dE| at t=1: {reach[0]:.1e}   t=20: {reach[19]:.1e}"
          f"   t=40: {reach[39]:.1e}")
    print(f"    farthest-to-nearest ratio: {reach[0] / reach[39]:.1f}x"
          f"   (the RNN's was ~300x, downhill)")

    banner("DEMO 2 --- A set, not a sequence")
    print("  Q, K, V see a BAG of vectors: shuffle the inputs and every")
    print("  attention output is identical. Order arrives only through the")
    print("  positional encodings added to the embeddings. Remove them:")
    print()
    X, y = make_recall(N_SEQ, 40, np.random.default_rng(140))
    Xt, yt = make_recall(N_SEQ, 40, np.random.default_rng(240))
    a_pos = fit(Attention(VOCAB, VOCAB, seed=0), X, y)
    a_nopos = fit(Attention(VOCAB, VOCAB, seed=0, use_pos=False), X, y)
    print(f"    recall (needs 'FIRST'):    with positions {accuracy(a_pos, Xt, yt):6.1%}"
          f"    without {accuracy(a_nopos, Xt, yt):6.1%}")
    Xm, ym = make_majority(N_SEQ, 41, np.random.default_rng(500))
    Xmt, ymt = make_majority(N_SEQ, 41, np.random.default_rng(501))
    m_nopos = fit(Attention(VOCAB, 2, seed=0, use_pos=False), Xm, ym)
    print(f"    majority (order-free):     without positions "
          f"{accuracy(m_nopos, Xmt, ymt):6.1%}")
    print()
    print("  'First' is not a property of a set -- without positions the")
    print("  model cannot express the question, and recall collapses toward")
    print("  chance. Majority voting needs no order, and the same blind")
    print("  model aces it. Positional encoding is not a detail; it is the")
    print("  entire concept of sequence, bolted back on.")

    banner("DEMO 3 --- The receipt, and the bill")
    print("  Where did the trained T = 40 recall model look? Its attention")
    print("  weights, averaged over the test set:")
    print()
    Wmap = net40.attention_map(test40[0])
    print(f"    weight on position 0 (the answer): {Wmap[:, 0].mean():.3f}")
    print(f"    average weight elsewhere:          {Wmap[:, 1:].mean():.4f}")
    print()
    print("  99% of the lookup lands on the one position that matters --")
    print("  a receipt you can read, where the RNN's memory was a vector")
    print("  of 24 unlabelled numbers.")
    print()
    print("  The bill. This model is Bahdanau's original: ONE query, O(T)")
    print("  per lookup. Modern self-attention gives EVERY position its")
    print("  own query -- T lookups over T positions. Exact multiplies in")
    print("  the attention core (scores + weighted sum, d_k = 16):")
    print()
    print("    T        one query      all T queries     growth")
    prev = None
    for T in (40, 80, 160, 320):
        one = 2 * (T + 1) * D_K
        full = 2 * (T + 1) ** 2 * D_K
        growth = f"x{full / prev:.1f}" if prev else "  --"
        prev = full
        print(f"    {T:3d}      {one:9,d}      {full:13,d}      {growth}")
    print()
    print("  Double the sequence, quadruple the bill: O(T^2). That is the")
    print("  price of making distance free, and the defining engineering")
    print("  constraint of the architecture that pays it -- the")
    print("  Transformer, where this section goes next.")


if __name__ == "__main__":
    main()
