"""
rnn.py --- companion code for "Recurrent Neural Networks (RNN / LSTM)"
(Deep Learning Architectures, Part 3).

Part 2 shared nine weights across 144 image positions. This part shares
one network across TIME: a recurrent cell reads a sequence one step at
a time, folding what it has seen into a hidden state --

    h_t = tanh( x_t Wx  +  h_{t-1} Wh  +  b )

-- the same Wx, Wh at every step, so the parameter count is independent
of sequence length, and what is learned at step 3 is known at step 300.
Training is backpropagation THROUGH TIME: unroll, then walk the chain
rule backwards along the sequence.

And there the section's recurring villain returns, worse. Part 1's
vanishing gradient descended a stack of layers; here every step of
distance multiplies the error signal by the same recurrent Jacobian.
The result is not graceful degradation -- it is a cliff, and this
script measures exactly where it stands.

The LSTM (Hochreiter & Schmidhuber, 1997) is the classic repair: a
separate CELL state updated ADDITIVELY, c_t = f*c_{t-1} + i*g, with
learned gates deciding what to keep (f), write (i), and reveal (o).
Backwards, the cell path multiplies by f -- a learned number near 1 --
instead of by a squashing matrix. A highway through time.

The stage: RECALL -- the first symbol of a sequence is the label; T
distractor symbols follow. To answer, the network must carry one fact
across T steps of noise.

Demonstrates:
  1. One cell, any length: both cells solve recall at T=10 perfectly,
     parameter counts never grow with T -- and a control where the
     answer sits at the END of a T=40 sequence, which the vanilla RNN
     aces (99.5%). Length is not the problem.
  2. The wall is DISTANCE: sweep the gap from 20 to 40 steps. The
     vanilla RNN goes 100% -> 100% -> 24% (chance); the LSTM never
     drops. At initialisation the RNN's gradient reaches t=1 attenuated
     ~300x; the LSTM's arrives ~10x stronger, before any training.
  3. Inside the cell: the measured forget gates (biased to remember at
     birth, top units ~0.89 after training) -- and the honest boundary:
     PARITY over 40 bits defeats both cells (~50%). Gates cure
     vanishing gradients, not every long-horizon lesson.

Everything is plain NumPy: both cells, full BPTT, Adam, and a startup
gradient audit for each. Dependencies: numpy. Runs in 3-4 minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
VOCAB = 4
HIDDEN = 24
N_SEQ = 400
EPOCHS = 1000
LR = 3e-3


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# The stage: recall. First symbol = the label, then T distractors.
# ---------------------------------------------------------------------------

def make_recall(n, T, rng, answer_last=False):
    sym = rng.integers(0, VOCAB, (n, T + 1))
    y = sym[:, -1].copy() if answer_last else sym[:, 0].copy()
    return np.eye(VOCAB)[sym], y


def make_parity(n, T, rng):
    bits = rng.integers(0, 2, (n, T))
    return np.eye(2)[bits], bits.sum(axis=1) % 2


# ---------------------------------------------------------------------------
# Shared bits: Adam, softmax head, gradient audit.
# ---------------------------------------------------------------------------

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


def softmax(L):
    Z = L - L.max(1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(1, keepdims=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------
# The vanilla recurrent cell, with full BPTT.
# ---------------------------------------------------------------------------

class RNN:
    def __init__(self, n_in, hidden, n_out, seed=0, lr=LR):
        r = np.random.default_rng(seed)
        self.H = hidden
        self.Wx = r.standard_normal((n_in, hidden)) * np.sqrt(1.0 / n_in)
        self.Wh = r.standard_normal((hidden, hidden)) * np.sqrt(1.0 / hidden)
        self.bh = np.zeros(hidden)
        self.Wo = r.standard_normal((hidden, n_out)) * np.sqrt(1.0 / hidden)
        self.bo = np.zeros(n_out)
        self.params = [self.Wx, self.Wh, self.bh, self.Wo, self.bo]
        self.opt = Adam(self.params, lr)

    def n_params(self):
        return sum(p.size for p in self.params)

    def forward(self, X):
        N, T1, _ = X.shape
        H = np.zeros((N, T1 + 1, self.H))
        for t in range(T1):
            H[:, t + 1] = np.tanh(X[:, t] @ self.Wx + H[:, t] @ self.Wh
                                  + self.bh)
        return H[:, -1] @ self.Wo + self.bo, H

    def probs(self, X):
        return softmax(self.forward(X)[0])

    def loss(self, X, y):
        p = self.probs(X)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-300).mean())

    def grads(self, X, y, want_reach=False):
        N, T1, _ = X.shape
        logits, H = self.forward(X)
        d = (softmax(logits) - np.eye(logits.shape[1])[y]) / N
        gWo = H[:, -1].T @ d
        gbo = d.sum(0)
        gWx = np.zeros_like(self.Wx)
        gWh = np.zeros_like(self.Wh)
        gbh = np.zeros_like(self.bh)
        dh = d @ self.Wo.T
        reach = []
        for t in range(T1 - 1, -1, -1):        # walk time backwards
            dz = dh * (1 - H[:, t + 1] ** 2)
            gWx += X[:, t].T @ dz
            gWh += H[:, t].T @ dz
            gbh += dz.sum(0)
            dh = dz @ self.Wh.T                # one more Jacobian each step
            if want_reach:
                reach.append(float(np.linalg.norm(dh)))
        g = [gWx, gWh, gbh, gWo, gbo]
        return (g, reach[::-1]) if want_reach else g

    def step(self, X, y):
        self.opt.step(self.grads(X, y))


# ---------------------------------------------------------------------------
# The LSTM cell: gated, additively-updated memory. Forget bias starts at
# +1 -- newborn LSTMs are biased to remember.
# ---------------------------------------------------------------------------

class LSTM:
    def __init__(self, n_in, hidden, n_out, seed=0, lr=LR):
        r = np.random.default_rng(seed)
        self.H = hidden
        D = n_in + hidden
        self.W = r.standard_normal((D, 4 * hidden)) * np.sqrt(1.0 / D)
        self.b = np.zeros(4 * hidden)
        self.b[hidden:2 * hidden] = 1.0
        self.Wo = r.standard_normal((hidden, n_out)) * np.sqrt(1.0 / hidden)
        self.bo = np.zeros(n_out)
        self.params = [self.W, self.b, self.Wo, self.bo]
        self.opt = Adam(self.params, lr)

    def n_params(self):
        return sum(p.size for p in self.params)

    def forward(self, X, cache=False):
        N, T1, _ = X.shape
        Hh = self.H
        h = np.zeros((N, Hh))
        c = np.zeros((N, Hh))
        caches, hs = [], [h]
        for t in range(T1):
            z = np.concatenate([X[:, t], h], axis=1)
            A = z @ self.W + self.b
            i = sigmoid(A[:, :Hh])
            f = sigmoid(A[:, Hh:2 * Hh])
            o = sigmoid(A[:, 2 * Hh:3 * Hh])
            g = np.tanh(A[:, 3 * Hh:])
            c = f * c + i * g                  # the additive highway
            tc = np.tanh(c)
            h = o * tc
            if cache:
                caches.append((z, i, f, o, g, c.copy(), tc))
            hs.append(h)
        return hs[-1] @ self.Wo + self.bo, caches, hs

    def probs(self, X):
        return softmax(self.forward(X)[0])

    def loss(self, X, y):
        p = self.probs(X)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-300).mean())

    def grads(self, X, y, want_reach=False):
        N, T1, D_in = X.shape
        Hh = self.H
        logits, caches, hs = self.forward(X, cache=True)
        d = (softmax(logits) - np.eye(logits.shape[1])[y]) / N
        gWo = hs[-1].T @ d
        gbo = d.sum(0)
        gW = np.zeros_like(self.W)
        gb = np.zeros_like(self.b)
        dh = d @ self.Wo.T
        dc = np.zeros((N, Hh))
        reach = []
        for t in range(T1 - 1, -1, -1):
            z, i, f, o, g, c, tc = caches[t]
            c_prev = caches[t - 1][5] if t > 0 else np.zeros((N, Hh))
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            dA = np.concatenate([dc * g * i * (1 - i),
                                 dc * c_prev * f * (1 - f),
                                 do * o * (1 - o),
                                 dc * i * (1 - g ** 2)], axis=1)
            gW += z.T @ dA
            gb += dA.sum(0)
            dh = (dA @ self.W.T)[:, D_in:]
            dc = dc * f                        # multiply by f, not a matrix
            if want_reach:
                reach.append(float(np.linalg.norm(dh) + np.linalg.norm(dc)))
        g_ = [gW, gb, gWo, gbo]
        return (g_, reach[::-1]) if want_reach else g_

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
    print(f"  [audit] RNN  BPTT vs central differences: "
          f"{gradcheck(RNN(VOCAB, 5, VOCAB, seed=1), Xs, ys):.2e}")
    print(f"  [audit] LSTM BPTT vs central differences: "
          f"{gradcheck(LSTM(VOCAB, 5, VOCAB, seed=1), Xs, ys):.2e}")

    banner("DEMO 1 --- One cell, any length")
    print("  Recall: the FIRST symbol is the answer; T distractors follow.")
    print("  T = 10 first, and note the parameter counts -- one cell is")
    print("  reused at every step, so T never appears in them.")
    print()
    X, y = make_recall(N_SEQ, 10, np.random.default_rng(110))
    Xt, yt = make_recall(N_SEQ, 10, np.random.default_rng(210))
    r_ = fit(RNN(VOCAB, HIDDEN, VOCAB, seed=0), X, y)
    l_ = fit(LSTM(VOCAB, HIDDEN, VOCAB, seed=0), X, y)
    print(f"    vanilla RNN : {r_.n_params():5d} params   "
          f"test {accuracy(r_, Xt, yt):6.1%}")
    print(f"    LSTM        : {l_.n_params():5d} params   "
          f"test {accuracy(l_, Xt, yt):6.1%}")
    print()
    print("  And a control before the main event: same T = 40 length, but")
    print("  the answer is the LAST symbol -- distance zero:")
    print()
    X, y = make_recall(N_SEQ, 40, np.random.default_rng(400),
                       answer_last=True)
    Xt, yt = make_recall(N_SEQ, 40, np.random.default_rng(401),
                         answer_last=True)
    r_ = fit(RNN(VOCAB, HIDDEN, VOCAB, seed=0), X, y)
    l_ = fit(LSTM(VOCAB, HIDDEN, VOCAB, seed=0), X, y)
    print(f"    vanilla RNN : {accuracy(r_, Xt, yt):6.1%}"
          f"      LSTM : {accuracy(l_, Xt, yt):6.1%}")
    print()
    print("  Forty steps of sequence are no obstacle at all -- as long as")
    print("  nothing must be REMEMBERED across them. Keep that in mind.")

    banner("DEMO 2 --- The wall is distance, and it is a cliff")
    print("  Same task, same budget; only the gap between the answer and")
    print("  the question grows:")
    print()
    print("    distance T    vanilla RNN    LSTM")
    lstm40 = None
    for T in (20, 30, 40):
        X, y = make_recall(N_SEQ, T, np.random.default_rng(100 + T))
        Xt, yt = make_recall(N_SEQ, T, np.random.default_rng(200 + T))
        r_ = fit(RNN(VOCAB, HIDDEN, VOCAB, seed=0), X, y)
        l_ = fit(LSTM(VOCAB, HIDDEN, VOCAB, seed=0), X, y)
        if T == 40:
            lstm40, rec40 = l_, (X, y)
        print(f"       {T}          {accuracy(r_, Xt, yt):6.1%}      "
              f"{accuracy(l_, Xt, yt):6.1%}")
    print()
    print("  No graceful decline: the vanilla RNN is perfect at 30 and at")
    print("  CHANCE at 40 (four symbols -> 25%). The mechanism is visible")
    print("  before training even starts -- the error signal's norm as it")
    print("  travels back to t = 1 on the T = 40 task, at initialisation:")
    print()
    X, y = make_recall(N_SEQ, 40, np.random.default_rng(140))
    _, nr = RNN(VOCAB, HIDDEN, VOCAB, seed=0).grads(X, y, want_reach=True)
    _, nl = LSTM(VOCAB, HIDDEN, VOCAB, seed=0).grads(X, y, want_reach=True)
    print("    step:      t=40      t=30      t=20      t=10      t=1")
    print("    RNN :   " + "  ".join(f"{nr[p-1]:8.1e}" for p in (40, 30, 20, 10, 1)))
    print("    LSTM:   " + "  ".join(f"{nl[p-1]:8.1e}" for p in (40, 30, 20, 10, 1)))
    print()
    print("  Every backward step multiplies the RNN's signal by the same")
    print("  tanh-squashed recurrent Jacobian -- Part 1's vanishing")
    print("  gradient, now applied 40 times by construction. The LSTM's")
    print("  signal arrives an order of magnitude stronger, and training")
    print("  only widens the gap: its cell path multiplies by the forget")
    print("  gate, a learned number near 1, not by a matrix.")

    banner("DEMO 3 --- Inside the cell: gates, a highway, and a boundary")
    print("  c_t = f*c_{t-1} + i*g  changes the backward pass: along the")
    print("  cell, gradient multiplies by f alone. The script initialises")
    print("  the forget bias at +1 -- newborn LSTMs are biased to remember")
    print("  -- and training pushes the units it uses higher. Measured on")
    print("  the trained T = 40 recall LSTM (per-unit forget gate, averaged")
    print("  over time):")
    print()
    li = LSTM(VOCAB, HIDDEN, VOCAB, seed=0)
    _, caches, _ = li.forward(rec40[0][:200], cache=True)
    pu0 = np.stack([c[2].mean(axis=0) for c in caches]).mean(axis=0)
    _, caches, _ = lstm40.forward(rec40[0][:200], cache=True)
    pu1 = np.stack([c[2].mean(axis=0) for c in caches]).mean(axis=0)
    print(f"    at init  : mean {pu0.mean():.2f}   top-3 units "
          f"{np.sort(pu0)[::-1][:3].round(3)}")
    print(f"    trained  : mean {pu1.mean():.2f}   top-3 units "
          f"{np.sort(pu1)[::-1][:3].round(3)}")
    print()
    print("  And the honest boundary of the fix. PARITY of 40 bits needs")
    print("  no long-range storage -- just one bit, updated every step --")
    print("  yet the credit assignment is brutal: every bit matters, and")
    print("  flipping any one flips the answer.")
    print()
    X, y = make_parity(N_SEQ, 40, np.random.default_rng(300))
    Xt, yt = make_parity(N_SEQ, 40, np.random.default_rng(301))
    r_ = fit(RNN(2, HIDDEN, 2, seed=0), X, y)
    l_ = fit(LSTM(2, HIDDEN, 2, seed=0), X, y)
    print(f"    parity, T = 40:  vanilla RNN {accuracy(r_, Xt, yt):.1%}"
          f"    LSTM {accuracy(l_, Xt, yt):.1%}   (chance: 50%)")
    print()
    print("  Both cells fail. Gates cure VANISHING GRADIENTS; they do not")
    print("  cure every hard lesson a long horizon can teach. Knowing the")
    print("  difference is knowing what an LSTM actually is: not memory")
    print("  magic -- an architectural guarantee that the error signal")
    print("  survives the trip.")


if __name__ == "__main__":
    main()
