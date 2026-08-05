"""
transformer.py --- companion code for "Transformers"
(Deep Learning Architectures, Part 5).

Nothing in this file is new. That is the point. The transformer
(Vaswani et al., 2017) is this section, assembled:

    token embeddings + sinusoidal positions      (Part 4)
    multi-head CAUSAL self-attention             (Part 4, multiplied:
                                                  every position queries
                                                  every earlier one)
    the MLP block between attention layers       (Part 1 -- and it holds
                                                  most of the parameters)
    residual connections                         (Part 1's vanishing
                                                  gradient, solved by
                                                  addition: x + f(x))
    layer normalisation                          (keep every residual
                                                  stream at unit scale)

plus one triangular mask: position t may look only at positions <= t,
which turns the whole machine into a next-token predictor -- a language
model. Every forward and backward pass here is hand-written NumPy --
LayerNorm Jacobian, softmax Jacobian, multi-head reshapes, the lot --
and audited against central differences at startup.

Demonstrates:
  1. The assembly, and why residuals are in it: a 4-layer model with
     residual connections reaches loss 0.20 on the task below; the same
     model WITHOUT them: 0.74. Addition is the bridge gradients cross.
  2. A transformer learns an ALGORITHM: string reversal, 98.6% exact
     match -- and the receipt is readable: of the two attention heads,
     head 1 learned the mirror itself (output i attends to input
     K-1-i: a clean anti-diagonal), head 0 assists.
  3. A transformer learns LANGUAGE: character-level training on a
     1,447-character corpus (this section's own sentences). Loss falls
     from 3.87 to ~0.12 and the machine writes -- reciting its corpus,
     wandering between memorised grooves at novel prompts, dissolving
     into alphabet soup at high temperature. At this scale it is a
     memoriser with style; scale the same code by a factor of a
     million and it is the machine you are talking to.

Everything is plain NumPy. Dependencies: numpy. Runs in ~4 minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


def softmax_last(S):
    Z = S - S.max(-1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(-1, keepdims=True)


def sinusoidal_pe(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle))


# ---------------------------------------------------------------------------
# The machine. Pre-LN decoder blocks:  x = x + attn(LN(x)); x = x + mlp(LN(x))
# ---------------------------------------------------------------------------

class Transformer:
    def __init__(self, vocab, d=32, heads=4, layers=2, d_ff=None, seed=0,
                 lr=3e-3, use_resid=True):
        r = np.random.default_rng(seed)
        self.V, self.d, self.h, self.L = vocab, d, heads, layers
        self.dk = d // heads
        self.dff = d_ff or 4 * d
        self.use_resid = use_resid

        def rnd(*shape, s=1.0):
            return r.standard_normal(shape) * np.sqrt(s / shape[0])

        self.emb = rnd(vocab, d)
        self.blocks = []
        for _ in range(layers):
            self.blocks.append(dict(
                g1=np.ones(d), b1=np.zeros(d),
                Wqkv=rnd(d, 3 * d), Wo=rnd(d, d),
                g2=np.ones(d), b2=np.zeros(d),
                W1=rnd(d, self.dff, s=2.0), bb1=np.zeros(self.dff),
                W2=rnd(self.dff, d), bb2=np.zeros(d)))
        self.gf, self.bf = np.ones(d), np.zeros(d)
        self.Wout, self.bout = rnd(d, vocab), np.zeros(vocab)

        self.params, self.names = [self.emb], ["emb"]
        for i, b in enumerate(self.blocks):
            for k in ("g1", "b1", "Wqkv", "Wo", "g2", "b2", "W1", "bb1",
                      "W2", "bb2"):
                self.params.append(b[k])
                self.names.append(f"{i}.{k}")
        self.params += [self.gf, self.bf, self.Wout, self.bout]
        self.names += ["gf", "bf", "Wout", "bout"]
        self.gi = {n: i for i, n in enumerate(self.names)}
        self.M = [np.zeros_like(p) for p in self.params]
        self.Vm = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.lr = lr

    def n_params(self):
        return sum(p.size for p in self.params)

    # ---- layer norm ----
    @staticmethod
    def ln_fwd(x, g, b):
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        inv = 1.0 / np.sqrt((xc ** 2).mean(-1, keepdims=True) + 1e-5)
        xh = xc * inv
        return xh * g + b, (xh, inv)

    @staticmethod
    def ln_bwd(dy, cache, g):
        xh, inv = cache
        dg = (dy * xh).sum(axis=tuple(range(dy.ndim - 1)))
        db = dy.sum(axis=tuple(range(dy.ndim - 1)))
        dxh = dy * g
        dx = inv * (dxh - dxh.mean(-1, keepdims=True)
                    - xh * (dxh * xh).mean(-1, keepdims=True))
        return dx, dg, db

    # ---- causal multi-head attention ----
    def attn_fwd(self, x, b):
        N, T, d = x.shape
        h, dk = self.h, self.dk
        q, k, v = np.split(x @ b["Wqkv"], 3, axis=-1)

        def sp(z):
            return z.reshape(N, T, h, dk).transpose(0, 2, 1, 3)

        q, k, v = sp(q), sp(k), sp(v)
        S = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)
        S = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -1e30, S)
        W = softmax_last(S)                       # (N, h, T, T)
        ctx = (W @ v).transpose(0, 2, 1, 3).reshape(N, T, d)
        return ctx @ b["Wo"], (x, q, k, v, W, ctx)

    def attn_bwd(self, dout, cache, b, grads, gi_prefix):
        x, q, k, v, W, ctxm = cache
        N, T, d = x.shape
        h, dk = self.h, self.dk
        grads[self.gi[gi_prefix + "Wo"]] += (
            ctxm.reshape(-1, d).T @ dout.reshape(-1, d))
        dctx = (dout @ b["Wo"].T).reshape(N, T, h, dk).transpose(0, 2, 1, 3)
        dW = dctx @ v.transpose(0, 1, 3, 2)
        dv = W.transpose(0, 1, 3, 2) @ dctx
        dS = W * (dW - (W * dW).sum(-1, keepdims=True)) / np.sqrt(dk)
        dq = dS @ k
        dk_ = dS.transpose(0, 1, 3, 2) @ q

        def mg(z):
            return z.transpose(0, 2, 1, 3).reshape(N, T, h * dk)

        dqkv = np.concatenate([mg(dq), mg(dk_), mg(dv)], axis=-1)
        grads[self.gi[gi_prefix + "Wqkv"]] += (
            x.reshape(-1, d).T @ dqkv.reshape(-1, 3 * d))
        return dqkv @ b["Wqkv"].T

    # ---- forward / loss / grads ----
    def forward(self, ids):
        N, T = ids.shape
        x = self.emb[ids] + sinusoidal_pe(T, self.d)
        caches = []
        for b in self.blocks:
            ln1, c1 = self.ln_fwd(x, b["g1"], b["b1"])
            a, ca = self.attn_fwd(ln1, b)
            x1 = x + a if self.use_resid else a
            ln2, c2 = self.ln_fwd(x1, b["g2"], b["b2"])
            hpre = ln2 @ b["W1"] + b["bb1"]
            hact = np.maximum(0, hpre)
            m = hact @ b["W2"] + b["bb2"]
            x2 = x1 + m if self.use_resid else m
            caches.append((x, c1, ca, x1, c2, ln2, hpre, hact))
            x = x2
        lnf, cf = self.ln_fwd(x, self.gf, self.bf)
        return lnf @ self.Wout + self.bout, (caches, cf, lnf)

    def loss(self, ids, targets):
        logits, _ = self.forward(ids)
        P = softmax_last(logits)
        N, T = targets.shape
        return float(-np.log(
            P[np.arange(N)[:, None], np.arange(T)[None, :], targets]
            + 1e-300).mean())

    def grads_(self, ids, targets):
        N, T = ids.shape
        logits, (caches, cf, lnf) = self.forward(ids)
        P = softmax_last(logits)
        onehot = np.zeros_like(P)
        onehot[np.arange(N)[:, None], np.arange(T)[None, :], targets] = 1
        dlog = (P - onehot) / (N * T)
        grads = [np.zeros_like(p) for p in self.params]
        grads[self.gi["Wout"]] += (
            lnf.reshape(-1, self.d).T @ dlog.reshape(-1, self.V))
        grads[self.gi["bout"]] += dlog.sum((0, 1))
        dx, dgf, dbf = self.ln_bwd(dlog @ self.Wout.T, cf, self.gf)
        grads[self.gi["gf"]] += dgf
        grads[self.gi["bf"]] += dbf
        for li in range(self.L - 1, -1, -1):
            b = self.blocks[li]
            x0, c1, ca, x1, c2, ln2, hpre, hact = caches[li]
            grads[self.gi[f"{li}.W2"]] += (
                hact.reshape(-1, self.dff).T @ dx.reshape(-1, self.d))
            grads[self.gi[f"{li}.bb2"]] += dx.sum((0, 1))
            dhpre = (dx @ b["W2"].T) * (hpre > 0)
            grads[self.gi[f"{li}.W1"]] += (
                ln2.reshape(-1, self.d).T @ dhpre.reshape(-1, self.dff))
            grads[self.gi[f"{li}.bb1"]] += dhpre.sum((0, 1))
            dx1_ln, dg2, db2 = self.ln_bwd(dhpre @ b["W1"].T, c2, b["g2"])
            grads[self.gi[f"{li}.g2"]] += dg2
            grads[self.gi[f"{li}.b2"]] += db2
            dx1 = dx1_ln + (dx if self.use_resid else 0)
            dln1 = self.attn_bwd(dx1, ca, b, grads, f"{li}.")
            dx0_ln, dg1, db1 = self.ln_bwd(dln1, c1, b["g1"])
            grads[self.gi[f"{li}.g1"]] += dg1
            grads[self.gi[f"{li}.b1"]] += db1
            dx = dx0_ln + (dx1 if self.use_resid else 0)
        np.add.at(grads[self.gi["emb"]], ids, dx)
        return grads

    def step(self, ids, targets):
        g = self.grads_(ids, targets)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, gr) in enumerate(zip(self.params, g)):
            self.M[i] = b1 * self.M[i] + (1 - b1) * gr
            self.Vm[i] = b2 * self.Vm[i] + (1 - b2) * gr ** 2
            p -= self.lr * (self.M[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.Vm[i] / (1 - b2 ** self.t)) + eps)


def gradcheck_sampled(net, ids, targets, n_per=6, seed=3):
    g = net.grads_(ids, targets)
    r = np.random.default_rng(seed)
    worst = 0.0
    checked = 0
    for P_, G in zip(net.params, g):
        flat, gflat = P_.reshape(-1), G.reshape(-1)
        for ix in r.choice(len(flat), min(n_per, len(flat)), replace=False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp = net.loss(ids, targets)
            flat[ix] = old - 1e-6
            lm = net.loss(ids, targets)
            flat[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
            checked += 1
    return worst, checked


# ---------------------------------------------------------------------------
# Stage 1: string reversal.  [s1..s8, SEP, s8..s1] -- learn the mirror.
# ---------------------------------------------------------------------------

K = 8
SEP = 6
VOC_REV = 7


def make_rev(n, rng):
    s = rng.integers(0, 6, (n, K))
    seq = np.concatenate([s, np.full((n, 1), SEP), s[:, ::-1]], axis=1)
    return seq[:, :-1], seq[:, 1:]


# ---------------------------------------------------------------------------
# Stage 2: a language. This section's own sentences, 1,447 characters.
# ---------------------------------------------------------------------------

CORPUS = (
    "the gradient is a promise: every weight moves in the direction that "
    "lowers the loss. a layer is an affine map and a nonlinearity, and "
    "without the nonlinearity the stack folds flat into one linear layer. "
    "the network learns by walking the chain rule backwards, layer by "
    "layer, keeping every activation as it goes. attention is a soft "
    "lookup: the query asks, the keys offer, the values answer, and the "
    "softmax decides how much of each answer to keep. distance stops "
    "existing when every position is one hop from every other. the forget "
    "gate is a learned number near one, and the cell state is a highway "
    "through time. width memorises and depth composes: the first layer "
    "builds curves from lines, the second builds the spiral from curves. "
    "the boundary lives in the low density gap, because that is the only "
    "place a wide empty street fits. an architectural prior is data you "
    "do not have to collect. the error signal must survive the trip from "
    "the loss to the first layer, and every step of distance multiplies "
    "it by the same squashing jacobian. the residual connection is the "
    "repair: the network learns corrections, not transformations, and "
    "the gradient flows through the addition untouched. the attention "
    "map is a receipt you can read: the model shows where it looked. "
    "one cell, any length: what is learned at step three is known at "
    "step three hundred. priors only pay when they are true: shuffle "
    "the pixels and the convolution loses everything it knew."
)
CHARS = sorted(set(CORPUS))
C2I = {c: i for i, c in enumerate(CHARS)}
DATA = np.array([C2I[c] for c in CORPUS])
T_CTX = 48


def lm_batch(rng, n=24):
    ix = rng.integers(0, len(DATA) - T_CTX - 1, n)
    return (np.stack([DATA[i:i + T_CTX] for i in ix]),
            np.stack([DATA[i + 1:i + T_CTX + 1] for i in ix]))


def generate(net, prompt, n, temp, seed):
    g = np.random.default_rng(seed)
    ids_ = [C2I[c] for c in prompt]
    for _ in range(n):
        logits, _ = net.forward(np.array([ids_[-T_CTX:]]))
        p = logits[0, -1] / temp
        p = np.exp(p - p.max())
        p /= p.sum()
        ids_.append(int(g.choice(len(CHARS), p=p)))
    return "".join(CHARS[i] for i in ids_)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    r0 = np.random.default_rng(0)
    ids = r0.integers(0, 11, (3, 7))
    tg = r0.integers(0, 11, (3, 7))
    worst, checked = gradcheck_sampled(
        Transformer(11, d=16, heads=2, layers=2, seed=1), ids, tg)
    print(f"  [audit] {checked} sampled gradients vs central differences: "
          f"{worst:.2e}")

    banner("DEMO 1 --- The assembly, and why the residuals are in it")
    print("  Embeddings + positions (Part 4), causal multi-head attention")
    print("  (Part 4, every position now a query), the MLP block (Part 1),")
    print("  LayerNorm -- and residual connections: x + f(x). Part 1's")
    print("  vanishing gradient, solved by addition. Measured, 4 layers")
    print("  deep on the reversal task (400 steps):")
    print()
    ids, tg = make_rev(128, np.random.default_rng(0))
    for resid in (True, False):
        net = Transformer(VOC_REV, d=32, heads=4, layers=4, seed=0,
                          use_resid=resid)
        for _ in range(400):
            net.step(ids, tg)
        tag = "with residuals   " if resid else "without residuals"
        print(f"    {tag}: loss {net.loss(ids, tg):.3f}")
    print()
    print("  Same parameters, same budget; the only difference is whether")
    print("  gradients may cross each block by addition or must fight")
    print("  through it. Residual streams are why 'deep' can mean 96")
    print("  layers and not 6.")

    banner("DEMO 2 --- A transformer learns an algorithm")
    print("  Reverse a string: read 8 symbols, a separator, then emit them")
    print("  backwards. One layer, TWO heads, trained 1,000 steps:")
    print()
    ids, tg = make_rev(256, np.random.default_rng(0))
    ids_t, tg_t = make_rev(512, np.random.default_rng(1))
    net = Transformer(VOC_REV, d=32, heads=2, layers=1, seed=0)
    for _ in range(1000):
        net.step(ids, tg)
    logits, (caches, _, _) = net.forward(ids_t)
    pred = logits.argmax(-1)
    em = float((pred[:, K:] == tg_t[:, K:]).all(axis=1).mean())
    tok = float((pred[:, K:] == tg_t[:, K:]).mean())
    print(f"    token accuracy {tok:.1%}   sequence exact-match {em:.1%}")
    print()
    print("  And the receipt. For each output step, where does each head")
    print("  put its attention peak over the 8 input positions?")
    print()
    _, (caches, _, _) = net.forward(ids_t[:64])
    Wl = caches[0][2][4]
    anti = [K - 1 - i for i in range(K)]
    for hh in range(2):
        peaks = [int(r.argmax()) for r in Wl[:, hh, K:2 * K, 0:K].mean(0)]
        tag = "   <-- the mirror, learned" if peaks == anti else ""
        print(f"    head {hh}: {peaks}{tag}")
    print()
    print("  Head 1 IS the algorithm: output i attends to input 8-1-i, a")
    print("  clean anti-diagonal nobody programmed. Head 0 assists. The")
    print("  division of labour between heads -- one mechanism, several")
    print("  specialists -- is exactly why multi-head beats single-head.")

    banner("DEMO 3 --- A transformer learns a language")
    print(f"  Character-level, trained on this section's own sentences:")
    print(f"  {len(CORPUS):,} characters, vocabulary of {len(CHARS)}. "
          f"Two layers, four")
    print("  heads, 66k parameters. Loss (nats/char; uniform = "
          f"{np.log(len(CHARS)):.2f}):")
    print()
    net = Transformer(len(CHARS), d=48, heads=4, layers=2, seed=0)
    rng = np.random.default_rng(0)
    Xe, Ye = lm_batch(np.random.default_rng(99), 64)
    marks = {0: None, 100: None, 500: None, 1500: None}
    for step in range(1500):
        if step in marks:
            marks[step] = net.loss(Xe, Ye)
        x, y = lm_batch(rng)
        net.step(x, y)
    marks[1500] = net.loss(Xe, Ye)
    print("    step:  " + "   ".join(f"{s}" for s in marks))
    print("    loss:  " + "   ".join(f"{v:.2f}" for v in marks.values()))
    print()
    print("  And the machine writes. Prompt 'the gradient ', temperature")
    print("  0.4:")
    print()
    out = generate(net, "the gradient ", 140, 0.4, seed=42)
    print(f'    "{out}"')
    print()
    print("  Prompt 'the moon ' -- which appears nowhere in the corpus --")
    print("  temperature 0.4:")
    print()
    out = generate(net, "the moon ", 110, 0.4, seed=42)
    print(f'    "{out}"')
    print()
    print("  Temperature 2.0:")
    print()
    out = generate(net, "the gradient ", 110, 2.0, seed=42)
    print(f'    "{out}"')
    print()
    print("  Honesty about what happened: 66k parameters on 1.4k characters")
    print("  is a memoriser -- it recites its corpus (with the occasional")
    print("  stutter), absorbs unseen prompts into the nearest memorised")
    print("  groove, and dissolves when the temperature melts its")
    print("  certainties. But the mechanism doing the reciting is the same")
    print("  one that, at a hundred million times the scale, wrote half")
    print("  the text you read this week. Scale changed the wattage;")
    print("  Parts 1 to 5 are the whole circuit.")


if __name__ == "__main__":
    main()
