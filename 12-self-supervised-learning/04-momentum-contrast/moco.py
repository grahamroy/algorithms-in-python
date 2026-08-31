"""
moco.py --- companion code for "Momentum Contrast (MoCo)"
(Self-Supervised Learning, Part 4 -- the section finale).

Part 1 established the crowd: negatives are the signal, and the
NT-Xent batch must carry them all, at O(B^2) cost. MoCo (He et al.,
2020) asks the economic question -- must the crowd fit in the batch?
-- and answers with a QUEUE: keep the last K key embeddings as
negatives, push the current batch in, pop the oldest out. A batch
of 8 can face a crowd of 1,024.

The catch that makes it an invention rather than a data structure:
the queue spans thousands of steps, so its entries were encoded by
OLD versions of the network. If the encoder drifts quickly, the
queue is incoherent -- today's queries are contrasted against
yesterday's coordinate system. MoCo's fix is Part 3's trick pointed
backwards: encode the keys with a slow EMA copy of the query
encoder. Consistency, not the queue, is the contribution -- and the
momentum ablation here shows it: keys from the live encoder score
at CHANCE; keys from a slow twin work.

The stage is shape world, and -- closing the section -- the script
retrains all four of its methods live (SimCLR's NT-Xent, MAE, DINO,
MoCo) for one final family table.

Demonstrates:
  1. The mechanism works, decoupled: MoCo with batch 32 and a
     1,024-deep queue reaches 96.0% / 99.6%; even batch EIGHT
     (with 32k cheap steps) reaches 96.0% / 99.7% -- the crowd
     size no longer has anything to do with the batch size.
  2. The consistency law. Key-encoder momentum 0 (keys from the
     live network): 32.9% -- chance. 0.9: 51.4%. 0.99: 89.7%.
     0.999: 95.4%. A queue is trivial; a queue whose entries
     remain mutually consistent across thousands of steps is the
     invention, and it needs the slowest twin of the section.
  3. The honest audit. On THIS stage the crowd saturates: queue
     16 and queue 4,096 score identically, and plain NT-Xent at
     batch 8, trained equally long, matches MoCo (95.9% / 99.9%
     vs 96.0% / 99.7%). Three classes need a dozen negatives.
     MoCo's machinery pays off where crowds must number in the
     tens of thousands and memory cannot hold the batch -- at
     ImageNet, not at shape world. Third lesson of this section
     in the same key: measure before you inherit machinery.
  4. The family, closed. All four methods, one table, and one
     taxonomy: every self-supervised method is an answer to
     "where does the yardstick for the second view come from?"
     SimCLR: the batch beside you. MoCo: a ledger of the recent
     past, kept honest by a slow twin. DINO: the slow twin
     alone. MAE: the data itself.

Everything is plain NumPy, gradients hand-derived and audited.
Dependencies: numpy. Runs in about five minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
SIDE = 16
D_IN = SIDE * SIDE
D_EMB = 32
HID = 96
TAU = 0.2
NOISE_PIX = 0.10
N_UNLAB = 4000
N_TEST = 1000
BATCH = 256
K_PROTO = 32
T_S = 0.2
T_T = 0.05
EMA_C = 0.9
N_PATCH = 16
P = 4
MASK_RATIO = 0.75
ROWIDX = np.arange(SIDE)

PIX_PATCH = np.zeros((SIDE, SIDE), dtype=int)
for _py in range(4):
    for _px in range(4):
        PIX_PATCH[_py * P:(_py + 1) * P, _px * P:(_px + 1) * P] = \
            _py * 4 + _px
PIX_PATCH_FLAT = PIX_PATCH.reshape(-1)


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# The stage: shape world, identical to Parts 1-3.
# ---------------------------------------------------------------------------

def draw_shape(cls, cx, cy, size, bright):
    img = np.zeros((SIDE, SIDE))
    s = int(size)
    if cls == 0:
        img[cy - s:cy + s + 1, cx] = 1.0
        img[cy, cx - s:cx + s + 1] = 1.0
    elif cls == 1:
        img[cy - s:cy + s + 1, cx - s] = 1.0
        img[cy - s:cy + s + 1, cx + s] = 1.0
        img[cy - s, cx - s:cx + s + 1] = 1.0
        img[cy + s, cx - s:cx + s + 1] = 1.0
    else:
        for d in range(-s, s + 1):
            img[cy + d, cx + d] = 1.0
            img[cy + d, cx - d] = 1.0
    return img * bright


def make_instances(n, rng):
    y = rng.integers(0, 3, n)
    imgs = np.zeros((n, SIDE, SIDE))
    factors = np.zeros((n, 4))
    for i in range(n):
        s = rng.integers(3, 6)
        cx = rng.integers(s, SIDE - s)
        cy = rng.integers(s, SIDE - s)
        b = rng.uniform(0.6, 1.4)
        imgs[i] = draw_shape(y[i], cx, cy, s, b)
        factors[i] = (cx, cy, s, b)
    return imgs, y, factors


def augment(imgs, rng, shift=True, jitter=True, noise=True):
    n = len(imgs)
    out = imgs
    if shift:
        d = rng.integers(-3, 4, (n, 2))
        rows = (ROWIDX[None, :] - d[:, 1:2]) % SIDE
        cols = (ROWIDX[None, :] - d[:, 0:1]) % SIDE
        out = out[np.arange(n)[:, None, None], rows[:, :, None],
                  cols[:, None, :]]
    if jitter:
        out = out * rng.uniform(0.7, 1.3, (n, 1, 1))
    if noise:
        out = out + NOISE_PIX * rng.standard_normal(out.shape)
    return out.reshape(n, -1)


# ---------------------------------------------------------------------------
# Plumbing, identical to Parts 1-3.
# ---------------------------------------------------------------------------

def lrelu(z):
    return np.where(z > 0, z, 0.2 * z)


def dlrelu(z):
    return np.where(z > 0, 1.0, 0.2)


class Adam:
    def __init__(self, params, lr, b1=0.5):
        self.p, self.lr, self.b1 = params, lr, b1
        self.M = [np.zeros_like(p) for p in params]
        self.V = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2, eps = self.b1, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.M[i] = b1 * self.M[i] + (1 - b1) * g
            self.V[i] = b2 * self.V[i] + (1 - b2) * g * g
            p -= self.lr * (self.M[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.V[i] / (1 - b2 ** self.t)) + eps)


class Net:
    def __init__(self, d_in, hidden, d_out, seed, lr):
        r = np.random.default_rng(seed)
        self.W1 = r.standard_normal((d_in, hidden)) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(hidden)
        self.W2 = r.standard_normal((hidden, hidden)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = r.standard_normal((hidden, d_out)) * np.sqrt(1.0 / hidden)
        self.b3 = np.zeros(d_out)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.opt = Adam(self.params, lr)

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        H1 = lrelu(Z1)
        Z2 = H1 @ self.W2 + self.b2
        H2 = lrelu(Z2)
        return H2 @ self.W3 + self.b3, (X, Z1, H1, Z2, H2)

    def backward(self, cache, dout):
        X, Z1, H1, Z2, H2 = cache
        gW3 = H2.T @ dout
        gb3 = dout.sum(0)
        dZ2 = (dout @ self.W3.T) * dlrelu(Z2)
        gW2 = H1.T @ dZ2
        gb2 = dZ2.sum(0)
        dZ1 = (dZ2 @ self.W2.T) * dlrelu(Z1)
        gW1 = X.T @ dZ1
        gb1 = dZ1.sum(0)
        return [gW1, gb1, gW2, gb2, gW3, gb3], dZ1 @ self.W1.T


class Encoder:
    def __init__(self, d_in=D_IN, hidden=HID, d_emb=D_EMB, seed=0,
                 lr=1e-3):
        self.net = Net(d_in, hidden, d_emb, seed=seed, lr=lr)

    def embed(self, X, cache=False):
        H, c = self.net.forward(X)
        nrm = np.sqrt((H ** 2).sum(1, keepdims=True)) + 1e-12
        Z = H / nrm
        if cache:
            return Z, (H, nrm, c)
        return Z

    def backward_emb(self, cach, dZ):
        H, nrm, c = cach
        Z = H / nrm
        dH = (dZ - Z * (dZ * Z).sum(1, keepdims=True)) / nrm
        grads, _ = self.net.backward(c, dH)
        return grads


def ntxent_grads(enc, Xa, Xb, tau=TAU):
    n = len(Xa)
    Za, ca = enc.embed(Xa, cache=True)
    Zb, cb = enc.embed(Xb, cache=True)
    Z = np.concatenate([Za, Zb])
    S = Z @ Z.T / tau
    np.fill_diagonal(S, -1e9)
    pos = np.concatenate([np.arange(n, 2 * n), np.arange(n)])
    E = np.exp(S - S.max(1, keepdims=True))
    Pm = E / E.sum(1, keepdims=True)
    loss = float(-np.log(Pm[np.arange(2 * n), pos] + 1e-300).mean())
    G = Pm.copy()
    G[np.arange(2 * n), pos] -= 1
    G /= (2 * n)
    dZ = ((G + G.T) @ Z) / tau
    ga = enc.backward_emb(ca, dZ[:n])
    gb = enc.backward_emb(cb, dZ[n:])
    return loss, [a + b for a, b in zip(ga, gb)]


def train_ntxent(imgs, seed=0, steps=2000, batch=BATCH):
    enc = Encoder(seed=seed)
    r1 = np.random.default_rng(1)
    for _ in range(steps):
        idx = r1.integers(0, len(imgs), batch)
        Xa = augment(imgs[idx], r1)
        Xb = augment(imgs[idx], r1)
        _, grads = ntxent_grads(enc, Xa, Xb)
        enc.net.opt.step(grads)
    return enc


def train_linear_probe(Z, y, n_cls, steps=300, lr=0.1, seed=0):
    r = np.random.default_rng(seed)
    W = r.standard_normal((Z.shape[1], n_cls)) * 0.01
    b = np.zeros(n_cls)
    opt = Adam([W, b], lr)
    for _ in range(steps):
        L = Z @ W + b
        L -= L.max(1, keepdims=True)
        Pm = np.exp(L)
        Pm /= Pm.sum(1, keepdims=True)
        G = Pm.copy()
        G[np.arange(len(y)), y] -= 1
        G /= len(y)
        opt.step([Z.T @ G, G.sum(0)])
    return W, b


def probe_acc(Ztr, ytr, Zte, yte, n_cls=3):
    W, b = train_linear_probe(Ztr, ytr, n_cls)
    return float(((Zte @ W + b).argmax(1) == yte).mean())


# ---------------------------------------------------------------------------
# Part 2's MAE and Part 3's DINO, minimal replicas for the family table.
# ---------------------------------------------------------------------------

def mask_batch(X, rng, ratio=MASK_RATIO):
    n = len(X)
    n_hide = int(round(N_PATCH * ratio))
    hide = np.zeros((n, N_PATCH), dtype=bool)
    for i in range(n):
        hide[i, rng.choice(N_PATCH, n_hide, replace=False)] = True
    pixhide = hide[:, PIX_PATCH_FLAT]
    Xm = np.where(pixhide, 0.0, X)
    return np.concatenate([Xm, hide.astype(float)], 1), pixhide


class MAE:
    def __init__(self, hidden=HID, d_emb=D_EMB, seed=0, lr=1e-3):
        self.enc = Net(D_IN + N_PATCH, hidden, d_emb, seed=seed, lr=lr)
        self.dec = Net(d_emb, hidden, D_IN, seed=seed + 100, lr=lr)

    def embed(self, X):
        inp = np.concatenate([X, np.zeros((len(X), N_PATCH))], 1)
        return self.enc.forward(inp)[0]

    def step(self, X, rng):
        inp, pixhide = mask_batch(X, rng)
        H, ce = self.enc.forward(inp)
        R, cd = self.dec.forward(H)
        diff = (R - X) * pixhide
        n_mask = pixhide.sum()
        gdec, dH = self.dec.backward(cd, 2 * diff / n_mask)
        genc, _ = self.enc.backward(ce, dH)
        self.enc.opt.step(genc)
        self.dec.opt.step(gdec)


def softmax_T(L, T):
    Z = L / T
    Z -= Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


class DINO:
    def __init__(self, hidden=HID, d_emb=D_EMB, k=K_PROTO, seed=0,
                 lr=1e-3, ema_net=0.99):
        self.net = Net(D_IN, hidden, d_emb, seed=seed, lr=lr)
        r = np.random.default_rng(seed + 500)
        self.Wp = r.standard_normal((d_emb, k)) * np.sqrt(1.0 / d_emb)
        self.bp = np.zeros(k)
        self.opt_p = Adam([self.Wp, self.bp], lr)
        self.t_params = [p.copy() for p in self.net.params]
        self.t_Wp = self.Wp.copy()
        self.t_bp = self.bp.copy()
        self.c = np.zeros(k)
        self.ema_net = ema_net

    def teacher_probs(self, X):
        saved = [p.copy() for p in self.net.params]
        for p, t in zip(self.net.params, self.t_params):
            p[...] = t
        H, _ = self.net.forward(X)
        for p, s in zip(self.net.params, saved):
            p[...] = s
        L = H @ self.t_Wp + self.t_bp
        return softmax_T(L - self.c, T_T), L

    def step(self, Xa, Xb):
        n = len(Xa)
        Pt_a, La = self.teacher_probs(Xa)
        Pt_b, Lb = self.teacher_probs(Xb)
        tot_gnet, tot_gp = None, None
        for Xs, Pt in ((Xb, Pt_a), (Xa, Pt_b)):
            H, ch = self.net.forward(Xs)
            Ls = H @ self.Wp + self.bp
            Ps = softmax_T(Ls, T_S)
            dL = (Ps - Pt) / T_S / (2 * n)
            gWp = H.T @ dL
            gbp = dL.sum(0)
            gnet, _ = self.net.backward(ch, dL @ self.Wp.T)
            tot_gnet = gnet if tot_gnet is None else [
                a + b for a, b in zip(tot_gnet, gnet)]
            tot_gp = [gWp, gbp] if tot_gp is None else [
                tot_gp[0] + gWp, tot_gp[1] + gbp]
        self.net.opt.step(tot_gnet)
        self.opt_p.step(tot_gp)
        m = self.ema_net
        for t, p in zip(self.t_params, self.net.params):
            t *= m
            t += (1 - m) * p
        self.t_Wp = m * self.t_Wp + (1 - m) * self.Wp
        self.t_bp = m * self.t_bp + (1 - m) * self.bp
        self.c = EMA_C * self.c + (1 - EMA_C) * \
            np.concatenate([La, Lb]).mean(0)

    def embed(self, X):
        saved = [p.copy() for p in self.net.params]
        for p, t in zip(self.net.params, self.t_params):
            p[...] = t
        H, _ = self.net.forward(X)
        for p, s in zip(self.net.params, saved):
            p[...] = s
        return H


# ---------------------------------------------------------------------------
# MoCo: query encoder, EMA key encoder, queue of past keys.
# ---------------------------------------------------------------------------

class MoCo:
    def __init__(self, hidden=HID, d_emb=D_EMB, seed=0, lr=1e-3,
                 m_key=0.999, queue_k=1024):
        self.enc = Encoder(hidden=hidden, d_emb=d_emb, seed=seed, lr=lr)
        self.k_params = [p.copy() for p in self.enc.net.params]
        self.m_key = m_key
        r = np.random.default_rng(seed + 900)
        Q = r.standard_normal((queue_k, d_emb))
        self.queue = Q / np.sqrt((Q ** 2).sum(1, keepdims=True))
        self.ptr = 0

    def key_embed(self, X):
        saved = [p.copy() for p in self.enc.net.params]
        for p, k in zip(self.enc.net.params, self.k_params):
            p[...] = k
        Z = self.enc.embed(X)
        for p, s in zip(self.enc.net.params, saved):
            p[...] = s
        return Z

    def loss_and_grads(self, Xq, Xk, tau=TAU):
        """InfoNCE of each query against its key (positive) and the
        whole queue (negatives). Gradients through the query only."""
        n = len(Xq)
        Zq, cq = self.enc.embed(Xq, cache=True)
        Zk = self.key_embed(Xk)
        l_pos = (Zq * Zk).sum(1, keepdims=True)
        l_neg = Zq @ self.queue.T
        L = np.concatenate([l_pos, l_neg], 1) / tau
        L -= L.max(1, keepdims=True)
        E = np.exp(L)
        Pm = E / E.sum(1, keepdims=True)
        loss = float(-np.log(Pm[:, 0] + 1e-300).mean())
        G = Pm.copy()
        G[:, 0] -= 1
        G /= n
        dZq = (G[:, :1] * Zk + G[:, 1:] @ self.queue) / tau
        grads = self.enc.backward_emb(cq, dZq)
        return loss, grads, Zk

    def step(self, Xq, Xk, tau=TAU):
        loss, grads, Zk = self.loss_and_grads(Xq, Xk, tau)
        self.enc.net.opt.step(grads)
        m = self.m_key
        for k, p in zip(self.k_params, self.enc.net.params):
            k *= m
            k += (1 - m) * p
        n, K = len(Zk), len(self.queue)
        idx = (self.ptr + np.arange(n)) % K
        self.queue[idx] = Zk
        self.ptr = (self.ptr + n) % K
        return loss

    def embed(self, X):
        return self.enc.embed(X)


def train_moco(imgs, seed=0, steps=2000, batch=32, **kw):
    mo = MoCo(seed=seed, **kw)
    r1 = np.random.default_rng(1)
    for _ in range(steps):
        idx = r1.integers(0, len(imgs), batch)
        Xq = augment(imgs[idx], r1)
        Xk = augment(imgs[idx], r1)
        mo.step(Xq, Xk)
    return mo


def gradcheck(seed=3):
    rng = np.random.default_rng(seed)
    mo = MoCo(hidden=8, d_emb=4, seed=seed, queue_k=7)
    Xq = rng.standard_normal((5, D_IN))
    Xk = rng.standard_normal((5, D_IN))
    _, grads, _ = mo.loss_and_grads(Xq, Xk)
    worst = 0.0
    r = np.random.default_rng(seed + 1)
    for P_, Gr in zip(mo.enc.net.params, grads):
        flat, gflat = P_.reshape(-1), Gr.reshape(-1)
        for ix in r.choice(len(flat), min(8, len(flat)), replace=False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp, _, _ = mo.loss_and_grads(Xq, Xk)
            flat[ix] = old - 1e-6
            lm, _, _ = mo.loss_and_grads(Xq, Xk)
            flat[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
    return worst


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo1(imgs, y, te_y, Xtr, Xte, few):
    banner("DEMO 1 --- A crowd that does not live in the batch")
    print("  Each query is scored against ONE positive key and a queue")
    print("  of the last 1,024 keys -- embeddings banked from previous")
    print("  steps, encoded by a slow EMA copy of the encoder. The")
    print("  batch can be tiny; the crowd stays huge:")
    print()
    mo32 = train_moco(imgs, steps=16000, batch=32)
    a15 = probe_acc(mo32.embed(Xtr)[few], y[few], mo32.embed(Xte), te_y)
    aall = probe_acc(mo32.embed(Xtr), y, mo32.embed(Xte), te_y)
    print(f"    MoCo, batch 32, queue 1024 : {a15:6.1%} / {aall:6.1%}")
    mo8 = train_moco(imgs, steps=32000, batch=8)
    a15b = probe_acc(mo8.embed(Xtr)[few], y[few], mo8.embed(Xte), te_y)
    aallb = probe_acc(mo8.embed(Xtr), y, mo8.embed(Xte), te_y)
    print(f"    MoCo, batch  8, queue 1024 : {a15b:6.1%} / {aallb:6.1%}")
    print()
    print("  (Part 1's NT-Xent at batch 256 scored 95.1% / 99.7%.)")
    print("  A batch of eight walks through training with a crowd of a")
    print("  thousand at its back -- negatives fully decoupled from the")
    print("  batch, at the cost of a queue and a second set of weights.")
    return mo32


def demo2(imgs, y, te_y, Xtr, Xte, few):
    banner("DEMO 2 --- The consistency law")
    print("  The queue spans thousands of steps, so its entries were")
    print("  written by OLD versions of the encoder. Encode the keys")
    print("  with the live network (momentum 0) and the ledger is")
    print("  incoherent; encode them with a slow EMA twin and it holds")
    print("  together. Same budget, only the key momentum moves:")
    print()
    print("    key momentum    probe15   probe-all")
    for m_key in (0.0, 0.9, 0.99, 0.999):
        mo = train_moco(imgs, m_key=m_key)
        a15 = probe_acc(mo.embed(Xtr)[few], y[few], mo.embed(Xte), te_y)
        aall = probe_acc(mo.embed(Xtr), y, mo.embed(Xte), te_y)
        print(f"       {m_key:5.3f}       {a15:6.1%}    {aall:6.1%}")
    print()
    print("  Momentum 0 is CHANCE -- not weak, dead. The queue itself is")
    print("  a data structure any of Section 00's articles could have")
    print("  supplied; a queue whose thousand entries stay mutually")
    print("  comparable while the network that wrote them keeps moving")
    print("  -- that is the invention, and it is Part 3's slow twin")
    print("  pointed backwards: DINO's teacher makes the TARGETS")
    print("  stable; MoCo's key encoder makes the LEDGER stable.")


def demo3(imgs, y, te_y, Xtr, Xte, few):
    banner("DEMO 3 --- The honest audit: does the queue pay HERE?")
    print("  Queue size, same budget:")
    print()
    print("    queue     probe15   probe-all")
    for qk in (16, 256, 4096):
        mo = train_moco(imgs, queue_k=qk)
        a15 = probe_acc(mo.embed(Xtr)[few], y[few], mo.embed(Xte), te_y)
        aall = probe_acc(mo.embed(Xtr), y, mo.embed(Xte), te_y)
        print(f"    {qk:5d}      {a15:6.1%}    {aall:6.1%}")
    print()
    print("  Flat. Sixteen negatives do what four thousand do, because")
    print("  three shape classes saturate a crowd early. And the control")
    print("  experiment: plain NT-Xent at batch 8 -- fourteen in-batch")
    print("  negatives, no queue, no twin -- given the same long budget:")
    print()
    e8 = train_ntxent(imgs, steps=32000, batch=8)
    a15 = probe_acc(e8.embed(Xtr)[few], y[few], e8.embed(Xte), te_y)
    aall = probe_acc(e8.embed(Xtr), y, e8.embed(Xte), te_y)
    print(f"    NT-Xent, batch 8, 32k steps: {a15:6.1%} / {aall:6.1%}")
    print()
    print("  It matches MoCo. On shape world the queue buys nothing --")
    print("  and that is the finding, stated honestly: MoCo solves a")
    print("  SCALE problem. Its machinery pays off where useful crowds")
    print("  number in the tens of thousands and no device holds the")
    print("  batch -- ImageNet, not a 16x16 toy. Third time this section")
    print("  has caught folklore not transferring (MAE's 75%, REINFORCE,")
    print("  now the queue): measure before you inherit machinery.")


def demo4(imgs, y, te_y, Xtr, Xte, few, mo32):
    banner("DEMO 4 --- The family, closed")
    print("  All four methods of this section, canonical configs from")
    print("  their own articles, retrained live on the same 4,000")
    print("  unlabelled images:")
    print()
    enc = train_ntxent(imgs)
    Xtr_flat = imgs.reshape(len(imgs), -1)
    mae = MAE(seed=0)
    rm = np.random.default_rng(1)
    for _ in range(3000):
        idx = rm.integers(0, len(imgs), BATCH)
        mae.step(Xtr_flat[idx], rm)
    di = DINO(seed=0)
    rd = np.random.default_rng(1)
    for _ in range(3000):
        idx = rd.integers(0, len(imgs), BATCH)
        Xa = augment(imgs[idx], rd)
        Xb = augment(imgs[idx], rd)
        di.step(Xa, Xb)
    print("                          probe15   probe-all   yardstick")
    rows = [
        ("SimCLR / NT-Xent (P1) ", enc, "the batch beside you"),
        ("MAE (P2)              ", mae, "the data itself"),
        ("DINO (P3)             ", di, "a slow twin's targets"),
        ("MoCo (P4)             ", mo32, "a ledger, kept by a twin"),
    ]
    for name, model, yard in rows:
        a15 = probe_acc(model.embed(Xtr)[few], y[few],
                        model.embed(Xte), te_y)
        aall = probe_acc(model.embed(Xtr), y, model.embed(Xte), te_y)
        print(f"    {name} {a15:6.1%}    {aall:6.1%}    {yard}")
    print()
    print("  Every self-supervised method in this section is one answer")
    print("  to the same question: when two views of the world should")
    print("  agree, where does the yardstick for 'agree' come from?")
    print("  SimCLR buys a crowd per batch. MoCo banks the crowd and")
    print("  keeps the bank honest with a slow twin. DINO drops the")
    print("  crowd and chases the twin directly. MAE never compares to")
    print("  anyone -- the missing three quarters of the image is")
    print("  yardstick enough. Four pretexts, one stage, no labels:")
    print("  the section closes where its data began -- with the")
    print("  structure of the data doing the teaching.")


def main() -> None:
    print(f"  [audit] MoCo backward (query side, through the queue")
    print(f"          softmax) vs central differences: {gradcheck():.2e}")
    rng = np.random.default_rng(0)
    imgs, y, factors = make_instances(N_UNLAB, rng)
    te_imgs, te_y, te_f = make_instances(N_TEST,
                                         np.random.default_rng(50))
    Xtr = augment(imgs, np.random.default_rng(54))
    Xte = augment(te_imgs, np.random.default_rng(55))
    r = np.random.default_rng(2)
    few = np.concatenate([r.choice(np.where(y == c)[0], 5, False)
                          for c in range(3)])
    mo32 = demo1(imgs, y, te_y, Xtr, Xte, few)
    demo2(imgs, y, te_y, Xtr, Xte, few)
    demo3(imgs, y, te_y, Xtr, Xte, few)
    demo4(imgs, y, te_y, Xtr, Xte, few, mo32)


if __name__ == "__main__":
    main()
