"""
dino.py --- companion code for "Self-Distillation (DINO)"
(Self-Supervised Learning, Part 3).

Part 1 ended on a measured cliffhanger: keep the pull (make two
views of the same image agree) and delete the crowd of negatives,
and the embedding collapses ~150-fold -- every image mapped to one
point, loss perfectly minimised, representation worthless. DINO
(Caron et al., 2021) walks exactly that tightrope and does not
fall. Two networks, STUDENT and TEACHER, same architecture. The
teacher is not trained at all: it is an exponential moving average
of the student -- the student's own past, distilled. Each step:

    teacher sees view A  ->  soft targets (sharpened, centred)
    student sees view B  ->  cross-entropy toward the targets
    (and symmetrically, B -> A)

No negatives, no contrast, no reconstruction. What stops the
collapse is a pair of opposing tweaks to the teacher's softmax:
CENTRING subtracts a running mean of teacher logits (kills the
run-to-one-prototype solution) and SHARPENING uses a lower
temperature than the student (kills the everything-uniform
solution). Either alone fails in its own characteristic way; the
two together balance.

The stage is shape world, and the script retrains Part 1's
contrastive encoder AND Part 1's pull-only collapse live, so all
three sit on one measured table.

Demonstrates:
  1. Agreement without the crowd. DINO reaches 94.5% on the
     15-label probe and 99.9% with all labels -- matching the
     NT-Xent contrastive encoder (95.1% / 99.7%) with not a
     single negative -- while the pull-only baseline collapses
     to a normalised spread of 0.0011 on the same budget.
  2. The tightrope, as a 2x2. Centring alone: UNIFORM collapse
     -- per-sample entropy pinned at log K = 3.47, probe 54.7%.
     Sharpening alone: ONE-HOT collapse -- per-sample entropy
     0.00, most mass on one prototype, probe 69.8%. Neither:
     collapse. Both: confident AND diverse (entropy 0.14 per
     sample, 2.09 across the batch), probe 99.9%. The DINO
     paper's ablation, reproduced to the entropy.
  3. The teacher's pace. Momentum 0 (teacher = student) breaks;
     0.9-0.99 works; 0.999 is too slow for this budget. And an
     honest check: the folklore that the EMA teacher OUTPERFORMS
     its student does not reproduce here -- the teacher trails
     early (54.7% vs 59.2% at step 100) and converges to a tie.
     Its job at this scale is not to be better; it is to be
     STABLE -- a target that moves slowly enough to be chased.

Everything is plain NumPy, gradients hand-derived and audited.
Dependencies: numpy. Runs in about seven minutes.
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
STEPS = 3000
BATCH = 256
K_PROTO = 32
T_S = 0.2
T_T = 0.05
EMA_NET = 0.99
EMA_C = 0.9
ROWIDX = np.arange(SIDE)


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# The stage: shape world, identical to Parts 1-2.
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
# Plumbing, identical to Parts 1-2.
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


def norm_spread(Z):
    """Spread of L2-normalised embeddings -- comparable across models."""
    Zn = Z / (np.sqrt((Z ** 2).sum(1, keepdims=True)) + 1e-12)
    return float(Zn.std(0).mean())


# ---------------------------------------------------------------------------
# DINO: student, EMA teacher, centred + sharpened targets.
# ---------------------------------------------------------------------------

def softmax_T(L, T):
    Z = L / T
    Z -= Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


class DINO:
    def __init__(self, hidden=HID, d_emb=D_EMB, k=K_PROTO, seed=0,
                 lr=1e-3, ema_net=EMA_NET, center=True, sharpen=True):
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
        self.center = center
        self.sharpen = sharpen

    def student_logits(self, X, cache=False):
        H, ch = self.net.forward(X)
        L = H @ self.Wp + self.bp
        if cache:
            return L, H, ch
        return L

    def teacher_logits(self, X):
        saved = [p.copy() for p in self.net.params]
        for p, t in zip(self.net.params, self.t_params):
            p[...] = t
        H, _ = self.net.forward(X)
        for p, s in zip(self.net.params, saved):
            p[...] = s
        return H @ self.t_Wp + self.t_bp

    def teacher_probs(self, X):
        L = self.teacher_logits(X)
        Lc = L - self.c if self.center else L
        T = T_T if self.sharpen else T_S
        return softmax_T(Lc, T), L

    def loss_and_grads(self, Xa, Xb):
        """Symmetric distillation loss and student grads (teacher and
        centre frozen -- their updates are not gradient steps)."""
        n = len(Xa)
        Pt_a, La = self.teacher_probs(Xa)
        Pt_b, Lb = self.teacher_probs(Xb)
        tot_gnet, tot_gp, loss = None, None, 0.0
        for Xs, Pt in ((Xb, Pt_a), (Xa, Pt_b)):
            Ls, H, ch = self.student_logits(Xs, cache=True)
            Ps = softmax_T(Ls, T_S)
            loss += float(-(Pt * np.log(Ps + 1e-12)).sum(1).mean()) / 2
            dL = (Ps - Pt) / T_S / (2 * n)
            gWp = H.T @ dL
            gbp = dL.sum(0)
            dH = dL @ self.Wp.T
            gnet, _ = self.net.backward(ch, dH)
            tot_gnet = gnet if tot_gnet is None else [
                a + b for a, b in zip(tot_gnet, gnet)]
            tot_gp = [gWp, gbp] if tot_gp is None else [
                tot_gp[0] + gWp, tot_gp[1] + gbp]
        return loss, tot_gnet, tot_gp, La, Lb

    def step(self, Xa, Xb):
        loss, gnet, gp, La, Lb = self.loss_and_grads(Xa, Xb)
        self.net.opt.step(gnet)
        self.opt_p.step(gp)
        m = self.ema_net
        for t, p in zip(self.t_params, self.net.params):
            t *= m
            t += (1 - m) * p
        self.t_Wp = m * self.t_Wp + (1 - m) * self.Wp
        self.t_bp = m * self.t_bp + (1 - m) * self.bp
        self.c = EMA_C * self.c + (1 - EMA_C) * \
            np.concatenate([La, Lb]).mean(0)
        return loss

    def embed(self, X, teacher=True):
        if teacher:
            saved = [p.copy() for p in self.net.params]
            for p, t in zip(self.net.params, self.t_params):
                p[...] = t
            H, _ = self.net.forward(X)
            for p, s in zip(self.net.params, saved):
                p[...] = s
            return H
        return self.net.forward(X)[0]


def collapse_stats(d, X):
    Pt, _ = d.teacher_probs(X)
    per = float(-(Pt * np.log(Pt + 1e-12)).sum(1).mean())
    mean_p = Pt.mean(0)
    of_mean = float(-(mean_p * np.log(mean_p + 1e-12)).sum())
    return per, of_mean


def train_dino(imgs, seed=0, steps=STEPS, **kw):
    d = DINO(seed=seed, **kw)
    r1 = np.random.default_rng(1)
    for _ in range(steps):
        idx = r1.integers(0, len(imgs), BATCH)
        Xa = augment(imgs[idx], r1)
        Xb = augment(imgs[idx], r1)
        d.step(Xa, Xb)
    return d


# ---------------------------------------------------------------------------
# Audit: student gradients at fixed teacher/centre.
# ---------------------------------------------------------------------------

def gradcheck(seed=3):
    rng = np.random.default_rng(seed)
    d = DINO(hidden=8, d_emb=4, k=5, seed=seed)
    Xa = rng.standard_normal((5, D_IN))
    Xb = rng.standard_normal((5, D_IN))
    _, gnet, gp, _, _ = d.loss_and_grads(Xa, Xb)

    def loss_only():
        return d.loss_and_grads(Xa, Xb)[0]

    worst = 0.0
    r = np.random.default_rng(seed + 1)
    for params, grads in ((d.net.params, gnet),
                          ([d.Wp, d.bp], gp)):
        for P_, Gr in zip(params, grads):
            flat, gflat = P_.reshape(-1), Gr.reshape(-1)
            for ix in r.choice(len(flat), min(8, len(flat)),
                               replace=False):
                old = flat[ix]
                flat[ix] = old + 1e-6
                lp = loss_only()
                flat[ix] = old - 1e-6
                lm = loss_only()
                flat[ix] = old
                worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
    return worst


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo1(imgs, y, te_y, Xtr, Xte, few):
    banner("DEMO 1 --- Agreement without the crowd")
    print("  Part 1 measured what SHOULD happen here: pull two views")
    print("  together with no negatives pushing back, and the embedding")
    print("  collapses to a point. DINO pulls with no negatives anyway")
    print("  -- toward a slow-moving EMA teacher whose targets are")
    print("  centred and sharpened. Training, with teacher-student")
    print("  probes along the way:")
    print()
    d = DINO(seed=0)
    r1 = np.random.default_rng(1)
    print("     step    teacher probe   student probe")
    for s in range(STEPS):
        idx = r1.integers(0, len(imgs), BATCH)
        Xa = augment(imgs[idx], r1)
        Xb = augment(imgs[idx], r1)
        d.step(Xa, Xb)
        if (s + 1) in (100, 500, 1000, 3000):
            at = probe_acc(d.embed(Xtr), y, d.embed(Xte), te_y)
            as_ = probe_acc(d.embed(Xtr, teacher=False), y,
                            d.embed(Xte, teacher=False), te_y)
            print(f"    {s + 1:5d}       {at:6.1%}          {as_:6.1%}")
    # rivals, retrained live
    enc = Encoder(seed=0)
    rc = np.random.default_rng(1)
    for _ in range(2000):
        idx = rc.integers(0, len(imgs), BATCH)
        Xa = augment(imgs[idx], rc)
        Xb = augment(imgs[idx], rc)
        _, grads = ntxent_grads(enc, Xa, Xb)
        enc.net.opt.step(grads)
    pull = Encoder(seed=0)
    rp = np.random.default_rng(1)
    for _ in range(2000):
        idx = rp.integers(0, len(imgs), BATCH)
        Xa = augment(imgs[idx], rp)
        Xb = augment(imgs[idx], rp)
        Za, ca_ = pull.embed(Xa, cache=True)
        Zb, cb_ = pull.embed(Xb, cache=True)
        df = Za - Zb
        ga = pull.backward_emb(ca_, 2 * df / len(Xa))
        gb = pull.backward_emb(cb_, -2 * df / len(Xa))
        pull.net.opt.step([a + b for a, b in zip(ga, gb)])
    r5 = np.random.default_rng(2)
    print()
    print("                            probe15   probe-all   spread")
    rows = [
        ("pull-only (Part 1)   ", pull.embed(Xtr), pull.embed(Xte)),
        ("DINO (this article)  ", d.embed(Xtr), d.embed(Xte)),
        ("contrastive (Part 1) ", enc.embed(Xtr), enc.embed(Xte)),
    ]
    for name, Ztr, Zte in rows:
        a15 = probe_acc(Ztr[few], y[few], Zte, te_y)
        aall = probe_acc(Ztr, y, Zte, te_y)
        print(f"    {name}  {a15:6.1%}    {aall:6.1%}     "
              f"{norm_spread(Zte):.4f}")
    print()
    print("  The pull-only row is Part 1's collapse, reproduced. DINO")
    print("  is the same pull -- no negatives anywhere -- and it matches")
    print("  the full contrastive encoder. The crowd turned out to be")
    print("  optional; what was NOT optional is the subject of DEMO 2.")
    return d


def demo2(imgs, y, te_y, Xtr, Xte):
    banner("DEMO 2 --- The tightrope: centring x sharpening")
    print("  Two failure modes wait on either side. Remove SHARPENING")
    print("  and the safest agreement is the uniform distribution --")
    print("  every image, every prototype, equal mass. Remove CENTRING")
    print("  and the safest agreement is a single prototype -- every")
    print("  image, one answer. H(per) = mean per-sample entropy;")
    print("  H(mean) = entropy of the average prediction; log K = 3.47:")
    print()
    print("    recipe             probe-all   H(per)   H(mean)   verdict")
    for center, sharpen, name in (
            (True, True, "centre + sharpen"),
            (True, False, "centre only     "),
            (False, True, "sharpen only    "),
            (False, False, "neither         ")):
        d2 = train_dino(imgs, center=center, sharpen=sharpen)
        aall = probe_acc(d2.embed(Xtr), y, d2.embed(Xte), te_y)
        per, om = collapse_stats(d2, Xte[:500])
        if per > 3.0 and om > 3.0:
            verdict = "uniform collapse"
        elif per < 0.05 and om < 1.0:
            verdict = "one-hot collapse"
        elif per > 3.0:
            verdict = "collapse"
        else:
            verdict = "healthy"
        print(f"    {name}    {aall:6.1%}     {per:5.2f}    {om:5.2f}"
              f"    {verdict}")
    print()
    print("  Centring alone: per-sample entropy pinned at log K -- the")
    print("  teacher says 'anything is fine' and the student agrees to")
    print("  say nothing. Sharpening alone: entropy zero, one prototype")
    print("  eats the batch -- confident and empty. Together they hold")
    print("  the only working point: confident about each image (0.14),")
    print("  diverse across images (2.09). The tightrope is real, it is")
    print("  two-sided, and both guy-ropes are cheap: a running mean")
    print("  and a temperature.")


def demo3(imgs, y, te_y, Xtr, Xte):
    banner("DEMO 3 --- The teacher's pace")
    print("  The teacher is the student's own past, averaged with")
    print("  momentum m. How much past is right, on this budget?")
    print()
    print("    momentum    probe-all    spread")
    for m_ema in (0.0, 0.9, 0.99, 0.999):
        d3 = train_dino(imgs, ema_net=m_ema)
        aall = probe_acc(d3.embed(Xtr), y, d3.embed(Xte), te_y)
        print(f"     {m_ema:5.3f}      {aall:6.1%}     "
              f"{norm_spread(d3.embed(Xte)):.4f}")
    print()
    print("  Momentum 0 -- the teacher IS the student -- and the chase")
    print("  degenerates: a target that moves with every step teaches")
    print("  little. 0.9 to 0.99 works; 0.999 remembers too much for a")
    print("  3,000-step run. And note the honest reading of DEMO 1's")
    print("  trace: the folklore says the EMA teacher outperforms its")
    print("  student, but here the teacher TRAILS early (54.7% vs 59.2%")
    print("  at step 100) and converges to a tie. At this scale the")
    print("  teacher's job is not to be better. It is to be STABLE --")
    print("  slow enough to be chased, which is what the momentum-0 row")
    print("  loses. BYOL, SimSiam and DINO differ in decoration, but")
    print("  every negative-free method that works has this one thing:")
    print("  an asymmetry that keeps the target out of lockstep with")
    print("  the learner.")


def main() -> None:
    print(f"  [audit] DINO student backward (through both softmaxes) vs")
    print(f"          central differences: {gradcheck():.2e}")
    rng = np.random.default_rng(0)
    imgs, y, factors = make_instances(N_UNLAB, rng)
    te_imgs, te_y, te_f = make_instances(N_TEST,
                                         np.random.default_rng(50))
    Xtr = augment(imgs, np.random.default_rng(54))
    Xte = augment(te_imgs, np.random.default_rng(55))
    r = np.random.default_rng(2)
    few = np.concatenate([r.choice(np.where(y == c)[0], 5, False)
                          for c in range(3)])
    demo1(imgs, y, te_y, Xtr, Xte, few)
    demo2(imgs, y, te_y, Xtr, Xte)
    demo3(imgs, y, te_y, Xtr, Xte)


if __name__ == "__main__":
    main()
