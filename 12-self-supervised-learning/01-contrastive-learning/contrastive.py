"""
contrastive.py --- companion code for "Contrastive Learning"
(Self-Supervised Learning, Part 1).

Section 11 built architectures that classify and architectures that
create. Every one of them either needed labels or needed nothing --
this section is about the third way: models that build their own
supervision out of raw, unlabelled data. Contrastive learning
(SimCLR: Chen et al., 2020; CLIP: Radford et al., 2021) makes the
move with a disarmingly simple rule: two AUGMENTED VIEWS of the
same example should agree, and views of different examples should
not. The loss (NT-Xent / InfoNCE) is just softmax classification
where the "class" is: which of these 2N views is your sibling?

    L_i = -log  exp(z_i . z_j / tau) / sum_k exp(z_i . z_k / tau)

The deep content is not the loss; it is the AUGMENTATION MENU.
Declaring "a shifted, brightened, noised copy is still the same
thing" is a statement about what should NOT matter -- and the
encoder learns to be invariant to exactly what the menu varies,
and to keep exactly what it preserves. The augmentations are the
supervision.

The stage (new for this section): "shape world" -- 16x16 images of
three shapes (cross, hollow square, diagonal X) at random position,
size and brightness, plus pixel noise. Augmentations: random shift,
brightness jitter, fresh noise. No label is used for pretraining.

Demonstrates:
  1. The promise. Contrastive pretraining on 4,000 unlabelled
     images, then a linear probe on FIFTEEN labels, lands within
     five points of full supervision on all 4,000 (95.1% vs
     99.7%; 94.4% mean over five label draws) -- while the same
     encoder trained supervised on those 15 labels alone manages
     84.9%. The unlabelled pile did the work.
  2. The menu is the supervision. Remove the shift augmentation
     and the class probe collapses from ~100% to ~52% -- yet
     nearest-neighbour purity stays ~98%: the encoder still
     separates instances, but it now files images by WHERE they
     are as much as by WHAT they are (neighbour shares position
     quadrant ~97%). Invariance, not information, is what the
     augmentation buys; without it the classes stop being
     linearly readable.
  3. Collapse. Remove the negatives (pull positives together,
     nothing else) and the embedding's spread across examples
     falls ~150x -- every image maps to the same point, loss
     perfectly minimised, representation worthless. The crowd
     of negatives is what holds the space open (and the whole
     BYOL/DINO literature -- Part 3 -- is about surviving
     without it).
  4. Mini-CLIP. A second encoder embeds 7-number "captions"
     (class one-hot + position + size + brightness); the
     symmetric InfoNCE aligns the two towers. Zero-shot
     classification via prototype captions: 95.1%. Badly-worded
     prompts (corner position) drop it to 68.0% -- prompt
     engineering, measured. Caption-to-image retrieval among
     1,000 candidates: median rank 15.

Everything is plain NumPy, gradients hand-derived (through the
L2-normalisation and both InfoNCE losses) and audited at startup.
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
STEPS = 2000
BATCH = 256
ROWIDX = np.arange(SIDE)


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# The stage: shape world.
# ---------------------------------------------------------------------------

def draw_shape(cls, cx, cy, size, bright):
    img = np.zeros((SIDE, SIDE))
    s = int(size)
    if cls == 0:                     # cross
        img[cy - s:cy + s + 1, cx] = 1.0
        img[cy, cx - s:cx + s + 1] = 1.0
    elif cls == 1:                   # hollow square
        img[cy - s:cy + s + 1, cx - s] = 1.0
        img[cy - s:cy + s + 1, cx + s] = 1.0
        img[cy - s, cx - s:cx + s + 1] = 1.0
        img[cy + s, cx - s:cx + s + 1] = 1.0
    else:                            # diagonal X
        for d in range(-s, s + 1):
            img[cy + d, cx + d] = 1.0
            img[cy + d, cx - d] = 1.0
    return img * bright


def make_instances(n, rng):
    y = rng.integers(0, 3, n)
    imgs = np.zeros((n, SIDE, SIDE))
    factors = np.zeros((n, 4))       # cx, cy, size, brightness
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
# Plumbing, as in Section 11.
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
    """in -> h -> h -> out, leaky-relu hidden, linear out; hand backprop."""

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
    """Net + L2 normalisation onto the unit sphere."""

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


# ---------------------------------------------------------------------------
# The losses: NT-Xent (one tower) and symmetric InfoNCE (two towers).
# ---------------------------------------------------------------------------

def ntxent_grads(enc, Xa, Xb, tau=TAU):
    """Views Xa[i] and Xb[i] are the positive pair; everyone else in
    the doubled batch is a negative."""
    n = len(Xa)
    Za, ca = enc.embed(Xa, cache=True)
    Zb, cb = enc.embed(Xb, cache=True)
    Z = np.concatenate([Za, Zb])
    S = Z @ Z.T / tau
    np.fill_diagonal(S, -1e9)
    pos = np.concatenate([np.arange(n, 2 * n), np.arange(n)])
    E = np.exp(S - S.max(1, keepdims=True))
    P = E / E.sum(1, keepdims=True)
    loss = float(-np.log(P[np.arange(2 * n), pos] + 1e-300).mean())
    G = P.copy()
    G[np.arange(2 * n), pos] -= 1
    G /= (2 * n)
    dZ = ((G + G.T) @ Z) / tau
    ga = enc.backward_emb(ca, dZ[:n])
    gb = enc.backward_emb(cb, dZ[n:])
    return loss, [a + b for a, b in zip(ga, gb)]


def clip_grads(img_enc, txt_enc, Xi, Ct, tau=TAU):
    """CLIP's symmetric loss: images classify captions and captions
    classify images, positives on the diagonal."""
    n = len(Xi)
    Zi, ci = img_enc.embed(Xi, cache=True)
    Zt, ct = txt_enc.embed(Ct, cache=True)
    S = Zi @ Zt.T / tau
    E = np.exp(S - S.max(1, keepdims=True))
    P = E / E.sum(1, keepdims=True)
    S2 = S.T
    E2 = np.exp(S2 - S2.max(1, keepdims=True))
    P2 = E2 / E2.sum(1, keepdims=True)
    diag = np.arange(n)
    loss = float(-(np.log(P[diag, diag] + 1e-300).mean()
                   + np.log(P2[diag, diag] + 1e-300).mean()) / 2)
    G = P.copy()
    G[diag, diag] -= 1
    G2 = P2.copy()
    G2[diag, diag] -= 1
    dS = (G + G2.T) / (2 * n)
    gi = img_enc.backward_emb(ci, dS @ Zt / tau)
    gt = txt_enc.backward_emb(ct, dS.T @ Zi / tau)
    return loss, gi, gt


def pretrain(enc, imgs, rng, steps=STEPS, batch=BATCH, augs=None,
             tau=TAU):
    augs = augs or {}
    for _ in range(steps):
        idx = rng.integers(0, len(imgs), batch)
        Xa = augment(imgs[idx], rng, **augs)
        Xb = augment(imgs[idx], rng, **augs)
        _, grads = ntxent_grads(enc, Xa, Xb, tau=tau)
        enc.net.opt.step(grads)
    return enc


# ---------------------------------------------------------------------------
# Probes and yardsticks.
# ---------------------------------------------------------------------------

def train_linear_probe(Z, y, n_cls, steps=300, lr=0.1, seed=0):
    r = np.random.default_rng(seed)
    W = r.standard_normal((Z.shape[1], n_cls)) * 0.01
    b = np.zeros(n_cls)
    opt = Adam([W, b], lr)
    for _ in range(steps):
        L = Z @ W + b
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        P /= P.sum(1, keepdims=True)
        G = P.copy()
        G[np.arange(len(y)), y] -= 1
        G /= len(y)
        opt.step([Z.T @ G, G.sum(0)])
    return W, b


def probe_acc(Ztr, ytr, Zte, yte, n_cls=3):
    W, b = train_linear_probe(Ztr, ytr, n_cls)
    return float(((Zte @ W + b).argmax(1) == yte).mean())


def nn_purity(Z, y, quad):
    S = Z @ Z.T
    np.fill_diagonal(S, -1e9)
    nn = S.argmax(1)
    return float((y[nn] == y).mean()), float((quad[nn] == quad).mean())


# ---------------------------------------------------------------------------
# Audits.
# ---------------------------------------------------------------------------

def gradcheck(seed=3):
    rng = np.random.default_rng(seed)
    enc = Encoder(hidden=8, d_emb=4, seed=seed)
    Xa = rng.standard_normal((5, D_IN))
    Xb = rng.standard_normal((5, D_IN))
    _, grads = ntxent_grads(enc, Xa, Xb)
    worst = 0.0
    r = np.random.default_rng(seed + 1)
    for P_, Gr in zip(enc.net.params, grads):
        flat, gflat = P_.reshape(-1), Gr.reshape(-1)
        for ix in r.choice(len(flat), min(8, len(flat)), replace=False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp, _ = ntxent_grads(enc, Xa, Xb)
            flat[ix] = old - 1e-6
            lm, _ = ntxent_grads(enc, Xa, Xb)
            flat[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
    return worst


def gradcheck_clip(seed=4):
    rng = np.random.default_rng(seed)
    ie = Encoder(hidden=8, d_emb=4, seed=seed)
    te = Encoder(d_in=7, hidden=8, d_emb=4, seed=seed + 100)
    Xi = rng.standard_normal((5, D_IN))
    Ct = rng.standard_normal((5, 7))
    _, gi, gt = clip_grads(ie, te, Xi, Ct)
    worst = 0.0
    r = np.random.default_rng(seed + 1)
    for enc, grads in ((ie, gi), (te, gt)):
        for P_, Gr in zip(enc.net.params, grads):
            flat, gflat = P_.reshape(-1), Gr.reshape(-1)
            for ix in r.choice(len(flat), min(8, len(flat)),
                               replace=False):
                old = flat[ix]
                flat[ix] = old + 1e-6
                lp, _, _ = clip_grads(ie, te, Xi, Ct)
                flat[ix] = old - 1e-6
                lm, _, _ = clip_grads(ie, te, Xi, Ct)
                flat[ix] = old
                worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
    return worst


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo1(imgs, y, te_imgs, te_y):
    banner("DEMO 1 --- The promise: fifteen labels")
    print("  Pretrain on 4,000 UNLABELLED shape images: two augmented")
    print("  views per image, NT-Xent, no label anywhere. Then freeze")
    print("  the encoder and train a linear probe -- a single softmax")
    print("  layer -- on just 5 labels per class. Against it: the same")
    print("  encoder architecture trained supervised, with the same")
    print("  augmentations, on the same 15 labels; and the ceiling,")
    print("  supervised on all 4,000:")
    print()
    enc = Encoder(seed=0)
    pretrain(enc, imgs, np.random.default_rng(1))
    Xtr = augment(imgs, np.random.default_rng(54))
    Xte = augment(te_imgs, np.random.default_rng(55))
    Ztr = enc.embed(Xtr)
    Zte = enc.embed(Xte)
    r = np.random.default_rng(2)
    few = np.concatenate([r.choice(np.where(y == c)[0], 5, False)
                          for c in range(3)])
    accs5 = []
    for dd in range(5):
        rd = np.random.default_rng(2 + dd)
        fd = np.concatenate([rd.choice(np.where(y == c)[0], 5, False)
                             for c in range(3)])
        accs5.append(probe_acc(Ztr[fd], y[fd], Zte, te_y))
    rows = [("contrastive + probe, 15 labels ",
             probe_acc(Ztr[few], y[few], Zte, te_y)),
            ("contrastive + probe, all labels",
             probe_acc(Ztr, y, Zte, te_y))]
    for idx, name in ((few, "supervised, 15 labels          "),
                      (np.arange(len(imgs)),
                       "supervised, all 4,000 labels   ")):
        sup = Encoder(seed=0)
        W = np.random.default_rng(3).standard_normal((D_EMB, 3)) * 0.01
        b = np.zeros(3)
        opt = Adam([W, b], 1e-3)
        r2 = np.random.default_rng(4)
        for _ in range(STEPS):
            bidx = idx[r2.integers(0, len(idx), min(BATCH, len(idx)))]
            Xb_ = augment(imgs[bidx], r2)
            Z, cach = sup.embed(Xb_, cache=True)
            L = Z @ W + b
            L -= L.max(1, keepdims=True)
            P = np.exp(L)
            P /= P.sum(1, keepdims=True)
            G = P.copy()
            G[np.arange(len(bidx)), y[bidx]] -= 1
            G /= len(bidx)
            genc = sup.backward_emb(cach, G @ W.T)
            sup.net.opt.step(genc)
            opt.step([Z.T @ G, G.sum(0)])
        Zs = sup.embed(Xte)
        rows.append((name, float(((Zs @ W + b).argmax(1)
                                  == te_y).mean())))
    for name, acc in rows:
        print(f"    {name}: {acc:6.1%}")
    print()
    print(f"  (Robustness: over five different draws of the 15 labels")
    print(f"   the probe averages {np.mean(accs5):.1%}, "
          f"range {min(accs5):.1%}-{max(accs5):.1%}.)")
    print()
    print("  Fifteen labels plus four thousand unlabelled images land")
    print("  within five points of four thousand labels; the same")
    print("  fifteen labels alone stop ten points lower still. The")
    print("  unlabelled pile did the work -- the whole economic")
    print("  argument for this section in four rows.")
    return enc, Xte, Zte


def demo2(imgs, y, factors, te_imgs, te_y, te_f):
    banner("DEMO 2 --- The augmentation menu IS the supervision")
    print("  Declaring 'a shifted copy is the same image' is a claim")
    print("  about what should not matter. Withdraw it and watch what")
    print("  the encoder decides to care about. Class probe on augmented")
    print("  views; nearest-neighbour checks on clean (unshifted) views:")
    print()
    quad_te = ((te_f[:, 0] > SIDE / 2).astype(int) * 2
               + (te_f[:, 1] > SIDE / 2).astype(int))
    Xte_pos = augment(te_imgs, np.random.default_rng(53), shift=False)
    Xtr = augment(imgs, np.random.default_rng(54))
    Xte = augment(te_imgs, np.random.default_rng(55))
    print("    menu          class probe   NN same class   NN same quadrant")
    for name, augs in [("full menu ", {}),
                       ("no shift  ", dict(shift=False)),
                       ("noise only", dict(shift=False, jitter=False))]:
        e = Encoder(seed=0)
        pretrain(e, imgs, np.random.default_rng(1), augs=augs)
        ca = probe_acc(e.embed(Xtr), y, e.embed(Xte), te_y)
        cp, qp = nn_purity(e.embed(Xte_pos), te_y, quad_te)
        print(f"    {name}     {ca:6.1%}        {cp:6.1%}          "
              f"{qp:6.1%}")
    print()
    print("  (Chance: class 33%, quadrant ~25%.) Remove the shift and")
    print("  the class probe halves -- but look at the neighbours: the")
    print("  encoder STILL separates instances almost perfectly. It has")
    print("  simply filed them by where they are as much as by what")
    print("  they are (quadrant purity 97%). The information never")
    print("  disappeared; the INVARIANCE did, and with it the linear")
    print("  readability of class. Augmentation does not add signal --")
    print("  it declares which differences are noise, and the encoder")
    print("  believes the declaration exactly.")


def demo3(imgs, enc_full, Xte):
    banner("DEMO 3 --- Take away the crowd and the space collapses")
    print("  The NT-Xent denominator is a crowd of negatives: every")
    print("  other image in the batch pushes back. Keep the pull (make")
    print("  positive views agree) and delete the push, same budget:")
    print()
    e3 = Encoder(seed=0)
    r3 = np.random.default_rng(1)
    for _ in range(STEPS):
        idx = r3.integers(0, len(imgs), BATCH)
        Xa = augment(imgs[idx], r3)
        Xb = augment(imgs[idx], r3)
        Za, ca_ = e3.embed(Xa, cache=True)
        Zb, cb_ = e3.embed(Xb, cache=True)
        d = Za - Zb
        ga = e3.backward_emb(ca_, 2 * d / len(Xa))
        gb = e3.backward_emb(cb_, -2 * d / len(Xa))
        e3.net.opt.step([a + b for a, b in zip(ga, gb)])
    s_pull = e3.embed(Xte).std(0).mean()
    s_full = enc_full.embed(Xte).std(0).mean()
    print(f"    embedding spread, pull-only  : {s_pull:.4f}")
    print(f"    embedding spread, contrastive: {s_full:.4f}")
    print()
    print(f"  A {s_full / s_pull:.0f}x collapse: with nothing to push")
    print("  against, the perfect solution is to map EVERY image to the")
    print("  same point -- loss zero, representation worthless. This is")
    print("  the failure the whole negative-free literature (BYOL,")
    print("  SimSiam, DINO -- Part 3) is engineered to dodge without")
    print("  paying for the crowd.")


def demo4(imgs, y, factors, te_imgs, te_y, te_f):
    banner("DEMO 4 --- Mini-CLIP: two towers and a zero shot")
    print("  Give every image a 7-number 'caption' (class one-hot +")
    print("  position + size + brightness) and train a SECOND encoder")
    print("  on captions, aligning the two towers with CLIP's symmetric")
    print("  InfoNCE: images classify captions, captions classify")
    print("  images. 6,000 steps, then evaluate things the model was")
    print("  never trained to do:")
    print()

    def captions(yv, f, rng):
        cap = np.zeros((len(yv), 7))
        cap[np.arange(len(yv)), yv] = 1.0
        cap[:, 3] = f[:, 0] / SIDE
        cap[:, 4] = f[:, 1] / SIDE
        cap[:, 5] = f[:, 2] / 6.0
        cap[:, 6] = f[:, 3] / 1.4
        return cap + 0.05 * rng.standard_normal(cap.shape)

    img_enc = Encoder(seed=0)
    txt_enc = Encoder(d_in=7, hidden=48, seed=200)
    r = np.random.default_rng(1)
    for _ in range(6000):
        idx = r.integers(0, len(imgs), BATCH)
        Xi = augment(imgs[idx], r)
        Ct = captions(y[idx], factors[idx], r)
        _, gi, gt = clip_grads(img_enc, txt_enc, Xi, Ct)
        img_enc.net.opt.step(gi)
        txt_enc.net.opt.step(gt)

    Xte = augment(te_imgs, np.random.default_rng(55))
    Zte_i = img_enc.embed(Xte)
    proto = np.zeros((3, 7))
    proto[np.arange(3), np.arange(3)] = 1.0
    proto[:, 3:5] = 0.5
    proto[:, 5] = 4.0 / 6.0
    proto[:, 6] = 1.0 / 1.4
    zs = float(((Zte_i @ txt_enc.embed(proto).T).argmax(1)
                == te_y).mean())
    pb = proto.copy()
    pb[:, 3:5] = 0.1
    zsb = float(((Zte_i @ txt_enc.embed(pb).T).argmax(1)
                 == te_y).mean())
    Cte = captions(te_y, te_f, np.random.default_rng(60))
    Sim = txt_enc.embed(Cte) @ Zte_i.T
    order = (-Sim).argsort(1)
    rank = (order == np.arange(len(te_y))[:, None]).argmax(1)
    print(f"    zero-shot classification, neutral prompts : {zs:6.1%}")
    print(f"    zero-shot classification, corner prompts  : {zsb:6.1%}")
    print(f"    caption->image retrieval among 1,000      : top-1 "
          f"{(rank == 0).mean():.1%},")
    print(f"                       top-10 {(rank < 10).mean():.1%}, "
          f"median rank {int(np.median(rank)) + 1}")
    print()
    print("  No classifier was ever trained -- the 'classes' are three")
    print("  sentences shown to the caption tower at test time, and the")
    print("  image tower agrees with the right one 95% of the time.")
    print("  Write the sentences badly (claim the shape sits in a")
    print("  corner) and 27 points vanish: prompt engineering is not a")
    print("  folk ritual, it is measurable geometry in the shared")
    print("  embedding space. And the same space does retrieval for")
    print("  free: the right image out of a thousand lands at median")
    print("  rank 15. One loss, three products -- the CLIP economy.")


def main() -> None:
    wa = gradcheck()
    wb = gradcheck_clip()
    print(f"  [audit] NT-Xent backward (through L2-norm) vs central")
    print(f"          differences: {wa:.2e}")
    print(f"  [audit] CLIP two-tower backward vs central differences: "
          f"{wb:.2e}")
    rng = np.random.default_rng(0)
    imgs, y, factors = make_instances(N_UNLAB, rng)
    te_imgs, te_y, te_f = make_instances(N_TEST,
                                         np.random.default_rng(50))
    enc, Xte, _ = demo1(imgs, y, te_imgs, te_y)
    demo2(imgs, y, factors, te_imgs, te_y, te_f)
    demo3(imgs, enc, Xte)
    demo4(imgs, y, factors, te_imgs, te_y, te_f)


if __name__ == "__main__":
    main()
