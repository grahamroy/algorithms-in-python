"""
mae.py --- companion code for "Masked Autoencoding"
(Self-Supervised Learning, Part 2).

Part 1 manufactured supervision by IGNORING: declare which
differences do not matter, and demand agreement across them. Masked
autoencoding (BERT: Devlin et al., 2019, for text; MAE: He et al.,
2022, for images) manufactures it the opposite way -- by
RECONSTRUCTING: hide most of the input and demand the model predict
what is missing.

    input:  image with 12 of 16 patches blanked (75%)
    task :  regress the hidden pixels;  loss only on what was hidden

No labels, no augmentation menu, no negatives -- the supervision is
the data's own redundancy. An encoder squeezes the visible quarter
into a 32-number code; a decoder must rebuild the hidden three
quarters from it. Whatever survives that bottleneck is the
representation.

The stage is Part 1's shape world, on purpose, and the script
retrains Part 1's contrastive encoder live, so the section's two
philosophies -- learn by ignoring vs learn by reconstructing --
meet on one measured table.

Demonstrates:
  1. The probes say failure. The masked autoencoder reconstructs
     well, but its LINEAR probe scores 49.2% on 15 labels and
     67.6% on all 4,000 -- against the contrastive encoder's
     95.1% and 99.7% on the identical stage. Reconstruction
     features are not linearly organised by class.
  2. The fine-tune flip. Unfreeze and fine-tune both encoders on
     the same 15 labels: MAE leaps 49.2% -> 87.5% -- past the
     84.9% supervised-from-scratch baseline -- while contrastive
     DROPS 95.1% -> 91.1% (its frozen features were already
     better than what 15 labels can teach). The 2x2 is the
     section's chiasmus: agreement features are ready to wear,
     reconstruction features are raw material.
  3. Why: the reconstructor keeps everything. Shift an image and
     the contrastive embedding moves 0.35x a typical inter-image
     distance; the MAE embedding moves 0.83x -- almost as far as
     jumping to a different image -- and a probe reads position
     from it at 94.2%. Reconstruction needs position, so position
     is kept; invariance was never asked for, so it never
     appears. Equivariant vs invariant, measured.
     And the honest footnote: on this stage the famous 75% mask
     ratio barely moves the probes (25% -> 90% all within a few
     points) -- shape world's redundancy is too regular for the
     ratio law to bite; it bought difficulty (recon error 3x),
     not better features. Folklore does not always transfer.

Everything is plain NumPy, all gradients hand-derived and audited.
Dependencies: numpy. Runs in about five minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import copy
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
# The stage: shape world, identical to Part 1.
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
# Plumbing, identical to Part 1.
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
    """Part 1's contrastive encoder: Net + L2 normalisation."""

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


def pretrain_contrastive(enc, imgs, rng, steps=STEPS, batch=BATCH):
    for _ in range(steps):
        idx = rng.integers(0, len(imgs), batch)
        Xa = augment(imgs[idx], rng)
        Xb = augment(imgs[idx], rng)
        _, grads = ntxent_grads(enc, Xa, Xb)
        enc.net.opt.step(grads)
    return enc


# ---------------------------------------------------------------------------
# The masked autoencoder.
# ---------------------------------------------------------------------------

def mask_batch(X, rng, ratio=MASK_RATIO):
    n = len(X)
    n_hide = int(round(N_PATCH * ratio))
    hide = np.zeros((n, N_PATCH), dtype=bool)
    for i in range(n):
        hide[i, rng.choice(N_PATCH, n_hide, replace=False)] = True
    pixhide = hide[:, PIX_PATCH_FLAT]
    Xm = np.where(pixhide, 0.0, X)
    inp = np.concatenate([Xm, hide.astype(float)], 1)
    return inp, pixhide


class MAE:
    def __init__(self, hidden=HID, d_emb=D_EMB, seed=0, lr=1e-3):
        self.enc = Net(D_IN + N_PATCH, hidden, d_emb, seed=seed, lr=lr)
        self.dec = Net(d_emb, hidden, D_IN, seed=seed + 100, lr=lr)

    def embed(self, X):
        """Representation for probing: the full image, empty mask."""
        inp = np.concatenate([X, np.zeros((len(X), N_PATCH))], 1)
        return self.enc.forward(inp)[0]

    def step(self, X, rng, ratio=MASK_RATIO):
        inp, pixhide = mask_batch(X, rng, ratio)
        H, ce = self.enc.forward(inp)
        R, cd = self.dec.forward(H)
        diff = (R - X) * pixhide
        n_mask = pixhide.sum()
        loss = float((diff ** 2).sum() / n_mask)
        gdec, dH = self.dec.backward(cd, 2 * diff / n_mask)
        genc, _ = self.enc.backward(ce, dH)
        self.enc.opt.step(genc)
        self.dec.opt.step(gdec)
        return loss


# ---------------------------------------------------------------------------
# Probes.
# ---------------------------------------------------------------------------

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
# Audit.
# ---------------------------------------------------------------------------

def gradcheck(seed=3):
    rng = np.random.default_rng(seed)
    m = MAE(hidden=8, d_emb=4, seed=seed)
    X = rng.standard_normal((5, D_IN))
    inp, pixhide = mask_batch(X, np.random.default_rng(seed + 5))

    def loss_only():
        H, _ = m.enc.forward(inp)
        R, _ = m.dec.forward(H)
        return float((((R - X) * pixhide) ** 2).sum() / pixhide.sum())

    H, ce = m.enc.forward(inp)
    R, cd = m.dec.forward(H)
    diff = (R - X) * pixhide
    gdec, dH = m.dec.backward(cd, 2 * diff / pixhide.sum())
    genc, _ = m.enc.backward(ce, dH)
    worst = 0.0
    r = np.random.default_rng(seed + 1)
    for params, grads in ((m.enc.params, genc), (m.dec.params, gdec)):
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
# Fine-tuning helpers.
# ---------------------------------------------------------------------------

def finetune_mae(m, imgs, y, few, Xte, te_y):
    netc = copy.deepcopy(m.enc)
    W = np.random.default_rng(3).standard_normal((D_EMB, 3)) * 0.01
    b = np.zeros(3)
    opt = Adam([W, b], 1e-3)
    r3 = np.random.default_rng(4)
    for _ in range(1000):
        bidx = few[r3.integers(0, len(few), 64)]
        Xb = augment(imgs[bidx], r3)
        inp = np.concatenate([Xb, np.zeros((len(Xb), N_PATCH))], 1)
        Z, cach = netc.forward(inp)
        L = Z @ W + b
        L -= L.max(1, keepdims=True)
        Pm = np.exp(L)
        Pm /= Pm.sum(1, keepdims=True)
        G = Pm.copy()
        G[np.arange(len(bidx)), y[bidx]] -= 1
        G /= len(bidx)
        gnet, _ = netc.backward(cach, G @ W.T)
        netc.opt.step(gnet)
        opt.step([Z.T @ G, G.sum(0)])
    inp_te = np.concatenate([Xte, np.zeros((len(Xte), N_PATCH))], 1)
    Zf = netc.forward(inp_te)[0]
    return float(((Zf @ W + b).argmax(1) == te_y).mean())


def finetune_contrastive(enc, imgs, y, few, Xte, te_y):
    netc = copy.deepcopy(enc.net)
    W = np.random.default_rng(3).standard_normal((D_EMB, 3)) * 0.01
    b = np.zeros(3)
    opt = Adam([W, b], 1e-3)
    r3 = np.random.default_rng(4)
    for _ in range(1000):
        bidx = few[r3.integers(0, len(few), 64)]
        Xb = augment(imgs[bidx], r3)
        H, cach = netc.forward(Xb)
        nrm = np.sqrt((H ** 2).sum(1, keepdims=True)) + 1e-12
        Z = H / nrm
        L = Z @ W + b
        L -= L.max(1, keepdims=True)
        Pm = np.exp(L)
        Pm /= Pm.sum(1, keepdims=True)
        G = Pm.copy()
        G[np.arange(len(bidx)), y[bidx]] -= 1
        G /= len(bidx)
        dZ = G @ W.T
        dH = (dZ - Z * (dZ * Z).sum(1, keepdims=True)) / nrm
        gnet, _ = netc.backward(cach, dH)
        netc.opt.step(gnet)
        opt.step([Z.T @ G, G.sum(0)])
    H, _ = netc.forward(Xte)
    Z = H / (np.sqrt((H ** 2).sum(1, keepdims=True)) + 1e-12)
    return float(((Z @ W + b).argmax(1) == te_y).mean())


def supervised_scratch(imgs, y, few, Xte, te_y):
    sup = Encoder(seed=0)
    W = np.random.default_rng(3).standard_normal((D_EMB, 3)) * 0.01
    b = np.zeros(3)
    opt = Adam([W, b], 1e-3)
    r2 = np.random.default_rng(4)
    for _ in range(STEPS):
        bidx = few[r2.integers(0, len(few), min(BATCH, len(few)))]
        Xb_ = augment(imgs[bidx], r2)
        Z, cach = sup.embed(Xb_, cache=True)
        L = Z @ W + b
        L -= L.max(1, keepdims=True)
        Pm = np.exp(L)
        Pm /= Pm.sum(1, keepdims=True)
        G = Pm.copy()
        G[np.arange(len(bidx)), y[bidx]] -= 1
        G /= len(bidx)
        genc = sup.backward_emb(cach, G @ W.T)
        sup.net.opt.step(genc)
        opt.step([Z.T @ G, G.sum(0)])
    Zs = sup.embed(Xte)
    return float(((Zs @ W + b).argmax(1) == te_y).mean())


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo1(imgs, y, te_imgs, te_y, Xtr, Xte, few):
    banner("DEMO 1 --- Hide three quarters, and what the probes say")
    print("  Blank 12 of 16 patches, regress the hidden pixels, score")
    print("  only what was hidden. 3,000 steps on 4,000 unlabelled")
    print("  images -- then the same linear probes as Part 1, with")
    print("  Part 1's contrastive encoder retrained live beside it:")
    print()
    Xtr_flat = imgs.reshape(len(imgs), -1)
    m = MAE(seed=0)
    r1 = np.random.default_rng(1)
    losses = []
    for _ in range(3000):
        idx = r1.integers(0, len(imgs), BATCH)
        losses.append(m.step(Xtr_flat[idx], r1))
    enc = Encoder(seed=0)
    pretrain_contrastive(enc, imgs, np.random.default_rng(1))
    print(f"    masked-pixel reconstruction mse: "
          f"{np.mean(losses[-100:]):.4f}  (pixel noise^2 = 0.0100)")
    print()
    print("                       probe, 15 labels   probe, all 4,000")
    probes = {}
    for name, Ztr, Zte in (
            ("MAE        ", m.embed(Xtr), m.embed(Xte)),
            ("contrastive", enc.embed(Xtr), enc.embed(Xte))):
        a15 = probe_acc(Ztr[few], y[few], Zte, te_y)
        aall = probe_acc(Ztr, y, Zte, te_y)
        probes[name.strip()] = a15
        print(f"    {name}         {a15:6.1%}            {aall:6.1%}")
    print()
    print(f"  The reconstruction works -- the model fills hidden")
    print(f"  strokes it has never seen -- and the probes call the")
    print(f"  representation a failure anyway:")
    print(f"  {probes['MAE']:.1%} against {probes['contrastive']:.1%} on"
          f" fifteen labels. Reconstruction")
    print("  features are not linearly organised by class. Hold the")
    print("  verdict until DEMO 2.")
    return m, enc, probes


def demo2(m, enc, imgs, y, te_y, Xte, few, probes):
    banner("DEMO 2 --- The fine-tune flip")
    print("  Unfreeze everything and fine-tune both encoders on the")
    print("  SAME fifteen labels (1,000 small steps), plus Part 1's")
    print("  supervised-from-scratch baseline:")
    print()
    ft_mae = finetune_mae(m, imgs, y, few, Xte, te_y)
    ft_con = finetune_contrastive(enc, imgs, y, few, Xte, te_y)
    sup = supervised_scratch(imgs, y, few, Xte, te_y)
    print("                       frozen probe   fine-tuned")
    print(f"    MAE                  {probes['MAE']:6.1%}        "
          f"{ft_mae:6.1%}")
    print(f"    contrastive          {probes['contrastive']:6.1%}"
          f"        {ft_con:6.1%}")
    print(f"    from scratch            --         {sup:6.1%}")
    print()
    print(f"  The chiasmus, measured. Fine-tuning transforms the masked")
    print(f"  autoencoder -- {probes['MAE']:.1%} to {ft_mae:.1%}, past "
          f"the from-scratch")
    print("  baseline: the class information was IN there, just not")
    print("  linearly arranged, and fifteen labels sufficed to unlock")
    print(f"  it. Fine-tuning HURTS the contrastive encoder -- "
          f"{probes['contrastive']:.1%} to")
    print(f"  {ft_con:.1%}: its frozen features already beat anything "
          f"fifteen")
    print("  labels can teach, and gradient steps on so few examples")
    print("  only overfit. Agreement features are ready to wear;")
    print("  reconstruction features are raw material. The same split")
    print("  holds at scale: MAE backbones want fine-tuning, CLIP-style")
    print("  towers get frozen and probed.")


def demo3(m, enc, imgs, y, factors, te_imgs, te_y, te_f, Xtr, Xte, few):
    banner("DEMO 3 --- The reconstructor keeps everything")
    print("  Why are the probes so different? Shift each test image and")
    print("  measure how far its embedding moves, as a fraction of the")
    print("  typical distance BETWEEN two different images:")
    print()
    Xc = augment(te_imgs, np.random.default_rng(53), shift=False)
    Xs = augment(te_imgs.copy(), np.random.default_rng(53), shift=True)
    for name, embed in (("contrastive", enc.embed), ("MAE        ",
                                                     m.embed)):
        Zc, Zs = embed(Xc), embed(Xs)
        d_pair = float(np.sqrt(((Zc - Zs) ** 2).sum(1)).mean())
        d_all = float(np.sqrt(((Zc[:200, None, :]
                                - Zc[None, :200, :]) ** 2)
                              .sum(-1)).mean())
        print(f"    {name}: shift moves the embedding "
              f"{d_pair / d_all:.2f}x a typical distance")
    xq_tr = (factors[:, 0] > SIDE / 2).astype(int)
    xq_te = (te_f[:, 0] > SIDE / 2).astype(int)
    Xtr_pos = augment(imgs, np.random.default_rng(52), shift=False)
    pa = probe_acc(m.embed(Xtr_pos), xq_tr, m.embed(Xc), xq_te, 2)
    print()
    print("  The contrastive embedding barely notices the shift; the")
    print("  MAE embedding moves almost as far as jumping to a")
    print("  DIFFERENT image. Reconstruction needs position -- so")
    print("  position is kept (a linear probe reads the shape's half")
    print(f"  of the canvas from the MAE embedding at {pa:.1%}).")
    print("  Invariant vs equivariant: Part 1 chose what to forget;")
    print("  the reconstructor is not allowed to forget anything.")
    print()
    print("  And the honest footnote -- the famous 75% mask ratio, on")
    print("  this stage:")
    print()
    Xtr_flat = imgs.reshape(len(imgs), -1)
    print("    mask ratio   recon-mse   probe15   probe-all")
    for ratio in (0.25, 0.5, 0.75, 0.9):
        m2 = MAE(seed=0)
        r2 = np.random.default_rng(1)
        losses = []
        for _ in range(3000):
            idx = r2.integers(0, len(imgs), BATCH)
            losses.append(m2.step(Xtr_flat[idx], r2, ratio=ratio))
        Zt, Ze = m2.embed(Xtr), m2.embed(Xte)
        a15 = probe_acc(Zt[few], y[few], Ze, te_y)
        aall = probe_acc(Zt, y, Ze, te_y)
        print(f"       {ratio:.0%}       {np.mean(losses[-100:]):.4f}"
              f"     {a15:6.1%}    {aall:6.1%}")
    print()
    print("  Reconstruction gets 3x harder from 25% to 90%; the probes")
    print("  barely move. Shape world is so redundant that even a")
    print("  quarter of the pixels gives the game away -- MAE's 75%")
    print("  law lives in the statistics of natural images, and it")
    print("  does not automatically ride along to other data. Measure")
    print("  before you inherit a hyperparameter.")


def main() -> None:
    print(f"  [audit] masked-reconstruction backward vs central")
    print(f"          differences: {gradcheck():.2e}")
    rng = np.random.default_rng(0)
    imgs, y, factors = make_instances(N_UNLAB, rng)
    te_imgs, te_y, te_f = make_instances(N_TEST,
                                         np.random.default_rng(50))
    Xtr = augment(imgs, np.random.default_rng(54))
    Xte = augment(te_imgs, np.random.default_rng(55))
    r = np.random.default_rng(2)
    few = np.concatenate([r.choice(np.where(y == c)[0], 5, False)
                          for c in range(3)])
    m, enc, probes = demo1(imgs, y, te_imgs, te_y, Xtr, Xte, few)
    demo2(m, enc, imgs, y, te_y, Xte, few, probes)
    demo3(m, enc, imgs, y, factors, te_imgs, te_y, te_f, Xtr, Xte, few)


if __name__ == "__main__":
    main()
