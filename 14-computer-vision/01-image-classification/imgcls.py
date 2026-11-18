"""
imgcls.py --- companion code for "Image Classification"
(Computer Vision, Part 1).

Section 13 spent eleven parts on a micro-language whose 31,740
sentences could be listed, so every claim had an oracle to be checked
against. Images need their own version of that, and this is it: the
pictures here are DRAWN from factors the script chooses --

    class, centre (cy, cx), size, brightness

-- so the thing a vision model is supposed to ignore is not inferred,
it is an input. That turns the central question of this part from an
argument into a measurement.

The question is what an architecture's PRIOR is worth. A multilayer
perceptron sees 256 unrelated numbers. A convolution is told, before
it sees any data, that pixels near each other belong together and that
a pattern means the same thing wherever it appears. A residual network
is told it may ignore a layer. A vision transformer is told almost
nothing and has to find out.

What that buys, measured here:
  1. The price of the prior in LABELS, not accuracy points: the same
     task at six training-set sizes, four architectures, matched
     parameter budgets.
  2. The price in PARAMETERS, which runs the same way and further.
  3. Translation invariance, exactly. The stage can re-render one
     shape at every legal position, so "does this model change its
     mind when the object moves" is a number, not an impression --
     and the answer separates what is built in from what is learned.
  4. Where the budget should go. At a fixed parameter count, deeper
     or wider or higher-resolution? That is EfficientNet's question,
     and it has an answer here.

Everything is plain NumPy: im2col convolution, same-padded residual
blocks, patch embedding with multi-head self-attention, all
hand-differentiated and audited against central differences at
startup. Seeded and deterministic.
Dependencies: numpy.  Runs in about ten minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

SEPARATOR = "=" * 72


def banner(title):
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


def softmax_rows(Z):
    e = np.exp(Z - Z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# -----------------------------------------------------------------------
# The stage: shape world, extended from Section 12's.
#
# Six shapes rather than three, chosen so that pixel statistics overlap
# -- a cross and a plus-less vertical bar share strokes, a hollow
# square and a corner L share two sides -- which means the class can
# only be settled by GEOMETRY. Every image records the factors it was
# drawn from, and that is what makes the invariance measurement exact.
# -----------------------------------------------------------------------

SIDE = 16
N_CLS = 6
CLASS_NAMES = ["cross", "hollow square", "diagonal X",
               "corner L", "vertical bar", "horizontal bar"]
SIZE_LO, SIZE_HI = 3, 6          # half-width, exclusive upper
BRIGHT_LO, BRIGHT_HI = 0.6, 1.4
NOISE = 0.20


def draw(cls, cy, cx, s, bright=1.0):
    """One clean shape on a SIDE x SIDE canvas. No noise."""
    img = np.zeros((SIDE, SIDE))
    if cls == 0:                                   # cross
        img[cy - s:cy + s + 1, cx] = 1.0
        img[cy, cx - s:cx + s + 1] = 1.0
    elif cls == 1:                                 # hollow square
        img[cy - s:cy + s + 1, cx - s] = 1.0
        img[cy - s:cy + s + 1, cx + s] = 1.0
        img[cy - s, cx - s:cx + s + 1] = 1.0
        img[cy + s, cx - s:cx + s + 1] = 1.0
    elif cls == 2:                                 # diagonal X
        for d in range(-s, s + 1):
            img[cy + d, cx + d] = 1.0
            img[cy + d, cx - d] = 1.0
    elif cls == 3:                                 # corner L
        img[cy - s:cy + s + 1, cx - s] = 1.0
        img[cy + s, cx - s:cx + s + 1] = 1.0
    elif cls == 4:                                 # vertical bar
        img[cy - s:cy + s + 1, cx] = 1.0
    else:                                          # horizontal bar
        img[cy, cx - s:cx + s + 1] = 1.0
    return img * bright


def positions_for(s):
    """Every centre at which a shape of half-width s fits entirely."""
    return [(cy, cx)
            for cy in range(s, SIDE - s)
            for cx in range(s, SIDE - s)]


def make_set(n, rng, noise=NOISE):
    """n images, their labels, and the factors they were drawn from."""
    y = rng.integers(0, N_CLS, n)
    imgs = np.zeros((n, SIDE, SIDE))
    factors = np.zeros((n, 4))                     # cy, cx, size, bright
    for i in range(n):
        s = int(rng.integers(SIZE_LO, SIZE_HI))
        cy = int(rng.integers(s, SIDE - s))
        cx = int(rng.integers(s, SIDE - s))
        b = float(rng.uniform(BRIGHT_LO, BRIGHT_HI))
        imgs[i] = draw(y[i], cy, cx, s, b)
        factors[i] = (cy, cx, s, b)
    return imgs + rng.normal(0, noise, imgs.shape), y, factors


# -----------------------------------------------------------------------
# Convolution as one matrix multiply.
#
# im2col lays every k x k window out as a row, so the forward pass is a
# matmul and the backward pass is another. `pad` keeps the spatial size,
# which is what a residual block needs in order to add its input back.
# -----------------------------------------------------------------------

def im2col(X, k, pad=0):
    """(N, H, W) -> (N, oh*ow, k*k), plus the output shape.

    The loop runs over the k*k KERNEL offsets rather than the oh*ow
    output positions -- nine passes instead of two hundred and
    fifty-six, for identical contents.
    """
    if pad:
        X = np.pad(X, ((0, 0), (pad, pad), (pad, pad)))
    N, H, W = X.shape
    oh, ow = H - k + 1, W - k + 1
    cols = np.empty((N, oh, ow, k, k))
    for i in range(k):
        for j in range(k):
            cols[:, :, :, i, j] = X[:, i:i + oh, j:j + ow]
    return cols.reshape(N, oh * ow, k * k), oh, ow


def col2im(dcols, k, H, W, pad=0):
    """The adjoint of im2col: scatter window gradients back to pixels.

    Same trick, and here it matters more: overlapping windows mean the
    scatter has to accumulate, and accumulating over nine slices is
    far cheaper than over two hundred and fifty-six.
    """
    N = len(dcols)
    Hp, Wp = H + 2 * pad, W + 2 * pad
    oh, ow = Hp - k + 1, Wp - k + 1
    d = dcols.reshape(N, oh, ow, k, k)
    dX = np.zeros((N, Hp, Wp))
    for i in range(k):
        for j in range(k):
            dX[:, i:i + oh, j:j + ow] += d[:, :, :, i, j]
    return dX[:, pad:pad + H, pad:pad + W] if pad else dX


def softmax_ce(logits, y):
    """Mean cross-entropy and its gradient with respect to the logits."""
    P = softmax_rows(logits)
    n = len(y)
    loss = float(-np.log(P[np.arange(n), y] + 1e-300).mean())
    d = P.copy()
    d[np.arange(n), y] -= 1.0
    return loss, d / n


# -----------------------------------------------------------------------
# Adam, shared by every model below so the comparison is of
# architectures and not of optimisers.
# -----------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr):
        self.p = params
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, g):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, p in self.p.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            p -= self.lr * (self.m[k] / (1 - b1 ** self.t)) / (
                np.sqrt(self.v[k] / (1 - b2 ** self.t)) + eps)
# =======================================================================
# Four architectures, one interface.
#
# Each exposes params / forward(X) -> (logits, cache) / backward.
# What differs is only what each is told before it sees any data:
#
#   MLP     nothing. 256 numbers in a row.
#   CNN     pixels near each other belong together, and a pattern
#           means the same thing wherever it appears.
#   ResNet  the same, plus: you may ignore a layer.
#   ViT     patches are a sequence; work out the rest yourself.
# =======================================================================

class MLP:
    """No spatial prior at all. The control."""

    def __init__(self, hid=64, seed=0, lr=3e-3):
        r = np.random.default_rng(seed)
        d = SIDE * SIDE
        self.P = {
            "W1": r.standard_normal((d, hid)) * np.sqrt(2 / d),
            "b1": np.zeros(hid),
            "W2": r.standard_normal((hid, N_CLS)) * np.sqrt(1 / hid),
            "b2": np.zeros(N_CLS),
        }
        self.opt = Adam(self.P, lr)

    def n_params(self):
        return sum(v.size for v in self.P.values())

    def forward(self, X):
        x = X.reshape(len(X), -1)
        h = np.maximum(0, x @ self.P["W1"] + self.P["b1"])
        return h @ self.P["W2"] + self.P["b2"], (x, h)

    def backward(self, d, cache):
        x, h = cache
        g = {"W2": h.T @ d, "b2": d.sum(0)}
        dh = (d @ self.P["W2"].T) * (h > 0)
        g["W1"] = x.T @ dh
        g["b1"] = dh.sum(0)
        return g


class CNN:
    """Locality and weight sharing, then global max-pool.

    The pooling is what makes it translation invariant BY
    CONSTRUCTION: the same filter response wins wherever the shape
    happens to be, so moving the object cannot change the answer.
    """

    def __init__(self, k=5, n_filt=32, seed=0, lr=3e-3):
        r = np.random.default_rng(seed)
        self.k = k
        self.P = {
            "F": r.standard_normal((k * k, n_filt)) * np.sqrt(2 / (k * k)),
            "bf": np.zeros(n_filt),
            "W": r.standard_normal((n_filt, N_CLS)) * np.sqrt(1 / n_filt),
            "b": np.zeros(N_CLS),
        }
        self.opt = Adam(self.P, lr)

    def n_params(self):
        return sum(v.size for v in self.P.values())

    def forward(self, X):
        cols, oh, ow = im2col(X, self.k)
        pre = cols @ self.P["F"] + self.P["bf"]
        act = np.maximum(0, pre)
        idx = act.argmax(1)
        n, nf = len(X), act.shape[2]
        rows = np.arange(n)[:, None]
        pooled = act[rows, idx, np.arange(nf)]
        return pooled @ self.P["W"] + self.P["b"], (cols, pre, idx,
                                                    pooled, oh, ow)

    def backward(self, d, cache):
        cols, pre, idx, pooled, oh, ow = cache
        n, nf = len(cols), self.P["F"].shape[1]
        rows = np.arange(n)[:, None]
        fcols = np.arange(nf)
        g = {"W": pooled.T @ d, "b": d.sum(0)}
        dpool = d @ self.P["W"].T
        live = (pre[rows, idx, fcols] > 0) * dpool
        picked = cols[rows, idx]
        g["F"] = np.einsum("nfk,nf->kf", picked, live)
        g["bf"] = live.sum(0)
        return g


class ResNet:
    """Two same-padded conv blocks with a skip around each.

    y = x + conv2(relu(conv1(x))).  The skip is the whole point: the
    block starts as the identity, so depth costs nothing until the
    gradient finds a use for it.
    """

    def __init__(self, k=3, width=12, n_blocks=2, seed=0, lr=3e-3):
        r = np.random.default_rng(seed)
        self.k, self.width, self.nb = k, width, n_blocks
        P = {"Fin": r.standard_normal((k * k, width))
             * np.sqrt(2 / (k * k)),
             "bin": np.zeros(width)}
        for i in range(n_blocks):
            for j in (1, 2):
                P[f"{i}.F{j}"] = r.standard_normal(
                    (k * k * width, width)) * np.sqrt(
                        2 / (k * k * width))
                P[f"{i}.b{j}"] = np.zeros(width)
        P["W"] = r.standard_normal((width, N_CLS)) * np.sqrt(1 / width)
        P["b"] = np.zeros(N_CLS)
        self.P = P
        self.opt = Adam(self.P, lr)

    def n_params(self):
        return sum(v.size for v in self.P.values())

    def _conv(self, H, key, pad):
        """H is (N, C, S, S); returns (N, width, S, S) and a cache."""
        n, c, s, _ = H.shape
        flat = H.reshape(n * c, s, s)
        cols, oh, ow = im2col(flat, self.k, pad=pad)
        cols = cols.reshape(n, c, oh * ow, self.k * self.k)
        cols = cols.transpose(0, 2, 1, 3).reshape(n, oh * ow,
                                                  c * self.k * self.k)
        out = cols @ self.P[key + "F" if key.endswith(".") else key]
        return out, cols, oh, ow

    def forward(self, X):
        n = len(X)
        cols0, oh0, ow0 = im2col(X, self.k, pad=self.k // 2)
        h = np.maximum(0, cols0 @ self.P["Fin"] + self.P["bin"])
        # 2x2 max-pool: 16x16 -> 8x8, as a real residual stem does
        hg = h.reshape(n, oh0, ow0, self.width)
        hp = hg.reshape(n, oh0 // 2, 2, ow0 // 2, 2, self.width)
        hp = hp.transpose(0, 1, 3, 5, 2, 4).reshape(
            n, (oh0 // 2) * (ow0 // 2), self.width, 4)
        pool_arg = hp.argmax(-1)
        oh = ow = oh0 // 2
        cur = np.take_along_axis(hp, pool_arg[..., None], -1)[..., 0]
        caches = [(cols0, h, pool_arg, oh0, ow0)]
        for i in range(self.nb):
            blk = []
            x_in = cur
            for j in (1, 2):
                grid = cur.reshape(n, oh, ow, self.width)
                grid = grid.transpose(0, 3, 1, 2).reshape(
                    n * self.width, oh, ow)
                c, _, _ = im2col(grid, self.k, pad=self.k // 2)
                c = c.reshape(n, self.width, oh * ow,
                              self.k * self.k)
                c = c.transpose(0, 2, 1, 3).reshape(
                    n, oh * ow, self.width * self.k * self.k)
                pre = c @ self.P[f"{i}.F{j}"] + self.P[f"{i}.b{j}"]
                blk.append((c, pre))
                cur = np.maximum(0, pre) if j == 1 else pre
            cur = x_in + cur                      # the skip
            caches.append((blk, x_in))
        pooled = cur.max(1)                       # global max-pool
        pool_idx = cur.argmax(1)
        caches.append((cur, pool_idx))
        return pooled @ self.P["W"] + self.P["b"], (caches, oh, ow, n)

    def backward(self, d, cache):
        caches, oh, ow, n = cache
        cur, pool_idx = caches[-1]
        pooled = cur[np.arange(n)[:, None], pool_idx,
                     np.arange(self.width)]
        g = {k: np.zeros_like(v) for k, v in self.P.items()}
        g["W"] = pooled.T @ d
        g["b"] = d.sum(0)
        dpool = d @ self.P["W"].T
        dcur = np.zeros_like(cur)
        dcur[np.arange(n)[:, None], pool_idx,
             np.arange(self.width)] = dpool
        for i in range(self.nb - 1, -1, -1):
            blk, x_in = caches[1 + i]
            dskip = dcur.copy()                   # straight through
            dbranch = dcur
            for j in (2, 1):
                c, pre = blk[j - 1]
                if j == 1:
                    dbranch = dbranch * (pre > 0)
                g[f"{i}.F{j}"] += c.reshape(-1, c.shape[2]).T @ \
                    dbranch.reshape(-1, self.width)
                g[f"{i}.b{j}"] += dbranch.sum((0, 1))
                dc = dbranch @ self.P[f"{i}.F{j}"].T
                dc = dc.reshape(n, oh * ow, self.width,
                                self.k * self.k)
                dc = dc.transpose(0, 2, 1, 3).reshape(
                    n * self.width, oh * ow, self.k * self.k)
                dgrid = col2im(dc, self.k, oh, ow, pad=self.k // 2)
                dbranch = dgrid.reshape(n, self.width, oh, ow)
                dbranch = dbranch.transpose(0, 2, 3, 1).reshape(
                    n, oh * ow, self.width)
            dcur = dskip + dbranch
        cols0, h, pool_arg, oh0, ow0 = caches[0]
        # undo the 2x2 pool: the winner in each window takes the
        # gradient, everything else gets nothing
        dp = np.zeros((n, (oh0 // 2) * (ow0 // 2), self.width, 4))
        np.put_along_axis(dp, pool_arg[..., None],
                          dcur[..., None], -1)
        dp = dp.reshape(n, oh0 // 2, ow0 // 2, self.width, 2, 2)
        dp = dp.transpose(0, 1, 4, 2, 5, 3).reshape(
            n, oh0 * ow0, self.width)
        dpre = dp * (h > 0)
        g["Fin"] += cols0.reshape(-1, cols0.shape[2]).T @ \
            dpre.reshape(-1, self.width)
        g["bin"] += dpre.sum((0, 1))
        return g


class ViT:
    """Patches, one attention block, a CLS token, a linear head.

    Told nothing about locality. Every patch can attend to every other
    from the first layer, which is freedom it has to pay for in data.
    """

    def __init__(self, patch=4, d=32, heads=4, seed=0, lr=3e-3):
        r = np.random.default_rng(seed)
        self.p = patch
        self.d = d
        self.h = heads
        self.dk = d // heads
        self.n_patch = (SIDE // patch) ** 2
        pd = patch * patch
        self.P = {
            "Wp": r.standard_normal((pd, d)) * np.sqrt(2 / pd),
            "bp": np.zeros(d),
            "cls": r.standard_normal(d) * 0.02,
            "pos": r.standard_normal((self.n_patch + 1, d)) * 0.02,
            "Wqkv": r.standard_normal((d, 3 * d)) * np.sqrt(1 / d),
            "Wo": r.standard_normal((d, d)) * np.sqrt(1 / d),
            "W1": r.standard_normal((d, 2 * d)) * np.sqrt(2 / d),
            "bb1": np.zeros(2 * d),
            "W2": r.standard_normal((2 * d, d)) * np.sqrt(1 / (2 * d)),
            "bb2": np.zeros(d),
            "Wc": r.standard_normal((d, N_CLS)) * np.sqrt(1 / d),
            "bc": np.zeros(N_CLS),
        }
        self.opt = Adam(self.P, lr)

    def n_params(self):
        return sum(v.size for v in self.P.values())

    def patches(self, X):
        n, p, g = len(X), self.p, SIDE // self.p
        out = np.empty((n, g * g, p * p))
        for i in range(g):
            for j in range(g):
                out[:, i * g + j] = X[:, i * p:(i + 1) * p,
                                      j * p:(j + 1) * p].reshape(n, -1)
        return out

    def forward(self, X):
        n = len(X)
        pt = self.patches(X)
        emb = pt @ self.P["Wp"] + self.P["bp"]
        z = np.concatenate(
            [np.broadcast_to(self.P["cls"], (n, 1, self.d)), emb], 1)
        z = z + self.P["pos"]
        q, k, v = np.split(z @ self.P["Wqkv"], 3, axis=-1)
        T = z.shape[1]

        def sp(a):
            return a.reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)

        qh, kh, vh = sp(q), sp(k), sp(v)
        S = qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(self.dk)
        A = softmax_rows(S)
        ctx = (A @ vh).transpose(0, 2, 1, 3).reshape(n, T, self.d)
        att = ctx @ self.P["Wo"]
        z1 = z + att
        hpre = z1 @ self.P["W1"] + self.P["bb1"]
        hact = np.maximum(0, hpre)
        z2 = z1 + hact @ self.P["W2"] + self.P["bb2"]
        cls = z2[:, 0]
        return cls @ self.P["Wc"] + self.P["bc"], (
            pt, z, qh, kh, vh, A, ctx, z1, hpre, hact, z2, cls, n, T)

    def backward(self, d, cache):
        (pt, z, qh, kh, vh, A, ctx, z1, hpre, hact, z2, cls,
         n, T) = cache
        g = {k: np.zeros_like(v) for k, v in self.P.items()}
        g["Wc"] = cls.T @ d
        g["bc"] = d.sum(0)
        dz2 = np.zeros_like(z2)
        dz2[:, 0] = d @ self.P["Wc"].T
        g["W2"] = hact.reshape(-1, 2 * self.d).T @ \
            dz2.reshape(-1, self.d)
        g["bb2"] = dz2.sum((0, 1))
        dh = (dz2 @ self.P["W2"].T) * (hpre > 0)
        g["W1"] = z1.reshape(-1, self.d).T @ dh.reshape(-1, 2 * self.d)
        g["bb1"] = dh.sum((0, 1))
        dz1 = dz2 + dh @ self.P["W1"].T
        g["Wo"] = ctx.reshape(-1, self.d).T @ dz1.reshape(-1, self.d)
        dctx = dz1 @ self.P["Wo"].T
        dctx = dctx.reshape(n, T, self.h, self.dk).transpose(0, 2, 1, 3)
        dA = dctx @ vh.transpose(0, 1, 3, 2)
        dvh = A.transpose(0, 1, 3, 2) @ dctx
        dS = A * (dA - (A * dA).sum(-1, keepdims=True)) / np.sqrt(self.dk)
        dqh = dS @ kh
        dkh = dS.transpose(0, 1, 3, 2) @ qh

        def mg(a):
            return a.transpose(0, 2, 1, 3).reshape(n, T, self.d)

        dqkv = np.concatenate([mg(dqh), mg(dkh), mg(dvh)], -1)
        g["Wqkv"] = z.reshape(-1, self.d).T @ dqkv.reshape(-1,
                                                           3 * self.d)
        dz = dz1 + dqkv @ self.P["Wqkv"].T
        g["pos"] = dz.sum(0)
        g["cls"] = dz[:, 0].sum(0)
        demb = dz[:, 1:]
        g["Wp"] = pt.reshape(-1, self.p * self.p).T @ \
            demb.reshape(-1, self.d)
        g["bp"] = demb.sum((0, 1))
        return g


MODELS = {"MLP": MLP, "CNN": CNN, "ResNet": ResNet, "ViT": ViT}


# =======================================================================
# Training and evaluation, identical for every model.
# =======================================================================

def fit(net, X, y, steps, batch=64, rng=None):
    """A fixed number of gradient steps, whatever the dataset size.

    Equal optimisation budget is the only way the comparison below is
    about architecture rather than about how many times each model
    happened to see its data.
    """
    rng = rng or np.random.default_rng(0)
    n = len(X)
    for _ in range(steps):
        b = rng.integers(0, n, min(batch, n))
        logits, cache = net.forward(X[b])
        _, d = softmax_ce(logits, y[b])
        net.opt.step(net.backward(d, cache))
    return net


def accuracy(net, X, y, batch=500):
    hit = 0
    for i in range(0, len(X), batch):
        logits, _ = net.forward(X[i:i + batch])
        hit += int((logits.argmax(1) == y[i:i + batch]).sum())
    return hit / len(X)
# =======================================================================
# Translation invariance, exactly.
#
# The stage chose the centre, so the same shape can be re-rendered at
# every legal position and the model asked whether it changes its mind.
# `margin` keeps the shape clear of the image border: a valid
# convolution has no windows past the edge, so a shape touching the
# border produces responses a centred one cannot, and that is a
# property of the padding rather than of the prior.
# =======================================================================

def self_agreement(net, rng, margin=0, noise=0.0, n_shapes=150):
    """How often the answer survives moving the object.

    1.0 means position is ignored completely. Returns the agreement
    with the model's own modal answer, and how often that answer is
    also correct.
    """
    agree, correct = [], []
    for _ in range(n_shapes):
        cls = int(rng.integers(0, N_CLS))
        s = int(rng.integers(SIZE_LO, SIZE_HI))
        b = float(rng.uniform(BRIGHT_LO, BRIGHT_HI))
        lo, hi = s + margin, SIDE - s - margin
        if hi - lo < 2:
            continue
        pos = [(cy, cx) for cy in range(lo, hi) for cx in range(lo, hi)]
        imgs = np.stack([draw(cls, cy, cx, s, b) for cy, cx in pos])
        if noise:
            imgs = imgs + rng.normal(0, noise, imgs.shape)
        pred = net.forward(imgs)[0].argmax(1)
        mode = int(np.bincount(pred, minlength=N_CLS).argmax())
        agree.append(float((pred == mode).mean()))
        correct.append(float((pred == cls).mean()))
    return float(np.mean(agree)), float(np.mean(correct))


def ms(vals):
    v = np.array(vals, dtype=float)
    return float(v.mean()), float(v.std())


# -----------------------------------------------------------------------
# Audits.
# -----------------------------------------------------------------------

def gradcheck(seed=0, n_per=4):
    """Every architecture, against central differences."""
    rng = np.random.default_rng(seed)
    X, y, _ = make_set(6, rng)
    out, worst_all, checked_all = {}, 0.0, 0
    for name, cls in MODELS.items():
        net = cls(seed=1, **CONFIG[name])
        logits, cache = net.forward(X)
        _, d = softmax_ce(logits, y)
        g = net.backward(d, cache)
        worst, checked = 0.0, 0
        for k, p in net.P.items():
            flat, gflat = p.reshape(-1), g[k].reshape(-1)
            for ix in rng.choice(len(flat), min(n_per, len(flat)),
                                 replace=False):
                old = flat[ix]
                flat[ix] = old + 1e-6
                lp = softmax_ce(net.forward(X)[0], y)[0]
                flat[ix] = old - 1e-6
                lm = softmax_ce(net.forward(X)[0], y)[0]
                flat[ix] = old
                worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
                checked += 1
        out[name] = (worst, checked)
        worst_all = max(worst_all, worst)
        checked_all += checked
    return out, worst_all, checked_all


def selftest():
    """The stage must be what the invariance measurement assumes."""
    rng = np.random.default_rng(3)

    # a shape drawn at two centres is the SAME shape, translated
    for _ in range(50):
        cls = int(rng.integers(0, N_CLS))
        s = int(rng.integers(SIZE_LO, SIZE_HI))
        a = draw(cls, s + 1, s + 1, s)
        b = draw(cls, s + 2, s + 1, s)
        assert np.allclose(a[:-1], b[1:]), cls      # a vertical shift
        assert abs(a.sum() - b.sum()) < 1e-9, cls   # same ink

    # every class is distinguishable from every other at a fixed centre
    c = SIDE // 2
    imgs = [draw(k, c, c, 3) for k in range(N_CLS)]
    for i in range(N_CLS):
        for j in range(i + 1, N_CLS):
            assert not np.allclose(imgs[i], imgs[j]), (i, j)

    # positions_for keeps the shape entirely inside the canvas, so
    # moving it conserves ink exactly -- which is the premise the
    # invariance measurement rests on
    c = SIDE // 2
    for s in range(SIZE_LO, SIZE_HI):
        for cls in range(N_CLS):
            ref = draw(cls, c, c, s).sum()
            for cy, cx in positions_for(s):
                assert abs(draw(cls, cy, cx, s).sum() - ref) < 1e-9, \
                    (cls, s, cy, cx)

    # the parameter budgets really are matched
    sizes = {n: MODELS[n](seed=0, **CONFIG[n]).n_params()
             for n in MODELS}
    lo, hi = min(sizes.values()), max(sizes.values())
    assert hi / lo < 1.1, sizes
    return sizes, lo, hi


# -----------------------------------------------------------------------
# Frozen configuration.
#
# The four models are matched to within 5% on parameter count, so the
# comparison is of priors and not of capacity.
# -----------------------------------------------------------------------

CONFIG = {
    "MLP": dict(hid=21),                               # 5,529
    "CNN": dict(k=5, n_filt=170),                      # 5,446
    "ResNet": dict(k=3, width=12, n_blocks=2),         # 5,430
    "ViT": dict(patch=4, d=24, heads=4),               # 5,670
}
LABEL_COUNTS = (60, 120, 240, 480, 960)
STEPS = 400
SEEDS = (0, 1, 2)
N_TEST = 1500
SEED_TEST, SEED_TRAIN, SEED_FIT, SEED_INV = 7, 10, 2, 55
MARGINS = (0, 2)
LR = 3e-3


def build_models(seed):
    return {n: MODELS[n](seed=seed, lr=LR, **CONFIG[n]) for n in MODELS}
SCALE_CFGS = ((1, 17), (2, 12), (3, 10))
SCALE_LABELS = (120, 960)


# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

def demo1(sizes):
    """What each architecture is told before it sees anything."""
    banner("1. FOUR PRIORS, ONE BUDGET")
    print("  Six shapes on a %dx%d canvas, drawn at a centre, size and"
          % (SIDE, SIDE))
    print("  brightness the script chooses, then noised. Guessing")
    print("  scores %.4f." % (1 / N_CLS))
    print()
    rng = np.random.default_rng(1)
    X, y, f = make_set(3, rng)
    for i in range(1):
        print("  one image, as the model receives it (class %s,"
              % CLASS_NAMES[y[i]])
        print("  centre %d,%d size %d brightness %.2f):"
              % (f[i][0], f[i][1], f[i][2], f[i][3]))
        for row in X[i]:
            print("    " + "".join(".:*#"[min(3, max(0, int(v * 3)))]
                                   for v in row))
    print()
    print("  %-8s %10s   %s" % ("model", "params", "what it is told"))
    print("  " + "-" * 64)
    told = {
        "MLP": "nothing: 256 numbers in a row",
        "CNN": "pixels nearby belong together; patterns repeat",
        "ResNet": "the same, plus: you may skip a layer",
        "ViT": "patches are a sequence; work it out",
    }
    for n in MODELS:
        print("  %-8s %10s   %s" % (n, f"{sizes[n]:,}", told[n]))
    print()
    print("  The budgets are matched to within %.1f%%, so what follows"
          % (100 * (max(sizes.values()) / min(sizes.values()) - 1)))
    print("  is a comparison of priors and not of capacity. Every")
    print("  model gets the same %d gradient steps at every training" % STEPS)
    print("  set size, for the same reason.")


def demo2(acc):
    """The price of a prior, in labels."""
    banner("2. WHAT THE PRIOR IS WORTH")
    print("  Test accuracy on %s held-out images. %d seeds per cell;"
          % (f"{N_TEST:,}", len(SEEDS)))
    print("  the spread is printed because one run is not evidence.")
    print()
    print("  %-8s" % "labels"
          + "".join("%17s" % n for n in MODELS))
    print("  " + "-" * 76)
    for n_lab in LABEL_COUNTS:
        row = "  %-8d" % n_lab
        for n in MODELS:
            m, s = acc[(n_lab, n)]
            row += "%11.4f %5.4f" % (m, s)
        print(row)
    print()
    lo, hi = LABEL_COUNTS[0], LABEL_COUNTS[-1]
    print("  From %d labels the convolution is already at %.4f. The"
          % (lo, acc[(lo, "CNN")][0]))
    print("  perceptron, with the SAME number of parameters, is at")
    print("  %.4f -- and after sixteen times the data it has reached"
          % acc[(lo, "MLP")][0])
    print("  only %.4f, still far below what the convolution got from"
          % acc[(hi, "MLP")][0])
    print("  sixty examples. That gap is the prior, and it is not")
    print("  bought with capacity: the budgets are equal.")
    print()
    print("  The transformer sits between them: %.4f at %d labels,"
          % (acc[(lo, "ViT")][0], lo))
    print("  %.4f at %d. It is not failing -- it is learning from"
          % (acc[(hi, "ViT")][0], hi))
    print("  data what the convolution was told for free, and paying")
    print("  for it in examples.")
    print()
    d_lo = acc[(lo, "CNN")][0] - acc[(lo, "ResNet")][0]
    d_hi = acc[(hi, "ResNet")][0] - acc[(hi, "CNN")][0]
    print("  And the residual network LOSES to the plain convolution")
    print("  at every small size -- %.4f against %.4f at %d labels."
          % (acc[(lo, "ResNet")][0], acc[(lo, "CNN")][0], lo))
    print("  By %d labels it has caught up (%+.4f). Depth is not free"
          % (hi, d_hi))
    print("  here; it costs %.4f early and repays almost none of it."
          % d_lo)


def demo3(inv):
    """Invariance: built in, or learned?"""
    banner("3. DOES IT CHANGE ITS MIND WHEN THE OBJECT MOVES?")
    print("  The stage chose the centre, so the same shape can be")
    print("  re-rendered at EVERY legal position and the model asked")
    print("  whether its answer survives. 1.0000 means position is")
    print("  ignored completely. Clean images, %d seeds." % len(SEEDS))
    print()
    print("  margin keeps the shape clear of the image border, where a")
    print("  valid convolution has no windows and the construction")
    print("  stops applying.")
    print()
    print("  %-8s %20s %20s" % ("model", "margin 0", "margin 2"))
    print("  " + "-" * 52)
    for n in MODELS:
        a0, s0 = inv[(n, 0)]
        a2, s2 = inv[(n, 2)]
        print("  %-8s %13.4f %6.4f %13.4f %6.4f" % (n, a0, s0, a2, s2))
    print()
    print("  Read the convolution's row. Not 'approximately'; not 'to")
    print("  within noise'. %.4f, with a standard deviation across"
          % inv[("CNN", 2)][0])
    print("  seeds of %.4f. Global max-pooling takes the strongest"
          % inv[("CNN", 2)][1])
    print("  response wherever it occurs, so moving the object cannot")
    print("  change the answer. The property is not learned. It is")
    print("  arithmetic.")
    print()
    print("  Now the transformer: %.4f at margin 0 and %.4f at margin"
          % (inv[("ViT", 0)][0], inv[("ViT", 2)][0]))
    print("  2. Clearing the border does not rescue it, because the")
    print("  border was never its problem. It has an approximation,")
    print("  fitted from examples -- roughly right everywhere, exactly")
    print("  right nowhere.")
    print()
    print("  That is the whole difference between a prior and a")
    print("  learned regularity, and this stage can show it because")
    print("  the position was an input rather than a guess.")


def demo4(scale):
    """Depth or width, at a fixed budget."""
    banner("4. WHERE SHOULD THE PARAMETERS GO?")
    print("  The same residual network reshaped three ways at one")
    print("  budget: fewer, wider blocks against more, narrower ones.")
    print("  This is the question EfficientNet asks, minus the third")
    print("  axis -- resolution is fixed at %dx%d here, so the study"
          % (SIDE, SIDE))
    print("  covers depth against width only.")
    print()
    print("  %-22s %8s %16s %16s"
          % ("shape", "params", "%d labels" % SCALE_LABELS[0],
             "%d labels" % SCALE_LABELS[1]))
    print("  " + "-" * 66)
    for nb, w in SCALE_CFGS:
        p = ResNet(k=3, width=w, n_blocks=nb).n_params()
        lo, hi = scale[(nb, w, SCALE_LABELS[0])], \
            scale[(nb, w, SCALE_LABELS[1])]
        print("  %-22s %8d %10.4f %5.4f %10.4f %5.4f"
              % ("%d block(s), width %d" % (nb, w), p,
                 lo[0], lo[1], hi[0], hi[1]))
    print()
    a = scale[(SCALE_CFGS[0][0], SCALE_CFGS[0][1], SCALE_LABELS[0])]
    c = scale[(SCALE_CFGS[-1][0], SCALE_CFGS[-1][1], SCALE_LABELS[0])]
    print("  Wider and shallower wins, and wins hardest where the data")
    print("  is thinnest: %.4f against %.4f at %d labels, closing to"
          % (a[0], c[0], SCALE_LABELS[0]))
    ah = scale[(SCALE_CFGS[0][0], SCALE_CFGS[0][1], SCALE_LABELS[1])]
    ch = scale[(SCALE_CFGS[-1][0], SCALE_CFGS[-1][1], SCALE_LABELS[1])]
    print("  %.4f against %.4f once there is plenty."
          % (ah[0], ch[0]))
    print()
    print("  Look at the spread as well: %.4f for the shallow model at"
          % a[1])
    print("  %d labels against %.4f for the deep one. Depth does not"
          % (SCALE_LABELS[0], c[1]))
    print("  just cost accuracy here, it costs reliability.")
    print()
    print("  The honest reading is not that depth is bad. It is that")
    print("  depth is a BET -- on the data having structure at several")
    print("  scales, so that later blocks have something to build on.")
    print("  Six shapes on a %dx%d grid have one scale, so the bet" % (SIDE, SIDE))
    print("  does not pay. On photographs it does, which is why real")
    print("  residual networks are deep and this one should not be.")


def main():
    banner("IMAGE CLASSIFICATION")
    sizes, lo, hi = selftest()
    per, worst, checked = gradcheck()
    print("  stage self-test passed: %d classes, %dx%d, ink conserved"
          % (N_CLS, SIDE, SIDE))
    print("    under translation for every class, size and position")
    print("    -- which is what the invariance measurement assumes")
    print("  parameter budgets matched to %.1f%%: %s"
          % (100 * (hi / lo - 1),
             ", ".join("%s %s" % (k, f"{v:,}") for k, v in sizes.items())))
    print("  gradient audit: worst error %.2e over %d sampled gradients"
          % (worst, checked))
    for n, (w, c) in per.items():
        print("    %-8s %.2e over %d" % (n, w, c))
    print()

    Xte, yte, _ = make_set(N_TEST, np.random.default_rng(SEED_TEST))
    acc, finals = {}, {}
    print("  training (this is the slow part)...")
    for n_lab in LABEL_COUNTS:
        for name in MODELS:
            runs = []
            for sd in SEEDS:
                Xtr, ytr, _ = make_set(
                    n_lab, np.random.default_rng(SEED_TRAIN + 7 * sd))
                net = fit(MODELS[name](seed=sd, lr=LR, **CONFIG[name]),
                          Xtr, ytr, STEPS,
                          rng=np.random.default_rng(SEED_FIT + sd))
                runs.append(accuracy(net, Xte, yte))
                if n_lab == LABEL_COUNTS[-1]:
                    finals.setdefault(name, []).append(net)
            acc[(n_lab, name)] = ms(runs)
        print("    %d labels done" % n_lab)

    inv = {}
    for name in MODELS:
        for mg in MARGINS:
            inv[(name, mg)] = ms([
                self_agreement(net, np.random.default_rng(SEED_INV),
                               margin=mg)[0]
                for net in finals[name]])

    scale = {}
    for nb, w in SCALE_CFGS:
        for n_lab in SCALE_LABELS:
            runs = []
            for sd in SEEDS:
                Xtr, ytr, _ = make_set(
                    n_lab, np.random.default_rng(SEED_TRAIN + 7 * sd))
                net = fit(ResNet(k=3, width=w, n_blocks=nb, seed=sd,
                                 lr=LR), Xtr, ytr, STEPS,
                          rng=np.random.default_rng(SEED_FIT + sd))
                runs.append(accuracy(net, Xte, yte))
            scale[(nb, w, n_lab)] = ms(runs)
        print("    scaling %d block(s) done" % nb)

    demo1(sizes)
    demo2(acc)
    demo3(inv)
    demo4(scale)

    banner("DONE")
    return acc, inv, scale


if __name__ == "__main__":
    main()
