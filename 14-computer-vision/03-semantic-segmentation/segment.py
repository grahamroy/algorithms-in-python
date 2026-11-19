"""Semantic segmentation from scratch: what pooling costs, and what
a skip connection buys back.

Part 1 of this section measured a convolution's translation
invariance and found it exact: re-render the same shape at every
legal position and the answer never changes, 1.0000 with zero
variance across seeds.  That invariance is bought by pooling, which
is a deliberate destruction of position -- and it was the right
trade, because the question was "what is this" and position was the
nuisance.

Change the question to "what is each PIXEL" and the same mechanism
becomes the bill.  A segmentation head has to say where the boundary
falls to within a pixel, using features that were built by throwing
away exactly that.  U-Net's one idea is to hand the decoder the
encoder's feature map from before the pooling.

So this script holds the architecture, the parameter count, the
optimiser and the budget fixed, and changes only whether that skip
exists.  And because the stage DREW every region, it knows each
pixel's exact distance to the nearest boundary -- which turns "where
does the error live" from an impression into a measurement.

Two of this article's results exist because a first answer was
wrong, and the script says so where it reports them:

  * At 400 steps the skip looks essential (0.8477 against 0.0735).
    It is not: the loser was undertrained.  Worse, at 800 steps the
    comparison REVERSES.  Three seeds show why -- early training
    here is so variable that a small-budget comparison measures the
    seed, not the architecture.
  * The depth comparison first ran with depth 3 at 29% under budget,
    because uniform widths cannot hit the target there.  It is now
    bracketed on both sides.

Dependencies: numpy.  Runs in about twenty-eight minutes.
"""

import sys
import time
import itertools
from collections import OrderedDict

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


def softmax_last(Z):
    e = np.exp(Z - Z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def fmt_pm(v):
    v = np.asarray(v, float)
    return "%.4f +-%.4f" % (v.mean(), v.std())


# -----------------------------------------------------------------------
# The stage.
#
# Six filled shapes rather than Parts 1 and 2's strokes.  A stroke is
# one pixel wide, so every one of its pixels touches a boundary and
# "where does the error live" has no interior to compare against.
#
# Two of the six are PIERCED: their solid twin minus a small central
# hole.  They have an identical outer silhouette and differ on only a
# handful of pixels near the middle, which means a pixel on the RIM
# can only be labelled correctly by a model whose receptive field
# reaches the hole.  That makes receptive field measurable instead of
# merely arithmetic.
# -----------------------------------------------------------------------

SIDE = 32
CLASS_NAMES = ["background", "disc", "square", "triangle",
               "diamond", "pierced disc", "pierced square"]
N_CLS = len(CLASS_NAMES)
PIERCED = (5, 6)
TWIN = {5: 1, 6: 2}                   # pierced -> its solid twin
R_LO, R_HI = 4, 8                     # radius, exclusive upper
MAX_OBJ = 3
NOISE = 0.15
MIN_GAP = 2


def hole_radius(r):
    return max(1, r // 3)


def shape_mask(cls, r):
    """A (2r+1, 2r+1) boolean mask for one filled shape."""
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disc = yy ** 2 + xx ** 2 <= r * r
    square = np.ones_like(disc, bool)
    if cls == 1:
        return disc
    if cls == 2:
        return square
    if cls == 3:
        return (yy + r) >= (np.abs(xx) * 2)
    if cls == 4:
        return np.abs(yy) + np.abs(xx) <= r
    h = hole_radius(r)
    return (disc if cls == 5 else square) & ~(yy ** 2 + xx ** 2
                                              <= h * h)


def scene(rng, n_obj=None, min_gap=MIN_GAP):
    """An image and its label map.

    The label map is not an annotation -- it is what the script chose
    before it drew anything, so there is no annotator to disagree
    with and no boundary wobble to inherit.
    """
    n = int(rng.integers(1, MAX_OBJ + 1)) if n_obj is None else n_obj
    lab = np.zeros((SIDE, SIDE), np.int64)
    placed = []
    for _ in range(300):
        if len(placed) == n:
            break
        cls = int(rng.integers(1, N_CLS))
        r = int(rng.integers(R_LO, R_HI))
        cy = int(rng.integers(r, SIDE - r))
        cx = int(rng.integers(r, SIDE - r))
        if any(abs(cy - p[1]) <= r + p[3] + min_gap and
               abs(cx - p[2]) <= r + p[3] + min_gap for p in placed):
            continue
        lab[cy - r:cy + r + 1, cx - r:cx + r + 1][shape_mask(cls, r)] = cls
        placed.append((cls, cy, cx, r))
    return (lab > 0).astype(float), lab, placed


def boundary_distance(lab):
    """Exact 4-connected distance to the nearest DIFFERENT label.

    Seed every pixel that touches another region, then dilate inside
    same-label regions.  The loop runs once per distance band, not
    once per pixel.  Knowing this exactly is the whole reason the
    error decomposition below is possible.
    """
    diff = np.zeros_like(lab, bool)
    ne = lab[:-1, :] != lab[1:, :]
    diff[:-1, :] |= ne
    diff[1:, :] |= ne
    ne = lab[:, :-1] != lab[:, 1:]
    diff[:, :-1] |= ne
    diff[:, 1:] |= ne
    dist = np.where(diff, 0, -1).astype(np.int64)
    cur, d = diff.copy(), 0
    while True:
        nxt = np.zeros_like(cur)
        nxt[:-1, :] |= cur[1:, :] & (lab[:-1, :] == lab[1:, :])
        nxt[1:, :] |= cur[:-1, :] & (lab[1:, :] == lab[:-1, :])
        nxt[:, :-1] |= cur[:, 1:] & (lab[:, :-1] == lab[:, 1:])
        nxt[:, 1:] |= cur[:, :-1] & (lab[:, 1:] == lab[:, :-1])
        nxt &= dist < 0
        if not nxt.any():
            break
        d += 1
        dist[nxt] = d
        cur = nxt
    return dist


def make_set(n, seed):
    rng = np.random.default_rng(seed)
    X, Y, D = [], [], []
    for _ in range(n):
        img, lab, _ = scene(rng)
        X.append(img + rng.normal(0, NOISE, img.shape))
        Y.append(lab)
        D.append(boundary_distance(lab))
    return np.stack(X), np.stack(Y), np.stack(D)


# -----------------------------------------------------------------------
# Convolution, pooling, and the two operations a decoder needs.
# -----------------------------------------------------------------------

def im2col(X, k, pad=0):
    if pad:
        X = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    N, C, H, W = X.shape
    oh, ow = H - k + 1, W - k + 1
    cols = np.empty((N, C, k, k, oh, ow))
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = X[:, :, i:i + oh, j:j + ow]
    return (cols.transpose(0, 4, 5, 1, 2, 3)
                .reshape(N, oh * ow, C * k * k)), oh, ow


def col2im(dcols, shape, k, oh, ow, pad=0):
    N, C, H, W = shape
    d = dcols.reshape(N, oh, ow, C, k, k).transpose(0, 3, 4, 5, 1, 2)
    out = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for i in range(k):
        for j in range(k):
            out[:, :, i:i + oh, j:j + ow] += d[:, :, i, j]
    return out[:, :, pad:pad + H, pad:pad + W] if pad else out


def maxpool2(X):
    N, C, H, W = X.shape
    f = (X.reshape(N, C, H // 2, 2, W // 2, 2)
          .transpose(0, 1, 2, 4, 3, 5)
          .reshape(N, C, H // 2, W // 2, 4))
    return f.max(-1), f.argmax(-1)


def maxunpool2(d, idx, shape):
    N, C, H, W = shape
    f = np.zeros((N, C, H // 2, W // 2, 4))
    np.put_along_axis(f, idx[..., None], d[..., None], -1)
    return (f.reshape(N, C, H // 2, W // 2, 2, 2)
             .transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H, W))


def upsample2(X):
    """Nearest neighbour, 2x -- and deliberately PARAMETER-FREE.

    A transposed convolution would add capacity to the decoder, and
    the comparison below would then be measuring that capacity rather
    than the skip connection.
    """
    return np.repeat(np.repeat(X, 2, axis=2), 2, axis=3)


def upsample2_back(d):
    """Adjoint: every input cell was copied to four outputs, so the
    four gradients sum back onto it."""
    N, C, H, W = d.shape
    return (d.reshape(N, C, H // 2, 2, W // 2, 2)
             .transpose(0, 1, 2, 4, 3, 5)
             .reshape(N, C, H // 2, W // 2, 4).sum(-1))


class Conv:
    """k x k same-padded convolution, bias, optional ReLU."""

    def __init__(self, cin, cout, k, rng, relu=True):
        self.k, self.cin, self.cout, self.relu = k, cin, cout, relu
        self.W = rng.normal(0, np.sqrt(2.0 / (cin * k * k)),
                            (cin * k * k, cout))
        self.b = np.full(cout, 0.01)

    def params(self):
        return [self.W, self.b]

    def forward(self, X):
        cols, oh, ow = im2col(X, self.k, pad=self.k // 2)
        A = cols @ self.W + self.b
        H = np.maximum(A, 0) if self.relu else A
        out = H.reshape(X.shape[0], oh, ow,
                        self.cout).transpose(0, 3, 1, 2)
        return out, (X.shape, cols, A, oh, ow)

    def backward(self, dout, cache):
        shape, cols, A, oh, ow = cache
        N = shape[0]
        dH = dout.transpose(0, 2, 3, 1).reshape(N, oh * ow, self.cout)
        dA = dH * (A > 0) if self.relu else dH
        gW = np.einsum("nij,nik->jk", cols, dA)
        gb = dA.sum((0, 1))
        dX = col2im(dA @ self.W.T, shape, self.k, oh, ow,
                    pad=self.k // 2)
        return dX, [gW, gb]


class SegNet:
    """Encoder-decoder, with or without skip connections.

    depth D means D poolings down and D upsamplings back.  With
    skips=True the decoder is handed the encoder's feature map at the
    matching resolution; that concatenation is the ONLY structural
    difference between the two architectures compared here.
    """

    def __init__(self, depth=2, width=8, k=3, skips=True, rng=None):
        rng = rng or np.random.default_rng(0)
        self.depth, self.skips, self.width = depth, skips, width
        self.enc, self.dec, chans = [], [], []
        cin = 1
        for d in range(depth):
            c = width * (d + 1)
            self.enc.append(Conv(cin, c, k, rng))
            chans.append(c)
            cin = c
        self.mid = Conv(cin, cin, k, rng)
        for d in reversed(range(depth)):
            self.dec.append(Conv(cin + (chans[d] if skips else 0),
                                 chans[d], k, rng))
            cin = chans[d]
        self.head = Conv(cin, N_CLS, 1, rng, relu=False)
        self.layers = self.enc + [self.mid] + self.dec + [self.head]

    def n_params(self):
        return int(sum(p.size for l in self.layers for p in l.params()))

    def params(self):
        return [p for l in self.layers for p in l.params()]

    def flat_grads(self, grads):
        return [g for gs in grads for g in gs]

    def receptive_field(self):
        """How much input can reach one output pixel.  Each 3x3 conv
        adds 2 at its current scale; each pooling doubles the scale of
        everything after it."""
        rf, stride = 1, 1
        for _ in range(self.depth):
            rf += 2 * stride
            stride *= 2
        rf += 2 * stride
        for _ in range(self.depth):
            stride //= 2
            rf += 2 * stride
        return rf

    def forward(self, X):
        h = X[:, None]
        caches, skips, pools = [], [], []
        for layer in self.enc:
            h, c = layer.forward(h)
            caches.append(c)
            skips.append(h)
            p, idx = maxpool2(h)
            pools.append((idx, h.shape))
            h = p
        h, c = self.mid.forward(h)
        caches.append(c)
        for i, layer in enumerate(self.dec):
            h = upsample2(h)
            if self.skips:
                h = np.concatenate([h, skips[self.depth - 1 - i]], 1)
            h, c = layer.forward(h)
            caches.append(c)
        logits, c = self.head.forward(h)
        caches.append(c)
        return logits.transpose(0, 2, 3, 1), (caches, pools)

    def backward(self, dlogits, cache):
        caches, pools = cache
        grads = [None] * len(self.layers)
        d, g = self.head.backward(dlogits.transpose(0, 3, 1, 2),
                                  caches[-1])
        grads[-1] = g
        dskip = [None] * self.depth
        for i in range(len(self.dec) - 1, -1, -1):
            d, g = self.dec[i].backward(d, caches[1 + self.depth + i])
            grads[self.depth + 1 + i] = g
            if self.skips:
                c = d.shape[1] - self.enc[self.depth - 1 - i].cout
                dskip[self.depth - 1 - i] = d[:, c:]
                d = d[:, :c]
            d = upsample2_back(d)
        d, g = self.mid.backward(d, caches[self.depth])
        grads[self.depth] = g
        for i in range(self.depth - 1, -1, -1):
            idx, shape = pools[i]
            d = maxunpool2(d, idx, shape)
            if self.skips and dskip[i] is not None:
                d = d + dskip[i]
            d, g = self.enc[i].backward(d, caches[i])
            grads[i] = g
        return grads


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr = params, lr
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, g in enumerate(grads):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mh = self.m[i] / (1 - b1 ** self.t)
            vh = self.v[i] / (1 - b2 ** self.t)
            self.p[i] -= self.lr * mh / (np.sqrt(vh) + eps)


def loss_and_grad(logits, lab, weights=None):
    """Per-pixel softmax cross-entropy, optionally class-weighted."""
    N, H, W, C = logits.shape
    flat = softmax_last(logits).reshape(-1, C)
    y = lab.reshape(-1)
    w = np.ones(y.size) if weights is None else weights[y]
    ll = -np.log(flat[np.arange(y.size), y] + 1e-12)
    loss = float((ll * w).sum() / w.sum())
    d = flat.copy()
    d[np.arange(y.size), y] -= 1.0
    d *= (w / w.sum())[:, None]
    return loss, d.reshape(N, H, W, C)


# -----------------------------------------------------------------------
# Training and metrics.
# -----------------------------------------------------------------------

def fit(net, X, Y, steps, lr, seed, weights=None, batch=8, opt=None):
    """Returns the optimiser so training can CONTINUE at a checkpoint.

    Rebuilding Adam at every checkpoint would restart its moments and
    smuggle a warm-up into the middle of the curve.
    """
    opt = opt or Adam(net.params(), lr)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, len(X), batch)
        logits, cache = net.forward(X[idx])
        L, dl = loss_and_grad(logits, Y[idx], weights)
        opt.step(net.flat_grads(net.backward(dl, cache)))
    return opt, L


def predict(net, X, chunk=40):
    out = []
    for i in range(0, len(X), chunk):
        out.append(net.forward(X[i:i + chunk])[0].argmax(-1))
    return np.concatenate(out, 0)


def metrics(pred, Y, D):
    ious = []
    for c in range(N_CLS):
        inter = ((pred == c) & (Y == c)).sum()
        union = ((pred == c) | (Y == c)).sum()
        ious.append(inter / union if union else np.nan)
    fg = Y > 0
    acc_by_d = {}
    for k in range(6):
        m = (D == k) & fg
        acc_by_d[k] = (float((pred[m] == Y[m]).mean()) if m.any()
                       else np.nan)
    return dict(miou=float(np.nanmean(ious[1:])),
                pixacc=float((pred == Y).mean()),
                fg_acc=float((pred[fg] == Y[fg]).mean()),
                acc_by_d=acc_by_d, ious=ious)


def match_width(depth, target, skips, lo=3, hi=40):
    best, bw = None, None
    for w in range(lo, hi + 1):
        n = SegNet(depth=depth, width=w, skips=skips,
                   rng=np.random.default_rng(0)).n_params()
        if best is None or abs(n - target) < abs(best - target):
            best, bw = n, w
    return bw, best


# =======================================================================
# Self-tests.
# =======================================================================

def selftest_stage():
    banner("SELF-TEST 1: the stage knows exactly what it drew")
    rng = np.random.default_rng(0)
    counts, depth_ok = [], {}
    for _ in range(600):
        _, lab, placed = scene(rng)
        counts.append(len(placed))
    print("600 scenes, objects per scene mean %.4f (min %d max %d)"
          % (np.mean(counts), min(counts), max(counts)))

    # solid shapes must HAVE an interior, or the decomposition below
    # has nothing to compare the boundary against
    print()
    print("share of each shape's pixels at distance >= 2 from an edge")
    for c in range(1, N_CLS):
        row = []
        for r in range(R_LO, R_HI):
            lab = np.zeros((SIDE, SIDE), np.int64)
            m = shape_mask(c, r)
            lab[12 - r:12 + r + 1, 12 - r:12 + r + 1][m] = c
            d = boundary_distance(lab)
            row.append(((lab == c) & (d >= 2)).sum() / m.sum())
        depth_ok[c] = max(row)
        print("  %-15s %s" % (CLASS_NAMES[c],
                              " ".join("r=%d %.2f" % (r, v)
                                       for r, v in
                                       zip(range(R_LO, R_HI), row))))
    for c in (1, 2, 3, 4):
        assert depth_ok[c] > 0.25, "solid shapes need an interior"

    print()
    print("a pierced shape is its twin minus a hole near the centre:")
    for p, s in TWIN.items():
        for r in (R_LO, R_HI - 1):
            mp, ms = shape_mask(p, r), shape_mask(s, r)
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            far = np.sqrt(yy ** 2 + xx ** 2)[ms & ~mp]
            assert (mp & ~ms).sum() == 0, "must be a strict subset"
            print("  r=%d %-15s differs on %2d px, all within %.1f of"
                  " centre; its rim is %d px out"
                  % (r, CLASS_NAMES[p], (ms & ~mp).sum(), far.max(), r))
            assert far.max() < 0.5 * r, "the hole must be central"
    print()
    print("So a RIM pixel can only be labelled correctly by a model")
    print("whose receptive field reaches the middle of the shape.")


def selftest_metric():
    banner("SELF-TEST 2: the metric cannot be fooled by the majority")
    X, Y, D = make_set(60, 3)
    bg = np.zeros_like(Y)
    m = metrics(bg, Y, D)
    print("predicting BACKGROUND everywhere:")
    print("  pixel accuracy %.4f   <- which is why it is not reported"
          % m["pixacc"])
    print("  mean IoU       %.4f   <- which is why this one is"
          % m["miou"])
    assert m["miou"] < 1e-9, "mIoU must punish the majority answer"
    assert m["pixacc"] > 0.8, "the imbalance should be severe"

    perfect = metrics(Y, Y, D)
    print("\nperfect prediction: mIoU %.4f, pixel accuracy %.4f"
          % (perfect["miou"], perfect["pixacc"]))
    assert abs(perfect["miou"] - 1.0) < 1e-12
    shift = np.roll(Y, 1, axis=2)
    print("the same labels shifted one pixel sideways: mIoU %.4f"
          % metrics(shift, Y, D)["miou"])


def selftest_adjoints():
    banner("SELF-TEST 3: the decoder's two new operations are adjoint")
    r = np.random.default_rng(7)
    a = r.normal(0, 1, (2, 3, 4, 4))
    b = r.normal(0, 1, (2, 3, 8, 8))
    lhs = float((upsample2(a) * b).sum())
    rhs = float((a * upsample2_back(b)).sum())
    print("upsample:  <Ua, b> = %.6f   <a, U*b> = %.6f   diff %.2e"
          % (lhs, rhs, abs(lhs - rhs)))
    assert abs(lhs - rhs) < 1e-10, "upsample adjoint is wrong"

    z = r.normal(0, 1, (2, 3, 8, 8))
    p, idx = maxpool2(z)
    g = r.normal(0, 1, p.shape)
    up = maxunpool2(g, idx, z.shape)
    print("max-pool:  routed %d gradients to %d inputs, sum %.6f"
          " vs %.6f" % (g.size, int((up != 0).sum()), up.sum(),
                        g.sum()))
    assert int((up != 0).sum()) == g.size
    assert abs(up.sum() - g.sum()) < 1e-10
    print("\nBoth are exact, so any gradient disagreement below is")
    print("not coming from the decoder's plumbing.")


def selftest_gradients():
    banner("SELF-TEST 4: the hand-derived gradients are right")
    # A SMALL instance, on purpose.  A full 32x32 batch holds ~25,000
    # pre-activations per layer, so the smallest |pre-activation| is
    # near zero by sheer density and no step size avoids both kink
    # crossings and roundoff.  The chain rule under test is the same.
    # Scaled inputs, also on purpose.  At unit scale the LINEAR
    # variant below drives the logits far enough to saturate the
    # softmax, and both gradients collapse to ~0 -- their relative
    # difference is then meaningless rather than small.  ReLU hides
    # that by clipping; the linear check does not.
    r = np.random.default_rng(5)
    X = r.normal(0, 0.3, (2, 12, 12))
    Y = r.integers(0, N_CLS, (2, 12, 12))
    eps = 1e-5

    print("Part A -- LINEAR activations.")
    print("This removes the ReLU kink, which isolates the parts a")
    print("segmentation decoder adds: the skip concatenation, the")
    print("upsampling, and the pooling adjoint.  It does NOT make the")
    print("loss smooth -- max-pool still has a corner wherever two")
    print("inputs tie for the maximum, which is why the error below")
    print("stops falling with eps instead of vanishing.")
    print()
    worst_lin = 0.0
    for skips in (False, True):
        net = SegNet(depth=2, width=3, skips=skips,
                     rng=np.random.default_rng(1))
        for layer in net.layers:
            layer.relu = False
        w = _audit(net, X, Y, eps=eps, n=8)
        worst_lin = max(worst_lin, w)
        print("  skips=%-5s  worst relative error %.3e" % (skips, w))
    assert worst_lin < 1e-6, "the decoder plumbing is wrong"

    print()
    print("Part B -- the real ReLU network.")
    print("A ReLU is not differentiable at 0, so a central difference")
    print("is only valid if no unit sits within eps of the kink.")
    print("Measure that margin and assert it rather than hoping.")
    print()
    worst = 0.0
    for skips in (False, True):
        net = SegNet(depth=2, width=3, skips=skips,
                     rng=np.random.default_rng(1))
        margin = _margin(net, X)
        w = _audit(net, X, Y, eps=eps, n=8)
        worst = max(worst, w)
        print("  skips=%-5s  margin %.2e (%.0fx eps)  worst %.3e"
              % (skips, margin, margin / eps, w))
        assert margin > 10 * eps, "the audit point sits on the kink"
    assert worst < 1e-6, "gradients disagree with central differences"
    print()
    print("Both architectures check out to better than 1e-6.")
    print()
    print("This mattered.  The first audit ran on a full 32x32 batch")
    print("at eps=1e-4, reported ~2e-03, and looked like a bug in the")
    print("skip connections.  It was not: the margin there is ~1e-06,")
    print("so the difference was straddling the kink.  The backward")
    print("pass had been correct all along.")


def _margin(net, X):
    """Smallest |pre-activation| over every ReLU in the network."""
    _, (caches, _) = net.forward(X)
    vals = [np.abs(c[2]).min()
            for layer, c in zip(net.layers, caches) if layer.relu]
    return float(min(vals))


def _audit(net, X, Y, eps, n):
    logits, cache = net.forward(X)
    L, dl = loss_and_grad(logits, Y)
    grads = net.flat_grads(net.backward(dl, cache))
    ps = net.params()
    r = np.random.default_rng(3)
    worst = 0.0
    for pi, p in enumerate(ps):
        for _ in range(n):
            idx = tuple(int(r.integers(0, s)) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps
            Lp, _ = loss_and_grad(net.forward(X)[0], Y)
            p[idx] = old - eps
            Lm, _ = loss_and_grad(net.forward(X)[0], Y)
            p[idx] = old
            num = (Lp - Lm) / (2 * eps)
            den = max(1e-9, abs(num) + abs(grads[pi][idx]))
            worst = max(worst, abs(num - grads[pi][idx]) / den)
    return worst


# =======================================================================
# Measurements.
# =======================================================================

def measure_imbalance(Ytr):
    banner("MEASUREMENT 1: what the majority answer already scores")
    freq = np.bincount(Ytr.ravel(), minlength=N_CLS).astype(float)
    freq /= freq.sum()
    for c in range(N_CLS):
        print("  %-16s %.4f" % (CLASS_NAMES[c], freq[c]))
    print()
    print("Background is %.4f of every pixel, so a model that says"
          % freq[0])
    print("'background' and nothing else already scores %.4f pixel"
          % freq[0])
    print("accuracy.  Pixel accuracy is therefore not reported below.")
    print("Mean IoU over the six real classes gives that model 0.0000,")
    print("which is the right answer.")
    return freq


def measure_curve(Xtr, Ytr, Xte, Yte, Dte):
    banner("MEASUREMENT 2: the comparison, and when to take it")
    w_s, n_s = match_width(DEPTH, TARGET, True)
    w_n, n_n = match_width(DEPTH, TARGET, False)
    print("U-Net           depth %d width %2d  %5d params  RF %d px"
          % (DEPTH, w_s, n_s,
             SegNet(depth=DEPTH, width=w_s,
                    rng=np.random.default_rng(0)).receptive_field()))
    print("encoder-decoder depth %d width %2d  %5d params  RF %d px"
          % (DEPTH, w_n, n_n,
             SegNet(depth=DEPTH, width=w_n,
                    rng=np.random.default_rng(0)).receptive_field()))
    print()
    print("Each run is trained ONCE and evaluated along the way, with")
    print("the optimiser carried through, so the rows below are one")
    print("trajectory rather than independent runs.")
    print()
    res, nets = {}, {}
    for skips in (True, False):
        w = w_s if skips else w_n
        for s in SEEDS:
            net = SegNet(depth=DEPTH, width=w, skips=skips,
                         rng=np.random.default_rng(s + 1))
            opt, done, t0 = None, 0, time.time()
            for ck in CKPTS:
                opt, _ = fit(net, Xtr, Ytr, ck - done, LR, s, opt=opt)
                done = ck
                res[(skips, s, ck)] = metrics(predict(net, Xte),
                                              Yte, Dte)
            nets[(skips, s)] = net
            print("  %-16s seed %d  " % ("U-Net" if skips
                                         else "encoder-decoder", s) +
                  "  ".join("%d:%.4f" % (ck, res[(skips, s, ck)]["miou"])
                            for ck in CKPTS) +
                  "   [%.0fs]" % (time.time() - t0))
            sys.stdout.flush()

    print()
    print("mean IoU by budget (%d seeds)" % len(SEEDS))
    print("steps    U-Net              encoder-decoder     gap")
    for ck in CKPTS:
        a = [res[(True, s, ck)]["miou"] for s in SEEDS]
        b = [res[(False, s, ck)]["miou"] for s in SEEDS]
        print("%5d    %s    %s    %+.4f"
              % (ck, fmt_pm(a), fmt_pm(b), np.mean(a) - np.mean(b)))
    early = CKPTS[len(CKPTS) // 2 - 1]
    ea = [res[(True, s, early)]["miou"] for s in SEEDS]
    print()
    print("Read the early rows and you can conclude anything you")
    print("like.  At %d steps the U-Net's three seeds are %s" %
          (early, ", ".join("%.4f" % v for v in ea)))
    print("-- a spread of %.4f, which is wider than the final gap."
          % (max(ea) - min(ea)))
    if np.mean(ea) < np.mean([res[(False, s, early)]["miou"]
                              for s in SEEDS]):
        print("At that budget the U-Net's MEAN is actually BEHIND, and")
        print("the ranking reverses again by the end.  A single run at")
        print("a small budget measures the seed, not the architecture.")
    return res, nets, (n_s, n_n)


def measure_boundary(res):
    banner("MEASUREMENT 3: where the difference actually lives")
    ck = CKPTS[-1]
    print("Accuracy on foreground pixels, split by exact distance to")
    print("the nearest boundary, at %d steps.  Distance 0 means the" % ck)
    print("pixel touches another region.")
    print()
    print("architecture       " + "   ".join("d=%d" % k
                                             for k in range(6)))
    rows = {}
    for skips in (True, False):
        acc = [np.nanmean([res[(skips, s, ck)]["acc_by_d"][k]
                           for s in SEEDS]) for k in range(6)]
        rows[skips] = acc
        print("  %-16s " % ("U-Net" if skips else "encoder-decoder") +
              " ".join("%.4f" % v for v in acc))
    gap = [rows[True][k] - rows[False][k] for k in range(6)]
    print("  %-16s " % "difference" +
          " ".join("%+.4f" % v for v in gap))
    print()
    elsewhere = max(abs(g) for g in gap[1:])
    print("The difference at the boundary is %+.4f.  The largest"
          % gap[0])
    print("difference at any other distance is %.4f -- %.0f times"
          % (elsewhere, gap[0] / elsewhere if elsewhere else 0))
    print("smaller.")
    print()
    if gap[0] > 5 * elsewhere:
        print("A skip connection buys boundary accuracy and almost")
        print("nothing else.  That is what the mechanism predicts:")
        print("pooling discards position, only the pixels next to a")
        print("boundary need it back, and the interior was never in")
        print("any difficulty.")
    else:
        print("The difference is NOT concentrated at the boundary,")
        print("which contradicts the usual explanation of what a skip")
        print("connection does.")
    return rows, gap


def probe(cls, r, n, seed):
    """Scenes holding exactly one shape, plus a mask of its rim."""
    rng = np.random.default_rng(seed)
    m = shape_mask(cls, r)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    rim_local = m & (np.sqrt(yy ** 2 + xx ** 2) > 0.6 * r)
    X, Y, R = [], [], []
    for _ in range(n):
        cy = int(rng.integers(r, SIDE - r))
        cx = int(rng.integers(r, SIDE - r))
        lab = np.zeros((SIDE, SIDE), np.int64)
        rim = np.zeros((SIDE, SIDE), bool)
        lab[cy - r:cy + r + 1, cx - r:cx + r + 1][m] = cls
        rim[cy - r:cy + r + 1, cx - r:cx + r + 1][rim_local] = True
        img = (lab > 0).astype(float)
        X.append(img + rng.normal(0, NOISE, img.shape))
        Y.append(lab)
        R.append(rim)
    return np.stack(X), np.stack(Y), np.stack(R)


def measure_depth(Xtr, Ytr, Xte, Yte, Dte, nets):
    banner("MEASUREMENT 4: receptive field, and what depth costs")
    print("Depth buys reach.  At a fixed parameter budget it is paid")
    print("for in width, so this is a trade and not a free lunch.")
    print()
    print("Uniform widths cannot hit %d parameters at depth 3 -- the"
          % TARGET)
    print("two nearest are 7,043 and 10,962 -- so depth 3 is run at")
    print("BOTH, and only loses if it loses at both.")
    print()
    rows = OrderedDict()
    for depth, width in DEPTH_CFGS:
        key = (depth, width)
        mious = []
        for s in SEEDS:
            if key == (DEPTH, MAIN_WIDTH) and (True, s) in nets:
                net = nets[(True, s)]       # already trained above
            else:
                net = SegNet(depth=depth, width=width, skips=True,
                             rng=np.random.default_rng(s + 1))
                fit(net, Xtr, Ytr, STEPS, LR, s)
                nets[(depth, width, s)] = net
            mious.append(metrics(predict(net, Xte), Yte,
                                 Dte)["miou"])
        n = SegNet(depth=depth, width=width,
                   rng=np.random.default_rng(0)).n_params()
        rf = SegNet(depth=depth, width=width,
                    rng=np.random.default_rng(0)).receptive_field()
        rows[key] = (n, rf, mious)
        print("  depth %d width %2d  %5d params  RF %2d px  mIoU %s"
              % (depth, width, n, rf, fmt_pm(mious)) +
              ("   seeds %s" % " ".join("%.4f" % v for v in mious)
               if np.std(mious) > 0.05 else ""))
        sys.stdout.flush()

    print()
    print("Rim accuracy on the PIERCED shapes, by radius.  A rim pixel")
    print("is only correct if the model saw the hole, which is up to r")
    print("pixels away -- so this is the receptive field, measured.")
    print()
    print("depth  RF    " + "   ".join("r=%d" % r
                                       for r in range(R_LO, R_HI)))
    rim = OrderedDict()
    for depth, width in DEPTH_CFGS:
        row = []
        for r in range(R_LO, R_HI):
            accs = []
            for s in SEEDS:
                key = ((True, s) if (depth, width) == (DEPTH, MAIN_WIDTH)
                       else (depth, width, s))
                net = nets[key]
                tot = ok = 0
                for cls in PIERCED:
                    Xp, Yp, Rp = probe(cls, r, 24, 400 + r)
                    p = predict(net, Xp)
                    tot += Rp.sum()
                    ok += (p[Rp] == cls).sum()
                accs.append(ok / max(1, tot))
            row.append(float(np.mean(accs)))
        rim[(depth, width)] = row
        print("  %d w%-2d %2dpx  %s"
              % (depth, width, rows[(depth, width)][1],
                 "  ".join("%.4f" % v for v in row)))
        sys.stdout.flush()

    shallow = rim[DEPTH_CFGS[0]]
    print()
    if shallow[0] - shallow[-1] > 0.15:
        print("The shallow model is fine on small shapes and fails on")
        print("large ones (%.4f at r=%d, %.4f at r=%d).  Its receptive"
              % (shallow[0], R_LO, shallow[-1], R_HI - 1))
        print("field is %d px, a radius of %d, and the hole it needs to"
              % (rows[DEPTH_CFGS[0]][1], rows[DEPTH_CFGS[0]][1] // 2))
        print("see sits up to %d px away.  The failure appears exactly"
              % (R_HI - 1))
        print("where the shape outgrows what the model can see.")
    else:
        print("Rim accuracy does not fall with radius, so the")
        print("receptive-field explanation is not supported here.")

    print()
    print("Does it fail in the predicted direction? What does the")
    print("shallow model call a large pierced shape rim?")
    print()
    twin_is_top_error = []
    for cls in PIERCED:
        counts = np.zeros(N_CLS)
        for s in SEEDS:
            key = ((True, s) if DEPTH_CFGS[0] == (DEPTH, MAIN_WIDTH)
                   else (DEPTH_CFGS[0][0], DEPTH_CFGS[0][1], s))
            Xp, Yp, Rp = probe(cls, R_HI - 1, 24, 400 + R_HI - 1)
            counts += np.bincount(predict(nets[key], Xp)[Rp],
                                  minlength=N_CLS)
        counts /= counts.sum()
        order = np.argsort(-counts)[:2]
        print("  %-15s -> %s" % (CLASS_NAMES[cls],
                                 ",  ".join("%s %.4f"
                                            % (CLASS_NAMES[c], counts[c])
                                            for c in order)))
        # the most common WRONG answer, whatever it is
        wrong = [c for c in np.argsort(-counts) if c != cls]
        top_wrong = wrong[0]
        twin_is_top_error.append(top_wrong == TWIN[cls])
        print("     most common error: '%s' (%.4f); its solid twin"
              " is '%s'" % (CLASS_NAMES[top_wrong], counts[top_wrong],
                            CLASS_NAMES[TWIN[cls]]))
    print()
    if all(twin_is_top_error):
        print("For both shapes the most common error IS the solid")
        print("twin, which is precisely what 'it cannot see the hole'")
        print("predicts -- the model reports the silhouette it can")
        print("see and misses the evidence that would change the")
        print("answer.")
    elif any(twin_is_top_error):
        print("Only one of the two is confused with its solid twin,")
        print("so the mechanism is suggestive here rather than shown.")
    else:
        print("Neither shape's commonest error is its solid twin, so")
        print("these rim mistakes are not the hole going unseen --")
        print("the explanation above does not survive its own test.")
    return rows, rim


def measure_weighting(Xtr, Ytr, Xte, Yte, Dte, res):
    banner("MEASUREMENT 5: does re-weighting the loss help?")
    freq = np.bincount(Ytr.ravel(), minlength=N_CLS).astype(float)
    freq /= freq.sum()
    print("Background outnumbers every real class by more than 15 to")
    print("1, so a plain per-pixel loss spends most of its gradient on")
    print("the easy answer.  The standard fix is to weight each class")
    print("by the inverse of its frequency.")
    print()
    schemes = OrderedDict()
    schemes["plain"] = None
    inv = 1.0 / freq
    schemes["inverse frequency"] = inv / inv.mean()
    sq = 1.0 / np.sqrt(freq)
    schemes["sqrt inverse"] = sq / sq.mean()

    out = OrderedDict()
    ck = CKPTS[-1]
    for name, weights in schemes.items():
        if name == "plain":
            mi = [res[(True, s, ck)]["miou"] for s in SEEDS]
            fg = [res[(True, s, ck)]["fg_acc"] for s in SEEDS]
        else:
            mi, fg = [], []
            for s in SEEDS:
                net = SegNet(depth=DEPTH, width=MAIN_WIDTH, skips=True,
                             rng=np.random.default_rng(s + 1))
                fit(net, Xtr, Ytr, STEPS, LR, s, weights=weights)
                m = metrics(predict(net, Xte), Yte, Dte)
                mi.append(m["miou"])
                fg.append(m["fg_acc"])
        out[name] = (mi, fg)
        print("  %-20s mIoU %s   foreground acc %s"
              % (name, fmt_pm(mi), fmt_pm(fg)))
        sys.stdout.flush()

    base_mi = np.mean(out["plain"][0])
    base_sd = np.std(out["plain"][0])
    best = max(out, key=lambda k: np.mean(out[k][0]))
    gap = np.mean(out[best][0]) - base_mi
    print()
    print("Best scheme on mIoU is '%s', %+.4f over plain, against"
          % (best, gap))
    print("plain's own seed spread of %.4f." % base_sd)
    if best == "plain" or gap <= max(base_sd, np.std(out[best][0])):
        print("That is inside the noise.  On this stage the schemes")
        print("cannot be told apart, and naming a winner would be")
        print("reporting a decimal point.")
    else:
        print("That clears the seed spread, so it is doing real work.")

    fg_base = np.mean(out["plain"][1])
    fg_best = max(out, key=lambda k: np.mean(out[k][1]))
    fg_gap = np.mean(out[fg_best][1]) - fg_base
    fg_sd = max(np.std(out["plain"][1]), np.std(out[fg_best][1]))
    print()
    if fg_best != "plain" and fg_gap > fg_sd:
        print("On FOREGROUND accuracy, though, '%s' gains %+.4f"
              % (fg_best, fg_gap))
        print("with a worst spread of %.4f -- a small effect, but a"
              % fg_sd)
        print("consistent one.  The weighting moves the errors off the")
        print("rare classes without moving the overlap much.")
    else:
        print("Foreground accuracy tells the same inconclusive story.")
    return out


# =======================================================================
# Configuration, frozen.
# =======================================================================

DEPTH = 2
TARGET = 9000
MAIN_WIDTH = 8                 # depth 2, skips, 9,991 params
DEPTH_CFGS = ((1, 18), (2, 8), (3, 4), (3, 5))
STEPS = 1600
CKPTS = (200, 400, 800, 1600)
SEEDS = (0, 1, 2)
LR = 3e-3
N_TRAIN, N_TEST = 300, 120
SEED_TRAIN, SEED_TEST = 10, 7


def main():
    t0 = time.time()
    banner("SEMANTIC SEGMENTATION: THE BILL FOR THROWING AWAY POSITION")
    print("canvas %dx%d   labels %d (background + %d shapes)"
          % (SIDE, SIDE, N_CLS, N_CLS - 1))
    print("objects 1-%d per scene, radius %d-%d, noise %.2f"
          % (MAX_OBJ, R_LO, R_HI - 1, NOISE))
    print("train %d scenes   test %d scenes   %d steps   %d seeds"
          % (N_TRAIN, N_TEST, STEPS, len(SEEDS)))

    selftest_stage()
    selftest_metric()
    selftest_adjoints()
    selftest_gradients()

    Xtr, Ytr, Dtr = make_set(N_TRAIN, SEED_TRAIN)
    Xte, Yte, Dte = make_set(N_TEST, SEED_TEST)

    measure_imbalance(Ytr)
    res, nets, params = measure_curve(Xtr, Ytr, Xte, Yte, Dte)
    measure_boundary(res)
    measure_depth(Xtr, Ytr, Xte, Yte, Dte, nets)
    measure_weighting(Xtr, Ytr, Xte, Yte, Dte, res)

    banner("DONE")
    print("total runtime %.1f minutes" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
