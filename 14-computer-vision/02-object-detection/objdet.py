"""Object detection from scratch: the assignment problem underneath.

Part 1 of this section compared four architectures on a question with
exactly one answer per image: which of six shapes is this?  Every model
there ended in a softmax over six classes, and the training target was
simply the label the stage had drawn from.

Put two shapes in the picture and that whole arrangement collapses.  The
output is no longer one label but a variable-length list of boxes, and
-- the part that is easy to miss -- THERE IS NO LONGER A TRAINING TARGET.
A detector emits a fixed number of predictions (64 here).  A scene
contains some other number of objects (2.46 on average).  Before any
loss can be evaluated, something has to decide which prediction is
answerable for which object.  That decision is called assignment, and
it is not a detail of the optimiser: it is the design decision that
separates YOLO from Faster R-CNN from DETR.

So this script holds the architecture, the loss terms, the parameter
count and the step budget fixed, and changes only the assignment rule:

  centre      the cell containing an object's centre owns it (YOLOv1).
              Two centres in one cell and the second object is DROPPED.
  many        every cell whose centre falls inside the box is positive
              and they all regress to it (the anchor-based family).
              Manufactures duplicates by design.
  hungarian   an exact one-to-one bipartite match against the model's
              CURRENT predictions (DETR).  The assignment moves while
              the model trains.

What makes the measurements here sharp is the same thing that made
Part 1's invariance measurement sharp: the script draws the scenes, so
it knows the true answer.  Not just where the objects are, but which
predictor was supposed to own each one.  That turns "the detector
missed it" into an attributable event -- the owner stayed silent, or it
fired in the wrong place, or it fired with the wrong label -- which is
a decomposition no real benchmark can perform.

Dependencies: numpy.  Runs in about nineteen minutes.
"""

import sys
import time
import itertools

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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def softmax_rows(Z):
    e = np.exp(Z - Z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def fmt_pm(v):
    v = np.asarray(v, float)
    return "%.4f +-%.4f" % (v.mean(), v.std())


# -----------------------------------------------------------------------
# The stage: Part 1's shape world, now with several shapes per canvas.
#
# The canvas doubles to 32x32 so there is room for up to four objects.
# The six classes are unchanged, which means a detector cannot cheat on
# appearance -- a cross and a vertical bar share strokes, a hollow
# square and a corner L share two sides -- and still has to say WHERE.
#
# Shapes are kept from touching (min_gap) for one reason: it makes the
# ground truth a clean list.  When boxes may overlap, "how many objects
# are there" stops having a single defensible answer, and every number
# downstream inherits that ambiguity.  The cost of the choice is that
# this stage is uncrowded, which is measured rather than assumed in the
# grid-resolution section below.
# -----------------------------------------------------------------------

SIDE = 32
N_CLS = 6
CLASS_NAMES = ["cross", "hollow square", "diagonal X",
               "corner L", "vertical bar", "horizontal bar"]
SIZE_LO, SIZE_HI = 2, 5          # half-width, exclusive upper
MAX_OBJ = 4
NOISE = 0.15


def stamp(img, cls, cy, cx, s, bright=1.0):
    """Add one shape to an existing canvas, centred at (cy, cx)."""
    if cls == 0:                                    # cross
        img[cy - s:cy + s + 1, cx] += bright
        img[cy, cx - s:cx + s + 1] += bright
    elif cls == 1:                                  # hollow square
        img[cy - s:cy + s + 1, cx - s] += bright
        img[cy - s:cy + s + 1, cx + s] += bright
        img[cy - s, cx - s:cx + s + 1] += bright
        img[cy + s, cx - s:cx + s + 1] += bright
    elif cls == 2:                                  # diagonal X
        for d in range(-s, s + 1):
            img[cy + d, cx + d] += bright
            img[cy + d, cx - d] += bright
    elif cls == 3:                                  # corner L
        img[cy - s:cy + s + 1, cx - s] += bright
        img[cy + s, cx - s:cx + s + 1] += bright
    elif cls == 4:                                  # vertical bar
        img[cy - s:cy + s + 1, cx] += bright
    else:                                           # horizontal bar
        img[cy, cx - s:cx + s + 1] += bright
    return img


def scene(rng, n_obj=None, min_gap=1):
    """A canvas and the boxes it was drawn from.

    Returns (clean image, [(cls, cy, cx, s), ...]).  The object list IS
    the ground truth -- it is what the stage chose, not an annotation.
    """
    n = int(rng.integers(1, MAX_OBJ + 1)) if n_obj is None else n_obj
    img = np.zeros((SIDE, SIDE))
    objs = []
    for _ in range(400):
        if len(objs) == n:
            break
        cls = int(rng.integers(0, N_CLS))
        s = int(rng.integers(SIZE_LO, SIZE_HI))
        cy = int(rng.integers(s, SIDE - s))
        cx = int(rng.integers(s, SIDE - s))
        if any(abs(cy - o[1]) <= s + o[3] + min_gap and
               abs(cx - o[2]) <= s + o[3] + min_gap for o in objs):
            continue
        stamp(img, cls, cy, cx, s)
        objs.append((cls, cy, cx, s))
    return np.clip(img, 0.0, 1.0), objs


def make_scenes(n, seed, noise=NOISE):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        img, objs = scene(rng)
        out.append((img + rng.normal(0, noise, img.shape), objs))
    return out


def obj_to_box(o):
    """(cls, cy, cx, s) -> [y0, x0, y1, x1], inclusive pixel bounds."""
    _, cy, cx, s = o
    return np.array([cy - s, cx - s, cy + s, cx + s], float)


def iou(a, b):
    y0 = max(a[0], b[0])
    x0 = max(a[1], b[1])
    y1 = min(a[2], b[2])
    x1 = min(a[3], b[3])
    ih = max(0.0, y1 - y0 + 1.0)
    iw = max(0.0, x1 - x0 + 1.0)
    inter = ih * iw
    aa = (a[2] - a[0] + 1.0) * (a[3] - a[1] + 1.0)
    bb = (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0)
    return inter / (aa + bb - inter)


# -----------------------------------------------------------------------
# The output grid.
#
# The backbone pools twice, so a 32x32 canvas leaves an 8x8 feature map
# and the detector has 64 places to speak from.  One predictor per cell
# is the cheapest possible answer to "how do we emit a variable number
# of things from a fixed-size network": emit a fixed number, and let
# most of them say nothing.
# -----------------------------------------------------------------------

GRID = 8
CELL = SIDE // GRID                 # 4 pixels
N_PRED = GRID * GRID                # 64 predictors
N_OUT = 1 + N_CLS + 3               # objectness, classes, (dy, dx, size)
S_MID = (SIZE_LO + SIZE_HI - 1) / 2.0
S_HALF = max(1.0, (SIZE_HI - 1 - SIZE_LO) / 2.0)


def cell_centre(p):
    gy, gx = divmod(p, GRID)
    return gy * CELL + (CELL - 1) / 2.0, gx * CELL + (CELL - 1) / 2.0


def owner_cell(o):
    """The cell the centre-assignment rule would put this object in."""
    _, cy, cx, _ = o
    return int((cy // CELL) * GRID + (cx // CELL))


# -----------------------------------------------------------------------
# Convolution as one matrix multiply, as in Part 1 -- but now the
# feature maps have channels, so im2col carries a channel axis and the
# loop still runs over the KERNEL rather than over output positions.
# -----------------------------------------------------------------------

def im2col(X, k, pad=0):
    """(N,C,H,W) -> ((N, oh*ow, C*k*k), oh, ow)."""
    if pad:
        X = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    N, C, H, W = X.shape
    oh, ow = H - k + 1, W - k + 1
    cols = np.empty((N, C, k, k, oh, ow))
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = X[:, :, i:i + oh, j:j + ow]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N, oh * ow, C * k * k)
    return cols, oh, ow


def col2im(dcols, shape, k, oh, ow, pad=0):
    """Adjoint of im2col: overlapping windows must accumulate."""
    N, C, H, W = shape
    d = dcols.reshape(N, oh, ow, C, k, k).transpose(0, 3, 4, 5, 1, 2)
    out = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    for i in range(k):
        for j in range(k):
            out[:, :, i:i + oh, j:j + ow] += d[:, :, i, j]
    return out[:, :, pad:pad + H, pad:pad + W] if pad else out


def maxpool2(X):
    N, C, H, W = X.shape
    r = X.reshape(N, C, H // 2, 2, W // 2, 2)
    f = r.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H // 2, W // 2, 4)
    return f.max(-1), f.argmax(-1)


def maxunpool2(d, idx, shape):
    N, C, H, W = shape
    f = np.zeros((N, C, H // 2, W // 2, 4))
    np.put_along_axis(f, idx[..., None], d[..., None], -1)
    r = f.reshape(N, C, H // 2, W // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5)
    return r.reshape(N, C, H, W)


class Detector:
    """conv-pool-conv-pool, then a 1x1 head applied at every cell.

    32x32 -> 16x16 -> 8x8, so the 64 predictors fall out of the
    architecture rather than being bolted on.  The head is shared
    across cells, which is the same weight-sharing argument Part 1
    made for convolution: a shape near the top-left and the same shape
    near the bottom-right should be described by the same function.
    """

    def __init__(self, c1=16, c2=32, k=3, rng=None):
        rng = rng or np.random.default_rng(0)
        self.k = k
        self.W1 = rng.normal(0, np.sqrt(2.0 / (k * k)), (k * k, c1))
        self.W2 = rng.normal(0, np.sqrt(2.0 / (c1 * k * k)),
                             (c1 * k * k, c2))
        self.W3 = rng.normal(0, np.sqrt(1.0 / c2), (c2, N_OUT))
        # A small POSITIVE bias, not zero.  The canvas is 97% empty, so
        # with zero biases most pre-activations are exactly zero, the
        # ReLU blocks them, and the first layer only ever sees gradient
        # from the few windows containing ink.  See the gradient audit.
        self.b1 = np.full(c1, 0.01)
        self.b2 = np.full(c2, 0.01)
        self.b3 = np.zeros(N_OUT)
        self.params = [self.W1, self.b1, self.W2, self.b2,
                       self.W3, self.b3]

    def n_params(self):
        return int(sum(p.size for p in self.params))

    def forward(self, X):
        N = X.shape[0]
        X4 = X[:, None]
        c1, oh1, ow1 = im2col(X4, self.k, pad=self.k // 2)
        A1 = c1 @ self.W1 + self.b1
        M1 = np.maximum(A1, 0).reshape(N, oh1, ow1, -1).transpose(
            0, 3, 1, 2)
        P1, i1 = maxpool2(M1)
        c2, oh2, ow2 = im2col(P1, self.k, pad=self.k // 2)
        A2 = c2 @ self.W2 + self.b2
        M2 = np.maximum(A2, 0).reshape(N, oh2, ow2, -1).transpose(
            0, 3, 1, 2)
        P2, i2 = maxpool2(M2)
        F = P2.transpose(0, 2, 3, 1).reshape(N, N_PRED, -1)
        out = F @ self.W3 + self.b3
        return out, (X4, c1, A1, M1.shape, i1, P1.shape,
                     c2, A2, M2.shape, i2, F)

    def backward(self, dout, cache):
        (X4, c1, A1, s1, i1, sP1, c2, A2, s2, i2, F) = cache
        N = X4.shape[0]
        gW3 = np.einsum("nij,nik->jk", F, dout)
        gb3 = dout.sum((0, 1))
        dP2 = (dout @ self.W3.T).reshape(
            N, GRID, GRID, -1).transpose(0, 3, 1, 2)
        dA2 = maxunpool2(dP2, i2, s2).transpose(0, 2, 3, 1).reshape(
            N, s2[2] * s2[3], -1) * (A2 > 0)
        gW2 = np.einsum("nij,nik->jk", c2, dA2)
        gb2 = dA2.sum((0, 1))
        dP1 = col2im(dA2 @ self.W2.T, sP1, self.k,
                     s2[2], s2[3], pad=self.k // 2)
        dA1 = maxunpool2(dP1, i1, s1).transpose(0, 2, 3, 1).reshape(
            N, s1[2] * s1[3], -1) * (A1 > 0)
        gW1 = np.einsum("nij,nik->jk", c1, dA1)
        gb1 = dA1.sum((0, 1))
        return [gW1, gb1, gW2, gb2, gW3, gb3]


class Adam:
    def __init__(self, params, lr=3e-3):
        self.p = params
        self.lr = lr
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


# -----------------------------------------------------------------------
# The loss -- which is only a function once an assignment exists.
#
# Three terms.  Objectness is scored at every one of the 64 predictors;
# class and box are scored ONLY where the assignment put an object.
# L_NOOBJ down-weights the predictors that are meant to stay quiet,
# which is YOLO's answer to the fact that most predictors are negative.
# Whether that helps here is measured rather than assumed.
# -----------------------------------------------------------------------

L_OBJ, L_CLS, L_BOX = 1.0, 1.0, 5.0
L_NOOBJ = 1.0

# Does a predictor's box have to lie inside its own cell?
#
# YOLOv1 put a sigmoid on the centre offset, which does exactly that.
# It is free under the centre rule, whose targets are inside the cell
# by definition -- but it makes the one-to-many rule's outer targets
# UNREACHABLE, and then scores the detector on boxes it was never able
# to emit.  The default is therefore an unbounded (linear) offset, the
# same for every rule, so the comparison is about assignment and not
# about decoding.  The coupling is measured in its own section.
BOUND_BOX = False


def xy_decode(ry, rx):
    """Raw head outputs -> centre offset in cells, and d(offset)/d(raw)."""
    if BOUND_BOX:
        py, px = sigmoid(ry), sigmoid(rx)
        return py, px, py * (1 - py), px * (1 - px)
    return ry, rx, np.ones_like(np.asarray(ry, float)), \
        np.ones_like(np.asarray(rx, float))


def targets_for(objs):
    """(predictor, class, dy, dx, size) under the centre rule."""
    rows = []
    for (cls, cy, cx, s) in objs:
        gy, gx = cy // CELL, cx // CELL
        rows.append((int(gy * GRID + gx), cls,
                     (cy % CELL) / CELL, (cx % CELL) / CELL,
                     (s - S_MID) / S_HALF))
    return rows


def box_row(p, cls, cy, cx, s):
    """Encode an object as a target for an ARBITRARY predictor p.

    The offset is measured in cells and may exceed 1, which is the
    point: a predictor two cells away still has a well-defined target.
    It is only clipped when the decoder cannot reach outside the cell.
    """
    gy, gx = divmod(p, GRID)
    ty = (cy - gy * CELL) / CELL
    tx = (cx - gx * CELL) / CELL
    if BOUND_BOX:
        ty, tx = np.clip(ty, 0.0, 1.0), np.clip(tx, 0.0, 1.0)
    return (p, cls, float(ty), float(tx), (s - S_MID) / S_HALF)


def loss_and_grad(out, assign):
    """assign[n] = [(predictor, class, dy, dx, size_norm), ...]."""
    N = out.shape[0]
    dout = np.zeros_like(out)
    tot = 0.0
    obj_t = np.zeros(out.shape[:2])
    for n, rows in enumerate(assign):
        for r in rows:
            obj_t[n, r[0]] = 1.0
    o = sigmoid(out[:, :, 0])
    w = np.where(obj_t > 0, L_OBJ, L_OBJ * L_NOOBJ)
    tot += -(w * (obj_t * np.log(o + 1e-12) +
                  (1 - obj_t) * np.log(1 - o + 1e-12))).sum()
    dout[:, :, 0] = w * (o - obj_t)
    for n, rows in enumerate(assign):
        for (p, c, dy, dx, sn) in rows:
            pr = softmax_rows(out[n, p, 1:1 + N_CLS][None])[0]
            tot += L_CLS * -np.log(pr[c] + 1e-12)
            g = pr.copy()
            g[c] -= 1.0
            dout[n, p, 1:1 + N_CLS] += L_CLS * g
            raw = out[n, p, 1 + N_CLS:]
            py, px, jy, jx = xy_decode(raw[0], raw[1])
            diff = np.array([py - dy, px - dx, raw[2] - sn])
            tot += L_BOX * 0.5 * (diff ** 2).sum()
            dout[n, p, 1 + N_CLS + 0] += L_BOX * diff[0] * jy
            dout[n, p, 1 + N_CLS + 1] += L_BOX * diff[1] * jx
            dout[n, p, 1 + N_CLS + 2] += L_BOX * diff[2]
    return tot / N, dout / N


# -----------------------------------------------------------------------
# The three assignment rules.  This is the whole article.
# -----------------------------------------------------------------------

def assign_centre(objs, out_n=None):
    """YOLOv1.  The cell holding the centre owns the object.

    Static: it depends only on the ground truth, so it can be computed
    once before training.  If two centres share a cell the second
    object is dropped -- the representation has nowhere to put it.
    """
    rows, taken = [], set()
    for t in targets_for(objs):
        if t[0] in taken:
            continue
        taken.add(t[0])
        rows.append(t)
    return rows


def assign_many(objs, out_n=None, radius=1.5):
    """The anchor-based family.  Every cell whose centre lies inside
    the object (plus a little slack) is a positive, and all of them
    regress to the same box.

    Also static, and it manufactures duplicates on purpose: several
    predictors are trained to report the same object, so something
    downstream has to remove the copies.  That something is NMS.
    """
    claim = {}
    for (cls, cy, cx, s) in objs:
        for p in range(N_PRED):
            py, px = cell_centre(p)
            if abs(py - cy) <= s + radius and abs(px - cx) <= s + radius:
                d = (py - cy) ** 2 + (px - cx) ** 2
                if p not in claim or d < claim[p][0]:
                    claim[p] = (d, box_row(p, cls, cy, cx, s))
    return [v[1] for v in claim.values()]


def hungarian(cost):
    """Exact rectangular assignment by shortest augmenting paths.

    cost is (n_rows, n_cols) with n_rows <= n_cols; returns the column
    chosen for each row, minimising the total.  Verified against brute
    force in the self-test, because an assignment rule that is only
    approximately optimal would quietly undercut the comparison.
    """
    n, m = cost.shape
    INF = 1e18
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.full(m + 1, -1, int)
    way = np.zeros(m + 1, int)
    for i in range(n):
        p[m] = i
        j0 = m
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, -1
            for j in range(m):
                if used[j]:
                    continue
                cur = cost[i0, j] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == -1:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == m:
                break
    out = np.full(n, -1, int)
    for j in range(m):
        if p[j] != -1:
            out[p[j]] = j
    return out


def assign_hungarian(objs, out_n):
    """DETR.  Solve an optimal one-to-one match between objects and
    predictors, scoring each pairing by how well that predictor is
    ALREADY doing on that object.

    Dynamic: the cost matrix is built from the current predictions, so
    the assignment is re-solved every step and moves as the model
    learns.  Nothing tells predictor 19 to specialise in the top-left;
    it gets there by being repeatedly cheapest for top-left objects.
    """
    if not objs:
        return []
    pr = softmax_rows(out_n[:, 1:1 + N_CLS])
    sc = sigmoid(out_n[:, 0])
    dy, dx, _, _ = xy_decode(out_n[:, 1 + N_CLS + 0],
                             out_n[:, 1 + N_CLS + 1])
    sn = out_n[:, 1 + N_CLS + 2]
    idx = np.arange(N_PRED)
    py = (idx // GRID) * CELL + dy * CELL
    px = (idx % GRID) * CELL + dx * CELL
    ps = sn * S_HALF + S_MID
    cost = np.zeros((len(objs), N_PRED))
    for oi, (cls, cy, cx, s) in enumerate(objs):
        cost[oi] = (-np.log(pr[:, cls] + 1e-9)
                    - np.log(sc + 1e-9)
                    + 0.2 * (np.abs(py - cy) + np.abs(px - cx)
                             + np.abs(ps - s)))
    col = hungarian(cost)
    return [box_row(int(col[oi]), cls, cy, cx, s)
            for oi, (cls, cy, cx, s) in enumerate(objs)]


RULES = {"centre": assign_centre,
         "many": assign_many,
         "hungarian": assign_hungarian}
STATIC = {"centre": True, "many": True, "hungarian": False}


# -----------------------------------------------------------------------
# Decoding, suppression, and average precision.
# -----------------------------------------------------------------------

def decode(out_n):
    """One image's head output -> [(score, class, box, predictor), ...]."""
    score = sigmoid(out_n[:, 0])
    cls = softmax_rows(out_n[:, 1:1 + N_CLS]).argmax(1)
    dy, dx, _, _ = xy_decode(out_n[:, 1 + N_CLS + 0],
                             out_n[:, 1 + N_CLS + 1])
    sn = np.clip(out_n[:, 1 + N_CLS + 2] * S_HALF + S_MID,
                 SIZE_LO, SIZE_HI - 1)
    gy, gx = np.arange(N_PRED) // GRID, np.arange(N_PRED) % GRID
    cy = gy * CELL + dy * CELL
    cx = gx * CELL + dx * CELL
    return [(float(score[p]), int(cls[p]),
             np.array([cy[p] - sn[p], cx[p] - sn[p],
                       cy[p] + sn[p], cx[p] + sn[p]]), int(p))
            for p in range(N_PRED)]


def nms(dets, thresh=0.3):
    """Greedy suppression, class-agnostic because in this world real
    objects never overlap -- so any overlap IS a duplicate."""
    kept = []
    for d in sorted(dets, key=lambda t: -t[0]):
        if all(iou(d[2], k[2]) <= thresh for k in kept):
            kept.append(d)
    return kept


def average_precision(all_dets, all_gt, iou_t=0.5):
    """All-point-interpolated AP, pooled over the test set.  A
    detection counts only if its class is right AND its box overlaps
    the object by at least iou_t; each object can be claimed once."""
    flat = []
    n_gt = sum(len(g) for g in all_gt)
    for i, dets in enumerate(all_dets):
        for d in dets:
            flat.append((d[0], i, d[1], d[2]))
    if n_gt == 0 or not flat:
        return 0.0
    flat.sort(key=lambda t: -t[0])
    used = [set() for _ in all_gt]
    tp = np.zeros(len(flat))
    for j, (sc, i, c, box) in enumerate(flat):
        best, bj = 0.0, -1
        for gi, g in enumerate(all_gt[i]):
            if gi in used[i] or g[0] != c:
                continue
            v = iou(box, obj_to_box(g))
            if v > best:
                best, bj = v, gi
        if best >= iou_t:
            used[i].add(bj)
            tp[j] = 1.0
    ctp = np.cumsum(tp)
    rec = ctp / n_gt
    prec = ctp / np.arange(1, len(flat) + 1)
    mrec = np.concatenate(([0.0], rec))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


# -----------------------------------------------------------------------
# Training and evaluation.
# -----------------------------------------------------------------------

def fit(net, data, rule, steps, batch=32, lr=3e-3, seed=0, track=False):
    """A fixed STEP budget, as in Part 1: every rule gets the same
    number of updates, so a difference cannot be a difference in
    training length."""
    opt = Adam(net.params, lr)
    rng = np.random.default_rng(seed)
    fn = RULES[rule]
    pre = [fn(d[1], None) for d in data] if STATIC[rule] else None
    hist, moves, prev = [], 0, {}
    for _ in range(steps):
        idx = rng.integers(0, len(data), batch)
        X = np.stack([data[i][0] for i in idx])
        out, cache = net.forward(X)
        if pre is not None:
            A = [pre[i] for i in idx]
        else:
            A = [fn(data[i][1], out[j]) for j, i in enumerate(idx)]
            if track:
                for j, i in enumerate(idx):
                    key = tuple(sorted(r[0] for r in A[j]))
                    if i in prev and prev[i] != key:
                        moves += 1
                    prev[i] = key
        L, dout = loss_and_grad(out, A)
        opt.step(net.backward(dout, cache))
        hist.append(L)
    return hist, moves


def forward_all(net, data, chunk=125):
    outs = []
    for i in range(0, len(data), chunk):
        X = np.stack([d[0] for d in data[i:i + chunk]])
        outs.append(net.forward(X)[0])
    return np.concatenate(outs, 0)


def evaluate(net, data, nms_t=0.3, score_t=0.5):
    out = forward_all(net, data)
    raw = [decode(out[i]) for i in range(len(data))]
    gt = [d[1] for d in data]
    kept = [nms(d, nms_t) for d in raw]
    n_gt = sum(len(g) for g in gt)
    fired = sum(1 for d in raw for x in d if x[0] > score_t)
    fired_k = sum(1 for d in kept for x in d if x[0] > score_t)
    return (dict(ap=average_precision(kept, gt),
                 ap_off=average_precision(raw, gt),
                 dets=fired / n_gt, kept=fired_k / n_gt),
            raw, kept, gt)


def decompose(kept, gt, score_t=0.5, iou_t=0.5):
    """Attribute every ground-truth object to exactly one outcome.

    Possible only because the stage drew the objects: `owner` is the
    predictor the centre rule holds answerable, so a detection can be
    checked against who was supposed to produce it.
    """
    n = dict(detected=0, bad_box=0, bad_class=0, silent=0, total=0,
             by_owner=0)
    dup = 0
    for dets, objs in zip(kept, gt):
        fired = [d for d in dets if d[0] > score_t]
        for o in objs:
            n["total"] += 1
            gb = obj_to_box(o)
            best, bd = 0.0, None
            for d in fired:
                v = iou(d[2], gb)
                if v > best:
                    best, bd = v, d
            if bd is None:
                n["silent"] += 1
            elif best >= iou_t and bd[1] == o[0]:
                n["detected"] += 1
                if bd[3] == owner_cell(o):
                    n["by_owner"] += 1
            elif best >= iou_t:
                n["bad_class"] += 1
            else:
                n["bad_box"] += 1
        dup += max(0, len(fired) - len(objs))
    n["dup"] = dup / len(gt)
    return n


def score_quality(raw, gt, good_t=0.5, bad_t=0.1):
    """How well does objectness rank useful predictors above useless?

    For every predictor, find its best IoU against any real object.
    `good` means it already overlaps an object well enough to count;
    `bad` means it overlaps nothing.  The statistic is the probability
    that a random good predictor outscores a random bad one -- an AUC,
    so it is free of any score threshold.  It also reports how much
    IoU the single most confident predictor per image actually has.
    """
    good, bad, top_iou = [], [], []
    for dets, objs in zip(raw, gt):
        boxes = [obj_to_box(o) for o in objs]
        best = []
        for d in dets:
            v = max([iou(d[2], b) for b in boxes], default=0.0)
            best.append(v)
            if v >= good_t:
                good.append(d[0])
            elif v < bad_t:
                bad.append(d[0])
        k = int(np.argmax([d[0] for d in dets]))
        top_iou.append(best[k])
    good, bad = np.asarray(good), np.asarray(bad)
    if not len(good) or not len(bad):
        return float("nan"), float(np.mean(top_iou)), len(good)
    # AUC by rank, exact ties counted as half
    wins = (good[:, None] > bad[None, :]).sum()
    ties = (good[:, None] == bad[None, :]).sum()
    auc = (wins + 0.5 * ties) / (len(good) * len(bad))
    return float(auc), float(np.mean(top_iou)), len(good)


# =======================================================================
# Self-tests.  Each one guards a claim the article makes.
# =======================================================================

def selftest_stage():
    banner("SELF-TEST 1: the stage's ground truth is unambiguous")
    rng = np.random.default_rng(0)
    worst = 0.0
    counts = []
    for _ in range(2000):
        _, objs = scene(rng)
        counts.append(len(objs))
        for a, b in itertools.combinations(objs, 2):
            worst = max(worst, iou(obj_to_box(a), obj_to_box(b)))
    print("scenes 2000   objects/scene mean %.4f  min %d  max %d"
          % (np.mean(counts), min(counts), max(counts)))
    print("worst box-box IoU between two real objects: %.4f" % worst)
    assert worst == 0.0, "objects overlap; ground truth is ambiguous"
    # every object fits inside the canvas
    rng = np.random.default_rng(1)
    for _ in range(500):
        _, objs = scene(rng)
        for o in objs:
            b = obj_to_box(o)
            assert b[0] >= 0 and b[1] >= 0, "box off the top/left"
            assert b[2] < SIDE and b[3] < SIDE, "box off the bottom/right"
    print("every box lies inside the canvas: OK")
    print("ground truth is a clean list, so counting objects is "
          "well defined.")


def selftest_metric():
    banner("SELF-TEST 2: average precision is calibrated")
    data = make_scenes(60, 3)
    gt = [d[1] for d in data]
    perfect = [[(1.0, o[0], obj_to_box(o), 0) for o in g] for g in gt]
    ap_perfect = average_precision(perfect, gt)
    print("AP of the ground truth scored against itself: %.4f"
          % ap_perfect)
    assert abs(ap_perfect - 1.0) < 1e-12, "perfect detections must be 1"

    rng = np.random.default_rng(0)
    rand = []
    for g in gt:
        row = []
        for _ in range(3):
            s = int(rng.integers(SIZE_LO, SIZE_HI))
            cy = int(rng.integers(s, SIDE - s))
            cx = int(rng.integers(s, SIDE - s))
            row.append((float(rng.random()), int(rng.integers(0, N_CLS)),
                        obj_to_box((0, cy, cx, s)), 0))
        rand.append(row)
    print("AP of random boxes with random labels:            %.4f"
          % average_precision(rand, gt))

    shifted = [[(1.0, o[0], obj_to_box(o) + np.array([3., 3., 3., 3.]), 0)
                for o in g] for g in gt]
    print("AP of correct labels on boxes shifted 3 px:       %.4f"
          % average_precision(shifted, gt))
    wrong = [[(1.0, (o[0] + 1) % N_CLS, obj_to_box(o), 0) for o in g]
             for g in gt]
    print("AP of exact boxes with every label rotated:       %.4f"
          % average_precision(wrong, gt))
    assert average_precision(wrong, gt) == 0.0

    # The ranking statistic used in Measurement 7, on rankings whose
    # answer is known by construction.
    one = [[(0, 8, 8, 3)]]
    hit = obj_to_box((0, 8, 8, 3))
    miss = np.array([SIDE - 4.0, SIDE - 4.0, SIDE - 2.0, SIDE - 2.0])
    cases = (("confident where it is right", 0.9, 0.1, 1.0),
             ("confident where it is wrong", 0.1, 0.9, 0.0),
             ("no opinion either way", 0.5, 0.5, 0.5))
    print()
    for label, sg, sb, want in cases:
        auc, _, _ = score_quality([[(sg, 0, hit, 0), (sb, 0, miss, 1)]],
                                  one)
        print("ranking AUC, %-30s %.4f" % (label + ":", auc))
        assert abs(auc - want) < 1e-12, "AUC miscomputed"
    print()
    print("the metric rewards nothing it should not.")


def selftest_hungarian():
    banner("SELF-TEST 3: the matcher really is optimal")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(400):
        n = int(rng.integers(1, 5))
        m = int(rng.integers(n, 10))
        C = rng.normal(0, 1, (n, m))
        got = C[np.arange(n), hungarian(C)].sum()
        best = min(sum(C[i, c] for i, c in enumerate(p))
                   for p in itertools.permutations(range(m), n))
        worst = max(worst, abs(got - best))
    print("shortest-augmenting-path vs brute force over 400 random")
    print("rectangular cost matrices: worst gap %.3e" % worst)
    assert worst < 1e-12, "the matcher is not optimal"
    print("so `hungarian` below is genuinely the best one-to-one match,")
    print("not a greedy approximation to it.")


def selftest_gradients():
    banner("SELF-TEST 4: the hand-derived gradients are right")
    data = make_scenes(6, 100)
    X = np.stack([d[0] for d in data])
    A = [assign_centre(d[1]) for d in data]
    net = Detector(c1=6, c2=10, rng=np.random.default_rng(1))

    # WHERE the audit is taken matters here, and it is worth a word.
    # A central difference assumes the function is smooth between the
    # two probes.  ReLU is not smooth at 0, and this stage parks an
    # enormous number of units exactly there: the canvas is mostly
    # empty, so a window containing no ink has a pre-activation of
    # precisely zero.
    clean = np.stack([scene(np.random.default_rng(100 + i))[0]
                      for i in range(6)])
    cc, _, _ = im2col(clean[:, None], 3, pad=1)
    print("clean canvas that is empty:               %.1f%%"
          % (100.0 * (clean == 0).mean()))
    print("pre-activations exactly 0 at zero bias:   %.1f%%"
          % (100.0 * ((cc @ net.W1) == 0).mean()))
    print()
    print("A bias shifts every pre-activation in its channel at once,")
    print("so any unit sitting on the kink flips state and poisons the")
    print("difference -- which reports a disagreement the backward pass")
    print("is not responsible for.  The audit therefore runs at a")
    print("GENERIC point, with the biases offset clear of zero.")

    r = np.random.default_rng(42)
    net.b1[:] = r.normal(0.5, 0.05, net.b1.shape)
    net.b2[:] = r.normal(0.5, 0.05, net.b2.shape)
    net.b3[:] = r.normal(0.0, 0.10, net.b3.shape)

    out, cache = net.forward(X)
    L, dout = loss_and_grad(out, A)
    g = net.backward(dout, cache)
    names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    eps = 1e-4
    cc2, _, _ = im2col(X[:, None], 3, pad=1)
    near = int((np.abs(cc2 @ net.W1 + net.b1) < eps).sum())
    print("units within eps of the kink at that point: %d" % near)
    assert near == 0, "audit point still sits on the kink"
    print()
    worst, n_checked = 0.0, 0
    print()
    for pi, (nm, p) in enumerate(zip(names, net.params)):
        flat, gf = p.reshape(-1), g[pi].reshape(-1)
        idxs = (range(p.size) if p.size <= 24
                else np.random.default_rng(pi).integers(0, p.size, 24))
        errs = []
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            Lp, _ = loss_and_grad(net.forward(X)[0], A)
            flat[i] = old - eps
            Lm, _ = loss_and_grad(net.forward(X)[0], A)
            flat[i] = old
            num = (Lp - Lm) / (2 * eps)
            den = max(1e-9, abs(num) + abs(gf[i]))
            errs.append(abs(num - gf[i]) / den)
        n_checked += len(errs)
        worst = max(worst, max(errs))
        print("  %-3s %6d params   worst relative error %.3e"
              % (nm, p.size, max(errs)))
    print("\nworst over %d sampled gradients: %.3e" % (n_checked, worst))
    assert worst < 1e-6, "gradients disagree with central differences"
    print("loss at the audit point: %.6f" % L)


# =======================================================================
# Measurements.
# =======================================================================

def measure_representation():
    banner("MEASUREMENT 1: what a fixed-size output can even represent")
    print("A classifier emits ONE answer.  Before asking whether it")
    print("would be accurate, ask whether it could be right at all.")
    print()
    rng = np.random.default_rng(5)
    grids = (1, 2, 4, 8, 16)
    lost = {g: 0 for g in grids}
    total = 0
    for _ in range(4000):
        _, objs = scene(rng)
        total += len(objs)
        for g in grids:
            cell = SIDE / g
            seen = set()
            for o in objs:
                key = (int(o[1] // cell), int(o[2] // cell))
                if key in seen:
                    lost[g] += 1
                seen.add(key)
    print("4000 scenes, %d objects" % total)
    print("  grid    predictors   cell    objects it cannot hold")
    rows = []
    for g in grids:
        r = lost[g] / total
        rows.append((g, g * g, SIDE / g, r))
        print("  %2dx%-2d      %5d      %4.1f px        %.4f"
              % (g, g, g * g, SIDE / g, r))
    print()
    print("One predictor -- the Part 1 architecture -- loses %.4f of"
          % (lost[1] / total))
    print("every object in this world before training begins.")
    print()
    print("The 8x8 grid loses nothing, so the dropped-object failure")
    print("is NOT what separates the rules below.  It costs little to")
    print("check when it would bite:")
    print()
    print("  objects   spacing   4x4 grid   8x8 grid   16x16 grid")
    dense = []
    for n_obj, gap in ((4, 1), (8, -1), (12, -2)):
        out = []
        for g in (4, 8, 16):
            rng = np.random.default_rng(5)
            cell = SIDE / g
            tot = los = 0
            for _ in range(1200):
                _, objs = scene(rng, n_obj=n_obj, min_gap=gap)
                tot += len(objs)
                seen = set()
                for o in objs:
                    key = (int(o[1] // cell), int(o[2] // cell))
                    if key in seen:
                        los += 1
                    seen.add(key)
            out.append(los / tot)
        dense.append((n_obj, gap, out))
        print("  %5d     %5d      %.4f     %.4f     %.4f"
              % (n_obj, gap, out[0], out[1], out[2]))
    print()
    print("Grid resolution is a bet on how crowded the world is.  Make")
    print("the wrong bet and the loss cannot see the objects you drop.")
    return rows, dense


NEG_SHARE = float("nan")


def measure_imbalance(train, test):
    global NEG_SHARE
    banner("MEASUREMENT 2: most predictors must learn to say nothing")
    pos = np.mean([len(d[1]) for d in train])
    NEG_SHARE = 100.0 * (1.0 - pos / N_PRED)
    print("predictors per image      %d" % N_PRED)
    print("objects per image         %.4f" % pos)
    print("negatives                 %.2f%%  (a ratio of %.1f to 1)"
          % (100 * (1 - pos / N_PRED), (N_PRED - pos) / pos))
    print()
    print("YOLO's answer was to down-weight the no-object term.  Does")
    print("it help here?  Three weights, two seeds, equal budgets.")
    print()
    global L_NOOBJ
    print("Measured at TWO budgets, because the answer depends on")
    print("where you stop: each run is evaluated part-way and at the")
    print("end, so a weight that merely learns faster cannot pass for")
    print("a weight that ends up better.")
    print()
    rows = []
    for w in (1.0, 0.5, 0.2):
        L_NOOBJ = w
        early, late, dets = [], [], []
        for s in (0, 1):
            net = Detector(rng=np.random.default_rng(s + 1))
            fit(net, train, "centre", AUX_STEPS, seed=s)
            early.append(evaluate(net, test)[0]["ap"])
            # Resume the SAME net to the full budget.  Note `fit` builds
            # a fresh Adam, so the moments restart at the checkpoint:
            # the two columns are comparable to EACH OTHER (all three
            # weights are treated identically) but not to Measurement 4,
            # which trains straight through.
            fit(net, train, "centre", STEPS - AUX_STEPS, seed=s + 100)
            r, _, _, _ = evaluate(net, test)
            late.append(r["ap"])
            dets.append(r["dets"])
        rows.append((w, np.mean(early), np.mean(late), np.mean(dets),
                     np.std(early), np.std(late)))
        print("  weight %.1f   AP at %d steps %s   at %d steps %s"
              % (w, AUX_STEPS, fmt_pm(early), STEPS, fmt_pm(late)))
    L_NOOBJ = 1.0
    print()
    best = max(rows, key=lambda r: r[2])
    base = [r for r in rows if r[0] == 1.0][0]
    gap = best[2] - base[2]
    spread = max(best[5], base[5])
    print("  best at %d steps: weight %.1f (%.4f)"
          % (STEPS, best[0], best[2]))
    print("  no down-weighting:  weight 1.0 (%.4f)" % base[2])
    print("  gap %.4f   worst seed spread of the two %.4f"
          % (gap, spread))
    print()
    if best[0] == 1.0:
        print("Down-weighting the negatives does not help at all here,")
        print("even though it makes the detector fire more often.")
    elif gap <= spread:
        print("The best weight beats no weighting by %.4f, which is"
              % gap)
        print("SMALLER than the %.4f spread between its own two seeds."
              % spread)
        print("At two seeds that is not a result.  The honest reading")
        print("is that this stage cannot tell the difference, and")
        print("reporting %.1f as the better setting would be reporting"
              % best[0])
        print("noise with a decimal point on it.")
    else:
        print("Down-weighting to %.1f beats no weighting by %.4f,"
              % (best[0], gap))
        print("clear of the %.4f seed spread." % spread)
    print()
    print("Either way the trick is aimed at a problem of a different")
    print("size: this stage has %.0f negatives per positive, while an"
          % ((N_PRED - pos) / pos))
    print("anchor-based detector carries ~100k anchors for a handful of")
    print("objects, nearer 10,000 to 1.")
    return rows


def clipped_share(rule, n=1500, seed=5):
    """Share of a rule's positive predictors whose target centre lies
    outside their own cell.  Requires BOUND_BOX off, so the targets
    come back unclipped."""
    rng = np.random.default_rng(seed)
    fn = RULES[rule]
    zero = np.zeros((N_PRED, N_OUT))
    tot = out_of = 0
    worst = 0.0
    for _ in range(n):
        _, objs = scene(rng)
        for (p, c, dy, dx, sn) in fn(objs, zero):
            tot += 1
            if not (0.0 <= dy <= 1.0 and 0.0 <= dx <= 1.0):
                out_of += 1
            worst = max(worst, abs(dy - 0.5), abs(dx - 0.5))
    return out_of / max(1, tot), tot, worst


def measure_coupling(train, test):
    global BOUND_BOX
    banner("MEASUREMENT 3: what confining a box to its cell is worth")
    print("YOLOv1 ran the centre offset through a sigmoid, confining")
    print("each predictor's box to its own %d px cell.  Ask first how"
          % CELL)
    print("many of each rule's targets that puts out of reach.")
    print()
    assert not BOUND_BOX, "measure the targets unclipped"
    share = {}
    for rule in ("centre", "many", "hungarian"):
        share[rule], tot, worst = clipped_share(rule)
        print("  %-11s %6d positives, %.4f outside their own cell,"
              "  furthest %.2f cells" % (rule, tot, share[rule], worst))
    print()
    free = [r for r, v in share.items() if v < 0.01]
    hit = sorted([r for r, v in share.items() if v >= 0.01],
                 key=lambda r: -share[r])
    print("The bound is free for %s -- its target is inside the cell"
          % ", ".join(free))
    print("by definition.  For %s it is not: those" % ", ".join(hit))
    print("positives are trained toward a box they cannot emit, and")
    print("then scored on the box they do emit.  It catches both of")
    print("the rules that let a predictor own a DISTANT object -- the")
    print("one-to-many rule by construction, and the optimal match")
    print("because nothing obliges it to pick the cell the centre")
    print("happens to fall in.")
    print()
    print("That looks like a reason to expect the bound to hurt them.")
    print("Train every rule both ways and see: %d steps, two seeds."
          % AUX_STEPS)
    print()
    out = {}
    for bound in (True, False):
        BOUND_BOX = bound
        for rule in ("centre", "many", "hungarian"):
            aps = []
            for s in (0, 1):
                net = Detector(rng=np.random.default_rng(s + 1))
                fit(net, train, rule, AUX_STEPS, seed=s)
                aps.append(evaluate(net, test)[0]["ap"])
            out[(bound, rule)] = aps
            print("  box in cell %-5s  %-11s AP %s"
                  % (str(bound), rule, fmt_pm(aps)))
            sys.stdout.flush()
    BOUND_BOX = False
    print()
    worth = {}
    for rule in ("centre", "many", "hungarian"):
        worth[rule] = (np.mean(out[(True, rule)])
                       - np.mean(out[(False, rule)]))
        print("  the cell bound is worth %+.4f to %-10s "
              "(it clips %.4f of that rule's targets)"
              % (worth[rule], rule, share[rule]))
    print()
    helped_most = max(worth, key=worth.get)
    clipped_most = max(share, key=share.get)
    if min(worth.values()) > 0.0:
        print("The prediction above was WRONG, and the measurement is")
        print("worth more than the prediction.  Confining a box to its")
        print("own cell helps EVERY rule -- including the rules whose")
        print("targets it mangles.")
        print()
        if helped_most == clipped_most:
            print("It helps %s most (%+.4f), which is precisely the rule"
                  % (helped_most, worth[helped_most]))
            print("whose targets it clips hardest (%.4f).  Reaching a"
                  % share[clipped_most])
            print("CLIPPED target a predictor can actually emit beats")
            print("reaching a CORRECT one it has to find by regressing a")
            print("long way.  The bound is a prior, and it buys more")
            print("than target fidelity costs.")
        else:
            print("It helps %s most (%+.4f) while clipping %s hardest,"
                  % (helped_most, worth[helped_most], clipped_most))
            print("so the benefit is not simply the clipping reversed.")
    else:
        print("The bound helps some rules and hurts others, so the")
        print("decoder cannot be chosen independently of the rule.")
    print()
    rank_b = sorted(("centre", "many", "hungarian"),
                    key=lambda r: -np.mean(out[(True, r)]))
    rank_u = sorted(("centre", "many", "hungarian"),
                    key=lambda r: -np.mean(out[(False, r)]))
    print("  ranking with the bound:    " + " > ".join(rank_b))
    print("  ranking without the bound: " + " > ".join(rank_u))
    if rank_b == rank_u:
        print()
        print("The ranking is the same either way, which is the only")
        print("thing the comparison below needs from this section: its")
        print("conclusion does not rest on the decoder.")
    else:
        print()
        print("The ranking DIFFERS, so the comparison below has to be")
        print("read as a statement about one decoder, not about")
        print("assignment in general.")
    print()
    print("Everything below uses the unbounded offset, for every rule,")
    print("so the comparison is about assignment alone.")
    return out


def measure_rules(train, test):
    banner("MEASUREMENT 4: the assignment rule IS the design decision")
    print("Same backbone, same head, same loss terms, same %d steps,"
          % STEPS)
    print("same parameter count.  Only the rule changes.")
    print()
    res, saved = {}, {}
    for rule in ("centre", "many", "hungarian"):
        acc = []
        for s in SEEDS:
            t0 = time.time()
            net = Detector(rng=np.random.default_rng(s + 1))
            _, moves = fit(net, train, rule, STEPS, seed=s, track=True)
            r, raw, kept, gt = evaluate(net, test)
            if s == SEEDS[0]:
                saved[rule] = raw          # reused by the NMS sweep
            d = decompose(kept, gt)
            tot = d["total"]
            acc.append(dict(
                ap=r["ap"], ap_off=r["ap_off"], dets=r["dets"],
                kept=r["kept"], dup=d["dup"], moves=moves,
                detected=d["detected"] / tot, silent=d["silent"] / tot,
                bad_box=d["bad_box"] / tot,
                bad_class=d["bad_class"] / tot,
                by_owner=d["by_owner"] / max(1, d["detected"]),
                sec=time.time() - t0))
            print("  %-10s seed %d   AP %.4f   positives/object %5.2f"
                  "   [%3.0fs]"
                  % (rule, s, r["ap"],
                     len(RULES[rule](train[0][1], np.zeros(
                         (N_PRED, N_OUT)))) / max(1, len(train[0][1])),
                     time.time() - t0))
            sys.stdout.flush()
        res[rule] = acc

    def col(rule, key):
        return np.array([a[key] for a in res[rule]])

    print()
    print("%-11s %-17s %-17s %-11s" %
          ("rule", "AP (NMS on)", "AP (NMS off)", "dup/image"))
    for rule in res:
        print("%-11s %-17s %-17s %-11s"
              % (rule, fmt_pm(col(rule, "ap")),
                 fmt_pm(col(rule, "ap_off")),
                 fmt_pm(col(rule, "dup"))))
    print()
    print("What NMS is worth, per rule (AP with minus AP without):")
    gain = {}
    for rule in res:
        g = col(rule, "ap") - col(rule, "ap_off")
        gain[rule] = g.mean()
        print("  %-11s %+.4f +-%.4f   (%.2f duplicates/image)"
              % (rule, g.mean(), g.std(), col(rule, "dup").mean()))
    print()
    helped = sorted([r for r, g in gain.items() if g > 0.005],
                    key=lambda r: -gain[r])
    hurt = sorted([r for r, g in gain.items() if g < -0.005],
                  key=lambda r: gain[r])
    if helped and hurt:
        print("Suppression is not a universal good: it repays %s"
              % ", ".join(helped))
        print("and costs %s.  The rules it repays are the rules that"
              % ", ".join(hurt))
        print("manufacture duplicates in the first place.")
    elif helped:
        print("Suppression repays %s and is neutral elsewhere."
              % ", ".join(helped))
    elif hurt:
        print("At this budget suppression costs every rule it changes,")
        print("including the one that manufactures duplicates -- so the")
        print("duplicates are not yet the binding constraint.")
    else:
        print("Suppression changes no rule by more than 0.005 here.")
    print()
    print("The Hungarian assignment is re-solved every step, and moves:")
    for rule in res:
        m = col(rule, "moves")
        print("  %-11s %8.0f changes of owner during training"
              % (rule, m.mean()))
    return res, saved


def measure_errors(res):
    banner("MEASUREMENT 5: attributing every miss")
    print("The stage drew the objects, so the predictor that was")
    print("supposed to own each one is known.  Every object therefore")
    print("lands in exactly one bucket.")
    print()
    print("%-11s %-9s %-9s %-9s %-9s %-9s" %
          ("rule", "found", "silent", "bad box", "bad cls", "by owner"))
    for rule, acc in res.items():
        def m(k):
            return np.mean([a[k] for a in acc])
        print("%-11s %-9.4f %-9.4f %-9.4f %-9.4f %-9.4f"
              % (rule, m("detected"), m("silent"), m("bad_box"),
                 m("bad_class"), m("by_owner")))
        tot = m("detected") + m("silent") + m("bad_box") + m("bad_class")
        assert abs(tot - 1.0) < 1e-9, "buckets do not partition"
    print()
    print("Columns 2-5 partition the objects exactly (they sum to 1).")
    print("'by owner' is the share of successful detections that came")
    print("from the predictor the centre rule holds answerable.")
    print()
    for rule, acc in res.items():
        buckets = {k: np.mean([a[k] for a in acc])
                   for k in ("silent", "bad_box", "bad_class")}
        worst = max(buckets, key=buckets.get)
        label = {"silent": "nothing fired at all",
                 "bad_box": "the box was too far off",
                 "bad_class": "the label was wrong"}[worst]
        print("  %-11s loses most objects because %s (%.4f)"
              % (rule, label, buckets[worst]))
    print()
    if all(max(("silent", "bad_box", "bad_class"),
               key=lambda k: np.mean([a[k] for a in acc])) == "silent"
           for acc in res.values()):
        print("Every rule fails the same way: by SILENCE rather than by")
        print("bad geometry.  Once a predictor speaks it usually gets")
        print("the box and the label right; the hard part is getting it")
        print("to speak at all, against a loss in which staying quiet")
        print("is the correct answer %.1f%% of the time." % NEG_SHARE)
    else:
        print("The rules do not fail the same way, which is the point:")
        print("the assignment rule chooses which error you get.")


def measure_nms(saved, test):
    banner("MEASUREMENT 6: the suppression threshold")
    print("The seed-%d detector from each rule above, re-scored; no"
          % SEEDS[0])
    print("retraining, only the threshold moves.  IoU 1.0 suppresses")
    print("nothing at all.")
    print()
    thresholds = (0.0, 0.1, 0.3, 0.5, 0.7, 1.0)
    gt = [d[1] for d in test]
    print("rule        " + "".join("%8.1f" % t for t in thresholds))
    rows = {}
    for rule, raw in saved.items():
        aps = [average_precision([nms(d, t) for d in raw], gt)
               for t in thresholds]
        rows[rule] = aps
        print("%-11s " % rule + "".join("  %.4f" % a for a in aps))
        sys.stdout.flush()
    print()
    best = {r: thresholds[int(np.argmax(v))] for r, v in rows.items()}
    for r, b in best.items():
        print("  %-11s best at IoU %.1f  (%.4f, against %.4f unsuppressed)"
              % (r, b, max(rows[r]), rows[r][-1]))
    print()
    wants = [r for r, b in best.items() if b < 1.0]
    if wants and len(wants) < len(best):
        print("%s want suppression; the rest are best left alone."
              % ", ".join(wants))
        print("NMS is not post-processing that happens to help: it is")
        print("the repair bill for an assignment rule that trained")
        print("several predictors to report the same object.")
    elif not wants:
        print("No rule here prefers any suppression at all -- every one")
        print("peaks with it switched off.")
    else:
        print("Every rule prefers some suppression at this budget.")
    return thresholds, rows


def measure_scores(saved, test):
    banner("MEASUREMENT 7: is the gap confidence, or geometry?")
    print("The one-to-many rule finds nearly as many objects as the")
    print("centre rule and scores far worse, so what it loses is")
    print("precision, not recall.  Two things could cause that.  Its")
    print("confidence might fail to separate the predictors that")
    print("overlap an object from the ones that overlap nothing -- a")
    print("ranking failure.  Or its confident predictors might simply")
    print("hold worse boxes -- a geometry failure.  The two columns")
    print("below separate those.")
    print()
    print("%-11s %-12s %-14s %-10s" %
          ("rule", "ranking AUC", "IoU of the", "usable"))
    print("%-11s %-12s %-14s %-10s" %
          ("", "", "most confident", "predictors"))
    gt = [d[1] for d in test]
    rows = {}
    for rule, raw in saved.items():
        auc, top, n_good = score_quality(raw, gt)
        rows[rule] = (auc, top, n_good)
        print("%-11s %-12.4f %-14.4f %-10d" % (rule, auc, top, n_good))
    print()
    print("AUC 1.0 would mean every predictor overlapping an object")
    print("outscores every predictor overlapping nothing; 0.5 is a")
    print("coin toss.  The second column is the overlap actually held")
    print("by each image's single most confident predictor -- and an")
    print("AP hit needs 0.5000 of it.")
    print()
    aucs = {r: v[0] for r, v in rows.items()}
    tops = {r: v[1] for r, v in rows.items()}
    if min(aucs.values()) > 0.90:
        print("Every rule ranks near-an-object above near-nothing well")
        print("(worst %.4f, by %s).  So the AP gap is NOT a confidence"
              % (min(aucs.values()), min(aucs, key=aucs.get)))
        print("failure -- and `many` ranks BEST of the three (%.4f),"
              % max(aucs.values()))
        print("which refutes the obvious guess that its duplicates are")
        print("confidently wrong.")
        print()
        lo = min(tops, key=tops.get)
        hi = max(tops, key=tops.get)
        print("The geometry column does separate them.  %s's most"
              % lo)
        print("confident predictor averages %.4f overlap against %s's"
              % (tops[lo], hi))
        print("%.4f.  An AP hit needs 0.5000, so %s is sitting just"
              % (tops[hi], lo))
        print("above the line: its confident boxes are good enough to")
        print("be plausible and not good enough to count, which is")
        print("what turns duplicates into false positives.")
        print()
        print("Train several predictors on one object and none of them")
        print("specialises in it.  That is the cost one-to-many")
        print("assignment pays for never going silent, and it is what")
        print("`centerness` and IoU-aware scoring were later invented")
        print("to recover.")
    else:
        worst = min(aucs, key=aucs.get)
        print("%s ranks its own predictors poorly (%.4f), so for that"
              % (worst, aucs[worst]))
        print("rule the gap is a confidence failure rather than a")
        print("geometric one.")
    return rows


# =======================================================================
# Configuration, frozen.
# =======================================================================

STEPS = 1000            # the main comparison: every rule gets these
AUX_STEPS = 600         # the two side studies, where only a gap matters
SEEDS = (0, 1, 2)
N_TRAIN, N_TEST = 800, 250
SEED_TRAIN, SEED_TEST = 10, 7


def main():
    t0 = time.time()
    banner("OBJECT DETECTION: WHERE THE TARGETS COME FROM")
    print("canvas %dx%d   classes %d   objects 1-%d"
          % (SIDE, SIDE, N_CLS, MAX_OBJ))
    print("grid %dx%d -> %d predictors, %d px cells, %d outputs each"
          % (GRID, GRID, N_PRED, CELL, N_OUT))
    net = Detector()
    print("detector parameters: %d" % net.n_params())
    print("train scenes %d   test scenes %d   steps %d   seeds %s"
          % (N_TRAIN, N_TEST, STEPS, list(SEEDS)))

    selftest_stage()
    selftest_metric()
    selftest_hungarian()
    selftest_gradients()

    train = make_scenes(N_TRAIN, SEED_TRAIN)
    test = make_scenes(N_TEST, SEED_TEST)

    measure_representation()
    measure_imbalance(train, test)
    measure_coupling(train, test)
    res, saved = measure_rules(train, test)
    measure_errors(res)
    measure_nms(saved, test)
    measure_scores(saved, test)

    banner("DONE")
    print("total runtime %.1f minutes" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
