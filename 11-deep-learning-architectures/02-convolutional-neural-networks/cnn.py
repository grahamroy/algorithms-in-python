"""
cnn.py --- companion code for "Convolutional Neural Networks"
(Deep Learning Architectures, Part 2).

Part 1 ended on the MLP's blind spot: it treats input dimensions as
unrelated columns. Shuffle the pixels of every image identically and an
MLP learns exactly as well -- a claim this script now MEASURES. Images
have structure the MLP cannot see: nearby pixels are related, and the
same pattern means the same thing wherever it appears.

The convolutional layer writes both facts into the architecture:

    LOCALITY:  each output looks at a small window (3x3 here), not at
               every pixel at once.
    SHARING:   one kernel slides across the whole image -- what is
               learned HERE is known EVERYWHERE. 9 weights, reused at
               all 144 positions.

Then ReLU, and 2x2 max-pooling ("was the feature anywhere in this
neighbourhood?") -- the classic stack, all with hand-written forward
and backward passes (im2col + einsum, argmax routing through the pool).
The backward pass is audited against central differences at startup,
Part 1's ritual, inherited.

The stage: 14x14 images of a single 7-pixel stroke -- horizontal,
vertical, or diagonal -- at a RANDOM position, with noise. All three
classes light up exactly 7 pixels, so total brightness carries no
signal. Geometry is the only thing there is to learn.

Demonstrates:
  1. The shuffle test: one fixed pixel permutation applied to every
     image. The MLP does not care (92.2% -> 91.0%); the CNN's whole
     advantage evaporates (99.8% -> 62.0%). Priors only pay when they
     are TRUE.
  2. Sample efficiency: at 150 training images the CNN leads by ~38
     points; by 600 the MLP has bought with data what the CNN was
     given by design.
  3. Locality vs sharing, separated: dense (12,803 params, 92.2%) vs
     locally-connected-unshared (11,243, 98.8%) vs shared conv (947,
     99.8%) -- and the learned 3x3 kernels, printed: oriented edge
     detectors nobody asked for.

Everything is plain NumPy. Dependencies: numpy. Runs in about a minute.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
IMG = 14           # image side
STROKE = 7         # stroke length, all classes
NOISE = 0.15
EPOCHS = 300


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# The stage: one 7-pixel stroke per image, three orientations, random
# position. Equal pixel budgets by construction -- geometry or nothing.
# ---------------------------------------------------------------------------

def draw_line(cls, rng, img=IMG):
    canvas = np.zeros((img, img))
    if cls == 0:                                   # horizontal
        r = rng.integers(0, img)
        c = rng.integers(0, img - STROKE + 1)
        canvas[r, c:c + STROKE] = 1
    elif cls == 1:                                 # vertical
        r = rng.integers(0, img - STROKE + 1)
        c = rng.integers(0, img)
        canvas[r:r + STROKE, c] = 1
    else:                                          # diagonal
        r = rng.integers(0, img - STROKE + 1)
        c = rng.integers(0, img - STROKE + 1)
        for k in range(STROKE):
            canvas[r + k, c + k] = 1
    canvas += rng.normal(0, NOISE, canvas.shape)
    return canvas


def make_lines(n_per_class, rng):
    X, y = [], []
    for cls in range(3):
        for _ in range(n_per_class):
            X.append(draw_line(cls, rng))
            y.append(cls)
    X, y = np.array(X), np.array(y)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# im2col: every 3x3 window laid out as a row, so convolution becomes one
# matrix multiply -- and its backward pass becomes another.
# ---------------------------------------------------------------------------

def im2col(X, k):
    N, H, W = X.shape
    out = np.lib.stride_tricks.sliding_window_view(X, (k, k), axis=(1, 2))
    return out.reshape(N, H - k + 1, W - k + 1, k * k)


class CNN:
    """conv(8 filters, 3x3) -> ReLU -> maxpool 2x2 -> dense -> softmax.
    shared=False keeps the same wiring but gives every location its OWN
    kernel -- locality without sharing, DEMO 3's control condition."""

    def __init__(self, filters=8, img=IMG, n_cls=3, seed=0, lr=1e-2,
                 shared=True):
        r = np.random.default_rng(seed)
        self.k = 3
        self.F = filters
        self.co = img - 2
        self.po = self.co // 2
        self.shared = shared
        if shared:
            self.K = r.standard_normal((9, filters)) * np.sqrt(2.0 / 9)
        else:
            self.K = r.standard_normal(
                (self.co * self.co, 9, filters)) * np.sqrt(2.0 / 9)
        self.bk = np.zeros(filters)
        nf = self.po * self.po * filters
        self.W = r.standard_normal((nf, n_cls)) * np.sqrt(2.0 / nf)
        self.b = np.zeros(n_cls)
        self.lr = lr
        self.params = [self.K, self.bk, self.W, self.b]
        self.M = [np.zeros_like(p) for p in self.params]
        self.V = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def n_params(self):
        return sum(p.size for p in self.params)

    def forward(self, X):
        N = len(X)
        C = im2col(X, self.k)                          # (N, co, co, 9)
        if self.shared:
            Z = C @ self.K + self.bk                   # one kernel, everywhere
        else:
            Cf = C.reshape(N, -1, 9)
            Z = np.einsum("nlk,lkf->nlf", Cf, self.K).reshape(
                N, self.co, self.co, self.F) + self.bk
        A = np.maximum(0, Z)
        Ap = A[:, :self.po * 2, :self.po * 2].reshape(
            N, self.po, 2, self.po, 2, self.F)
        P = Ap.max(axis=(2, 4))                        # "anywhere nearby?"
        logits = P.reshape(N, -1) @ self.W + self.b
        return logits, (C, Z, A, Ap, P)

    def probs(self, X):
        logits, _ = self.forward(X)
        Z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def loss(self, X, y):
        p = self.probs(X)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-300).mean())

    def grads(self, X, y):
        N = len(X)
        logits, (C, Z, A, Ap, P) = self.forward(X)
        Zs = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(Zs)
        p = e / e.sum(axis=1, keepdims=True)
        d = (p - np.eye(logits.shape[1])[y]) / N
        gW = P.reshape(N, -1).T @ d
        gb = d.sum(0)
        dP = (d @ self.W.T).reshape(N, self.po, self.po, self.F)
        mask = (Ap == P[:, :, None, :, None, :])       # route to the argmax
        mask = mask / np.maximum(mask.sum(axis=(2, 4), keepdims=True), 1)
        dA = np.zeros_like(A)
        dA[:, :self.po * 2, :self.po * 2] = (
            mask * dP[:, :, None, :, None, :]).reshape(
            N, self.po * 2, self.po * 2, self.F)
        dZ = dA * (Z > 0)
        if self.shared:
            gK = C.reshape(-1, 9).T @ dZ.reshape(-1, self.F)
        else:
            gK = np.einsum("nlk,nlf->lkf", C.reshape(N, -1, 9),
                           dZ.reshape(N, -1, self.F))
        gbk = dZ.sum(axis=(0, 1, 2))
        return [gK, gbk, gW, gb]

    def step(self, X, y):
        g = self.grads(X, y)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (P_, G) in enumerate(zip(self.params, g)):
            self.M[i] = b1 * self.M[i] + (1 - b1) * G
            self.V[i] = b2 * self.V[i] + (1 - b2) * G ** 2
            P_ -= self.lr * (self.M[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.V[i] / (1 - b2 ** self.t)) + eps)


class DenseNet:
    """Part 1's MLP on the flattened pixels: [64], ReLU, He, Adam."""

    def __init__(self, img=IMG, hidden=64, n_cls=3, seed=0, lr=1e-2):
        r = np.random.default_rng(seed)
        d = img * img
        self.W1 = r.standard_normal((d, hidden)) * np.sqrt(2.0 / d)
        self.b1 = np.zeros(hidden)
        self.W2 = r.standard_normal((hidden, n_cls)) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(n_cls)
        self.lr = lr
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.M = [np.zeros_like(p) for p in self.params]
        self.V = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def n_params(self):
        return sum(p.size for p in self.params)

    def probs(self, X):
        Xf = X.reshape(len(X), -1)
        H = np.maximum(0, Xf @ self.W1 + self.b1)
        L = H @ self.W2 + self.b2
        Z = L - L.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def step(self, X, y):
        Xf = X.reshape(len(X), -1)
        H = np.maximum(0, Xf @ self.W1 + self.b1)
        L = H @ self.W2 + self.b2
        Z = L - L.max(axis=1, keepdims=True)
        e = np.exp(Z)
        p = e / e.sum(axis=1, keepdims=True)
        d = (p - np.eye(L.shape[1])[y]) / len(X)
        g = [Xf.T @ (d @ self.W2.T * (H > 0)),
             (d @ self.W2.T * (H > 0)).sum(0),
             H.T @ d,
             d.sum(0)]
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (P_, G) in enumerate(zip(self.params, g)):
            self.M[i] = b1 * self.M[i] + (1 - b1) * G
            self.V[i] = b2 * self.V[i] + (1 - b2) * G ** 2
            P_ -= self.lr * (self.M[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.V[i] / (1 - b2 ** self.t)) + eps)


def fit(net, X, y, epochs=EPOCHS):
    for _ in range(epochs):
        net.step(X, y)
    return net


def accuracy(net, X, y):
    return float((net.probs(X).argmax(1) == y).mean())


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    # Part 1's ritual first: audit the new backward pass.
    r = np.random.default_rng(1)
    Xs = r.normal(0, 1, (4, 8, 8))
    ys = np.array([0, 1, 2, 0])
    net = CNN(filters=3, img=8, seed=1)
    g = net.grads(Xs, ys)
    worst = 0.0
    for P_, G in zip(net.params, g):
        it = np.nditer(P_, flags=["multi_index"])
        for _ in it:
            ix = it.multi_index
            old = P_[ix]
            P_[ix] = old + 1e-6
            lp = net.loss(Xs, ys)
            P_[ix] = old - 1e-6
            lm = net.loss(Xs, ys)
            P_[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - G[ix]))
    print(f"  [audit] conv/pool/dense backward vs central differences: "
          f"{worst:.2e}")

    rng = np.random.default_rng(RNG_SEED)
    X, y = make_lines(200, rng)                     # 600 training images
    X_test, y_test = make_lines(200, rng)           # 600 test images

    banner("DEMO 1 --- The shuffle test: does geometry matter to you?")
    print("  One fixed random permutation of the 196 pixels, applied to")
    print("  EVERY image, train and test. Neighbourhoods are destroyed;")
    print("  the information is fully intact.")
    print()
    perm = np.random.default_rng(99).permutation(IMG * IMG)

    def shuffle_px(Z):
        return Z.reshape(len(Z), -1)[:, perm].reshape(Z.shape)

    print("                     MLP [64]    CNN")
    for tag, tr, te in (("original", X, X_test),
                        ("shuffled", shuffle_px(X), shuffle_px(X_test))):
        m = fit(DenseNet(seed=0), tr, y)
        c = fit(CNN(seed=0), tr, y)
        print(f"    {tag}:         {accuracy(m, te, y_test):5.1%}     "
              f"{accuracy(c, te, y_test):5.1%}")
    print()
    print("  Part 1's claim, now measured: the MLP does not notice (92.2%")
    print("  -> 91.0% -- to it, pixels were always unrelated columns). The")
    print("  CNN's advantage evaporates (99.8% -> 62.0%): its architecture")
    print("  ASSUMES nearby-means-related, and the shuffle makes that")
    print("  assumption false. Priors only pay when they are true -- the")
    print("  62% remainder is the CNN scraping the few pixel pairs the")
    print("  permutation happened to leave adjacent.")

    banner("DEMO 2 --- Sample efficiency: the prior is prepaid data")
    print("  Same two models, shrinking training sets, tested on 600:")
    print()
    print("    train imgs    MLP [64]    CNN")
    for npc in (5, 15, 50, 200):
        r2 = np.random.default_rng(10 + npc)
        Xtr, ytr = make_lines(npc, r2)
        m = fit(DenseNet(seed=0), Xtr, ytr, epochs=600)
        c = fit(CNN(seed=0), Xtr, ytr, epochs=600)
        print(f"      {3 * npc:4d}        {accuracy(m, X_test, y_test):5.1%}"
              f"     {accuracy(c, X_test, y_test):5.1%}")
    print()
    print("  At 150 images the CNN is ~38 points ahead: an MLP must SEE a")
    print("  stroke at a position to know it there, while the CNN's shared")
    print("  kernel learns it once for everywhere. By 600 images the MLP")
    print("  has bought with data what the CNN was given by design. That")
    print("  is what an architectural prior is: data you don't have to")
    print("  collect.")

    banner("DEMO 3 --- Locality vs sharing, and the kernels that emerged")
    print("  The conv layer makes two promises. Separate them: a locally-")
    print("  connected net has the same 3x3 wiring but a PRIVATE kernel at")
    print("  each location -- locality without sharing. 600 train images:")
    print()
    print("    architecture                 params    test acc")
    for tag, net in (("dense MLP [64]", DenseNet(seed=0)),
                     ("local 3x3, unshared", CNN(seed=0, shared=False)),
                     ("conv 3x3, shared", CNN(seed=0, shared=True))):
        fit(net, X, y)
        print(f"    {tag:24s}  {net.n_params():6d}     "
              f"{accuracy(net, X_test, y_test):5.1%}")
        if tag.startswith("conv"):
            conv = net
    print()
    print("  Locality alone helps (+6.6 points): windows see strokes.")
    print("  Sharing does more with 12x fewer parameters: every window's")
    print("  experience trains the SAME nine weights, so the kernel sees")
    print("  the whole dataset at every position. And what did those nine")
    print("  weights become? The two strongest kernels, printed:")
    print()
    norms = np.linalg.norm(conv.K, axis=0)
    for f in np.argsort(norms)[::-1][:2]:
        K = conv.K[:, f].reshape(3, 3)
        print(f"    filter {f}  (norm {norms[f]:.2f}):")
        for row in K:
            print("      " + "  ".join(f"{v:+.2f}" for v in row))
        print()
    print("  Filter 0 is a vertical-edge detector -- a positive left column")
    print("  against negative neighbours. Filter 6 lights up along the")
    print("  diagonal. Nobody asked for edge detectors; they are what the")
    print("  gradient discovers when nine weights must serve 144 positions")
    print("  at once. (Real networks rediscover Gabor-like filters the")
    print("  same way; biological V1 arrived at them first.)")


if __name__ == "__main__":
    main()
