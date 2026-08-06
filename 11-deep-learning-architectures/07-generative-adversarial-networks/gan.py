"""
gan.py --- companion code for "Generative Adversarial Networks"
(Deep Learning Architectures, Part 7).

Every architecture in this section so far CLASSIFIED: given a point,
a picture, a sentence, a graph -- name it. This article pivots to
architectures that CREATE, and it starts with the strangest training
signal in deep learning: no loss function is written down for the
generator at all. Instead, two networks play a game (Goodfellow et
al., 2014). A GENERATOR (the forger) maps random noise z to fake
data G(z). A DISCRIMINATOR (the detective) is a plain binary
classifier: real or fake? The detective's mistakes are the forger's
gradient -- each network is the other's loss function.

The stage: the section's own two spirals (Parts 1-2 classified them;
this article learns to DRAW them from noise), plus the classic ring
of 8 Gaussians for the failure studies.

Demonstrates:
  1. The game works: from noise to spirals. Sample quality reaches
     98.7% (fraction of generated points within 0.10 of the true
     curve) against the real data's own anchor of 99.5%. The
     detective ends at 50.4% accuracy -- a coin flip, which is what
     victory looks like. And the score OSCILLATES on the way
     (99.4% at step 4000, 93.7% at 5000, back to 98.7%; coverage
     swings 60-87% and never reaches the real data's 93.3%):
     a game orbits an equilibrium, it does not descend.
  2. The frozen forger -- the section's vanishing gradient, fourth
     appearance (Part 1: depth, Part 3: time, Part 6: over-smoothing;
     now it vanishes through the OPPONENT'S CONFIDENCE). When the
     detective wins early, the original minimax loss log(1-D(G(z)))
     hands the forger a gradient proportional to D(G(z)) ~ 1e-3.
     On SGD the forger freezes forever: |grad| decays to ~1e-38,
     final quality 0.0%. The non-saturating fix -log D(G(z)) --
     in the original GAN paper -- recovers 93.1% from the identical
     start. Adam's scale normalisation masks the disease (99.8%
     even with the saturating loss) but the standard cure is the
     non-saturating loss.
  3. Mode collapse is a RACE. Same architecture, same data, only
     the detective's learning rate drops: 1e-3 covers 8/8 modes on
     every seed; 3e-5 covers 0-1. In between, the forger visibly
     FLEES around the ring -- dominant mode 6, 5, (none), 7, 5,
     1, 0, 7, 0, 3, 6, 0 across the twelve checkpoints, never
     settling. The equilibrium is a balance of speeds, not a
     bottom to descend to.

Everything is plain NumPy, both backward passes hand-derived and
audited at startup. Dependencies: numpy. Runs in under a minute.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
TURNS = 2.5
NOISE = 0.04
R0 = 0.12
RING_K = 8
RING_R = 1.0
RING_SD = 0.05
HID = 32
D_Z = 2


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# The stages: the section's two spirals, and the ring of 8 Gaussians.
# ---------------------------------------------------------------------------

def make_spirals(n_per_class, noise, rng, turns=TURNS, r0=R0):
    out = []
    for cls in range(2):
        t = rng.uniform(0, 1, n_per_class) ** 0.5
        theta = t * turns * 2 * np.pi + cls * np.pi
        r = r0 + (1 - r0) * t
        out.append(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1))
    X = np.concatenate(out) + rng.normal(0, noise, (2 * n_per_class, 2))
    idx = rng.permutation(len(X))
    return X[idx]


def spiral_data(n, rng):
    return make_spirals(n // 2, NOISE, rng)


def spiral_reference(n=4000, turns=TURNS, r0=R0):
    """Dense noise-free curve points on both arms, with (arm, t) tags."""
    t = np.linspace(0, 1, n)
    pts, arm, tt = [], [], []
    for cls in range(2):
        theta = t * turns * 2 * np.pi + cls * np.pi
        r = r0 + (1 - r0) * t
        pts.append(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1))
        arm.append(np.full(n, cls))
        tt.append(t)
    return np.concatenate(pts), np.concatenate(arm), np.concatenate(tt)


def make_ring(n, rng, k=RING_K, radius=RING_R, sd=RING_SD):
    c = rng.integers(0, k, n)
    ang = 2 * np.pi * c / k
    ctr = radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return ctr + sd * rng.standard_normal((n, 2))


def ring_centers(k=RING_K, radius=RING_R):
    ang = 2 * np.pi * np.arange(k) / k
    return radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)


# ---------------------------------------------------------------------------
# Honest yardsticks. Both are anchored by scoring the REAL data too.
# ---------------------------------------------------------------------------

REF, REF_ARM, REF_T = spiral_reference()
N_BINS = 30


def spiral_metrics(X, thresh=0.10):
    """(quality, coverage): points near the curve; arm-bins occupied."""
    d2 = ((X[:, None, :] - REF[None, :, :]) ** 2).sum(-1)
    nn = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(X)), nn])
    ok = dist < thresh
    quality = float(ok.mean())
    bins = (REF_ARM[nn] * N_BINS + np.minimum(
        (REF_T[nn] * N_BINS).astype(int), N_BINS - 1)).astype(int)
    hit = np.zeros(2 * N_BINS, dtype=int)
    np.add.at(hit, bins[ok], 1)
    coverage = float((hit >= 3).mean())
    return quality, coverage


def ring_metrics(X, sd=RING_SD):
    """(modes covered, on-mode mass, per-mode counts within 3 sigma)."""
    C = ring_centers()
    d = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    nearest = d.argmin(1)
    close = d[np.arange(len(X)), nearest] < 3 * sd
    counts = np.zeros(RING_K, dtype=int)
    np.add.at(counts, nearest[close], 1)
    modes = int((counts >= 0.01 * len(X)).sum())
    return modes, float(close.mean()), counts


# ---------------------------------------------------------------------------
# Plumbing: activations, optimisers, one MLP class for both players.
# ---------------------------------------------------------------------------

def lrelu(z):
    return np.where(z > 0, z, 0.2 * z)


def dlrelu(z):
    return np.where(z > 0, 1.0, 0.2)


def sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def softplus(z):
    return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))


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


class SGD:
    def __init__(self, params, lr):
        self.p, self.lr = params, lr

    def step(self, grads):
        for p, g in zip(self.p, grads):
            p -= self.lr * g


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
        """Param grads AND the gradient w.r.t. the input -- the forger
        needs the latter: its learning signal arrives THROUGH the
        detective's input socket."""
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


# ---------------------------------------------------------------------------
# The GAN: a forger, a detective, and two ways to phrase the forger's loss.
# ---------------------------------------------------------------------------

class GAN:
    def __init__(self, d_data=2, d_z=D_Z, hidden=HID, seed=0,
                 lr_d=1e-3, lr_g=1e-3, saturating=False):
        self.d_z = d_z
        self.saturating = saturating
        self.G = Net(d_z, hidden, d_data, seed=seed, lr=lr_g)
        self.D = Net(d_data, hidden, 1, seed=seed + 100, lr=lr_d)

    def d_logit(self, X):
        out, cache = self.D.forward(X)
        return out[:, 0], cache

    # -- losses, for the audits
    def d_loss(self, Xr, Xf):
        lr_, _ = self.d_logit(Xr)
        lf_, _ = self.d_logit(Xf)
        return float(softplus(-lr_).mean() + softplus(lf_).mean())

    def g_loss(self, Z):
        Xf, _ = self.G.forward(Z)
        lf, _ = self.d_logit(Xf)
        if self.saturating:
            return float(-softplus(lf).mean())      # log(1 - D(G(z)))
        return float(softplus(-lf).mean())          # -log D(G(z))

    # -- the detective's move: binary cross-entropy, real=1 fake=0
    def d_step(self, Xr, Xf):
        lr_, cr = self.d_logit(Xr)
        lf_, cf = self.d_logit(Xf)
        dlr = (sigmoid(lr_) - 1)[:, None] / len(lr_)
        dlf = sigmoid(lf_)[:, None] / len(lf_)
        gr, _ = self.D.backward(cr, dlr)
        gf, _ = self.D.backward(cf, dlf)
        self.D.opt.step([a + b for a, b in zip(gr, gf)])

    # -- the forger's move: backprop THROUGH the frozen detective
    def g_grads(self, Z):
        Xf, cg = self.G.forward(Z)
        lf, cd = self.d_logit(Xf)
        n = len(lf)
        if self.saturating:
            dl = -sigmoid(lf)[:, None] / n          # ~0 when D is sure
        else:
            dl = (sigmoid(lf) - 1)[:, None] / n     # ~-1/n when D is sure
        _, dX = self.D.backward(cd, dl)
        gg, _ = self.G.backward(cg, dX)
        return gg

    def g_step(self, Z):
        self.G.opt.step(self.g_grads(Z))

    def sample(self, n, rng):
        return self.G.forward(rng.standard_normal((n, self.d_z)))[0]


def d_accuracy(gan, data_fn, n, rng):
    Xr = data_fn(n, rng)
    Xf = gan.sample(n, rng)
    lr_, _ = gan.d_logit(Xr)
    lf_, _ = gan.d_logit(Xf)
    return float(np.concatenate([(lr_ > 0), (lf_ <= 0)]).mean())


# ---------------------------------------------------------------------------
# Audit: both hand backprops against central differences.
# ---------------------------------------------------------------------------

def gradcheck(seed=3):
    rng = np.random.default_rng(seed)
    gan = GAN(hidden=8, seed=seed)
    Xr = rng.standard_normal((6, 2))
    Z = rng.standard_normal((6, 2))
    Xf = gan.G.forward(Z)[0]

    lr_, cr = gan.d_logit(Xr)
    lf_, cf = gan.d_logit(Xf)
    dlr = (sigmoid(lr_) - 1)[:, None] / len(lr_)
    dlf = sigmoid(lf_)[:, None] / len(lf_)
    gr, _ = gan.D.backward(cr, dlr)
    gf, _ = gan.D.backward(cf, dlf)
    gD = [a + b for a, b in zip(gr, gf)]
    worst_d = 0.0
    r = np.random.default_rng(seed + 1)
    for P, Gd in zip(gan.D.params, gD):
        flat, gflat = P.reshape(-1), Gd.reshape(-1)
        for ix in r.choice(len(flat), min(10, len(flat)), replace=False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp = gan.d_loss(Xr, Xf)
            flat[ix] = old - 1e-6
            lm = gan.d_loss(Xr, Xf)
            flat[ix] = old
            worst_d = max(worst_d, abs((lp - lm) / 2e-6 - gflat[ix]))

    worst_g = 0.0
    for sat in (False, True):
        gan.saturating = sat
        gG = gan.g_grads(Z)
        for P, Gg in zip(gan.G.params, gG):
            flat, gflat = P.reshape(-1), Gg.reshape(-1)
            for ix in r.choice(len(flat), min(10, len(flat)), replace=False):
                old = flat[ix]
                flat[ix] = old + 1e-6
                lp = gan.g_loss(Z)
                flat[ix] = old - 1e-6
                lm = gan.g_loss(Z)
                flat[ix] = old
                worst_g = max(worst_g, abs((lp - lm) / 2e-6 - gflat[ix]))
    gan.saturating = False
    return worst_d, worst_g


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo1() -> None:
    banner("DEMO 1 --- The game: from noise to spirals")
    print("  A forger maps 2-d Gaussian noise to points; a detective")
    print("  classifies real vs fake; each is the other's loss function.")
    print("  The target is this section's own two-spirals -- Parts 1 and 2")
    print("  classified them, tonight we draw them. Quality = generated")
    print("  points within 0.10 of the true curve; coverage = 60 arc-bins")
    print("  occupied. The REAL data itself anchors both yardsticks.")
    print()
    gan = GAN(seed=0)
    rng = np.random.default_rng(1)
    print("     step   quality   coverage   detective acc")
    for s in range(6000):
        Xr = spiral_data(128, rng)
        Z = rng.standard_normal((128, gan.d_z))
        Xf, _ = gan.G.forward(Z)
        gan.d_step(Xr, Xf)
        gan.g_step(rng.standard_normal((128, gan.d_z)))
        if (s + 1) % 1000 == 0:
            Xs = gan.sample(2000, np.random.default_rng(99))
            q, c = spiral_metrics(Xs)
            acc = d_accuracy(gan, spiral_data, 1000,
                             np.random.default_rng(98))
            print(f"    {s + 1:5d}    {q:6.1%}    {c:6.1%}       {acc:6.1%}")
    Xreal = spiral_data(2000, np.random.default_rng(97))
    q, c = spiral_metrics(Xreal)
    print(f"     real    {q:6.1%}    {c:6.1%}       (anchor)")
    print()
    print("  The forger ends within a point of the real data's own quality")
    print("  score, and the detective ends at a coin flip -- in this game")
    print("  that is what victory looks like. But note the path: quality")
    print("  dips to 93.7% at step 5000 AFTER 99.4% at 4000, and coverage")
    print("  swings 60-87% without ever reaching the real data's 93.3% --")
    print("  the spiral's tails keep being lost and re-found. A game does")
    print("  not descend a loss; it orbits an equilibrium. There is no")
    print("  single number to watch going down -- the section's first")
    print("  architecture where the training curve cannot be trusted.")


def demo2() -> None:
    banner("DEMO 2 --- The frozen forger: the vanishing gradient, "
           "4th appearance")
    print("  The original minimax loss for the forger is log(1-D(G(z))),")
    print("  whose gradient is proportional to D(G(z)) -- the detective's")
    print("  belief that the fake is real. If the detective wins early,")
    print("  that belief is ~0, and so is the forger's gradient. We stage")
    print("  the bad start honestly: the forger's first drafts land far")
    print("  from the data (in high dimension this happens for free), and")
    print("  the detective studies them for 500 steps before the forger")
    print("  moves. After the head start, mean D(fake) = 6.4e-04.")
    print()
    print("    optimiser   forger loss        final quality   coverage")
    rows = []
    for opt_name in ("sgd", "adam"):
        for sat in (False, True):
            gan = GAN(seed=0, saturating=sat)
            gan.G.b3 += np.array([2.5, 2.5])
            if opt_name == "sgd":
                gan.G.opt = SGD(gan.G.params, 0.05)
            rng = np.random.default_rng(1)
            for _ in range(500):
                Xr = spiral_data(128, rng)
                Z = rng.standard_normal((128, gan.d_z))
                Xf, _ = gan.G.forward(Z)
                gan.d_step(Xr, Xf)
            norms = []
            for s in range(6000):
                Xr = spiral_data(128, rng)
                Zb = rng.standard_normal((128, gan.d_z))
                Xf, _ = gan.G.forward(Zb)
                gan.d_step(Xr, Xf)
                Z2 = rng.standard_normal((128, gan.d_z))
                gg = gan.g_grads(Z2)
                if s in (9, 199, 999, 2999, 5999):
                    norms.append((s + 1, float(np.sqrt(
                        sum((g ** 2).sum() for g in gg)))))
                gan.G.opt.step(gg)
            Xs = gan.sample(2000, np.random.default_rng(99))
            q, c = spiral_metrics(Xs)
            lbl = "saturating    " if sat else "non-saturating"
            print(f"    {opt_name.upper():4s}        {lbl}     "
                  f"{q:6.1%}        {c:6.1%}")
            rows.append((opt_name, sat, norms))
    print()
    print("  The forger's gradient norm, SGD rows only:")
    for opt_name, sat, norms in rows:
        if opt_name != "sgd":
            continue
        lbl = "saturating    " if sat else "non-saturating"
        print(f"    {lbl}: " + "  ".join(
            f"{s}:{n:.0e}" for s, n in norms))
    print()
    print("  On SGD the saturating forger is frozen solid -- gradient")
    print("  norms around 1e-38, quality 0.0%, forever. The non-saturating")
    print("  loss -log D(G(z)) faces the identical detective and walks")
    print("  home: its gradient approaches -1, not 0, when the detective")
    print("  is confident. Part 1 lost gradients to depth, Part 3 to time,")
    print("  Part 6 to over-smoothing; a GAN loses them to the OPPONENT'S")
    print("  CONFIDENCE. Adam largely masks the disease (its update is")
    print("  scale-normalised), but the cure that shipped in the original")
    print("  paper is the non-saturating loss.")


def demo3() -> None:
    banner("DEMO 3 --- Mode collapse is a race")
    print("  New stage: 8 Gaussians on a ring. The forger need only pick")
    print("  one mode the detective currently trusts and pour everything")
    print("  into it -- unless the detective adapts fast enough to punish")
    print("  repetition. Same architecture, same data; the only knob is")
    print("  the detective's learning rate. 3 seeds each:")
    print()
    print("    detective lr    modes covered      on-mode mass")
    for lr_d in (1e-3, 3e-4, 1e-4, 3e-5):
        mm = []
        for seed in range(3):
            gan = GAN(seed=seed, lr_d=lr_d, lr_g=1e-3)
            rng = np.random.default_rng(seed + 10)
            for _ in range(4000):
                Xr = make_ring(64, rng)
                Z = rng.standard_normal((64, gan.d_z))
                Xf, _ = gan.G.forward(Z)
                gan.d_step(Xr, Xf)
                gan.g_step(rng.standard_normal((64, gan.d_z)))
            Xs = gan.sample(2000, np.random.default_rng(99))
            m, hq, _ = ring_metrics(Xs)
            mm.append((m, hq))
        print(f"       {lr_d:7.0e}      " +
              "  ".join(f"{m}/8" for m, _ in mm) + "      " +
              "  ".join(f"{hq:3.0%}" for _, hq in mm))
    print()
    print("  A clean dose-response: slow the detective 30x and the forger")
    print("  stops bothering to learn the distribution at all. And what")
    print("  collapse looks like from inside -- the slow-detective run,")
    print("  dominant mode over time:")
    print()
    gan = GAN(seed=0, lr_d=1e-4, lr_g=1e-3)
    rng = np.random.default_rng(10)
    trace = []
    for s in range(6000):
        Xr = make_ring(64, rng)
        Z = rng.standard_normal((64, gan.d_z))
        Xf, _ = gan.G.forward(Z)
        gan.d_step(Xr, Xf)
        gan.g_step(rng.standard_normal((64, gan.d_z)))
        if (s + 1) % 500 == 0:
            Xs = gan.sample(2000, np.random.default_rng(99))
            m, hq, counts = ring_metrics(Xs)
            dom = int(np.argmax(counts)) if counts.sum() else -1
            trace.append((s + 1, m, dom))
    print("     step    modes covered    dominant mode")
    for s, m, d in trace:
        print(f"    {s:5d}        {m}/8            "
              f"{'--' if d < 0 else d:>2}")
    print()
    print("  The forger camps on one mode, the slow detective eventually")
    print("  learns to reject it, the forger flees to another -- around")
    print("  and around the ring, never covering it. Mode collapse is not")
    print("  a wrong loss; it is a lost race. Every practical GAN trick")
    print("  -- two-timescale learning rates, more detective steps per")
    print("  forger step, minibatch statistics -- is a way of keeping the")
    print("  detective ahead. The equilibrium is a balance of speeds,")
    print("  not a bottom to descend to.")


def main() -> None:
    wd, wg = gradcheck()
    print(f"  [audit] detective backward vs central differences: {wd:.2e}")
    print(f"  [audit] forger backward (through the detective)  : {wg:.2e}")
    demo1()
    demo2()
    demo3()


if __name__ == "__main__":
    main()
