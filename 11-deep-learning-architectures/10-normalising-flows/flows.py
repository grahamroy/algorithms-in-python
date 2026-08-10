"""
flows.py --- companion code for "Normalising Flows"
(Deep Learning Architectures, Part 10).

The section's generative quartet closes with the purist. The GAN
(Part 7) never wrote a loss down; the VAE (Part 8) wrote a bound;
diffusion (Part 9) regressed its way around the problem. The
normalising flow (RealNVP: Dinh et al., 2017) refuses every
approximation: build the map from noise to data out of INVERTIBLE
steps only, and the model computes the exact probability of every
point in the plane, to the digit:

    log p(x) = log N( f(x); 0, I ) + log |det df/dx|

Each step here is an affine coupling: one coordinate passes through
untouched and sets the scale and shift for the other,

    y_b = x_b * exp(s(x_a)) + t(x_a),     log|det| = s(x_a)

-- trivially invertible, log-det read off the wire, and alternating
which coordinate leads lets a stack of them warp noise into data.
Training maximises exact log-likelihood. No adversary, no bound,
no schedule.

The stage is Parts 7-9's, on purpose, and the script retrains the
GAN, the VAE and the diffusion model live for the quartet's final
table.

Demonstrates:
  1. The exact number. Training by exact NLL on the spirals, then
     the receipts no other architecture in this section can print:
     the model's density numerically integrates to 1.0000 over the
     plane; the round trip x -> z -> x reproduces the data to
     3.94e-15 (floating-point roundoff -- the autoencoder that
     cannot lie); and individual points are priced: an on-arm
     point at p=0.144, a point two units off the manifold at
     log p = -147701.
  2. The topology tax. A flow is a continuous bijection -- it can
     bend the plane but never tear it. Depth helps (off-manifold
     mass falls 7.3% -> 1.1% from 2 to 8 couplings, and 16 buys
     nothing more), but no depth can cut noise into islands: on
     the 8-Gaussian ring the flow holds 8/8 modes on every seed
     yet pays 42-67% of its mass off-mode in filaments -- and the
     pricing receipt makes the tax exact: the empty midpoint
     between two modes is priced at p=1.11 against p=1.22 at a
     mode centre, where the TRUE density is 9.5e-13. Twelve
     orders of magnitude of topology tax, measurable only
     because the flow prices points at all.
  3. The quartet. All four generative architectures of this
     section, retrained live, one table: the GAN sharpest but
     18% of the arc uncovered; the VAE covering but soft;
     diffusion sharp AND covering at 20 passes; the flow one
     pass, exact likelihood, and honest middle scores. Four
     answers to where the difficulty lives -- the opponent, the
     encoder, the schedule, or the Jacobian.

Everything is plain NumPy, all backward passes hand-derived and
audited at startup (through the couplings AND the log-det terms).
Dependencies: numpy. Runs in about ten minutes -- the three ring
flows are most of the bill.
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
SIG_X = 0.1
T_STEPS = 100
T_EMB = 16
LOG2PI = np.log(2 * np.pi)


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# The stages and yardsticks, identical to Parts 7-9.
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


REF, REF_ARM, REF_T = spiral_reference()
N_BINS = 30


def spiral_metrics(X, thresh=0.10):
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


def off_manifold(X, thresh=0.15):
    d = np.sqrt(((X[:, None, :] - REF[None, :, :]) ** 2).sum(-1)).min(1)
    return float((d > thresh).mean())


def ring_metrics(X, sd=RING_SD):
    C = ring_centers()
    d = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1))
    nearest = d.argmin(1)
    close = d[np.arange(len(X)), nearest] < 3 * sd
    counts = np.zeros(RING_K, dtype=int)
    np.add.at(counts, nearest[close], 1)
    modes = int((counts >= 0.01 * len(X)).sum())
    return modes, float(close.mean()), counts


# ---------------------------------------------------------------------------
# Plumbing, identical to Parts 7-9.
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


# ---------------------------------------------------------------------------
# Parts 7-9's generators, minimal replicas for the quartet.
# ---------------------------------------------------------------------------

class GAN:
    def __init__(self, d_data=2, d_z=2, hidden=32, seed=0,
                 lr_d=1e-3, lr_g=1e-3):
        self.d_z = d_z
        self.G = Net(d_z, hidden, d_data, seed=seed, lr=lr_g)
        self.D = Net(d_data, hidden, 1, seed=seed + 100, lr=lr_d)

    def d_step(self, Xr, Xf):
        lr_, cr = self.D.forward(Xr)
        lf_, cf = self.D.forward(Xf)
        dlr = (sigmoid(lr_[:, 0]) - 1)[:, None] / len(Xr)
        dlf = sigmoid(lf_[:, 0])[:, None] / len(Xf)
        gr, _ = self.D.backward(cr, dlr)
        gf, _ = self.D.backward(cf, dlf)
        self.D.opt.step([a + b for a, b in zip(gr, gf)])

    def g_step(self, Z):
        Xf, cg = self.G.forward(Z)
        lf, cd = self.D.forward(Xf)
        dl = (sigmoid(lf[:, 0]) - 1)[:, None] / len(Z)
        _, dX = self.D.backward(cd, dl)
        gg, _ = self.G.backward(cg, dX)
        self.G.opt.step(gg)

    def sample(self, n, rng):
        return self.G.forward(rng.standard_normal((n, self.d_z)))[0]


class VAE:
    def __init__(self, d_data=2, d_z=2, hidden=32, seed=0, lr=1e-3,
                 beta=1.0, sig_x=SIG_X):
        self.d_z, self.beta, self.s2 = d_z, beta, sig_x ** 2
        self.enc = Net(d_data, hidden, 2 * d_z, seed=seed, lr=lr)
        self.dec = Net(d_z, hidden, d_data, seed=seed + 100, lr=lr)

    def step(self, X, rng):
        n = len(X)
        eps = rng.standard_normal((n, self.d_z))
        out, ce = self.enc.forward(X)
        mu, lv = out[:, :self.d_z], out[:, self.d_z:]
        z = mu + np.exp(0.5 * lv) * eps
        xhat, cd = self.dec.forward(z)
        dxhat = (xhat - X) / (self.s2 * n)
        gdec, dz = self.dec.backward(cd, dxhat)
        dmu = dz + self.beta * mu / n
        dlv = dz * eps * 0.5 * np.exp(0.5 * lv) \
            + self.beta * 0.5 * (np.exp(lv) - 1) / n
        genc, _ = self.enc.backward(ce, np.concatenate([dmu, dlv], 1))
        self.enc.opt.step(genc)
        self.dec.opt.step(gdec)

    def sample(self, n, rng):
        return self.dec.forward(rng.standard_normal((n, self.d_z)))[0]


def make_schedule(T=T_STEPS, b0=1e-4, b1=0.25):
    betas = np.linspace(b0, b1, T)
    alphas = 1 - betas
    return betas, alphas, np.cumprod(alphas)


def t_embed(t_idx, T=T_STEPS, d=T_EMB):
    t = np.asarray(t_idx, dtype=float)[:, None] / T
    freqs = 2 ** np.arange(d // 2)
    ang = t * freqs[None, :] * np.pi
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


class Diffusion:
    def __init__(self, d_data=2, hidden=128, seed=0, lr=1e-3, T=T_STEPS,
                 clip=1.3, ema=0.999):
        self.T = T
        self.betas, self.alphas, self.abar = make_schedule(T)
        self.clip = clip
        self.net = Net(d_data + T_EMB, hidden, d_data, seed=seed, lr=lr)
        self.ema_decay = ema
        self.ema_params = [p.copy() for p in self.net.params]

    def predict(self, Xt, t_idx, use_ema=False):
        inp = np.concatenate([Xt, t_embed(t_idx, self.T)], axis=1)
        if use_ema:
            saved = [p.copy() for p in self.net.params]
            for p, e in zip(self.net.params, self.ema_params):
                p[...] = e
            out = self.net.forward(inp)
            for p, s in zip(self.net.params, saved):
                p[...] = s
            return out
        return self.net.forward(inp)

    def step(self, X0, rng):
        t_idx = rng.integers(0, self.T, len(X0))
        eps = rng.standard_normal(X0.shape)
        sa = np.sqrt(self.abar[t_idx])[:, None]
        sb = np.sqrt(1 - self.abar[t_idx])[:, None]
        pred, cache = self.predict(sa * X0 + sb * eps, t_idx)
        grads, _ = self.net.backward(cache, 2 * (pred - eps) / len(X0))
        self.net.opt.step(grads)
        d = self.ema_decay
        for e, p in zip(self.ema_params, self.net.params):
            e *= d
            e += (1 - d) * p

    def sample(self, n, rng, n_steps=None):
        steps = (np.arange(self.T) if n_steps is None else
                 np.unique(np.linspace(0, self.T - 1, n_steps)
                           .round().astype(int)))
        X = rng.standard_normal((n, 2))
        for i in range(len(steps) - 1, -1, -1):
            t = steps[i]
            t_prev = steps[i - 1] if i > 0 else -1
            ab_t = self.abar[t]
            ab_p = self.abar[t_prev] if t_prev >= 0 else 1.0
            pred, _ = self.predict(X, np.full(n, t), use_ema=True)
            x0_hat = np.clip((X - np.sqrt(1 - ab_t) * pred)
                             / np.sqrt(ab_t), -self.clip, self.clip)
            if i > 0:
                beta_eff = 1 - ab_t / ab_p
                mean = (np.sqrt(ab_p) * beta_eff / (1 - ab_t) * x0_hat
                        + np.sqrt(1 - beta_eff) * (1 - ab_p) / (1 - ab_t)
                        * X)
                var = beta_eff * (1 - ab_p) / (1 - ab_t)
                X = mean + np.sqrt(var) * rng.standard_normal(X.shape)
            else:
                X = x0_hat
        return X


# ---------------------------------------------------------------------------
# The flow: affine couplings, exact log-likelihood, hand backprop.
# ---------------------------------------------------------------------------

class Coupling:
    """One coordinate passes through and prices the other:
    y_b = x_b * exp(s(x_a)) + t(x_a); s squashed by 1.5*tanh."""

    def __init__(self, keep, hidden, seed, lr):
        self.keep = keep
        self.net = Net(1, hidden, 2, seed=seed, lr=lr)

    def forward(self, X):
        a = X[:, self.keep:self.keep + 1]
        b = X[:, 1 - self.keep:2 - self.keep]
        out, cache = self.net.forward(a)
        sr, t = out[:, :1], out[:, 1:]
        s = 1.5 * np.tanh(sr)
        yb = b * np.exp(s) + t
        Y = np.empty_like(X)
        Y[:, self.keep:self.keep + 1] = a
        Y[:, 1 - self.keep:2 - self.keep] = yb
        return Y, s, (a, b, sr, s, t, cache)

    def backward(self, cach, dY, dlogdet):
        a, b, sr, s, t, cache = cach
        da_pass = dY[:, self.keep:self.keep + 1]
        dyb = dY[:, 1 - self.keep:2 - self.keep]
        es = np.exp(s)
        db = dyb * es
        ds = dyb * b * es + dlogdet
        dt = dyb
        dsr = ds * 1.5 * (1 - np.tanh(sr) ** 2)
        grads, dinp = self.net.backward(cache, np.concatenate(
            [dsr, dt], 1))
        dX = np.empty_like(dY)
        dX[:, self.keep:self.keep + 1] = da_pass + dinp
        dX[:, 1 - self.keep:2 - self.keep] = db
        return grads, dX

    def inverse(self, Y):
        a = Y[:, self.keep:self.keep + 1]
        yb = Y[:, 1 - self.keep:2 - self.keep]
        out, _ = self.net.forward(a)
        sr, t = out[:, :1], out[:, 1:]
        s = 1.5 * np.tanh(sr)
        b = (yb - t) * np.exp(-s)
        X = np.empty_like(Y)
        X[:, self.keep:self.keep + 1] = a
        X[:, 1 - self.keep:2 - self.keep] = b
        return X


class Flow:
    def __init__(self, n_layers=8, hidden=64, seed=0, lr=1e-3):
        self.layers = [Coupling(k % 2, hidden, seed + k, lr)
                       for k in range(n_layers)]

    def logp(self, X, keep_caches=False):
        Z = X
        logdet = np.zeros(len(X))
        caches = []
        for L in self.layers:
            Z, s, c = L.forward(Z)
            logdet += s[:, 0]
            caches.append(c)
        lp = -0.5 * (Z ** 2).sum(1) - LOG2PI + logdet
        if keep_caches:
            return lp, Z, caches
        return lp

    def nll(self, X):
        return float(-self.logp(X).mean())

    def step_(self, X):
        n = len(X)
        lp, Z, caches = self.logp(X, keep_caches=True)
        dZ = Z / n
        dlogdet = -np.ones((n, 1)) / n
        all_grads = []
        dY = dZ
        for L, c in zip(reversed(self.layers), reversed(caches)):
            grads, dY = L.backward(c, dY, dlogdet)
            all_grads.append((L, grads))
        return float(-lp.mean()), all_grads

    def step(self, X):
        loss, all_grads = self.step_(X)
        for L, g in all_grads:
            L.net.opt.step(g)
        return loss

    def sample(self, n, rng):
        X = rng.standard_normal((n, 2))
        for L in reversed(self.layers):
            X = L.inverse(X)
        return X

    def to_z(self, X):
        Z = X
        for L in self.layers:
            Z, _, _ = L.forward(Z)
        return Z


def gradcheck(seed=3):
    rng = np.random.default_rng(seed)
    f = Flow(n_layers=4, hidden=6, seed=seed)
    X = rng.standard_normal((6, 2))
    _, all_grads = f.step_(X)
    worst = 0.0
    r = np.random.default_rng(seed + 1)
    for L, grads in all_grads:
        for P, Gr in zip(L.net.params, grads):
            flat, gflat = P.reshape(-1), Gr.reshape(-1)
            for ix in r.choice(len(flat), min(8, len(flat)),
                               replace=False):
                old = flat[ix]
                flat[ix] = old + 1e-6
                lp = f.nll(X)
                flat[ix] = old - 1e-6
                lm = f.nll(X)
                flat[ix] = old
                worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
    return worst


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo1():
    banner("DEMO 1 --- The exact number")
    print("  Train by maximising exact log-likelihood on the spirals --")
    print("  no bound, no adversary. Eval NLL on held-out data, plus the")
    print("  shared yardsticks on 2,000 samples (drawn in ONE inverse")
    print("  pass each):")
    print()
    f = Flow(n_layers=8, hidden=64, seed=0)
    rng = np.random.default_rng(1)
    Xev = spiral_data(2000, np.random.default_rng(96))
    print("     step   eval NLL   q@0.10   q@0.05   coverage")
    for s in range(8000):
        f.step(spiral_data(128, rng))
        if (s + 1) % 2000 == 0:
            Xs = f.sample(2000, np.random.default_rng(99))
            q, c = spiral_metrics(Xs)
            q5, _ = spiral_metrics(Xs, thresh=0.05)
            print(f"    {s + 1:5d}    {f.nll(Xev):6.3f}    {q:6.1%}   "
                  f"{q5:6.1%}    {c:6.1%}")
    print()
    print("  (The eval NLL wobbles -- 1.297 at step 4000 after 1.247 --")
    print("  exactness of the LOSS does not make the OPTIMISATION")
    print("  monotone. The number is honest; the descent is still SGD.)")
    print()
    print("  Now the receipts no other architecture in this section can")
    print("  print. First: is it actually a probability density?")
    print()
    g = np.linspace(-2.5, 2.5, 500)
    GX, GY = np.meshgrid(g, g)
    P = np.stack([GX.ravel(), GY.ravel()], 1)
    lp = f.logp(P)
    dA = (g[1] - g[0]) ** 2
    print(f"    integral of p(x) over the plane : "
          f"{np.exp(lp).sum() * dA:.4f}")
    Z = f.to_z(Xev)
    Xback = Z.copy()
    for L in reversed(f.layers):
        Xback = L.inverse(Xback)
    print(f"    round trip max |x - f_inv(f(x))|: "
          f"{np.abs(Xback - Xev).max():.2e}")
    print(f"    z of real data: mean {Z.mean(0).round(3)}, "
          f"var {Z.var(0).round(3)}")
    print()
    print("  It integrates to 1.0000: a genuine density, verified by")
    print("  quadrature. The round trip is exact to floating-point")
    print("  roundoff -- Part 8's VAE reconstructed with error 0.002;")
    print("  the flow is the autoencoder that CANNOT lie. And it prices")
    print("  individual points:")
    print()
    pts = np.array([[0.12, 0.0], [0.0, 0.0], [2.0, 2.0]])
    names = ["on an arm       ", "between the arms", "far off-manifold"]
    for nm, p, l in zip(names, pts, f.logp(pts)):
        print(f"    log p ({p[0]:5.2f},{p[1]:5.2f})  {nm} : {l:11.3f}")
    print()
    print("  The far point is priced at log p = -147701 -- charged into")
    print("  oblivion, as it should be. But look at the middle row: the")
    print("  empty spot BETWEEN the arms costs nearly the same as a spot")
    print("  ON an arm. Hold that thought; it is DEMO 2's subject.")
    return f


def demo2(f):
    banner("DEMO 2 --- The topology tax")
    print("  A flow is a continuous bijection: it can bend the plane,")
    print("  stretch it, fold it back -- but never TEAR it. Depth buys")
    print("  precision (off-manifold = samples farther than 0.15):")
    print()
    print("    couplings   eval NLL   q@0.10   off-manifold")
    Xev = spiral_data(2000, np.random.default_rng(96))
    for K in (2, 4, 8, 16):
        fk = Flow(n_layers=K, hidden=64, seed=0)
        rng = np.random.default_rng(1)
        for _ in range(8000):
            fk.step(spiral_data(128, rng))
        Xs = fk.sample(2000, np.random.default_rng(99))
        q, _ = spiral_metrics(Xs)
        print(f"       {K:2d}       {fk.nll(Xev):6.3f}    {q:6.1%}      "
              f"{off_manifold(Xs):5.1%}")
    print()
    print("  Two couplings are crude; eight polish the off-manifold mass")
    print("  down to 1.1%; sixteen buy nothing more. But no depth can do")
    print("  what the spirals do not demand and the ring DOES: cut one")
    print("  connected blob of noise into eight islands. The ring,")
    print("  twelve couplings, three seeds:")
    print()
    frs = []
    for seed in range(3):
        fr = Flow(n_layers=12, hidden=64, seed=seed, lr=3e-4)
        rng = np.random.default_rng(seed + 10)
        for _ in range(16000):
            fr.step(make_ring(256, rng))
        Xs = fr.sample(2000, np.random.default_rng(99))
        m, hq, _ = ring_metrics(Xs)
        frs.append(fr)
        print(f"    seed {seed}: modes {m}/8   on-mode mass {hq:.0%}")
    print()
    print("  Every mode, every seed -- a flow cannot drop an island,")
    print("  because dropping it would tear the map. But 42-67% of the")
    print("  mass lands OFF-mode: the islands stay connected by")
    print("  filaments of probability the data never asked for. And")
    print("  because a flow prices points exactly, the tax can be")
    print("  itemised (seed 0):")
    print()
    fr = frs[0]
    C = ring_centers()
    mid = (C[0] + C[1]) / 2
    mid = mid / np.linalg.norm(mid)
    pts = np.array([C[0], mid, [0.0, 0.0]])
    names = ["at a mode centre  ", "BETWEEN two modes ",
             "at the ring centre"]
    lps = fr.logp(pts)
    for nm, l in zip(names, lps):
        print(f"    p {nm}: {np.exp(l):9.2e}")
    d2s = ((mid[None, :] - C) ** 2).sum(1)
    true_p = (np.exp(-d2s / (2 * RING_SD ** 2))
              / (8 * 2 * np.pi * RING_SD ** 2)).sum()
    print(f"    TRUE p between modes: {true_p:9.2e}")
    print()
    print("  The model prices the empty midpoint at 1.11 -- 91% of a")
    print("  mode centre's 1.22 -- where the true density is 9.5e-13.")
    print("  Twelve orders of magnitude of topology tax, printed to the")
    print("  digit by the only architecture capable of printing it. The")
    print("  flow's superpower and its signature failure are the same")
    print("  fact: the map must be smooth, invertible, and whole.")


def demo3(f):
    banner("DEMO 3 --- The quartet")
    print("  Four articles, four generative architectures, one stage.")
    print("  The GAN, VAE and diffusion model retrained live (same seeds")
    print("  and budgets as Parts 7-9; diffusion sampled at its best")
    print("  20-pass setting), the flow from DEMO 1, the real data, and")
    print("  the pure-noise floor:")
    print()
    gan = GAN(seed=0)
    grng = np.random.default_rng(1)
    for _ in range(6000):
        Xr = spiral_data(128, grng)
        Zb = grng.standard_normal((128, gan.d_z))
        Xf, _ = gan.G.forward(Zb)
        gan.d_step(Xr, Xf)
        gan.g_step(grng.standard_normal((128, gan.d_z)))
    v = VAE(seed=0)
    vrng = np.random.default_rng(1)
    for _ in range(6000):
        v.step(spiral_data(128, vrng), vrng)
    dm = Diffusion(seed=0)
    drng = np.random.default_rng(1)
    for _ in range(16000):
        dm.step(spiral_data(128, drng), drng)
    rows = [
        ("GAN (Part 7)        ", 1,
         gan.sample(2000, np.random.default_rng(99))),
        ("VAE (Part 8)        ", 1,
         v.sample(2000, np.random.default_rng(99))),
        ("diffusion (Part 9)  ", 20,
         dm.sample(2000, np.random.default_rng(99), n_steps=20)),
        ("flow (this article) ", 1,
         f.sample(2000, np.random.default_rng(99))),
        ("real data           ", 0,
         spiral_data(2000, np.random.default_rng(97))),
        ("pure noise (floor)  ", 0,
         np.random.default_rng(99).standard_normal((2000, 2))),
    ]
    print("                          passes   q@0.10   q@0.05   coverage")
    for name, passes, Xx in rows:
        q10, c10 = spiral_metrics(Xx)
        q5, _ = spiral_metrics(Xx, thresh=0.05)
        ptxt = f"{passes:4d}" if passes else "   -"
        print(f"    {name}  {ptxt}    {q10:6.1%}   {q5:6.1%}    {c10:6.1%}")
    print()
    print("  The section's generative story in one table. The GAN:")
    print("  sharpest, cheapest, and 18% of the arc missing -- the")
    print("  mode-seeking game. The VAE: everything covered, softly --")
    print("  the honest bound. Diffusion: sharp AND covering, at twenty")
    print("  passes -- the trade refused, paid in compute. The flow: one")
    print("  pass, honest middle scores on every yardstick -- and the")
    print("  only row that can tell you, exactly, how probable any of")
    print("  its opinions are. Four architectures, four answers to the")
    print("  same question: where should the difficulty live? In the")
    print("  opponent, the encoder, the schedule -- or the Jacobian.")


def main() -> None:
    print(f"  [audit] flow backward (couplings + log-det) vs central")
    print(f"          differences: {gradcheck():.2e}")
    f = demo1()
    demo2(f)
    demo3(f)


if __name__ == "__main__":
    main()
