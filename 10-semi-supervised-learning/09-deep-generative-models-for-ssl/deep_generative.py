"""
deep_generative.py --- companion code for "Deep Generative Models for SSL"
(Semi-Supervised Learning, Part 9).

Part 4 (EM) laid out the generative bargain: commit to a story of how the
data was MADE -- p(x|class) -- and every unlabelled point becomes evidence
about that story. It also exposed the fine print: the model believes its
story, not your labels. Part 4 kept the story polite (each class is a
Gaussian) and the data agreeable (blobs). This part breaks the polite
version on purpose -- two moons are not Gaussians -- and then UPGRADES THE
STORY instead of abandoning it:

    each class's density becomes a tiny VARIATIONAL AUTOENCODER --
    a neural decoder that bends a 1-D latent variable into a curve
    in data space. A moon IS a 1-D curve plus noise.

The mixture is trained EM-style, exactly as in Part 4:
  E-step: responsibilities from each class model's likelihood
          (here the ELBO, a lower bound on log p(x|class)),
          with the 8 labelled points clamped.
  M-step: each VAE takes a gradient step on responsibility-weighted data.

Two training details matter (both measured, not decorative):
  * warm start -- each VAE first fits ONLY its own 4 labelled points, so
    the story grows outward from the labels rather than both components
    fitting everything symmetrically and never separating;
  * anchor -- labelled points get extra weight in the M-step, so the
    clamp is felt by the gradients, not just the responsibilities.

Demonstrates (two moons, Part 1's exact data and 8-label draws):
  1. The wrong story: Part 4's semi-supervised Gaussian mixture on moons.
     Unlabelled data still helps (81.1% vs the 77.4% supervised MLP of
     Part 8), but the story is the ceiling.
  2. The upgraded story: mixture of two tiny VAEs -- 98.0% mean, every
     draw at 97.6%+, the track's best inductive scoreboard, and a better
     density: the VAEs' lower bound beats the GMM's exact log-density.
  3. The payoff no discriminative method offers: sweep the 1-D latent and
     the decoder WALKS THE MOON -- it rediscovered the arc parametrisation
     (cos t, sin t) it was never shown.

Everything (VAE forward/backward, reparameterisation, EM loop, GMM) is
plain NumPy. Dependencies: numpy. Runs in about half a minute.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
SIG_X = 0.15                              # observation noise (the data's own)
LOGC = -np.log(2 * np.pi * SIG_X ** 2)    # 2-D Gaussian normaliser
EPOCHS = 3400
WARMUP = 400
LAMBDA = 10.0                             # labelled-point anchor weight


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Part 1's exact data pipeline
# ---------------------------------------------------------------------------

def make_moons(n_per_class, noise, rng):
    t = rng.uniform(0, np.pi, n_per_class)
    top = np.stack([np.cos(t), np.sin(t)], axis=1)
    t = rng.uniform(0, np.pi, n_per_class)
    bottom = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([top, bottom])
    X += rng.normal(0, noise, X.shape)
    y = np.concatenate([np.zeros(n_per_class, dtype=int),
                        np.ones(n_per_class, dtype=int)])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# DEMO 1's model: Part 4's semi-supervised Gaussian mixture, full covariance,
# labelled points clamped in the E-step.
# ---------------------------------------------------------------------------

def gauss_logpdf(X, mu, S):
    d = X - mu
    Si = np.linalg.inv(S)
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * (np.einsum('ij,jk,ik->i', d, Si, d)
                   + logdet + 2 * np.log(2 * np.pi))


def gmm_ssl(X, li, yl, iters=100):
    n = len(X)
    mus = np.array([X[li][yl == c].mean(axis=0) for c in (0, 1)])
    Ss = [np.cov(X.T) + 1e-6 * np.eye(2)] * 2
    pis = np.array([0.5, 0.5])
    for _ in range(iters):
        logp = np.stack([np.log(pis[c]) + gauss_logpdf(X, mus[c], Ss[c])
                         for c in (0, 1)], axis=1)
        logp -= logp.max(axis=1, keepdims=True)
        R = np.exp(logp)
        R /= R.sum(axis=1, keepdims=True)
        R[li] = np.eye(2)[yl]                       # the clamp
        Nk = R.sum(axis=0)
        pis = Nk / n
        mus = (R.T @ X) / Nk[:, None]
        Ss = []
        for c in (0, 1):
            d = X - mus[c]
            Ss.append((R[:, c][:, None] * d).T @ d / Nk[c]
                      + 1e-6 * np.eye(2))
    return mus, Ss, pis


def gmm_logjoint(Xq, mus, Ss, pis):
    return np.stack([np.log(pis[c]) + gauss_logpdf(Xq, mus[c], Ss[c])
                     for c in (0, 1)], axis=1)


# ---------------------------------------------------------------------------
# DEMO 2's model: a tiny VAE per class. Encoder x -> (mu, log v) for a 1-D
# latent; decoder z -> xhat with fixed observation noise SIG_X. All gradients
# by hand, including through the reparameterisation z = mu + sqrt(v) * eps.
# ---------------------------------------------------------------------------

class Net:
    """One tanh hidden layer, linear output. grads() also returns the
    gradient at the input -- the decoder's input gradient is what carries
    the reconstruction error back to the latent."""

    def __init__(self, n_in, hidden, n_out, lr, seed):
        r = np.random.default_rng(seed)
        self.P = [r.standard_normal((n_in, hidden)) * np.sqrt(1.0 / n_in),
                  np.zeros(hidden),
                  r.standard_normal((hidden, n_out)) * np.sqrt(1.0 / hidden),
                  np.zeros(n_out)]
        self.M = [np.zeros_like(p) for p in self.P]
        self.V = [np.zeros_like(p) for p in self.P]
        self.t = 0
        self.lr = lr

    def fwd(self, X):
        H = np.tanh(X @ self.P[0] + self.P[1])
        return H @ self.P[2] + self.P[3], H

    def grads(self, X, H, dout, w):
        """Per-sample output gradients dout, per-sample weights w."""
        dw = dout * w[:, None]
        gW2 = H.T @ dw
        gb2 = dw.sum(axis=0)
        dH = dw @ self.P[2].T * (1 - H ** 2)
        gW1 = X.T @ dH
        gb1 = dH.sum(axis=0)
        dX = dH @ self.P[0].T / np.maximum(w[:, None], 1e-12)  # unweighted
        return [gW1, gb1, gW2, gb2], dX

    def adam(self, g):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, gi) in enumerate(zip(self.P, g)):
            self.M[i] = b1 * self.M[i] + (1 - b1) * gi
            self.V[i] = b2 * self.V[i] + (1 - b2) * gi ** 2
            p -= self.lr * (self.M[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.V[i] / (1 - b2 ** self.t)) + eps)


class VAE:
    def __init__(self, seed, lr=5e-3, hidden=16):
        self.enc = Net(2, hidden, 2, lr, seed)         # x -> [mu, log v]
        self.dec = Net(1, hidden, 2, lr, seed + 100)   # z -> xhat

    def elbo(self, X, rng, K=1):
        """Monte-Carlo ELBO per point: E_q[log p(x|z)] - KL(q || N(0,1))."""
        EL = np.zeros(len(X))
        mv, _ = self.enc.fwd(X)
        mu, logv = mv[:, :1], np.clip(mv[:, 1:], -8, 4)
        v = np.exp(logv)
        for _ in range(K):
            eps = rng.standard_normal(mu.shape)
            z = mu + np.sqrt(v) * eps
            xh, _ = self.dec.fwd(z)
            rec = LOGC - ((X - xh) ** 2).sum(axis=1) / (2 * SIG_X ** 2)
            kl = 0.5 * (mu ** 2 + v - logv - 1).sum(axis=1)
            EL += rec - kl
        return EL / K

    def step(self, X, w, rng):
        """One weighted Adam step on -ELBO (one reparameterised sample)."""
        mv, He = self.enc.fwd(X)
        mu, logv = mv[:, :1], np.clip(mv[:, 1:], -8, 4)
        v = np.exp(logv)
        eps = np.random.default_rng(
            rng.integers(1 << 31)).standard_normal(mu.shape)
        z = mu + np.sqrt(v) * eps
        xh, Hd = self.dec.fwd(z)
        dxh = (xh - X) / SIG_X ** 2                    # d(-rec)/dxhat
        gdec, dz = self.dec.grads(z, Hd, dxh, w)
        dmu = dz + mu                                  # + dKL/dmu
        dlogv = dz * eps * np.sqrt(v) / 2 + 0.5 * (v - 1)
        dmv = np.concatenate([dmu, dlogv], axis=1)
        genc, _ = self.enc.grads(X, He, dmv, w)
        self.dec.adam(gdec)
        self.enc.adam(genc)


def movae_ssl(X, li, yl, epochs=EPOCHS, seed=0):
    """EM-style training of a two-component mixture of VAEs."""
    rng = np.random.default_rng(seed + 77)
    vaes = [VAE(seed * 13 + 1), VAE(seed * 13 + 2)]
    Ycl = np.eye(2)[yl]
    for ep in range(epochs):
        if ep < WARMUP:
            for c, v in enumerate(vaes):               # own labels only
                Xi = X[li][yl == c]
                v.step(Xi, np.full(len(Xi), 1.0 / len(Xi)), rng)
            continue
        # E-step: responsibilities from the ELBOs, labelled points clamped
        EL = np.stack([v.elbo(X, rng, K=4) for v in vaes], axis=1)
        R = np.exp(EL - EL.max(axis=1, keepdims=True))
        R /= R.sum(axis=1, keepdims=True)
        R[li] = Ycl
        W = R.copy()
        W[li] *= LAMBDA                                # the anchor
        # M-step: one weighted gradient step per component
        for c, v in enumerate(vaes):
            v.step(X, W[:, c] / W[:, c].sum(), rng)
    return vaes


def movae_logjoint(vaes, Xq, seed=0, K=16):
    rng = np.random.default_rng(seed + 999)
    return np.stack([v.elbo(Xq, rng, K=K) for v in vaes],
                    axis=1) + np.log(0.5)


def logsumexp_rows(L):
    m = L.max(axis=1, keepdims=True)
    return m[:, 0] + np.log(np.exp(L - m).sum(axis=1))


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_moons(250, noise=0.15, rng=rng)
    X_test, y_test = make_moons(250, noise=0.15, rng=rng)

    def draw_labels(draw):
        d_rng = np.random.default_rng(draw)
        return np.concatenate([
            d_rng.choice(np.where(y == 0)[0], 4, replace=False),
            d_rng.choice(np.where(y == 1)[0], 4, replace=False)])

    banner("DEMO 1 --- The wrong story: Gaussians on moons")
    print("  Part 4's semi-supervised mixture -- two full-covariance")
    print("  Gaussians, EM with the 8 labels clamped -- on data that is")
    print("  not Gaussian. All 500 unlabelled points join in.")
    print()
    gmm_accs, gmm_fits = [], []
    for draw in range(5):
        li = draw_labels(draw)
        mus, Ss, pis = gmm_ssl(X, li, y[li])
        gmm_fits.append((mus, Ss, pis))
        pred = gmm_logjoint(X_test, mus, Ss, pis).argmax(axis=1)
        gmm_accs.append(float((pred == y_test).mean()))
    print("    draw:  " + "   ".join(f"{a:5.1%}" for a in gmm_accs)
          + f"    mean {np.mean(gmm_accs):.1%}")
    print()
    print("  Better than the 77.4% supervised network of Part 8 -- even a")
    print("  wrong story extracts something from unlabelled data. But it is")
    print("  a CEILING: each Gaussian must drape one ellipse over a curved")
    print("  moon, and its tails claim the other class's horn. More EM")
    print("  iterations cannot fix it. The story is the limit.")

    banner("DEMO 2 --- The upgraded story: a tiny VAE per class")
    print("  Same EM loop, new p(x|class): a neural decoder bends a 1-D")
    print("  latent into a curve (a moon IS a curve plus noise). E-step:")
    print("  responsibilities from each VAE's ELBO, labels clamped. M-step:")
    print("  responsibility-weighted gradient steps. Each VAE warm-starts")
    print("  on its own 4 labelled points, so the story grows outward")
    print("  from the labels.")
    print()
    print("    draw   Gaussian story   VAE story")
    v_accs, vae_fits = [], []
    for draw in range(5):
        li = draw_labels(draw)
        vaes = movae_ssl(X, li, y[li], seed=draw)
        vae_fits.append(vaes)
        pred = movae_logjoint(vaes, X_test, seed=draw).argmax(axis=1)
        va = float((pred == y_test).mean())
        v_accs.append(va)
        print(f"      {draw}        {gmm_accs[draw]:6.1%}         {va:6.1%}")
    print(f"     mean       {np.mean(gmm_accs):6.1%}         "
          f"{np.mean(v_accs):6.1%}")
    print()
    mus, Ss, pis = gmm_fits[0]
    gmm_ll = float(logsumexp_rows(gmm_logjoint(X_test, mus, Ss, pis)).mean())
    vae_ll = float(logsumexp_rows(
        movae_logjoint(vae_fits[0], X_test, seed=0, K=64)).mean())
    print(f"  Mean test log-density (draw 0):")
    print(f"    Gaussian story : {gmm_ll:.2f}   (exact)")
    print(f"    VAE story      : {vae_ll:.2f}   (its ELBO lower bound)")
    print()
    print("  Every draw lands at 97.6% or better -- the best inductive")
    print("  scoreboard in this track, level with transductive label")
    print("  propagation. And the VAEs' LOWER BOUND on the log-density")
    print("  beats the Gaussians' exact value: the story is simply truer.")

    banner("DEMO 3 --- The payoff: sweep the latent, walk the moon")
    print("  A discriminative model ends at a boundary. A generative model")
    print("  can be played back. Decode z from -2 to +2 (draw 0's model)")
    print("  and measure each point's distance from its moon's arc centre")
    print("  (true arcs: radius 1.0, noise 0.15):")
    print()
    zs = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    centres = [np.array([0.0, 0.0]), np.array([1.0, 0.5])]
    for c, v in enumerate(vae_fits[0]):
        xh, _ = v.dec.fwd(zs)
        r = np.linalg.norm(xh - centres[c], axis=1)
        print(f"    class-{c} decoder:")
        for zi, (xx, yy), ri in zip(zs[:, 0], xh, r):
            print(f"      z = {zi:+.1f}  ->  ({xx:+.3f}, {yy:+.3f})"
                  f"   radius {ri:.3f}")
        print()
    print("  Each decoder traces its arc end to end at radius ~1: the 1-D")
    print("  latent became the arc parameter t, though no one ever showed")
    print("  the model (cos t, sin t). It over-reaches slightly at |z| = 2")
    print("  (radius 1.1-1.2) -- the prior's tails stretch past the horns.")
    print("  This is what 'modelling how the data was made' buys: the same")
    print("  object that classifies can also GENERATE the data it believes")
    print("  in -- and its mistakes are inspectable geometry, not opaque")
    print("  weights.")


if __name__ == "__main__":
    main()
