"""
em.py --- companion code for "Expectation-Maximisation (EM)"
(Semi-Supervised Learning, Part 4).

Parts 1-3 were discriminative: train a classifier, be careful what you feed
it. EM changes the worldview. Posit a GENERATIVE model -- a story of how the
data was produced: "each class is a Gaussian; a point is born by picking a
class, then sampling from its Gaussian." Under that story, the missing labels
are just MISSING DATA, and EM (Dempster, Laird & Rubin, 1977) is the classic
recipe for maximum likelihood with missing data:

    E-step: with the current parameters, compute each unlabelled point's
            RESPONSIBILITIES -- P(class | x), a soft, fractional membership.
            (Labelled points keep their known label: responsibility 1.)
    M-step: refit the parameters by weighted maximum likelihood, every
            point counting fractionally toward every class.

Repeat. The famous guarantee: each round can only INCREASE the observed-data
log-likelihood. The connection to this track: SELF-TRAINING (Part 1) is EM
with hard assignments -- promote your confident guesses and commit. EM never
commits: a point that is 58% class A counts as 0.58 of a point for A, forever
revisable.

Demonstrates:
  1. The setting: two overlapping Gaussian classes, 10 labels. Ten points
     cannot estimate two covariance matrices; 500 could.
  2. The loop, when the model is RIGHT: log-likelihood rises monotonically
     (the guarantee, visible), soft responsibilities in action, and 10
     labels + 490 unlabelled matching an all-labels oracle.
  3. The catch, when the model is WRONG (each class is really two clusters):
     the likelihood still rises every iteration -- and the accuracy is a
     coin flip, because the most likely story under a wrong model is not
     the true story. EM keeps its promise; the promise is about likelihood.

Everything (the generators, Gaussian likelihoods, the EM loop) is plain
NumPy. Dependencies: numpy. Runs in a few seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 1


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def make_blobs(n, rng):
    """Model-CORRECT world: one tilted Gaussian per class, overlapping."""
    y = np.arange(n) % 2
    rng.shuffle(y)
    mus = [np.array([-1.2, 0.0]), np.array([1.2, 0.0])]
    Ss = [np.array([[1.0, 0.6], [0.6, 1.0]]),
          np.array([[1.0, -0.6], [-0.6, 1.0]])]
    Ls = [np.linalg.cholesky(S) for S in Ss]
    X = np.stack([mus[c] + Ls[c] @ rng.standard_normal(2) for c in y])
    return X, y


def make_interleaved(n, rng):
    """Model-WRONG world: each class is itself TWO clusters, interleaved
    A, B, A, B along a line. No single Gaussian can say that."""
    centers = [(-3.0, 0), (-1.0, 1), (1.0, 0), (3.0, 1)]
    X = np.zeros((n, 2))
    y = np.zeros(n, dtype=int)
    for i in range(n):
        cx, cls = centers[rng.integers(4)]
        X[i] = [cx + rng.normal(0, 0.55), rng.normal(0, 0.55)]
        y[i] = cls
    return X, y


# ---------------------------------------------------------------------------
# Gaussian machinery
# ---------------------------------------------------------------------------

def gauss_logpdf(X, mu, S):
    d = X - mu
    Si = np.linalg.inv(S)
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * (np.einsum('ni,ij,nj->n', d, Si, d)
                   + logdet + 2 * np.log(2 * np.pi))


def fit_weighted(X, R, reg=1e-4):
    """Weighted maximum likelihood: the M-step. R[:, c] = how much of each
    point belongs to class c."""
    params = []
    for c in range(2):
        w = R[:, c]
        W = w.sum()
        mu = (w[:, None] * X).sum(0) / W
        d = X - mu
        S = ((w[:, None, None] * np.einsum('ni,nj->nij', d, d)).sum(0) / W
             + reg * np.eye(2))
        params.append((W / len(X), mu, S))
    return params


def posteriors(X, params):
    """The E-step: P(class | x) under the current parameters."""
    logp = np.stack([np.log(p) + gauss_logpdf(X, mu, S)
                     for p, mu, S in params], axis=1)
    m = logp.max(axis=1, keepdims=True)
    P = np.exp(logp - m)
    P /= P.sum(axis=1, keepdims=True)
    return P, logp


def observed_ll(X_lab, y_lab, X_unl, params):
    """The quantity EM guarantees to increase: labelled points contribute
    log p(x, y); unlabelled points contribute log p(x) = log sum_c p(x, c)."""
    _, logp_l = posteriors(X_lab, params)
    ll = logp_l[np.arange(len(y_lab)), y_lab].sum()
    _, logp_u = posteriors(X_unl, params)
    m = logp_u.max(axis=1)
    ll += (m + np.log(np.exp(logp_u - m[:, None]).sum(axis=1))).sum()
    return float(ll)


def accuracy(params, X, y):
    P, _ = posteriors(X, params)
    return float((P.argmax(axis=1) == y).mean())


# ---------------------------------------------------------------------------
# Semi-supervised EM
# ---------------------------------------------------------------------------

def semisup_em(X_lab, y_lab, X_unl, iters=30, X_test=None, y_test=None):
    R_lab = np.eye(2)[y_lab]
    params = fit_weighted(X_lab, R_lab, reg=1e-3)   # init from the labels
    history = []
    for it in range(1, iters + 1):
        R_unl, _ = posteriors(X_unl, params)                       # E-step
        X = np.concatenate([X_lab, X_unl])
        R = np.concatenate([R_lab, R_unl])
        params = fit_weighted(X, R, reg=1e-4)                      # M-step
        history.append((it, observed_ll(X_lab, y_lab, X_unl, params),
                        accuracy(params, X_test, y_test)))
    return params, history


def split_labels(X, y, rng, n_per_class=5):
    lab = np.concatenate([
        rng.choice(np.where(y == 0)[0], n_per_class, replace=False),
        rng.choice(np.where(y == 1)[0], n_per_class, replace=False)])
    unl = np.ones(len(X), dtype=bool)
    unl[lab] = False
    return lab, np.where(unl)[0]


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_blobs(500, rng)
    Xt, yt = make_blobs(500, rng)
    lab, unl = split_labels(X, y, rng)

    banner("DEMO 1 --- The setting: a generative story and 10 labels")
    print("  The story: each class is one tilted Gaussian; the two overlap.")
    print("  10 points are labelled, 490 are not. A Gaussian per class needs a")
    print("  mean and a 2x2 covariance -- five labelled points per class is a")
    print("  hopeless budget for estimating a covariance matrix.")
    print()
    p_lab = fit_weighted(X[lab], np.eye(2)[y[lab]], reg=1e-3)
    p_orc = fit_weighted(X, np.eye(2)[y], reg=1e-4)
    print(f"  Gaussians fit to the 10 labels     : {accuracy(p_lab, Xt, yt):6.1%}"
          f" test accuracy")
    print(f"  Oracle: fit to all 500 true labels : {accuracy(p_orc, Xt, yt):6.1%}"
          f" test accuracy")

    banner("DEMO 2 --- The loop, when the model is right")
    print("  E-step: soft responsibilities for the 490 unlabelled points.")
    print("  M-step: refit both Gaussians with every point counting")
    print("  fractionally. The observed log-likelihood can only go up:")
    print()
    p_em, hist = semisup_em(X[lab], y[lab], X[unl], X_test=Xt, y_test=yt)
    print("    iter   log-likelihood   test accuracy")
    for it, ll, acc in hist[:6] + [hist[-1]]:
        print(f"     {it:3d}      {ll:9.1f}        {acc:6.1%}")
    print()
    R_unl, _ = posteriors(X[unl], p_em)
    border = np.argmin(np.abs(R_unl.max(axis=1) - 0.58))
    print(f"  A borderline unlabelled point: P(A) = {R_unl[border, 0]:.2f}, "
          f"P(B) = {R_unl[border, 1]:.2f}.")
    print("  Self-training would commit it wholesale; EM counts it as a")
    print(f"  fraction of a point for each class -- forever revisable.")
    print()
    print("  Across three datasets (10 labels each):")
    print()
    print("    seed   10 labels only   semi-sup EM   oracle (500 labels)")
    for sd in range(3):
        r2 = np.random.default_rng(sd)
        X2, y2 = make_blobs(500, r2)
        Xt2, yt2 = make_blobs(500, r2)
        lab2, unl2 = split_labels(X2, y2, r2)
        a_lab = accuracy(fit_weighted(X2[lab2], np.eye(2)[y2[lab2]], 1e-3),
                         Xt2, yt2)
        p2, _ = semisup_em(X2[lab2], y2[lab2], X2[unl2],
                           X_test=Xt2, y_test=yt2)
        a_em = accuracy(p2, Xt2, yt2)
        a_orc = accuracy(fit_weighted(X2, np.eye(2)[y2], 1e-4), Xt2, yt2)
        print(f"      {sd}       {a_lab:6.1%}         {a_em:6.1%}        "
              f"{a_orc:6.1%}")
    print()
    print("  With the model right, 490 unlabelled points do the covariance")
    print("  estimation that 10 labels never could -- EM matches the oracle.")

    banner("DEMO 3 --- The catch: the most likely story can be wrong")
    print("  New world: each class is REALLY two clusters, interleaved")
    print("  A, B, A, B along a line. The one-Gaussian-per-class story cannot")
    print("  say that -- and the best two-Gaussian story is 'left vs right',")
    print("  which crosses BOTH classes. Same EM, same guarantee:")
    print()
    r3 = np.random.default_rng(0)
    X3, y3 = make_interleaved(500, r3)
    Xt3, yt3 = make_interleaved(500, r3)
    lab3, unl3 = split_labels(X3, y3, r3)
    p3, hist3 = semisup_em(X3[lab3], y3[lab3], X3[unl3],
                           X_test=Xt3, y_test=yt3, iters=40)
    print("    iter   log-likelihood   test accuracy")
    for it, ll, acc in hist3[:4] + [hist3[-1]]:
        print(f"     {it:3d}      {ll:9.1f}        {acc:6.1%}")
    print()
    p_true = fit_weighted(X3, np.eye(2)[y3], reg=1e-4)
    ll_em = observed_ll(X3[lab3], y3[lab3], X3[unl3], p3)
    ll_true = observed_ll(X3[lab3], y3[lab3], X3[unl3], p_true)
    print(f"  Log-likelihood of EM's fitted story      : {ll_em:9.1f}")
    print(f"  Log-likelihood of the TRUE class Gaussians: {ll_true:9.1f}")
    print()
    print("  The likelihood rose every single iteration, exactly as promised --")
    print("  to a coin-flip classifier. And EM's wrong story genuinely IS more")
    print("  likely than the truth under this model. EM did its job perfectly;")
    print("  the job was 'find the most likely story', and with a misspecified")
    print("  model the most likely story is not the true one.")


if __name__ == "__main__":
    main()
