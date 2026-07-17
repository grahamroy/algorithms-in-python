"""
multiview.py --- companion code for "Multiview Learning"
(Semi-Supervised Learning, Part 3).

Co-training (Part 2) USED two views to trade pseudo-labels. Multiview learning
promotes the views to the object of study: if two independent views of the
same example agree on something, that something is probably REAL -- so learn
the REPRESENTATION on which the views agree, using no labels at all, and let
the few labels you have work in that space.

The classical engine is Canonical Correlation Analysis (CCA, Hotelling 1936):
given paired views A and B, find projection directions w_A, w_B maximising

    corr( A @ w_A ,  B @ w_B )

-- the directions along which the two views move together. Correlation needs
no labels, so the entire unlabelled mountain trains the representation.

Why that helps: each view is full of view-SPECIFIC structure (nuisance --
lighting in one sensor, phrasing in one document field) that is often LOUDER
than the signal. Variance-based methods like PCA, run on one view, faithfully
keep the loudest directions -- the nuisance. But nuisance is view-specific:
it does not correlate ACROSS views. What correlates is what the views share.
Here, the only thing they share is the class.

Demonstrates (10-D views whose nuisance is 3x louder than the signal):
  1. The setting: with 10 labels, raw space and PCA space are near-chance --
     and even 600 labels in raw space are mediocre.
  2. CCA from unlabelled pairs alone finds the class direction (correlation
     0.95 with the hidden latent; PCA: 0.05) -- and 10 labels in that space
     BEAT 600 labels in the raw one.
  3. The catch: agreement is not relevance. Give both views a shared
     background louder than the class and CCA's TOP direction locks onto it
     -- truncate to one component and accuracy is a coin flip, while the
     class survives, demoted, in component 2.

Everything (the generator, CCA via whitening + SVD, PCA, k-NN) is plain
NumPy. Dependencies: numpy. Runs in a few seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Data: each example is a hidden class latent z (what the views share) plus,
# per view, LOUD view-specific nuisance structure. Optionally a shared
# nuisance u -- a "background" both views see -- for the failure demo.
# ---------------------------------------------------------------------------

def make_views(n, rng, shared_nuisance=0.0, d=10, k_nuis=4,
               sig=1.0, nuis=3.0):
    y = np.arange(n) % 2
    rng.shuffle(y)
    z = (2 * y - 1) + rng.normal(0, 0.4, n)      # the class latent (shared)
    u = rng.normal(0, 1, n)                      # a shared background latent
    views = []
    for _ in range(2):
        a_sig = rng.standard_normal(d)
        a_sig /= np.linalg.norm(a_sig)
        a_shn = rng.standard_normal(d)
        a_shn /= np.linalg.norm(a_shn)
        Q = rng.standard_normal((k_nuis, d))     # view-specific nuisance dirs
        N = rng.normal(0, 1, (n, k_nuis))
        V = (sig * z[:, None] * a_sig[None, :]
             + shared_nuisance * u[:, None] * a_shn[None, :]
             + nuis * N @ Q / np.sqrt(k_nuis)
             + rng.normal(0, 0.3, (n, d)))
        views.append(V)
    return views[0], views[1], y, z


# ---------------------------------------------------------------------------
# CCA from scratch: whiten each view, SVD the cross-covariance.
# Returns the projections and the canonical correlations.
# ---------------------------------------------------------------------------

def cca(A, B, k=2, reg=1e-3):
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    n = len(A)
    Caa = A.T @ A / n + reg * np.eye(A.shape[1])
    Cbb = B.T @ B / n + reg * np.eye(B.shape[1])
    Cab = A.T @ B / n

    def inv_sqrt(C):
        w, V = np.linalg.eigh(C)
        return V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-10))) @ V.T

    Wa_, Wb_ = inv_sqrt(Caa), inv_sqrt(Cbb)
    U, S, Vt = np.linalg.svd(Wa_ @ Cab @ Wb_)
    return Wa_ @ U[:, :k], Wb_ @ Vt[:k].T, S[:k]


def pca(A, k=2):
    A = A - A.mean(axis=0)
    w, V = np.linalg.eigh(A.T @ A / len(A))
    return V[:, ::-1][:, :k]


def knn_acc(X_train, y_train, X_query, y_query, k=3):
    X_train = np.atleast_2d(X_train)
    X_query = np.atleast_2d(X_query)
    d = ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :k]
    pred = (np.take(y_train, idx).mean(axis=1) > 0.5).astype(int)
    return float((pred == y_query).mean())


def z_alignment(projection, z):
    """|correlation| between a 1-D projection and the hidden class latent."""
    return float(abs(np.corrcoef(projection, z)[0, 1]))


def experiment(seed, shared_nuisance=0.0, n_per_class=5):
    rng = np.random.default_rng(seed)
    A, B, y, z = make_views(1000, rng, shared_nuisance=shared_nuisance)
    tr, te = slice(0, 600), slice(600, 1000)
    lab = np.concatenate([
        rng.choice(np.where(y[tr] == 0)[0], n_per_class, replace=False),
        rng.choice(np.where(y[tr] == 1)[0], n_per_class, replace=False)])
    return A, B, y, z, tr, te, lab


def main() -> None:
    A, B, y, z, tr, te, lab = experiment(RNG_SEED)

    banner("DEMO 1 --- The setting: loud views, quiet signal, 10 labels")
    print("  Every example has two 10-dimensional views. Each view contains the")
    print("  shared class signal -- and view-specific nuisance 3x LOUDER than it.")
    print("  600 example pairs are available; 10 of them are labelled.")
    print()
    raw = knn_acc(A[tr][lab], y[tr][lab], A[te], y[te])
    P = pca(A[tr], 2)
    pca_acc = knn_acc(A[tr][lab] @ P, y[tr][lab], A[te] @ P, y[te])
    oracle = knn_acc(A[tr], y[tr], A[te], y[te])
    print(f"  k-NN in raw view A (10-D), 10 labels : {raw:6.1%}")
    print(f"  k-NN in PCA-2 of view A,   10 labels : {pca_acc:6.1%}")
    print(f"  k-NN in raw view A, ALL 600 labels   : {oracle:6.1%}")
    print()
    print("  Ten labels are hopeless in the raw space, and PCA -- which keeps the")
    print("  LOUDEST directions -- faithfully keeps the nuisance. Even 600 labels")
    print("  cannot fully rescue a space where distance is mostly noise.")

    banner("DEMO 2 --- Agreement finds the signal: CCA from unlabelled pairs")
    print("  CCA sees only the 600 UNLABELLED pairs -- no labels anywhere -- and")
    print("  asks: along which directions do the two views move together?")
    print()
    Wa, Wb, S = cca(A[tr], B[tr], 2)
    print(f"  Canonical correlations found : {S[0]:.2f}, {S[1]:.2f}   "
          f"(one strong shared direction)")
    align_pca = z_alignment(A[te] @ P[:, 0], z[te])
    align_cca = z_alignment((A[te] @ Wa)[:, 0], z[te])
    print(f"  |corr| with the hidden class latent:  PCA-1 {align_pca:.2f}   "
          f"CCA-1 {align_cca:.2f}")
    print()
    print("  The shared direction IS the class. Now give the 10 labels the")
    print("  CCA space to work in, across three datasets:")
    print()
    print("    seed   10 labels, CCA space   600 labels, raw space")
    for sd in range(3):
        A2, B2, y2, z2, tr2, te2, lab2 = experiment(sd)
        Wa2, _, _ = cca(A2[tr2], B2[tr2], 2)
        cca_acc = knn_acc((A2[tr2] @ Wa2)[lab2], y2[tr2][lab2],
                          A2[te2] @ Wa2, y2[te2])
        orc = knn_acc(A2[tr2], y2[tr2], A2[te2], y2[te2])
        print(f"      {sd}          {cca_acc:6.1%}                {orc:6.1%}")
    print()
    print("  Ten labels in the RIGHT space beat six hundred in the wrong one.")
    print("  The unlabelled mountain did the heavy lifting -- it taught the")
    print("  representation; the labels only had to draw one line in it.")

    banner("DEMO 3 --- The catch: agreement is not relevance")
    print("  Now both views also share a BACKGROUND latent 4x louder than the")
    print("  class (same lighting, same season...). CCA still finds what the")
    print("  views share -- ranked by agreement strength:")
    print()
    print("    seed   corr 1   corr 2   |comp-1 vs class|   k=1 acc    k=2 acc")
    for sd in range(3):
        A3, B3, y3, z3, tr3, te3, lab3 = experiment(sd, shared_nuisance=4.0)
        Wa3, _, S3 = cca(A3[tr3], B3[tr3], 2)
        al = z_alignment((A3[te3] @ Wa3)[:, 0], z3[te3])
        acc1 = knn_acc((A3[tr3] @ Wa3)[lab3, :1], y3[tr3][lab3],
                       (A3[te3] @ Wa3)[:, :1], y3[te3])
        acc2 = knn_acc((A3[tr3] @ Wa3)[lab3], y3[tr3][lab3],
                       A3[te3] @ Wa3, y3[te3])
        print(f"      {sd}     {S3[0]:.2f}     {S3[1]:.2f}        {al:.2f}"
              f"             {acc1:6.1%}     {acc2:6.1%}")
    print()
    print("  Component 1 (corr 0.99) is now the BACKGROUND -- its alignment with")
    print("  the class is ~0. Keep only the top component and accuracy is a coin")
    print("  flip; the class survives, demoted to component 2. CCA ranks shared")
    print("  directions by agreement, not by usefulness -- keep enough components,")
    print("  and know what your views share besides the thing you care about.")


if __name__ == "__main__":
    main()
