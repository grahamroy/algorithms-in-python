"""
tsne.py --- companion code for "t-SNE"
(Unsupervised Learning, Part 4).

Three demos:
  1. From-scratch t-SNE on a 300-point subsample of digits
     (naive O(n^2) per iteration, kept small for speed).
  2. scikit-learn's Barnes-Hut t-SNE on the full digits
     dataset (1797 points).
  3. PCA on the same dataset for comparison, with a KNN-in-2D
     accuracy score that quantifies how cleanly each method
     preserves digit-class neighbourhoods.

Dependencies: numpy, scikit-learn. Runs in ~30 seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as SkTSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# From-scratch t-SNE
# ---------------------------------------------------------------------------

def _hbeta(D_row, beta):
    """Compute conditional p-values and their entropy for one row,
    given precision beta = 1 / (2 sigma^2)."""
    P = np.exp(-D_row * beta)
    sumP = P.sum()
    if sumP < 1e-12:
        H = 0.0
        P = np.zeros_like(P)
    else:
        H = np.log(sumP) + beta * (D_row * P).sum() / sumP
        P = P / sumP
    return H, P


def _conditional_probs(X, perplexity=30.0, tol=1e-5, max_iter=50):
    """Compute conditional p_{j|i} matrix where each row has the
    requested perplexity. Binary search over beta for each row."""
    n = X.shape[0]
    # Squared Euclidean distances
    sum_X = (X ** 2).sum(axis=1)
    D = sum_X[:, None] + sum_X[None, :] - 2 * X @ X.T
    np.fill_diagonal(D, 0.0)

    P = np.zeros((n, n))
    log_perp = np.log(perplexity)

    for i in range(n):
        beta_min, beta_max = -np.inf, np.inf
        beta = 1.0
        # Take only the distances from i to j != i
        Di = np.delete(D[i], i)
        for _ in range(max_iter):
            H, Pi = _hbeta(Di, beta)
            H_diff = H - log_perp
            if abs(H_diff) < tol:
                break
            if H_diff > 0:
                beta_min = beta
                beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2
        # Insert zero at i and store
        P_row = np.zeros(n)
        P_row[np.arange(n) != i] = Pi
        P[i] = P_row
    return P


def tsne_from_scratch(X, n_components=2, perplexity=30.0,
                      n_iter=600, lr=200.0, momentum=0.5,
                      early_exaggeration=4.0, ee_iters=100,
                      seed=RNG_SEED, verbose=True):
    """Minimal from-scratch t-SNE.

    Naive O(n^2) per iteration. Use scikit-learn's TSNE for
    anything bigger than a few hundred points.
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    # P-values
    P_cond = _conditional_probs(X, perplexity=perplexity)
    P = (P_cond + P_cond.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    if early_exaggeration > 1.0:
        P *= early_exaggeration

    # Low-dimensional init
    Y = 0.0001 * rng.standard_normal((n, n_components))
    dY_prev = np.zeros_like(Y)

    for it in range(n_iter):
        # Pairwise low-d squared distances
        sum_Y = (Y ** 2).sum(axis=1)
        D_low = sum_Y[:, None] + sum_Y[None, :] - 2 * Y @ Y.T
        # q_{ij} numerator and Q matrix
        num = 1.0 / (1.0 + D_low)
        np.fill_diagonal(num, 0.0)
        Q = num / num.sum()
        Q = np.maximum(Q, 1e-12)

        # Gradient
        PQ = (P - Q) * num
        grad = 4.0 * ((np.diag(PQ.sum(axis=1)) - PQ) @ Y)

        dY = momentum * dY_prev - lr * grad
        Y = Y + dY
        # Re-centre
        Y = Y - Y.mean(axis=0)
        dY_prev = dY

        if it + 1 == ee_iters and early_exaggeration > 1.0:
            P = P / early_exaggeration
        if verbose and (it + 1) % 100 == 0:
            kl = (P * np.log(P / Q)).sum()
            print(f"    iter {it + 1:>4}  KL = {kl:.3f}")

    # Final KL
    sum_Y = (Y ** 2).sum(axis=1)
    D_low = sum_Y[:, None] + sum_Y[None, :] - 2 * Y @ Y.T
    num = 1.0 / (1.0 + D_low)
    np.fill_diagonal(num, 0.0)
    Q = num / num.sum()
    Q = np.maximum(Q, 1e-12)
    final_kl = float((P * np.log(P / Q)).sum())
    return Y, final_kl


# ---------------------------------------------------------------------------
# Helper: KNN-in-2D classification accuracy
# ---------------------------------------------------------------------------

def knn_2d_accuracy(Y, labels, k=15):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Y, labels, cv=5)
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_from_scratch(X, y):
    banner("DEMO 1 --- t-SNE from scratch on the digits dataset")

    # Subsample for speed (the naive implementation is O(n^2) per iter)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(X.shape[0], size=300, replace=False)
    Xs = X[idx]
    ys = y[idx]

    print(f"  Subsample for speed : {len(Xs)} points")
    print(f"  Perplexity          : 30")
    print(f"  Iterations          : 600")
    Y, kl = tsne_from_scratch(Xs, perplexity=30.0, n_iter=600,
                              verbose=True)
    acc = knn_2d_accuracy(Y, ys, k=15)
    print(f"  Final KL divergence : {kl:.3f}")
    print(f"  KNN accuracy in 2D  : {acc:.3f}  "
          f"(15-NN classifier on the embedding)")


def demo_sklearn(X, y):
    banner("DEMO 2 --- Same data (full 1797), scikit-learn TSNE "
           "(Barnes-Hut)")

    tsne = SkTSNE(n_components=2, perplexity=30.0,
                  max_iter=1000, init="pca",
                  random_state=RNG_SEED)
    Y = tsne.fit_transform(X)
    acc = knn_2d_accuracy(Y, y, k=15)
    print(f"  Perplexity          : 30")
    print(f"  Iterations          : 1000")
    print(f"  Final KL divergence : {tsne.kl_divergence_:.3f}")
    print(f"  KNN accuracy in 2D  : {acc:.3f}")


def demo_pca(X, y):
    banner("DEMO 3 --- Same data, PCA (2 components, from Part 3)")

    pca = PCA(n_components=2).fit(X)
    Y = pca.transform(X)
    acc = knn_2d_accuracy(Y, y, k=15)
    print(f"  Cumulative variance explained : "
          f"{pca.explained_variance_ratio_.sum():.3f}")
    print(f"  KNN accuracy in 2D            : {acc:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    digits = load_digits()
    X = digits.data
    y = digits.target
    demo_from_scratch(X, y)
    demo_sklearn(X, y)
    demo_pca(X, y)
    print()


if __name__ == "__main__":
    main()
