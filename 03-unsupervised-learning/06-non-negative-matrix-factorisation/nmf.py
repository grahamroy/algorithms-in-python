"""
nmf.py --- companion code for "Non-Negative Matrix Factorisation"
(Unsupervised Learning, Part 6).

Three demos:
  1. NMF from scratch (Lee & Seung multiplicative updates) on a
     4-category subset of the 20-newsgroups dataset. Display the
     top words per discovered topic.
  2. Comparison with scikit-learn's NMF (coordinate descent).
  3. Reconstruction error as a function of the number of
     components k.

Dependencies: numpy, scikit-learn. Runs in well under a minute.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF as SkNMF
from sklearn.feature_extraction.text import TfidfVectorizer


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CATEGORIES = [
    "rec.autos",
    "alt.atheism",
    "soc.religion.christian",
    "rec.sport.baseball",
]


def make_dataset():
    data = fetch_20newsgroups(
        subset="train", categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=RNG_SEED,
    )
    vec = TfidfVectorizer(max_df=0.7, min_df=2, stop_words="english",
                          max_features=1000)
    X = vec.fit_transform(data.data)
    return X.toarray(), vec.get_feature_names_out()


# ---------------------------------------------------------------------------
# NMF from scratch via Lee & Seung multiplicative updates
# ---------------------------------------------------------------------------

class NMF:
    """Non-negative Matrix Factorisation via Lee & Seung
    multiplicative updates on the squared Frobenius objective."""

    def __init__(self, n_components, n_iter=200, tol=1e-4,
                 random_state=RNG_SEED, eps=1e-12):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.eps = eps

    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = self.n_components
        rng = np.random.default_rng(self.random_state)

        # Random non-negative init scaled to the data
        scale = float(np.sqrt(X.mean() / k))
        W = rng.uniform(0.0, 2 * scale, size=(n, k))
        H = rng.uniform(0.0, 2 * scale, size=(k, d))

        prev_err = np.inf
        for it in range(self.n_iter):
            # Update H first, then W (Lee & Seung 2001)
            WtX = W.T @ X
            WtWH = W.T @ W @ H + self.eps
            H *= WtX / WtWH

            XHt = X @ H.T
            WHHt = W @ H @ H.T + self.eps
            W *= XHt / WHHt

            err = float(np.linalg.norm(X - W @ H, ord="fro"))
            if abs(prev_err - err) < self.tol:
                break
            prev_err = err

        self.components_ = H
        self.reconstruction_err_ = err
        return W


# ---------------------------------------------------------------------------
# Print top words per topic
# ---------------------------------------------------------------------------

def print_topics(H, feature_names, top_n=8, indent="    "):
    for i, comp in enumerate(H):
        top_idx = np.argsort(comp)[::-1][:top_n]
        words = " ".join(feature_names[j] for j in top_idx)
        print(f"{indent}Topic {i}: {words}")


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_from_scratch(X, feature_names):
    banner("DEMO 1 --- NMF from scratch on 20-newsgroups TF-IDF "
           "(4 categories)")

    k = 4
    print(f"  Data shape : {X.shape[0]} documents × {X.shape[1]} "
          f"vocabulary (TF-IDF)")
    print(f"  Components : {k}")
    print(f"  Iterations : 200 (multiplicative updates)")

    model = NMF(n_components=k, n_iter=200, random_state=RNG_SEED)
    W = model.fit_transform(X)
    print(f"  Final reconstruction error : "
          f"{model.reconstruction_err_:.2f} (Frobenius)")
    print(f"  Topic words (top 8 per component):")
    print_topics(model.components_, feature_names, top_n=8)


def demo_sklearn(X, feature_names):
    banner("DEMO 2 --- Same data, scikit-learn NMF (coordinate descent)")

    sk = SkNMF(n_components=4, init="nndsvd", solver="cd",
               max_iter=200, random_state=RNG_SEED)
    W = sk.fit_transform(X)
    print(f"  Iterations : 200")
    print(f"  Final reconstruction error : "
          f"{sk.reconstruction_err_:.2f} (Frobenius)")
    print(f"  Topic words (top 8 per component):")
    print_topics(sk.components_, feature_names, top_n=8)


def demo_k_sweep(X):
    banner("DEMO 3 --- How many topics? Reconstruction error vs k")

    print(f"  {'k':>4}   {'reconstruction error':>20}")
    print(f"  {'---':>4}   {'--------------------':>20}")
    for k in (2, 4, 6, 8, 10, 15, 20):
        model = SkNMF(n_components=k, init="nndsvd", solver="cd",
                      max_iter=200, random_state=RNG_SEED)
        model.fit(X)
        print(f"  {k:>4}   {model.reconstruction_err_:>20.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    X, feature_names = make_dataset()
    demo_from_scratch(X, feature_names)
    demo_sklearn(X, feature_names)
    demo_k_sweep(X)
    print()


if __name__ == "__main__":
    main()
