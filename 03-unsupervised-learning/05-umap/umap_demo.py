"""
umap_demo.py --- companion code for "UMAP"
(Unsupervised Learning, Part 5).

Four demos:
  1. UMAP on the digits dataset with timing + KNN-in-2D accuracy.
  2. scikit-learn TSNE for direct comparison on the same data.
  3. PCA for the linear-baseline comparison.
  4. UMAP's transform() method: fit on 80% of the data, embed
     the held-out 20%, score KNN accuracy on the test split.

Dependencies: numpy, scikit-learn, umap-learn.
Install with: pip install umap-learn
Runs in ~30 seconds.
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

import umap


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def knn_2d_accuracy(Y, labels, k=15):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Y, labels, cv=5)
    return float(scores.mean())


def knn_2d_holdout_accuracy(Y_train, y_train, Y_test, y_test, k=15):
    knn = KNeighborsClassifier(n_neighbors=k).fit(Y_train, y_train)
    return float((knn.predict(Y_test) == y_test).mean())


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_umap(X, y):
    banner("DEMO 1 --- UMAP (umap-learn) on the digits dataset")

    n_neighbors = 15
    min_dist = 0.1
    print(f"  Data shape          : {X.shape[0]} samples, "
          f"{X.shape[1]} features")
    print(f"  n_neighbors         : {n_neighbors}")
    print(f"  min_dist            : {min_dist}")

    t0 = time.perf_counter()
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=2,
                        random_state=RNG_SEED)
    Y = reducer.fit_transform(X)
    dt = time.perf_counter() - t0
    print(f"  Wall time (fit)     : {dt:.2f} s")

    acc = knn_2d_accuracy(Y, y, k=15)
    print(f"  KNN accuracy in 2D  : {acc:.3f}")


def demo_tsne(X, y):
    banner("DEMO 2 --- scikit-learn TSNE (Barnes-Hut) for comparison")

    perplexity = 30.0
    print(f"  Perplexity          : {perplexity}")

    t0 = time.perf_counter()
    tsne = TSNE(n_components=2, perplexity=perplexity,
                max_iter=1000, init="pca",
                random_state=RNG_SEED)
    Y = tsne.fit_transform(X)
    dt = time.perf_counter() - t0
    print(f"  Wall time (fit)     : {dt:.2f} s")

    acc = knn_2d_accuracy(Y, y, k=15)
    print(f"  KNN accuracy in 2D  : {acc:.3f}")


def demo_pca(X, y):
    banner("DEMO 3 --- scikit-learn PCA (2 components)")

    t0 = time.perf_counter()
    pca = PCA(n_components=2).fit(X)
    Y = pca.transform(X)
    dt = time.perf_counter() - t0

    print(f"  Cumulative variance explained : "
          f"{pca.explained_variance_ratio_.sum():.3f}")
    print(f"  Wall time (fit)               : {dt:.2f} s")

    acc = knn_2d_accuracy(Y, y, k=15)
    print(f"  KNN accuracy in 2D            : {acc:.3f}")


def demo_umap_transform(X, y):
    banner("DEMO 4 --- UMAP transform on held-out new points")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED, stratify=y
    )

    print(f"  Fit on {len(X_train)} train points; "
          f"transform {len(X_test)} held-out test points")

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        n_components=2,
                        random_state=RNG_SEED)
    Y_train = reducer.fit_transform(X_train)
    Y_test = reducer.transform(X_test)

    acc = knn_2d_holdout_accuracy(Y_train, y_train, Y_test, y_test)
    print(f"  KNN accuracy on held-out test set in 2D : {acc:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    digits = load_digits()
    X = digits.data
    y = digits.target
    demo_umap(X, y)
    demo_tsne(X, y)
    demo_pca(X, y)
    demo_umap_transform(X, y)
    print()


if __name__ == "__main__":
    main()
