"""
lda.py --- companion code for "Latent Dirichlet Allocation"
(Advanced Unsupervised Learning, Part 5).

Three demos:
  1. LDA fit on a 12-document toy corpus across 3 topics
     (cooking, machine learning, sports), reporting perplexity
     and convergence.
  2. Top words per inferred topic.
  3. Per-document topic mixture (the matrix theta).

Dependencies: numpy, scikit-learn. Runs in under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Toy corpus: 4 documents per topic, 3 topics
# ---------------------------------------------------------------------------

CORPUS = [
    # Cooking
    "recipe flour butter sugar egg mix dough bake oven cook recipe oven flour",
    "bake cake recipe oven flour butter egg sugar mix dough cook",
    "oven recipe flour cook butter egg sugar bake mix dough cake butter",
    "flour recipe butter sugar oven egg bake mix cook dough cake",
    # Machine learning
    "model train neural network layer accuracy dataset learning weight loss",
    "neural network train model layer loss accuracy weight dataset learning gradient",
    "model dataset train neural learning accuracy network layer weight loss",
    "train model neural network loss layer weight accuracy dataset learning gradient",
    # Sports
    "goal match team player score league ball win coach game",
    "team player match goal score league ball win coach game stadium",
    "match team goal player league score ball win coach game",
    "player goal team match score league ball coach win game stadium",
]
LABELS = ['cooking', 'cooking', 'cooking', 'cooking',
          'ml', 'ml', 'ml', 'ml',
          'sports', 'sports', 'sports', 'sports']
TOPIC_NAMES = ['cooking', 'ml', 'sports']


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_fit():
    banner("DEMO 1 --- LDA on a 12-document toy corpus")

    vec = CountVectorizer()
    X = vec.fit_transform(CORPUS)
    vocab = vec.get_feature_names_out()

    print(f"  Vocabulary size : {len(vocab)}")
    print(f"  Documents       : {X.shape[0]}")
    print(f"  Topics          : 3")
    print(f"  Method          : variational EM (sklearn)")

    lda = LatentDirichletAllocation(n_components=3,
                                    learning_method="batch",
                                    random_state=RNG_SEED,
                                    max_iter=100)
    lda.fit(X)
    # Per-document log-likelihood normalised
    perplexity = lda.perplexity(X)
    print(f"  Converged in    : {lda.n_iter_} iterations")
    print(f"  Perplexity      : {perplexity:.2f}")
    return lda, X, vocab


def demo_top_words(lda, vocab, top_n=10):
    banner("DEMO 2 --- Top words per topic")

    # Map inferred topics to canonical names by best-match per-doc
    # Skipped for the simple toy: just print topics in order.
    for k, comp in enumerate(lda.components_):
        top_idx = np.argsort(comp)[::-1][:top_n]
        top_words = ", ".join(vocab[i] for i in top_idx)
        print(f"  Topic {k}: {top_words}")


def demo_doc_mixtures(lda, X):
    banner("DEMO 3 --- Per-document topic mixture (rounded)")

    theta = lda.transform(X)
    # Determine which inferred topic best matches each true label
    # by averaging theta within each true class
    avg_by_label = {}
    for lbl in TOPIC_NAMES:
        idx = [i for i, l in enumerate(LABELS) if l == lbl]
        avg_by_label[lbl] = theta[idx].mean(axis=0)
    label_to_topic = {}
    used = set()
    for lbl in TOPIC_NAMES:
        order = np.argsort(-avg_by_label[lbl])
        for k in order:
            if int(k) not in used:
                label_to_topic[lbl] = int(k)
                used.add(int(k))
                break
    # Reorder columns
    col_order = [label_to_topic[name] for name in TOPIC_NAMES]
    theta_ord = theta[:, col_order]

    header = "  doc  " + "  ".join(f"{n:<8}" for n in TOPIC_NAMES) + \
             "inferred-label"
    print(header)
    print("  ---  " + "  ".join("-" * 8 for _ in TOPIC_NAMES) +
          "--------------")
    for i, row in enumerate(theta_ord):
        winner = TOPIC_NAMES[int(np.argmax(row))]
        row_str = "  ".join(f"{v:<8.2f}" for v in row)
        print(f"   {i:>2}  {row_str}{winner}")


def main() -> None:
    lda, X, vocab = demo_fit()
    demo_top_words(lda, vocab)
    demo_doc_mixtures(lda, X)
    print()


if __name__ == "__main__":
    main()
