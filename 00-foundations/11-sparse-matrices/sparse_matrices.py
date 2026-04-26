"""
sparse_matrices.py --- companion code for "Sparse Matrices"
(Foundations, Part 11).

Three demos:
  1. Build COO, convert to CSR, compare to dense memory footprint.
  2. SpMV benchmark: sparse vs dense on the same matrix at the crossover.
  3. TF-IDF on a small corpus, returning a CSR matrix.

Dependencies: numpy, scipy --- the only non-stdlib packages in this
foundations track. Runs in well under a second.
"""

from collections import Counter
from time import perf_counter
import math
import re

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


SEPARATOR = "=" * 64


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Demo 1 --- COO -> CSR conversion + memory comparison
# ---------------------------------------------------------------------------

def demo_coo_to_csr() -> None:
    banner("DEMO 1 --- COO -> CSR + memory comparison")

    # The 4 x 5 matrix from the article body, built explicitly in COO.
    rows = np.array([0, 0, 1, 2, 2, 3])
    cols = np.array([0, 3, 2, 1, 4, 0])
    data = np.array([5, 7, 9, 2, 8, 1], dtype=np.float64)

    coo = coo_matrix((data, (rows, cols)), shape=(4, 5))
    print("COO matrix (built from row, col, data triples):")
    print(coo.toarray())
    print()

    csr = coo.tocsr()
    print("CSR layout:")
    print(f"  indptr  : {csr.indptr.tolist()}")
    print(f"  indices : {csr.indices.tolist()}")
    print(f"  data    : {csr.data.tolist()}")
    print()
    print('Row 2 lookup -- "non-zeros in row 2 live in indices[3:5]":')
    print(f"  indices[3:5] = {csr.indices[3:5].tolist()}")
    print(f"  data[3:5]    = {csr.data[3:5].tolist()}")
    print()

    # Memory comparison at scale: 50,000 x 50,000 matrix at 0.1% density
    m, n = 50_000, 50_000
    density = 0.001
    nnz = int(m * n * density)
    dense_bytes = m * n * 8
    sparse_bytes = nnz * (8 + 4) + (m + 1) * 4  # data(8) + indices(4) + indptr
    print(f"Storage at scale ({m:,} x {n:,}, density {density * 100:.1f}%):")
    print(f"  Dense  : {dense_bytes / (1024 ** 3):>10.2f} GB")
    print(f"  Sparse : {sparse_bytes / (1024 ** 2):>10.2f} MB")
    ratio = dense_bytes / sparse_bytes
    print(f"  Ratio  : {ratio:>10.0f}x smaller")


# ---------------------------------------------------------------------------
# Demo 2 --- SpMV benchmark
# ---------------------------------------------------------------------------

def make_random_sparse(m: int, n: int, density: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    nnz = int(m * n * density)
    rows = rng.integers(0, m, size=nnz)
    cols = rng.integers(0, n, size=nnz)
    data = rng.standard_normal(nnz)
    return coo_matrix((data, (rows, cols)), shape=(m, n)).tocsr()


def demo_spmv_benchmark() -> None:
    banner("DEMO 2 --- SpMV: sparse wins at low density")

    # Large sparse-only benchmark: dense would not fit
    m, n = 50_000, 50_000
    density = 0.001
    A_large = make_random_sparse(m, n, density, seed=1)
    x_large = np.random.default_rng(2).standard_normal(n)
    print(f"Large benchmark: {m:,} x {n:,} matrix, "
          f"{density * 100:.1f}% density ({A_large.nnz:,} nnz)")
    print(f"  Dense storage  : {m * n * 8 / (1024 ** 2):>10.2f} MB  "
          f"(would not fit in benchmark; skipped)")
    print(f"  Sparse storage : "
          f"{A_large.data.nbytes / (1024 ** 2) + A_large.indices.nbytes / (1024 ** 2) + A_large.indptr.nbytes / (1024 ** 2):>10.2f} MB")
    start = perf_counter()
    _ = A_large @ x_large
    elapsed = perf_counter() - start
    print(f"  Sparse SpMV    : {elapsed * 1000:>10.2f} ms")
    print()

    # Smaller comparison where dense actually fits
    m_s, n_s = 5_000, 5_000
    density_s = 0.01
    A_small = make_random_sparse(m_s, n_s, density_s, seed=3)
    A_small_dense = A_small.toarray()
    x_small = np.random.default_rng(4).standard_normal(n_s)

    print(f"Smaller benchmark: {m_s:,} x {n_s:,} matrix, "
          f"{density_s * 100:.1f}% density ({A_small.nnz:,} nnz)")
    print(f"  Dense storage  : "
          f"{A_small_dense.nbytes / (1024 ** 2):>10.2f} MB")
    print(f"  Sparse storage : "
          f"{(A_small.data.nbytes + A_small.indices.nbytes + A_small.indptr.nbytes) / (1024 ** 2):>10.2f} MB")
    print()

    # Time both -- average over a few runs to get a stable reading
    n_runs = 5

    start = perf_counter()
    for _ in range(n_runs):
        _ = A_small_dense @ x_small
    dense_t = (perf_counter() - start) / n_runs

    start = perf_counter()
    for _ in range(n_runs):
        _ = A_small @ x_small
    sparse_t = (perf_counter() - start) / n_runs

    print(f"  Dense  SpMV    : {dense_t * 1000:>10.2f} ms")
    print(f"  Sparse SpMV    : {sparse_t * 1000:>10.2f} ms")
    speedup = dense_t / sparse_t if sparse_t > 0 else float("inf")
    print(f"  Sparse speedup : {speedup:>10.1f}x")


# ---------------------------------------------------------------------------
# Demo 3 --- TF-IDF on a small corpus
# ---------------------------------------------------------------------------

DOCS = [
    "machine learning algorithms power modern artificial intelligence systems",
    "deep neural networks transform computer vision and natural language processing",
    "reinforcement learning trains agents through reward signals from the environment",
    "knowledge graphs connect entities through typed relations for symbolic reasoning",
    "consciousness studies explore the nature of subjective experience",
    "the hard problem asks why physical processes give rise to felt experience",
    "integrated information theory proposes a mathematical measure of consciousness",
    "the global workspace broadcasts information across cognitive modules",
    "higher order theories require thoughts about thoughts for conscious experience",
    "the attention schema models the brains own attention as a representation",
    "panpsychism holds that consciousness pervades the fundamental physical world",
    "the chinese room argues that symbol manipulation cannot produce understanding",
    "philosophical zombies challenge the link between behaviour and experience",
    "embodied cognition emphasises the role of the body in shaping thought",
    "predictive processing frames the brain as a hierarchical inference engine",
    "free energy minimisation underpins active inference and adaptive behaviour",
    "transformers use self attention to model long range dependencies in sequences",
    "convolutional neural networks excel at processing grid structured visual data",
    "recurrent networks maintain hidden state across temporal sequences",
    "graph neural networks aggregate features from neighbouring nodes",
    "diffusion models generate images by reversing a noising process step by step",
    "variational autoencoders learn compact latent representations of complex data",
    "generative adversarial networks pit a generator against a discriminator",
    "language models predict the next token given a prefix of text",
    "tokenisation splits raw text into subword units that vocabulary indexes",
    "embeddings map discrete symbols into dense continuous vector spaces",
    "softmax over logits produces a probability distribution across classes",
    "gradient descent minimises a loss function by following partial derivatives",
    "backpropagation computes gradients efficiently through the chain rule",
    "stochastic optimisation samples gradients from minibatches of training data",
    "regularisation penalises model complexity to improve generalisation",
    "cross validation estimates how well a model performs on unseen data",
    "bias and variance describe two sources of generalisation error",
    "overfitting occurs when a model memorises rather than learns the training data",
    "early stopping halts training when validation loss stops improving",
    "dropout randomly zeros activations to reduce co adaptation of neurons",
    "batch normalisation stabilises training by rescaling layer inputs",
    "residual connections allow gradients to flow through very deep networks",
    "attention weights tell the model where to focus across the input",
    "self attention computes pairwise relations within a single sequence",
    "positional encodings inject sequence order into otherwise permutation invariant attention",
    "multi head attention runs several attention computations in parallel",
    "encoder decoder architectures map an input sequence to an output sequence",
    "sequence to sequence models translate text from one language to another",
    "beam search keeps the top candidate prefixes during sequence generation",
    "temperature sampling adjusts the entropy of next token predictions",
    "retrieval augmented generation grounds language model outputs in external documents",
    "vector databases index dense embeddings for approximate nearest neighbour search",
    "hierarchical navigable small world graphs accelerate similarity search at scale",
    "locality sensitive hashing buckets similar vectors into the same hash bin",
    "product quantisation compresses embeddings into compact discrete codes",
    "inverted file indexes partition vectors into clusters for faster lookup",
    "fine tuning adapts a pretrained model to a downstream task on smaller data",
    "low rank adaptation injects trainable low rank updates into frozen weights",
    "reinforcement learning from human feedback aligns models with human preferences",
    "direct preference optimisation simplifies preference based language model training",
    "in context learning lets language models learn from examples in the prompt",
    "chain of thought prompts encourage models to reason step by step",
    "tool use lets language models call external functions during inference",
    "agents combine planning reasoning and tool calls into multi step workflows",
    "evaluation harnesses measure model capabilities across diverse benchmarks",
    "data deduplication removes near duplicates before training large language models",
    "curriculum learning orders training data from easy to hard examples",
    "knowledge distillation trains a small student model to mimic a large teacher",
    "model compression reduces parameter count for deployment on resource constrained devices",
    "quantisation lowers the numerical precision of weights without losing much accuracy",
    "pruning removes redundant weights or attention heads from a trained network",
    "matrix factorisation decomposes a user item matrix into latent factor matrices",
    "collaborative filtering recommends items based on similar users behaviour",
    "two tower retrieval encodes queries and items separately for efficient lookup",
    "sequential recommenders model user behaviour as ordered sequences of interactions",
    "k means clustering partitions data into compact spherical groups",
    "principal component analysis finds orthogonal directions of maximum variance",
    "singular value decomposition factors a matrix into rotations and a diagonal scaling",
    "non negative matrix factorisation discovers parts based representations",
    "spectral clustering uses graph laplacian eigenvectors to partition data",
    "density based clustering identifies clusters of arbitrary shape in feature space",
    "gaussian mixture models softly assign points to several elliptical clusters",
    "expectation maximisation iteratively fits latent variable probabilistic models",
    "variational inference approximates intractable posteriors with tractable distributions",
    "markov chain monte carlo samples from complex probability distributions",
    "gaussian processes model functions with a distribution over latent representations",
    "bayesian neural networks place distributions over weights to capture uncertainty",
    "monte carlo tree search balances exploration and exploitation in game trees",
    "q learning estimates expected returns for state action pairs",
    "policy gradient methods directly optimise a parameterised policy by following its gradient",
    "actor critic algorithms combine value estimation with policy optimisation",
    "proximal policy optimisation bounds the per step change to a stable update region",
    "soft actor critic adds an entropy bonus to encourage exploratory policies",
    "deep deterministic policy gradient handles continuous action reinforcement learning",
    "twin delayed networks reduce overestimation bias in value based methods",
    "experience replay decorrelates training samples by buffering past transitions",
    "prioritised replay samples transitions in proportion to their td error magnitude",
    "actor learner architectures distribute the learner and many actors across machines",
    "model based reinforcement learning learns a world model to plan in imagination",
    "offline reinforcement learning trains policies from previously collected experience",
    "self supervised learning derives training signals from the data itself",
    "contrastive learning pulls similar examples together and pushes dissimilar ones apart",
    "masked autoencoders learn representations by reconstructing hidden patches",
    "diffusion model training adds noise to clean data and learns to denoise",
    "decoder only transformers underpin modern large generative language models",
]


def tokenise(text: str):
    return re.findall(r"[a-z]+", text.lower())


def build_tfidf_csr(documents):
    """Build a TF-IDF matrix in CSR format from scratch."""
    # Build vocabulary
    vocab = {}
    for doc in documents:
        for tok in tokenise(doc):
            if tok not in vocab:
                vocab[tok] = len(vocab)

    n_docs = len(documents)
    n_vocab = len(vocab)

    # Compute document frequency for each term
    doc_freq = Counter()
    tokenised = []
    for doc in documents:
        toks = tokenise(doc)
        tokenised.append(toks)
        for tok in set(toks):
            doc_freq[tok] += 1

    # IDF
    idf = {term: math.log((1 + n_docs) / (1 + df)) + 1
           for term, df in doc_freq.items()}

    # Build COO triples for TF-IDF
    rows, cols, data = [], [], []
    for doc_id, toks in enumerate(tokenised):
        if not toks:
            continue
        counts = Counter(toks)
        n_terms = len(toks)
        # Compute the L2 norm of the TF-IDF row to L2-normalise it
        weights = {}
        for tok, c in counts.items():
            tf = c / n_terms
            weights[tok] = tf * idf[tok]
        norm = math.sqrt(sum(w * w for w in weights.values()))
        if norm == 0:
            continue
        for tok, w in weights.items():
            rows.append(doc_id)
            cols.append(vocab[tok])
            data.append(w / norm)

    coo = coo_matrix(
        (np.array(data), (np.array(rows), np.array(cols))),
        shape=(n_docs, n_vocab),
    )
    return coo.tocsr(), vocab


def demo_tfidf() -> None:
    banner("DEMO 3 --- TF-IDF as a sparse CSR matrix")

    matrix, vocab = build_tfidf_csr(DOCS)
    n_docs, n_vocab = matrix.shape
    nnz = matrix.nnz
    density = nnz / (n_docs * n_vocab)

    print(f"TF-IDF: {n_docs} documents over a {n_vocab:,}-word vocabulary")
    print()
    print(f"  Shape           : {matrix.shape}")
    print(f"  Non-zeros       : {nnz:,}")
    print(f"  Density         : {density * 100:.2f}%")

    sparse_bytes = (
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    )
    dense_bytes = n_docs * n_vocab * 8
    print(f"  Memory (sparse) : {sparse_bytes / 1024:>8.1f} KB")
    print(f"  Memory (dense)  : {dense_bytes / 1024:>8.1f} KB")
    print(f"  Ratio           : {dense_bytes / sparse_bytes:>8.1f}x smaller")
    print()

    # Show top-k weighted terms in the first document
    inv_vocab = {idx: term for term, idx in vocab.items()}
    row0 = matrix.getrow(0)
    indices = row0.indices
    weights = row0.data
    pairs = sorted(zip(weights, indices), reverse=True)[:5]
    print(f"  First document  : {DOCS[0]!r}")
    print(f"  Top-5 weighted terms in that document:")
    for w, idx in pairs:
        print(f"    {inv_vocab[idx]:<20} -> {w:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo_coo_to_csr()
    demo_spmv_benchmark()
    demo_tfidf()
    print()


if __name__ == "__main__":
    main()
