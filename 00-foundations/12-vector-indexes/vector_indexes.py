"""
vector_indexes.py --- companion code for "Vector Indexes (ANN)"
(Foundations, Part 12).

Four demos, all built from scratch in numpy:
  1. Brute-force kNN baseline (timing ground truth).
  2. IVF (Inverted File Index): k-means partitioning, search top
     `nprobe` clusters, compare recall + latency to brute force.
  3. PQ (Product Quantisation): split each vector into m subvectors,
     quantise each to 256 codes, look up distances via tables.
  4. NSW (simplified Navigable Small World, single-layer): build a
     k-nearest-neighbour graph, search by greedy descent.

Dependencies: numpy.  Runs in a few seconds on a laptop.
"""

from time import perf_counter
import math
import numpy as np


SEPARATOR = "=" * 64


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Shared corpus + brute-force kNN
# ---------------------------------------------------------------------------

N = 50_000          # corpus size
D = 128             # embedding dim
N_CLUSTERS = 50     # generate corpus from K well-separated Gaussian clusters
N_QUERIES = 100     # number of queries to time
K = 10              # k for recall@k
SEED = 7


def make_clustered_corpus(n, d, n_clusters, seed):
    """Generate vectors that look more like real embeddings: clustered around
    a moderate number of centres in d-dim space, with intra-cluster noise.
    This is what IVF / NSW / PQ are actually tuned for."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_clusters, d)).astype(np.float32) * 5.0
    assignments = rng.integers(0, n_clusters, size=n)
    noise = rng.standard_normal((n, d)).astype(np.float32) * 0.5
    return centres[assignments] + noise


CORPUS = make_clustered_corpus(N, D, N_CLUSTERS, SEED)
# Queries are drawn from the same distribution (a different sample of clusters)
QUERIES = make_clustered_corpus(N_QUERIES, D, N_CLUSTERS, SEED + 1)


def brute_force_knn(corpus, queries, k):
    """Exact kNN by L2 distance. Returns indices of k nearest per query."""
    # ||q - x||^2 = ||q||^2 - 2 q . x + ||x||^2
    # Compute squared distances using broadcasting
    q_norm = np.einsum("ij,ij->i", queries, queries)[:, None]
    c_norm = np.einsum("ij,ij->i", corpus, corpus)[None, :]
    dists = q_norm + c_norm - 2.0 * queries @ corpus.T
    # Argpartition for top-k (faster than full sort)
    top = np.argpartition(dists, k, axis=1)[:, :k]
    # Sort the top-k by actual distance for stable ordering
    row_idx = np.arange(top.shape[0])[:, None]
    sort_within = np.argsort(dists[row_idx, top], axis=1)
    return top[row_idx, sort_within]


def recall_at_k(approx_indices, true_indices, k):
    """Average fraction of true top-k recovered in approx top-k."""
    recalls = []
    for a, t in zip(approx_indices, true_indices):
        recalls.append(len(set(a[:k]) & set(t[:k])) / k)
    return float(np.mean(recalls))


# ---------------------------------------------------------------------------
# Demo 1 --- Brute-force baseline
# ---------------------------------------------------------------------------

def demo_brute_force():
    banner("DEMO 1 --- Brute-force kNN (the baseline)")

    print(f"Corpus: {N:,} vectors x {D} dims (float32)")
    print(f"        ~{CORPUS.nbytes / (1024 ** 2):.1f} MB")
    print(f"Queries: {N_QUERIES}")
    print()

    start = perf_counter()
    truth = brute_force_knn(CORPUS, QUERIES, K)
    elapsed = perf_counter() - start
    print(f"  Brute-force kNN time: {elapsed * 1000:>8.1f} ms total"
          f"  ({elapsed * 1000 / N_QUERIES:.1f} ms/query)")
    print(f"  This is the recall=1.0 baseline.")
    return truth


# ---------------------------------------------------------------------------
# Demo 2 --- IVF (k-means partitioning + bucket search)
# ---------------------------------------------------------------------------

def kmeans(data, k, n_iter=15, seed=0):
    """Lloyd's algorithm. Returns (centroids, labels)."""
    rng = np.random.default_rng(seed)
    # Initialise with k random points from the data
    init_idx = rng.choice(len(data), size=k, replace=False)
    centroids = data[init_idx].copy()
    labels = np.zeros(len(data), dtype=np.int32)
    for _ in range(n_iter):
        # Assign step: nearest centroid for each point
        d = (
            np.einsum("ij,ij->i", data, data)[:, None]
            + np.einsum("ij,ij->i", centroids, centroids)[None, :]
            - 2.0 * data @ centroids.T
        )
        labels = np.argmin(d, axis=1).astype(np.int32)
        # Update step: mean of assigned points
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = data[mask].mean(axis=0)
    return centroids, labels


def build_ivf(corpus, k_clusters, seed=0):
    centroids, labels = kmeans(corpus, k_clusters, n_iter=10, seed=seed)
    buckets = [np.where(labels == j)[0] for j in range(k_clusters)]
    return centroids, buckets


def ivf_search(query, centroids, buckets, corpus, nprobe, k):
    """Find nprobe nearest centroids, brute-force search those buckets."""
    # Distance to centroids
    d_cent = np.sum((centroids - query) ** 2, axis=1)
    nearest_centroids = np.argpartition(d_cent, nprobe)[:nprobe]
    # Concatenate the candidate indices
    candidates = np.concatenate([buckets[c] for c in nearest_centroids])
    if len(candidates) == 0:
        return np.zeros(k, dtype=np.int32)
    cand_vecs = corpus[candidates]
    d = np.sum((cand_vecs - query) ** 2, axis=1)
    if len(candidates) <= k:
        order = np.argsort(d)
        return candidates[order]
    top = np.argpartition(d, k)[:k]
    order = np.argsort(d[top])
    return candidates[top[order]]


def demo_ivf(truth):
    banner("DEMO 2 --- IVF (Inverted File Index)")

    k_clusters = 100
    nprobe = 10
    print(f"IVF: {N:,} vectors, K={k_clusters} clusters, nprobe={nprobe}")

    start = perf_counter()
    centroids, buckets = build_ivf(CORPUS, k_clusters, seed=1)
    build_time = perf_counter() - start
    print(f"  Build time: {build_time:.2f} s")
    print(f"  Avg bucket size: {N / k_clusters:.0f} vectors")
    print()

    start = perf_counter()
    results = np.array([
        ivf_search(q, centroids, buckets, CORPUS, nprobe, K)
        for q in QUERIES
    ])
    elapsed = perf_counter() - start
    rec = recall_at_k(results, truth, K)
    print(f"  {N_QUERIES} queries: {elapsed * 1000:>8.1f} ms total"
          f"  ({elapsed * 1000 / N_QUERIES:.1f} ms/query)")
    print(f"  Recall@{K}: {rec:.2f}")

    # Try a higher nprobe to show the recall/latency curve
    print()
    print("  Sweeping nprobe (recall vs latency):")
    print(f"    {'nprobe':<8} {'time (ms/q)':<15} {'recall@10':<10}")
    for np_val in [4, 10, 20, 40]:
        start = perf_counter()
        results = np.array([
            ivf_search(q, centroids, buckets, CORPUS, np_val, K)
            for q in QUERIES
        ])
        elapsed = perf_counter() - start
        rec = recall_at_k(results, truth, K)
        print(f"    {np_val:<8} {elapsed * 1000 / N_QUERIES:<15.2f} "
              f"{rec:<10.2f}")


# ---------------------------------------------------------------------------
# Demo 3 --- PQ (Product Quantisation)
# ---------------------------------------------------------------------------

def train_pq(corpus, m, k_codes=256, seed=0):
    """
    Train m sub-quantisers, each with k_codes centroids.
    Returns codebooks of shape (m, k_codes, sub_d).
    """
    n, d = corpus.shape
    assert d % m == 0
    sub_d = d // m
    codebooks = np.zeros((m, k_codes, sub_d), dtype=np.float32)
    for j in range(m):
        sub = corpus[:, j * sub_d:(j + 1) * sub_d]
        cents, _ = kmeans(sub, k_codes, n_iter=8, seed=seed + j)
        codebooks[j] = cents
    return codebooks


def encode_pq(corpus, codebooks):
    """Encode each vector to m bytes (one code per subvector)."""
    n, d = corpus.shape
    m = codebooks.shape[0]
    sub_d = d // m
    codes = np.zeros((n, m), dtype=np.uint8)
    for j in range(m):
        sub = corpus[:, j * sub_d:(j + 1) * sub_d]
        cents = codebooks[j]
        # Distance from each subvector to all 256 centroids
        d_sub = (
            np.einsum("ij,ij->i", sub, sub)[:, None]
            + np.einsum("ij,ij->i", cents, cents)[None, :]
            - 2.0 * sub @ cents.T
        )
        codes[:, j] = np.argmin(d_sub, axis=1).astype(np.uint8)
    return codes


def pq_search(query, codes, codebooks, k):
    """Estimate distance to every code via per-subspace lookup tables."""
    m = codebooks.shape[0]
    sub_d = codebooks.shape[2]
    # Lookup table: for each subspace, distance from query to each of 256 codes
    table = np.zeros((m, codebooks.shape[1]), dtype=np.float32)
    for j in range(m):
        q_sub = query[j * sub_d:(j + 1) * sub_d]
        diff = codebooks[j] - q_sub
        table[j] = np.sum(diff * diff, axis=1)
    # Estimated distance for each corpus vector: sum table[j, codes[i, j]]
    # Use advanced indexing for vectorised lookup
    n_corpus = codes.shape[0]
    dists = np.zeros(n_corpus, dtype=np.float32)
    for j in range(m):
        dists += table[j, codes[:, j]]
    top = np.argpartition(dists, k)[:k]
    order = np.argsort(dists[top])
    return top[order]


def demo_pq(truth):
    banner("DEMO 3 --- PQ (Product Quantisation)")

    m = 8                # 8 subvectors of 16 dims each
    k_codes = 256        # 1 byte per code
    print(f"PQ: m={m} subvectors x {k_codes} codes each "
          f"({D // m}-dim subspaces)")

    start = perf_counter()
    codebooks = train_pq(CORPUS, m, k_codes, seed=2)
    train_time = perf_counter() - start
    print(f"  Codebook training: {train_time:.2f} s")

    start = perf_counter()
    codes = encode_pq(CORPUS, codebooks)
    encode_time = perf_counter() - start
    print(f"  Encode all corpus : {encode_time:.2f} s")
    print()

    original_size = CORPUS.nbytes
    compressed_size = codes.nbytes + codebooks.nbytes
    print(f"  Original corpus : {original_size / (1024 ** 2):.1f} MB"
          f"  ({D} dims x 4 bytes = {D * 4} B/vec)")
    print(f"  PQ codes        : {codes.nbytes / (1024 ** 2):.2f} MB"
          f"  ({m} bytes/vec)")
    print(f"  + codebooks     : {codebooks.nbytes / 1024:.1f} KB"
          f"  ({m} x {k_codes} x {D // m} floats)")
    print(f"  Compression     : {original_size / compressed_size:.1f}x")
    print()

    start = perf_counter()
    results = np.array([pq_search(q, codes, codebooks, K) for q in QUERIES])
    elapsed = perf_counter() - start
    rec = recall_at_k(results, truth, K)
    print(f"  {N_QUERIES} queries: {elapsed * 1000:>8.1f} ms total"
          f"  ({elapsed * 1000 / N_QUERIES:.1f} ms/query)")
    print(f"  Recall@{K}: {rec:.2f}")
    print()
    print("  PQ alone trades a lot of recall for compression. In production,")
    print("  PQ is paired with IVF or HNSW for the candidate generation,")
    print("  then PQ-approximated distances rerank the candidates.")


# ---------------------------------------------------------------------------
# Demo 4 --- NSW (simplified Navigable Small World, single layer)
# ---------------------------------------------------------------------------

def build_nsw_graph(corpus, m_neighbours, seed=0):
    """
    Build a single-layer kNN graph by random sampling: for each node, pick
    sample_size random candidates and keep the m_neighbours nearest. Plus
    a couple of long-range random shortcuts per node so the graph stays
    globally navigable across distant clusters.

    This is much faster than full brute-force kNN per node (O(N * sample * D)
    instead of O(N^2 * D)) and gives a graph of comparable quality for ANN
    search.
    """
    n = len(corpus)
    rng = np.random.default_rng(seed)
    sample_size = m_neighbours * 6
    neighbours = []
    for i in range(n):
        candidates = rng.choice(n, size=sample_size, replace=False)
        candidates = candidates[candidates != i][:sample_size - 1]
        diffs = corpus[candidates] - corpus[i]
        d = np.einsum("ij,ij->i", diffs, diffs)
        nearest = candidates[np.argpartition(d, m_neighbours)[:m_neighbours]]
        nbrs = list(nearest.tolist())
        # Add 2 random long-range shortcuts
        for _ in range(2):
            j = int(rng.integers(0, n))
            if j != i and j not in nbrs:
                nbrs.append(j)
        neighbours.append(nbrs)
    return neighbours


def nsw_search(query, corpus, graph, k, ef_search=64, seed=0):
    """Greedy walk with a candidate beam of size ef_search. Returns top-k."""
    rng = np.random.default_rng(seed)
    n = len(corpus)
    # Start from a random entry point
    entry = int(rng.integers(0, n))
    visited = {entry}
    # Distance helper
    def dist(idx):
        diff = corpus[idx] - query
        return float(np.sum(diff * diff))
    # Beam of (distance, node_idx)
    candidates = [(dist(entry), entry)]
    best = list(candidates)

    while candidates:
        candidates.sort()
        d_curr, curr = candidates.pop(0)
        # Stop if the closest candidate is worse than the worst in best
        if best and d_curr > max(b[0] for b in best) and len(best) >= ef_search:
            break
        for nb in graph[curr]:
            if nb in visited:
                continue
            visited.add(nb)
            d_nb = dist(nb)
            candidates.append((d_nb, nb))
            best.append((d_nb, nb))
            if len(best) > ef_search:
                best.sort()
                best = best[:ef_search]
    best.sort()
    return np.array([idx for _, idx in best[:k]])


def demo_nsw(truth):
    banner("DEMO 4 --- NSW (simplified Navigable Small World)")

    m_neighbours = 16
    print(f"NSW: {N:,} vectors, single layer, M={m_neighbours} neighbours/node")

    start = perf_counter()
    graph = build_nsw_graph(CORPUS, m_neighbours, seed=3)
    build_time = perf_counter() - start
    print(f"  Build time (brute-force kNN per node): {build_time:.2f} s"
          f"  (one-off)")
    print()

    start = perf_counter()
    results = np.array([
        nsw_search(q, CORPUS, graph, K, ef_search=64, seed=4 + i)
        for i, q in enumerate(QUERIES)
    ])
    elapsed = perf_counter() - start
    rec = recall_at_k(results, truth, K)
    print(f"  {N_QUERIES} queries (greedy NSW): {elapsed * 1000:>8.1f} ms total"
          f"  ({elapsed * 1000 / N_QUERIES:.1f} ms/query)")
    print(f"  Recall@{K}: {rec:.2f}")
    print()
    print("  Production HNSW with multiple layers and larger M typically")
    print("  delivers 50-200x speedup at recall@10 above 0.99.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    truth = demo_brute_force()
    demo_ivf(truth)
    demo_pq(truth)
    demo_nsw(truth)
    print()


if __name__ == "__main__":
    main()
