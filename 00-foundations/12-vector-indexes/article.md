# Vector Indexes (ANN) — HNSW, IVF, PQ, and How RAG Actually Retrieves

### *Algorithms in Python --- Foundations, Part 12*

---

In Part 8 we built a KD-tree to find the nearest neighbour of a
query point in O(log n) — and noted in passing that it stops working
somewhere around 20 dimensions. Part 11 spent the whole article on
sparse matrices. This article is about the structure that picks up
where both leave off: how to find the nearest neighbour of a
**dense** vector in a corpus of millions or billions of dense
vectors, when the dimensionality is in the hundreds or thousands and
exact search is hopeless.

Every modern vector database — FAISS, Pinecone, Chroma, Weaviate,
LanceDB, Milvus — is built around three or four ideas in this
article. Every RAG pipeline uses one of them between the embedding
model and the LLM. Every "find similar images / songs / documents"
feature on a major platform is doing approximate nearest-neighbour
(ANN) search at the bottom of the stack. The three structures that
matter are **HNSW**, **IVF**, and **PQ**, plus **LSH** which we met
briefly in Part 10. Each one trades some exact recall for a dramatic
speedup, and modern production systems combine two or three of them
to get the latency, recall, and memory profile they actually need.

This is also the last foundation. After this article we leave the
data-structure layer and start building the algorithms that run on
top.

---

## The problem: nearest neighbour in high dimensions

Given a query vector `q` in `d` dimensions and a corpus of `N`
vectors, find the `k` nearest neighbours of `q` under some distance
or similarity metric — usually cosine similarity for embeddings, L2
distance for image features, or dot product for some recommender
embeddings.

Brute force is O(N · d) per query. For N = 1 million and d = 768
(a typical sentence-embedding dimension), that is 768 million
floating-point operations per query, or about 50 ms on a laptop.
Push N to a billion — the scale of any real document index — and
brute force becomes a few seconds per query. At even modest
throughput, that is unaffordable.

We covered KD-trees in Part 8 as the classic exact-NN structure.
They work beautifully up to about 20 dimensions and degrade to
brute force above ~50, because the recursive partitioning stops
pruning the search space as the curse of dimensionality kicks in.
Any `d > 50` problem needs **approximate** NN search — give up
strict exactness in exchange for sub-linear query time.

The metric you optimise is **recall@k**: of the true k nearest
neighbours, what fraction does your approximate search return? A
typical production target is recall@10 above 0.95 with query
latency under 10 ms on a billion-vector corpus. The structures in
this article are how that target gets met.

---

## The three families

Three families of algorithms dominate ANN search:

- **Graph-based** (HNSW, NSW): build a graph where each vector is
  connected to its nearest neighbours; search by greedy walk.
- **Partition-based** (IVF, IMI): cluster the vectors and search
  only the buckets nearest the query.
- **Quantisation-based** (PQ, OPQ, ScaNN): compress the vectors so
  distance computations are cheap, often combined with one of the
  above.

LSH (Part 10) is a fourth family that has fallen out of favour for
dense vectors but is still used for set similarity (MinHash) and
for some streaming workloads. The three above are what you find in
production today.

We will look at each in turn, then how they combine.

---

## HNSW — the graph that walks itself to the answer

**Hierarchical Navigable Small World** (HNSW) is the highest-quality
algorithm for medium-scale dense ANN. It is the default in most
modern vector databases (Qdrant, Weaviate, Milvus) and the
engine inside FAISS's `IndexHNSWFlat`.

The structure is a *multi-layer* graph. Every vector lives at layer
0 (the bottom). A small random sample also lives at layer 1, a
smaller sample at layer 2, and so on — the layer count is
geometrically distributed, so most vectors are only at the bottom
and only a handful reach the top.

```
  Layer 2:    A . . . . . . F . . . . . . J          (sparse)
              |             |             |
  Layer 1:    A . . C . . . F . . . H . . J          (medium)
              |   |         |   |   |     |
  Layer 0:    A B C D E F G H I J K L M N O          (everyone)
```

Within each layer, every vector has edges to a few of its nearest
neighbours *at that layer*. Vertical edges connect a vector to its
copies in the layers above (when they exist).

**Search.** Start at a fixed entry point at the top layer. Greedily
walk to the neighbour closest to the query, repeat until no
neighbour is closer than the current vertex. Drop down one layer
and repeat. Continue until you reach layer 0, where you do a final
greedy walk to find the k best.

The clever bit: the top layer is so sparse that one greedy walk
covers a lot of distance in the embedding space cheaply. Each
layer below refines the location. By the time you hit layer 0, you
are already in a tiny neighbourhood and only need to walk a few
steps to find the true nearest neighbours.

**Insertion.** New vectors are assigned a layer geometrically (most
end up only at layer 0, some reach higher), then their neighbours
at each layer are found by the same greedy search and connected.
The construction is O(N · log N · M) where M is the per-layer
out-degree.

**Why it works.** HNSW is a "navigable small world" — the
neighbourhoods are short-range but the cross-layer edges create
shortcuts that make the graph diameter logarithmic. It is the
algorithmic equivalent of "six degrees of separation" applied to
embedding space.

**Memory.** Roughly `N · M · 8 bytes` for the adjacency lists, plus
the vectors themselves. For N = 1 M, d = 768, M = 32, that is about
3.3 GB of vectors plus 256 MB of graph — fits on a workstation,
delivers recall@10 above 0.99 at sub-millisecond latency.

The companion script implements a simplified single-layer NSW
(navigable small-world) graph on 50,000 clustered vectors and
shows the greedy search working end to end:

```
NSW (simplified, single layer): 50,000 vectors in 128 dimensions
  Build time              :  1.34 s   (one-off; sampled kNN graph)
  100 queries (greedy NSW):  525 ms   (average 5.3 ms / query)
  Recall@10               :  0.13
```

The recall is modest (a sampled NSW build cannot compete with
proper HNSW iterative insertion), but the *mechanics* are the
point: greedy walk on a small-world graph, no exhaustive scan.
Production HNSW with multiple layers, iterative insertion, and a
larger out-degree typically lands at recall@10 above 0.99 with
50–200× speedup over brute force at corpus sizes of millions or
more — well past where numpy's vectorised brute force stops being
competitive.

---

## IVF — partition first, search the buckets

**Inverted File Index** (IVF) is the partition-based approach. It
is simpler than HNSW and the workhorse inside FAISS for very large
corpora.

Run k-means on the vector corpus to find K cluster centroids (a
typical choice is `K = sqrt(N)`). Each vector is assigned to its
nearest centroid, and the index becomes a list of K buckets:
*"vectors closer to centroid 1," "vectors closer to centroid 2,"*
and so on. To search, find the `nprobe` centroids nearest to the
query and brute-force-search only the vectors in those buckets.

```
              centroid 0    centroid 1    centroid 2
                 *              *              *
            [v17, v83, ...]  [v3, v22, ...]  [v9, v41, ...]
            (5,200 vectors)  (4,800 vectors) (5,500 vectors)

  query Q --> nearest centroid: 1
              (with nprobe=2: also probe centroid 2)
              search ~10,000 vectors instead of all N
```

The trade-off is clear: small `nprobe` gives fast queries but lower
recall (the true neighbour might live in a bucket you didn't visit);
large `nprobe` gives high recall but pulls more vectors into the
brute-force scan. Production systems usually run `nprobe` between 8
and 64, depending on the recall target.

**Build cost.** O(N · K · d · iters) for the k-means clustering,
typically a few minutes for tens of millions of vectors. **Query
cost.** O(K · d) to find the nearest centroids plus O(nprobe · N/K
· d) to scan the buckets — significantly less than O(N · d) when
K is well-chosen.

The companion script builds an IVF index with K = 100 clusters
over a 50,000-vector clustered corpus (50 well-separated Gaussian
clusters in 128 dimensions, the kind of structure real embeddings
exhibit) and reports the recall-vs-latency curve:

```
IVF: 50,000 vectors, K=100 clusters
  Build time             :  0.46 s   (one-off k-means)
  Avg bucket size        :  500 vectors

  Sweeping nprobe (recall vs latency per query):
    nprobe   time/query   recall@10
       4       1.5 ms       0.66
      10       3.2 ms       1.00
      20       5.8 ms       1.00
      40      10.8 ms       1.00
```

At `nprobe=10`, IVF recovers the true nearest neighbours
*perfectly* on this clustered data — exactly the regime IVF was
designed for. The trade-off curve is the part to internalise: small
`nprobe` is fast and approximate; larger `nprobe` is slower and
exact. Production systems pick the point on this curve that hits
their latency and recall targets.

---

## PQ — compress the vectors so distance is cheap

**Product Quantisation** (PQ) is the third family — orthogonal to
HNSW and IVF, in that it compresses the vectors themselves and is
typically *combined* with one of the others.

Take each `d`-dimensional vector and split it into `m` subvectors
of dimension `d/m`. For each subvector, train a separate k-means
with 256 centroids. Now every subvector can be replaced by the
1-byte index of its nearest centroid, so the entire `d`-dimensional
vector compresses to `m` bytes — typically `8 bytes` for `d = 128`
or `16 bytes` for `d = 768`.

```
Original vector (768 dims, 4 bytes each = 3072 bytes)
  [x_0, x_1, ..., x_767]

Split into m=8 subvectors of 96 dims each:
  [sub_0 | sub_1 | sub_2 | sub_3 | sub_4 | sub_5 | sub_6 | sub_7]

Each subvector quantised to one of 256 codes:
  [37   |  192 |   8   |  101 |   77  |  201 |   3   |  155]
                                                                
Compressed to 8 bytes total --- a 384x reduction.
```

**Distance computation.** For a query vector, precompute the
distance from each query subvector to all 256 codes for that
subvector, giving an `m × 256` lookup table. Then estimating the
distance to any compressed corpus vector is `m` table lookups — no
floating-point arithmetic needed. For `m = 8` that is 8 lookups
per distance, vastly faster than the 768 multiply-adds of full
distance computation.

**The trade-off.** PQ is *lossy*. The recall@10 for PQ alone on a
million-vector corpus typically lands around 0.6-0.8, which is
unacceptable for most workloads. The standard production solution
is **IVF-PQ** or **HNSW-PQ**: use IVF or HNSW to narrow the
candidate set, then re-rank the candidates with PQ-approximated
distances. The combination delivers near-exact recall at a fraction
of the memory cost of storing the full vectors.

The companion script trains a PQ codebook on the same 50,000
clustered vectors and shows the memory savings:

```
PQ: m=8 subvectors x 256 codes (16-dim subspaces)
  Original corpus     : 24.4 MB     (128 dims x 4 bytes per vec)
  PQ codes            :  0.38 MB    (8 bytes per vec)
  Compression ratio   : ~48x

  100 queries with PQ distance lookup:
    Average latency   : 1.5 ms / query
    Recall@10         : 0.05
```

A 48× compression — but recall collapses on this small,
clustered-but-noisy corpus. That is exactly the point of PQ: it is
a *memory* tool, not a *recall* tool. The standard production
pattern is **IVF-PQ** or **HNSW-PQ**: use IVF or HNSW to narrow
the candidate set to a few hundred vectors, then use the PQ
distance approximation only to rerank those candidates, falling
back to full distance computation on the top few. The recall of
the candidate generator carries the system; PQ pays only for the
memory.

---

## LSH — covered in Part 10, mentioned for completeness

**Locality-Sensitive Hashing** is the fourth family and was the
state of the art for ANN throughout the 2000s. It uses hash
functions designed so that *similar* vectors collide more often
than dissimilar ones. The structure is a hash table; the search
looks up the query's hash bucket and brute-forces the candidates.

Modern dense-vector workloads have largely moved past LSH —
HNSW, IVF, and PQ deliver higher recall at lower memory cost on
the same hardware. LSH is still strong for **set similarity**
(MinHash, covered in Part 10) and for some streaming applications
where the index needs to absorb updates faster than HNSW can
handle them.

---

## Combining the three: how production vector search really looks

Real production vector indexes combine these primitives:

- **IVF-Flat**: IVF partitioning, full vectors stored per bucket.
  Highest recall, highest memory.
- **IVF-PQ**: IVF partitioning, PQ-compressed vectors per bucket.
  Lower memory, slight recall hit. The default for billion-scale
  FAISS indexes.
- **HNSW-Flat**: HNSW graph over full vectors. Highest quality at
  medium scale (millions of vectors). The default in Qdrant, Weaviate,
  and modern vector DBs.
- **HNSW-PQ**: HNSW graph with PQ-compressed vectors. Used when
  you need sub-millisecond queries on hundred-million-scale corpora
  and can tolerate a small recall hit.
- **ScaNN** (Google): IVF + a learned anisotropic quantiser. The
  state of the art on most academic benchmarks; the engine behind
  Google's enterprise vector search.

The right combination depends on **N, d, latency target, recall
target, and update frequency**. A startup serving a few hundred
thousand documents with rare updates can use HNSW-Flat in RAM. A
hyperscaler indexing a few billion documents needs IVF-PQ on disk
with aggressive sharding. The algorithms are the same; the
parameter choices and the storage layer differ.

---

## Big-O and recall summary

[[BIG-O TABLE IMAGE]]

Two takeaways. **HNSW gives the best recall/latency at medium
scale**, at the cost of memory (the graph itself) and slow
inserts. **IVF scales better to very large corpora**, at the cost
of lower recall unless you crank `nprobe`. **PQ is a memory tool**
that you stack on top of either, paying recall for compression.
LSH still has its niche in set similarity but is rarely the right
choice for dense vectors today.

---

## Real-world ML and AI connections

ANN search is the substrate under most "find me something similar"
features in modern AI.

**RAG retrieval.** Every retrieval-augmented generation pipeline
embeds the user's question, runs an ANN search over a vector
database, and feeds the top-k retrieved chunks to the LLM as
context. The vector database is HNSW or IVF-PQ. The embedding
model is something like `text-embedding-3-small` (OpenAI),
`embed-english-v3` (Cohere), or one of the open-source BGE / GTE
families. The full pipeline — embedding, ANN search, LLM context
construction — is the dominant pattern for grounding LLMs in
private data.

**Semantic search at scale.** Google, Bing, and every enterprise
search product runs ANN over learned embeddings as a parallel path
to keyword search. The scoring is a fusion of the two; the
embedding side handles "find documents about X" queries that
keyword matching misses.

**Image and video similarity.** Pinterest's visual search,
Google Lens, every "find similar products" feature on Amazon and
Etsy, and the duplicate-detection step in any photo library all
run ANN over learned image embeddings. Pinterest's PinSage uses
HNSW internally; Amazon's product graph uses IVF-PQ.

**Recommender candidate generation.** The first pass of a modern
recommender — generating a few hundred candidates from a catalogue
of millions of items — is an ANN over user and item embeddings.
The downstream ranker re-scores them with a heavier model. YouTube,
Netflix, Spotify, and Pinterest all run two-tower retrieval over
ANN indexes for this stage.

**Embedding cache for LLM serving.** Some inference-serving stacks
cache embeddings of frequently-seen prompts and use ANN to look
up cached responses for near-duplicate queries. The trick is the
same as deduplication — ANN tells you "have I seen this before,
approximately?" — and shaves real money off the inference bill.

**Anomaly detection.** Network monitoring, fraud detection, and
some classes of ML monitoring use ANN to flag points that have no
near neighbours in a corpus of "normal" examples. The structures
are the same, the use case is inverted: instead of "find me what
is closest," it is "find me what is far from everything."

**Code search.** GitHub Copilot, Sourcegraph, and several IDE
plugins embed code snippets and run ANN over the embedding space
to surface similar implementations across a codebase. The
"similar code" suggestions you get in modern IDEs are HNSW
queries underneath.

---

## When NOT to use ANN

ANN is the right tool for "find approximately similar things at
scale." It is the wrong tool when:

**The corpus is small.** Below ~10,000 vectors, brute force in
numpy is faster than any ANN structure once you account for
construction cost. Use brute force, profile, only switch when
profiling tells you to.

**You need exact answers.** Database joins, audit-grade
deduplication, anything where a missed neighbour is unacceptable —
use exact NN. ANN is for the recall-tolerant 99% of similarity
problems, not the strict 1%.

**The vectors are sparse.** Sparse vectors (Part 11) have their
own indexing world — inverted indexes, BM25, the classical IR
stack. ANN is for dense embeddings. Mixing the two requires care.

**The data updates faster than you can rebuild.** HNSW is
incremental but slow to insert; IVF needs the centroids
re-trained periodically as the distribution shifts. If your
corpus is a fast-moving stream (real-time event ingest, live chat
logs), consider a hybrid where the bulk of the corpus uses ANN
and the recent tail uses brute force, then merge.

**You cannot tune the parameters honestly.** Every ANN structure
has knobs — `M` and `efConstruction` for HNSW, `K` and `nprobe`
for IVF, `m` and codebook size for PQ — and the right values
depend on your data distribution. Defaults from a tutorial often
under-perform by 20-50%. Tune on your actual data.

---

## What comes next: foundations done

Twelve foundations down. Zero to go.

You now have, in twelve articles, the data structures that every
modern ML and AI system runs on:

- **Arrays, matrices, tensors** (Parts 1-3) — the numerical
  substrate.
- **Linked lists, graphs, hash tables, queues, trees, knowledge
  graphs** (Parts 4-9) — the classical structures that organise
  computation.
- **Probabilistic structures, sparse matrices, vector indexes**
  (Parts 10-12) — the modern shapes that let those structures
  scale to ML workloads.

The foundations track is complete. The next article begins the
**algorithms** track with **linear regression** — the simplest
supervised algorithm, the one that every other one builds on top
of, and a chance to revisit the matrix algebra of Parts 1-2 with
everything we now know about storage, sparsity, and indexing.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**vector_indexes.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/12-vector-indexes/vector_indexes.py)

Run it with:

```bash
python vector_indexes.py
```

It needs `numpy` — the same dependency we used in Parts 1-3 (Part 11
added `scipy`). The script implements brute-force kNN, simplified IVF, PQ,
and a single-layer NSW graph, all from scratch, and benchmarks
each on a 50,000-vector clustered corpus in 128 dimensions. The
takeaways worth pinning to the wall: **IVF achieves perfect
recall at moderate latency on clustered data**, **PQ gives ~48×
memory compression** (paired with IVF or HNSW in production),
and **the simplified single-layer NSW shows the greedy-walk
mechanic** that real HNSW exploits at scale. At this small corpus
size numpy's vectorised brute force is hard to beat on raw latency
— the structures' speed wins arrive at the millions-to-billions
vector scale where production vector databases actually live.

---

*This is Part 12 of the Algorithms in Python series, Foundations track — the final foundation. The companion script `vector_indexes.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 11](https://medium.com/@grahamjroy/sparse-matrices-when-most-of-your-data-is-zero-85cebc669d78) covered sparse matrices. The next article opens the algorithms track with linear regression — and revisits everything we have built about storage, sparsity, and indexing, now in service of a model that learns.*
