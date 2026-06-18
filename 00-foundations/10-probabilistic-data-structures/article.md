# Probabilistic Data Structures — Trading Exactness for Sublinear Memory

### *Algorithms in Python --- Foundations, Part 10*

---

A Python `set` answers two questions exactly: *is this item in the
collection?* and *how many distinct items have I seen?* It is fast,
clean, and I have used it ten thousand times. It also gets quietly
ruinous when the collection is a billion URLs you have already
crawled, or every event in a multi-terabyte log stream, or every
distinct user that has ever opened your app. The exact answer was
never the bottleneck. The memory was.

This article is about the data structures that solve that problem by
giving up the *exact* answer and giving you a *probabilistically
correct* one, in dramatically less memory. The trade is real and
explicit: a small, bounded chance of being wrong, in exchange for
sublinear or sometimes constant space regardless of how much data
you have seen. Nothing else gets you to the scale modern systems run
at.

There are three of these structures that every ML and infrastructure
engineer should know:

- **Bloom filters** — answer *"have I seen this item?"* with one-sided
  false positives. The structure that web crawlers, database query
  planners, and CDN caches use to avoid pointless work.
- **Count-Min Sketch** — answer *"how often has this item appeared?"*
  with bounded over-estimation. Behind every heavy-hitter detector in
  ad systems, network monitors, and recommendation pipelines.
- **HyperLogLog** — answer *"how many distinct items have I seen?"*
  with a few KB of memory regardless of stream size. The shape inside
  Redis `PFCOUNT`, BigQuery's `APPROX_COUNT_DISTINCT`, and Presto.

All three rely on the same trick: pass the item through one or more
hash functions and aggregate the *signatures* rather than the items
themselves. By the end of this article you will have built each one
in stdlib Python and measured the gap between theory and practice on
real-shaped workloads.

---

## The shape of the trade

Part 6 introduced hash tables and made one promise: O(1) lookup if
your hash distributes evenly. Probabilistic data structures keep that
promise but invert the goal. A hash table uses the hash to *find* the
item; a probabilistic structure uses the hash to *summarise* it,
throw the item away, and answer questions from the summary.

Concretely, every structure in this article has two parameters:

```
ε (epsilon)  — the accuracy you want
δ (delta)    — the probability that you exceed ε
```

The structure's memory grows logarithmically (or in some cases not at
all) with the *number of items*, but inversely with `ε` and `δ`. Want
twice the accuracy? Pay roughly twice the memory. Want a billion
items in your stream? Pay nothing extra.

That asymmetry is the whole reason these structures exist. A Python
`set` of one billion 64-bit integers needs ~24 GB of RAM. A
HyperLogLog estimating the same cardinality needs about 12 KB. The
HLL is wrong by ~1%, but for most decisions — *do we need to scale
the cluster?*, *which cohort is growing fastest?*, *is the dedup
ratio holding up?* — that 1% does not matter.

---

## Bloom filters — "have I seen this?"

A Bloom filter is a fixed-size **bit array** of `m` bits, plus `k`
independent hash functions. To insert an item, hash it `k` times,
take each hash modulo `m`, and set those `k` bits to 1. To check
membership, hash again and read those same `k` bits.

```
                  bit array of size m
            +---+---+---+---+---+---+---+---+---+---+
   insert  -> 0 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
            +---+---+---+---+---+---+---+---+---+---+
                ^           ^               ^
            h1(x)        h2(x)           h3(x)    (k=3)
```

If *any* of the `k` bits is 0, the item is **definitely not in the
set**. If *all* `k` bits are 1, the item is **probably in the set** —
but might be a false positive, because some other items' insertions
could have collectively set those same bits. Crucially, there are no
false negatives. If you have inserted an item, the filter will say
yes when asked.

That asymmetry is the entire point. *"Probably yes, definitely no"*
is exactly what you want for a cache pre-check, a "have I crawled
this URL" guard, or a "does the database row exist on disk" filter.
You only fall through to the expensive lookup when the Bloom says
yes; when it says no, you skip the work entirely.

### The math

For a Bloom filter with `m` bits, `k` hash functions, and `n`
inserted items, the false-positive rate is:

```
P(false positive) ≈ (1 - e^(-k·n/m))^k
```

This drops rapidly as `m` grows and is minimised when:

```
k = (m/n) · ln 2
```

A standard rule of thumb: for a target false-positive rate `p`, use
`m = -n·ln p / (ln 2)^2` bits and `k = (m/n)·ln 2` hash functions.
Plug in `p = 0.01` and `n = 1_000_000` and you get `m ≈ 9.6 million
bits` (about 1.2 MB) and `k ≈ 7`. One million elements with 1% false
positive in 1.2 MB of RAM. A `set` would need ~80 MB at minimum.

The companion script builds a Bloom filter for one million strings
and measures the actual false-positive rate against the theoretical
prediction:

```
Bloom filter --- m=9,585,059 bits (1.14 MB), k=7 hashes

Inserted 1,000,000 unique strings.
Querying 100,000 strings that were NOT inserted...

  False positives observed : 979
  False positive rate (FPR): 0.00979
  Theoretical FPR          : 0.01004
  Ratio (observed/theory)  : 0.98x
```

Observed and theoretical false-positive rates within 2% of each
other on a stream of a million items. The formula is doing exactly
what it claims.

### Where it shows up

**Database row existence checks.** Cassandra, RocksDB, LevelDB, and
HBase keep a Bloom filter per SSTable on disk. Before reading the
table to look up a key, the storage engine checks the Bloom filter
in memory. If it says "no," the disk read is skipped entirely. The
saved IOs are the difference between a system that scales to
petabytes and one that does not.

**Web crawlers.** Common Crawl and any production crawler keeps a
Bloom filter of URLs already fetched. A few hundred bits per URL
across a billion-URL frontier is comfortably in RAM; the equivalent
hash set is not.

**CDN cache pre-checks.** "Has this content been cached at the
edge?" gets asked billions of times per day per data centre. Bloom
filters in front of the cache layer answer "definitely no" cheaply
and skip the round-trip to the cache server.

**Spell checkers.** Older Unix `spell` used a Bloom-filter-like
structure (Bloom's original 1970 paper proposed exactly this use
case). Even today, many tokenisers and lexicon checks in NLP
pipelines use Bloom-style filters to do fast "is this a known word?"
membership tests over million-word vocabularies.

---

## Count-Min Sketch — "how often have I seen this?"

A Bloom filter answers a yes/no question. **Count-Min Sketch** (CMS)
generalises the same idea to counts. It is a `d × w` matrix of
counters, plus `d` independent hash functions, one per row.

```
            w columns
        +---+---+---+---+---+---+---+---+
  row 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  row 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  row 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |   d rows
  row 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
        +---+---+---+---+---+---+---+---+
```

To increment the count for an item, hash it `d` times — once per row
— and increment the cell at `(row, hash(item) mod w)` in each row.
To estimate the count of an item, hash it the same way and return
the **minimum** of the `d` cells.

The minimum is the magic. Each cell over-counts because of
collisions (other items' increments land in the same cell). But the
*minimum across rows* is the cell that suffered the fewest
collisions. So CMS gives you an estimate that is always ≥ the true
count and almost never much larger.

### The math

For width `w` and depth `d`, the estimate satisfies:

```
true_count ≤ estimate ≤ true_count + ε · N    with probability 1 - δ
```

where `N` is the total number of insertions, `ε = e/w`, and
`δ = e^(-d)`. Choosing `w = 2718` and `d = 5` gives you about 0.1%
over-estimation with probability 99.3% — five rows of ~2700
counters, total memory under 60 KB regardless of how many distinct
items you stream through.

The companion script streams a Zipfian distribution of word counts
through a CMS and shows that the top heavy hitters are recovered
exactly while the long tail is approximated cheaply:

```
Count-Min Sketch --- width=2718, depth=5 (~53 KB)

Streamed 100,000 words drawn from a Zipfian (alpha=1.2) over
a vocabulary of 5,000 distinct words.

Top-10 heavy hitters (true vs CMS-estimated):

  rank  word          true_count   cms_estimate   over_count
     1  the              21,496        21,496             0
     2  of                9,329         9,332             3
     3  and               5,724         5,725             1
     4  to                4,062         4,064             2
     5  in                2,967         2,967             0
     6  a                 2,504         2,504             0
     7  is                2,020         2,020             0
     8  that              1,782         1,782             0
     9  for               1,550         1,552             2
    10  it                1,409         1,409             0
```

On the heavy tail, CMS returns essentially exact counts because
collisions are diluted across thousands of cells. On the long tail,
estimates are slightly inflated, but for "find the heavy hitters"
queries — which is what CMS is optimised for — the rank ordering is
preserved.

### Where it shows up

**Heavy-hitters in ad systems.** Find the top-k advertisers, the
top-k publishers, the top-k creatives by impressions. CMS plus a
priority-queue of size `k` (Part 7's heap) gives you streaming
top-k in fixed memory.

**Network monitoring.** Detect heavy flows in a router that sees a
million packets per second. The sketch sits in the data plane,
tracks per-source-IP byte counts, and the control plane reads the
top hitters periodically.

**Recommendation systems.** Estimate item-item co-occurrence counts
for collaborative filtering when the item catalogue is too large to
materialise the full matrix. The same trick powers some of the
candidate-generation passes in Spotify's and YouTube's recommender
stacks.

**Online ML feature stats.** When training online models on streams
you often want recent feature frequencies — *"how often has this
user_id appeared in the last hour?"* CMS gives you that in fixed
memory regardless of how many user IDs exist, and several large
recommender systems use CMS for exactly this.

---

## HyperLogLog — "how many distinct items have I seen?"

The third question is the cardinality question: not *which* items
appeared, but *how many distinct ones*. Naively this needs O(n)
memory — store every distinct item — but **HyperLogLog** (HLL) gets
a few-percent estimate in O(log log n) bits. That is not a typo; the
"log log" in the name is the asymptotic memory complexity.

The intuition is shockingly simple. Hash every item to a uniformly
random 64-bit value and look at the **leading zeros**. If you have
seen many distinct items, you will have seen at least one that
hashed to a value with many leading zeros. Specifically, the
probability that a random hash starts with `k` zeros is `2^(-k)`.
So the maximum leading-zero count `L` you have observed gives you a
ballpark estimate of `2^L` for the cardinality.

That estimator is high-variance, so HLL splits the hash space into
`m` buckets ("registers"), tracks the maximum leading-zero count in
each, and combines them via a harmonic mean. The result is an
estimator whose **standard error** is approximately `1.04 / sqrt(m)`.
With `m = 16384` (16 kilobytes) you get about 0.8% error on
cardinalities up to billions of distinct items.

### The math

Each register is 5–6 bits (it stores a leading-zero count up to ~64).
With `m` registers the cardinality estimate is:

```
estimate ≈ α_m · m^2 / Σ(2^(-register_i))
```

where `α_m` is a precomputed bias-correction constant. The standard
error is:

```
σ ≈ 1.04 / sqrt(m)
```

So `m = 1024` gives ~3% error in ~700 bytes if registers are packed
at 5–6 bits each; `m = 16384` gives ~0.8% error in ~12 KB; `m =
65536` gives ~0.4% error in ~48 KB. The companion script uses one
byte per register for stdlib simplicity (so the demo footprint is
1 KB instead of the packed ~700 bytes), and trains it on a million
distinct strings:

```
HyperLogLog --- m=1024 registers (~1024 bytes), expected error ~3.25%

Streaming 1,000,000 distinct strings...

  True cardinality   : 1,000,000
  HLL estimate       :   972,188
  Error              :     27,812 (2.78%)
  Memory footprint   :     1.00 KB
```

A million distinct items estimated within 2.78% — comfortably
inside the theoretical ~3.25% standard-error bound — using one
kilobyte of memory. Scale the registers to 16 KB and you would
land within 1% on a billion-item stream. There is no other data
structure on the planet with that memory profile for this problem.

### Where it shows up

**Redis `PFCOUNT`.** Redis ships HLL as a first-class data type.
`PFADD` adds an item, `PFCOUNT` returns the cardinality estimate,
and `PFMERGE` unions two HLLs. Every analytics dashboard built on
Redis with a "unique users today" widget is reading from HLLs.

**BigQuery `APPROX_COUNT_DISTINCT` / Presto / Trino.** Distributed
query engines use HLL whenever you write `COUNT(DISTINCT x)` because
the exact answer is too expensive at the data sizes they handle.
The query planner usually rewrites it under a flag if the column
cardinality is large enough.

**Site analytics.** Google Analytics, Mixpanel, Amplitude, Plausible
— every product analytics tool that reports "monthly active users"
or "unique events" computes those numbers via HLL or a close
relative.

**ML feature engineering at scale.** Counting distinct values of
high-cardinality features (user IDs, ad IDs, search queries) is one
of the most common operations in feature engineering. HLL gives you
those counts in fixed memory, which makes streaming feature stores
tractable.

---

## Big-O and memory summary

[[BIG-O TABLE IMAGE]]

Three takeaways. **Bloom filters trade memory for false positives**:
~10 bits per item gets you a 1% FPR. **Count-Min Sketch trades width
for over-estimation error**: a few KB of counters delivers
heavy-hitter counts with bounded over-counting. **HyperLogLog trades
register count for cardinality precision**: 12 KB gets you ~1%
accuracy on billion-item streams. All three are O(1) per insert and
per query, regardless of how much data has flowed through.

The thing that makes these structures *useful* rather than just
clever is that the parameters are knobs you tune to your tolerance.
You decide what error rate is acceptable, and the structure tells you
exactly how much memory that costs. Most exact algorithms do not give
you that knob.

---

## Real-world ML and AI connections

These structures show up in surprising places once you start looking.

**Feature hashing in ML.** scikit-learn's `HashingVectorizer` and
Vowpal Wabbit's feature handling both use a hashing trick that is
algorithmically a Bloom-filter-style insertion: hash the feature
name, mod by a vocabulary size, accumulate values at that index. The
collisions are noise, but with a large enough vocabulary the noise
washes out and you avoid storing an explicit token-to-index map. The
exact same trick is what made the original "feature hashing for ad
prediction at scale" papers practical.

**Locality-Sensitive Hashing (LSH) for ANN search.** LSH is the
probabilistic cousin of vector indexes — instead of building a graph
or KD-tree (Parts 8 and the upcoming Part 12), you hash vectors so
that similar vectors are likely to land in the same bucket. The
hash family is constructed so collisions are *features*, not bugs.
LSH is still used for billion-scale similarity search in MinHash
deduplication of training corpora, near-duplicate document detection
in news pipelines, and approximate nearest neighbour at the edge of
RAG systems.

**MinHash for set similarity at scale.** MinHash estimates the
Jaccard similarity between two sets by hashing each element and
keeping the minimum hash value across the set. Compare two MinHashes
and you get an estimate of the underlying sets' similarity, in
constant time regardless of set size. Used heavily in LLM
training-data deduplication: Common Crawl's near-duplicate filter is built
on MinHash, and so are the dedup steps in the C4, Pile, and RedPajama
corpora.

**Streaming dedup for LLM training data.** Bloom filters and MinHash
combined are how a 50 TB pre-training corpus gets deduplicated in a
single pass. Each document goes through MinHash to compute a
similarity signature, the signature is checked against a Bloom
filter of seen signatures, and duplicates are dropped. The whole
pipeline runs in fixed memory regardless of corpus size.

**Sum-trees and Count-Min in prioritised experience replay.** Part
7 noted that prioritised experience replay uses sum-trees for
weighted sampling. Some implementations also use Count-Min
sketches to estimate per-trajectory frequencies for importance
weighting, especially in distributed RL where the replay buffer is
sharded across workers and exact counts would require coordination.

**Rate limiting at scale.** Token-bucket rate limiters are exact;
sliding-window rate limiters at scale (think OpenAI's API limits,
Cloudflare's edge limits) often use Count-Min Sketches per
(user, time-window) so that the rate limiter itself doesn't become
the bottleneck. A few KB per node, a few hundred million decisions
per second.

**Cardinality-aware query planning.** Postgres and modern OLAP
engines use HLL-style sketches in their statistics collection. The
planner needs to estimate row counts for join ordering; HLL gives it
those counts after a single sample pass over the data, instead of
materialising a sort or distinct-aggregate.

---

## When NOT to use a probabilistic data structure

These structures are powerful, but they have sharp edges.

**When you need the exact answer.** Audit logs, billing, financial
reconciliation, anything that touches money or compliance — use
exact structures. A Bloom filter that says "yes" to a transaction
that never happened is a bug, not a feature.

**When the false positive (or over-estimate) propagates.** If your
downstream pipeline cannot tolerate occasional wrong answers, do not
introduce them upstream. The "small bounded error" property of these
structures is only useful if downstream code treats the answers as
estimates.

**When you cannot tune the parameters honestly.** Bloom filters
degrade catastrophically once you exceed the design `n`. Count-Min
becomes useless if `w` is much smaller than the vocabulary's
heavy-tailed range. HLL has a "small cardinality" regime where it
under-estimates and needs a switchover to linear counting. Each
structure has a sweet spot; be sure you are inside it.

**When the data fits in memory anyway.** A `set` of ten thousand
items is faster than a Bloom filter for the same operation and gives
you exact answers. Probabilistic structures shine at scales where
the exact equivalent does not fit; below that, the simpler tool
wins every time.

**When you cannot rotate or reset the structure.** Bloom filters
fill up monotonically — once a bit is set, it stays set. A
long-running Bloom filter eventually saturates. Production systems
either age out the filter (counting Bloom filters) or rotate
through a pool of filters periodically. Plan for this from the
start.

---

## What comes next

Ten foundations down, two to go. Part 11 is **Sparse Matrices** —
CSR, CSC, COO, and why the right representation when most of your
data is zero is a different shape entirely. The format that powers
TF-IDF, recommender matrices, GNN adjacency, and sparse
autoencoders, and where the dense-vs-sparse crossover actually
lives.

Then Part 12, **Vector Indexes (ANN)** — HNSW, IVF, PQ, and LSH.
The structures inside FAISS, Pinecone, Chroma, and every modern
RAG retriever, picking up where this article left off on
locality-sensitive hashing.

After Part 12 the foundations are complete and we move to
algorithms. Linear regression, then everything that builds on it.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**probabilistic_data_structures.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/10-probabilistic-data-structures/probabilistic_data_structures.py)

Run it with:

```bash
python probabilistic_data_structures.py
```

It finishes in a few seconds on a laptop — SHA-256 hashing of over a
million items dominates the runtime. The companion script
builds all three structures from stdlib only — Bloom filter on a
million items measuring observed vs theoretical false-positive rate,
Count-Min Sketch over a Zipfian word stream recovering top-10 heavy
hitters, and HyperLogLog estimating the cardinality of a million
distinct strings in under 1 KB of memory. The headline numbers are
worth pinning to the wall: **1.2 MB for a million-item set with 1%
FPR**, **54 KB for streaming top-k over an unbounded vocabulary**,
and **a few KB for billion-scale distinct counts**. Nothing else gets
you to those memory profiles.

---

*This is Part 10 of the Algorithms in Python series, Foundations track. The companion script `probabilistic_data_structures.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 9](https://grahamjroy.medium.com/knowledge-graphs-where-symbols-embeddings-and-rag-meet-9ff0b2502434) covered knowledge graphs. Part 11 will look at sparse matrices — the format underneath every TF-IDF, recommender, and GNN that ever ran at scale.*
