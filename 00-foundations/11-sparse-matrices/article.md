# Sparse Matrices — When Most of Your Data is Zero

### *Algorithms in Python --- Foundations, Part 11*

---

In Part 2 we treated matrices as dense rectangular arrays of numbers
— the mental model NumPy gives you and the one most linear algebra
courses teach. That model is correct, and it is also a trap. Most of
the matrices in modern ML are not dense. A typical TF-IDF matrix on
a million-document corpus has 99.99% zeros. A user-item recommender
matrix at any real platform is over 99.9% empty. A graph adjacency
matrix for a billion-node knowledge graph is so sparse the dense
representation does not fit on Earth's hard drives, let alone in
RAM.

This article is about the data structures that exploit that
sparsity. They are not exotic — they are the substrate of
scikit-learn's text features, every recommender system that ever
shipped, every graph neural network in PyTorch Geometric and DGL,
and the sparse-attention mechanisms that let modern transformers
handle long contexts without their memory exploding. Three formats
do most of the work — **COO**, **CSR**, and **CSC** — and knowing
which one fits which job is the difference between an ML pipeline
that runs and one that gets killed by the OOM reaper.

We will build one of each in `scipy.sparse`, time them against the
dense equivalent on a real benchmark, and end with a TF-IDF demo
showing why sparse is the default representation for any text
pipeline at scale.

---

## What "sparse" actually means

A matrix is **sparse** when most of its entries are zero. The
quantity that matters is the **number of non-zeros**, written
`nnz`. For an `m × n` matrix:

```
density = nnz / (m · n)
```

A matrix with `density < 0.01` (less than 1% non-zero) is
universally considered sparse. In practice the matrices that benefit
from sparse representations are far sparser than that — TF-IDF,
recommender, and graph matrices typically run at `density < 0.001`.

The dense storage cost is always `O(m · n)` regardless of how many
zeros you have. Sparse storage is `O(nnz)` — proportional only to
the *non*-zeros. For a million-document TF-IDF matrix with a
50,000-word vocabulary at 0.1% density, those numbers are:

```
Dense  : 1,000,000 × 50,000 × 8 bytes  =  400 GB
Sparse : 50,000,000 × ~16 bytes        =  ~800 MB
```

Three orders of magnitude. The dense version does not fit anywhere;
the sparse version fits on a phone. That gap is the entire reason
the formats in this article exist.

---

## COO — the format you build in

The simplest sparse format is **Coordinate List** (COO). Three
parallel arrays:

```
rows = [0, 0, 1, 2, 2, 3]     ← row index of each non-zero
cols = [0, 3, 2, 1, 4, 0]     ← column index of each non-zero
data = [5, 7, 9, 2, 8, 1]     ← the values
```

Together they describe the non-zero entries of a 4×5 matrix:

```
        col 0  col 1  col 2  col 3  col 4
row 0 [   5      .      .      7      .  ]
row 1 [   .      .      9      .      .  ]
row 2 [   .      2      .      .      8  ]
row 3 [   1      .      .      .      .  ]
```

Building a COO matrix is `O(1)` per non-zero — just append to the
three arrays. That makes COO the format you *construct* in: when
you are reading a sparse dataset out of a file, scraping pairs from
some upstream source, or accumulating triples from a stream, COO is
the right intermediate representation.

The catch is that *querying* a COO matrix is slow. To find all
non-zeros in row 2, you scan the entire `rows` array looking for
`2`. That is `O(nnz)` per row lookup, and useless for any operation
that walks the matrix structure repeatedly.

In `scipy.sparse`, COO is the format used to read in MatrixMarket
files, to add new entries in batch, and as the input to most "build
a sparse matrix" constructors. After construction you almost always
convert to CSR or CSC for the actual computation.

---

## CSR — the format you compute in

**Compressed Sparse Row** (CSR) is the workhorse. Three arrays again,
but laid out so that all non-zeros for a given row are contiguous in
memory.

```
indptr  = [0, 2, 3, 5, 6]     ← row i's non-zeros live in
                                indices[indptr[i] : indptr[i+1]]
indices = [0, 3, 2, 1, 4, 0]  ← column index of each non-zero
data    = [5, 7, 9, 2, 8, 1]  ← the values
```

The same 4×5 matrix encoded differently. `indptr[i]` is the start
position of row `i`'s entries in the `indices` and `data` arrays;
`indptr[i+1]` is where they end. Row 2's non-zeros are at positions
3 to 5, which gives `indices[3:5] = [1, 4]` and `data[3:5] = [2, 8]`
— column 1 has value 2, column 4 has value 8. The `indptr` array has
length `m + 1`, so the total storage is still `O(nnz + m)`.

CSR has two properties that make it the default for compute:

**Fast row slicing.** Getting all entries of row `i` is
`O(nnz_in_row_i)` — read a contiguous slice of `indices` and
`data`. No scan of the whole matrix.

**Fast sparse matrix-vector multiplication (SpMV).** The most
common operation on a sparse matrix is multiplying it by a dense
vector — `y = A @ x`. With CSR you walk the rows in order, and for
each row you read its non-zero columns from `indices`, look up the
corresponding entries of `x`, multiply by `data`, and sum. The
inner loop is cache-friendly because `indices` and `data` are
contiguous, and the only random-access pattern is the lookups into
`x` — and even those are bounded by `nnz_in_row` per row.

The companion script benchmarks SpMV on a 50,000 × 50,000 matrix
at 0.1% density (2.5 million non-zeros) against the dense
equivalent and shows where sparse wins:

```
Large benchmark: 50,000 x 50,000 matrix, 0.1% density (2,498,787 nnz)
  Dense storage  :   19,073.49 MB  (would not fit in benchmark; skipped)
  Sparse storage :       28.79 MB
  Sparse SpMV    :        3.38 ms

Smaller benchmark: 5,000 x 5,000 matrix, 1.0% density
  Dense storage  :      190.73 MB
  Sparse storage :        2.87 MB

  Dense  SpMV    :        3.08 ms
  Sparse SpMV    :        0.21 ms
  Sparse speedup :       14.5x
```

At 1% density on a small matrix the sparse version is already
fourteen times faster. At 0.1% density on a large matrix the dense
version is 19 GB — it simply does not exist in any practical sense
— and that is the regime real ML pipelines actually live in.

---

## CSC — the format you slice columns with

**Compressed Sparse Column** (CSC) is CSR's mirror image. Same
three-array layout, but indexed by column instead of row:

```
indptr  = [0, 2, 3, 4, 5, 6]  ← column j's non-zeros live in
                                indices[indptr[j] : indptr[j+1]]
indices = [0, 3, 2, 1, 0, 2]  ← row index of each non-zero
data    = [5, 1, 2, 9, 7, 8]  ← the values
```

CSC gives you fast column slicing and fast multiplication of
`A^T @ x` — the column-oriented mirror of CSR's row-oriented
strengths. It is the right format when your computation walks
columns more than rows: certain optimisation solvers, some
implementations of LASSO and coordinate descent, and a few
graph-partitioning algorithms.

The trade-off between CSR and CSC is the same as the trade-off
between row-major and column-major dense layouts: choose the one
whose dominant access pattern matches your algorithm. Most ML code
ends up using CSR because the dominant operation is "row of
features × weight vector" or "row of one-hot embedding × weight
matrix," both of which are row-walks.

`scipy.sparse` lets you convert between formats in linear time, so
you can build in COO and switch to CSR or CSC at the boundary
between construction and computation.

---

## Three other formats worth knowing

`scipy.sparse` ships seven formats. Three more are worth a passing
mention:

**LIL** (List of Lists) — one Python list per row of (column,
value) pairs. Easy to mutate one entry at a time, useful when you
are filling a sparse matrix incrementally with random-access writes.
Painfully slow for everything else; convert to CSR before any
computation.

**DOK** (Dictionary of Keys) — a `dict` keyed on `(row, col)`
tuples. Same use case as LIL — building a sparse matrix one cell at
a time when you do not know the structure in advance. Internally
just a Python dict, with all the overhead that entails.

**BSR** (Block Sparse Row) — CSR but each "non-zero" is a small
dense block instead of a single value. Useful when your sparsity
pattern has natural block structure, e.g. block-diagonal Hessians
in optimisation, or factor graphs in probabilistic ML. A niche but
beautiful structure when it fits.

In practice, 95% of ML code paths use CSR. A handful of
construction-heavy pipelines use COO or LIL as a staging ground.
CSC shows up in solvers. The other formats are tools for narrower
problems.

---

## TF-IDF — the canonical sparse matrix in ML

The classical use case for sparse matrices in ML is **TF-IDF
vectorisation** of text. Tokenise a corpus, compute term-frequency
(TF) and inverse-document-frequency (IDF), and store the result as
a `documents × vocabulary` matrix where each entry is the TF-IDF
weight of one term in one document.

For any non-trivial corpus, that matrix is overwhelmingly zero. A
1,000-document corpus over a 10,000-word vocabulary has 10 million
*possible* entries; if the average document contains 200 distinct
terms, only 200,000 of them are non-zero. That is 2% density, and
already worth a sparse representation. Real corpora — Wikipedia,
Common Crawl, an enterprise document store — typically run an order
of magnitude sparser.

The companion script builds a small TF-IDF matrix from scratch and
reports the density:

```
TF-IDF: 101 documents over a 570-word vocabulary

  Shape           : (101, 570)
  Non-zeros       : 974
  Density         : 1.69%
  Memory (sparse) :  11.8 KB
  Memory (dense)  : 449.8 KB
  Ratio           :  38.1x smaller
```

A 38× memory saving on a hundred-document toy corpus, and the
saving grows linearly with corpus size and vocabulary growth.
scikit-learn's `TfidfVectorizer` returns a CSR matrix by default
for exactly this reason — you can fit it on a million documents on
a laptop.

The same shape powers every recommender's user-item matrix (sparse
because most users have rated almost no items), every GNN's
adjacency matrix (sparse because most nodes are not directly
connected), and every sparse autoencoder's hidden-layer activations
(sparse by L1 regularisation on the activations).

---

## The dense-vs-sparse crossover

A natural question: at what density does sparse become the wrong
choice? The answer depends on the operation, but a useful rule of
thumb for SpMV is:

```
Sparse storage cost  ≈  3 × nnz × 8 bytes   (data + indices + indptr)
Dense storage cost   ≈  m × n × 8 bytes
```

Sparse is cheaper on memory whenever `3 · nnz < m · n`, i.e.
density below roughly 33%. For computation, the crossover is
narrower — sparse SpMV has more overhead per non-zero than dense
matrix-vector multiplication, so the dense version is faster up to
densities of about 1-10% depending on hardware. Below that, sparse
wins on both axes.

The practical heuristic: if your density is below 1%, use sparse.
If it is above 30%, use dense. Between those two, profile both on
your actual data and your actual operation, because the answer
depends on cache behaviour and BLAS implementation.

---

## Big-O and memory summary

[[BIG-O TABLE IMAGE]]

The thing to read off this table is the asymmetry. Construction is
fastest in COO and LIL (the formats designed for it); computation
is fastest in CSR and CSC (the formats designed for it). The
`scipy.sparse` library makes the conversion cheap, so the canonical
ML pipeline is *build in COO, convert to CSR, compute in CSR*. Any
attempt to compute directly on a COO or LIL matrix will be slow.

---

## Real-world ML and AI connections

Sparse matrices are the unsung backbone of a huge slice of the ML
stack.

**TF-IDF and bag-of-words.** Every classical NLP pipeline before
deep learning ran on a sparse term-document matrix. They still do,
inside any pipeline that uses TF-IDF as a baseline or as a feature
for a tree-based model. scikit-learn's `CountVectorizer`,
`HashingVectorizer`, and `TfidfVectorizer` all return CSR.

**Recommender systems.** The user-item rating matrix at Spotify,
Netflix, Amazon, Pinterest, or any platform with millions of users
and items is sparse — most users have rated a vanishingly small
fraction of available items. Matrix factorisation (SVD, ALS),
neural collaborative filtering, and modern two-tower retrieval all
start from a sparse rating matrix and learn dense embeddings on top.

**Graph adjacency matrices in GNNs.** PyTorch Geometric and DGL
represent graph adjacency as sparse matrices, because for any
graph with bounded degree the adjacency is overwhelmingly empty.
Message passing — the core operation of a GNN layer — is sparse
matrix multiplication: aggregate features from neighbours by
multiplying the sparse adjacency matrix by the dense feature matrix.
The whole field would be infeasible without sparse SpMV.

**Sparse attention in long-context transformers.** Vanilla
attention is `O(n²)` in sequence length, and that quadratic cost is
the reason long-context LLMs are hard. **Sparse attention** patterns
— Longformer, BigBird, sparse Mixture-of-Experts routing — represent
the attention matrix as a sparse mask, multiplied lazily by the
keys and values. The matrix never gets materialised at full
density. Every modern long-context transformer ships with some form
of sparse attention under the hood.

**One-hot encodings and feature hashing.** A categorical feature
with thousands of levels expanded to one-hot is a column of vectors
each with a single non-zero. The natural representation is sparse,
and scikit-learn returns sparse matrices from `OneHotEncoder` and
`HashingVectorizer` for exactly this reason. The downstream linear
model (logistic regression, linear SVM) has sparse-aware code paths
that are an order of magnitude faster than the dense equivalent.

**Sparse autoencoders for interpretability.** Anthropic's and
OpenAI's recent interpretability work on LLMs uses sparse
autoencoders to decompose dense neural network activations into a
sparse basis of interpretable features. The sparsity is enforced by
an L1 penalty during training; the result is a sparse activation
matrix where most features are zero on most inputs. Reading the
non-zeros tells you what the network is "thinking" — a structural
trick that is conceptually identical to factorising a TF-IDF matrix.

**Solvers in classical ML.** The kernel matrix in an SVM, the
Hessian in second-order optimisation, the design matrix in LASSO
regression — all sparse for typical workloads. scikit-learn's
LASSO, ridge, and SVM solvers all have CSR-aware paths. The
quadratic-programming solvers behind support vector machines are
implementations of the same family of sparse-matrix algorithms that
power CFD and finite-element analysis.

---

## When NOT to use sparse

Sparse representations are powerful but not free.

**When the matrix is actually dense.** The sparse formats add
per-non-zero overhead — index storage, irregular memory access. A
matrix that is 20% non-zero is not really sparse, and the dense
representation will be both smaller (no index overhead) and faster
(BLAS, vectorisation, cache locality). Density above ~30% means
dense.

**When you need random-access writes.** CSR and CSC are
read-optimised. Setting a single entry — `A[i, j] = v` — requires
shifting the `indices` and `data` arrays of the affected row or
column, which is `O(nnz_in_row)` per write. If your access pattern
is random-write, build in LIL or DOK instead, then convert.

**When you need many element-wise operations on the structure.**
Sparse element-wise multiplication preserves sparsity, but
element-wise *addition* of a dense scalar destroys it (every zero
becomes that scalar). The same goes for element-wise functions like
`exp` or `log`. Beware of any operation that turns a zero into a
non-zero.

**When the matrix-vector product cost dwindles compared to other
work.** SpMV is fast, but the constant factor is meaningfully
larger than dense BLAS. If your matrix is small enough to fit
densely and SpMV is not the bottleneck, the simpler representation
wins. This often comes up in middle-sized neural network layers
(say, 4096 × 4096) where dense is universally faster despite the
weights being mostly small.

**When you cannot guarantee the sparsity pattern.** If the matrix
*could* fill up at runtime — say, a recommender's rating matrix as
a power user adds hundreds of ratings — you might end up with a
nominally-sparse matrix that has become dense in practice. The
conversion overhead and the failure mode are nasty. Either cap the
sparsity (truncate per-user ratings) or use a hybrid representation
that can switch.

---

## What comes next

Eleven foundations down, one to go. Part 12 is **Vector Indexes
(ANN)** — HNSW, IVF, PQ, and LSH. The structures inside FAISS,
Pinecone, Chroma, and every modern RAG retriever, picking up where
this article and Part 10 leave off on the dense-vs-sparse and
exact-vs-probabilistic trade-offs.

After Part 12 the foundations are complete. The first algorithm
article is **linear regression** — and it returns to the matrix
algebra of Parts 1 and 2 with everything we have learned about
storage, sparsity, and scale layered on top.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**sparse_matrices.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/11-sparse-matrices/sparse_matrices.py)

Run it with:

```bash
python sparse_matrices.py
```

It needs `numpy` and `scipy` — the only two non-stdlib
dependencies in this whole foundations track, and the only ones any
serious ML project ships without. The script builds a small COO
matrix and converts to CSR, benchmarks sparse SpMV against the
dense equivalent at the 1%-density crossover, and ends with a
TF-IDF demo on a hundred-document toy corpus showing the resulting
CSR matrix's density and memory footprint. The headline numbers
worth pinning to the wall: **~15× SpMV speedup at 1% density**,
**~660× memory reduction on a 50,000² matrix at 0.1% density**,
and **TF-IDF matrices that fit on a phone where the dense
equivalent would crush a workstation**.

---

*This is Part 11 of the Algorithms in Python series, Foundations track. The companion script `sparse_matrices.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 10](https://medium.com/@grahamjroy/probabilistic-data-structures-trading-exactness-for-sublinear-memory-410e07b57cb6) covered probabilistic data structures. Part 12 — the final foundation — will look at vector indexes for approximate nearest-neighbour search: HNSW, IVF, PQ, and LSH, the structures inside every modern RAG retriever.*
