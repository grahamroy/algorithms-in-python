# Two-Tower Retrieval — How Web-Scale Recommenders Actually Serve

### *Algorithms in Python --- Recommender Systems, Part 3*

---

Part 2 ended on a problem. Neural Collaborative Filtering gave
us a richer model than matrix factorisation — a learned
interaction function instead of a fixed dot product — and it
ranked better. But it broke the one property that makes
recommendation *serveable* at scale: because NeuMF's score is
an arbitrary neural function of `(user, item)`, you cannot
pre-compute item representations and index them. To find a
user's top items you have to run the network once for *every*
item in the catalogue. For a hundred-million-item catalogue,
that is a hundred million forward passes per recommendation.
Impossible in a 50-millisecond serving budget.

**Two-Tower Retrieval** is the architecture that resolves the
tension. It keeps NCF's central lesson — *learn rich
embeddings with deep networks* — but deliberately constrains
the final scoring step back to a **dot product**. A "user
tower" (a deep network) maps the user and their context to a
vector. An "item tower" (a separate deep network) maps each
item to a vector in the same space. The predicted affinity is
the dot product of the two vectors. The towers can be
arbitrarily deep and non-linear; the *combination* is forced
to be a single inner product.

That one constraint changes everything about serving. Item
vectors do not depend on the user, so you compute them **once**,
offline, for the entire catalogue and load them into an
approximate-nearest-neighbour index (HNSW, IVF, PQ — the
Foundations Part 12 machinery). At request time you run the
user tower *once* to get the query vector, then the ANN index
returns the top-k items in **sub-linear** time. A
hundred-million-item retrieval becomes a single embedding plus
one ANN lookup — milliseconds, not minutes.

This is the model behind YouTube's candidate generation,
Google Play's recommendations, every large-scale
"retrieve-then-rank" pipeline, and the retrieval half of most
RAG systems. This article builds it from first principles. We
will set up the retrieval problem and the in-batch-softmax
training trick that makes it work, implement both towers and
the sampled-softmax loss from scratch in numpy, evaluate
recall on a synthetic dataset, and finish with the
production realities — the retrieve-then-rank split, the
embedding-staleness problem, and where two-tower stops and
the heavier ranker takes over.

---

## Retrieval vs ranking

The key reframing: at web scale, recommendation is **two
stages**, not one.

**Retrieval (candidate generation).** From a catalogue of
`10⁸` items, cheaply select a few hundred plausible
candidates for this user. Must be *fast* and *high-recall* —
it is fine to include some mediocre candidates, but a great
item missed here can never be recommended. This is where
two-tower lives.

**Ranking.** Take those few hundred candidates and score them
precisely with a heavy model (a NeuMF-style interaction
network, a gradient-boosted tree over hand-crafted features,
a transformer). Must be *accurate* — it decides the final
order — but only runs on hundreds of items, so it can afford
to be expensive.

The two stages have opposite cost profiles, and the two-tower
model is purpose-built for the first. Its job is not to be the
most accurate scorer; its job is to *not miss* the good items
while running fast enough to scan the whole catalogue. The
dot-product constraint is the price of admission to ANN
indexing, and recall — not ranking accuracy — is the metric
that matters.

---

## The architecture

Two independent networks producing vectors in a shared space:

```
User tower:   u = f_θ(user features, context)      ∈ ℝ^d
Item tower:   v = g_φ(item features)               ∈ ℝ^d
Score:        s(user, item) = u · v
```

`f_θ` and `g_φ` are typically multi-layer perceptrons over
embedded categorical features (user id, item id, plus
side-features like genre, device, time of day). They share
*nothing* — separate parameters, separate inputs — except the
output dimensionality `d` and the geometry of the space they
map into. The towers only ever meet at the final dot product.

Two design points matter enormously in practice:

**Side features beat pure IDs for cold-start.** A pure-ID
two-tower is just matrix factorisation with extra steps. The
win comes from feeding *content features* into the towers — an
item the model has never been trained on still gets a
reasonable vector from its genre, text, and metadata. This is
the structural advantage over classical MF, which has nothing
to say about an unseen ID.

**Normalisation and temperature.** Vectors are usually
L2-normalised so the dot product becomes a cosine similarity,
and the logits are divided by a learned or tuned
**temperature** `τ` before the softmax. Temperature controls
how sharply the model separates positives from negatives; it
is one of the most important hyperparameters in two-tower
training.

---

## Training: in-batch negatives and sampled softmax

Here is the elegant trick that makes two-tower training
practical. We have positives — `(user, item)` pairs that
interacted. We need negatives, and the catalogue is far too
large to score them all. The in-batch approach: within a
mini-batch of `B` positive pairs, treat **every other item in
the batch** as a negative for each user.

```
For a batch of B pairs {(u_1, v_1), ..., (u_B, v_B)}:
    Compute the B × B score matrix  S_ij = u_i · v_j
    Row i is a softmax over the batch:
        the diagonal S_ii is the positive
        the off-diagonal S_ij (j ≠ i) are negatives
    Loss = cross-entropy pushing each row's mass onto the diagonal
```

For each user, the loss says "score your true item higher
than the `B − 1` other items in this batch." This is a
**sampled softmax** — the batch is a random sample of the
catalogue serving as negatives — and it costs only one `B × B`
matmul, reusing item vectors we already computed for the
positives. No separate negative-sampling pass, no extra
forward passes. Bigger batches give more negatives and better
retrieval, which is one reason two-tower training scales with
hardware.

One subtlety: popular items appear as in-batch negatives more
often, so the model over-penalises them. Production systems
apply **logQ correction** — subtract the log of each item's
sampling probability from its logit — to debias. Our
implementation keeps it simple, but the correction is standard
in libraries like TensorFlow Recommenders.

---

## A worked example

The companion script builds a synthetic retrieval dataset —
2,000 users, 1,000 items, interactions generated so that each
user has affinity for a latent-factor neighbourhood — trains
a two-tower model with in-batch softmax from scratch, and
evaluates retrieval quality with the standard metric:
**Recall@K** (is the held-out item among the top K retrieved
from the *full* catalogue?).

```
DEMO --- Two-tower retrieval on synthetic data
  Users                : 2000
  Items                : 1000
  Embedding dim        : 32
  Batch size           : 256   (255 in-batch negatives per positive)
  Training pairs       : 18000
  Epochs               : 40

  Method                       Recall@10    Recall@50    Recall@100
  -------------------------   ----------   ----------   -----------
  Popularity (baseline)            0.083        0.298         0.500
  Two-tower (in-batch softmax)     0.197        0.806         0.951
  Exhaustive dot-product check     0.197        0.806         0.951
```

Three observations.

**Two-tower beats the popularity baseline at every K.**
Recall@10 of 0.197 vs 0.083 (≈2.4×), and the gap widens with
K — by Recall@50 it is 0.806 vs 0.298. A learned per-user
retrieval surfaces the held-out item far more reliably than
recommending globally popular items. That gap is the value of
personalised retrieval. (Recall@10 is intrinsically hard here:
the held-out item competes with the user's *other* in-training
favourites from the same neighbourhood, which legitimately
rank above it.)

**The ANN-free check confirms the embeddings are sound.** The
"exhaustive dot-product check" scores the query vector against
*every* item by brute force and gets identical recall to the
trained retrieval — confirming the recall ceiling is set by
the *embeddings*, not by any approximation. In production you
would swap the brute-force scan for an HNSW index and accept a
~1% recall loss for a 100–1000× speed-up; here we keep it
exact so the number is interpretable.

**Recall climbs steeply with K.** 0.20 → 0.81 → 0.95 from
K=10 to 100. This is exactly the retrieval mindset: the model
does not need the true item at rank 1, it needs it *somewhere*
in the candidate set that the downstream ranker will reorder.
A retrieval Recall@100 of 0.95 means the ranker gets a shot
at the right answer 95% of the time.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The numbers that make two-tower the web-scale answer:

**Training**: in-batch softmax is `O(B² · d)` per batch for
the score matrix plus the tower forward passes — the `B²`
comes from reusing the batch as its own negatives, which is
*cheaper* than sampling separate negatives. Scales linearly
in the number of training pairs.

**Item indexing (offline, one-off)**: run the item tower over
the whole catalogue once — `O(m · cost_of_tower)` — then build
the ANN index, roughly `O(m · log m)` for HNSW. Done in a
batch job, refreshed periodically.

**Serving (per request)**: one user-tower forward pass
(`O(cost_of_tower)`) plus one ANN lookup (`O(log m)`
effective). This is the whole point — **sub-linear in the
catalogue size**. Contrast with NeuMF's `O(m · cost_of_network)`
per request, which is linear and therefore hopeless at `m =
10⁸`.

**Memory**: `O(m · d)` for the item vectors in the index. At
`m = 10⁸` and `d = 64` floats that is ~25 GB — large but
shardable across machines, and quantisation (PQ) cuts it by
4–32×.

The asymmetry is the design: pay a heavy offline cost to index
items, so that each online request is a single embedding and a
single sub-linear lookup.

---

## Real-world ML and AI connections

**YouTube candidate generation.** Covington, Adams & Sargin's
2016 *Deep Neural Networks for YouTube Recommendations*
described exactly this two-stage design — a deep candidate
generator producing embeddings for ANN retrieval, then a
separate ranking network. It is the canonical industrial
reference for the architecture.

**TensorFlow Recommenders (TFRS).** Google's open-source
library is built around the two-tower retrieval model with
in-batch softmax and logQ correction as the default. The
"retrieval" and "ranking" tasks are first-class objects in the
API.

**Dense retrieval for RAG.** The retrieval half of every
retrieval-augmented-generation system is a two-tower model in
disguise: a query encoder and a document encoder map into a
shared space, documents are indexed offline, and the query
embedding does an ANN lookup. DPR (Dense Passage Retrieval,
Karpukhin et al. 2020) is two-tower retrieval applied to
question answering.

**Embedding-based search.** Product search, image search,
and semantic text search all use the two-tower pattern:
encode the query, encode the corpus offline, retrieve by
nearest neighbour. The "encode both sides into a shared
space, retrieve by dot product" recipe is one of the most
reused ideas in applied ML.

**Ad retrieval and candidate generation at scale.** Meta,
Pinterest, and most large platforms run two-tower retrieval
as the first stage of their ads and feed pipelines, often
with hundreds of millions of candidates.

**Contrastive representation learning.** In-batch softmax is
the same objective as CLIP's image-text contrastive loss and
SimCLR's self-supervised loss — a batch of positives, every
other element as a negative, cross-entropy onto the diagonal.
Two-tower retrieval is contrastive learning pointed at
recommendation.

---

## When NOT to use two-tower retrieval

**When the catalogue is small.** If you only have a few
thousand items, you can afford to score them all with a heavy
ranker — skip retrieval entirely and just rank. The two-tower
machinery is overhead you do not need below ~10⁴–10⁵ items.

**When you need the most accurate score.** The dot-product
constraint that buys fast retrieval also caps expressiveness.
Two-tower is deliberately *less* accurate than a
cross-feature interaction model; that is why it feeds a
ranker rather than serving final scores. Do not use it as
your only model when ranking quality is the goal.

**When user-item cross features are essential.** Two-tower
cannot model features that depend on the user and item
*jointly* (e.g. "did this user click this exact item's
category in the last hour?") because the towers never see each
other's inputs until the dot product. Such features belong in
the ranker.

**When you cannot tolerate stale item embeddings.** Item
vectors are computed offline and refreshed periodically. For
items whose relevance changes minute-to-minute (breaking
news, live events, rapidly-shifting prices) the index can lag
reality. Streaming-update or hybrid approaches are needed.

**When you have no side features and little data.** A
pure-ID two-tower with sparse data is just a more expensive
matrix factorisation. Use MF directly until you have content
features and enough interactions to justify the towers.

---

## What comes next

Part 4 — the final article in the Recommender Systems track —
is **Sequential Recommenders**. Everything so far has treated
a user's interactions as an unordered *set*: matrix
factorisation, NCF, and two-tower all ignore the order in
which a user clicked things. Sequential recommenders drop that
assumption and model behaviour as a *sequence*, predicting the
next interaction from the ordered history — the recommendation
analogue of language modelling, and the architecture (GRU4Rec,
SASRec, BERT4Rec) behind "because you just watched…" and
session-based recommendation.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**two_tower.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/07-recommender-systems/03-two-tower-retrieval/two_tower.py)

Run it with:

```bash
pip install numpy
python two_tower.py
```

It needs only `numpy`. The script implements a two-tower
retrieval model from scratch — separate user and item MLP
towers over embedding tables, L2 normalisation, in-batch
sampled-softmax training with a hand-written Adam optimiser
and full backpropagation — then evaluates Recall@10/50/100
against a popularity baseline by retrieving from the full
catalogue, and confirms the trained retrieval matches an
exhaustive brute-force dot-product scan. The headline insight
worth pinning to the wall: **two-tower keeps NCF's deep
embeddings but forces the final score back to a dot product,
so item vectors can be indexed offline and retrieved by
approximate nearest neighbour in sub-linear time — the
architecture that makes web-scale, retrieve-then-rank
recommendation actually serveable**.

---

*This is Part 3 of the Recommender Systems track in the Algorithms in Python series. The companion script `two_tower.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 2 covered Neural Collaborative Filtering — the richer model whose serving cost two-tower exists to fix. Part 4 will look at Sequential Recommenders — modelling user behaviour as an ordered sequence rather than an unordered set.*
