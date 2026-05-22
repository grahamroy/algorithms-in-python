# Association Rule Mining — Beer, Diapers, and the Apriori Algorithm

### *Algorithms in Python --- Unsupervised Learning, Part 8*

---

For the last seven articles we have been working with numeric
feature vectors — points in `d`-dimensional Euclidean space,
clustered, projected, embedded, factorised, and visualised.
Today the data type changes. Each "data point" is a **set** —
the items in a single shopping basket, the pages a user
visited in one session, the medical codes attached to a single
patient visit, the genes co-expressed in one cell. The
question is no longer "which points are close in space?" but
"which combinations of items frequently appear together, and
which items predict the presence of others?"

**Association Rule Mining** is the family of algorithms that
answers those questions. Agrawal, Imielinski & Swami's 1993
paper *Mining Association Rules between Sets of Items in Large
Databases* established the framework: count which itemsets
appear frequently together in a transaction database, then
turn the frequent itemsets into rules of the form *"if a
basket contains {milk, bread}, it probably contains {butter}"*.
The algorithm Agrawal & Srikant proposed in 1994 to do this
efficiently — **Apriori** — was for nearly two decades the
default way to run market-basket analysis on retail data, and
it is still the entry point to a whole field of pattern-mining
algorithms (FP-Growth, ECLAT, the more modern frequent-pattern
miners).

The technique entered popular awareness through the (probably
apocryphal) "beer and diapers" story: a major US retailer's
data-mining team supposedly discovered that men who shopped on
Friday evenings often bought beer and diapers together, and
re-arranged store layouts accordingly. The story has been
repeated, debunked, and re-told for thirty years; whether or
not it is literally true, it is the canonical example of what
association rule mining is for: discovering non-obvious
co-purchase patterns that influence business decisions.

This article builds Apriori from first principles. We will
define support, confidence, and lift; walk through the
algorithm's clever "monotone downward closure" property that
lets it prune the exponential search space; implement it from
scratch on a small grocery dataset; compare with the
production-grade `mlxtend` library; and finish with where the
algorithm fits in 2026 — narrower than it was in 1996, but
still very much in use.

---

## The setup: transactions and itemsets

A **transaction database** is a collection of transactions,
where each transaction is a set of items drawn from a universe
of possible items `I`. Example:

```
T1 = {milk, bread, butter}
T2 = {bread, butter}
T3 = {milk, bread, eggs}
T4 = {bread, eggs}
T5 = {milk, eggs}
T6 = {bread, butter, eggs}
T7 = {milk, bread}
T8 = {bread, butter}
```

Eight transactions, four distinct items (`milk`, `bread`,
`butter`, `eggs`). The order of items inside a transaction
does not matter — these are sets — and quantities are
ignored (you either bought bread or you did not; the algorithm
does not care about *how much*).

An **itemset** is any subset of `I`. The itemset `{bread,
butter}` is a 2-itemset; the itemset `{milk, bread, eggs}` is
a 3-itemset; and so on. With four items there are `2⁴ - 1 = 15`
possible non-empty itemsets. With a realistic supermarket
inventory (`|I| ≈ 50,000`) the number of possible itemsets is
astronomically large — far too large to enumerate explicitly.

Association rule mining works in two phases:

1. **Find all frequent itemsets.** An itemset is *frequent* if
   it appears in at least a minimum fraction (the **minimum
   support**) of transactions.
2. **Generate rules.** For each frequent itemset, generate
   candidate rules of the form `A → B` (where `A ∪ B` is the
   itemset and `A ∩ B = ∅`), and keep those with **confidence**
   above a threshold.

The first phase is the hard one. Apriori is the algorithm that
makes it tractable.

---

## Three metrics: support, confidence, lift

Three numbers characterise any rule `A → B`:

**Support.** The fraction of transactions that contain both
`A` and `B`:

```
support(A → B) = | transactions containing A ∪ B | / N
```

In the eight-transaction example above:
`support(bread → butter) = |{T1, T2, T6, T8}| / 8 = 0.50`.

**Confidence.** Among transactions that contain `A`, the
fraction that also contain `B`:

```
confidence(A → B) = support(A ∪ B) / support(A)
                  = P(B | A)
```

If `support(bread) = 7/8 = 0.875` and `support(bread, butter)
= 4/8 = 0.50`, then `confidence(bread → butter) = 0.50 / 0.875
≈ 0.57` — 57% of bread-buying transactions also contain butter.

**Lift.** Confidence normalised by the baseline probability of
`B`:

```
lift(A → B) = confidence(A → B) / support(B)
            = P(B | A) / P(B)
```

A lift of `1.0` means `A` provides no information about `B`
(buying `A` does not change the probability of `B`). Lift `>1`
means `A` is a *positive* indicator of `B` (the items
co-occur more than chance). Lift `<1` means `A` is a *negative*
indicator (the items co-occur less than chance — buying one
makes you less likely to buy the other).

In practice, sorting candidate rules by **lift** is usually
what surfaces interesting findings, because high-support
high-confidence rules involving common items (e.g.
`anything → bread` in a supermarket where 90% of baskets
contain bread) tend to be obvious and uninteresting. Lift
isolates the *surprising* associations.

---

## The Apriori algorithm

The brute-force way to find all frequent itemsets is to
enumerate every subset of `I` and count its support — `2^|I|`
candidates. For `|I| = 50,000` this is impossible.

Apriori's key insight is the **monotone downward closure
property** (the "Apriori property"):

> If an itemset is frequent, all of its subsets are also
> frequent. Equivalently, if any subset of an itemset is
> *infrequent*, the itemset itself is infrequent.

This means we can build the frequent itemsets level by level
— first all frequent singletons, then frequent pairs (only
considering pairs of frequent singletons), then frequent
triples (only considering combinations of frequent pairs),
and so on. At each level we prune candidates whose subsets
are not all known to be frequent.

```
apriori(transactions, min_support):
    # Level 1: all single items
    L_1 = { {i} : support({i}) ≥ min_support }
    L = [L_1]
    k = 2
    while L[k - 2]:                  # while previous level non-empty
        # Candidate generation: join L_{k-1} with itself
        C_k = generate_candidates(L[k - 2])
        # Prune candidates whose subsets aren't all in L_{k-1}
        C_k = prune(C_k, L[k - 2])
        # Count support in one pass over transactions
        L_k = { c in C_k : support(c) ≥ min_support }
        L.append(L_k)
        k += 1
    return union(L)
```

The candidate-generation step pairs every two `(k−1)`-itemsets
that share `k−2` items to form a candidate `k`-itemset. The
pruning step then removes any candidate that has any
`(k−1)`-subset not in `L_{k-1}` — because if any such subset
were infrequent, the candidate itself must be infrequent (by
the Apriori property), so we can drop it without checking.

The number of database passes equals the size of the largest
frequent itemset (typically `≤ 5` in real retail data). Each
pass scans all transactions and increments counts for the
candidate itemsets it contains. This is the algorithm's main
cost.

Once we have all frequent itemsets, rule generation is
straightforward: for every frequent itemset `F` and every
non-empty proper subset `A ⊂ F`, the rule `A → F \ A` has
support `support(F)` and confidence `support(F) / support(A)`.
Keep the rules whose confidence exceeds the threshold.

---

## A worked example

The companion script uses a small synthetic grocery dataset
(50 transactions over 10 items — a tractable size for showing
the algorithm in action) and runs both a from-scratch Apriori
and `mlxtend`'s production implementation.

```
DEMO 1 --- Apriori from scratch on a small grocery dataset
  Transactions   : 50
  Items          : 10
  min_support    : 0.2  (must appear in >=10 transactions)
  Frequent itemsets discovered: 41
  Examples (top 5 by support):
    {bread}                     support=0.760
    {eggs}                      support=0.580
    {milk}                      support=0.500
    {butter}                    support=0.480
    {bread, butter}             support=0.460
```

```
DEMO 2 --- Rule generation (min_confidence=0.6, sorted by lift)
  antecedent       ->  consequent          conf    lift
  ---------------      --------------    ------  ------
  {beer, bread}    ->  {butter, soda}     0.667    3.03
  {butter, soda}   ->  {beer, bread}      0.909    3.03
  {beer}           ->  {chips, soda}      0.611    2.78
  {chips, soda}    ->  {beer}             1.000    2.78
  {butter, soda}   ->  {beer}             0.909    2.53
```

```
DEMO 3 --- Same data, mlxtend Apriori + association_rules
  Frequent itemsets discovered: 41   (matches from-scratch)
  Rules with confidence >= 0.6:  57   (matches from-scratch)
  Top rule by lift: {beer, bread} -> {butter, soda}  (lift = 3.03)
```

Three observations.

**Apriori discovered 41 frequent itemsets out of `2¹⁰ - 1 =
1023` possible itemsets.** That is the win of the monotone
pruning: even on a tiny dataset, the algorithm doesn't have
to check most of the search space.

**The highest-lift rules cluster around the "snack basket"
pattern.** People who buy `{chips, soda}` *always* buy beer in
this dataset (confidence `1.000`), and they do so at lift
`2.78` — meaning beer is 2.8× more common in chips+soda
baskets than in the general population. Pull these rules
together and the algorithm has rediscovered the "people who
buy snacks for the game also buy beer" co-purchase pattern,
without any prior knowledge of what beer or chips are.

**The from-scratch implementation matches mlxtend exactly on
both itemset count (41) and rule count (57).** Apriori is one
of the few ML algorithms where reproducibility across
implementations is essentially guaranteed — given the same
data, support threshold, and confidence threshold, all
correct implementations return the same itemsets and rules.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The dominant costs:

**Per-pass scan.** Each pass through the transaction database
costs `O(N · |C_k| · k)` — for each of `N` transactions, check
which of `|C_k|` candidate `k`-itemsets it contains, with each
check costing `O(k)`. This is the per-iteration cost.

**Number of passes.** Equals the size of the largest frequent
itemset — typically small (`≤ 5`) on real retail data, can be
larger on dense biomedical or web-log data.

**Candidate generation.** `O(|L_{k-1}|² · k)` for the join and
prune steps. Subset enumeration of each candidate requires
checking `O(k)` subsets against the previous level.

**Memory.** `O(|L|)` for storing all frequent itemsets across
levels. The candidate sets `C_k` and `L_k` typically fit
comfortably in memory even on million-transaction databases.

**The pathological worst case.** If `min_support` is set very
low (or the data is very dense), the number of frequent
itemsets can grow exponentially in `|I|` — Apriori then
generates massive candidate sets and slows to a crawl. The
practical answer is to raise `min_support` until the itemset
count is tractable, or switch to **FP-Growth** — a tree-based
algorithm that avoids candidate generation entirely and is
typically 5-10× faster than Apriori on dense data.

For very large transaction databases (Walmart-scale retail,
clickstream logs from a major web property), the modern
recommendations are:

- **FP-Growth** — no candidate generation, two database
  passes total, much faster than Apriori on dense data.
- **ECLAT** — vertical data layout (each item maps to the set
  of transactions containing it); fast on sparse data.
- **Top-K closed pattern miners** — instead of enumerating
  all frequent itemsets, find the top-K "closed" itemsets
  (those that are not subsumed by a larger itemset with the
  same support).

In `mlxtend` and other libraries, FP-Growth is the default when
the candidate generation in Apriori becomes the bottleneck.

---

## Real-world ML and AI connections

Association rule mining is the algorithm that built
recommendation systems before collaborative filtering took
over. It is also still in production wherever pattern-mining
matters:

**Market-basket analysis.** The original use case and still
the most common. Retailers, supermarkets, e-commerce — all
run Apriori or FP-Growth on their transaction logs to
discover co-purchase patterns. Outputs feed product placement,
promotion design, basket-builder UI components, and
cross-sell recommendations.

**Click-stream analysis.** Web analytics platforms use
association rule mining to find common click sequences and
page-co-visits. "Users who viewed `/pricing` and `/docs`
often go to `/signup`" is a rule with practical product
implications.

**Medical billing fraud detection.** A patient's billing record
is a set of procedure codes; rules of the form *"if codes A
and B appear together, code C usually follows"* let auditors
spot bills that violate the expected co-occurrence patterns.

**Inventory and supply chain.** Co-occurrence rules in
purchase orders identify products that should be stocked
together, reducing stock-outs of complementary items.

**Bioinformatics.** Gene-expression data, where each cell is
a "transaction" of expressed genes; association rules surface
co-regulated gene modules. Drug-discovery pipelines use
similar techniques on chemical-feature sets.

**Recommender systems.** Modern recommender systems (matrix
factorisation, neural collaborative filtering, sequence
models) have largely displaced association rules as the
primary recommendation engine — but Apriori-style co-occurrence
mining is often the **baseline** against which more
sophisticated methods are compared, and remains the right
choice when the data is small, the interpretability bar is
high, or the recommendation logic must be auditable.

**Educational data mining.** Student-course enrolment data,
quiz-answer co-occurrence patterns, learning-pathway
discovery — all natural fits for association rule mining.

The pattern: ARM is the right tool when you have **set-valued
transactions** and want **interpretable co-occurrence rules**.
For continuous embeddings, dense feature vectors, or any
non-set data, reach for something else.

---

## When NOT to use association rule mining

The technique has well-defined limits:

**When the data isn't set-valued.** ARM is designed for
unordered sets of categorical items. For sequences (where
order matters), use **sequential pattern mining** (PrefixSpan,
SPADE). For continuous features, the algorithm does not
apply.

**When you need ranking or personalisation.** Association
rules are global — they tell you "in general, A predicts B".
For per-user recommendations based on personal preferences,
collaborative filtering or matrix factorisation is the right
tool.

**When the data is too dense or the minimum support too
low.** Apriori's exponential blow-up on dense data is well
known. Switch to FP-Growth, or raise the support threshold
until the itemset count is manageable.

**When the rare interesting rules are below the support
threshold.** ARM can only find rules involving items that
appear frequently enough together. The "beer and diapers"
combination would not have been found unless the support
threshold was tuned low enough to include it — and tuning
that low typically explodes the candidate count. **Top-K
mining** or **rare-rule mining** algorithms specifically
target these.

**When the items have known hierarchy that matters.**
Standard ARM treats items as opaque labels. If your items
have a taxonomy (e.g. specific products under categories),
**generalised association rule mining** uses the hierarchy
explicitly.

**When statistical significance matters.** High-confidence
rules with low support are often coincidences. The
statistical-significance tests for association rules
(chi-squared, Fisher's exact test, false-discovery-rate
corrections) are not built into the standard algorithm —
treat the output as a candidate list, not as proven
relationships.

---

## What comes next

This article closes out the **Unsupervised Learning** track
(8 articles). Track 04 — **Advanced Unsupervised Learning** —
opens with **DBSCAN**, the density-based clustering algorithm
that identifies clusters by density-connected regions and
handles arbitrary shapes natively (without spectral
clustering's eigenvector machinery). After DBSCAN: Gaussian
Mixture Models, Autoencoders, Anomaly Detection, and Latent
Dirichlet Allocation.

The unsupervised toolkit we built across this track —
K-Means, hierarchical clustering, PCA, t-SNE, UMAP, NMF,
spectral clustering, and association rule mining — covers
the standard repertoire for discovering structure in
unlabelled data. The advanced track sharpens those tools
and adds the probabilistic, density-based, and deep-learning
approaches that handle the cases the basic toolkit struggles
with.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**apriori.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/03-unsupervised-learning/08-association-rule-mining/apriori.py)

Run it with:

```bash
pip install numpy mlxtend pandas
python apriori.py
```

It needs `numpy`, `pandas`, and `mlxtend` (the standard
Python library for association rule mining; not part of
scikit-learn). The script implements Apriori from scratch
with level-by-level frequent-itemset generation and the
Apriori-property pruning, applies it to a 50-transaction
10-item grocery dataset, generates rules sorted by lift, and
compares against `mlxtend`'s production implementation (the
two find identical itemsets and rules). The headline insight
worth pinning to the wall: **Apriori turns the exponential
search for frequent itemsets into a level-by-level
enumeration by exploiting the monotone-downward-closure
property — if an itemset is infrequent, all its supersets
are too — and association rules sorted by lift surface the
non-obvious co-occurrence patterns that drive market-basket
analysis to this day**.

---

*This is Part 8 of the Unsupervised Learning track in the Algorithms in Python series, and the final article in that track. The companion script `apriori.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 7](https://medium.com/p/a205e1fc8d0e) covered Spectral Clustering. The next track — Advanced Unsupervised Learning — opens with DBSCAN, the density-based clustering algorithm that handles arbitrary cluster shapes natively.*
