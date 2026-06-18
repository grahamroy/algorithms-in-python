# Matrix Factorisation — The Recommender System That Started It All

### *Algorithms in Python --- Recommender Systems, Part 1*

---

In October 2006, Netflix offered a million dollars to anyone
who could improve their movie-recommendation algorithm by 10%.
Three years and a thousand teams later, the winning entry was
an ensemble of methods, but the single biggest innovation —
the one that pulled the leaderboard out of its early plateau
and reshaped recommender systems forever — was **matrix
factorisation**.

The setup is simple to state. Build an `n × m` matrix `R` where
rows are users, columns are items (movies, products, songs),
and entries are ratings. The matrix is almost entirely empty
— a typical user has rated maybe 0.1% of the available items.
The recommendation task is to fill in the missing entries: for
each user-item pair we have not observed, predict what rating
the user would give if they tried it. Recommend the items with
the highest predicted ratings.

Matrix factorisation says: find two smaller matrices `U` (`n ×
k`) and `V` (`m × k`) such that `R ≈ U · Vᵀ`. Each user is now
represented by a `k`-dimensional **latent factor vector** `u_i`;
each item by a `k`-dimensional vector `v_j`. The predicted
rating of user `i` for item `j` is the dot product `u_i · v_j`.
Train `U` and `V` to fit the observed ratings; use them to
predict the missing ones.

This sounds modest. It is one of the most consequential
algorithmic ideas in modern applied ML. Matrix factorisation
underlies Spotify's "Discover Weekly", Amazon's "people who
bought this also bought", LinkedIn's job recommendations, and
the early generations of every collaborative filtering pipeline
that powered the recommendation web before deep learning took
over. Modern systems use richer architectures (neural
collaborative filtering, two-tower models, sequence-aware
recommenders — the rest of this track) but the
matrix-factorisation foundation persists.

This article builds matrix factorisation from first principles.
We will set up the explicit-feedback problem, walk through the
two standard fitting algorithms — **Alternating Least Squares**
(ALS) and **Stochastic Gradient Descent** (SGD, in the
FunkSVD style that won the Netflix Prize) — implement ALS from
scratch on a synthetic ratings matrix, compare with
truth-holdout error, and finish with the implicit-feedback variant
that handles "clicks rather than ratings" and the production
considerations (cold start, regularisation, scalability) that
matter in practice.

---

## The setup: explicit feedback

Each user `i ∈ {1, ..., n}` has rated some subset `Ω_i` of
items, giving a rating `r_{i,j}` on a numeric scale (typically
1–5 stars). The full ratings matrix `R ∈ ℝ^{n × m}` has values
only at the observed `(i, j)` pairs in `Ω = ∪_i {i} × Ω_i`;
elsewhere it is empty.

We assume there is a **latent structure**: each user has
preferences over abstract "tastes" (action vs drama, romance vs
sci-fi, indie vs mainstream), each item has loadings on those
tastes, and the rating is approximately the dot product of the
two. Pick a dimensionality `k` (the number of latent factors,
typically 10–200) and fit:

```
r̂_{i,j} = u_i · v_j   where u_i, v_j ∈ ℝ^k
```

The training objective minimises squared error on the *observed*
ratings, with L2 regularisation to prevent overfitting:

```
L(U, V) = Σ_{(i,j) ∈ Ω} (r_{i,j} - u_i · v_j)²
         + λ (Σ_i ‖u_i‖² + Σ_j ‖v_j‖²)
```

The missing-data nature of `R` is the key challenge. We cannot
just compute a regular SVD of `R` — there are no values where
it's empty. We have to fit `U` and `V` from only the observed
entries, then use them to fill in the rest.

---

## Algorithm 1: Alternating Least Squares (ALS)

The trick that makes the optimisation tractable: **fix one
matrix and solve for the other** in closed form. Then swap and
repeat.

With `V` fixed, the optimisation over `U` decouples per user.
For each user `i`, the optimal `u_i` minimises:

```
Σ_{j ∈ Ω_i} (r_{i,j} - u_i · v_j)² + λ ‖u_i‖²
```

This is **ridge regression**: predicting the observed ratings
of user `i` from the (fixed) item vectors of the items they've
rated, with L2 regularisation. The closed-form solution is:

```
u_i = (V_{Ω_i}ᵀ V_{Ω_i} + λ I)⁻¹ · V_{Ω_i}ᵀ r_{i,Ω_i}
```

where `V_{Ω_i}` stacks the item vectors of items rated by user
`i`, and `r_{i,Ω_i}` is the vector of those ratings. Each
per-user update is a small `k × k` matrix inverse.

Symmetrically, with `U` fixed, the optimal `v_j` is the same
formula with `U_{Ω_j}` and `r_{Ω_j, j}` (the users who rated
item `j` and their ratings of it).

The full algorithm:

```
ALS(R, k, λ, n_iter):
    Initialise U, V randomly
    for iter in 1..n_iter:
        # Update all users with V fixed
        for i in 1..n:
            u_i = ridge_regression(V_{Ω_i}, r_{i, Ω_i}, λ)
        # Update all items with U fixed
        for j in 1..m:
            v_j = ridge_regression(U_{Ω_j}, r_{Ω_j, j}, λ)
    return U, V
```

ALS converges monotonically (each step minimises the loss
exactly given the other matrix), the per-user / per-item
updates are independent so the whole thing parallelises
trivially, and the only hyperparameters are `k`, `λ`, and the
iteration count.

The Netflix paper used essentially this algorithm. Spark MLlib
ships an ALS implementation that scales to billion-rating
datasets.

---

## Algorithm 2: Stochastic Gradient Descent (FunkSVD)

The alternative, made famous by Simon Funk's blog post during
the Netflix Prize: optimise the loss directly with SGD over
individual ratings.

For each observed `(i, j, r_{i,j})` in random order, compute
the prediction error `e_{i,j} = r_{i,j} - u_i · v_j` and
update:

```
u_i ← u_i + η · (e_{i,j} · v_j - λ · u_i)
v_j ← v_j + η · (e_{i,j} · u_i - λ · v_j)
```

Step in the direction that reduces the squared error, with the
L2 regulariser pulling factors toward zero. Iterate over all
ratings, multiple epochs.

SGD is conceptually simpler than ALS, scales better in some
regimes (one rating at a time vs full per-user regression),
and is the algorithm that won the Netflix Prize. It's also the
foundation of the **gradient-based** training in every deep
recommender that followed.

For typical dataset sizes either algorithm works. ALS is
easier to parallelise across machines; SGD is easier to
distribute via mini-batching. The recommendation libraries
(Surprise, implicit, LensKit) typically ship both.

---

## A worked example

The companion script generates a synthetic ratings matrix from
known latent factors — 200 users, 100 items, true rank 5 —
adds noise, masks 60% of entries to simulate sparsity, fits
both ALS and SGD with `k = 5`, and reports the RMSE on a
held-out test set:

```
DEMO --- Matrix factorisation on synthetic ratings
  Users           : 200
  Items           : 100
  True latent dim : 5
  Observed cells  : 7983 of 20000 (39.9% density)
  Train/test split: 6387/1596

  Method                       k   iters     train RMSE      test RMSE
  -----------------------   ----   -----    -----------    -----------
  Baseline (global mean)       —       —          2.245          2.246
  ALS                          5      20          0.281          0.369
  SGD (FunkSVD)                5     200          0.347          0.448
```

Three observations.

**Both methods dramatically beat the global-mean baseline.**
Predicting every missing rating with the overall average gives
RMSE 2.25; ALS gets 0.37. That's the value of personalisation
in one number — a model that knows nothing except observed
ratings beats the population average by 84% on this synthetic
problem.

**ALS slightly edges out SGD here** (0.37 vs 0.45 test RMSE).
On real data the gap is usually small and depends on
hyperparameter tuning. ALS converges in fewer iterations (20
vs 200) because each step is a closed-form ridge regression;
SGD's smaller per-step updates need more epochs to reach the
same loss.

**Training RMSE is meaningfully lower than test RMSE** (0.28
vs 0.37). This is the regularisation working — without `λ`
the train error would be near zero and the test error would
explode. Picking `λ` by cross-validation is the most
important hyperparameter choice.

---

## Implicit feedback

In production we usually don't have ratings. We have *clicks*,
*plays*, *purchases* — implicit signals that a user *interacted*
with an item but no explicit "I rate this 4 stars".

The implicit-feedback variant (Hu, Koren & Volinsky, 2008)
reformulates the problem. Treat the observation `r_{i,j}` as a
confidence-weighted preference:

- If the user interacted with the item (`r_{i,j} > 0`),
  preference `p_{i,j} = 1`, confidence `c_{i,j} = 1 + α · r_{i,j}`.
- If no interaction observed (`r_{i,j} = 0`), preference
  `p_{i,j} = 0`, confidence `c_{i,j} = 1`.

The loss becomes:

```
L = Σ_{all (i,j)} c_{i,j} · (p_{i,j} - u_i · v_j)² + λ (‖U‖² + ‖V‖²)
```

Two key differences from explicit feedback. First, *every*
user-item pair contributes a loss term (not just observed
ones), with low confidence on unobserved pairs. Second, the
target is binary preference (interacted vs not) rather than a
real-valued rating.

The Hu-Koren-Volinsky ALS update has the same shape as standard
ALS but with confidence weighting. The `implicit` Python library
implements this efficiently. This is the algorithm behind most
production "people who clicked also clicked" recommenders.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The relevant scales:

**ALS per iteration**: `O(n · |Ω_i| · k² + m · |Ω_j| · k²)`
where `|Ω_i|` is the average number of ratings per user and
`|Ω_j|` per item. The `k²` per rating comes from building each
entity's `k × k` Gram matrix; the `k × k` solve itself adds an
`O(k³)` term per entity. For typical `k ≤ 100` this is fast.

**SGD per epoch**: `O(|Ω| · k)` — one update per observed
rating. Faster than ALS per iteration for small `k`, but needs
more iterations.

**Memory**: `O((n + m) · k)` for the factor matrices, `O(|Ω|)`
for the observed ratings. Even with `n = m = 10⁶` and `k = 100`,
the factor matrices are ~800MB total — fits on one machine.

**Prediction**: `O(k)` per user-item query (one dot product).
A million queries per second on commodity hardware.

For genuinely huge datasets (billions of ratings, hundreds of
millions of users/items) Spark's `MLlib ALS` does distributed
ALS across a cluster.

---

## Real-world ML and AI connections

**The Netflix Prize.** The 2006–2009 competition that put
matrix factorisation on the map. Bell et al's BellKor's
Pragmatic Chaos team won with an ensemble of ~100 models,
roughly half of which were matrix-factorisation variants.

**Spotify Discover Weekly.** The early generations relied
heavily on implicit-feedback matrix factorisation on listening
data. Modern incarnations add deep models, but the
matrix-factorisation foundation remains.

**LinkedIn job recommendations**, **Amazon "people who bought
this"**, **YouTube watch suggestions (pre-deep-learning era)**
— all collaborative filtering systems that started life as
matrix factorisation.

**Google's TensorFlow Recommenders.** The library's first
example tutorial implements matrix factorisation, then extends
to neural variants.

**Implicit-feedback systems in production today.** Most
"frequently-bought-together" widgets on e-commerce sites are
either pure implicit MF or hybrid MF + content features.

**Cold-start hybrid systems.** Pure MF cannot recommend to
new users (no observed ratings to fit `u_i`). Production
systems combine MF with content-based features (item metadata,
user demographics) to handle the cold-start problem.

**Embedding-based ML at scale.** Matrix factorisation is the
conceptual ancestor of every "learn a low-dimensional embedding
per entity" technique — word embeddings, node embeddings on
graphs, two-tower retrieval models, the encoder half of
modern recommender stacks. Once you understand MF, the rest of
the recommender literature is a series of generalisations.

---

## When NOT to use matrix factorisation

**When you have no interaction data.** Brand-new platform, no
user history. MF needs observed ratings or implicit interactions
to fit factors. Cold-start solutions (content-based filtering,
demographic models, popularity baselines) are the entry point;
MF kicks in after enough interactions accumulate.

**When the data is very dense.** Standard MF assumes most
entries are unobserved (the sparse-matrix regime). Dense data
might be better served by direct prediction methods or
deep models.

**When item attributes matter more than collaborative signal.**
Pure MF only uses the rating matrix; if your problem is really
about "find me items similar to this one based on content",
content-based filtering or learned content embeddings might
work better.

**When recommendations need to be diverse / serendipitous.**
MF tends to amplify popularity biases. The top-10 predicted
ratings for most users will look very similar. Production
systems layer diversity / novelty / freshness re-rankers on top.

**When you need explainable recommendations.** "We recommend
this because users with similar latent factor 7 liked items
with high factor-7 loading" is not an explanation a human can
understand. Content-based or rule-based systems are more
transparent.

**When the data has strong sequence structure.** "What will
the user watch next given what they watched today?" is a
sequential question that standard MF treats as i.i.d. Use
sequence-aware recommenders (covered later in this track).

---

## What comes next

Part 2 of the Recommender Systems track is **Neural
Collaborative Filtering** — replacing the dot-product
prediction `u_i · v_j` with a deep neural network that takes
user and item embeddings as input and outputs a predicted
rating. The latent factors are learned with backpropagation;
the prediction function is non-linear; the rest is the same
matrix-factorisation idea generalised.

Then comes **Two-Tower Retrieval** (the scalable architecture
behind every modern web-scale recommender), and **Sequential
Recommenders** (treating user behaviour as a sequence and
predicting the next interaction).

---

## The complete code

The full script is on GitHub — grab it and run it:

[**matrix_factorisation.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/07-recommender-systems/01-matrix-factorisation/matrix_factorisation.py)

Run it with:

```bash
pip install numpy
python matrix_factorisation.py
```

It needs only `numpy`. The script generates a synthetic
ratings matrix from known latent factors, masks 60% of
entries to simulate sparsity, fits both ALS (with closed-form
ridge-regression per-user updates) and SGD (FunkSVD style)
from scratch, and reports test-set RMSE for both versus the
global-mean baseline. Both methods dramatically beat the
baseline; ALS edges out SGD here (0.37 vs 0.45 test RMSE),
converging in far fewer iterations. The
headline insight worth pinning to the wall: **matrix
factorisation decomposes the sparse ratings matrix into
user-factor and item-factor matrices whose dot product
predicts ratings; ALS gives closed-form per-entity updates
that parallelise trivially; SGD over observed ratings is the
Netflix-Prize-winning alternative; the implicit-feedback
variant powers most production recommenders today**.

---

*This is Part 1 of the Recommender Systems track in the Algorithms in Python series. The companion script `matrix_factorisation.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). The previous track — Time Series & Forecasting — closed with the Temporal Fusion Transformer. Part 2 of this track will look at Neural Collaborative Filtering — replacing the dot product with a deep network.*
