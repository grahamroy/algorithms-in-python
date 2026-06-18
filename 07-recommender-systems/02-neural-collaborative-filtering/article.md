# Neural Collaborative Filtering — When the Dot Product Isn't Enough

### *Algorithms in Python --- Recommender Systems, Part 2*

---

Matrix factorisation gave us a beautiful idea: represent every
user and every item as a vector in the same `k`-dimensional
space, and predict the affinity between them with a **dot
product**. It won the Netflix Prize, it powers a generation of
production recommenders, and it is the foundation everything
else in this track builds on.

But the dot product is a *fixed*, *linear* way of combining two
vectors. `u_i · v_j = Σ_d u_{i,d} v_{j,d}` — multiply matching
dimensions, add them up. It treats every latent dimension as
independent and weights them all equally. There is no room for
"this user likes action films *only when* they're also short",
no room for interactions *between* latent dimensions, no room
for any non-linear structure in how a taste maps to a rating.

In 2017, He et al. asked the obvious question: **what if we let
a neural network learn the interaction function instead of
hard-coding it as a dot product?** Keep the embeddings — one
learned vector per user, one per item — but replace `u_i · v_j`
with a small trained network that takes the two embeddings as
input and outputs a score. The embeddings are still learned by
backpropagation; the difference is that the *combination* step
is now flexible and non-linear. They called it **Neural
Collaborative Filtering** (NCF), and it became the template for
the deep-learning generation of recommender systems.

This article builds NCF from first principles. We will set up
the implicit-feedback problem and the log-loss objective, then
construct the three models from the paper — **GMF**
(Generalised Matrix Factorisation, the dot product as a special
case of a neural layer), **MLP** (concatenate the embeddings
and learn the interaction with a multi-layer perceptron), and
**NeuMF** (fuse the two into one model) — implement all three
from scratch in numpy with negative sampling and
backpropagation, and evaluate them on a synthetic problem whose
ground-truth interaction is deliberately non-linear, using the
standard leave-one-out HR@10 / NDCG@10 ranking protocol.

---

## The setup: implicit feedback and a learned interaction

Matrix factorisation, as we built it in Part 1, predicted a
real-valued rating. NCF is designed for the more common
production setting: **implicit feedback**. We observe that a
user clicked, played, or bought an item — a binary signal — and
we never observe an explicit star rating. The matrix `Y` has

```
y_{i,j} = 1   if user i interacted with item j
y_{i,j} = 0   otherwise (unobserved)
```

The zeros are not "disliked"; they are "unknown". This is the
same regime as implicit-feedback matrix factorisation, but NCF
attacks it as a **binary classification** problem: learn a
model `ŷ_{i,j} = f(i, j)` that outputs the probability that user
`i` would interact with item `j`.

Every model shares the same first step. Each user `i` maps to a
learned **embedding** `p_i ∈ ℝ^k` (a row of a user embedding
table `P`), each item `j` to `q_j ∈ ℝ^k` (a row of an item table
`Q`). These are exactly the latent factor vectors of matrix
factorisation. What changes is the **interaction function** `f`
that turns `(p_i, q_j)` into a score. Matrix factorisation fixes
`f(p_i, q_j) = p_i · q_j`. NCF *learns* it.

The output is squashed through a sigmoid to a probability, and
the model is trained to minimise **binary cross-entropy** (log
loss) over observed positives and sampled negatives:

```
L = − Σ_{(i,j) ∈ Ω⁺ ∪ Ω⁻}
        y_{i,j} log ŷ_{i,j} + (1 − y_{i,j}) log (1 − ŷ_{i,j})
```

where `Ω⁺` is the set of observed interactions and `Ω⁻` is a set
of negatives sampled from the unobserved entries. We come back
to negative sampling below; first, the three interaction
functions.

---

## Model 1: GMF — the dot product as a neural layer

Start by writing matrix factorisation in a form that makes the
generalisation obvious. The dot product `p_i · q_j` is the sum
of the **element-wise product** `p_i ⊙ q_j`:

```
p_i · q_j = Σ_d (p_i ⊙ q_j)_d = 1ᵀ (p_i ⊙ q_j)
```

So a dot product is: take the element-wise product vector, then
collapse it to a scalar with a sum — equivalently, a dot with
the all-ones vector. **Generalised Matrix Factorisation** makes
two changes. First, replace the all-ones vector with a *learned*
weight vector `h ∈ ℝ^k`, so different latent dimensions can
matter more or less. Second, allow a non-linear output
activation:

```
φ_GMF = p_i ⊙ q_j            (element-wise product, in ℝ^k)
ŷ_{i,j} = σ(hᵀ φ_GMF + b)    (learned weights, sigmoid output)
```

Set `h = 1`, drop the bias, and use an identity output and you
recover plain matrix factorisation exactly. GMF is therefore a
strict generalisation: it can represent the dot product, plus a
little more (per-dimension weighting). But the interaction is
still **bilinear** — it only ever multiplies a user dimension by
the *same* item dimension. No cross-dimension interaction, no
non-linearity in how the factors combine. GMF is the dot
product wearing a neural-network hat.

---

## Model 2: MLP — learning the interaction

To capture interactions the dot product cannot, NCF takes a
different route: **concatenate** the user and item embeddings
into one `2k`-vector and feed it through a standard multi-layer
perceptron with non-linear (ReLU) hidden layers.

```
z_0 = [ p_i ; q_j ]                      (concatenate, in ℝ^{2k})
z_1 = ReLU(W_1 z_0 + b_1)
z_2 = ReLU(W_2 z_1 + b_2)
   ⋮
ŷ_{i,j} = σ(hᵀ z_L + b)
```

The difference from GMF is fundamental. The element-wise product
forces the model to compare dimension `d` of the user with
dimension `d` of the item and nothing else. Concatenation throws
both full vectors at the network and lets the weight matrices
learn *whatever* combination of user and item dimensions
predicts interaction — including cross terms (user dimension 3
with item dimension 7) and, through the ReLU non-linearities,
genuinely non-linear functions of the embeddings. A multi-layer
perceptron is a universal function approximator; in principle it
can learn any interaction function, not just the bilinear one.

The cost is that the MLP has to discover the right interaction
from scratch, and learning good embeddings *through* several
non-linear layers from sparse binary feedback is a harder
optimisation problem than learning them under a clean dot
product. We will see this tension in the results.

The typical architecture follows a **tapered** shape: the first
hidden layer is widest and each subsequent layer halves, so the
network compresses the concatenated embeddings down toward the
scalar score. Here we use `2k → 32 → 16 → 1`. (We call this
single funnel a "tower" only once below — in NeuMF, where two
of them run side by side.)

---

## Model 3: NeuMF — fusing the two

GMF gives you the reliable, well-behaved bilinear signal. The
MLP gives you flexible non-linear capacity. **NeuMF** (Neural
Matrix Factorisation) refuses to choose and fuses both into a
single model.

The key design decision: give the two towers **separate
embeddings**. Forcing GMF and MLP to share one embedding table
would couple their (quite different) representational needs and
hurt both. So NeuMF learns a GMF user/item embedding *and* an
independent MLP user/item embedding. Each tower runs to its last
layer; the two final vectors are concatenated and passed to a
single output layer:

```
φ_GMF = p_i^G ⊙ q_j^G                       (GMF tower)
φ_MLP = ReLU(W_2 ReLU(W_1 [p_i^M ; q_j^M]))  (MLP tower, last layer)
ŷ_{i,j} = σ( hᵀ [ φ_GMF ; φ_MLP ] + b )      (fuse and predict)
```

The output layer learns how much to trust the linear GMF signal
versus the non-linear MLP signal, *per dimension*. In the
original paper NeuMF is pre-trained — train GMF and MLP
separately, then initialise NeuMF from their weights and
fine-tune — which squeezes out the last bit of accuracy. Our
implementation trains it end-to-end from random initialisation,
which is simpler and already enough to show the effect.

---

## Training: negative sampling and log loss

There is one problem with the binary-classification framing:
the observed data is *all positives*. Every entry in `Ω⁺` has
label 1. A classifier trained only on positives learns to output
1 for everything. We need negatives.

We do not have explicit negatives (a zero means "unknown", not
"disliked"), so we **sample** them. For each observed positive
`(i, j)`, draw a few items the user has not interacted with and
label them 0. The standard ratio is around four negatives per
positive. Each epoch resamples a fresh set, so over training the
model sees many different unobserved pairs as negatives, which
is a cheap, effective approximation to ranking every unobserved
item below the observed ones.

With positives and sampled negatives in hand, every model is
trained the same way: forward pass to the sigmoid probability,
binary cross-entropy loss, backpropagation through the
interaction function and into the embedding tables, and a
gradient step. We use **Adam** — adaptive per-parameter learning
rates make training these small networks robust without much
tuning, which is why it is the default optimiser for deep
recommenders. The gradient with respect to an embedding row is
**sparse**: only the users and items appearing in the
mini-batch get updated, so the per-step cost scales with the
batch, not with the size of the embedding tables.

---

## A worked example

The companion script generates a synthetic implicit-feedback
dataset — 500 users, 300 items — but with a deliberately
**non-linear ground truth**. The true affinity between a user
and an item is a *sum of rectified group-wise dot products*:
the latent dimensions are split into groups, each group
contributes `ReLU(u_g · v_g)`, and the affinities are summed. No
single dot product can reproduce a sum of rectified dot
products, but an MLP — which computes exactly sums of rectified
linear combinations — can. This is the regime NCF was built for.

The highest-affinity items become each user's observed
interactions; we hold out one interaction per user for testing,
train on the rest with four sampled negatives per positive, and
evaluate with the standard **leave-one-out** protocol: rank each
user's held-out item against 99 items they never touched, and
report **HR@10** (was the held-out item in the top 10?) and
**NDCG@10** (how *highly* was it ranked?).

```
DEMO --- Neural Collaborative Filtering on synthetic implicit feedback
  Users                : 500
  Items                : 300
  Interaction type     : implicit (non-linear teacher)
  Observed interactions: 12500 (8.3% density)
  Eval protocol        : leave-one-out, 1 + 99 negatives, HR@10 / NDCG@10

  Model                        params     HR@10    NDCG@10
  ------------------------   --------   -------   --------
  Popularity (baseline)             —     0.248      0.126
  GMF (dot-product gen.)        12817     0.750      0.537
  MLP (learned interaction)     14401     0.566      0.354
  NeuMF (GMF + MLP fusion)      27217     0.750      0.594
```

Four observations.

**Personalisation crushes popularity.** Recommending the most
popular items gets HR@10 0.25 — a held-out item lands in the top
10 a quarter of the time, barely above the 0.10 you would get by
chance. Every learned model more than doubles it. As with matrix
factorisation, the headline value of collaborative filtering is
visible in one number.

**GMF is a strong baseline — the dot product is hard to beat.**
Even though the ground truth is non-linear, GMF reaches HR@10
0.75. A flexible, well-trained bilinear model captures a great
deal of the structure, and on this problem it comfortably
out-ranks the pure MLP. This is the honest and important lesson
of NCF: the dot product is not a weakling you trivially
improve on. Decades of recommender systems ran on it for good
reason.

**The pure MLP underperforms — capacity is not the same as
accuracy.** The MLP *should* win on this dataset: the teacher
is a sum of rectified group-wise dot products, exactly the
class of function a ReLU MLP can represent and a single dot
product cannot. It still lands at HR@10 0.57, below GMF. The
MLP has the representational power; it just cannot reliably
*find* it. Learning good embeddings through several non-linear
layers from sparse binary feedback is a genuinely harder
optimisation problem, and the extra flexibility buys
overfitting and training difficulty before it buys accuracy.
He et al. found the same thing — MLP alone is not reliably
better than MF — and it is precisely *why* they proposed the
fusion.

**NeuMF gets the best of both.** Fusing the towers matches GMF's
HR@10 (0.75) and clearly wins on NDCG@10 (0.594 vs 0.537) — it
ranks the held-out item *higher* on average. The output layer
leans on GMF for the reliable linear signal and on the MLP for
the non-linear residual the dot product misses, and the
combination ranks better than either tower alone. That is the
entire thesis of the paper in one row of a table.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Let `n` users, `m` items, `k` the embedding dimension, `|Ω⁺|`
the observed interactions, and `H` the total width of the MLP
hidden layers.

**GMF forward/backward**: `O(k)` per `(i, j)` pair — one
element-wise product and a dot. Identical asymptotics to plain
matrix factorisation; the learned weight vector adds nothing to
the order.

**MLP forward/backward**: `O(k + H²)`-ish per pair, dominated by
the dense hidden-layer matrix multiplies. For the small towers
typical of recommenders (`H` in the low hundreds) this is still
cheap, but it is a constant-factor heavier than GMF.

**Training**: `O(E · |Ω⁺| · (1 + n_neg) · cost_per_pair)` for
`E` epochs with `n_neg` sampled negatives per positive. Negative
sampling multiplies the data by `(1 + n_neg)`; everything else
is standard mini-batch SGD. Embedding gradients are **sparse** —
`O(batch · k)` memory and updates per step, independent of the
table sizes.

**Memory**: `O((n + m) · k)` for the embedding tables (NeuMF
doubles this — separate tables per tower) plus the small dense
MLP weights. The embeddings dominate, exactly as in matrix
factorisation.

**Prediction / serving**: this is the catch. GMF and MF can
serve top-N with a single matrix multiply and, crucially, with
**approximate nearest-neighbour** indexes over the item
embeddings — because the score is a dot product, sub-linear
retrieval is possible. NeuMF's score is an arbitrary neural
function of `(i, j)`, so scoring a user against `m` items is `m`
separate forward passes, with **no** ANN shortcut. NCF buys
accuracy at the cost of cheap retrieval — the tension the
two-tower architecture (Part 3) exists to resolve.

---

## Real-world ML and AI connections

**The paper itself.** *Neural Collaborative Filtering* (He,
Liao, Zhang, Nie, Hu, Chua, WWW 2017) is one of the most-cited
recommender-systems papers of the deep-learning era. It
reframed collaborative filtering as learning an interaction
function and gave the field the GMF / MLP / NeuMF vocabulary.

**The embedding-plus-MLP template.** The NCF recipe — learn an
embedding per categorical entity, concatenate, push through an
MLP — is the backbone of nearly every deep recommender that
followed: YouTube's deep candidate and ranking networks, the
deep half of **Wide & Deep** (Google Play), **DeepFM**, and the
ranking stacks at essentially every large platform.

**Two-tower retrieval (next in this track).** NCF's serving
problem — you cannot ANN-index an arbitrary neural score —
directly motivates the two-tower design, where user and item
each get an independent tower and the final score is forced back
to a dot product *so that* fast retrieval works. Two-tower is
NCF's lesson applied at web scale.

**Factorisation Machines and their neural descendants.** GMF's
"weighted element-wise product" sits next to FMs, which model
all pairwise feature interactions with shared embeddings; DeepFM
and xDeepFM fuse that idea with an MLP in the same spirit as
NeuMF fuses GMF and MLP.

**Log loss + negative sampling everywhere.** The
binary-classification-with-sampled-negatives framing is the same
machinery behind word2vec's skip-gram with negative sampling,
contrastive learning, and most retrieval training. NCF is a
clean place to see it.

**Pre-training and fusion.** NeuMF's "train the parts, then fuse
and fine-tune" is an early, concrete instance of the
pre-train-then-combine pattern that now dominates large-scale
ML.

---

## When NOT to use Neural Collaborative Filtering

**When retrieval has to be fast over millions of items.** This
is the big one. NeuMF's score is not a dot product, so you
cannot use an ANN index to fetch candidates — you would have to
score every item with a forward pass. For large catalogues use a
two-tower / dot-product model for retrieval and save the
neural-interaction model for re-ranking a short candidate list.

**When the dot product already fits.** If the true interaction
is mostly linear — and a great deal of collaborative signal is —
matrix factorisation or GMF will match a neural model at a
fraction of the cost and with far easier training. Our own
results make the point: GMF tied NeuMF on HR@10. Reach for the
MLP only when you have evidence of non-linear structure and
enough data to learn it.

**When data is scarce.** Neural interaction functions have more
parameters and more ways to overfit. On small or very sparse
datasets, regularised matrix factorisation is usually the safer,
stronger choice. NCF wants volume.

**When you need cold-start handling out of the box.** Like MF,
NCF learns an embedding *per known user and item* and has
nothing to say about an entity it never saw in training.
Cold-start still needs side features (content, demographics) —
which, incidentally, slot naturally into the MLP's input, one of
NCF's genuine advantages over pure MF.

**When you need explainability.** "The fused output layer
weighted the MLP's third hidden unit at 0.4" explains nothing.
If recommendations must be justified to users or auditors,
simpler or content-based models are more transparent.

---

## What comes next

Part 3 of the Recommender Systems track is **Two-Tower
Retrieval** — the architecture that takes NCF's central lesson
(learn rich embeddings) but deliberately keeps the final scoring
a dot product, so that retrieving the best items from a
hundred-million-item catalogue becomes an approximate
nearest-neighbour lookup instead of a hundred million forward
passes. It is the model that makes web-scale recommendation
actually serveable, and it is the reason every large platform
runs a *retrieve-then-rank* pipeline: a cheap two-tower model to
fetch candidates, a heavy NCF-style model to rank them.

Then comes **Sequential Recommenders**, which drop the
assumption that a user's interactions are an unordered set and
model them as a sequence — predicting the *next* interaction
from the history.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**neural_collaborative_filtering.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/07-recommender-systems/02-neural-collaborative-filtering/neural_collaborative_filtering.py)

Run it with:

```bash
pip install numpy
python neural_collaborative_filtering.py
```

It needs only `numpy`. The script generates a synthetic
implicit-feedback dataset with a deliberately non-linear
ground-truth interaction, then builds all three NCF models —
GMF, MLP, and NeuMF — from scratch, including embedding tables,
ReLU hidden layers, a hand-written Adam optimiser, negative
sampling, and full backpropagation. It trains each model with
binary cross-entropy and evaluates them with the leave-one-out
HR@10 / NDCG@10 ranking protocol against a popularity baseline.
The headline insight worth pinning to the wall: **Neural
Collaborative Filtering replaces matrix factorisation's fixed
dot product with a learned interaction function; GMF is the dot
product generalised, the MLP learns non-linear interactions the
dot product cannot, and NeuMF fuses both — but the dot product
is a strong baseline, the pure MLP is not automatically better,
and the fusion's win comes with the loss of cheap dot-product
retrieval that the next model exists to recover**.

---

*This is Part 2 of the Recommender Systems track in the Algorithms in Python series. The companion script `neural_collaborative_filtering.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 1 introduced Matrix Factorisation — the dot product NCF generalises. Part 3 of this track will look at Two-Tower Retrieval — keeping the dot product on purpose, so web-scale retrieval stays fast.*
