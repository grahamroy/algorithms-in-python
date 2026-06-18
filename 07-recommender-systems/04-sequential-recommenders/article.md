# Sequential Recommenders — Predicting What You'll Want Next

### *Algorithms in Python --- Recommender Systems, Part 4*

---

Everything in this track so far has thrown away time. Matrix
factorisation, Neural Collaborative Filtering, and two-tower
retrieval all treat a user's history as an **unordered set**:
the model knows *that* you watched a documentary, a thriller,
and a comedy, but not *in what order*, and not that you watched
the thriller five minutes ago. The prediction is "what does
this user like in general?" — a static taste profile.

But behaviour is a sequence. What you want *next* depends
heavily on what you just did. Someone who has watched three
episodes of a series wants episode four, not a static-taste
recommendation. Someone who just bought a tent is in the
market for a sleeping bag, regardless of their long-run
preferences. Session context — the *order* and *recency* of
recent actions — carries signal that set-based models are
structurally blind to.

**Sequential Recommenders** model the history as an ordered
sequence and predict the next interaction from it. The
problem becomes almost identical to **language modelling**:
treat each item as a token, a user's history as a sentence,
and "predict the next item" as "predict the next word." Every
architecture that conquered language modelling has a direct
recommender analogue — RNNs gave us **GRU4Rec** (Hidasi et
al., 2016), the transformer gave us **SASRec** (self-attentive,
Kang & McAuley 2018) and **BERT4Rec** (masked, Sun et al.
2019). Session-based recommendation, "because you just
watched…", next-basket prediction, and the sequence models
inside modern feeds are all this idea.

This article builds a self-attentive sequential recommender —
a small SASRec — from first principles. We will frame
next-item prediction as autoregressive sequence modelling, walk
through causal self-attention over an item-embedding sequence,
implement the model from scratch in numpy with full
backpropagation, train it with the standard next-item objective,
and evaluate with leave-one-out HR@10 / NDCG@10 against a
"recommend the most popular" baseline and a "recommend the most
recent item's neighbours" heuristic.

---

## Next-item prediction as language modelling

Frame the data exactly like a language model. Each user `u`
has a chronological sequence of item interactions
`S_u = [i_1, i_2, ..., i_n]`. The task: given the prefix
`[i_1, ..., i_{t-1}]`, predict `i_t`. Slide that over the whole
sequence and every position becomes a training example — the
**autoregressive** objective, identical to next-token
prediction in a GPT-style model.

```
Input :  [i_1, i_2, i_3, ..., i_{n-1}]
Target:  [i_2, i_3, i_4, ..., i_n]
```

Position `t` sees only items `1..t` (a *causal* constraint — no
peeking at the future) and predicts item `t+1`. At serving
time, feed the user's full history and read off the
distribution over items for the next position; the top-k are
the recommendations.

This framing immediately inherits the entire language-modelling
toolbox: embeddings, positional encodings, self-attention,
causal masking, softmax over a vocabulary (here the item
catalogue). The only real differences from text are that the
"vocabulary" is the item set (often millions, needing sampled
softmax) and that there is no natural-language pretraining to
lean on.

---

## Self-attention over an item sequence

SASRec's core is a single idea: when predicting the next item,
let the model **attend** to the most relevant items in the
history, regardless of how far back they are. A user's last
action matters most, but an item from twenty steps ago can
still be decisive (the series they are slowly working through).
Self-attention learns those dependencies directly.

The mechanics, per position:

```
1. Embed each item:    x_t = ItemEmbed[i_t] + PosEmbed[t]
2. Project to queries, keys, values:
       q_t = x_t W_Q,   k_t = x_t W_K,   v_t = x_t W_V
3. Attention weights (causal — only over j ≤ t):
       a_{t,j} = softmax_j ( q_t · k_j / √d )
4. Context vector:     c_t = Σ_{j ≤ t} a_{t,j} · v_j
5. Feed-forward + residual → representation r_t
6. Next-item score:    score(r_t, item) = r_t · ItemEmbed[item]
```

Position `t`'s representation `r_t` is a learned,
attention-weighted blend of everything the user did up to `t`, and the
prediction for the next item is the dot product of `r_t` with
each candidate item embedding (note the tying: the same item
embeddings serve as both input and output vocabulary, which
regularises the model and halves the parameters).

The **causal mask** is what makes this a valid sequence model:
position `t` can attend to positions `1..t` but never `t+1`
onward, so the model can be trained on all positions at once
(every prefix predicts its successor) without leaking the
answer. This is the same masked self-attention as a GPT
decoder, applied to items instead of words.

---

## Why attention beats the RNN predecessor

GRU4Rec, the first deep sequential recommender, used a
recurrent network: process items one at a time, carrying a
hidden state. It works, but it has the RNN's weaknesses — the
hidden state is a bottleneck that must compress the entire
history into a fixed vector, long-range dependencies fade, and
training is sequential (slow).

Self-attention fixes all three. There is no compression
bottleneck — every past item is directly accessible via
attention. Long-range dependencies are one attention hop away,
not many recurrent steps. And every position trains in
parallel. The same reasons transformers beat RNNs for language
apply verbatim to sequential recommendation, which is why
SASRec and BERT4Rec displaced GRU4Rec within a couple of years.

The one thing the RNN does naturally and attention does not is
*know the order* — attention is permutation-invariant by
itself. That is what **positional embeddings** (step 1 above)
restore: a learned vector per position that tells the model
*where* in the sequence each item sits.

---

## A worked example

The companion script generates synthetic user sequences with a
deliberately **order-dependent** structure: items live in
latent "genres", and a user's next item tends to follow from
the genre of their *recent* items, not their whole history. A
set-based model that ignores order cannot capture this; a
sequential one can. We train a small single-head SASRec and
evaluate with leave-one-out — hold out each user's last item,
rank it against 99 sampled negatives, report HR@10 and NDCG@10.

```
DEMO --- Sequential recommendation on synthetic sessions
  Users                : 1000
  Items                : 500
  Avg sequence length  : 17
  Model                : 1-layer self-attention, d=32, causal mask
  Eval                 : leave-one-out, 1 + 99 negatives

  Method                         HR@10     NDCG@10
  ---------------------------   --------   --------
  Popularity (baseline)           0.144      0.069
  Most-recent-item neighbours     0.604      0.374
  SASRec (self-attention)         0.788      0.439
```

Three observations.

**Order carries most of the signal.** The "most-recent-item
neighbours" heuristic — a first-order transition model that
recommends whatever most often *followed* the user's last item
in training — already reaches HR@10 0.604, more than four times
the popularity baseline. On order-dependent data, even a crude
use of recency beats a sophisticated set-based model. This is
the structural point of the whole article in one row.

**Self-attention beats the recency heuristic decisively.**
SASRec reaches HR@10 0.788 vs 0.604 — it does not just look at
the last item, it learns *which* past items matter for the next
one and weights them, capturing dependencies a single-step
transition model misses. The held-out item lands in the top 10
nearly four times in five, and the NDCG edge (0.439 vs 0.374)
shows it also ranks the right item *higher*, not just inside
the window.

**The popularity baseline collapses.** HR@10 0.144 — close to
the 0.10 you would expect from 1-in-10 sampling against 99
negatives. When the signal is sequential, a static "most
popular" recommendation is almost worthless, which is exactly
why session-based and next-item models exist.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Sequential models pay for their expressiveness in the
sequence-length term:

**Self-attention is `O(L² · d)`** per sequence, where `L` is
the sequence length and `d` the embedding dimension — every
position attends to every earlier position, the same quadratic
cost that limits transformer context windows. For the short
sequences typical of recommendation (`L` capped at 50–200 most
recent items) this is cheap; the cap is a standard engineering
choice.

**Training** is `O(n_seq · L² · d)` for `n_seq` sequences,
parallelised across positions and sequences on a GPU — the
parallel-over-positions property is exactly why attention
replaced the RNN.

**Serving** is one forward pass over the user's (truncated)
history, `O(L² · d)`, producing the next-item representation
`r`. Then scoring `r` against the catalogue is the *same*
retrieval problem as the rest of this track: a dot product per
item, so for large catalogues you reach straight back for the
ANN index from Part 3. SASRec's output is a single query
vector — two-tower retrieval applies directly.

**Memory** is `O(L² )` for the attention matrix plus
`O(m · d)` for item embeddings.

The recommendation-specific trick: cap `L`. A user with ten
years of history is truncated to their most recent ~100 items,
which bounds the quadratic cost and, conveniently, matches the
intuition that ancient history rarely predicts the next click.

---

## Real-world ML and AI connections

**SASRec and BERT4Rec in production.** Self-attentive
sequential models are the standard modern approach to
session-based and next-item recommendation. SASRec (causal,
GPT-style) and BERT4Rec (bidirectional, masked-item training)
are the two reference architectures, widely deployed and
extended.

**YouTube, TikTok, and feed ranking.** Modern feed and
short-video systems lean heavily on sequence models of recent
watch history — "what you engaged with in this session"
dominates the next recommendation. The sequential signal is
why these feeds feel responsive within a single session.

**E-commerce next-basket prediction.** "Customers who bought
these items, in this order, next bought…" is a sequential
problem. Alibaba, Amazon, and most large retailers run
session-based sequence models for in-session recommendation.

**The language-model convergence.** Sequential recommendation
has converged architecturally with language modelling — same
transformers, same causal masking, same next-token objective.
Recent work treats recommendation explicitly as a generative
sequence task (generative retrieval, "semantic IDs",
LLM-as-recommender), pushing the analogy all the way.

**Session-based recommendation for cold users.** Sequence
models shine for logged-out or new users with no long-term
profile but an active session — the within-session sequence is
all the signal there is, and a sequential model uses it
directly where set-based models have nothing.

**Retrieval still applies.** A sequential model produces a
next-item *query vector*; turning that into recommendations
over a huge catalogue is the two-tower retrieval problem from
Part 3. The sequence model is the "user tower" with a
time-aware history encoder — the two ideas compose.

---

## When NOT to use sequential recommenders

**When order genuinely doesn't matter.** Some domains are
close to order-invariant — long-run taste dominates and
recency adds little (a user's film-genre preferences shift
slowly). There, matrix factorisation or two-tower over the
interaction set is simpler and just as good. Sequence models
earn their cost only when *sequence* carries signal.

**When sequences are too short.** A user with two or three
interactions gives the attention mechanism almost nothing to
work with. Below a handful of events, fall back to set-based
or content models; sequential models want history.

**When you need maximum retrieval throughput at huge scale.**
The `O(L²)` encoder per request is heavier than a single
two-tower forward pass. Many systems use a sequential model to
produce the query vector but still rely on two-tower-style ANN
retrieval downstream, rather than scoring the catalogue with
the sequence model directly.

**When interpretability matters.** Attention weights are
suggestive but not faithful explanations. "We recommended this
because attention head 1 weighted your third-last item at 0.3"
is not an account you can give a user or an auditor.
Rule-based or content-based systems are more transparent.

**When training data or compute is scarce.** Sequential
transformers have more parameters and more ways to overfit
than MF, and they want volume and GPUs. On small datasets a
simple recency heuristic — as our own results show — is a
remarkably strong, cheap baseline.

---

## What comes next

This is the final article in the **Recommender Systems** track.
Four models, each fixing the previous one's limitation: matrix
factorisation (the latent-factor dot product), neural
collaborative filtering (a learned interaction function),
two-tower retrieval (deep embeddings that still index for
web-scale serving), and sequential recommenders (modelling
order and recency). Together they are the backbone of the
recommendation systems that shape what billions of people
read, watch, and buy.

The next track is **Reinforcement Learning** — a different
paradigm entirely. Where everything so far has learned from a
fixed dataset of past behaviour, RL learns from *interaction*:
an agent takes actions, receives rewards, and improves its
policy over time. It opens with Q-Learning, the foundational
value-based method, and builds toward the deep RL that plays
games and controls robots.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**sequential_recommender.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/07-recommender-systems/04-sequential-recommenders/sequential_recommender.py)

Run it with:

```bash
pip install numpy
python sequential_recommender.py
```

It needs only `numpy`. The script implements a single-head
self-attentive sequential recommender (a small SASRec) from
scratch — item and positional embeddings, causal-masked
self-attention, a feed-forward block, tied input/output item
embeddings, and full backpropagation with a hand-written Adam
optimiser — trains it with the next-item objective on
synthetic order-dependent sessions, and evaluates HR@10 /
NDCG@10 against a popularity baseline and a recency heuristic.
The headline insight worth pinning to the wall: **sequential
recommenders treat a user's history as an ordered sequence and
predict the next item exactly as a language model predicts the
next word; causal self-attention learns which past items
matter for the next one; on order-dependent behaviour it
decisively beats set-based models, and its output is a query
vector that plugs straight into two-tower retrieval**.

---

*This is Part 4 of the Recommender Systems track in the Algorithms in Python series, and the final article of the track. The companion script `sequential_recommender.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 3 covered Two-Tower Retrieval — whose ANN serving the sequential model's query vector plugs into. The next track opens Reinforcement Learning with Q-Learning.*
