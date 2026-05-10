# K-Nearest Neighbours — When Memorising Beats Modelling

### *Algorithms in Python --- Supervised Learning, Part 4*

---

In Part 3 we built a Naive Bayes classifier that did all of its
work at training time — count classes, count features per class,
build a per-class log-probability table — and then made
predictions in microseconds with a tiny vector-vector dot
product. Today we look at the algorithm that takes the *exact
opposite* approach: zero training, all the work at prediction
time. There is no model to fit; the training set *is* the model.
Asked to label a new example, it digs through every training
example, finds the closest ones, and lets them vote.

The algorithm is **K-Nearest Neighbours** (KNN). It is the most
intuitive supervised-learning algorithm ever proposed: things
that look like other things tend to *be* other things. It
pre-dates the field of machine learning as a name — the basic idea
was formalised by Fix and Hodges in 1951, before anyone called
this "ML" — and it has never gone away. Modern recommender
systems, retrieval-augmented language models, image search, and
half of the anomaly-detection literature still build on KNN at
their core, just with much faster ways of finding the nearest
neighbours.

This article builds KNN from first principles. We will walk
through the algorithm, the three big design choices (distance
metric, *k*, vote weighting), implement it from scratch in numpy,
look at why it breaks in high dimensions and how the KD-trees
from [Foundations Part 8 — Trees](https://medium.com/@grahamjroy/trees-hierarchical-structure-for-decisions-search-and-database-indexes-64767b20394f)
and the ANN indexes from
[Foundations Part 12 — Vector Indexes](https://medium.com/@grahamjroy/vector-indexes-the-data-structures-behind-vector-search-9ca81830f658)
fix the worst of it, and finish with the surprisingly long list
of places KNN is still doing real work in production.

---

## The KNN algorithm

KNN is a *lazy learner*. The training step is just storing the
data:

```
fit(X_train, y_train)  →  remember (X_train, y_train)
```

That is it. No parameters are estimated, no objective is
minimised, no convergence is checked. Training is O(n · d) only
because we have to *read* the data into memory.

The prediction step is where the work happens. Given a new point
`x`:

```
1.  Compute the distance from x to every training point.
2.  Sort the training points by distance, ascending.
3.  Take the first k.
4.  Classification → majority vote of their labels.
    Regression     → mean (or weighted mean) of their labels.
```

For classification, ties are usually broken by the closest
neighbour, by lowest class index, or by reducing *k* by one. For
regression, the prediction is simply the mean target value of
the *k* nearest training points.

The whole algorithm fits in five lines of numpy:

```python
def predict_one(x_query, X_train, y_train, k=5):
    d = np.linalg.norm(X_train - x_query, axis=1)
    idx = np.argpartition(d, k)[:k]
    return Counter(y_train[idx]).most_common(1)[0][0]
```

Three things make KNN unusually attractive:

- **No training time.** You can ship a KNN classifier the moment
  you have labelled data. There is no hyperparameter search for
  weights, no convergence to babysit.
- **Trivially incremental.** Adding a new labelled example is
  appending to an array. No retraining loop.
- **Local decision boundary.** The decision surface bends to fit
  the data exactly — it is non-parametric, with no global form
  imposed.

The cost is paid at prediction time, and as we will see, that
cost can be brutal as the data grows.

---

## Three design choices

Three knobs control how a KNN classifier behaves: the distance
metric, the value of *k*, and how the votes are weighted.

### Distance metric

The metric defines what "near" means, and this is more
consequential than people realise. A model with the wrong
distance is solving the wrong problem.

**Euclidean (L2)**. The straight-line distance, `√Σ (x_i - y_i)²`.
The default for continuous features that all live on roughly the
same scale. Standardise your features first or one large-scale
feature dominates the distance.

**Manhattan (L1)**. The grid distance, `Σ |x_i - y_i|`. Less
sensitive to outliers than L2 because it doesn't square the
gaps. Often a better default in high-dimensional data, where
L2's behaviour degrades faster.

**Cosine distance**. `1 - (x · y) / (‖x‖ · ‖y‖)`. Compares the
*direction* of the vectors, ignoring magnitude. The right choice
for text (TF-IDF vectors), embeddings from neural networks, and
anywhere the magnitude is an artefact of normalisation.

**Hamming**. The fraction of positions where two vectors differ.
For categorical or binary features (genome variants, feature
flags, set-membership encodings).

**Mahalanobis**. Like Euclidean but pre-multiplied by the inverse
covariance matrix of the training data. Treats correlated
features as a single "direction" — the right metric when you
care about statistical distance rather than geometric distance.

The metric is part of the model. Switching from Euclidean to
cosine on the same dataset can flip predictions wholesale.

### Choice of k

The value of *k* sets the bias-variance tradeoff cleanly:

- **k = 1** has zero bias and high variance. Every training point
  defines its own region; the boundary is wiggly and overfits
  individual noisy points.
- **k = n** (the full training set) has high bias and zero
  variance. Every prediction is just the majority class —
  effectively a constant model.
- **k between** is where the action is. Usually we pick *k* by
  cross-validation. Common sweet spots are 5, 7, or 11 for
  small/medium datasets; the rule of thumb `k ≈ √n` is decent for
  a starting point.

Odd values are conventional for binary classification to avoid
ties.

### Vote weighting

By default each of the *k* neighbours casts an equal vote, but
this throws away signal. A neighbour at distance 0.01 is almost
certainly the right answer; one at distance 4.7 is barely better
than chance. **Distance-weighted voting** weights each neighbour
by `1 / (d + ε)`:

```
score(class c) = Σ_{i ∈ k-nearest, y_i = c}  1 / (d_i + ε)
```

The tiny `ε` avoids division by zero when a query coincides
exactly with a training point. Distance weighting smooths the
boundary, gives finer-grained probability estimates, and is
usually a free improvement.

---

## A worked example

The companion script generates a 2D synthetic dataset (300
points, 3 well-separated Gaussian classes) and walks the
classifier through it. Here is the from-scratch core:

```python
class KNearestNeighbours:
    def __init__(self, k=5, weights="uniform"):
        self.k = k
        self.weights = weights  # "uniform" or "distance"

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)

    def predict(self, X_query):
        out = []
        for x in X_query:
            # 1. distances to every training point
            d = np.linalg.norm(self.X - x, axis=1)
            # 2. indices of the k smallest
            idx = np.argpartition(d, self.k)[:self.k]
            labels = self.y[idx]
            if self.weights == "uniform":
                out.append(Counter(labels).most_common(1)[0][0])
            else:
                w = 1.0 / (d[idx] + 1e-12)
                scores = {}
                for lab, wi in zip(labels, w):
                    scores[lab] = scores.get(lab, 0.0) + wi
                out.append(max(scores, key=scores.get))
        return np.array(out)
```

That is the entire algorithm. Run it on the demo dataset and you
get:

```
DEMO 1 --- KNN from scratch on 3-class synthetic data
  Training set: 400 examples, 2 features
  Test set    : 100 examples
  k = 5, weights = uniform
  Accuracy    : 0.870  (87/100)

DEMO 2 --- Same data, scikit-learn KNeighborsClassifier
  Accuracy    : 0.870  (87/100)

DEMO 3 --- Bias-variance sweep over k
  k =   1  acc = 0.860  (high variance, jagged boundary)
  k =   3  acc = 0.870
  k =   5  acc = 0.870
  k =   7  acc = 0.880
  k =  15  acc = 0.890
  k =  31  acc = 0.880
  k =  75  acc = 0.890
  k = 199  acc = 0.880  (high bias, oversmoothed boundary)
  k = 399  acc = 0.370  (majority-class baseline)
```

The scikit-learn predictions match ours on every test example.
The bias-variance sweep shows the textbook U-shape: tiny *k*
overfits a little, the middle (around k = 15–75) is the sweet
spot, and as *k* approaches the size of the training set the
classifier collapses into "always predict the majority class"
and accuracy tumbles to 37%.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Three notes worth pulling out. **Training is O(n · d) only
because we have to read the data into memory** — there is no
fitting step. **Brute-force prediction is O(n · d) per query**
which scales linearly in the training-set size; for a million
training examples and a 256-dimensional embedding, that is 256
million floating-point operations *per prediction*. **Memory is
O(n · d)** — the entire training set lives in RAM, no
compression.

This is why production KNN is almost never brute force. KD-trees
([Foundations Part 8 — Trees](https://medium.com/@grahamjroy/trees-hierarchical-structure-for-decisions-search-and-database-indexes-64767b20394f))
bring prediction down to roughly `O(d · log n)` on low-dimensional
data (say `d < 20`), and ball trees push the same idea a little
further. Above ~30 dimensions, both data structures collapse
into something close to brute force, and the practical answer
becomes the approximate-nearest-neighbour indexes from
[Foundations Part 12 — Vector Indexes](https://medium.com/@grahamjroy/vector-indexes-the-data-structures-behind-vector-search-9ca81830f658) —
HNSW, IVF, PQ. They give up exactness for an enormous speed-up,
typically 100× to 1000× faster with 95–99% recall.

---

## The curse of dimensionality

KNN's reliance on distances is also its weakness. In
high-dimensional spaces, distances behave in counter-intuitive ways:
*every* pair of points becomes roughly the same distance apart,
and the very concept of "nearest" loses its meaning.

Pick `d`-dimensional uniform random points in the unit cube. The
ratio of the maximum to minimum pairwise distance approaches 1
as `d` grows. By 100 dimensions, the closest training point and
the farthest training point are almost equidistant from any
query — and a 5-NN classifier is essentially picking 5 random
training points and voting.

There are two ways to fight this. The first is **reduce the
dimension**: PCA, autoencoders, learned embeddings (Word2Vec,
sentence transformers, image encoders) all compress raw features
into a low-dimensional space where distances mean what we want.
The second is **learn the right metric**: metric-learning losses
(triplet loss, contrastive loss, ArcFace) train a neural network
to produce embeddings where same-class pairs are close and
different-class pairs are far. KNN on the learned embedding then
works beautifully — and this is exactly the recipe behind face
recognition, image search, and recommendation systems.

---

## Real-world ML and AI connections

KNN is the algorithm that hides inside other systems:

**Item-item collaborative filtering.** Amazon's seminal 2003
paper *Item-to-Item Collaborative Filtering* describes building
a sparse similarity matrix between products and serving
recommendations as the *k* nearest items to whatever the user
just looked at. The core algorithm is KNN with cosine
similarity. Variants of it still power "you might also like"
panels at every major retailer.

**kNN language models.** Khandelwal et al's 2020 paper *Generalisation
through Memorisation: Nearest Neighbor Language Models* showed
that a transformer LM augmented with a kNN lookup over a giant
datastore of cached `(context, next-token)` pairs measurably
improves perplexity. The model uses learned representations from
the LM as the metric space; KNN does the retrieval. This is the
intellectual ancestor of every retrieval-augmented LLM in
production today.

**Image search and face recognition.** Compute embeddings with a
deep CNN trained via metric learning, store them in an HNSW
index, look up the *k* nearest at query time. Pinterest's visual
search, Google Photos' face grouping, every "find similar
images" feature you have ever clicked — all KNN over learned
embeddings.

**Anomaly detection.** Local Outlier Factor (LOF), k-distance
methods, and isolation-by-distance approaches all reduce to KNN
on the dataset and look at how *unusual* a point's neighbourhood
is. Production fraud-detection systems still ship LOF.

**Few-shot classification.** Prototypical Networks (Snell et al,
2017) embed each class's few-shot examples, average them into a
prototype, and classify new queries by 1-NN against the
prototypes. Modern few-shot learning is mostly variants on this.

**Vector databases.** Pinecone, Weaviate, Qdrant, Chroma, and
the FAISS library all exist because KNN over high-dimensional
embeddings is so useful. The substrate of every modern RAG
system is "embed your query, KNN against your document
embeddings, feed the top-k chunks to the LLM." That is just KNN
with a few extra steps.

The pattern is consistent: the heavy lifting (the embedding) is
done by a model trained for the task; KNN does the final
retrieval over that learned space.

---

## When NOT to use KNN

KNN's simplicity hides several brittle assumptions:

**When you have a lot of training data and tight latency.** A
brute-force KNN over a million training points takes hundreds of
milliseconds per query and grows linearly. ANN indexes help, but
they introduce their own complexity (build time, memory, recall
loss) that often pushes you toward a parametric model
altogether.

**When the data is genuinely high-dimensional with no learned
embedding.** Raw 4096-dimensional bag-of-words, raw
2048-dimensional ResNet activations without metric learning,
genome SNPs — distances are unreliable. Train an embedding
first or pick a different model.

**When features are heterogeneous in scale or type.** A Euclidean
distance over `[age, income, zip-code-as-int]` is meaningless.
Either standardise carefully, build a custom distance, or use a
tree-based model that doesn't care about scale.

**When classes are heavily imbalanced.** A 99-to-1 class ratio
will dominate the *k* nearest neighbours of almost any query
unless you use distance-weighted voting *and* tune *k* carefully.
Tree-based or cost-sensitive models handle this more gracefully.

**When you need to explain *why* a prediction came out a certain
way, and "look at these five training points" is not an
acceptable answer.** KNN's explanations are the neighbours
themselves. That is sometimes wonderful (it is *literally*
case-based reasoning) and sometimes useless (when the neighbours
are high-dimensional embeddings nobody can interpret).

**When you can't keep the entire training set in memory.** KNN
is non-parametric — there is no compressed model. Bigger
datasets mean bigger memory.

---

## What comes next

Part 5 of the supervised-learning track is **Decision Trees** —
the model that abandons distances entirely and learns the
classifier as a sequence of axis-aligned splits. Where KNN's
flexibility comes from local averaging, decision trees get
theirs from recursive partitioning, and where KNN struggles in
high dimensions, decision trees handle it without breaking a
sweat. Decision trees are also the gateway to the ensemble
family — random forests, gradient boosting (XGBoost, LightGBM,
CatBoost) — that wins on most tabular ML benchmarks.

After that we will round out the supervised-learning track and
move to the advanced supervised models (random forests, GBMs,
SVMs).

---

## The complete code

The full script is on GitHub — grab it and run it:

[**knn.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/01-supervised-learning/04-k-nearest-neighbours/knn.py)

Run it with:

```bash
pip install numpy scikit-learn
python knn.py
```

It needs `numpy` and `scikit-learn`. The script generates a 2D
three-class synthetic dataset, fits a from-scratch KNN
classifier, compares against scikit-learn's
`KNeighborsClassifier` (which agrees on every prediction), and
runs a *k*-sweep that reproduces the textbook bias-variance
U-shape — small *k* overfits, large *k* underfits, the middle
wins. The headline insight worth pinning to the wall: **KNN
moves all of the work from training to prediction, the metric
is a hyperparameter that matters as much as *k*, and the
algorithm is now mostly used as the retrieval step on top of a
learned embedding rather than on the raw features**.

---

*This is Part 4 of the Algorithms in Python series, Supervised Learning track. The companion script `knn.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 3](https://medium.com/p/e4b5a43e4e60) covered Naive Bayes. Part 5 will look at Decision Trees — the model that classifies by splitting the feature space rather than measuring distances in it.*
