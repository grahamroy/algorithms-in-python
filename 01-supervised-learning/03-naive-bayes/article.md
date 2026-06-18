# Naive Bayes — When the Wrong Assumption Wins

### *Algorithms in Python --- Supervised Learning, Part 3*

---

In Part 2 we trained a logistic regression by gradient descent on
log loss, and the model that came out was a linear classifier:
sigmoid of a weighted sum of features. Today we look at another
linear classifier that gets to the same shape via a completely
different route — no gradient descent, no iteration, just
*counting*. It is one of the oldest algorithms in machine
learning, it makes an obviously false assumption about the data,
and on a class of problems — especially text classification — it
beats more sophisticated models with embarrassing regularity.

The algorithm is **Naive Bayes**. The "Bayes" part comes from
Bayes' rule, which the model applies directly to compute
`P(class | features)`. The "naive" part is the assumption that
all features are *conditionally independent given the class* — an
assumption that is almost never true in practice. The whole story
of Naive Bayes is *why does the wrong assumption work so well?*

This article builds Naive Bayes from first principles. We will
revisit Bayes' rule, derive the multinomial variant that powers
text classifiers, implement it from scratch in numpy on a small
spam-classification corpus, and finish with the question of when
to reach for it (more often than you might think) and when to
walk away (image features, mostly). By the end the
"counting + Bayes' rule = classifier" pattern will be obvious,
and you will see why every NLP toolkit still ships with a Naive
Bayes baseline.

---

## Bayes' rule, applied to classification

Given features `x` and a candidate class `y`, Bayes' rule says:

```
P(y | x) = P(x | y) · P(y) / P(x)
```

Read it left to right. The probability that the class is `y`
given that we observed features `x` (the **posterior**) equals
the probability of seeing those features given the class
(the **likelihood**) times the prior probability of the class
(the **prior**), divided by the probability of seeing those
features at all (the **evidence**).

For classification we want the most likely class — the *argmax*
over `y` of `P(y | x)`. The evidence `P(x)` is the same for every
candidate class, so it drops out of the comparison:

```
ŷ = argmax_y  P(y | x)
   = argmax_y  P(x | y) · P(y)
```

Two terms to estimate from training data: `P(y)` (just count
classes) and `P(x | y)` (the harder one — modelling the joint
distribution of all features given the class). The hard term is
where the "naive" assumption rescues us.

---

## The independence assumption

Modelling `P(x | y)` properly means modelling how features
co-vary inside each class. With a thousand features, that is a
thousand-dimensional joint distribution per class. Estimating it
from data is hopeless without astronomical amounts of training
data, especially when most features are sparse or rare.

Naive Bayes makes a sweeping simplification:

```
P(x | y)  =  ∏_i  P(x_i | y)
```

The probability of seeing the joint feature vector is the product
of the per-feature probabilities, *as if every feature were
independent of every other one given the class*. This is the
"naive" assumption — and it is wrong. Two features that frequently
co-occur (for example, the words "credit" and "card" in a spam
email) are clearly not independent. But Naive Bayes pretends they
are anyway.

The resulting classifier is:

```
ŷ = argmax_y  P(y) · ∏_i  P(x_i | y)
```

In practice we work in **log space** to avoid underflow when
multiplying many small probabilities:

```
ŷ = argmax_y  log P(y) + Σ_i  log P(x_i | y)
```

Two terms again. `log P(y)` is the log-prior, easy to estimate
from class counts. `Σ_i log P(x_i | y)` is a sum of per-feature
log-probabilities, each of which is also estimated by counting
within the class. The whole classifier is *counting* —
no optimisation loop, no learning rate, no convergence check.

---

## Three flavours

What goes inside `P(x_i | y)` depends on what kind of feature
`x_i` is. Three variants cover the vast majority of practical
uses.

**Multinomial Naive Bayes** assumes each feature is a *count* —
how often a word appeared in a document, how often a click
happened in a session. The likelihood per class is a multinomial
distribution over the vocabulary, and the per-feature probability
is the proportion of that token's count among all token counts in
the class. This is the variant text classifiers use.

**Bernoulli Naive Bayes** assumes each feature is *binary* —
present or absent. A document is represented as a 0/1 vector
indicating which words appeared at least once, ignoring counts.
Useful when presence carries the signal but frequency does not
(short messages, hashtags, categorical flags).

**Gaussian Naive Bayes** assumes each feature is *continuous* and
follows a Gaussian distribution within each class. Estimate the
mean and variance of each feature per class; at prediction time,
plug into the Gaussian density formula. Useful for low-dimensional
numeric features when you want a fast, calibrated baseline.

In practice, multinomial dominates on text and Gaussian shows up
in introductory examples. Bernoulli is rarer but elegant for
short-text or feature-flag problems.

---

## Multinomial Naive Bayes on text

The canonical Naive Bayes problem is **spam classification**. You
have a corpus of emails labelled spam or ham; you want to predict
the label of a new email. The standard representation:

- Tokenise each email into words.
- Build a vocabulary `V` of all distinct tokens across the corpus.
- Represent each email as a vector of length `|V|` whose `i`-th
  entry is the count of token `i` in that email.

This produces an `n_documents × |V|` count matrix. With even a
modest vocabulary it is overwhelmingly zero — exactly the regime
[Foundations Part 11 — Sparse Matrices](https://medium.com/@grahamjroy/sparse-matrices-when-most-of-your-data-is-zero-85cebc669d78)
covered. Scikit-learn's `MultinomialNB` accepts CSR sparse
matrices directly.

The training step is just counting:

```
For each class y:
  P(y)        = (# documents in class y) / (total documents)
  count(w, y) = total occurrences of word w across all class-y documents
  total(y)    = total token count in class-y documents
  P(w | y)    = (count(w, y) + α) / (total(y) + α · |V|)
```

The `α` term is **Laplace smoothing** (also called *additive
smoothing*). It addresses what happens when a word in a test
email never appeared in any training email of class `y`: without
smoothing, `P(w | y) = 0`, and a single zero kills the whole
product. Adding `α` (typically 1) to every count and `α · |V|` to
the denominator ensures every word has a small positive
probability under every class.

Prediction: for a new document `d`, compute the log-posterior for
each class:

```
log P(y | d) ∝ log P(y) + Σ_{w ∈ d} count(w, d) · log P(w | y)
```

…and pick the argmax. That is the entire algorithm.

The companion script implements this from scratch on a small
synthetic SMS corpus (32 messages, half spam, half ham) and
reports its training and test performance:

```
Training Multinomial Naive Bayes from scratch...

  Vocabulary size : 123 unique tokens
  Class priors    : P(spam)=0.500  P(ham)=0.500
  Smoothing alpha : 1.0

Test set predictions:
  message                                            true   predicted
  "free entry to win 100 cash text now"              spam   spam   OK
  "please pick up bread on the way home"             ham    ham    OK
  "urgent call this number to claim your prize"      spam   spam   OK
  "running late will be home in 20 mins"             ham    ham    OK
  "congratulations you won a holiday voucher"        spam   spam   OK
  "can you grab milk thanks"                         ham    ham    OK
  "click here for free ringtones now"                spam   spam   OK
  "see you at 7 for dinner"                          ham    ham    OK

  Accuracy: 1.000  (8/8)
```

Eight test messages, all classified correctly, after fitting on
24 training examples. The same data through scikit-learn's
`MultinomialNB` produces the same predictions.

---

## Why does the wrong assumption work?

The independence assumption is provably wrong on most real data.
"Credit" and "card" are not independent in a spam corpus; "the"
and "a" are not independent in any English text. So why does the
classifier perform well, sometimes outstandingly well, on
real-world tasks?

Three things cooperate.

**Ranking is what matters, not calibration.** The classifier's
output is `argmax`, not the raw probability. Naive Bayes is
notoriously *over-confident* — it routinely returns probabilities
of 0.9999 when the true probability is more like 0.7 — but
the *ordering* of classes is what determines the label. Even when
the joint probabilities are wildly miscalibrated, the
class-with-the-highest-score is often still the right one.

**The errors of the independence assumption tend to cancel.** When
two features are correlated within both classes (say, "credit"
and "card" appear together more in *both* spam and ham than would
be predicted by independence), the resulting bias affects every
class similarly and largely cancels in the argmax. This is the
formal result behind Domingos and Pazzani's 1997 paper *"On the
Optimality of the Simple Bayesian Classifier under Zero-One
Loss"*, which showed Naive Bayes can be *optimal* even when its
assumptions are violated.

**High-dimensional sparse features are nearly independent.** In
text, most documents contain only a tiny fraction of the
vocabulary. The pairwise correlation of any two specific words
across the *full corpus* is small, because most documents have
neither. The naive assumption is far less wrong on sparse data
than on dense data, which is why Naive Bayes shines on text and
struggles on images.

The combination — argmax-only, errors cancel, sparse features —
explains the long-running puzzle of why a model with such a wrong
assumption keeps showing up at the top of practical leaderboards
on certain problem classes.

---

## Big-O and complexity

[[BIG-O TABLE IMAGE]]

Three notes. **Training is O(n · d) — a single pass through the
data.** No iteration, no convergence — Naive Bayes is the fastest
classifier to fit in this entire series. **Prediction is O(d · K)
per example**, where `K` is the number of classes; in text
classification with `K = 2` and `d` small (only the words in the
document, not the whole vocabulary), this is essentially free.
**Memory is O(K · |V|)** for the per-class log-probability tables.
On a 100,000-word vocabulary with 10 classes, that is 1 million
floats — about 8 MB. The model is tiny by any standard.

The contrast with logistic regression: gradient descent has to
iterate, and each iteration is also O(n · d). Naive Bayes
finishes in the time logistic regression takes for a single
epoch. On modest-sized text corpora (millions of documents,
hundreds of thousands of features) Naive Bayes trains in
seconds where logistic regression takes minutes.

---

## Real-world ML and AI connections

Naive Bayes is the algorithm that *won't go away*. It keeps
showing up because it is fast, it is calibrated enough for many
purposes, and it works.

**Spam filtering.** The historical use case. SpamAssassin's
original Bayesian filter, every email provider's first-generation
filter, and the spam classifiers still built into low-resource
mailservers — Multinomial or Bernoulli Naive Bayes. Modern Gmail
and Outlook layer deep models on top, but the Naive Bayes
baseline is still doing real work in the pipeline.

**Sentiment analysis baseline.** Before BERT and the LLM era,
sentiment analysis was overwhelmingly Multinomial NB on n-grams.
Even today, when an engineer says "let me start with a baseline,"
the baseline they fit in 30 seconds is almost certainly NB on
TF-IDF. Beating it by enough to justify a heavier model is the
real bar most fancy methods have to clear.

**Document categorisation.** Newspaper archives, customer-support
ticket triage, legal-document classification — all routinely
deployed as Naive Bayes when the corpus is large, the categories
are clear, and the latency budget is tight. Reuters' famous
`Reuters-21578` benchmark established Naive Bayes as the
near-state-of-the-art for text categorisation through the 1990s
and early 2000s.

**Language identification.** Detecting whether a piece of text is
French, German, or Polish is *almost a solved problem* with
Naive Bayes on character n-grams. Google's `cld3` and many
production language detectors are still trained on this
principle. The features — n-gram frequencies — happen to be
nearly independent in the way Naive Bayes likes.

**Medical and biological classification.** Disease screening from
symptom checklists, clinical-text classification (positive vs
negative chest X-ray reports), variant calling in bioinformatics
— all problems where Naive Bayes' interpretability and small
data footprint matter. The model gives you a per-feature
log-probability that is easy to audit, which is the same property
that keeps logistic regression in regulated industries.

**Real-time / edge classification.** Naive Bayes' tiny memory
footprint and constant-time prediction make it the right choice
when the model has to live on a microcontroller or run inside a
firewall packet inspector. ML on a Raspberry Pi for IoT
classification is often Naive Bayes.

**The "first model to ship" baseline.** The single most common
pattern: train Naive Bayes on day one, ship it as the v1 service,
spend the next quarter exploring deeper models. If they do not
beat NB by a clear margin, you do not ship them. This is true at
both startups and FAANG-scale teams.

---

## When NOT to use Naive Bayes

Naive Bayes is great at what it does, but it is also brittle in
specific ways:

**When features are densely correlated.** Image features
(adjacent pixels), audio features (adjacent samples), tabular
features with strong interaction effects — the independence
assumption is badly violated and the classifier suffers. Use a
tree-based model or a neural network.

**When you need calibrated probabilities.** Naive Bayes is
notoriously over-confident — it returns 0.9999 when the truth is
0.85. If your downstream system thresholds, ranks under
uncertainty, or feeds into a cost-sensitive decision, run
calibration on top (e.g. Platt scaling or isotonic regression),
or just use logistic regression instead.

**When zero-frequency words dominate.** Smoothing rescues *some*
unseen words, but if test documents are full of vocabulary that
never appeared in training, the smoothing assumption is doing all
the work and the predictions are essentially priors. This is
common when training and test corpora come from very different
distributions (training on news, testing on tweets).

**When the features are continuous and non-Gaussian.** Gaussian
Naive Bayes assumes per-class Gaussianity, which fails for skewed,
multi-modal, or heavy-tailed features. Either bucket the features
into bins and treat them as multinomial, or use a different
classifier entirely.

**When you have abundant labelled data and a strong model.**
Naive Bayes' bias does not melt away as you give it more data;
it just plateaus at whatever the independence assumption can
support. With millions of examples and a transformer or
gradient-boosted tree, you will beat NB by a meaningful margin —
just plan to actually beat it, not just run it.

---

## What comes next

Part 4 of the supervised-learning track is **K-Nearest
Neighbours** (KNN) — the simplest distance-based classifier and
a direct callback to the KD-trees from
[Foundations Part 8 — Trees](https://medium.com/@grahamjroy/trees-hierarchical-structure-for-decisions-search-and-database-indexes-64767b20394f).
KNN takes the opposite approach to Naive Bayes: zero training
(just store the data), all the work happens at prediction time
(find the *k* nearest training examples, vote). The contrast
between the two highlights an axis we will return to often —
*model in your fitted parameters* vs *model in your training
set*.

After KNN comes **Decision Trees**, the gateway to the
random-forest / gradient-boosting family that wins on most
tabular ML benchmarks.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**naive_bayes.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/01-supervised-learning/03-naive-bayes/naive_bayes.py)

Run it with:

```bash
pip install numpy scikit-learn
python naive_bayes.py
```

It needs `numpy` and `scikit-learn`. The script implements
Multinomial Naive Bayes from scratch on a tiny spam/ham SMS
corpus, classifies a held-out set, compares against scikit-learn's
`MultinomialNB` (which agrees on every prediction), and prints
the most informative tokens per class — the words that move the
log-posterior the most, like *free*, *text*, *click* on the
spam side and *the*, *at*, *home* on the ham side. The
headline insight worth pinning to the wall: **Naive Bayes is
just counting plus Bayes' rule, runs in a single pass through
the data, and sets the baseline that every fancier text
classifier still has to beat**.

---

*This is Part 3 of the Algorithms in Python series, Supervised Learning track. The companion script `naive_bayes.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 2](https://medium.com/p/c67849455e74) covered logistic regression. Part 4 will look at K-Nearest Neighbours — the lazy classifier that does no training and all of its work at prediction time.*
