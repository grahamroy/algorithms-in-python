# Anomaly Detection — Spotting the Points That Don't Belong

### *Algorithms in Python --- Advanced Unsupervised Learning, Part 4*

---

The first eleven articles in this series have asked the
algorithm to *summarise* the data — find the clusters, find
the directions of variance, reconstruct the input through a
bottleneck. **Anomaly detection** asks the opposite question:
which examples are *unlike* the rest? Which points don't fit
the pattern the rest of the data describes?

The problem is everywhere. Credit-card fraud: 99.95% of
transactions are legitimate; flag the 0.05% that aren't.
Manufacturing quality control: most parts are within tolerance;
catch the defective few. Network intrusion: most packets are
normal traffic; isolate the attack. Medical screening: most
images are healthy; surface the rare positive. In each case
the structure is the same — a vast majority of "normal"
examples and a tiny minority of "abnormal" ones — and the
labels for "abnormal" are either entirely missing or wildly
imbalanced. Supervised classification is the wrong shape;
unsupervised anomaly detection is the right one.

This article is a *survey* rather than a deep dive on a
single algorithm. We will walk through six distinct
families — statistical, density-based, distance-based,
isolation-based, reconstruction-based, and one-class — show what each
one does, when it is the right choice, and compare them
side-by-side on a 2-D toy problem with planted outliers.
The deepest article-length treatment in this series is
**Isolation Forest**, the algorithm most widely deployed in
production fraud and anomaly stacks; the others get briefer
coverage with pointers for further reading.

---

## The six families

Anomaly detection algorithms differ by what they consider
"normal" — and by how they score deviation from it.

### Statistical methods

The simplest approach. Assume the data follows some
distribution (typically Gaussian) and flag points that fall
in the tails.

- **Z-score.** `z_i = (x_i - μ) / σ`. Flag points with `|z| > 3`.
  Works for 1-D normally-distributed data; fails on
  multi-modal or skewed distributions.
- **IQR rule** (Tukey's fences). Flag points outside
  `[Q1 - 1.5·IQR, Q3 + 1.5·IQR]`. More robust than z-score
  to skewed data.
- **Mahalanobis distance.** Multivariate generalisation of
  z-score. Distance from the mean, weighted by inverse
  covariance. Works for elliptical Gaussian data.

These are the right tools for clean 1-D or low-d numeric
data with simple distributions. They fail on anything
non-Gaussian, multi-modal, or non-numeric.

### Density-based methods

Estimate the density of the data; flag points where the
density is low.

- **KDE thresholding.** Fit a kernel density estimator on
  the training data; new points with density below a
  threshold are anomalies.
- **GMM likelihood.** Fit a Gaussian Mixture Model
  (Part 2 of this track); points with low likelihood
  under the fitted mixture are anomalies. Calibrated,
  interpretable, scales modestly.

The big advantage of density methods: they give a
*calibrated probability* — useful for downstream
cost-sensitive decisions. The big disadvantage: density
estimation in high dimensions is hard.

### Distance-based methods

Score each point by how far it is from its neighbours.
Outliers have far-away neighbours.

- **k-NN distance.** Distance to the `k`-th nearest
  neighbour. Simple, interpretable, the default first
  attempt on small datasets.
- **Local Outlier Factor (LOF)** (Breunig et al, 2000).
  Compares the local density around each point to the local
  density around its neighbours. A point is flagged if its
  neighbourhood is much sparser than its neighbours'
  neighbourhoods — handles datasets with varying local
  density that fixed-distance methods cannot.

Distance methods scale to medium-size datasets with k-NN
acceleration but become impractical past `n ≈ 10⁵` without
approximation.

### Isolation-based methods

Anomalies are *easier to isolate* than normal points — fewer
random splits separate them from the rest.

- **Isolation Forest** (Liu, Ting & Zhou, 2008). Build many
  random trees that recursively split the data; the average
  path length to isolate each point is the anomaly score
  (shorter = more anomalous). Sub-linear at prediction
  time, handles high dimensions, and is the workhorse of
  modern production anomaly detection.
- **Extended Isolation Forest** (Hariri et al, 2019).
  Random *hyperplanes* (not axis-aligned cuts) for
  smoother decision boundaries.

Isolation Forest is the single most-used anomaly detector
in 2026. Fast to train, fast to score, surprisingly accurate
across very different domains.

### Reconstruction-based methods

Train a model to reconstruct normal data; anomalies are
points the model cannot reconstruct well.

- **PCA reconstruction error.** Project to top-K
  components, reconstruct, compute MSE. Cheap,
  interpretable, and the default for high-dim data with
  linear structure.
- **Autoencoder reconstruction error.** Same idea with a
  non-linear autoencoder (Part 3 of this track). Better
  for image / audio / sequence data where structure is
  non-linear.
- **VAE-based anomaly scores.** Use the encoded latent's
  log-likelihood under a prior, or the reconstruction
  log-likelihood under the decoder.

Reconstruction-based methods dominate computer-vision
anomaly detection (industrial inspection, medical imaging).

### One-class methods

Train a classifier with only the "normal" class; everything
outside its decision boundary is anomalous.

- **One-Class SVM.** Find the smallest hypersphere (or
  hyperplane, in kernel space) containing the training
  data. Points outside it are anomalies. Sensitive to
  hyperparameter tuning; less common in production now
  than Isolation Forest.

---

## A worked example

The companion script generates 500 inlier points from a 2-D
Gaussian mixture plus 25 planted outliers uniformly
distributed in a larger box, and runs six anomaly detectors
on the data — each scoring every point, then comparing the
top-25 flagged anomalies against the ground-truth outliers.

```
DEMO --- Six anomaly detectors on synthetic data
  Dataset       : 500 inliers (mixture of 2 Gaussians) + 25 outliers
  Top-25 flagged points compared against true outliers

  Detector                      Precision@25      AUC
  ----------------------        ------------    -----
  Mahalanobis distance                  0.76    0.902
  GMM log-likelihood                    0.84    0.968
  k-NN distance (k=5)                   0.84    0.959
  Local Outlier Factor                  0.84    0.970
  Isolation Forest                      0.80    0.970
  One-Class SVM (RBF)                   0.88    0.971
```

Four things to pull out.

**Five of the six methods are clustered between 0.80 and
0.88 precision@25.** They each get 20–22 of the 25
outliers in the top 25 and a handful of false alarms. The
missed outliers are points placed near the boundary of the
inlier mixture — borderline by construction. No detector
gets every outlier; this is realistic on noisy data.

**Mahalanobis distance is the weakest at 0.76 precision and
0.902 AUC.** It assumes a single Gaussian, but the inliers
are a mixture of two. The "valley" between the two
clusters has high Mahalanobis distance from the global
mean even though it's perfectly normal under the actual
two-Gaussian model. The right fix is to use a model that
matches the data structure — which is exactly what GMM
log-likelihood does (jumping to AUC 0.968).

**AUC is the more honest comparison.** Precision-at-K
depends on the choice of K; AUC integrates over all
thresholds. The five non-Mahalanobis methods all sit at
AUC 0.959+ — essentially the same ranking quality, with
the residual differences down to where the borderline
outliers land.

**There is no single winner.** On this 2-D Gaussian-mixture
dataset, OCSVM happens to top the precision leaderboard
but the ranking is fragile across re-seeds. On a
high-dimensional dataset, k-NN becomes unreliable (curse of
dimensionality) and Isolation Forest takes over. On image /
audio data, autoencoder reconstruction MSE wins. Match the
algorithm to the data structure.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The six families have very different cost profiles:

- **Statistical methods** are `O(n · d)` to fit, `O(d)` per
  query. Essentially free.
- **GMM density** is `O(I · n · K · d²)` to fit, `O(K · d²)`
  per query — moderate cost, well-defined.
- **k-NN distance** is `O(n · d)` per query with brute
  force, `O(log n · d)` with k-d trees in low dimensions,
  approximate-NN methods for high dimensions.
- **LOF** is `O(n² · d)` for the all-pairs distance
  computation — the most expensive of the family. Doesn't
  scale to large `n` without sub-sampling or approximation.
- **Isolation Forest** is `O(n · log n)` per tree, with
  the random-tree construction allowing sub-linear
  prediction at scale.
- **PCA reconstruction** is the cost of PCA plus a
  single matrix multiply per query.

For a million-point production deployment, Isolation
Forest is almost always the right choice. For
small-to-medium curated data where interpretability matters,
GMM or LOF win.

---

## Real-world ML and AI connections

**Fraud detection.** Credit-card transactions, account
takeover, money laundering — typically a stacked pipeline:
Isolation Forest for fast cheap filtering, then a deeper
model (gradient-boosted trees or graph neural networks) on
the suspicious subset.

**Network intrusion detection.** Traffic logs scored by
distance-based or density-based methods; anomalies
investigated further.

**Manufacturing quality control.** Image-based defect
detection via autoencoder reconstruction MSE; the
"AnomaLib" library (Intel) is the standard open-source
implementation.

**Medical imaging.** Train an autoencoder or VAE on healthy
scans; high reconstruction error on a new scan flags
pathology.

**Time-series anomalies.** Forecast next-value with an
ARIMA / Prophet / LSTM; large prediction error = anomaly.
Underlies most application-monitoring tools (Datadog,
New Relic).

**Drug safety / pharmacovigilance.** Adverse-event reports
clustered with anomaly methods to surface unusual
patterns.

**Outlier detection in scientific data.** Sensor errors,
mislabelled examples, contamination — anomaly detection
runs before serious analysis on essentially any real
dataset.

---

## When NOT to use unsupervised anomaly detection

**When you have labels for some anomalies.** Switch to
supervised classification with class-imbalance handling
(SMOTE, class weighting, calibrated thresholds). The label
signal is usually stronger than any unsupervised score.

**When you have a clear precise definition of "anomaly".**
Rules-based detection is often simpler, faster, more
auditable.

**When the data is too small.** Below ~100 points,
distinguishing "anomalous" from "tail of normal" is
statistically hopeless. Get more data.

**When you cannot validate the detector's output.** Any
unsupervised score needs human inspection of top
candidates to tune the threshold. Without that loop the
detector ships *something* — but you have no idea if it is
detecting the right something.

---

## What comes next

Part 5 of the Advanced Unsupervised Learning track is
**Latent Dirichlet Allocation (LDA)** — the probabilistic
topic model that decomposes a corpus of documents into a
mixture of topics, with each topic itself a distribution
over words. The mathematical machinery is closely related
to the EM algorithm we used for GMMs in Part 2 and the
variational methods we will encounter in the next track.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**anomaly_detection.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/04-advanced-unsupervised-learning/04-anomaly-detection/anomaly_detection.py)

Run it with:

```bash
pip install numpy scikit-learn
python anomaly_detection.py
```

It needs `numpy` and `scikit-learn`. The script generates a
500+25 inlier/outlier dataset and runs six anomaly
detectors — Mahalanobis, GMM log-likelihood, k-NN distance,
Local Outlier Factor, Isolation Forest, and One-Class SVM —
reporting precision-at-25 and AUC for each. The headline
insight worth pinning to the wall: **anomaly detection is a
family of approaches rather than a single algorithm;
Isolation Forest is the right default for production
scale, LOF and density-based methods for small-medium
interpretability-sensitive work, autoencoder reconstruction
for image and audio data**.

---

*This is Part 4 of the Advanced Unsupervised Learning track in the Algorithms in Python series. The companion script `anomaly_detection.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 3 of this track covered Autoencoders. Part 5 will look at Latent Dirichlet Allocation — the probabilistic topic model that decomposes documents into mixtures of topics.*
