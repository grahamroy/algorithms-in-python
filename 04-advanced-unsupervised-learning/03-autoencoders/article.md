# Autoencoders — Neural Networks That Learn to Reconstruct Themselves

### *Algorithms in Python --- Advanced Unsupervised Learning, Part 3*

---

In Part 3 of the basic Unsupervised track we built PCA — the
linear dimensionality reducer that finds the axes of maximum
variance via SVD of the centred data matrix. PCA is fast,
deterministic, and gives you an interpretable orthonormal
basis. Its single limitation is the word "linear": it can
only find directions that are linear combinations of the
original features. Data that lives on a curved manifold —
images, audio, anything with non-trivial structure — gets
smeared rather than compressed.

**Autoencoders** are the neural-network answer. Two networks
trained jointly: an **encoder** that maps the input to a
low-dimensional *latent code*, and a **decoder** that maps the
code back to a reconstruction of the input. The objective is
to minimise the reconstruction error — the squared difference
between the input and what comes out the other side. The
network is forced to discover, in the latent code, whatever
compressed representation lets the decoder rebuild the input
most accurately.

When the encoder and decoder are linear maps with a
squared-error loss, the optimal solution is *exactly* PCA. When they
are non-linear (typical: a few dense layers with ReLU
activations), the autoencoder learns a non-linear compression
that can capture curved manifolds PCA cannot. And — unlike
t-SNE / UMAP — the encoder is a *parametric function*: once
trained you can apply it to new data with a single forward
pass, no re-fitting.

Autoencoders are the conceptual ancestor of every modern
self-supervised technique. Variational Autoencoders, masked
autoencoders (the trick behind BERT-style pretraining),
denoising autoencoders (the trick behind diffusion models),
and the encoder half of the Transformer all live in the same
family. This article builds the simplest version — a fully
connected autoencoder trained on the digits dataset — and
walks through the key variants and where they end up.

---

## The architecture

An autoencoder is two networks stacked into one optimisation
loop:

```
x  →  Encoder  →  z  →  Decoder  →  x_hat
                  ↑                   ↓
                  |                   |
                  +--- minimise ‖x - x_hat‖² ---+
```

`x` is the input (e.g. a 64-dimensional flattened image).
`z` is the **latent code** — a vector with `d_latent` ≪ `d_input`
dimensions. `x_hat` is the reconstructed input, the same shape
as `x`. The loss is mean-squared reconstruction error.

The encoder and decoder are typically multi-layer perceptrons
(dense layers + non-linearities), with the architecture roughly
symmetric. A common shape:

```
Encoder:  64 → 32 → 16 → d_latent     (with ReLU / Tanh)
Decoder:  d_latent → 16 → 32 → 64     (with ReLU + linear output)
```

The shrinking-then-expanding shape is the *bottleneck* —
dimensions are squeezed down to `d_latent` then expanded back
out. The decoder cannot recover the original `x` perfectly
from a code that smaller than the input; it can only recover
what fits through the bottleneck. Training the network to
minimise reconstruction error therefore forces the bottleneck
to carry as much information about `x` as possible.

---

## The training loop

Standard mini-batch SGD or Adam on the reconstruction loss:

```
for epoch in 1..n_epochs:
    for batch in shuffled(training_data):
        z = encoder(batch)
        x_hat = decoder(z)
        loss = mean_squared_error(batch, x_hat)
        backprop(loss, encoder.params + decoder.params)
        optimiser.step()
```

Standard backpropagation through both networks treated as one
graph. The encoder learns to extract features that make
reconstruction easy; the decoder learns to invert that
extraction. Together they discover a compressed representation
that preserves the input's structure.

After training, the encoder can be used standalone — as a
non-linear feature extractor, a visualisation tool, or the
starting point for downstream classification.

---

## Variants

Several modifications change what the autoencoder learns.

**Denoising autoencoder** (Vincent et al, 2008). Add random
noise to the input before feeding it to the encoder; require
the decoder to output the *clean* original. The bottleneck
must now capture features robust to noise — and the network
learns more useful representations as a side effect. This is
the algorithmic ancestor of diffusion models and of BERT's
masked-token pretraining.

**Sparse autoencoder.** Add an L1 penalty on the latent code,
encouraging most code dimensions to be zero. The active
dimensions become specialised "feature detectors". Used in
classical unsupervised feature learning before deep
pretraining took over.

**Contractive autoencoder.** Add a penalty on the Frobenius
norm of the encoder Jacobian. The encoder is encouraged to
produce locally-stable codes — small perturbations of `x`
should not change `z` much. Robust to noise without the
denoising trick.

**Variational autoencoder (VAE)** (Kingma & Welling, 2014).
The encoder outputs the *parameters of a probability
distribution* over `z` rather than a deterministic code. The
decoder samples from that distribution. A KL-divergence
regulariser keeps the encoded distribution close to a
standard Gaussian. The result: a *generative* autoencoder
you can sample from to produce new data. The conceptual
foundation of every latent-variable generative model.

**Masked autoencoder.** The training task is to predict a
masked-out portion of the input from the unmasked rest.
Vision: He et al's MAE (2021) masks random patches of an
image. Text: BERT masks random tokens. The trained encoder is
the workhorse representation network for downstream tasks.

The simple deterministic autoencoder this article implements
is the simplest member of a very large family.

---

## A worked example

The companion script trains a small autoencoder on the
scikit-learn digits dataset (the same 1797 × 64 dataset we
used in PCA and UMAP). Architecture: 64 → 32 → 8 → 32 → 64,
ReLU activations, Adam optimiser, 300 epochs.

```
DEMO 1 --- Autoencoder from scratch on the digits dataset
  Architecture       : 64 → 32 → 8 → 32 → 64  (ReLU + linear out)
  Optimiser          : Adam, lr=1e-3, batch_size=64
  Epochs             : 300
  Final train loss   : 0.2453
  Final test loss    : 0.4063
```

```
DEMO 2 --- PCA at the same bottleneck size for comparison
  n_components       : 8
  Train MSE          : 0.4392
  Test MSE           : 0.5104
```

```
DEMO 3 --- KNN-in-latent-space classification accuracy
  Autoencoder latent (8d)  KNN accuracy : 0.903
  PCA latent (8d)          KNN accuracy : 0.919
  Raw 64-d pixels          KNN accuracy : 0.983
```

Three observations.

**The autoencoder achieves a lower reconstruction error than
PCA at the same bottleneck size** (0.41 vs 0.51 test MSE on
standardised features). The non-linear encoder discovers a
slightly more compact compression than PCA's linear
projection — modestly so on the digits dataset, where the
structure is mostly linear and PCA already does well.

**Downstream KNN classification tells a different story.**
PCA's linear latent (91.9%) slightly *beats* the
autoencoder's non-linear latent (90.3%). On a small dataset
with linear-ish structure, an over-trained autoencoder can
overfit reconstruction at the cost of discriminative
features. The lesson: **lower reconstruction error does not
guarantee a better latent space for downstream tasks**. The
benefit of autoencoders shows up on larger, more genuinely
non-linear datasets (Fashion-MNIST, ImageNet, audio
spectrograms).

**Training is much slower than PCA.** PCA fits in 0.01
seconds; the autoencoder takes ~10 seconds for 300 epochs.
On large datasets the gap grows — and the trade-off is
typically worth it only when the data has substantial
non-linear structure that PCA cannot capture.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Two costs dominate:

**Training.** Per batch: one forward pass through `L` layers
each of size ~`d` is `O(d²)` per layer, so `O(L · d²)` per
batch. Backprop is the same order. With `n / B` batches per
epoch and `E` epochs, total cost is `O(E · n · L · d² / B)`.
For a small autoencoder on the digits dataset (300 epochs,
1437 train, 64-d input, 4 layers) that is ~milliseconds per
batch, seconds total.

**Encoding new data.** One forward pass through the encoder:
`O(L · d²)` per example. Sub-millisecond on small models, 
linear in the input size for large ones.

Autoencoders scale to large data with mini-batch training. The
deep-learning ecosystem (PyTorch, JAX) provides GPU
acceleration that takes million-sample autoencoders from days
to minutes.

---

## Real-world ML and AI connections

Autoencoders and their descendants are everywhere in modern
ML:

**Self-supervised pretraining.** Every modern
transformer-based language model (BERT, GPT, T5) pretrains with some
form of masked-input reconstruction — the masked autoencoder
recipe. The pretrained encoder is then fine-tuned for
downstream tasks. This is the single biggest application
area for the autoencoder idea.

**Anomaly detection.** Train an autoencoder on "normal" data;
points with high reconstruction error are anomalies. Used in
manufacturing quality control, fraud detection, network
intrusion detection — anywhere "normal" is well-defined and
abnormal is rare.

**Image and video compression.** Neural image codecs (e.g.
Ballé et al's hyperprior models, JPEG-AI) are conceptually
autoencoders with entropy-coded latent representations. The
modern follow-up to JPEG / HEIC.

**Generative modelling via VAE.** Variational autoencoders
produce sampleable generative models from imagery, audio,
text, and molecular structures. The first wave of practical
deep generative models before GANs and diffusion took over.

**Recommendation systems.** Train an autoencoder on user-item
interaction matrices; the latent codes give compact user and
item embeddings for downstream collaborative filtering.

**Drug discovery.** Train a VAE on molecular structures
(SMILES strings or graph representations); sample from the
latent space to generate novel candidate molecules. An active
area of pharma ML research.

**Diffusion models.** Modern image / video / audio generation
(Stable Diffusion, DALL-E 3, Midjourney, Sora) all use a
*denoising* autoencoder trained to predict the noise added
to an image at every diffusion step. The autoencoder
machinery — bottleneck, reconstruction loss, neural-network
parameterisation — is the foundation.

The pattern: deterministic autoencoders are rare in
production today, but their descendants (VAEs, denoising
autoencoders, masked autoencoders) dominate generative ML
and self-supervised pretraining.

---

## When NOT to use autoencoders

**When the data is small.** Neural networks need lots of data.
For `n < 10⁴` PCA + a downstream model is usually faster,
simpler, and just as accurate.

**When you need a linear feature space.** Autoencoders learn
non-linear codes. If downstream tasks require linear
interpretation (regression on the codes, simple feature
importance), PCA is the right tool.

**When training time is a hard constraint.** PCA fits in
milliseconds; even small autoencoders take seconds to
minutes. For production pipelines with tight retrain budgets,
PCA wins.

**When you don't have a clear architecture.** Picking the
right bottleneck size, depth, activation, and regularisation
is an art. For a one-off project without time to tune,
simpler dimensionality reduction is more reliable.

---

## What comes next

Part 4 of the Advanced Unsupervised Learning track is
**Anomaly Detection** as a topic in its own right. Autoencoders
are one of several approaches; Isolation Forest, One-Class
SVM, Local Outlier Factor, and density-based methods all
deserve coverage. The next article surveys the family.

After anomaly detection comes Latent Dirichlet Allocation
(LDA) for probabilistic topic modelling, then the Bayesian /
Probabilistic / Causal Methods track begins.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**autoencoder.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/04-advanced-unsupervised-learning/03-autoencoders/autoencoder.py)

Run it with:

```bash
pip install numpy scikit-learn
python autoencoder.py
```

It needs `numpy` and `scikit-learn`. The script implements a
small autoencoder from scratch in numpy with manual
backpropagation, ReLU activations, and Adam optimisation;
trains it on the digits dataset; compares reconstruction
error against PCA at the same bottleneck size; and uses the
learned latent codes for KNN classification — where PCA's
linear latent narrowly beats the autoencoder's (0.919 vs
0.903), a reminder that lower reconstruction error does not
guarantee a more discriminative latent
space. The headline insight worth pinning to the wall:
**an autoencoder is two networks trained to compress and
reconstruct the input; the bottleneck forces the encoder to
discover useful features; non-linear autoencoders beat PCA
on data with curved manifolds and parameterise the
encoder so it generalises to new data unlike t-SNE / UMAP**.

---

*This is Part 3 of the Advanced Unsupervised Learning track in the Algorithms in Python series. The companion script `autoencoder.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Part 2 of this track covered Gaussian Mixture Models. Part 4 will look at Anomaly Detection — the family of algorithms (autoencoders included) used to flag unusual points relative to a learned model of "normal" data.*
