# Deep Generative Models for SSL — The Story Learns to Bend

### *Algorithms in Python --- Semi-Supervised Learning, Part 9*

---

Part 4 laid out the generative bargain: commit to a story of how
the data was *made* — a density `p(x|class)` for every class —
and suddenly every unlabelled point is evidence, because
unlabelled points still had to be *made by something*. It also
printed the fine print in bold: **the model believes its story,
not your labels**. On that article's Gaussian blobs the story
was true, and eight labels plus EM beat an oracle trained on
fifty times more.

This part breaks the polite version on purpose. Two moons are
not Gaussians — no ellipse drapes over a crescent — so Part 4's
mixture walks into a ceiling it cannot iterate its way through.
Then, instead of abandoning the bargain the way the rest of this
track did (committees, graphs, adversaries), we **upgrade the
story**: each class's density becomes a tiny **variational
autoencoder** — a neural decoder that bends a 1-D latent
variable into a curve in data space. A moon *is* a 1-D curve
plus noise. Same EM loop, same clamp, same eight labels; the
only thing that changes is what the story is allowed to say.

The scoreboard jumps from a capped 81.1% to **98.0%** — the best
inductive result in this track — and the closing demo does
something no discriminative method in this series can: it plays
the model backwards and *draws the data it believes in*.

---

## The upgraded story: a VAE in one breath

A **VAE** (Kingma & Welling, 2013) is a density model built
from two small networks. A *decoder* takes a latent variable
`z ~ N(0, 1)` and maps it to a point in data space — bend and
stretch the latent line until it lies along the data. An
*encoder* runs the other way, mapping each data point `x` to a
Gaussian guess `q(z|x)` over which latent could have produced
it. Exact log-likelihood is intractable, so the VAE trains on
the **ELBO**, a lower bound:

```
log p(x)  >=  E_q [ log p(x|z) ]  -  KL( q(z|x) || N(0,1) )
              reconstruction         stay honest about z
```

Gradients flow through the sampling step by the
**reparameterisation trick** — write `z = mu + sigma * eps` with
`eps ~ N(0,1)`, and the randomness moves out of the path the
derivative takes. The companion script does this by hand: the
decoder's *input* gradient carries the reconstruction error back
to the latent, then into the encoder's two heads. Machinery this
track already owns — Part 8 backpropagated to the input to build
an adversary; here the same trick trains a generator.

Semi-supervised use is Part 4's recipe verbatim, with the ELBO
standing in for the Gaussian log-density:

- **E-step**: responsibilities for every unlabelled point from
  each class-VAE's ELBO; the 8 labelled points are clamped.
- **M-step**: each VAE takes one gradient step on
  responsibility-weighted data.

Two training details matter, and both earned their place by
failure. The naive version — both VAEs initialised randomly,
responsibilities from epoch one — collapsed to 40.2%: two
flexible models each fit *all* the data half-weighted,
responsibilities never separated, and the mixture ended as
symmetric mush. Flexibility broke the identifiability that
rigid Gaussians got for free. The fixes are measured, not
decorative: a **warm start** (each VAE first fits only its own
four labelled points, so the story grows outward from the
labels), and an **anchor** (labelled points get 10× weight in
the M-step, so the clamp reaches the gradients, not just the
responsibilities).

---

## A worked example: the same eight labels, a truer story

The stage is Part 1's exact two-moons data and its five random
8-label draws.

### The wrong story, measured

```
DEMO 1 --- The wrong story: Gaussians on moons
    draw:  83.8%   73.0%   82.2%   83.0%   83.6%    mean 81.1%
```

Two things are true at once, and both matter. Even the wrong
story extracts signal from unlabelled data — 81.1% beats the
77.4% supervised network of Part 8, because pulling each
ellipse toward its 250-point mass fixes more than it breaks.
But the story is a **ceiling**: each Gaussian must drape one
ellipse over a curved moon, its tails claim the other class's
horn, and no amount of EM fixes geometry the story cannot
express. Part 4's fine print, now with teeth.

### The upgraded story, measured

```
DEMO 2 --- The upgraded story: a tiny VAE per class
    draw   Gaussian story   VAE story
      0         83.8%          97.6%
      1         73.0%          98.0%
      2         82.2%          98.0%
      3         83.0%          98.2%
      4         83.6%          98.4%
     mean        81.1%          98.0%

  Mean test log-density (draw 0):
    Gaussian story : -1.81   (exact)
    VAE story      : -1.54   (its ELBO lower bound)
```

Every draw lands at 97.6% or better — including draw 1, the one
that broke self-training (59.6%) and capped the Gaussian story
at 73.0%. This is the strongest inductive scoreboard in the
track: level with Part 5's transductive label propagation
(~98.3%), ahead of VAT's 95.4%, and unlike either it hands you
a full generative model of each class at the end.

The log-density line is the quiet clincher. The VAEs report a
*lower bound* on their density; the Gaussians report their
density *exactly* — and the bound still wins by 0.27 nats per
point. The comparison is rigged against the VAE and it wins
anyway. The story is simply truer.

### Play it backwards

```
DEMO 3 --- The payoff: sweep the latent, walk the moon
    class-0 decoder:
      z = -2.0  ->  (-1.096, +0.121)   radius 1.103
      z = -1.0  ->  (-0.967, +0.400)   radius 1.046
      z = +0.0  ->  (+0.029, +0.944)   radius 0.945
      z = +1.0  ->  (+0.937, +0.276)   radius 0.977
      z = +2.0  ->  (+1.127, -0.188)   radius 1.142

    class-1 decoder:
      z = -2.0  ->  (+2.110, +0.415)   radius 1.113
      z = -1.0  ->  (+1.890, -0.010)   radius 1.026
      z = +0.0  ->  (+1.151, -0.434)   radius 0.946
      z = +1.0  ->  (+0.284, -0.119)   radius 0.946
      z = +2.0  ->  (-0.205, +0.260)   radius 1.229
```

Decode `z` from −2 to +2 and each network **walks its moon end
to end** at radius ≈ 1 from the arc's centre (the true arcs:
radius 1.0, noise 0.15). The 1-D latent became the arc
parameter `t`, though nothing ever showed the model
`(cos t, sin t)` — it rediscovered the data's parametrisation
from 500 dots and eight names. The flaws are honest too: at
`|z| = 2` the decodes over-reach to radius 1.1–1.2, the prior's
tails stretching past the horns. A discriminative model's
mistakes hide in weights; a generative model's mistakes are
*inspectable geometry*.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

**The E-step** costs one ELBO pass per component over all `N`
points — a few tiny matrix multiplies each, times `K` Monte
Carlo samples (the script averages 4 to steady the
responsibilities). **The M-step** is one weighted
forward/backward per component. Everything scales linearly in
points, epochs, and network size; the script's 3,400 epochs on
500 points take about twenty seconds in NumPy.

**Against the Gaussian mixture**: EM with Gaussians has a
*closed-form* M-step and converges in dozens of iterations;
the VAE mixture pays thousands of gradient steps for a story
no closed form can tell. That is the whole trade.

**The knobs**: latent dimension (1 here, and *deliberately* —
matching the data's true manifold is the story's strength),
observation noise `sigma_x` (fixed to the data's known 0.15;
learnable in general), the warm-start length, and the anchor
weight. The last two are identifiability repairs, not tuning —
without them the mixture doesn't underperform, it *dissolves*.

---

## From toy to literature

This two-VAE mixture is the ancestor-in-miniature of Kingma et
al.'s **M2 model** (2014), the paper that made deep generative
SSL respectable: there, class is a latent variable in a single
conditional VAE, the classifier `q(y|x)` is learned jointly,
and unlabelled points marginalise over classes exactly the way
our responsibilities do. Its descendants — semi-supervised
GANs, class-conditional diffusion — carry the same bargain to
images and molecules. In today's practice, consistency methods
(Part 8's family) usually win pure classification benchmarks;
the generative route earns its keep when you want more than a
boundary from the same training run: sampling, density-based
anomaly detection, missing-feature imputation, or a latent
space you can inspect. You pay for the story; you get to *read*
the story.

---

## What comes next

Part 10, the **Transductive SVM**, takes the track's oldest
geometric intuition — the boundary belongs in the low-density
gap — and hands it to the margin machinery of the series' SVM
article: place the widest street that both respects the labels
and avoids ploughing through unlabelled points. Where VAT enforced
that idea with an adversary and this part with a story, the
TSVM states it as pure optimisation.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**deep_generative.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/10-semi-supervised-learning/09-deep-generative-models-for-ssl/deep_generative.py)

Run it with:

```bash
pip install numpy
python deep_generative.py
```

It needs only `numpy` and runs in about half a minute.
Everything is from scratch: the VAE's encoder and decoder with
hand-derived gradients through the reparameterisation trick,
the ELBO, the EM loop with clamped responsibilities, and the
Gaussian-mixture baseline it dethrones. The headline insight
worth pinning to the wall: **deep generative SSL keeps Part 4's
bargain and upgrades the story — swap each Gaussian for a tiny
VAE whose decoder bends a 1-D latent into a curve, train the
mixture with the same clamped EM, and the ceiling breaks: 81.1%
becomes 98.0% on eight labels, the VAE's density lower bound
beats the Gaussian's exact density, and the model can be played
backwards — sweep the latent and it walks the moon it was never
shown — provided you repair what flexibility costs:
identifiability, restored here by warm-starting each class's
story on its own labels and anchoring the labelled points in
every gradient step**.

---

*This is Part 9 of the Semi-Supervised Learning track in the Algorithms in Python series. The companion script `deep_generative.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It upgrades [Part 4](https://medium.com/p/dc0679e0bf4e)'s generative bargain with neural machinery, is evaluated on [Part 1](https://medium.com/p/eeca5accd031)'s exact label draws, and posts the track's best inductive scoreboard ahead of [Part 8](https://medium.com/p/5ad699fa2684)'s VAT. Part 10 will look at the Transductive SVM.*
