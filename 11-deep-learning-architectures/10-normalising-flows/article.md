# Normalising Flows — The Model That Prices Every Point in the Plane

### *Algorithms in Python --- Deep Learning Architectures, Part 10*

---

The section's generative quartet closes with the purist. The GAN
(Part 7) never wrote a loss down. The VAE (Part 8) wrote one, but
only a bound. Diffusion (Part 9) regressed its way around the
problem. The **normalising flow** (RealNVP: Dinh et al., 2017)
refuses every one of those compromises: build the map from noise
to data out of *invertible* steps only, and the model computes
the exact probability of any point you show it — to the digit:

```
log p(x)  =  log N( f(x); 0, I )  +  log |det df/dx|
```

Read it as bookkeeping. Push the data point `x` through an
invertible map `f` to noise-space; score it under a standard
Gaussian; then add the log-determinant of the Jacobian — the
exact accounting of how much `f` stretched or squeezed space on
the way. Every unit of probability is conserved; nothing is
approximated; the training loss *is* the log-likelihood.

The catch is the construction: every layer must be invertible,
and its Jacobian determinant affordable. The **affine coupling**
layer is the trick that made this practical. Split the
coordinates; one half passes through untouched and sets the
scale and shift for the other:

```
y_b = x_b * exp(s(x_a)) + t(x_a),      log|det| = s(x_a)
```

Inverting it is one subtraction and one division — the
passthrough half is unchanged, so `s` and `t` can be recomputed
exactly. The log-determinant is read off the wire. `s` and `t`
come from an ordinary little MLP that never needs inverting,
because the *wiring*, not the network, carries the invertibility.
Stack the couplings, alternating which coordinate leads, and
noise warps into data. Sampling is one inverse pass; encoding is
one forward pass; both are exact.

The companion script hand-derives the backward pass through the
couplings *and* the log-det terms, audited at startup (1.86e-08
against central differences), and the stage is Parts 7–9's — the
two spirals, the ring, the shared yardsticks — with the GAN, VAE
and diffusion model retrained live for the quartet's final table.

---

## A worked example: the number, the tax, the quartet

### The exact number

```
     step   eval NLL   q@0.10   q@0.05   coverage
     2000     1.247     95.7%    53.4%     98.3%
     4000     1.297     90.1%    50.5%     98.3%
     6000     1.217     94.3%    53.7%    100.0%
     8000     1.208     96.7%    54.9%     98.3%
```

The loss is exact — and the eval NLL still wobbles (1.297 at
step 4000 after 1.247), because exactness of the *objective*
does not make the *optimisation* monotone. The number is honest;
the descent is still SGD. Then the receipts, which no other
architecture in this section can print:

```
    integral of p(x) over the plane : 1.0000
    round trip max |x - f_inv(f(x))|: 3.94e-15
    z of real data: mean [0.032 0.016], var [0.864 1.221]
```

Three different kinds of exactness. The model's density,
numerically integrated over the plane, comes back **1.0000** — a
genuine probability distribution, verified by quadrature, not by
faith. The round trip `x → z → x` reproduces the data to
3.94e-15 — floating-point roundoff. Part 8's VAE reconstructed
with error 0.002 and called it good; the flow is the autoencoder
that *cannot lie*, because reconstruction is algebra, not
learning. And real data pushed to noise-space lands almost
exactly on the standard Gaussian the model promised.

Then it prices individual points:

```
    log p ( 0.12, 0.00)  on an arm        :      -1.935
    log p ( 0.00, 0.00)  between the arms :      -1.988
    log p ( 2.00, 2.00)  far off-manifold : -147701.380
```

A point two units off the manifold is charged into oblivion —
log-probability minus one hundred and forty-seven *thousand* —
exactly as it should be. But look at the middle row. The empty
spot *between* the arms is priced at nearly the same as a spot
*on* an arm. That is not a bug in training. Hold the thought.

### The topology tax

A flow is a continuous bijection: it can bend the plane, stretch
it, fold it — but never **tear** it. Depth buys precision:

```
    couplings   eval NLL   q@0.10   off-manifold
        2        1.326     89.6%       7.3%
        4        1.211     95.0%       1.8%
        8        1.208     96.7%       1.1%
       16        1.224     94.7%       2.5%
```

Two couplings are crude, eight polish stray mass down to 1.1%,
sixteen buy nothing further. But no depth can do what the ring
of eight Gaussians demands: cut one connected blob of noise into
eight islands. Twelve couplings, three seeds:

```
    seed 0: modes 8/8   on-mode mass 45%
    seed 1: modes 8/8   on-mode mass 33%
    seed 2: modes 8/8   on-mode mass 58%
```

Every mode, every seed — a flow *cannot* drop an island, because
dropping one would tear the map; mode collapse is topologically
forbidden. But 42–67% of the mass lands off-mode, in filaments
of probability connecting islands the data never connected. And
because the flow prices points exactly, the tax can be itemised:

```
    p at a mode centre  :  1.22e+00
    p BETWEEN two modes :  1.11e+00
    p at the ring centre:  5.56e-06
    TRUE p between modes:  9.51e-13
```

The model prices the empty midpoint between two modes at 1.11 —
91% of a mode centre's 1.22 — where the true density is
9.5e-13. **Twelve orders of magnitude of topology tax**, printed
to the digit by the only architecture capable of printing it.
This is the article's punchline pair: the flow's superpower and
its signature failure are the same fact. The map must be smooth,
invertible and whole — so it can tell you the exact price of
anything, including the exact price of its own inability to
tear.

### The quartet

Four articles, four generative architectures, one stage — GAN,
VAE and diffusion retrained live with their own seeds and
budgets, diffusion at its best 20-pass setting:

```
                          passes   q@0.10   q@0.05   coverage
    GAN (Part 7)             1     98.7%    58.0%     81.7%
    VAE (Part 8)             1     89.5%    51.3%     98.3%
    diffusion (Part 9)      20     98.7%    55.6%    100.0%
    flow (this article)      1     96.7%    54.9%     98.3%
    real data                -     99.5%    77.3%     93.3%
    pure noise (floor)       -     40.7%    22.9%     95.0%
```

The section's generative story in six rows. The GAN: sharpest,
cheapest, 18% of the arc missing — the mode-seeking game. The
VAE: everything covered, softly — the honest bound. Diffusion:
sharp *and* covering at twenty passes — the trade refused, paid
in compute. The flow: one pass, honest middle scores on every
yardstick — and the only row that can tell you, exactly, how
probable any of its opinions are. Four answers to the same
question: where should the difficulty live? In the opponent, in
the encoder, in the schedule — or in the Jacobian.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Everything a flow does costs one pass: sampling is one inverse
sweep through the couplings, `O(K · h²)`; exact density
evaluation — the thing nobody else sells at any price — is one
forward sweep of the same cost; training needs no sampling loop,
no second network, no schedule. The structural bill is paid
elsewhere: coupling layers move only half the coordinates per
step, so flows need depth `K` where a free architecture needs
less, and every hidden representation must stay the same
dimension as the data — no bottlenecks, no compression, which at
image scale makes flows memory-hungry (Glow's multi-scale
factoring exists precisely to fight this). The general lesson of
the table: constraints move cost around, never remove it. The
GAN paid in stability, the VAE in blur, diffusion in passes; the
flow pays in architecture.

---

## Where the exact numbers went

Flows never took the image-generation crown — RealNVP and Glow
made striking faces in 2017–18, but the topology tax and the
equal-width constraint kept them behind GANs, then diffusion.
They won everywhere the *number* matters more than the picture.
Density estimation and anomaly detection: score a point, flag
the cheap ones — the receipt DEMO 1 printed is the product.
Simulation-based inference in physics and astronomy, where
posteriors must be exact and reusable. Variational posteriors
inside other models — a flow can replace the VAE's diagonal
Gaussian with something honest about correlations. Molecular
Boltzmann generators, where free-energy differences demand real
log-likelihoods. And the idea's descendants run the field:
continuous flows became neural ODEs; flow matching — training a
continuous flow with a diffusion-style regression instead of a
Jacobian — now trains some of the strongest image and video
models, a literal merger of this article's mechanism with Part
9's loss. The purist's constraint relaxed, the purist's exact
change kept.

---

## What comes next

This closes Section 11's generative quartet, and the section
itself is one article from complete. Section 12 turns to
**self-supervised learning** — Part 1, Contrastive Learning
(SimCLR / CLIP): what representations can be learned from no
labels at all, just agreement between two views of the same
thing — the idea that quietly powers the encoders every model
in this section takes for granted.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**flows.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/11-deep-learning-architectures/10-normalising-flows/flows.py)

Run it with:

```bash
pip install numpy
python flows.py
```

It needs only `numpy` and runs in about ten minutes — the three
ring flows are most of the bill. Everything is from scratch: the
coupling layers with their hand-derived backward pass through
the log-det terms, the exactness receipts (quadrature,
round-trip, pointwise pricing), the depth sweep, the ring runs,
and live retrains of Parts 7–9's GAN, VAE and diffusion model
for the quartet table. The headline insight worth pinning to the
wall: **build the generator out of invertible steps and the
model becomes a genuine probability density — integrating to
1.0000 by quadrature, reconstructing to 3.94e-15, pricing an
off-manifold point at log p = -147701 — but the same
unbreakable map cannot tear noise into islands, so it holds
8/8 ring modes on every seed while pricing the empty gap
between modes at twelve orders of magnitude above the truth:
the exact-likelihood superpower and the topology tax are one
and the same fact, and the quartet table says the rest — four
architectures, four places to hide the difficulty: the
opponent, the encoder, the schedule, or the Jacobian**.

---

*This is Part 10 of the Deep Learning Architectures track in the Algorithms in Python series, closing the generative quartet begun with [Part 7](https://medium.com/p/084802e78ace)'s GAN, [Part 8](https://medium.com/p/eb97ec9742ff)'s VAE and [Part 9](https://medium.com/p/7bffa6259472)'s diffusion model — all three retrained live inside the companion script `flows.py` in the [series repository](https://github.com/grahamroy/algorithms-in-python), on the spirals that [Part 1](https://medium.com/p/9c0d36baa5bc) classified. Section 12 opens with Contrastive Learning.*
