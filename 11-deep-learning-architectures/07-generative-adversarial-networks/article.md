# Generative Adversarial Networks — The Loss Function Is Another Network

### *Algorithms in Python --- Deep Learning Architectures, Part 7*

---

Six articles into this section, every architecture has done the
same job with a different geometry: given a point, a picture, a
sequence, a graph — *name it*. This article pivots the section
from architectures that classify to architectures that
**create**, and it begins with the strangest training signal in
deep learning: for the network doing the creating, *no loss
function is ever written down*.

Instead, two networks play a game (Goodfellow et al., 2014). A
**generator** — call it the forger — maps random noise `z` to a
fake data point `G(z)`. A **discriminator** — the detective — is
a perfectly ordinary binary classifier: real or fake? The whole
arrangement fits in one line:

```
min_G max_D   E_x[ log D(x) ]  +  E_z[ log(1 - D(G(z))) ]
```

The detective maximises; the forger minimises. The detective's
training step is Part 1's bread and butter — binary
cross-entropy on a stack of layers. The forger's step is the
novelty: its gradient arrives *through the detective* — take
the detective's verdict on a fake, backpropagate through the
detective's frozen weights to its input socket, and keep going
into the forger. The detective's mistakes are the forger's
learning signal. Each network is the other's loss function.

That one design choice buys something remarkable — a training
signal for "make this look real" without anyone defining
"real" — and this article measures what it costs: a score that
orbits instead of descends, a gradient that can vanish through
the *opponent's confidence*, and a failure mode — mode collapse
— that turns out to be a race, not a loss-function bug.

---

## The stage: draw the spirals, then a ring

The target distribution is this section's own **two spirals** —
the stage Parts 1 and 2 *classified*. Tonight nobody classifies:
the forger must learn to *draw* them, from 2-d Gaussian noise,
one point at a time. Both players are small two-hidden-layer
MLPs, and both hand-written backward passes are audited at
startup (detective 3.75e-10, forger-through-detective 1.68e-10
against central differences).

Because a GAN has no trustworthy loss to read, the script builds
two external yardsticks — **quality** (fraction of generated
points within 0.10 of the true spiral curve) and **coverage**
(how many of 60 bins along the arms receive points) — and
anchors both by scoring *the real data itself*. For the failure
studies, a second stage: eight Gaussians on a ring, the standard
mode-collapse laboratory.

---

## A worked example: the game, the freeze, the race

### From noise to spirals

```
DEMO 1 --- The game: from noise to spirals
     step   quality   coverage   detective acc
     1000     95.0%     85.0%        45.6%
     2000     98.7%     81.7%        47.9%
     3000     97.0%     60.0%        48.7%
     4000     99.4%     86.7%        55.5%
     5000     93.7%     70.0%        45.4%
     6000     98.7%     81.7%        50.4%
     real     99.5%     93.3%       (anchor)
```

The game works: after six thousand alternating steps the
forger's samples score within a point of the real data's own
quality anchor, and the detective is reduced to **50.4%** — a
coin flip, which in this game is what victory looks like. When
the fakes match the data, the best possible detective can only
guess.

But read the *path*, because it is the article's first lesson.
Quality hits 99.4% at step 4000, falls to 93.7% at 5000, and
recovers; coverage swings between 60% and 87% and never reaches
the real data's 93.3% — the spiral's faint outer tails keep
being lost and re-found. Nothing is broken. A gradient-descent
loss curve goes *down*; a game **orbits** its equilibrium. This
is the section's first architecture where the training curve
cannot be trusted, and it is why GAN practice leans on external
yardsticks (FID, precision/recall — our quality and coverage
are their toy cousins) rather than on the loss.

### The frozen forger

The section has now met the vanishing gradient three times:
Part 1 lost it to depth, Part 3 to time, Part 6 to
over-smoothing. Fourth appearance, strangest venue: a GAN can
lose its gradient through the **opponent's confidence**.

The forger's original minimax loss is `log(1 - D(G(z)))`, and
its gradient is proportional to `D(G(z))` — the detective's
belief that the fake is real. Early in training the forger's
drafts are garbage, the detective rejects them with near
certainty, that belief is nearly zero — and so, exactly when
the forger has everything to learn, it is handed nothing to
learn from. The script stages the bad start honestly: the
forger's first drafts land far from the data (in high dimension
this happens for free), and the detective studies them for 500
steps before the forger moves. Mean `D(fake)` after the head
start: 6.4e-04. Then the same game, four ways:

```
    optimiser   forger loss        final quality   coverage
    SGD         non-saturating      93.1%         91.7%
    SGD         saturating           0.0%          0.0%
    ADAM        non-saturating     100.0%         83.3%
    ADAM        saturating          99.8%         76.7%
```

The SGD rows are the theorem made visible. The saturating
forger's gradient norm over training:

```
    non-saturating: 10:2e+00  200:9e-02  1000:6e-02  3000:2e-01  6000:2e-01
    saturating    : 10:7e-03  200:3e-90  1000:1e-33  3000:4e-53  6000:3e-38
```

Frozen solid — norms around 1e-38 at the end, quality 0.0%,
*forever*: the detective grows more certain, which shrinks the
gradient, which keeps the forger where it is, which lets the
detective grow more certain. The **non-saturating** fix — train
the forger to *maximise* `log D(G(z))` instead — faces the
identical detective from the identical start and walks home to
93.1%: its gradient approaches −1, not 0, precisely when the
detective is confident. Same optimum, opposite failure mode,
and it shipped in the original GAN paper for exactly this
reason.

The Adam rows are the honest footnote: Adam's update is
scale-normalised, so it largely masks the disease even with the
saturating loss — one reason the folklore "always use the
non-saturating loss" and the folklore "GANs train fine with
Adam" coexist. The disease is real; adaptive optimisers are a
partial bandage; the cure is the loss.

### Mode collapse is a race

The forger's laziest winning move is to find one output the
detective currently trusts and produce *only that*. On the
eight-Gaussian ring this failure has a name — mode collapse —
and the cleanest way to cause it is not to change any loss but
to change a **speed**. Same architecture, same data, three
seeds per row; the only knob is the detective's learning rate:

```
DEMO 3 --- Mode collapse is a race
    detective lr    modes covered      on-mode mass
         1e-03      8/8  8/8  8/8      81%  69%  80%
         3e-04      8/8  6/8  8/8      71%  44%  68%
         1e-04      3/8  3/8  4/8       7%   8%  11%
         3e-05      1/8  0/8  0/8       2%   0%   0%
```

A clean dose-response. At matched speed the ring is covered,
every seed. Slow the detective thirty-fold and the forger stops
learning the distribution entirely — not because the objective
changed, but because the player enforcing it fell behind. And
watching the slow-detective run from inside shows what collapse
actually *is*:

```
     step    modes covered    dominant mode
      500        1/8             6
     1000        2/8             5
     1500        0/8            --
     2000        1/8             7
     2500        1/8             5
     3000        2/8             1
     3500        2/8             0
     4000        3/8             7
     4500        3/8             0
     5000        3/8             3
     5500        2/8             6
     6000        4/8             0
```

The forger camps on one mode; the slow detective eventually
learns to reject it; the forger flees to another — around and
around the ring, never covering it. Mode collapse is not a
wrong loss; it is a **lost race**. Read through that lens,
the practical GAN toolbox stops looking like folklore:
two-timescale learning rates, extra detective steps per forger
step, minibatch statistics handed to the detective — every one
of them is a way of keeping the detective ahead. The
equilibrium is a balance of speeds, not a bottom to descend to.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The economics that kept GANs alive: **sampling is one forward
pass** — `O(L · d²)` and you have your image, against the
hundreds of sequential denoising passes a diffusion model will
charge for the same picture (Part 9's story). Training costs
three network passes per step — detective on real, detective on
fake, forger through detective — and the parameter bill is two
networks, though only the forger ships; the detective is
training scaffolding, discarded at the end. The line with no
happy entry is evaluation: there is no loss you can trust, so
honest measurement means external statistics over *samples* —
at scale, FID's covariance step alone is an `O(d³)` reminder
that with GANs even the scoreboard must be built by hand.

---

## Where the forgeries went

For roughly six years this game *was* generative modelling.
DCGAN (2016) found the convolutional recipe that made it
stable enough to use; StyleGAN's faces became the public image
of "AI-generated"; pix2pix and CycleGAN turned the detective
into a general-purpose loss for image-to-image translation —
day to night, sketch to photo — where nobody could have written
the pixel loss by hand; SRGAN did the same for super-resolution.
The research arc bent toward this article's two failures:
WGAN's critic replaces the detective's probability with a score
whose gradient does not saturate (the frozen forger, addressed
at the loss), and the two-timescale update rule — now standard
— is the race, addressed at the clock. The same forgery power
has an ugly edge — "deepfake" names the case where the fake is
of a person — and that is an argument about deployment, not
architecture. Diffusion models have since taken the
image-generation crown, but the adversarial *idea* outlived the
throne: a learned detective remains the field's best answer
whenever "does this look right?" has no formula — and the
one-pass forger still owns the applications where the answer
must arrive in milliseconds.

---

## What comes next

Part 8, **Variational Autoencoders**, generates from the
opposite temperament: it *does* write down an honest loss — the
ELBO, a compression bargain between reconstruction and a tidy
latent space — and pays for that honesty with blur where the
GAN bought sharpness with instability. The two are the section's
clearest trade: a loss you can trust versus samples you cannot
distinguish.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**gan.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/11-deep-learning-architectures/07-generative-adversarial-networks/gan.py)

Run it with:

```bash
pip install numpy
python gan.py
```

It needs only `numpy` and runs in under a minute. Everything is
from scratch: both players, both hand-derived backward passes
(the forger's gradient threaded through the detective's frozen
weights), the startup audits, the anchored quality and coverage
yardsticks, and every experiment above. The headline insight
worth pinning to the wall: **a GAN never writes down a loss for
its generator — the detective is the loss — and everything
strange about GAN training follows from that one substitution:
victory is a detective at a 50.4% coin flip while quality sits
a point under the data's own 99.5% anchor; the score orbits
the equilibrium instead of descending; the minimax gradient is
proportional to the detective's belief D(G(z)), so a confident
detective froze an SGD forger at 0.0% quality with gradients
near 1e-38 while the non-saturating loss recovered 93.1% from
the identical start; and mode collapse is a lost race, not a
wrong objective — slow the detective thirty-fold and coverage
falls from 8/8 on every seed to 0/8 while the forger flees
around the ring, so every practical GAN trick is one more way
of keeping the detective ahead**.

---

*This is Part 7 of the Deep Learning Architectures track in the Algorithms in Python series. The companion script `gan.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). The two spirals it learns to draw are the stage [Part 1](https://medium.com/p/9c0d36baa5bc) and [Part 2](https://medium.com/p/6f860b54044f) classified, and the vanishing gradient it meets through the detective's confidence is the section's oldest enemy ([Part 3](https://medium.com/p/9a17b15e8339), [Part 6](https://medium.com/p/5fe15dcbafad)). Part 8 will look at Variational Autoencoders.*
