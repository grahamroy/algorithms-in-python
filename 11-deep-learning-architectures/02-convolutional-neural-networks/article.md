# Convolutional Neural Networks — Learned Here, Known Everywhere

### *Algorithms in Python --- Deep Learning Architectures, Part 2*

---

Part 1 closed on a strange promise: shuffle the pixels of every
image *identically* and a multi-layer perceptron learns exactly
as well. To the MLP, an image was never a picture — just 196
unrelated columns of numbers that happen to arrive together. It
cannot know that pixel 37 sits beside pixel 38, and it spends
its capacity rediscovering, position by position, facts a human
would call obvious.

Images have two kinds of structure the MLP is blind to.
**Nearby pixels are related** — edges, strokes, textures are
local events. And **the same pattern means the same thing
wherever it appears** — a vertical stroke at the top-left is
the same stroke at the bottom-right. The **convolutional
layer** (LeCun, 1989) writes both facts directly into the
architecture:

- **Locality**: each output neuron looks at a small window —
  3×3 here — instead of the whole image.
- **Weight sharing**: *one* kernel slides across every
  position. Nine weights, reused at all 144 locations. What is
  learned here is known everywhere.

Add ReLU, then 2×2 **max-pooling** — a layer that answers "did
the feature occur anywhere in this neighbourhood?" and throws
away exactly where — and you have the stack that read your
cheques in the 1990s and won ImageNet in 2012.

The companion script builds it all by hand — the im2col trick
that turns convolution into one matrix multiply, the backward
pass with argmax routing through the pool, audited against
central differences at **2.59e-10** — and then runs the
experiment Part 1 promised.

---

## The stage: geometry or nothing

Every image is 14×14, containing a single 7-pixel stroke —
**horizontal, vertical, or diagonal** — at a random position,
under noise. Three classes, all lighting exactly 7 pixels, so
total brightness carries no signal at all. The *only* learnable
thing is geometry: which way does the stroke run, wherever it
happens to be?

---

## A worked example: the shuffle, the budget, the kernels

### The shuffle test

One fixed random permutation of the 196 pixel positions,
applied to every image — train and test alike. Neighbourhoods
are destroyed; information is perfectly intact.

```
DEMO 1 --- The shuffle test: does geometry matter to you?
                     MLP [64]    CNN
    original:         92.2%     99.8%
    shuffled:         91.0%     62.0%
```

The MLP does not notice — 92.2% to 91.0%, the promised
non-event, because to it pixels were always unrelated columns
and the shuffle just renames them. The CNN's advantage
**evaporates**: 99.8% to 62.0%. Its architecture *assumes*
nearby-means-related, and the shuffle makes the assumption
false. This is the honest statement of what an architectural
prior is: a bet about the data, paid out only when true. (The
62% remainder is the CNN scraping the few pixel pairs the
permutation happened to leave adjacent.)

### Sample efficiency: the prior is prepaid data

```
DEMO 2 --- Sample efficiency: the prior is prepaid data
    train imgs    MLP [64]    CNN
        15        36.0%     48.2%
        45        50.8%     65.8%
       150        59.7%     97.2%
       600        95.8%     99.8%
```

At 150 training images the CNN leads by ~38 points, and the
mechanism is visible in the failure: an MLP must *see* a stroke
at a position to know it there — each location is a separate
fact to learn — while the CNN's shared kernel learns the stroke
once, for everywhere. By 600 images the MLP has bought with
data what the CNN was given by design. An architectural prior
is data you don't have to collect.

### Locality vs sharing — and the kernels nobody asked for

The conv layer makes two separable promises, so the script
separates them: a **locally-connected** network keeps the same
3×3 wiring but gives every location a *private* kernel —
locality without sharing.

```
DEMO 3 --- Locality vs sharing
    architecture                 params    test acc
    dense MLP [64]              12,803     92.2%
    local 3x3, unshared         11,243     98.8%
    conv 3x3, shared               947     99.8%
```

Locality alone is worth +6.6 points — small windows see strokes
cleanly. Sharing then does more with **12× fewer parameters**,
because every window's experience trains the same nine weights:
the kernel effectively sees the whole dataset at every
position. 947 parameters beat 12,803.

And what did those nine weights become?

```
    filter 0  (norm 2.82):        filter 6  (norm 2.58):
      +1.12  -0.76  -0.52           +1.27  -1.00  -0.83
      +1.39  -0.67  -0.69           -1.06  +0.95  +0.09
      +1.60  -0.58  -0.29           -0.85  +0.21  +0.73
```

Filter 0 is a **vertical-edge detector** — a strongly positive
left column against negative neighbours. Filter 6 lights up
along the diagonal. Nobody asked for edge detectors; they are
what gradient descent discovers when nine weights must serve
144 positions at once. Full-scale networks rediscover
Gabor-like filters the same way — and biological V1 arrived at
the same answer first, which remains one of deep learning's
most pleasing coincidences.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

A conv layer's compute is `O(N · P · k² · F)` — positions ×
window × filters — but its *parameters* are just `O(k² · F)`,
independent of image size. That decoupling is the whole
economics of the architecture: cost scales with the image,
knowledge doesn't. The im2col trick trades memory (each pixel
copied k² times) to turn the slide into one matrix multiply —
the same trade every deep learning framework makes under the
name of speed. Pooling is free of parameters entirely; its
backward pass just routes gradient to whichever input won the
max.

---

## From cheques to ImageNet — and what dethroned it

The lineage is compact: LeNet-5 reading American cheques
(1998), a long winter, then AlexNet (2012) — the same
architecture, three orders of magnitude more compute — cutting
ImageNet error nearly in half and starting the modern era.
ResNet (2015) made CNNs effectively arbitrarily deep by letting
layers learn *corrections* rather than transformations. Today's
vision transformers relax the convolutional prior — with enough
data, ViTs *learn* locality rather than assume it, and at
internet scale that flexibility wins. But DEMO 2 is the
counterweight: below some data threshold the prepaid prior is
unbeatable, which is why CNNs still rule small-data vision,
edge devices, and medical imaging. The lesson generalises: the
less data you have, the more your architecture should already
know.

---

## What comes next

Part 3, **Recurrent Neural Networks**, takes weight sharing to
its other natural habitat: *time*. A sequence has the same two
structures an image has — local order matters, and the same
pattern means the same thing whenever it occurs. The RNN slides
one network along a sequence the way a kernel slides across an
image — and meets a harder version of Part 1's vanishing
gradient the moment the sequence gets long.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**cnn.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/11-deep-learning-architectures/02-convolutional-neural-networks/cnn.py)

Run it with:

```bash
pip install numpy
python cnn.py
```

It needs only `numpy` and runs in about a minute. Everything is
from scratch: the stroke-image generator, im2col convolution
with its full backward pass, argmax-routed max-pooling, the
locally-connected control, and the startup gradient audit.
The headline insight worth pinning to the wall: **a
convolutional layer is a bet written into the wiring — nearby
pixels are related (3×3 windows) and patterns mean the same
thing everywhere (nine shared weights serving 144 positions) —
and the bet pays exactly when true: 947 parameters beat 12,803
(99.8% vs 92.2%), a ~38-point lead at 150 training images, and
learned kernels that became edge detectors nobody requested —
while shuffling the pixels, which falsifies the bet, costs the
CNN its entire advantage (99.8% → 62.0%) and costs the MLP
nothing at all**.

---

*This is Part 2 of the Deep Learning Architectures track in the Algorithms in Python series. The companion script `cnn.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It measures the pixel-shuffle claim that closed [Part 1](https://medium.com/p/9c0d36baa5bc), whose MLP returns here as the baseline. Part 3 will look at Recurrent Neural Networks.*
