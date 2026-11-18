# Image Classification — What a Prior Is Worth

### *Algorithms in Python --- Computer Vision, Part 1*

---

The Natural Language Processing track spent eleven parts on a
micro-language whose 31,740 sentences could be listed, which
meant every claim had an oracle to check it against. Images
need their own version of that, and this section's is simpler
than it sounds: the pictures here are **drawn** from factors
the script chooses.

```
class, centre (cy, cx), size, brightness
```

So the thing a vision model is supposed to ignore is not
something to be inferred afterwards. It is an input. That
turns the central question of this article from an argument
into a measurement.

The question is what an architecture's **prior** is worth. A
multilayer perceptron sees 256 unrelated numbers. A
convolution is told, before it sees any data, that pixels
near each other belong together and that a pattern means the
same thing wherever it appears. A residual network is told it
may skip a layer. A vision transformer is told almost
nothing.

---

## A worked example: four priors, one budget

### The stage

Six shapes on a 16×16 canvas, chosen so their pixel
statistics overlap — a cross and a vertical bar share
strokes, a hollow square and a corner L share two sides — so
that only **geometry** settles the class. Guessing scores
0.1667.

![[STAGE IMAGE]]

Four models, matched to within **4.4%** on parameter count,
each given the same 400 gradient steps at every training-set
size. What differs is only what each was told in advance:

```
MLP      nothing: 256 numbers in a row
CNN      pixels nearby belong together; patterns repeat
ResNet   the same, plus: you may skip a layer
ViT      patches are a sequence; work it out
```

Matching the budgets matters more than it might appear. My
first run of this comparison had the residual network beating
the convolution handily — on five times the parameters. Equal
capacity reverses the result, and the reversal is the
interesting part.

### The price of a prior, measured in labels

![[ACCURACY TABLE IMAGE]]

From **60 labels** the convolution is at **0.9271**. The
perceptron, holding the same number of parameters, is at
**0.3064** — and after sixteen times the data it has reached
only **0.6276**, still well below what the convolution
managed from sixty examples.

That gap is the prior, and it was not bought with capacity.
Both models have about 5,500 parameters. One of them was
simply told something true about images before training
began.

The transformer sits between the two: 0.4500 at 60 labels,
0.8658 at 960. It is not broken. It is **learning from data
what the convolution was told for free**, and paying for it
in examples — which is exactly the behaviour that makes
vision transformers excellent at scale and poor below it.

Then the result I did not expect. **The residual network
loses to the plain convolution at every small size** —
0.8896 against 0.9271 at 60 labels, a gap of about fifteen
standard errors. By 960 labels it has drawn level
(+0.0002). Depth costs 0.0376 early here and repays almost
none of it.

### Does it change its mind when the object moves?

This is the measurement the stage was built for. Because the
script chose the centre, the same shape can be re-rendered at
**every legal position** and each model asked whether its
answer survives the move. A score of 1.0000 means position is
ignored completely.

![[INVARIANCE TABLE IMAGE]]

Read the convolution's row. Not "approximately". Not "within
noise". **1.0000, with a standard deviation across seeds of
0.0000.** Global max-pooling takes the strongest filter
response wherever it occurs, so moving the object cannot
change the answer. The property is not learned, and no amount
of training would improve it, because it is arithmetic.

Now the transformer: **0.9445** at margin 0 and **0.9505**
with the border cleared. Clearing the border does not rescue
it, because the border was never its problem. It has an
approximation fitted from examples — roughly right
everywhere, exactly right nowhere.

That is the whole difference between a prior and a learned
regularity, and this stage can show it only because the
position was an input rather than a guess.

The residual network is worth a glance too: 0.9991 at margin
0, 1.0000 with two pixels of clearance. Its padded
convolutions see a little of the canvas edge, and that is the
one place its construction stops applying.

### Where should the parameters go?

![[SCALING TABLE IMAGE]]

The same residual network reshaped three ways at one budget.
This is EfficientNet's question minus one axis — resolution
is fixed at 16×16 here, so the study covers depth against
width only, and says so.

Wider and shallower wins, and wins hardest where the data is
thinnest: **0.9716 against 0.8751** at 120 labels, closing to
0.9984 against 0.9949 once there is plenty. Look at the
spread as well — **0.0096 for the shallow model against
0.0498 for the deep one** at 120 labels. Depth does not only
cost accuracy here, it costs reliability.

The honest reading is not that depth is bad. Depth is a
**bet** — on the data having structure at several scales, so
that later blocks have something built by earlier ones to
work on. Six shapes on a 16×16 grid have one scale, so the
bet does not pay. On photographs it does, which is precisely
why real residual networks are deep and this one should not
be.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The line worth noticing is the first pair. A dense layer on
an `S×S` image costs `S²·H` parameters; a convolution costs
`k²·F` regardless of how big the image is. At 16×16 that is
the difference between 5,529 parameters buying one hidden
layer of 21 units and 5,446 buying 170 independent 5×5
feature detectors.

Weight sharing is not a memory optimisation that happens to
help accuracy. It is the *statement* that a pattern means the
same thing everywhere, and the parameter saving is what that
statement costs to make.

---

## Where this sits

LeCun's convolutional networks date from 1989, and the
argument for them was always the one measured above: build
the invariance in rather than learn it. ResNet arrived in
2015 to solve a problem this article's stage does not have —
that beyond a couple of dozen layers, plain deep networks got
*worse*, and the skip connection fixed it by letting a block
start as the identity. EfficientNet asked in 2019 how to
spend a fixed budget across depth, width and resolution
together rather than one at a time.

Vision transformers came in 2020 with the opposite bet:
discard the locality prior, treat patches as a sequence, and
let attention learn the structure. The finding that followed
them is the one reproduced here — they need far more data
than convolutions to reach the same place, because they are
reconstructing from examples what a convolution is given for
free.

The thread back through this series is Part 2 of Section 11,
which measured the same trade on two spirals and found that
priors beat flexibility at low data. This article prices that
in two currencies at once — labels and parameters — and adds
the thing that stage could not provide: an exact measurement
of what the prior actually *is*.

---

## What comes next

Part 2, **Object Detection**, breaks the assumption every
model here relied on: that there is exactly one thing in the
picture and the only question is what it is. Once an image
may contain several objects, or none, the output stops being
a label and becomes a variable-length list of boxes — and the
loss has to be rebuilt before anything can be trained at all.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**imgcls.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/14-computer-vision/01-image-classification/imgcls.py)

Run it with:

```bash
pip install numpy
python imgcls.py
```

It needs only `numpy` and runs in about ten minutes.
Everything is from scratch: im2col convolution, same-padded
residual blocks with max-pooling, patch embedding with
multi-head self-attention, and a shared Adam so the
comparison is of architectures rather than optimisers — all
hand-differentiated and audited against central differences
at startup (worst error 6.46e-10 over 128 sampled gradients),
with a self-test that proves the stage conserves ink under
translation, which is the premise the invariance measurement
rests on. The headline insight worth pinning to the wall: **a
convolution and a perceptron with the same 5,500 parameters
and the same training budget score 0.9271 and 0.3064 on sixty
labels, and the perceptron never catches up — because the
convolution was told, before it saw anything, that a pattern
means the same thing wherever it appears; and that telling is
exact rather than approximate, which this stage can prove by
re-rendering one shape at every legal position and finding
the convolution's answer unchanged 1.0000 of the time with
zero variance across seeds, while a vision transformer given
the identical images manages 0.9505 and never improves,
because it is approximating from examples what the other was
handed for free**.

---

*This is Part 1 of the Computer Vision track in the Algorithms in Python series. The companion script `imgcls.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It extends the "shape world" stage introduced in [contrastive learning](https://medium.com/p/37f7feeb811e), and picks up the prior-versus-flexibility thread from [convolutional neural networks](https://medium.com/p/6f860b54044f). Part 2 will look at Object Detection.*
