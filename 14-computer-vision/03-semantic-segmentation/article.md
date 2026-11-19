# Semantic Segmentation — The Bill for Throwing Away Position

### A convolution earns translation invariance by discarding where things are. Ask it where things are and the same mechanism sends the invoice.

![[HEADER]]

Part 1 of this section measured a convolution's translation invariance and found it was not an approximation. Re-render the same shape at every legal position and the answer never changes: 1.0000, with zero variance across seeds. That exactness is bought by pooling, which is a deliberate destruction of position — and it was the right trade, because the question was *what is this* and position was the nuisance.

Part 2 then broke the assumption that there is one thing in the picture, and found that the hard part was inventing a training target where none existed.

Here the target comes back for free. Every pixel carries a label; nothing has to be assigned, nothing has to be suppressed. What replaces that difficulty is the one Part 1 created: a segmentation head must say where a boundary falls to within a pixel, using features built by throwing away exactly that. U-Net's contribution is a single idea — hand the decoder the encoder's feature map from *before* the pooling.

So this script holds the architecture, the parameter count, the optimiser and the budget fixed, and changes only whether that skip exists. And because the stage drew every region, it knows each pixel's exact distance to the nearest boundary, which turns *where does the error live* from an impression into a measurement.

Everything below comes from one ~1,000-line NumPy script. No autograd, no frameworks; convolution is `im2col`, the gradients are hand-derived, and it runs in about twenty-eight minutes.

---

## The stage

Six filled shapes, one to three per scene, and a label for every pixel.

![[STAGE]]

Parts 1 and 2 drew one-pixel strokes. That will not do here: every pixel of a stroke touches a boundary, so *where does the error live* has no interior to compare against. These shapes are solid, and 0.27 to 0.54 of each one's pixels sit two or more pixels from any edge.

Two of the six are **pierced** — their solid twin, minus a small central hole.

![[PAIRS]]

They have an identical outer silhouette and differ on five pixels at r=4 and thirteen at r=7, all of them near the middle. So a pixel on the **rim** can only be labelled correctly by a model whose receptive field reaches the hole, several pixels away. That makes receptive field something to measure rather than something to assert, and it matters later.

The labels are not annotations. The script chose each shape's class, centre and radius and *then* drew the pixels, so every boundary is exact and there is no annotator to disagree with.

---

## Why not pixel accuracy

Background is **0.8216** of every pixel. The six real classes split the remainder between them, none of them larger than 0.0436 (pierced square) and none smaller than 0.0200 (triangle) — so background outnumbers the commonest shape by nearly nineteen to one, and the rarest by more than forty.

A model that says "background" and nothing else scores 0.8216 pixel accuracy. The same model scores 0.0000 mean IoU, which is the right answer, so mean IoU is what gets reported below. A self-test asserts both, because a metric that rewards the majority answer would quietly flatter every comparison in the article.

---

## The same comparison, four different conclusions

Here is the result I nearly published, twice.

![[CURVE TABLE]]

At 400 steps the skip connection looks essential: 0.4700 against 0.3265, and on a single seed it reads 0.8398 against 0.0289. I wrote that down. Then I gave the loser more budget and the gap collapsed, so I wrote down something more sophisticated — that skips buy *optimisation speed* rather than final quality.

Three seeds killed that too. At 800 steps the comparison **reverses**: the encoder-decoder is ahead by 0.0956. At 400 steps the U-Net's three seeds are 0.8398, 0.5149 and 0.0554 — a spread of 0.7844, which is wider than the entire effect being measured.

Early training here does not measure the architecture. It measures the seed. The only row worth quoting is the last one, where both have converged and the spreads have collapsed to ±0.0268 and ±0.0103: **the skip is worth +0.0639 mean IoU**.

That is a far smaller claim than the one the first run offered. It is also the only one that survives.

---

## Where the difference actually lives

Now the measurement this stage exists for. Every foreground pixel has an exact distance to the nearest boundary, so the error can be split by it.

![[BOUNDARY TABLE]]

The entire advantage is at distance zero. At the boundary the skip is worth **+0.0816**; the largest difference at any other distance is **0.0036**, twenty-three times smaller, and at three of the five remaining distances the plain encoder-decoder is fractionally *ahead*.

This is exactly what the mechanism predicts, and it is satisfying to see it come out so cleanly. Pooling discards position. Only the pixels adjacent to a boundary need position back — an interior pixel is surrounded by its own kind and was never in any difficulty. So a skip connection does not make the network better at segmentation in general. It repairs one specific thing, in one specific band, and leaves everything else where it found it.

It also closes the thread Part 1 opened. There, pooling bought exact translation invariance and the article called it a good deal. Here the same pooling costs 0.0816 at precisely the pixels where position is the answer, and the skip is what buys it back.

---

## Depth, and what it can see

Depth buys receptive field: 9 pixels at depth 1, 21 at depth 2, 45 at depth 3. At a fixed parameter budget it is paid for in width, so this is a trade.

![[DEPTH TABLE]]

Two things here. First, depth 2 wins — and depth 3 loses at **both** widths that bracket the budget, so the deficit is depth rather than capacity. At 10,962 parameters, more than the winner, it manages 0.4399 with a standard deviation of 0.3422, because one seed collapsed entirely. Squeezing a 32×32 image through a 4×4 bottleneck four channels wide is an optimisation problem as much as a representational one. That echoes Part 1, where a wider-shallower network beat a deeper one at a matched budget; here the optimum is in the middle rather than at the edge.

Second, and more precisely, the **rim** columns. A rim pixel of a pierced shape is labelled correctly only if the model can see the hole, which is up to r pixels away. Depth 1 has a receptive field of 9 pixels — a radius of 4 — and it behaves exactly as that arithmetic predicts: 0.9766 at r=4, falling to **0.6742** at r=7. Depth 2, with a radius of 10, is flat across the whole range (0.9872 to 0.9803).

The failure appears precisely where the shape outgrows what the model can see. And it fails in the predicted direction: at r=7 the shallow model's commonest error on a pierced disc is to call it a **disc** (0.0904 of its rim pixels), and on a pierced square to call it a **square** (0.4054). In both cases the most frequent mistake is the shape's *solid twin*. The model reports the silhouette it can see and misses the evidence that would change the answer, which is what "it cannot see the hole" means in practice. A receptive field is not a diagram in a paper; it is a hard limit on which questions a pixel is allowed to have an opinion about.

---

## Does re-weighting the loss help?

Background outnumbers every real class by more than fifteen to one, so a plain per-pixel loss spends most of its gradient on the easy answer. The standard fix is to weight each class by the inverse of its frequency.

![[WEIGHTING TABLE]]

Inverse-frequency weighting is nominally best, by +0.0156 — against plain's own seed spread of 0.0268. That is inside the noise, and so is the foreground-accuracy column. On this stage the schemes cannot be told apart, and naming a winner would be reporting a decimal point.

The honest reading is that the trick is aimed at a problem of a different size. Fifteen to one is a mild imbalance; the medical segmentation tasks the technique was built for routinely run at hundreds or thousands to one. Part 2 reached the same verdict about down-weighting the no-object term for the same reason, which is worth noticing: both are standard fixes, and this family of stages is too forgiving to show either one working.

---

## What it costs

![[BIG-O TABLE]]

---

## Verification

Four self-tests, each guarding a claim the article makes:

- **The stage knows what it drew.** Solid shapes carry real interiors; each pierced shape is a strict subset of its twin, differing only within half a radius of the centre.
- **The metric cannot be fooled.** Predicting background everywhere gives 0.8216 pixel accuracy and **0.0000** mean IoU. The ground truth against itself gives 1.0000; shifted one pixel sideways, 0.7477.
- **The decoder's two new operations are exact.** The nearest-neighbour upsample and its adjoint agree to **0.00e+00**, and max-unpool routes every gradient to exactly one input with the sum preserved.
- **The gradients are right**: worst relative error **1.8e-07** over the ReLU network, **2.2e-08** with linear activations.

That last line took four attempts, and each failure was worth more than the pass.

The first audit reported 2.6e-03 and looked like a bug in the skip connections. It was not. Isolating the parts showed the convolution, the upsample adjoint and the pooling adjoint were all exact, and a *linear* version of the whole network — skips included — checked out at 1e-09. The problem was that a ReLU has no derivative at zero, and a central difference that steps across the kink reports a disagreement the backward pass is not responsible for.

Two further corrections came out of chasing it. "Linear activations make the loss smooth" is false — max-pool still has a corner wherever two inputs tie for the maximum, which is why the linear error stopped falling as the step size shrank instead of vanishing with it. And on a full 32×32 batch there are roughly 25,000 pre-activations per layer, so the smallest one is near zero by sheer density and *no* step size avoids both the kink and floating-point roundoff. The audit therefore runs on a small instance, where the margin is 1.8e-04 — nineteen times the step size — and asserts that margin rather than hoping for it.

There was a fourth, quieter one. With linear activations and unit-scale inputs the logits saturate the softmax, both gradients collapse to roughly zero, and their *relative* difference reads 0.99 — meaningless rather than wrong. ReLU had been hiding that by clipping.

---

## What this does and does not show

The stage was built to make one thing visible and deletes a great deal to do it. Regions never overlap, nothing is occluded, there is a single scale, the shapes are convex or nearly so, and the whole world is 32×32 with clean boundaries. Real segmentation spends most of its engineering on exactly what was removed: ambiguous boundaries, thin structures, scale pyramids, and annotations that disagree with each other.

The measurement that drives this article — error by exact distance to a boundary — is also only available because the script drew the regions. On annotated data that distance inherits every wobble in the annotation, which is precisely the quantity being measured.

What survives is the shape of the argument. Pooling trades position for invariance; segmentation needs the position back; a skip connection buys it back, in one band, for a measurable amount. And a receptive field is a hard limit on what a pixel can know, visible the moment you build a stage where the answer depends on something far away.

---

## What comes next

Part 4, **Generative Vision**, turns the question inside out. Every model in this section so far has been handed an image and asked to describe it — a label, a box, a per-pixel map. Next the image is the output, and the loss stops being a comparison against a known answer, because there isn't one: a generated picture is not supposed to match a target, only to belong to a distribution.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**segment.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/14-computer-vision/03-semantic-segmentation/segment.py)

Run it with:

```bash
pip install numpy
python segment.py
```

It needs only `numpy` and runs in about twenty-eight minutes.
Everything is from scratch: im2col convolution, max-pooling and
its adjoint, parameter-free nearest-neighbour upsampling, skip
concatenation, a per-pixel softmax cross-entropy and an exact
4-connected boundary-distance transform — all hand-differentiated
and audited at startup against central differences on a small
instance whose ReLU margin is asserted, not assumed. The headline
insight worth pinning to the wall: **a skip connection is worth
+0.0639 mean IoU at convergence and +0.0816 at the boundary, while
the largest difference it makes at any other distance is 0.0036 —
so it does not make a network better at segmentation, it repairs
one band of pixels whose position pooling threw away; and the same
comparison read at 400 steps says skips are essential, at 800 steps
says they are harmful, and only at convergence says anything
true at all**.

---

*This is Part 3 of the Computer Vision track in the Algorithms in Python series. The companion script `segment.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It extends the stage from [image classification](https://medium.com/@grahamjroy/image-classification-what-a-prior-is-worth-a00de66cfa5e), whose exact translation-invariance measurement this article is the other half of, and follows [object detection](https://medium.com/@grahamjroy/object-detection-where-the-targets-come-from-78b28605ffcb), where the difficulty was inventing a training target rather than locating one. Part 4 will look at Generative Vision.*
