# Object Detection — Where the Targets Come From

### A detector has 64 places to speak from. A scene has 2.46 objects. Almost everything difficult about detection follows from that mismatch.

![[HEADER]]

[Part 1 of this section](https://medium.com/@grahamjroy/image-classification-what-a-prior-is-worth-a00de66cfa5e) put four architectures — a perceptron, a convolution, a residual network and a vision transformer — on the same question and measured what each one's prior was worth. Every one of them ended the same way: a softmax over six classes, trained against the label the stage had drawn from. The target was never in doubt. It was simply *there*, one per image, waiting to be compared against.

Now put two shapes on the canvas.

The output is no longer a label but a list, and the list has no fixed length. That much is obvious. What is less obvious, and what this article is actually about, is that **the training target disappears with it**. A detector emits a fixed number of predictions — sixty-four, here. A scene contains some other number of objects. Cross-entropy needs to know which prediction is answerable for which object before it can compute anything at all, and nothing in the data says.

That decision has a name — *assignment* — and it is not a detail of the optimiser. It is the design decision that separates YOLO from Faster R-CNN from DETR. So this script holds the architecture, the loss terms, the parameter count and the step budget fixed, and changes only the rule that assigns objects to predictors.

Everything below is measured by a single ~1,100-line NumPy script with no autograd and no frameworks. Convolution is `im2col`, the gradients are hand-derived and audited against central differences, and the bipartite matcher is checked against brute force. It runs in about nineteen minutes.

---

## The stage

Part 1's six shapes, on a canvas twice as wide, one to four per scene.

![[STAGE]]

The six classes are deliberately confusable — a cross and a vertical bar share strokes, a hollow square and a corner L share two sides — so a detector cannot settle the class on appearance alone. And the boxes are not annotations. The script *chose* the class, the centre and the size and then drew the pixels, which means the ground truth is exact rather than agreed.

One choice needs defending. Objects are kept from touching:

```python
if any(abs(cy - o[1]) <= s + o[3] + min_gap and
       abs(cx - o[2]) <= s + o[3] + min_gap
       for o in objs):
    continue
```

That costs realism, and it buys something I was not willing to give up. When boxes may overlap, "how many objects are there" stops having a single defensible answer, and every number downstream inherits the ambiguity. A self-test confirms the cost was actually paid: across 2,000 scenes, the worst IoU between two real objects is **0.0000**.

The price is that this stage is uncrowded, and I measure that rather than wave it away.

---

## What a fixed-size output can even represent

Before asking whether the Part 1 architecture would be *accurate* at this, ask whether it could be *right*.

A classifier emits one answer. If a scene holds three objects, two of them have nowhere to go — not "are predicted badly", but cannot be expressed. Generalise that to a grid of predictors and the question becomes: how often do two objects' centres land in the same cell?

![[REPRESENTATION TABLE]]

A single predictor — everything Part 1 built — cannot hold **0.6016** of the objects in this world. That is the number before a single gradient step, and no amount of training touches it.

The honest part of that table is the bottom. At 8×8 the figure is **0.0000**, so dropped objects are *not* what separates the rules compared later. The grid resolution is a bet on how crowded the world is, and it costs little to show where the bet would be lost:

![[CROWDING TABLE]]

Pack twelve overlapping objects into the same canvas and a 4×4 grid loses **0.1502** of them. This stage simply does not reach that regime, which is a limitation to state plainly rather than a result to claim.

---

## Most predictors must learn to say nothing

Sixty-four predictors, 2.4638 objects per image: **96.15%** of predictors should stay quiet, a ratio of 25 to 1. YOLO's answer was to down-weight the no-object term so the silent majority cannot drown out the few positives.

```python
w = np.where(obj_t > 0, L_OBJ, L_OBJ * L_NOOBJ)
```

Does it help here? Three weights, two seeds, each run scored part-way and at the end — so a weight that merely learns *faster* cannot pass for one that ends up *better*:

```
              600 steps        1000 steps
weight 1.0    0.7380 ±0.0295   0.8328 ±0.0007
weight 0.5    0.7872 ±0.0363   0.8519 ±0.0299
weight 0.2    0.7528 ±0.0514   0.8442 ±0.0175
```

Weight 0.5 is nominally best at both budgets, and I am not going to report that as a finding. It beats no weighting by 0.0191, and its own two seeds differ by 0.0299. The gap is inside the noise, and calling it a win would be reporting a decimal point.

What can be said is that the trick is aimed at a problem of a different size. This stage has 25 negatives per positive; an anchor-based detector carries on the order of 100,000 anchors for a handful of objects, nearer 10,000 to 1. The cure is calibrated to a disease this stage does not have.

---

## A detour that turned out to matter

YOLOv1 ran the centre offset through a sigmoid, which confines a predictor's box to its own 4px cell. For the centre rule that is free — its target is inside the cell by definition. For a rule that lets a *distant* predictor own an object, it is not:

```
centre      3747 positives,  0.0000 outside their own cell
many       22817 positives,  0.7443 outside their own cell
hungarian   3747 positives,  0.4665 outside their own cell
```

I expected this to be a confound. Three quarters of the one-to-many rule's positives are trained toward a box they cannot emit and then scored on the box they do emit; that looks like grounds for throwing the comparison out. So I trained every rule both ways.

```
                    with the bound    without
centre                  0.7770         0.7380
many                    0.4970         0.3766
hungarian               0.6893         0.6575
```

**The prediction was wrong.** Confining the box to its own cell helps *every* rule, and helps the one-to-many rule most (+0.1204) — precisely the rule whose targets it mangles hardest. Reaching a clipped target a predictor can actually emit beats reaching a correct one it has to find by regressing a long way. The bound is a prior, and it buys more than target fidelity costs.

What survives is the check that the comparison actually needs. The ranking is `centre > hungarian > many` with the bound and `centre > hungarian > many` without it. The conclusion does not rest on the decoder, so the comparison below uses the unbounded offset for every rule — the only setting in which all three can express their own targets — and pays the centre rule about 0.04 for the privilege.

---

## The three rules

Each takes the scene's objects and returns a list of `(predictor, class, dy, dx, size)`. That list is the target. Nothing else changes.

**Centre — YOLOv1.** The cell holding the centre owns the object. Static, so it can be computed once. A clash drops the object.

```python
def assign_centre(objs, out_n=None):
    rows, taken = [], set()
    for t in targets_for(objs):
        if t[0] in taken:
            continue          # no room: object dropped
        taken.add(t[0])
        rows.append(t)
    return rows
```

**Many — the anchor-based family.** Every cell whose centre falls inside the object is a positive, and all of them regress to the same box. Also static, and it manufactures duplicates on purpose.

```python
for p in range(N_PRED):
    py, px = cell_centre(p)
    if (abs(py - cy) <= s + radius and
            abs(px - cx) <= s + radius):
        d = (py - cy) ** 2 + (px - cx) ** 2
        if p not in claim or d < claim[p][0]:
            claim[p] = (d, box_row(p, cls, cy, cx, s))
```

**Hungarian — DETR.** Solve an optimal one-to-one match between objects and predictors, scoring each pairing by how well that predictor is *already* doing on that object.

```python
cost[oi] = (-np.log(pr[:, cls] + 1e-9)
            - np.log(sc + 1e-9)
            + 0.2 * (np.abs(py - cy)
                     + np.abs(px - cx)
                     + np.abs(ps - s)))
```

This one is dynamic. The cost matrix is built from the *current* predictions, so the assignment is re-solved every step and moves as the model learns. Nothing tells predictor 19 to specialise in the top-left; it gets there by being repeatedly cheapest for top-left objects.

An approximate matcher would quietly undercut the whole comparison, so the solver is checked against brute force on 400 random rectangular cost matrices. Worst gap: **0.000e+00**.

---

## The comparison

Same backbone, same head, same loss terms, same 1,000 steps, same 5,130 parameters. Three seeds.

![[RULES TABLE]]

The centre rule wins at **0.8458**, the optimal match reaches **0.7438**, and the one-to-many rule manages **0.4422**. But the column that matters most is the last pair.

**Suppression is not a universal good.** It repays the one-to-many rule by **+0.0874** and *costs* both one-to-one rules — −0.0196 and −0.0212 — consistently across every seed. Non-maximum suppression is not post-processing that happens to help. It is the repair bill for an assignment rule that trained several predictors to report the same object, and if you did not run up that bill, paying it makes things worse.

The threshold sweep says the same thing more sharply:

![[NMS TABLE]]

The one-to-many rule wants the most aggressive suppression on offer. The two one-to-one rules peak with it switched off entirely.

One number is worth pausing on. The Hungarian assignment changed its mind about who owns which object **3,903** times during training, while the two static rules changed zero times. That is not instability to be engineered away; it is the mechanism. The matching is a function of the model, and the model moves.

---

## Attributing every miss

Here is what the stage buys that a benchmark cannot. Because the script drew the objects, the predictor that was *supposed* to own each one is known. So "the detector missed it" stops being a single bucket.

![[ERRORS TABLE]]

Two things in that table surprised me.

The one-to-many rule **never goes silent** — 0.0000, against 0.1716 and 0.1544 for the one-to-one rules. Train five predictors per object and something always fires. It also *finds* almost as many objects as the winner (0.7358 against 0.7566). Its AP is half as good anyway, which means what it loses is precision, not recall: it pays for its coverage with 0.1752 bad boxes and 0.0889 wrong labels.

And the optimal match does not reinvent the centre rule. Only **0.2681** of its hits come from the cell containing the object's centre, against **0.9542** for the rule defined that way. Given free choice of which predictor answers for which object, it systematically picks someone else.

---

## Confidence, or geometry?

If the one-to-many rule finds the objects but loses precision, there are two candidate culprits. Its confidence might fail to separate predictors that overlap an object from predictors that overlap nothing — a ranking failure. Or its confident predictors might simply hold worse boxes — a geometry failure.

```
             ranking AUC    IoU of the most confident
centre          0.9930              0.7020
many            0.9976              0.5954
hungarian       0.9507              0.6973
```

The ranking AUC is the chance that a predictor overlapping a real object outscores one overlapping nothing. Every rule is near-perfect at it, and the one-to-many rule is the **best** of the three. The obvious guess — that its duplicates are confidently wrong — is simply false.

The geometry column separates them. Its most confident predictor averages **0.5954** overlap against the centre rule's **0.7020**, and an AP hit needs 0.5000. It is sitting just above the line: its confident boxes are good enough to look plausible and not good enough to count, which is exactly what turns a duplicate into a false positive.

Train several predictors on one object and none of them specialises in it. That is the price one-to-many assignment pays for never going silent, and it is what *centerness* and IoU-aware scoring were later invented to win back.

---

## What it costs

![[BIG-O TABLE]]

---

## Verification

Four self-tests guard the claims, and each one is a claim the article makes:

- **The ground truth is unambiguous.** Worst IoU between two real objects across 2,000 scenes: **0.0000**. Every box inside the canvas.
- **The metrics are calibrated.** AP of the ground truth against itself **1.0000**; random boxes **0.0005**; correct labels on boxes shifted 3px **0.0000**; exact boxes with every label rotated **0.0000**. The ranking AUC returns 1.0000, 0.0000 and 0.5000 on rankings whose answer is known by construction.
- **The matcher is optimal**, not greedy: worst gap against brute force **0.000e+00** over 400 trials.
- **The gradients are right**: worst relative error **5.198e-09** over 98 sampled gradients.

That last one took a detour worth repeating. The first audit reported a relative error of **1.0** on the convolution biases while every weight checked out at 1e-8 — the signature of a real bug. It was not one. The canvas is 97% empty, so with zero biases **90%** of the first layer's pre-activations are *exactly* zero: thousands of units sitting precisely on the ReLU kink at once. A bias shifts every pre-activation in its channel together, so a central difference straddles the kink and reports a disagreement the backward pass is not responsible for. Audited at a generic point, with the biases clear of zero, the same code returns 5.198e-09.

The fix carried a second benefit. Biases now initialise to a small positive value, because on a 97%-empty canvas a zero bias means the first layer only ever sees gradient from the few windows containing ink.

---

## What this does and does not show

The stage was built to make assignment visible, and it does that by removing almost everything else. Objects never overlap, they occupy a single scale, nothing is occluded, and the whole world is 32×32. Crowding, scale pyramids and occlusion are not incidental omissions — they are the parts of real detection that the simplification deliberately deletes, and they are where most of the engineering in a modern detector actually goes.

The `many` rule here is also the one-to-many *principle*, not a faithful RetinaNet: no anchor priors of differing scale and aspect, no IoU-threshold matching against them. A real anchor detector has machinery, precisely because the naive version behaves the way the table shows.

What does survive the simplification is the shape of the problem. There is no target until a rule invents one; that rule decides whether you get silence or duplicates; and the post-processing everyone treats as a fixed part of the pipeline turns out to be a bill that only one family of rules runs up.

---

## What comes next

Part 3, **Semantic Segmentation**, dissolves the problem this
article is about. When every pixel carries its own label the
target comes back, handed to you for free — no rule has to
invent it, and nothing has to be suppressed afterwards. What
replaces it is a different difficulty entirely, because the
thing a convolution throws away in order to recognise an
object anywhere is exactly the thing you now have to give
back: where, precisely, it was.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**objdet.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/14-computer-vision/02-object-detection/objdet.py)

Run it with:

```bash
pip install numpy
python objdet.py
```

It needs only `numpy` and runs in about nineteen minutes.
Everything is from scratch: im2col convolution with channels,
a shared 1x1 detection head, three assignment rules, greedy
non-maximum suppression, all-point-interpolated average
precision and an exact bipartite matcher by shortest
augmenting paths — all hand-differentiated and audited
against central differences at startup (worst error 5.198e-09
over 98 sampled gradients, taken at a generic point because
on a 97%-empty canvas a zero bias parks 90% of the
pre-activations exactly on the ReLU kink). Self-tests pin the
stage's ground truth, both metrics and the matcher before any
measurement runs. The headline insight worth pinning to the
wall: **the same network, the same loss and the same budget
score 0.8458, 0.7438 and 0.4422 depending only on which
predictor is told to answer for which object — and
non-maximum suppression, which everyone treats as a fixed
part of the pipeline, repays the one rule that manufactures
duplicates by +0.0874 while costing the two that do not
0.0196 and 0.0212 on every seed, because it was never
post-processing at all but the repair bill for an assignment
decision made long before**.

---

*This is Part 2 of the Computer Vision track in the Algorithms in Python series. The companion script `objdet.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It extends the "shape world" stage from [image classification](https://medium.com/@grahamjroy/image-classification-what-a-prior-is-worth-a00de66cfa5e), which measured what a convolution's prior is worth at sixty labels. Earlier in the series: [Fine-Tuning and LoRA — The Update Is Not Low-Rank](https://medium.com/@grahamjroy/fine-tuning-and-lora-the-update-is-not-low-rank-048d50f039aa) and [Retrieval-Augmented Generation — Reading or Reciting?](https://medium.com/@grahamjroy/retrieval-augmented-generation-reading-or-reciting-0686445e3d97). Part 3 will look at Semantic Segmentation.*
