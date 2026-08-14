# Masked Autoencoding — Learning by Filling In the Blanks

### *Algorithms in Python --- Self-Supervised Learning, Part 2*

---

Part 1 manufactured supervision by **ignoring**: declare which
differences do not matter — the augmentation menu — and demand
agreement across them. This article builds it the opposite way:
by **reconstructing**. Hide most of the input; demand the model
predict what is missing. For text, that recipe is BERT (Devlin
et al., 2019) — mask words, predict words — and it trained the
encoders that started the modern NLP era. For images it is the
masked autoencoder, MAE (He et al., 2022): blank three quarters
of the patches and regress the missing pixels.

```
    input:  image with 12 of 16 patches blanked (75%)
    task :  predict the hidden pixels
    loss :  MSE, scored ONLY on what was hidden
```

No labels, no augmentation menu, no negatives, no crowd. The
supervision is the data's own redundancy: to fill in a hidden
patch of a cross you must know it *is* a cross, where it sits,
how bright it is. An encoder squeezes the visible quarter into
a 32-number code; a decoder rebuilds the hidden three quarters
from that code; whatever survives the squeeze is the
representation.

The stage is Part 1's shape world, on purpose, and the
companion script retrains Part 1's contrastive encoder live —
so the section's two philosophies, *learn by ignoring* and
*learn by reconstructing*, meet on one measured table. The
masked-reconstruction backward pass is hand-derived and audited
at startup (2.11e-10 against central differences).

---

## A worked example: the probes, the flip, the keeping

### Hide three quarters, and what the probes say

```
    masked-pixel reconstruction mse: 0.0467  (pixel noise^2 = 0.0100)

                       probe, 15 labels   probe, all 4,000
    MAE                  49.2%             67.6%
    contrastive          95.1%             99.7%
```

The reconstruction works — the model fills in hidden strokes it
has never seen. And the linear probes call the representation a
failure anyway: 49.2% against 95.1% on fifteen labels, thirty
points behind even with every label in the pile. Reconstruction
features are not linearly organised by class. If this article
stopped here, the verdict would be: use Part 1.

### The fine-tune flip

It doesn't stop here. Unfreeze everything and fine-tune both
encoders on the *same* fifteen labels, with Part 1's
supervised-from-scratch baseline alongside:

```
                       frozen probe   fine-tuned
    MAE                   49.2%         87.5%
    contrastive           95.1%         91.1%
    from scratch            --          84.9%
```

Read the two columns as a chiasmus, because the rows swap
places. Fine-tuning *transforms* the masked autoencoder — 49.2%
to 87.5%, past the from-scratch baseline: the class information
was in there all along, just not linearly arranged, and fifteen
labels sufficed to unlock it. Fine-tuning *hurts* the
contrastive encoder — 95.1% to 91.1%: its frozen features
already beat anything fifteen labels can teach, and gradient
steps on so few examples only overfit. Agreement features are
ready to wear; reconstruction features are raw material. The
same split holds nine orders of magnitude up: MAE backbones are
built to be fine-tuned, CLIP-style towers get frozen and
probed — and knowing which regime your downstream budget lives
in is precisely how you choose between these two pretexts.

### The reconstructor keeps everything

Why are the probes so different? Shift each test image and
measure how far its embedding moves, as a fraction of the
typical distance between two *different* images:

```
    contrastive: shift moves the embedding 0.35x a typical distance
    MAE        : shift moves the embedding 0.83x a typical distance
```

The contrastive embedding barely notices the shift. The MAE
embedding moves almost as far as jumping to a different image —
and a linear probe reads which half of the canvas the shape
sits in from the MAE embedding at 94.2%. Nothing surprising is
happening: reconstruction *needs* position, so position is
kept. Invariance was never asked for, so it never appears. The
contrastive encoder is **invariant** — it chose what to forget;
the masked autoencoder is **equivariant** — it is not allowed
to forget anything, because any detail it drops is a detail it
cannot redraw. That single design difference is the entire
probe gap, and the entire fine-tune flip.

And an honest footnote. MAE's most quoted number is its mask
ratio — 75%, chosen because natural images are so redundant
that easier ratios make the task trivial. On shape world:

```
    mask ratio   recon-mse   probe15   probe-all
       25%       0.0226      48.3%     68.7%
       50%       0.0296      48.7%     68.8%
       75%       0.0467      49.2%     67.6%
       90%       0.0609      49.6%     72.0%
```

Reconstruction gets three times harder from 25% to 90%; the
probes barely move. Shape world is so regular that even a
quarter of the pixels gives the whole game away — the 75% law
lives in the statistics of natural images, and it does not
automatically ride along to other data. Measure before you
inherit a hyperparameter.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Masked autoencoding's bill is refreshingly linear: one encoder
pass, one decoder pass, `O(B · L · h²)` per step, and no
`O(B²)` similarity matrix anywhere — the crowd of negatives
that dominated Part 1's budget simply is not on the invoice.
At transformer scale the accounting gets *better* still: a
ViT-based MAE encoder processes only the visible 25% of
tokens, quartering the cost of its most expensive component —
one reason MAE pretraining is cheap enough to be routine. The
decoder is scaffolding, discarded after pretraining, exactly
like Part 7's detective. The bill this pretext does pay sits
downstream: where a contrastive tower stops at a probe, a
masked backbone usually wants full fine-tuning — cheap per
task, but not free, and the right column of DEMO 2 is the
reason you pay it.

---

## Where the blanks went

Masked prediction is arguably the most consequential training
idea of the era, and it did not start with images. BERT's
masked words trained the encoder half of NLP; GPT's variant —
mask nothing, predict the *next* token — trained the rest, and
Section 13 will treat both properly. MAE brought the recipe to
vision with the 75% twist and made ViT pretraining cheap;
audio (HuBERT, wav2vec's masked steps), video (VideoMAE),
protein sequences (ESM) and time series followed, because the
recipe asks nothing of the data except redundancy. The
division of labour this article measured — reconstruction
backbones for fine-tuning, agreement towers for frozen probes
— is visible across the industry: the encoders behind search
and recommendation are contrastive; the backbones behind
detection, segmentation and medical imaging are masked
pretrained then fine-tuned. And the two philosophies merged:
DINOv2, today's default vision backbone, trains agreement and
masked prediction *together* — Part 3 meets Part 2 in one
loss, which is where this section is heading.

---

## What comes next

Part 3, **Self-Distillation (DINO)**, returns to Part 1's
agreement objective and removes the crowd: no negatives, no
contrast — just a student network chasing its own slowly-moving
teacher. Part 1 measured the 158× collapse that should cause;
DINO's centring-and-sharpening trick is the reason it doesn't,
and shape world will host the whole tightrope walk.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**mae.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/12-self-supervised-learning/02-masked-autoencoding/mae.py)

Run it with:

```bash
pip install numpy
python mae.py
```

It needs only `numpy` and runs in about five minutes.
Everything is from scratch: the patch masking, encoder,
decoder, the masked-pixel loss with hand-derived audited
gradients, a live retrain of Part 1's contrastive encoder,
linear probes, fine-tuning for all three contenders, and the
shift-sensitivity measurement. The headline insight worth
pinning to the wall: **hide three quarters of the input and
the model must learn what things are in order to redraw them —
but what it learns is arranged for redrawing, not for reading:
the linear probe scores 49.2% against the contrastive
encoder's 95.1% on the same fifteen labels, then fine-tuning
flips the table — MAE leaps to 87.5%, past the 84.9%
from-scratch baseline, while the contrastive encoder falls to
91.1% — because the reconstructor keeps everything (its
embedding moves 0.83× an image-to-image distance under a mere
shift, and position reads out at 94.2%) while the agreement
encoder kept only what its menu declared essential:
reconstruction features are raw material, agreement features
are ready to wear, and the fine-tuning budget downstream is
what should pick between them**.

---

*This is Part 2 of the Self-Supervised Learning track in the Algorithms in Python series. The companion script `mae.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It shares its shape-world stage with [Part 1](https://medium.com/p/37f7feeb811e)'s contrastive encoder — retrained live for every comparison — and its reconstruction machinery echoes Section 11's [VAE](https://medium.com/p/eb97ec9742ff). Part 3 will look at Self-Distillation (DINO).*
