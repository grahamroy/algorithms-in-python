# BERT-style Encoders — Reading in Both Directions

### *Algorithms in Python --- Natural Language Processing, Part 4*

---

Part 3 built a machine with two halves. The encoder read a
sentence; the decoder wrote a new one; attention was the
bridge between them. The article ended by pointing at the
halves and naming their descendants: BERT is the encoder,
GPT is the decoder.

This one throws the decoder away.

What is left is a stack that reads a sequence and emits a
vector per position — and immediately a problem. A decoder
trains itself on raw text for free: predict the next token,
and every token in the corpus is a labelled example.
An encoder has no next token. It has no output to compare
against. So what do you train it on?

```
    hide a word, and predict it from what surrounds it
```

That is **masked language modelling**, and it is the whole
idea behind BERT (Devlin et al., 2019). It is also Section
12's masked autoencoder, one modality over: hide part of
the input, reconstruct it, and the representations you
needed in order to reconstruct turn out to be the ones you
wanted all along.

Hiding a word has one consequence a decoder can never have.
Once the target is in the middle rather than at the end, the
model is free to read *forward* as well as back — and the
causal mask, which Part 3 needed so the decoder could not
cheat, becomes pure loss. BERT deletes it. Every position
attends to every other, in both directions.

The stage is still Part 1's synthetic micro-language, and it
was built for exactly this measurement. Agreement in that
language is carried by the verb form and the possessive,
both of which sit to the **right** of the subject:

```
    the king   rules  his    crown
    the kings  rule   their  crown
```

So hide the subject and the evidence for its number and
gender lies ahead of the gap, where only a bidirectional
reader can find it. Hide the possessive and the evidence
lies behind it, where anyone can. One corpus, two slots,
opposite verdicts — and no folklore required.

---

## A worked example: hide, predict, transfer

### The pretext and its recipe

Every sentence gets a `[CLS]` token in front and padding to
seven positions. Then 15% of the word positions are chosen,
and the input is corrupted at them — 80% become `[MASK]`,
10% become a random word, and 10% are left exactly as they
were. The model predicts the original token at every chosen
position:

```
    [CLS] the <[MASK]> <[MASK]> their queen
    [CLS] the mast drifts <[MASK]> ship
    [CLS] the <swords> defends his banners
    [CLS] the rooster harvests <[MASK]> seed
```

Angle brackets mark the chosen positions. The third line
shows the random replacement — `swords` is standing where
`lord` was — and the fourth kind of corruption, the 10% left
untouched, is the subtle one: a position may be scored
without being visibly disturbed, so the model can never
assume that what it reads is what it should keep.

Three encoders are pre-trained for 1,500 steps each: a
bidirectional one, a causal one wearing Part 3's triangular
mask, and a bidirectional one whose corruption is 100%
`[MASK]`. All three are 2 layers, 4 heads, d=64 — 114,223
parameters. The masked-LM loss, in nats, against 4.69 for a
uniform guess over the 109-token vocabulary:

![[LOSS CURVE IMAGE]]

Those are per-batch losses, which is why they wobble rather
than descend politely. The floor is not zero and should not
be: a masked subject is genuinely ambiguous — twelve nouns
could fill it — so part of that loss is irreducible. The gap
between the bidirectional and causal curves is the part that
is not.

Keep an eye on the third curve, because it matters later: the
100%-`[MASK]` encoder reaches the *lowest* loss of the three,
and — as the last section measures — transfers the worst.
Pre-training loss is a progress bar, not a target.

### What bidirectionality actually buys

Mask one slot at a time in 1,000 held-out sentences and
score the prediction against ground truth:

![[RECEIPT TABLE IMAGE]]

Read it by the middle column. Where the evidence lies to the
left, the two models are identical — both get the verb's
number and the possessive exactly right, every time. Where
the evidence lies to the right, the causal model collapses:
66.7% on the subject's number, which is precisely the base
rate for guessing *singular* every time, and 20.6% on
gender, which is worse than it sounds given that most
subjects have no gender to get wrong.

The causal model is not looking in the wrong place. There is
no place to look.

```
the distant [MASK] navigate their island
  gold: tides | bidir: tides | causal: compass

the noble [MASK] defend their banner
  gold: ladies | bidir: lords | causal: princess

the muddy [MASK] feeds the barns
  gold: plough | bidir: plough | causal: hen
```

Neither model can be certain *which* noun was hidden — and
the honest row in that table is the fourth one, where even
the bidirectional model gets the exact word only 45.5% of
the time. It is not guessing blindly: Part 1's role
signatures mean the object is the subject's signature
partner half the time, which is almost exactly the accuracy
we measure. What the bidirectional model reliably recovers
is the *grammar* — right number, right gender, right topic,
often the wrong noun.

And the receipt is readable, as it has been since Part 4 of
the Deep Learning track. Mean attention out of the masked
position, averaged over heads:

![[ATTENTION TABLE IMAGE]]

Layer 0 of the bidirectional model spreads its attention
evenly over the content words on both sides of the gap;
layer 1 concentrates half its weight on `their`, the token
that carries the number. The causal model's rows are zero
everywhere below the mask, because the mask forbids it — so
it spends its attention on the adjective behind it and on
the gap itself.

### Pre-train once, then spend labels

Pre-training is a means, not an end. The end is a task with
few labels. Here it is agreement acceptability — a
miniature of CoLA, the linguistic-acceptability task in
BERT's own GLUE benchmark. Half the sentences are generated
normally; half have their agreement broken:

```
    ok   the islands drift their storm
    bad  the noble banner defend the lords
    bad  the island navigates their captains
    ok   the wooden fence feeds the cow
```

Every word is in vocabulary and every sentence has the right
shape; only the agreement is wrong. Chance is 50%. Three
conditions get the same labels and the same 600 held-out
test sentences: the same architecture trained from scratch,
the pre-trained encoder with everything frozen except a
classifier head on `[CLS]`, and the pre-trained encoder
fine-tuned end to end.

![[LABELS TABLE IMAGE]]

At 100 labels, fine-tuning is 24 points ahead of the
identical architecture trained from scratch — roughly where
from-scratch gets with four times the labels. The
pre-trained encoder has never seen a label and never seen an
ungrammatical sentence. What it has is representations in
which agreement is already encoded, because predicting a
masked possessive *requires* knowing the subject's number.

The frozen probe is the control, and it is the interesting
failure. It never gets past 62%, at any label count. The
information is in there — fine-tuning the same weights
reaches 95.5% — but it is not linearly readable off the
`[CLS]` vector, because nothing in masked language
modelling ever gave `[CLS]` a job. That is the gap BERT's
second pre-training task, next-sentence prediction, was
there to fill. RoBERTa later dropped NSP and leaned on
fine-tuning instead, which is the column that works here
too. Mean-pooling the token vectors rather than reading
`[CLS]` does measurably better at this scale — 88.2% at 100
labels against 80.0%. `[CLS]` is a convention, not a
mechanism.

### The head was the bottleneck

There is a cheaper route, and it is the result I did not
expect to be this stark. Score a sentence by masking each
position in turn and summing the log probability of the word
that was really there — the pseudo-log-likelihood (Salazar
et al., 2020). No new parameters at all; the masked-LM head
already has an opinion:

![[PLL TABLE IMAGE]]

Twenty labels spent on fitting a single threshold beat
twenty labels spent on fine-tuning a classifier head, 90.5%
to 54.8%. The knowledge was in the encoder the whole time;
the randomly-initialised head was the bottleneck.

Honesty about the second row: the causal model scores these
sentences *better*. Every violation in this task is local
and visible from the left — a plural verb after a singular
subject is improbable to a left-to-right reader too. That is
not a crack in the earlier result, it is the boundary of it.
Bidirectionality is not a general superiority; it is an
advantage exactly where the evidence lies ahead, which is
what the fill-the-gap table measures and this task does not.

### Why 80/10/10 and not 100% [MASK]

The usual story about the 10% random replacement is that
without it the model would simply trust whatever word it is
shown, having only ever been asked about `[MASK]`. That
story is testable. Replace a subject with a noun from a
different topic, with no `[MASK]` anywhere in the sentence,
and read the masked-LM head at that position:

![[RECIPE TABLE IMAGE]]

The folklore is wrong, at least here. Neither model echoes
the word it was shown. Trained only ever to replace
`[MASK]`, the 100%-mask encoder still overrules a real token
that does not fit — it just does it less precisely, losing
ten points of topic accuracy. Downstream, fine-tuned on the
acceptability task, the gap persists but stays modest: 67.2%
against 61.0% at 50 labels, 80.0% against 70.8% at 100.

The direction matches BERT's ablation; the magnitude is a
few points on 600 test sentences and is not a result to bet
on. The recipe's published justification rests on ablations
over corpora a hundred million times this size. Reproducing
the direction is worth something. Reproducing the magnitude
would need BERT's budget.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Self-attention costs `O(n² · d)` per layer and the
feed-forward block `O(n · d²)`, so an `L`-layer encoder
forward is `O(L · n(n·d + d²))` — the same arithmetic as the
Transformer in Part 5 of the Deep Learning track, minus the
triangular mask. Dropping that mask costs nothing: the
scores are computed either way, and the mask only overwrites
half of them.

Two rows deserve more than a glance. The first is **signal
per pass**: masked language modelling is trained on 15% of
positions, where a causal language model gets a gradient
from every token in the sequence. BERT pays for its
bidirectionality in sample efficiency, which is why
pre-training it takes more passes over more text than
training a decoder of the same size.

The second is **scoring**. A causal model scores a sentence
in one forward pass, because its probabilities factorise
left to right. BERT has no such factorisation — scoring the
sentence above took `O(n)` forward passes, one per masked
position. The same property that makes an encoder good at
reading makes it bad at generating, and no amount of
engineering gets around it. BERT cannot write.

---

## Where BERT sits

BERT (Devlin et al., 2019) did not introduce the
Transformer, attention, or masking. It introduced a
*default*: pre-train one large encoder on unlabelled text,
then fine-tune it on whatever small labelled task you have.
Before BERT, a new NLP task meant a new architecture and a
corpus to go with it. After it, the task meant a head and a
few thousand labels. The GLUE leaderboard turned over in
months.

The lineage in this series is direct. Part 1 built word
vectors that were *static* — `bank` had one vector whether
it sat by a river or held your money. Part 2 decided which
subword pieces words decompose into; BERT's WordPiece is
literally Part 2's second algorithm. Part 3 built the
encoder-decoder pair, and this article keeps the encoder,
deletes the mask, and swaps the training signal from
translation to reconstruction. The result is the thing word
embeddings could not be: a vector *per occurrence*, computed
from the company that particular word is keeping.

What BERT's descendants changed was mostly the recipe, not
the machine. RoBERTa dropped NSP, trained longer, and masked
dynamically. ALBERT shared parameters across layers.
DeBERTa disentangled content from position. ELECTRA
replaced masked prediction with corruption detection — which
is, almost exactly, the acceptability task above. The
encoder stayed where it was.

---

## What comes next

Part 5, **GPT-style Decoders**, takes the other half. The
causal mask comes back, and with it the one thing this
article's encoder cannot do: generate. The question is what
you gain when the price of reading only backwards is paid
willingly — a gradient from every token, a sentence scored
in one pass, and text that runs on past its prompt.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**bert_encoder.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/13-natural-language-processing/04-bert-style-encoders/bert_encoder.py)

Run it with:

```bash
pip install numpy
python bert_encoder.py
```

It needs only `numpy` and runs in about four minutes.
Everything is from scratch: bidirectional multi-head
self-attention with a padding mask, LayerNorm, learned
positional embeddings, the masked-LM head and the `[CLS]`
head, the 80/10/10 corruption recipe, fine-tuning and frozen
probing, and pseudo-log-likelihood scoring — all
hand-differentiated and audited against central differences
at startup (worst error 8.22e-10 over 330 sampled
gradients). The headline insight worth pinning to the wall:
**bidirectionality is not a general improvement, it is a
precisely located one — on the slots whose evidence lies to
the left, a causal encoder ties the bidirectional one at
100%, and on the slot whose evidence lies to the right it
scores 66.7%, which is exactly the base rate for guessing
the majority class, against 99.8%; the pre-training that
buys this also buys transfer, taking a 100-label task from
55.7% to 80.0%, but the frozen probe never passes 62%,
because masked language modelling gives `[CLS]` nothing to
do — which is what next-sentence prediction was for, and why
fine-tuning, not probing, is how BERT is used; and the
cheapest route of all skips the head entirely, scoring
sentences straight from the masked-LM output at 90.5% with
twenty labels spent on a single threshold**.

---

*This is Part 4 of the Natural Language Processing track in the Algorithms in Python series. The companion script `bert_encoder.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It reuses Part 1's synthetic micro-language ([word embeddings](https://medium.com/p/164f7f542a3c), [BPE tokenisation](https://medium.com/p/5e6a51341455), [sequence-to-sequence](https://medium.com/p/bbdf65112819)). Part 5 will look at GPT-style Decoders.*
