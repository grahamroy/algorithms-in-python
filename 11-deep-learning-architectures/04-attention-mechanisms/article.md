# Attention Mechanisms — Distance Stops Existing

### *Algorithms in Python --- Deep Learning Architectures, Part 4*

---

Part 3 ended at a measured cliff. A recurrent network relates
step 40 to step 1 by *surviving* — one fact, carried through 39
hops of state, the error signal arriving ~300× weaker for the
trip, until at forty steps of distance the whole thing dropped
to chance. The failure wasn't capacity. It was the *route*:
information had to swim the whole channel to be used.

**Attention** (Bahdanau, Cho & Bengio, 2014) removes the route.
Step 40 does not wait for information to reach it — it **looks
at** step 1 directly. The mechanism is a soft dictionary
lookup, and three small matrices define it:

```
q = query   (what am I looking for?)          -- from the reader
K = keys    (what does each position offer?)  -- one per position
V = values  (what does each position say?)

weights = softmax( q · Kᵀ / √d_k )    -- a soft lookup
answer  = Σ_t  weights_t · V_t        -- a weighted average
```

Every position is exactly **one hop** from every other. There
is no channel to survive, no state to decay — distance stops
existing as an architectural concept. (The `√d_k` is a
housekeeping constant: dot products grow with dimension, and
unscaled they push the softmax into saturated, gradient-free
territory.)

This article builds one attention head from scratch — the
softmax Jacobian hand-derived, the whole backward pass audited
at **1.90e-10** — and runs it on Part 3's exact stage, where it
does something no architecture so far in this section could:
it makes the cliff *unbuildable*. Then it pays for its honesty
twice — once by showing what attention fundamentally cannot see
without help, and once by printing the quadratic bill that
defines the architecture built on top of it.

---

## A worked example: the cliff, the set, the receipt

### The cliff, unbuilt

Part 3's recall task, Part 3's budget: the first symbol is the
label, `T` distractors follow, and the RNN went from perfect at
30 steps of distance to chance at 40.

```
DEMO 1 --- The cliff, unbuilt
    distance T =  40:  100.0%
    distance T =  80:  100.0%
    distance T = 160:  100.0%
```

Flat — at four times the distance that destroyed the vanilla
RNN. And the gradient tells the same story before training
begins. On `T = 40` at initialisation:

```
    |dL/dE| at t=1: 1.8e-03   t=20: 2.1e-03   t=40: 8.2e-04
    farthest-to-nearest ratio: 2.2x   (the RNN's: ~300x, downhill)
```

The error signal reaches the *farthest* position as strongly as
the nearest — a ratio of 2.2 against the RNN's ~300, and even
that residue is positional-encoding texture, not decay. Where
Part 3's problem was "will the signal survive the trip," here
there is no trip.

### A set, not a sequence

Now the honest half of the bargain. Look back at the equations:
`q`, `K`, `V` see a **bag of vectors**. Shuffle the sequence
and every attention output is *identical* — the mechanism is
permutation-invariant by construction. Order does not exist for
it unless smuggled back in, which is the entire job of the
**positional encodings** (sinusoids of geometrically spaced
frequencies, added to the embeddings — a soft binary clock each
position wears). Remove them:

```
DEMO 2 --- A set, not a sequence
    recall (needs 'FIRST'):   with positions 100.0%   without  34.2%
    majority (order-free):    without positions 100.0%
```

Without positions, recall collapses toward chance — not because
the model is weak, but because *"first" is not a property of a
set*; the question has become inexpressible. And the control
proves the collapse is about order, not blindness: a majority
task (which symbol occurs more often — no order required) is
aced by the same position-free model, because a weighted
softmax average is a natural counting machine. Positional
encoding is not an implementation detail; it is the entire
concept of sequence, bolted back on.

### The receipt, and the bill

Where did the trained recall model actually look?

```
DEMO 3 --- attention weights, averaged over the test set
    weight on position 0 (the answer): 0.992
    average weight elsewhere:          0.0002
```

99.2% of the lookup lands on the one position that matters. The
RNN's memory was a vector of 24 unlabelled numbers; attention
leaves a **receipt you can read**. (One honest caveat the
literature earned the hard way: in deep multi-head stacks,
attention maps stop being straightforward explanations — but
the inspectability of a single head is real, and it is a large
part of why the mechanism conquered the field so quickly.)

Then the bill. This model is Bahdanau's original setting: *one*
query attending over `T` positions — `O(T)` per lookup. Modern
**self-attention** gives every position its own query — `T`
lookups over `T` positions. Exact multiply counts in the
attention core:

```
    T        one query      all T queries     growth
     40          1,312             53,792        --
     80          2,592            209,952      x3.9
    160          5,152            829,472      x4.0
    320         10,272          3,297,312      x4.0
```

Double the sequence, quadruple the bill: `O(T²)`. That is the
price of making distance free — and the defining engineering
constraint of the architecture that pays it.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The trade against Part 3 is exact and symmetrical. The RNN:
`O(T)` compute, `O(1)` distance-to-information, paid for with
sequential steps and a decaying signal. Attention: every
position one hop away, gradients intact, fully parallel across
positions — paid for with `O(T²)` score pairs and the memory to
hold them. Neither side of the ledger is free; the last twenty
years of sequence architecture is the story of which side you
would rather pay, and Part 3's Big-O table already whispered
the answer hardware chose.

---

## From translation alignment to "all you need"

Attention was born as a patch. 2014 neural translation squeezed
a whole French sentence through one fixed vector, and long
sentences degraded — so Bahdanau bolted a lookup onto the RNN:
let the decoder *glance back* at every source word as it
writes, and learn where to glance. The attention maps learned
soft word alignments nobody supervised, translation quality
jumped, and for three years attention lived as the RNN's
assistant. The 2017 heresy — *Attention Is All You Need*
(Vaswani et al.) — was to fire the RNN and keep the assistant:
stack self-attention with Part 1's MLPs, accept the quadratic
bill in exchange for total parallelism across a 10,000-GPU
cluster, and the modern era followed. Every large language
model running today is this article's mechanism, multiplied:
more heads, more layers, one causal mask — and Part 5
assembles exactly that machine.

---

## What comes next

Part 5, **Transformers**, takes this one head and builds the
full machine: multi-head self-attention (every position
querying every other, several lookups in parallel), the MLP
blocks from Part 1 stacked between them, residual connections
— Part 1's vanishing-gradient lesson, solved by addition — and
the causal mask that turns the whole thing into a language
model.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**attention.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/11-deep-learning-architectures/04-attention-mechanisms/attention.py)

Run it with:

```bash
pip install numpy
python attention.py
```

It needs only `numpy` and runs in about two minutes. Everything
is from scratch: embeddings, sinusoidal positional encodings,
the attention forward pass, the full hand-derived backward pass
(softmax Jacobian included) audited against central
differences, and Part 3's recall stage. The headline insight
worth pinning to the wall: **attention is a soft dictionary
lookup — softmax(q·Kᵀ/√d_k)·V — that puts every position one
hop from every other, so the RNN's cliff cannot be built: 100%
recall at distance 40, 80, and 160 with gradient reach flat
(2.2× vs the RNN's ~300×) and a readable receipt (99.2% of
attention on the answering position) — but the mechanism sees
a SET, collapsing to 34.2% the moment "first" must exist
without positional encodings, and giving every position its
own query costs O(T²): 53,792 multiplies at T=40, 3.3 million
at T=320**.

---

*This is Part 4 of the Deep Learning Architectures track in the Algorithms in Python series. The companion script `attention.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It dissolves the distance cliff measured in [Part 3](https://medium.com/p/9a17b15e8339) and prices the machinery the next part will assemble. Part 5 will look at Transformers.*
