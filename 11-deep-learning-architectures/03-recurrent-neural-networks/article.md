# Recurrent Neural Networks — The Wall Is Distance, Not Length

### *Algorithms in Python --- Deep Learning Architectures, Part 3*

---

Part 2 shared nine weights across every position of an image.
This part points the same idea down the other axis of the
world: **time**. A sequence has exactly the structure an image
has — local order matters, and the same pattern means the same
thing whenever it occurs — so the recurrent network shares one
cell across every step:

```
h_t = tanh( x_t·Wx  +  h_{t-1}·Wh  +  b )
```

The same `Wx` and `Wh` at step 3 and step 300. The hidden state
`h` is the network's running summary of everything so far —
its memory — and the parameter count never grows with sequence
length, for the same reason Part 2's kernel never grew with the
image. Training is **backpropagation through time**: unroll the
recurrence, then walk Part 1's chain rule backwards along the
sequence.

And right there, the section's recurring villain returns with a
sharper knife. Part 1's vanishing gradient decayed through a
*stack* of layers — six storeys, ×25,000 attenuation. In a
recurrent network, every step of *distance* between cause and
consequence multiplies the error signal by the same squashing
recurrent Jacobian — forty steps means forty multiplications,
by construction. This article measures what that does, finds
that the failure is not a slope but a **cliff**, and then opens
up the machine built to survive it: the **LSTM**.

---

## The stage: carry one fact through the noise

The task is **recall**: the *first* symbol of a sequence (one
of four) is the label; then `T` distractor symbols follow. To
answer at the end, the network must carry exactly one fact
across `T` steps of noise. `T` is a dial for *distance between
evidence and verdict* — and nothing else about the task changes
as it grows. Both cells' backward passes are audited against
central differences at startup (2.48e-10 and 2.56e-10 — Part
1's ritual, applied to BPTT).

---

## A worked example: the control, the cliff, the cell

### One cell, any length

```
DEMO 1 --- One cell, any length
    vanilla RNN :   796 params   test 100.0%     (recall, T=10)
    LSTM        :  2884 params   test 100.0%

    control -- answer is the LAST symbol, T=40:
    vanilla RNN :  99.5%      LSTM :  98.8%
```

At `T=10` both cells are perfect, with parameter counts a
fraction of Part 1's smallest useful MLP — and independent of
`T`. The control is the important line: a **forty-step**
sequence whose answer sits at the *end* is no obstacle at all
(99.5% for the vanilla cell). Length is not the problem. Hold
that thought.

### The wall is distance, and it is a cliff

```
DEMO 2 --- The wall is distance, and it is a cliff
    distance T    vanilla RNN    LSTM
       20          100.0%      100.0%
       30          100.0%       98.8%
       40           24.0%      100.0%
```

No graceful decline: the vanilla RNN is *perfect* at thirty
steps of distance and at **chance** at forty (four symbols —
25%). Combined with the control, the verdict is exact: the
vanilla cell fails precisely when information must *survive*
distance, not when sequences are merely long.

The mechanism is visible before training even begins — the
error signal's norm as it travels backwards to `t = 1` on the
`T = 40` task, at initialisation:

```
    step:      t=40      t=30      t=20      t=10      t=1
    RNN :    2.4e-02   3.7e-03   8.5e-04   2.2e-04   8.3e-05
    LSTM:    1.7e-02   5.9e-03   3.0e-03   1.4e-03   6.2e-04
```

The RNN's signal arrives at the first step attenuated ~300×;
the LSTM's arrives an order of magnitude stronger — *with
untrained, default gates*. Adam can rescale a small gradient,
but it cannot restore the information a squashing Jacobian has
destroyed forty times over; past the cliff, what reaches `t=1`
is as much noise as signal.

### Inside the cell

The LSTM (Hochreiter & Schmidhuber, 1997) survives the trip by
refusing to multiply by a matrix at all. It keeps a second
state — the **cell** `c` — updated *additively*, with three
learned gates deciding what to forget, write, and reveal:

```
i, f, o = sigmoid gates      g = tanh(candidate)
c_t = f * c_{t-1} + i * g        <-- the additive highway
h_t = o * tanh(c_t)
```

Walk the backward pass along the cell path and the step-to-step
factor is just `f` — elementwise, a learned number near 1 — not
a squashing matrix. The script initialises the forget bias at
+1, so newborn LSTMs are *biased to remember* (that is where
DEMO 2's order-of-magnitude head start came from), and training
pushes the units it relies on higher:

```
DEMO 3 --- forget gates on the trained T=40 recall LSTM
    at init  : mean 0.74   top-3 units [0.795 0.785 0.784]
    trained  : mean 0.79   top-3 units [0.888 0.874 0.872]
```

And then the honest boundary, because gates are not magic.
**Parity** of 40 bits needs no long-range storage — one bit,
updated every step — but its credit assignment is brutal: every
input matters and flipping any one flips the answer.

```
    parity, T = 40:  vanilla RNN 47.8%   LSTM 49.2%   (chance: 50%)
```

Both cells fail. Gates cure *vanishing gradients*; they do not
cure every hard lesson a long horizon can teach. Knowing the
difference is knowing what an LSTM actually is: not memory
magic — an architectural guarantee that the error signal
survives the trip.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Everything is linear in sequence length — `O(T · H²)` per
sequence forward, the same backwards (Part 1's law holds
through time) — with parameters `O(H²)`, independent of `T`.
The LSTM pays exactly 4× the vanilla cell (four gate blocks
through one fused matrix). The costs that actually shaped
history are the other two: BPTT must *store* every timestep's
activations (`O(T · H)` memory), and the computation is
**inherently sequential** — step `t` cannot start before step
`t−1` finishes, on any hardware, at any budget. Remember that
last one; it is the door the next two articles walk through.

---

## The GRU, and where the recurrence went

The **GRU** (Cho et al., 2014) is the LSTM's streamlined
sibling — two gates instead of three, no separate cell, ~25%
fewer parameters, and on most mid-sized tasks the same
accuracy; the principle (gated, additive state) is identical.
For a decade this family *was* sequence modelling: translation,
speech, language models. What ended the monopoly was not
accuracy but the Big-O table's last line — sequential
computation cannot be parallelised across a 10,000-GPU
cluster, and a mechanism that pays `O(T)` steps of latency to
relate the first token to the last was outrun by one that pays
`O(1)`. But recurrence is not history: its `O(H)` inference
state — constant memory regardless of context length — is
exactly why state-space models (Mamba and kin) have brought
the recurrent idea roaring back for long-context work. The
wheel is still turning.

---

## What comes next

Part 4, **Attention Mechanisms**, removes the middleman
entirely. The RNN relates step 40 to step 1 by *surviving* 39
hops of state; attention lets step 40 simply *look at* step 1
— every position queries every other directly, distance
becomes irrelevant, and the cliff in DEMO 2 becomes
unbuildable. The price is a new Big-O row — `O(T²)` — and the
subject of the two most consequential architecture articles in
this section.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**rnn.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/11-deep-learning-architectures/03-recurrent-neural-networks/rnn.py)

Run it with:

```bash
pip install numpy
python rnn.py
```

It needs only `numpy` and runs in 3–4 minutes. Everything is
from scratch: both cells, full backpropagation through time,
the forget-bias initialisation, Adam, the startup gradient
audits, and the recall and parity stages. The headline insight
worth pinning to the wall: **a recurrent network is weight
sharing across time — one cell, any length, 796 parameters —
and its failure mode is distance, not length: perfect recall
through 30 steps of noise, chance at 40, while a 40-step task
with the answer at the end stays trivial; the LSTM survives
because its cell state updates additively and its backward
pass multiplies by a learned forget gate near 1 instead of a
squashing matrix 40 times — an order-of-magnitude stronger
signal at t=1 before training even starts — but gates cure
vanishing gradients, not every long horizon: 40-bit parity
defeats both cells honestly**.

---

*This is Part 3 of the Deep Learning Architectures track in the Algorithms in Python series. The companion script `rnn.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It carries [Part 1](https://medium.com/p/9c0d36baa5bc)'s vanishing gradient into the time dimension and applies [Part 2](https://medium.com/p/6f860b54044f)'s weight sharing to sequences. Part 4 will look at Attention Mechanisms.*
