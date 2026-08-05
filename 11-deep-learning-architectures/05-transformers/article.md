# Transformers — The Section, Assembled

### *Algorithms in Python --- Deep Learning Architectures, Part 5*

---

Nothing in this article is new. That is the entire point of it.

The **transformer** (Vaswani et al., 2017 — *Attention Is All
You Need*) is not another architecture in the sequence this
section has been building; it *is* the sequence this section
has been building, bolted together:

- **Token embeddings + sinusoidal positions** — Part 4, where
  order had to be smuggled back into a set.
- **Multi-head causal self-attention** — Part 4's soft lookup,
  multiplied: every position now queries every earlier one, and
  several heads run in parallel so different specialists can
  emerge.
- **The MLP block** — Part 1's machine, sitting between every
  pair of attention layers and holding most of the parameters.
- **Residual connections** — `x + f(x)`: Part 1's vanishing
  gradient, solved by *addition*, because gradient flows through
  a sum untouched.
- **Layer normalisation** — keeping every residual stream at
  unit scale so the additions stay sane.

Plus exactly one new idea, and it is a triangle: the **causal
mask**. Position `t` may attend only to positions `≤ t` — which
converts the whole machine into a next-token predictor. Feed it
text, ask it to predict each character from the ones before,
and you have a **language model**.

The companion script builds all of it in NumPy — LayerNorm
Jacobian, softmax Jacobian, multi-head reshapes, every backward
pass by hand, 150 sampled gradients audited at **6.53e-10** —
and then runs the three demonstrations this section has been
driving toward: the assembly justified, an algorithm learned
and *read out of the weights*, and a machine that writes.

---

## A worked example: the bridge, the mirror, the writer

### Why the residuals are in the box

Every component earned its seat in earlier parts except one —
the residual connection has been asserted, never measured.
Four layers deep, on the reversal task below, same parameters,
same budget:

```
DEMO 1 --- The assembly, and why the residuals are in it
    with residuals   : loss 0.195
    without residuals: loss 0.739
```

The only difference is whether the gradient may cross each
block by addition or must fight through it. Part 1 showed six
sigmoid storeys starving the bottom of a stack; the residual
stream is the reason modern "deep" means 96 layers and not 6 —
every layer learns a *correction* to a signal that always
flows.

### A transformer learns an algorithm

Read 8 symbols, then a separator, then emit them in reverse.
One layer, **two** heads, a thousand training steps:

```
DEMO 2 --- A transformer learns an algorithm
    token accuracy 99.8%   sequence exact-match 98.6%

    attention peaks per output step, over the 8 inputs:
    head 0: [7, 6, 5, 4, 3, 2, 7, 7]
    head 1: [7, 6, 5, 4, 3, 2, 1, 0]   <-- the mirror, learned
```

Head 1 **is** the algorithm: producing output `i`, it attends
to input `8−1−i` — a clean anti-diagonal that nobody
programmed, sitting in the attention map like a receipt. Head 0
learned most of the same mirror and hands off the rest. This
division of labour — one mechanism, several specialists — is
the practical answer to why *multi*-head attention beats one
big head: each head is cheap, and each can commit to a
different relation.

### A transformer learns a language

The final demo trains the same machine — two layers, four
heads, 66k parameters — as a character-level language model on
a corpus with a certain familiarity: **this section's own
sentences**, 1,447 characters of them.

```
DEMO 3 --- A transformer learns a language
    step:  0      100    500    1500
    loss:  3.87   2.29   0.21   0.12     (nats/char; uniform = 3.37)
```

And then the machine writes. Prompt `"the gradient "`,
temperature 0.4:

```
"the gradient flows through the addition untouched. the
 attention map is a receipt you can read: the modell shows
 where it looked. one celll, any length: "
```

Prompt `"the moon "` — which appears nowhere in the corpus:

```
"the moon celll state is a highway through time. width
 memorises and depth composes: the first layer builds curves
 from "
```

Temperature 2.0:

```
"the gradient flowss tat step ofis atace is ane mplace is an
 affi linear layer. the network learns by walking the chavked. t"
```

Honesty about what happened: 66k parameters on 1.4k characters
is a **memoriser** — it recites its corpus (with the occasional
stutter — "modell", "celll" — where two grooves overlap),
absorbs an unseen prompt into the nearest memorised channel,
and dissolves into alphabet soup when temperature melts its
certainties. No intelligence was created in the making of this
demo. But the mechanism doing the reciting — embed, attend,
correct, predict, sample — is *identical* to the one that, at
roughly a hundred million times the scale, wrote half the text
you read this week. Scale changed the wattage. Parts 1 through
5 are the whole circuit.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Per layer: `O(T² · d)` for attention (Part 4's bill, now paid
in full) plus `O(T · d²)` for the MLP — and which term
dominates depends on whether the sequence or the width is
bigger, a fact that shapes real model design. Parameters are
`O(L · d²)`, independent of `T` as always. The memory that
hurts is the `O(T²)` of attention weights per layer per head —
the reason "context length" is a marketed number. And training
parallelises across *all* positions at once (the mask, not the
clock, enforces order), which is the property that ended the
recurrent era; generation, ironically, is still one token at a
time.

---

## What the assembly bought, historically

The 2017 paper's title was a provocation aimed at Part 3:
attention had been the RNN's assistant, and the authors fired
the RNN. What remained trained in parallel over a whole corpus
on a whole cluster — and it *scaled*: loss falling as smooth
power laws in parameters, data, and compute, further than
anyone expected. BERT read with it; GPT wrote with it; then
GPT-3, and the era in which this article is being read. Nearly
every large model since is this exact machine with engineering
around the bill — KV-caches to reuse yesterday's attention,
FlashAttention to stop materialising the `T²` matrix,
mixture-of-experts to grow the Part 1 blocks without paying
for them every token — and the residual stream is still the
spine of all of it.

---

## What comes next

Part 6, **Graph Neural Networks**, drops the last assumption
standing: that data comes in a grid or a line at all. Molecules,
social networks, and road maps arrive as nodes and edges — and
message passing turns out to be attention's cousin, with the
graph deciding who may look at whom.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**transformer.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/11-deep-learning-architectures/05-transformers/transformer.py)

Run it with:

```bash
pip install numpy
python transformer.py
```

It needs only `numpy` and runs in three to four minutes.
Everything is from scratch: pre-LN decoder blocks, causal
multi-head self-attention, LayerNorm with its hand-derived
Jacobian, the sampled gradient audit, string reversal, and the
character-level language model with temperature sampling. The
headline insight worth pinning to the wall: **a transformer is
this section assembled — Part 4's attention with every position
querying every earlier one, Part 1's MLP holding most of the
parameters, residual connections carrying the gradient across
96 layers by addition (measured: loss 0.195 with them, 0.739
without), and one causal triangle turning it into a next-token
predictor — a machine that learns algorithms you can read out
of its attention maps (a perfect anti-diagonal for string
reversal, 98.6% exact) and languages you can sample from (3.87
→ 0.12 nats/char), and whose scaled-up twin wrote half of what
you read this week**.

---

*This is Part 5 of the Deep Learning Architectures track in the Algorithms in Python series. The companion script `transformer.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It assembles [Part 1](https://medium.com/p/9c0d36baa5bc)'s MLP, [Part 4](https://medium.com/p/a27e38010fd8)'s attention, and the lessons of [Part 3](https://medium.com/p/9a17b15e8339) into the machine that ended the recurrent era. Part 6 will look at Graph Neural Networks.*
