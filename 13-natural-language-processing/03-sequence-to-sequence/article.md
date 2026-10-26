# Sequence-to-Sequence — The Encoder Reads, the Decoder Writes

### *Algorithms in Python --- Natural Language Processing, Part 3*

---

Part 2 decided which pieces a word should be built from.
BPE merged the most frequent pairs, WordPiece merged the
most surprising, Unigram pruned from above — three
algorithms, one promise: every word decomposes into known
fragments. But tokens in a row are not yet a *language*.
A tokeniser says what the pieces are; it says nothing
about what happens when you read a sequence of them and
need to produce another.

This article asks the next question:

```
    given an input sequence, produce an output sequence.
```

That is the Seq2Seq problem (Sutskever et al., 2014). An
**encoder** reads the input and compresses it into a
fixed-size vector. A **decoder** unrolls that vector into
output tokens, one at a time. Everything the decoder needs
must pass through that single vector — the *bottleneck*.

Bahdanau et al. (2015) broke the bottleneck with
**attention**: instead of one vector, the decoder peeks at
every encoder state and picks what is relevant at each
step. The model learns *alignment* — which input tokens
matter for each output token — as a side effect.

The stage is still Part 1's synthetic micro-language —
107 words, three topics — but with a new task:
**word-order translation** from SVO (subject-verb-object)
to SOV (subject-object-verb), mirroring real
English-to-Japanese/Korean translation. Short sentences
(5–6 words) and long sentences (two clauses joined by
"and", 11–13 words) test the bottleneck: the fixed
vector struggles on longer sequences; attention does not.

---

## A worked example: encode, decode, attend

### The task: SVO to SOV

The micro-language generates SVO sentences. The
translation task moves the verb to the end of each
clause:

```
    the thrones crown their queen
    -> the thrones their queen crown

    the mast drifts the ship
    -> the mast the ship drifts

    the lord defends his banners
    -> the lord his banners defends
```

For long sentences — two clauses joined by "and" — each
clause transforms independently:

```
    the castle crowns the king and the king defends
    his knight
    -> the castle the king crowns and the king his
    knight defends
```

The vocabulary is shared between source and target (107
tokens). Training uses 1,500 short pairs and 750 long
pairs; the test set has 400 short and 200 long.

### The encoder-decoder architecture

The encoder is an Elman RNN that reads the source
sequence left to right. At each step `t`, it updates a
hidden state:

```
    h_t = tanh(W_x · x_t + W_h · h_{t-1} + b)
```

where `x_t` is the embedding of the current token. After
reading the entire source, the encoder's final state
`h_T` contains — in theory — everything the decoder
needs.

The decoder is a second RNN, initialised with that final
state. At each step it receives the previous output
token's embedding, updates its own hidden state, and
predicts the next token through a softmax over the
vocabulary:

```
    s_t = tanh(W_x · y_{t-1} + W_h · s_{t-1} + b)
    P(y_t | y_{<t}, x) = softmax(W_o · s_t + b_o)
```

During training the decoder sees the *correct* previous
token (teacher forcing); during inference it feeds its
own predictions back.

### Without attention: the bottleneck

Trained for 30 epochs on all 2,250 pairs (lr = 0.002,
Adam, gradient clipping at 5.0), the no-attention model
converges — the loss drops from 2.05 to 0.95 — but the
numbers tell the story:

```
    Accuracy  short:   4.5%
    Accuracy  long:    0.0%
```

Not a typo. The model learns the right *format* — it
produces sequences of the right length with real words
— but the content is wrong:

```
    the noble ladies defend their banner
    gold:  the noble ladies their banner defend
    pred:  the noble kings their knight rule  (WRONG)

    the fertile cow grazes her tractor and the islands
    sail their harbour
    gold:  the fertile cow her tractor grazes and the
    islands their harbour sail
    pred:  the deep compass the waves anchors and the
    rooster his seed grazes  (WRONG)
```

The decoder receives one vector — 48 floating-point
numbers — and must reconstruct the entire output
sequence from it. For a 5-word sentence, that is roughly
10 numbers per word. For a 13-word sentence, fewer than
4. The information does not fit. The bottleneck is
real, and it is the reason Sutskever et al.'s original
architecture was eventually replaced.

### With attention: the bottleneck breaks

Bahdanau attention gives the decoder a window into the
full encoder. At each decoding step, the model computes
an *attention score* for every encoder state, then takes
a weighted sum:

```
    e_j = v^T · tanh(W_a · h_j + U_a · s_{t-1})
    a_j = softmax(e)_j
    c_t = sum_j  a_j · h_j
```

The decoder's input becomes the concatenation of the
previous token's embedding and the context vector `c_t`.
Now the decoder can *reach back* into the encoder at
every step — it is no longer bottlenecked.

Same architecture (d_emb = 32, d_hid = 48), same
training regime, same data. The loss drops from 1.96
to 0.001:

```
    Accuracy  short: 100.0%
    Accuracy  long:   99.5%
```

From 4.5% to 100% on short sequences, from 0.0% to
99.5% on long ones. The only change: attention.

```
    the distant tides navigate their island
    gold:  the distant tides their island navigate
    pred:  the distant tides their island navigate  (ok)

    the fertile cow grazes her tractor and the islands
    sail their harbour
    gold:  the fertile cow her tractor grazes and the
    islands their harbour sail
    pred:  the fertile cow her tractor grazes and the
    islands their harbour sail  (ok)
```

### The attention receipt

The attention weights are readable — they show where the
decoder *looks* when producing each output token:

![[ATTENTION TABLE IMAGE]]

Each row is one decoding step; each column is one encoder
position. Read down the columns: `distant` in the output
attends 0.99 to `distant` in the input. `tides` attends
0.99 to `tides`. `island` attends 0.99 to `island`.

The interesting row is `navigate` — the verb in the
output, which must move from the middle of the input
(position 4) to the end of the output (position 6). The
decoder assigns 1.00 to the verb's encoder position.
The model learned the alignment between input and output
positions as a side effect of the translation task.

This is the same "receipt" idea from Part 4 of the Deep
Learning track (Attention Mechanisms): attention weights
are human-readable because they sum to 1 and concentrate
where the model finds information. The difference is
that here, the receipt spans *two* sequences — encoder
and decoder — rather than self-attending within one.

---

## The bottleneck receipt

```
                        short     long
                        -----     ----
    no attention        4.5%    0.0%
    attention         100.0%   99.5%
```

Without attention, everything the decoder needs must pass
through one vector. Short sequences survive the
compression; long ones do not. Attention lets the decoder
reach back into the encoder at every step — the
bottleneck breaks, and accuracy on long sequences
recovers.

The loss curves tell the same story. Without attention,
the loss plateaus at 0.95 — the model cannot compress
long sequences into one vector, so it learns a generic
output distribution instead. With attention, the loss
reaches 0.001 — the model finds the exact mapping.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The encoder runs one RNN step per source token:
`O(T · d²)` where `T` is source length and `d` the
hidden dimension (the matrix-vector product at each step
is `O(d²)`). The decoder runs one RNN step per target
token: `O(T' · d²)`. Attention adds `O(T' · T · d)` —
at each of `T'` decoding steps, we score all `T` encoder
states. Total: `O((T + T') · d² + T' · T · d)`.

For comparison, the Transformer (Part 5 of the Deep
Learning track) replaces the sequential RNN with
parallel self-attention: `O(n² · d)` where `n` is the
full sequence length. Seq2Seq's RNN is cheaper per-step
but inherently sequential — each hidden state depends on
the previous one, so the encoder cannot be parallelised.
This is the architectural bottleneck that the Transformer
removed, and the reason Seq2Seq is no longer the
dominant paradigm. But the encoder-decoder *structure*
survived: T5, BART, and machine translation models
still use it.

---

## Where Seq2Seq sits

Seq2Seq (Sutskever et al., 2014) launched neural machine
translation. The original paper used LSTMs and reversed
the source sentence to help the decoder; Bahdanau et al.
(2015) added attention and the field never looked back.
Google's Neural Machine Translation system (Wu et al.,
2016) scaled the idea to production, and for three years
Seq2Seq with attention was the state of the art for
translation, summarisation, and dialogue.

The connection to this series is direct. Part 1 built
word embeddings — dense vectors that encode meaning.
Part 2 decided which subword pieces to use. This part
takes sequences of those pieces and produces new
sequences from them. The encoder is the same RNN from
Part 3 of the Deep Learning track (RNN / LSTM / GRU);
the attention mechanism is Part 4's; the decoder is the
autoregressive generation pattern that GPT will later
use without an encoder. Seq2Seq is where NLP became
*generative* — the first architecture that could read
a sentence and write a different one.

The two halves became the building blocks of everything
that followed. BERT (Part 4, next) is the encoder: it
reads bidirectionally and produces contextual
representations. GPT (Part 5) is the decoder: it writes
left-to-right, one token at a time. The Transformer
kept the encoder-decoder split but replaced the RNN with
self-attention — and that is the architecture we will
reach in two parts.

---

## What comes next

Part 4, **BERT-style Encoders**, takes the encoder half
of Seq2Seq and asks: what if we *only* encode? Masked
language modelling, bidirectional context, and the
pre-train-then-fine-tune paradigm that made transfer
learning the default in NLP.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**seq2seq.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/13-natural-language-processing/03-sequence-to-sequence/seq2seq.py)

Run it with:

```bash
pip install numpy
python seq2seq.py
```

It needs only `numpy` and runs in about two minutes.
Everything is from scratch: the Elman RNN encoder,
Bahdanau attention with learned alignment, the decoder
with teacher forcing and greedy inference, hand-derived
BPTT gradients audited against central differences
(worst-case relative error: 1.05e-06), and Adam with
gradient clipping. The headline insight worth pinning to
the wall: **the encoder-decoder bottleneck is real and
measurable — 4.5% accuracy on short sequences and 0.0%
on long ones when the decoder has only one vector to
work with; 100% and 99.5% when attention lets it reach
back into the encoder at every step; the attention
weights are human-readable (the verb in the output
attends 1.00 to the verb in the input, nouns attend
0.99 to their source positions), making the model's
alignment decisions as transparent as a lookup table;
this is the architecture that launched neural machine
translation and whose two halves — encoder and decoder
— became BERT and GPT**.

---

*This is Part 3 of the Natural Language Processing track in the Algorithms in Python series. The companion script `seq2seq.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It reuses Part 1's synthetic micro-language ([word embeddings](https://medium.com/p/164f7f542a3c), [BPE tokenisation](https://medium.com/p/5e6a51341455)). Part 4 will look at BERT-style Encoders.*
