# Retrieval-Augmented Generation — Reading, or Reciting?

### *Algorithms in Python --- Natural Language Processing, Part 8*

---

RAG is two parts bolted together, and this track has already
built both. Part 6 ended on a bi-encoder, whose sentence
vectors do not depend on what they are compared against and
so can be computed once and cached — the only reason search
over millions of documents is possible at all. Part 7 built
the generator and measured what it costs to turn a
distribution into words.

Put the first in front of the second and you have the whole
architecture:

```
question -> retriever -> document -> generator -> answer
```

The appeal is obvious. The model no longer has to know
things; it has to read them. Facts can be updated by editing
a store instead of retraining a network, and the answer comes
with a citation.

Then comes the question that is almost impossible to answer
in a real system:

> When the model answers, is it reading the document, or is
> it reciting what it already believed?

You cannot tell from an accuracy score, because the retrieved
document and the model's prior usually say the same thing.
Here, for once, they can be made to disagree on purpose.

---

## A worked example: a prior you can aim at

### The role signature is the prior

Part 1's micro-language gives every subject a **role
signature** — the object the language expects after it. The
oracle says exactly how strong that expectation is, and the
pre-trained decoder has absorbed it completely: asked to
finish `the king commands his`, the model names the role
signature **97.22%** of the time, with no document anywhere
in sight.

So a knowledge base can be built to fight it. Each of the
144 facts — one per subject-and-verb pair — names an object
of the same topic, so every document is a grammatical
sentence and the planted ones cannot be spotted by their
shape. What varies is whether the fact **agrees** with the
role signature or **contradicts** it:

```
question    : what does the king command
role says   : knight
agreeing doc: the king commands his knight
contra doc  : the king commands his castle
```

Two documents, one question, one model. On the agreeing
document, reading and reciting produce the same word. On the
contradicting one they produce different words, and the
difference between those two accuracies is grounding, with
nothing else varying.

One more guard: the 144 facts are split 108 train / 36 test
**by subject-and-verb pair**, so no test answer can be
recalled from fine-tuning. The document is the only place the
answer exists.

### The retrieval half

![[RETRIEVAL TABLE IMAGE]]

The questions are built so that overlap alone cannot solve
them: the question carries the verb's base form (`command`)
while the document carries the inflected one (`commands`).

The lexical baseline is not failing to *find* the document.
It is failing to **choose**. On average only 0.08 documents
outscore the right one — but 10.11 documents **tie** with it,
and a tie is settled by whatever order the store happens to
be in. Overlap sees `the` and the subject and nothing more.
Worse, several verb stems in this vocabulary (`crown`,
`plough`, `anchor`) are also nouns, so documents about
entirely unrelated subjects join the tie.

The bi-encoder has learned that `command` and `commands` are
the same relation, and that is the whole of its job. It is
also the whole argument for dense retrieval.

With recall@1 at **1.0000**, retrieval is solved at this
scale. That is convenient rather than disappointing: every
error from here on belongs to the generator, and there is no
need to argue about which half is at fault.

### Is it reading, or reciting?

![[CONDITIONS TABLE IMAGE]]

Three outcomes, counted separately: the model answered with
the **document's** object, with the **role signature**, or
with neither.

Start with the second row. The two numbers are identical, and
not by coincidence — on an agreeing fact the document says
exactly what the role signature says, so a model that reads
and a model that recites emit the same word. **That row
cannot distinguish them.** It is also the row that every
naturally-collected benchmark is built out of, because
documents found in the world mostly agree with what a model
trained on the world already believes.

Now the third row. Same generator, same question, a gold
document in front of it: **1.0000** when the document agrees
with the prior, **0.3056** when it does not. The document is
barely being read. The agreeing score was a coincidence that
looked like comprehension.

The fourth row is the failure mode promised at the end of
Part 7. Handed a document about the right subject and the
wrong relation, the model answers with that document's object
16.67% of the time and falls back on the role signature
50.00% of the time. Both are wrong, both are fluent, and
neither is supported by the text it was given.

### Where grounding actually comes from

None of this is a property of the architecture. It is a
property of what the generator was fine-tuned on. Sweeping
the share of **training** documents that contradict the prior
moves every column:

![[MIX TABLE IMAGE]]

![[GROUNDING CURVE IMAGE]]

The top row is the pipeline a careful engineer builds by
accident. Every training document agrees with the model's
prior — which is simply what collected evidence looks like —
and the result scores a flawless **1.0000** on agreeing facts
and **0.0000** on contradicting ones, reciting the role
signature every single time. It has learned that the document
is decoration.

Read the last two columns together, because this is the
finding worth keeping. As the training mix adds contradicting
evidence, **reciting the role signature falls cleanly from
1.0000 to 0.0278** — and **being misled by a wrong document
rises just as cleanly from 0.0000 to 0.8056**. These are not
two behaviours that happen to correlate. They are one
behaviour under two names. A model that will let a document
overrule what it knows is, by construction, a model that a
bad retrieval can walk into the wrong answer.

Which means the retriever is not a component you can improve
independently of the generator. It is the thing that decides
which of those two failure modes you get.

The middle of the sweep explains its own untidiness. The
"reads the document" column climbs unevenly, and the
"neither" column says why: losing faith in the prior happens
*before* learning to copy. At a 75% contradicting mix the
model recites the role signature only 25.00% of the time but
reads the document just 22.22% of the time, answering with
some third word in the remaining 52.78%. There is a stage in
grounding training where the model has stopped trusting what
it knew and has not yet started trusting what it was handed.

And grounding is not free. Follow the closed-book column
down: **0.6111 to 0.0833**. The same training that teaches a
model to defer to documents strips out its ability to answer
without one.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Part 6 ended its complexity table with the row that made this
article inevitable: scoring a query against `N` stored
sentences costs `N` dot products for a bi-encoder and `N`
full forward passes for a cross-encoder. That gap is the
reason the retrieval half of every production system is a
bi-encoder, and the reason anything needing real precision
re-ranks a shortlist with a cross-encoder afterwards.

The row that surprises people is the generator's. Prepending
a document makes the input longer, and attention is quadratic
in that length, so a 500-token passage in front of a 20-token
question costs 676 times the attention of the question alone.
Retrieval is cheap. Reading is not, and the bill arrives on
every single query rather than once at indexing time.

---

## Where RAG sits

The architecture is from 2020 — Lewis et al. named it, and
Karpukhin et al. published Dense Passage Retrieval in the
same year with the in-batch-negatives recipe used here. The
pitch was exactly the one above: move knowledge out of the
weights and into a store you can edit.

The caveat arrived more slowly. Longpre et al. showed in 2021
that when a retrieved passage is edited to disagree with what
a model learned in pre-training, models frequently follow the
memory rather than the passage — and that the rate depends on
how the system was trained, not on how good the retriever is.
This article is that experiment rebuilt small enough that the
prior is a known quantity rather than an inference.

The lineage in this series is direct, and it closes a loop.
Part 6's bi-encoder is the retriever. Part 7's decoder is the
generator. Part 6's artefact lesson — that a benchmark
measures the dataset as much as the model, and the only way
to know which is to run the ablation that removes the input —
reappears here wearing different clothes. There, the
hypothesis-only model exposed what the data leaked. Here, the
contradicting document exposes what the evaluation never
tested. Both are the same discipline: do not measure a system
on examples where the right answer and the easy answer
coincide.

---

## What comes next

Part 9, **Fine-tuning and LoRA**, takes the other route to
the same goal. RAG puts the knowledge beside the model, and
fine-tuning puts it inside. Having just measured what it
costs to
teach a model to defer to a document, the question becomes
what it costs to change the weights instead — and why almost
nobody changes all of them any more.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**rag.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/13-natural-language-processing/08-retrieval-augmented-generation/rag.py)

Run it with:

```bash
pip install numpy
python rag.py
```

It needs only `numpy` and runs in about six minutes.
Everything is from scratch: the bidirectional retriever with
mean pooling, in-batch-negative contrastive training, the
causal generator reading a `[SEP]`-joined document, question
and answer with its loss on the answer span alone, and the
knowledge base whose every fact is auditable against the
oracle that defines the language — all hand-differentiated
and checked against central differences at startup (worst
error 1.26e-09 over 150 sampled gradients, covering both the
generator head and the retriever's path through pooling and
normalisation). The headline insight worth pinning to the
wall: **a retrieval system fine-tuned only on documents that
agree with what its generator already believed scores a
perfect 1.0000 on agreeing facts and 0.0000 on contradicting
ones, which means the benchmark was measuring the prior and
not the pipeline; and the fix is not free, because the same
training that raises grounding from 0.0000 to 0.6667 raises
susceptibility to a wrong document from 0.0000 to 0.8056 —
trusting documents and being fooled by them are one behaviour,
and the retriever decides which one you get**.

---

*This is Part 8 of the Natural Language Processing track in the Algorithms in Python series. The companion script `rag.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It reuses Part 1's synthetic micro-language ([word embeddings](https://medium.com/p/164f7f542a3c), [BPE tokenisation](https://medium.com/p/5e6a51341455), [sequence-to-sequence](https://medium.com/p/bbdf65112819), [BERT-style encoders](https://medium.com/p/a0ba729fe954), [GPT-style decoders](https://medium.com/p/6da33883e00d), [natural language inference](https://medium.com/p/d4f9c54be0d4), [natural language generation](https://medium.com/p/48a45ec5b396)). Part 9 will look at Fine-tuning and LoRA.*
