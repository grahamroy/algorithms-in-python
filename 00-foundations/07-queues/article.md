# Queues — FIFO, Priority, and the One Abstraction Behind BFS, Replay Buffers and Schedulers

### *Algorithms in Python --- Foundations, Part 7*

---

You have already used a queue this week without thinking about it. Every time
you wrote `collections.deque`, every time you fed a batch into a training
loop, every time an async function awaited the next item off a work channel,
every time a background job waited its turn behind another — you used a
queue. It is the plumbing layer of almost every long-running system, and it
is strange that we almost never talk about it directly.

In Part 5 we ran BFS across a graph with a `deque` and moved on. In Part 6
we looked at how `dict` gets you O(1) lookups out of thin air. This article
is about the other half of that same mental toolkit — the data structure
that says *"I don't care which item is smallest or largest or closest to
anything. I care which one arrived first."*

That is a **queue**. And like hash tables, it is an abstraction with two or
three good implementations, each with a very specific shape.

---

## What is a queue?

A queue is an ordered collection of items with two operations: **enqueue**
adds an item to the back, **dequeue** removes an item from the front. First
in, first out — FIFO. If you have ever stood in a supermarket queue, you
already have the correct mental model.

```
 enqueue -->  [ D ][ C ][ B ][ A ]  --> dequeue
                back             front
```

Four items arrived in order A, B, C, D. The next `dequeue` returns A. That
is the entire contract.

Compare that to the **stack** we used for DFS in Part 5 — a LIFO structure
that pops the most recently added item. Same two operations, opposite
discipline. BFS explores neighbours in the order they were discovered
(queue). DFS dives along the most recent frontier (stack). Swap the
underlying data structure and you swap the algorithm's whole personality.

A queue's value is *fairness in time*. The thing that waited longest gets
served next. Every system that needs to be fair over a work stream — a
request handler, a print spooler, a training data pipeline, a replay
buffer, a message bus — ends up with a queue somewhere in it.

---

## Why a Python list is not a queue

At first glance, a Python list already looks like a queue. You can
`append` to the right, `pop(0)` from the left, and the contract is
satisfied. It works. It just does not scale.

```python
q = []
q.append("a")
q.append("b")
q.append("c")
first = q.pop(0)     # 'a'
```

The bug is buried in `pop(0)`. A Python list is a contiguous array of
pointers (see Part 1). When you remove the item at index `0`, every
remaining element has to shift one slot to the left so index `1` becomes
index `0`, index `2` becomes index `1`, and so on. That is O(n). Do it a
million times on a million-item queue and you have written something that
runs in O(n^2).

The companion script puts a number on it. Drain a 100,000-item queue using
`list.pop(0)` against `deque.popleft()`:

```
Draining 100,000 items:
  list.pop(0)      :  8085.88 ms
  deque.popleft()  :     1.72 ms
  deque speedup    :   4700.8x
```

Eight seconds against under two milliseconds. Four thousand times faster.
That is not a micro-optimisation — that is the difference between a
background worker that keeps up with its feed and one that falls behind
forever. If you remember one thing from this article: **never use a list
as a FIFO queue.** Python has a dedicated structure for it and it is called
`deque`.

---

## `collections.deque` — the workhorse

`deque` is short for double-ended queue: a queue that is O(1) at *both*
ends.

```python
from collections import deque

q = deque()
q.append("a")        # O(1) on the right
q.append("b")
q.appendleft("z")    # O(1) on the left
q.pop()              # O(1) on the right -> 'b'
q.popleft()          # O(1) on the left  -> 'z'
```

Used as a FIFO queue you only need two of these — `append` on the right and
`popleft` on the left. Used as a stack you would `append` and `pop`. Used
as a deque proper you use all four. Same object, same O(1) cost.

### How it is built

Internally, `deque` is not a single contiguous array. It is a **doubly
linked list of fixed-size blocks**. Each block is a small array (64 items
in current CPython), and the blocks themselves are linked to their
neighbours with forward and backward pointers. That is a direct callback
to Part 4 — the doubly linked list we built by hand is hiding inside
every `deque` you have ever used, just with blocks of items per node
instead of one.

```
  [block]  <->  [block]  <->  [block]  <->  [block]
   head                                       tail
```

Two consequences follow immediately. First, appending or popping at either
end is O(1) — you only ever touch the head or tail block, and allocate a
fresh block only when one fills up. Second, indexing into the middle is
*not* O(1). `dq[len(dq) // 2]` has to walk from the nearest end block by
block. For FIFO access patterns that does not matter; you almost never
index into a queue.

`deque` is thread-safe for `append` and `popleft` at the individual-call
level, which is why it is the structure `queue.Queue` is built on top of
when you need blocking semantics between threads.

### `maxlen` — the ring-buffer trick

`deque` takes an optional `maxlen` argument. Once the deque is full,
appending on one end silently evicts the opposite end to keep the length
fixed. That is a **ring buffer** in two lines of code.

```python
window = deque(maxlen=5)
for value in stream:
    window.append(value)
```

Running the companion script's second demo over a simulated sensor feed:

```
  step  incoming    window                                   avg
     1     19.62   [19.62]                                    19.62
     2     20.77   [19.62, 20.77]                             20.20
     3     19.66   [19.62, 20.77, 19.66]                      20.02
     4     19.53   [19.62, 20.77, 19.66, 19.53]               19.89
     5     18.60   [19.62, 20.77, 19.66, 19.53, 18.6]         19.64
     6     19.68   [20.77, 19.66, 19.53, 18.6, 19.68]         19.65
     7     21.67   [19.66, 19.53, 18.6, 19.68, 21.67]         19.83
     8     20.64   [19.53, 18.6, 19.68, 21.67, 20.64]         20.02
     9     21.56   [18.6, 19.68, 21.67, 20.64, 21.56]         20.43
    10     20.37   [19.68, 21.67, 20.64, 21.56, 20.37]        20.78
```

The first five readings fill the window. Every reading after that evicts
the oldest one on the left — no bookkeeping, no index arithmetic, no edge
cases at the boundaries. A rolling moving average, a streaming latency
histogram, the last N frames of an environment, a chat context window:
all of them, structurally, are a fixed-length deque with items falling
off the back.

This is worth underlining because so many people reach for indexed lists
and modulo arithmetic to implement ring buffers. Don't. CPython has one
built in.

---

## Priority queues — when "first in" isn't the right discipline

FIFO is the right discipline *most* of the time. But sometimes the order
you want is not arrival order — it is priority order. The job that matters
most should run next, regardless of when it arrived.

That is a **priority queue**, and the idiomatic Python implementation is
`heapq`.

```python
import heapq

heap = []
heapq.heappush(heap, (5, "send newsletter"))
heapq.heappush(heap, (1, "page on-call"))
heapq.heappush(heap, (3, "rebuild embeddings"))
priority, job = heapq.heappop(heap)    # (1, 'page on-call')
```

`heapq` is not a class. It is a module of functions that treat a plain
Python list as a **binary min-heap**. `heappush` and `heappop` are both
O(log n). You can peek at the smallest element in O(1) with `heap[0]`.

### The heap invariant

A binary heap is a binary tree with one rule: every parent is smaller than
or equal to its children. That is the whole thing. It does not have to be
sorted — siblings can appear in any order — only the parent-child
relationship has to hold.

```
           1
         /   \
        3     2
       / \   / \
      5   4 7   6
```

The smallest element is always at the root. `heappop` returns it, then
patches up the tree by moving the last element to the root and "sifting
it down" along the smaller-child path until the invariant is restored.
`heappush` appends to the end and "sifts up" until the invariant is
restored. Both walks are at most the height of the tree — `log n` for a
balanced binary tree — and the tree stays balanced automatically because
we fill it level by level.

The lovely CPython detail: the tree is not a tree at all. It is stored as
a plain flat list where the children of index `i` live at indices `2i+1`
and `2i+2`, and the parent of index `i` lives at `(i - 1) // 2`. No
pointer chasing, no node allocations, no balancing rotations. Excellent
cache locality. The "tree" is purely a way of reading the array.

This is a Part 8 preview. A heap is literally a **tree hiding inside an
array**. We will look at tree structure properly in the next article, but
the first tree you use is usually a heap.

### Ties and stable ordering

Heaps compare elements with `<`. If you push tuples, Python compares
element-wise: first the priority, then the next field, then the next. So
the standard pattern is:

```python
heapq.heappush(heap, (priority, job_id, payload))
```

`job_id` breaks ties when two jobs have the same priority, producing stable
ordering by arrival. Without it, Python would try to compare the payloads
themselves — which fails with `TypeError` the moment you push two
non-comparable objects with the same priority.

### Min-heap, max-heap

`heapq` is a min-heap. If you want the largest element first, negate the
priority on push and negate again on pop:

```python
heapq.heappush(heap, (-score, model_name))
neg_score, name = heapq.heappop(heap)
score = -neg_score
```

The companion script's third demo uses both. First a job scheduler: six
jobs go in with mixed priorities, and they come out in priority order with
ties broken by ID:

```
Popping in priority order (ties broken by job_id):
  priority=1  job_id=2  -> page on-call: DB master down
  priority=1  job_id=6  -> page on-call: 500s spiking
  priority=2  job_id=4  -> rotate TLS certificate
  priority=3  job_id=3  -> rebuild nightly embeddings
  priority=4  job_id=5  -> compress last week's logs
  priority=5  job_id=1  -> send weekly newsletter
```

Then a max-heap via negation, popping four models in descending score
order. Same data structure, opposite discipline.

---

## Big-O summary

[[BIG-O TABLE IMAGE]]

`deque` is O(1) on both ends and O(n) for anything in the middle.
`heapq` is O(log n) for push and pop, O(1) for peeking at the minimum,
and O(n) to build a heap from an existing list via `heapify` — faster
than pushing items one at a time, which would be O(n log n).

The worst case for a list-as-queue is the O(n) shift that made us reach
for `deque` in the first place. The worst case for `heapq` is essentially
never hit because the heap invariant is maintained on every operation —
unlike a hash table, there is no adversarial input that degrades it.

---

## Real-world ML and AI connections

Queues are everywhere in ML infrastructure once you start looking.

**BFS and its cousins.** We already saw this in Part 5. `collections.deque`
is what makes BFS run in O(V + E) instead of O(V^2). Replace the deque
with a list and `popleft` becomes O(V) and the whole algorithm becomes
O(V^2 + E). The choice of queue implementation is the difference between
a traversal that finishes on a web-scale graph and one that does not.

**Dijkstra and A\*.** The priority-queue generalisation of BFS. Part 5's
Dijkstra implementation reached for `heapq` without ceremony; now you
know why. The frontier is not "everything we discovered in order" — it
is "the unvisited vertex with the smallest tentative distance". A binary
heap delivers that in O(log V) per operation, and the whole routing
backbone of the internet runs on something topologically equivalent to
that code.

**Experience replay in reinforcement learning.** Deep Q-Networks and most
modern off-policy RL agents keep a **replay buffer** — the last N
environment transitions `(state, action, reward, next_state)`. During
training the agent samples minibatches from the buffer instead of training
only on the most recent transition. The buffer is, structurally, a
`deque(maxlen=N)`: the oldest experiences silently fall off the back as
new ones arrive. **Prioritised experience replay** (Schaul et al., 2016)
goes one step further and samples transitions with probability proportional
to their TD error, which requires a data structure that supports weighted
sampling — typically a **sum-tree**, which is yet another heap-shaped array.

**Message passing in GNN training.** Part 5's graph neural networks
propagate information from node to neighbour in layers. At serving time
on streaming graphs, the training loop often uses a queue of nodes whose
features need recomputing when their neighbours change. Framework
schedulers like PyTorch Geometric's `NeighborLoader` walk out from a seed
set with a BFS-like frontier, and the frontier is — of course — a deque.

**Job scheduling and asyncio.** Every async Python program is running on
top of an event loop whose core data structure is a pair of queues: a
FIFO for ready tasks and a priority queue (heap) keyed on scheduled time
for timers and sleeps. `asyncio.Queue` itself is a `deque`-backed
structure with an async waiting protocol on top. Celery, RQ, Kafka
consumer groups, Temporal — every job queue in your stack is the same
abstraction wearing different clothes.

**Training data pipelines.** `torch.utils.data.DataLoader` runs its worker
processes on top of a queue. Workers pull batches from a dataset and push
them to a shared queue; the main process pulls from that queue as the
training loop asks for the next batch. The queue's depth is the size of
the prefetch window — enough depth to hide disk latency, not so much that
you run out of RAM. The `DataLoader` is a producer-consumer pattern with
a bounded queue in the middle, almost verbatim the example in every
concurrency textbook.

**Beam search.** Sequence generation in language models often uses
beam search: keep the top-k most probable partial sequences at each step,
extend each by every possible next token, score the results, and keep the
top-k again. The "top-k" is a priority queue — usually a heap of size k
with a small `nsmallest` / `nlargest` wrapper. When you tune
`num_beams=5` in a HuggingFace generation call, you are sizing a heap.

**Token rate limiting and request shaping.** Serving an LLM at scale
means controlling which requests go through when. Token buckets, leaky
buckets, and weighted fair-queueing schedulers all sit on priority queues
keyed by earliest-deliverable-time. Your OpenAI or Anthropic API rate
limit is enforced by a stack of queues somewhere.

---

## When NOT to use a queue

Queues are excellent, but they are not the answer to everything.

**When you need random access.** A queue hides everything but its ends.
If you need to reach into the middle of the collection, use a list or
array. `deque[i]` does exist but it is O(n) in the general case.

**When order does not matter at all.** If you are collecting items you
will process in arbitrary order later, a `set` or a plain `list` is
simpler and usually faster. Do not pay for FIFO discipline you do not
need.

**When you need a sorted view, not just the smallest.** A heap gives you
the minimum cheaply, but iterating a heap in sorted order costs O(n log n)
— you essentially have to pop everything. For "all items in sorted order"
you want a sorted list or a balanced BST.

**When you need to update priorities.** A classic Dijkstra
implementation pushes duplicate entries onto the heap rather than
updating existing ones, because `heapq` does not support decrease-key.
For a few stale entries that is fine; for a workload where priorities
change constantly, look at an indexed priority queue or a structure like
a Fibonacci heap. In practice almost nobody builds one.

**When multiple producers and consumers need coordination.** `deque` and
`heapq` are bare data structures. For thread-safe or async-safe
queueing with blocking `put` and `get`, reach for `queue.Queue`,
`asyncio.Queue`, or `multiprocessing.Queue`. They are built on top of
the same primitives with synchronisation on top.

---

## What comes next

Seven data structures down, five to go. Part 8 is **Trees**.

We have already met two trees in disguise in this article. A binary heap
is a complete binary tree stored as an array. A `deque`'s internal
structure is a linked list of blocks, which is nearly a tree at its
degenerate minimum branching factor. Part 8 gets us to honest
hierarchical structure: decision trees and random forests (the everyday
workhorses of tabular ML), KD-trees and ball trees (nearest-neighbour
search), B-trees (the data structure your database's index is actually
built from), and the tree traversals that bring the BFS/DFS story full
circle.

And after Part 8 — knowledge graphs, the ninth of our twelve
foundations. Then probabilistic data structures, sparse matrices, and
vector indexes complete the data-structure layer before we start
building the algorithms that run on top.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**queues.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/07-queues/queues.py)

Run it with:

```bash
python queues.py
```

It finishes in a few seconds on a laptop — the deliberately slow
`list.pop(0)` benchmark dominates the runtime. The headline line —
`deque.popleft()` running several thousand times faster than `list.pop(0)`
on the same workload — is the sort of number you want pinned to the wall
next to your desk.

---

*This is Part 7 of the Algorithms in Python series, Foundations track. The companion script `queues.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 6](https://medium.com/@grahamjroy) covered hash tables. Part 8 will look at trees — the hierarchical structure that shows up in decision forests, nearest-neighbour search, and every database index on earth.*
