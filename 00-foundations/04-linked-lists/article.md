# Linked Lists — When Arrays Aren't Enough

### *Algorithms in Python --- Foundations, Part 4*

---

In the first three parts of this series we stored data in contiguous memory. An array laid
numbers end to end. A matrix stacked rows of numbers into a table. A tensor extended that
table into three, four, five dimensions. Every element lived right next to its neighbour in
one unbroken block of memory, and that arrangement made random access instant and
vectorised arithmetic blazing fast.

Now we break that assumption.

A linked list stores each element in its own independent object --- a **node** --- and
connects the nodes with pointers. Nothing is contiguous. Nothing is pre-allocated. You
cannot jump to position 500 without walking past the first 499 nodes. On paper, this
sounds like a step backwards. In practice, it unlocks an entirely different set of
trade-offs --- and those trade-offs matter the moment your problem involves frequent
insertion, flexible sizing, or structures more complex than a flat grid.

This article introduces the linked list, shows when it beats arrays, when it loses, and
why it keeps showing up in the architecture of modern AI systems.

---

## What is a linked list?

A linked list is a sequence of **nodes**. Each node holds two things: a piece of data and
a reference (pointer) to the next node. The last node points to nothing --- `None` in
Python.

```
 [10 | *]---> [20 | *]---> [30 | *]---> [40 | *]---> None
```

There is no index. There is no contiguous block. To reach the third element, you start at
the head and follow two pointers. This is fundamentally different from an array, where
position 2 is a single arithmetic offset from the start of memory.

Think of it like a scavenger hunt. An array is a bookshelf --- every book is right where
you expect it, and you can grab the seventh one without touching the first six. A linked
list is a treasure hunt where each clue tells you where the next one is. You have to
follow the chain. The upside is that adding a new clue at the start, or removing one from
the middle, is trivially easy --- you just rewrite a couple of notes. On a bookshelf, you
would have to slide every book over to make room.

In Python, building this from scratch takes two classes:

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
```

That is the entire structure. Everything else --- append, delete, search, reverse --- is
just pointer manipulation on top of these two pieces.

---

## Singly vs doubly linked lists

The structure above is a **singly linked list**: each node points forward to the next, and
that is it. You can traverse in one direction only --- head to tail.

A **doubly linked list** adds a second pointer. Each node knows both its successor *and*
its predecessor:

```
 None <---[* | A | *]<--->[* | B | *]<--->[* | C | *]---> None
```

The extra pointer costs more memory per node, but it buys you two important things.
First, you can traverse backwards. Second, and more critically, if you already hold a
reference to a node, you can remove it in O(1) --- you just rewire the neighbours. In a
singly linked list, deletion requires walking from the head to find the previous node,
which takes O(n).

This distinction is not academic. The doubly linked list is the backbone of structures
like the **LRU cache**, where you need to move and remove arbitrary nodes at constant
cost.

---

## Core operations and their complexity

Every data structure earns its place through the speed of its operations. Here is how a
singly linked list compares to an array:

| Operation | Singly linked list | Array (Python list) |
|---|---|---|
| Access by index | O(n) | O(1) |
| Prepend (insert at head) | O(1) | O(n) |
| Append (insert at tail) | O(n)* | Amortised O(1) |
| Insert at arbitrary position | O(n) walk + O(1) splice | O(n) shift |
| Delete by value | O(n) | O(n) |
| Search | O(n) | O(n) |

*\*With a tail pointer, append becomes O(1) as well.*

The standout difference is **prepend**. Inserting at the front of a linked list takes
constant time --- you create a node, point it at the current head, and update the head
pointer. Inserting at the front of a Python list takes O(n) because every existing element
must shift one position to the right.

### Prepend

```python
def prepend(self, data):
    new_node = Node(data)
    new_node.next = self.head
    self.head = new_node
    self.size += 1
```

Three assignments. No loops. No shifting. This is O(1) whether the list has five elements
or five million. Compare this to `list.insert(0, x)` in Python, which must copy every
element one position to the right before it can place the new value at index zero. On a
list with a million elements, that is a million memory moves --- every single time.

### Append

Appending to the tail of a singly linked list requires walking the entire chain to find
the last node:

```python
def append(self, data):
    new_node = Node(data)
    if self.head is None:
        self.head = new_node
    else:
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    self.size += 1
```

This is O(n) because of the walk. You can bring it down to O(1) by maintaining a `tail`
pointer alongside `head`, so you always know where the end is. Python's built-in list
already does this --- `list.append()` is amortised O(1). The point is not that linked
lists append faster (they do not, unless you keep a tail pointer), but that *prepend* is
where the linked list has its structural advantage.

### Delete

Deleting a value requires two steps: find the node, then rewire the pointers around it.

```python
def delete(self, data):
    if self.head.data == data:
        self.head = self.head.next
        self.size -= 1
        return
    current = self.head
    while current.next:
        if current.next.data == data:
            current.next = current.next.next
            self.size -= 1
            return
        current = current.next
```

The search is O(n), but the actual removal --- updating one pointer --- is O(1). In an
array, after finding the element you must also shift every subsequent element one position
to the left, adding another O(n) cost on top of the search. For workloads with frequent
deletions, this distinction matters.

### Search

```python
def search(self, data):
    current = self.head
    while current:
        if current.data == data:
            return True
        current = current.next
    return False
```

You start at the head and walk forward, checking each node. Worst case you visit every
node --- O(n). There is no shortcut. Arrays have the same O(n) search cost for unsorted
data, but they benefit from cache locality (more on that shortly), which makes the
constant factor smaller in practice.

### Reverse

Reversing a linked list in place is a classic operation --- and a common interview
question. The idea is simple: walk through the list, flipping each pointer to face the
other direction.

```python
def reverse(self):
    previous = None
    current = self.head
    while current:
        next_node = current.next
        current.next = previous
        previous = current
        current = next_node
    self.head = previous
```

Three pointer swaps per node. One pass through the list. O(n) total. No extra memory
beyond three temporary variables.

---

## When linked lists beat arrays

**Frequent insertion and deletion at the front.** If your workload involves repeatedly
adding or removing elements at the beginning of a sequence --- a task queue, a stream of
events, an undo history --- linked lists handle it in O(1). A Python list pays O(n) every
time.

**Unknown or highly variable size.** Arrays over-allocate to stay fast, but if the size
swings wildly, you waste memory or pay for repeated resizing. A linked list allocates
exactly one node per element, no more.

**Building more complex structures.** Trees, graphs, adjacency lists --- all of these are
built from nodes and pointers. Once you understand a linked list, you understand the
building block that underpins every non-linear data structure.

**No wasted capacity.** A Python list typically over-allocates by 12--25% to keep appends
fast. For small lists this is negligible, but if you are managing millions of short,
independent sequences --- say, the adjacency lists of a large graph --- that overhead adds
up. Each linked list node allocates exactly what it needs and nothing more.

---

## When arrays beat linked lists

**Random access.** If you need to read position *k* frequently, arrays win absolutely.
O(1) versus O(n) is not a close contest.

**Cache locality.** Array elements sit in contiguous memory. When the CPU loads one
element into cache, the neighbouring elements come along for free. Linked list nodes are
scattered across the heap, so every pointer hop is a potential cache miss. In practice,
this means an O(n) scan through an array is often *faster* than an O(n) scan through a
linked list of the same length, because the hardware is optimised for sequential memory
access.

**Vectorised operations.** NumPy, PyTorch, and every numerical library on earth depend on
contiguous memory to run parallel arithmetic. You cannot vectorise a linked list. If your
workload is batch matrix multiplication or element-wise transformations, arrays (and their
higher-dimensional generalisations) are the only sensible choice.

**Memory overhead per element.** Each linked list node carries the weight of at least one
pointer (8 bytes on a 64-bit system) in addition to the data itself. For a doubly linked
list, that doubles to two pointers. When your data is small --- say, a list of integers
--- the pointer overhead can exceed the data size. Arrays store raw values with zero
per-element bookkeeping, which is why they dominate when storage density matters.

**Sorting.** Sorting an array benefits from random access and cache-friendly memory. Merge
sort works well on linked lists (it is inherently sequential), but quicksort and other
partition-based algorithms suffer without O(1) indexing. In practice, if you need sorted
data and fast lookups, a sorted array or a balanced tree will outperform a linked list.

---

## AI and ML relevance

Linked lists rarely appear *directly* in a training loop. You will not pass a linked list
into `model.fit()`. But they are everywhere in the infrastructure that makes AI systems
work.

**Graphs and trees.** A graph is typically represented as an adjacency list --- an array
of linked lists, one per vertex. Graph neural networks (GNNs), knowledge graphs, and
dependency parsers all operate on this representation. Understanding linked lists is the
prerequisite for understanding graphs.

**Dynamic computation graphs.** When PyTorch executes a forward pass, it builds a
computation graph on the fly --- each operation creates a node, and edges link operations
to their inputs. This graph is not stored in a matrix. It is a dynamically constructed
structure of nodes and pointers, assembled as code runs and torn down after
backpropagation. The design philosophy is the same one that makes linked lists powerful:
allocate only what you need, wire it together with references, and restructure freely.

**Memory management.** The free lists used by memory allocators --- including Python's own
object allocator --- are singly linked lists. When you call `del` on a Python object, it
goes onto a free list. When you create a new object, the allocator pops a block off that
list. This is O(1) allocation, and it is linked list mechanics.

**Streaming data.** When data arrives continuously --- sensor readings, log events, market
ticks --- a linked list (or its close relative, the deque) lets you ingest at the front
and process at the back without resizing. Python's `collections.deque` is implemented as a
doubly linked list of fixed-size blocks, giving you O(1) operations at both ends.

**Attention mechanisms and sparse structures.** Transformer models compute attention over
sequences of tokens. While the standard implementation uses dense matrices, research into
sparse attention (Longformer, BigBird) represents the attention pattern as a graph ---
each token attends only to a subset of other tokens, and those connections are stored as
adjacency lists. At the implementation level, these are arrays of linked structures. The
shift from dense to sparse attention is, at its core, a shift from matrices to
pointer-based representations.

**Garbage collection.** CPython uses reference counting as its primary memory management
strategy, supplemented by a cycle-detecting garbage collector. The collector maintains
linked lists of tracked objects, grouped by generation. When it runs, it walks these lists
to identify and collect unreachable cycles. Every time you train a model in Python, the
garbage collector is quietly traversing linked lists in the background to keep memory
under control.

---

## Practical examples from the companion code

The Python script that accompanies this article demonstrates two practical applications.

### LRU cache

A Least Recently Used cache evicts the oldest unused entry when it runs out of space. The
classic implementation pairs a **doubly linked list** with a **hash map**: the list
maintains access order (most recent at the tail, least recent at the head), and the map
provides O(1) lookup by key. Every `get` and `put` is O(1).

This pattern appears in web servers, database query caches, and ML model serving. If your
inference server caches the last *k* tokenised prompts to avoid re-tokenising duplicates,
you are using an LRU cache.

### Undo stack

An undo mechanism is a natural fit for a linked list used as a stack. Each action is
prepended to the head. Undo pops from the head in O(1). There is no shifting, no
resizing --- just pointer updates. The companion script builds a small text editor undo
system where each action ("Type 'Hello'", "Bold text") is pushed onto the stack, and
calling undo peels back the most recent operation instantly.

You could use a Python list for this --- and in many applications, you should. But the
linked list version makes the mechanics explicit: there is a head, there is a chain of
previous actions, and undo is just moving the head one node forward. When you move to
more complex undo systems --- branching undo trees, for instance --- the pointer-based
approach generalises more naturally.

### Performance comparison

The script benchmarks 1,000 head insertions on collections of 10,000 to 100,000 elements.
The linked list completes in near-constant time regardless of collection size, while the
Python list slows linearly as it must shift every element on each insertion.

---

## What comes next

The linked list is the first data structure in this series that breaks away from
contiguous memory. Each node lives independently, connected only by a reference to its
neighbour. That idea --- independent entities connected by references --- is the
foundation of every non-linear structure in computer science.

In Part 5, we take the next step: **graphs**. A graph is a collection of nodes connected
by edges, and the most common way to store it is an array of linked lists. If the linked
list is a chain, a graph is a web --- and it is the data structure that powers social
networks, recommendation engines, knowledge bases, and an entire family of neural network
architectures.

---

## The complete code

The full script is on GitHub --- grab it here and run it yourself:

[**linked_lists.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/04-linked-lists/linked_lists.py)

Run it with:

```bash
python linked_lists.py
```

---

*This is Part 4 of the series "Algorithms in Python". You can find the full series and
source code at [github.com/grahamroy/algorithms-in-python](https://github.com/grahamroy/algorithms-in-python).*
