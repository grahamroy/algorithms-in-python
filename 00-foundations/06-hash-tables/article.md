# Hash Tables — How Python Finds Things in Constant Time

### *Algorithms in Python --- Foundations, Part 6*

---

Open any Python file you have written this week. Count the `{}` literals, the
`dict(...)` calls, the `set()` constructors, the `**kwargs`, the config
objects, the model state dicts, the `lru_cache` decorators, the word-to-index
lookups, the deduplication passes. Every single one is the same data
structure, used over and over, and almost none of us ever think about how it
works.

In the first five parts of this series we looked at data structures that
organise their contents spatially: arrays in contiguous memory, linked lists
chained with pointers, graphs pointing at whatever they like. This article is
about a data structure that does something stranger. Instead of searching for
where a key is stored, a **hash table** *computes* the location directly from
the key. It is the answer to *"can we look things up without actually
looking?"*, and the answer turns out to be yes, on average, in constant time.

This is what makes Python feel fast. Every `d[key]` access is a hash table
lookup. Every `set` membership test is a hash table lookup. Every attribute
access on an object is *also* a hash table lookup, because the object's
`__dict__` is a hash table. You are using them hundreds of times a second.

---

## The lookup problem

Suppose you have ten million records keyed by user ID and you want the record
for `"user_42991"`. How long does it take?

If the records sit in an unsorted array, you walk the whole thing until you
find a match — **O(n)**. If the array is sorted, you can do binary search
instead, finishing in about 24 comparisons — **O(log n)** — but you pay to
keep it sorted on every insert.

A hash table takes a completely different route. Instead of searching for the
key, it *computes* the key's address. You hand a hash function the string
`"user_42991"` and it hands back an integer, which you reduce into a slot
index in a fixed-size array. You go straight to that slot. No scanning, no
tree walks, **O(1)** average case.

The idea in one sentence: **compute the array index from the key directly,
instead of searching for it.** Everything else in this article is about
realising that idea in the face of collisions and growth.

---

## Hash functions

A **hash function** takes an input of any size and returns a fixed-size
integer. A good one has three properties.

**Deterministic.** Same input, same output during a run. Otherwise you could
store things but never retrieve them.

**Uniform.** Outputs spread evenly across the integer range. A hash function
that dumps half its inputs into the same bucket has given up on its own job.

**Fast.** You pay the hash cost on every insert, lookup and delete. A
cryptographic hash like SHA-256 is uniform and deterministic and also
completely unusable for a hash table — thousands of times too slow. Hash
tables use simple, fast, non-cryptographic mixers.

A subtler property is the **avalanche effect**: flipping one bit of the input
should change roughly half the bits of the output. Without it, similar keys
(`"user_1"`, `"user_2"`, `"user_3"`) pile into adjacent slots and lookups
slow down. Python's `hash()` demonstrates the effect nicely:

```
One-character change, completely different hash:
  hash(  'password') = -912499362905713490
  hash(  'Password') = 4071344120751255091
  hash(  'passw0rd') = 1284991106330832461
  hash( 'password!') = 6244609580141321781
```

Four strings that differ in one character each, four completely unrelated
integers. That is what you want.

### Python's `hash()` built-in

Every hashable object in Python has a `__hash__` method, and `hash(x)` calls
it. The rules are worth memorising:

- `hash(n)` for a small integer `n` is just `n`. Yes, really.
- `hash(0) == 0`. `hash(42) == 42`. `hash(-1) == -2` (a historical CPython
  quirk — `-1` is reserved as an error sentinel internally).
- `hash("hello")` gives a big signed integer that changes every time you
  restart Python.
- `hash((1, 2, 3))` combines the hashes of the tuple's elements.
- `hash(3.14)`, `hash(True)`, `hash(None)` all work.

Here is the output from the companion script:

```
hash(              0) = 0
hash(              1) = 1
hash(             42) = 42
hash(             -1) = -2
hash(1152921504606846976) = 1152921504606846976
hash(        'hello') = -8477081617862038534
hash(        'world') = -4552163429076350643
hash(       'hello!') = -739901268400795190
hash(      (1, 2, 3)) = 529344067295497451
hash(      (1, 2, 4)) = -4363729961677198915
hash(           3.14) = 322818021289917443
hash(           True) = 1
hash(           None) = 4238894112
```

Run it yourself and the string hashes will be different. Integer, tuple and
float hashes will be the same.

### Why string hashes are randomised

Until Python 3.2, the hash of a given string was the same in every process,
forever. That let attackers mount a **hash DoS** attack: submit thousands of
request parameters whose keys all hashed to the same bucket of the web
framework's dict, forcing O(n^2) work on a single malicious request. A few
hundred kilobytes of crafted input could tie up a server.

From Python 3.3 onwards, every process picks a random seed at startup
(`PYTHONHASHSEED`, pinnable via the environment variable of the same name).
The seed is mixed into string and bytes hashing, so two processes see two
completely different distributions of keys across buckets. An attacker who
cannot see your seed cannot craft a collision attack against you. Integer,
float and tuple hashes are not randomised because the attack only really
worked through web-facing string handling. That is why `hash("hello")` keeps
changing on you.

### Bad hash functions

It is worth seeing what a bad hash function does. Two archetypes:

```python
def hash1(key):
    return 0           # everything lands in the same bucket

def hash2(key):
    return len(key)    # "cat" and "dog" collide, all 3-letter words collide
```

`hash1` turns the hash table into a single long list. Every lookup walks from
the start — **O(n)**, strictly worse than an unsorted array because you also
pay for a pointless hash. `hash2` is less obviously broken but still
catastrophic: real string data clusters heavily by length, so you get a
handful of enormous buckets and everything else empty. The companion script's
collision stress test shows the first one in action.

---

## Collisions

Even with a perfect hash function, two different keys will eventually hash to
the same slot — the **pigeonhole principle**. If your table has 16 slots and
you insert more than 16 keys, collisions are guaranteed. The question is how
you handle them. There are two dominant strategies.

### Separate chaining

In **separate chaining**, each slot holds a *chain* — a linked list or a
dynamic array — of every entry that hashes there. Insert appends to the chain.
Lookup hashes the key, walks the chain, and compares keys one by one. If your
hash function is uniform and your load factor is reasonable, chains stay
very short and each lookup is still O(1) on average.

### Open addressing

In **open addressing**, every entry lives directly in a slot of the main
array. If slot `i` is already full when you try to insert, you **probe** for
the next empty slot: `i+1`, `i+2`, and so on (linear probing), or `i + k^2`
(quadratic probing), or a position dictated by a second hash function (double
hashing). Lookups follow the same probe sequence until they hit the key or an
empty slot.

CPython's `dict` and `set` both use open addressing with a clever
*perturbation* sequence that mixes in high bits of the hash on each probe,
avoiding the cache-unfriendly clustering linear probing suffers from. You
get great cache locality; deletion is trickier (you cannot simply empty a
slot, or lookups would terminate early — you need tombstones).

### A tiny picture

Here are eight slots, with three keys that all hash to slot 3. With separate
chaining, slot 3 holds a list of three entries and all other slots are
untouched:

```
slot:   0   1   2      3        4   5   6   7
       [ ] [ ] [ ]  [A->B->C]  [ ] [ ] [ ] [ ]
```

With linear probing in open addressing, A lands in slot 3, B bumps to slot 4,
C to slot 5:

```
slot:   0   1   2   3   4   5   6   7
       [ ] [ ] [ ] [A] [B] [C] [ ] [ ]
```

Both are O(1) on average. The companion script uses separate chaining because
it is simpler to write, simpler to read, and simpler to reason about.

---

## Load factor and resizing

The quality of a hash table depends on how full it is. Define the **load
factor** as

```
    alpha = n / m
```

where `n` is entries and `m` is slots. When alpha is small, collisions are
rare. As alpha approaches 1, chains get long (or probe sequences get long)
and supposedly constant-time operations start looking unmistakably linear.

Real hash tables prevent this with **automatic resizing**. When the load
factor crosses a threshold, allocate a new table roughly twice as big and
**rehash** every entry into the new slots. CPython's `dict` resizes at about
2/3. Our companion `HashTable` resizes at 0.75. The slot index for a given
key usually changes in the resized table, so there is no shortcut around
rebuilding it.

Resizing is O(n), but it happens rarely. Between resizes you insert `n`
entries at O(1) each, then pay one O(n) cost to double the table, then insert
another `n`, and so on. Summed across operations, total cost is linear in the
number of inserts, so the **amortised** cost per insert stays O(1). Same
story as Python `list.append` from Part 1: occasional expensive reallocation,
cheap on average. It works because each entry is only rehashed on the resize
that moves it, not on every subsequent one.

---

## Big-O summary

Here is the full table for a well-implemented hash table:

| Operation   | Average        | Worst case |
|-------------|----------------|------------|
| Lookup      | O(1)           | O(n)       |
| Insert      | O(1) amortised | O(n)       |
| Delete      | O(1)           | O(n)       |
| Iterate all | O(n + m)       | O(n + m)   |

The worst case kicks in only when every key lands in the same bucket —
something a decent hash function plus `PYTHONHASHSEED` randomisation makes
practically impossible in the absence of an adversary. Iteration is `O(n +
m)` rather than `O(n)` because you walk all `m` slots (most of them empty) to
find the `n` occupied ones.

One more fact worth stating clearly. **Since Python 3.7, `dict` preserves
insertion order as a language guarantee.** CPython became order-preserving in
3.6 as an implementation detail, and in 3.7 the language spec was amended to
require it. The machinery is a small extra index layer on top of the hash
table and costs almost nothing — ordered iteration essentially for free.

---

## Building one from scratch

The companion code for this article implements a hash table in about a hundred
lines, using separate chaining, automatic resizing, and Python's built-in
`hash()` for the hash function. Here is the core of it:

```python
class HashTable:
    INITIAL_SIZE = 8
    RESIZE_THRESHOLD = 0.75
    GROWTH_FACTOR = 2

    def __init__(self):
        self._size = self.INITIAL_SIZE
        self._count = 0
        self._slots = [[] for _ in range(self._size)]

    def _hash(self, key):
        return abs(hash(key)) % self._size

    def put(self, key, value):
        index = self._hash(key)
        bucket = self._slots[index]
        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._count += 1
        if self.load_factor() > self.RESIZE_THRESHOLD:
            self._resize(self._size * self.GROWTH_FACTOR)

    def get(self, key):
        index = self._hash(key)
        for existing_key, value in self._slots[index]:
            if existing_key == key:
                return value
        raise KeyError(key)

    def delete(self, key):
        index = self._hash(key)
        bucket = self._slots[index]
        for i, (existing_key, value) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                self._count -= 1
                return value
        raise KeyError(key)
```

Each slot is a Python list of `(key, value)` tuples. `put` hashes the key,
walks the bucket to check whether the key already exists (update) or not
(append). `get` and `delete` walk the same bucket. `_resize` allocates a
bigger array and reinserts everything at its new index.

Running the script produces output like this (yours will differ slightly
because of hash seed randomisation and the resulting bucket assignments):

```
================================================================
DEMO 1 — Basic operations and automatic resize
================================================================

Initial state: HashTable(size=8, count=0, load_factor=0.000)

Inserting 20 key-value pairs...

    [Resizing from 8 to 16 slots, load factor was 0.875]
    [Resizing from 16 to 32 slots, load factor was 0.812]

Final state: HashTable(size=32, count=20, load_factor=0.625)

Bucket occupancy (zero = empty slot):
  [1, 0, 0, 1, 0, 2, 0, 0, 1, 0, 1, 0, 0, 2, 0, 2, 0, 0, 2, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 2]

Retrievals:
  get('alpha') -> 1
  get('eta') -> 7
  get(404) -> 'four-oh-four'
  get(1010) -> 'ten-ten'

Membership tests:
  'alpha' in table -> True
  'missing' in table -> False
  404 in table -> True
  999 in table -> False

Deleting 'gamma'...
  delete('gamma') returned 3
  State after delete: HashTable(size=32, count=19, load_factor=0.594)
  'gamma' in table -> False

Updating 'alpha' from 1 to 99...
  State after update: HashTable(size=32, count=19, load_factor=0.594)
  get('alpha') -> 99
```

Notice the two resize events. The table starts with eight slots; the 7th
entry pushes it past 0.75 (`7/8 = 0.875`) and it doubles to 16. More inserts
cross the threshold again at `13/16 = 0.812` and it doubles to 32. Twenty
entries into 32 slots is a comfortable final load factor of 0.625. The
bucket-occupancy printout confirms a reasonable distribution — most buckets
hold zero or one entries, a few hold two. That is separate chaining doing
its job with a decent hash function.

### The collision stress test

The second demo reaches for the worst-case hammer. It subclasses
`HashTable` with a deliberately awful `_hash` that returns 0 for every key,
then inserts 2000 keys into both tables and times a lookup sweep:

```python
class BadHashTable(HashTable):
    def _hash(self, key):
        return 0
```

```
================================================================
DEMO 2 — Collision stress test: good hash vs bad hash
================================================================

Inserted 2000 keys into each table.

Good hash  -> HashTable(size=4096, count=2000, load_factor=0.488)
            lookup time for 2000 keys: 0.24 ms

Bad hash   -> HashTable(size=4096, count=2000, load_factor=0.488)
            lookup time for 2000 keys: 30.84 ms

Bad hash is 127x slower than the good hash.

Every key landed in the same bucket, so every lookup scans
a list of length n. That is the worst case a hash table has.

Good-hash bucket distribution:
  total slots      : 4096
  occupied slots   : 1559
  max bucket size  : 4
  mean bucket size : 0.49
```

Same data, same resize thresholds, same table size. The good hash takes
about a quarter of a millisecond. The bad hash takes thirty — two orders of
magnitude slower — because every lookup linearly scans a bucket holding all
2000 entries. Mean bucket size with the good hash is 0.49; the maximum here
was 4 (string hashes are salted per process, so the exact occupied-slot and
max-bucket figures vary slightly from run to run).
That gap — between a uniform distribution and everything piled in one corner
— is exactly what the PYTHONHASHSEED was introduced to protect against. A
hash DoS attack does not *quite* turn your dict into the bad hash table, but
it gets close enough that your server falls over.

---

## Real-world ML and AI connections

Dictionaries are the duct tape of machine learning code. Once you notice them
you can't stop noticing them.

**Feature hashing, also known as the hashing trick.** You have a categorical
feature with millions of possible values — user IDs, URLs, n-grams — and you
want to turn them into fixed-length vectors without building a vocabulary.
Feature hashing hashes each feature string, takes the result modulo a fixed
dimension `d`, and uses that as the index in a `d`-dimensional vector.
Scikit-learn's `HashingVectorizer` does this for text: a streaming,
memory-bounded featuriser that never has to see the full corpus. The price
is collisions — two features landing on the same index are added together —
but if `d` is large enough the resulting noise barely costs a decimal point
of accuracy.

**Caching and memoisation.** `functools.lru_cache` is a hash table of
arguments to return values with LRU eviction bolted on. Every memoised
dynamic programming solution uses one. Every cached LLM call, every cached
embedding lookup, every cached database query in your inference service. The
difference between memoising and not memoising is the difference between
O(2^n) and O(n) for classic Fibonacci and for thousands of less classic
recursions.

**Deduplication at scale.** Training data dedup is entirely hash-based.
Exact-match dedup is a `set` of content hashes: stream files in, hash
contents, skip repeats. For near-duplicate detection, **locality-sensitive
hashing (LSH)** — a cousin of ordinary hashing that deliberately puts
*similar* items in the same bucket — is the backbone. MinHash + LSH is the
technique that cleans the C4 and Common Crawl datasets modern LLMs train on.
Still a hash table; the hash function just does something more interesting
than mixing bits.

**Count-Min Sketch.** One of the most elegant data structures in computing,
and it is nothing but a stack of hash tables. You have a stream of events
and you want to know, for any event, roughly how many times you have seen
it. You cannot afford a counter per distinct event (billions of them).
Instead keep a small 2D table of counters, hash each event with several
different hash functions, and increment one counter per hash function. To
query, take the *minimum* of the counters — collisions can only inflate
counters, never decrease them. Memory is O(width x depth) independent of the
number of distinct items. It shows up in streaming analytics, word-frequency
estimation, and anywhere you need approximate counts on unbounded streams.

**Set operations for data leakage checks.** One of the most important sanity
checks in ML is ensuring your test set is not polluted by training examples:

```python
leakage = len(set(train_ids) & set(test_ids))
```

That intersection is O(n + m). The naive nested-loop alternative is O(n *
m) and becomes unrunnable on a few million rows. Whenever you see Python
code flying on set-like problems, a hash table is pulling the load.

**Dict as config, everywhere.** Every HuggingFace model config, every
PyTorch `state_dict`, every JSON payload, every kwargs bag, every logger
context, every feature flag bundle. You do not pick this data structure —
you use it because it is the medium of exchange.

---

## When NOT to use a hash table

Hash tables are magnificent, but they are not magical, and a few things they
cannot do.

**Sorted iteration.** A hash table gives you insertion order (Python 3.7+)
or nothing. For "everything in key order" use a balanced BST, a skip list,
or keep a sorted list on the side.

**Range queries.** "All keys between 100 and 200" is a fundamentally ordered
question, and the whole point of a hash table is that it throws order away.
For range queries use a tree (balanced BST, B-tree) or a sorted array with
binary search.

**Prefix queries.** "All keys starting with `foo_`" is the classic use case
for a **trie**. A trie walks shared prefixes in O(k) where `k` is the prefix
length; a hash table would have to scan everything.

**Hashable keys only.** You cannot use a list, dict, set or any mutable type
as a key. If a key could change after insertion, you could no longer find
the entry — the slot it lives in was determined by its *previous* hash.
Python enforces this by making mutable objects unhashable. Use a tuple.

---

## What comes next

This is the sixth article of Foundations. Six of the eight data structures
are on the board: arrays, matrices, tensors, linked lists, graphs, hash
tables. Part 7 will be **queues**.

You have already seen a queue if you read Part 5. The BFS traversal opened
with `from collections import deque` and treated that deque as a queue,
popping left and appending right. That worked because `collections.deque` is
not a Python list — it is a doubly linked list of fixed-size blocks, and
both ends are O(1). A list's `popleft` would have been O(n). Next article
we will pull the deque apart, look at the circular buffer it is made of,
and go from there to priority queues (binary heaps) and the single abstraction
that ties BFS, experience replay, message passing and job scheduling into
one FIFO story.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**hash_table.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/06-hash-tables/hash_table.py)

Run it with:

```bash
python hash_table.py
```

It runs in under a tenth of a second on a laptop. Try bumping the collision
test from 2000 to 20000 keys and watch the bad-hash time rise by a factor of
one hundred while the good-hash time barely moves. That gap, in a single
graph, is the entire point of this article.

---

*This is Part 6 of the Algorithms in Python series, Foundations track. The companion script `hash_table.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 5](https://medium.com/@grahamjroy) introduced graphs. Part 7 will look at queues — and show you why the `collections.deque` you used for BFS is so much faster than you'd think.*
