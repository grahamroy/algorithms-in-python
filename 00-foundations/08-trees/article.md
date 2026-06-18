# Trees — Hierarchical Structure for Decisions, Search, and Database Indexes

### *Algorithms in Python --- Foundations, Part 8*

---

At the end of Part 7 I claimed that a binary heap is "a tree hiding inside
an array." That was a hint about today. The heap is actually the simplest
tree most engineers ever use — flat in memory, balanced for free, with a
single invariant. Honest trees, the ones that branch arbitrarily and grow
unevenly, look very different in code and have very different performance
profiles.

But you have already used dozens of them. Your filesystem is a tree. The
DOM the browser parses is a tree. The expression `x * (y + z)` is parsed
into a tree before it is evaluated. Every `scikit-learn` decision tree classifier,
every B-tree index inside Postgres, every nearest-neighbour query inside
FAISS or scikit-learn — trees, all the way down.

In Part 5 we walked **graphs** and met BFS and DFS. A tree is just a graph
with two extra rules: it is connected, and it has no cycles. Everything
we learned about traversal carries over verbatim. What changes is the
*structure* — the parent/child relationship gives us a notion of hierarchy
that plain graphs do not have, and that hierarchy is what makes trees the
right data structure for an enormous family of problems.

This article is about that structure. Six tree shapes — binary search
trees, decision trees, random forests, KD-trees, B-trees, and tries — and
why each one exists.

---

## What is a tree?

A tree is a collection of **nodes** connected by **edges**, with one node
designated as the **root**, and every other node having exactly one parent.
Nodes with no children are **leaves**. The **depth** of a node is the
number of edges from the root to it; the **height** of the tree is the
maximum depth of any leaf.

```
              root
             /    \
            A      B
           / \      \
          C   D      E
                    / \
                   F   G    <-- leaves
```

Three vocabulary points. **Branching factor** is how many children a node
can have — a binary tree caps it at two, a B-tree allows hundreds.
**Balanced** means every leaf is at roughly the same depth, so traversals
take O(log n) instead of degenerating to O(n). **Subtree** is the recursive
gift: every node together with its descendants is itself a tree, which is
why almost every tree algorithm fits in five lines of recursion.

```python
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []
```

That is the entire structural definition. Specialisations add fields —
`left`/`right` for binary trees, `keys` and `children` lists for B-trees,
a fitted threshold for decision trees — but the spine is always the same.

---

## Tree traversal — BFS and DFS, with three flavours

In Part 5 we visited every node of a graph in two ways: breadth-first with
a queue, depth-first with a stack (or recursion). On a tree, both still
work — and DFS gains three sub-flavours depending on *when* you visit the
current node relative to its children.

```python
def preorder(node):    # visit BEFORE recursing
    if node is None: return
    print(node.value)
    preorder(node.left)
    preorder(node.right)

def inorder(node):     # visit BETWEEN recursions
    if node is None: return
    inorder(node.left)
    print(node.value)
    inorder(node.right)

def postorder(node):   # visit AFTER recursing
    if node is None: return
    postorder(node.left)
    postorder(node.right)
    print(node.value)
```

Three lines moved, three different orderings. Each one has a job.

**Pre-order** is what you want when the parent must be processed before
its children — copying a tree, serialising it, evaluating a top-down rule.

**In-order** on a binary search tree gives you the values in sorted order.
That is not a coincidence; it is the entire reason BSTs exist.

**Post-order** is what you want when children must be processed before the
parent — freeing memory, computing a recursive aggregate (subtree size,
maximum depth, evaluating an arithmetic expression bottom-up).

**Level-order**, or BFS, is the queue-based traversal from Part 7 ported
verbatim to a tree. Visit the root, enqueue its children, then visit them
in arrival order and enqueue *their* children. The companion script runs
all four traversals on the same hand-built tree so you can see the
orderings side by side.

---

## Binary search trees

A **binary search tree** (BST) is a binary tree with one rule: for every
node, all values in the left subtree are smaller, all values in the right
subtree are larger. That single invariant gives you O(log n) lookup,
insertion, and deletion — *if* the tree stays balanced.

```python
class BSTNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def insert(root, value):
    if root is None:
        return BSTNode(value)
    if value < root.value:
        root.left = insert(root.left, value)
    elif value > root.value:
        root.right = insert(root.right, value)
    return root

def search(root, value):
    if root is None or root.value == value:
        return root
    if value < root.value:
        return search(root.left, value)
    return search(root.right, value)
```

The "if balanced" caveat is doing real work. Insert the integers `1, 2, 3,
4, 5` into an empty BST in that order and you get a tree that is just a
linked list — every node has one child, lookup is O(n), and you have done
worse than a sorted array. This is why production code never uses a naive
BST. It uses a **self-balancing** variant — red-black trees in
`std::map`, `TreeMap`, and Linux's process scheduler; AVL trees in some
in-memory indexes; B-trees on disk. The invariant is the same; the
extra machinery is rotations that re-shape the tree on every insert to
keep its height at O(log n).

Python's standard library does not ship a self-balancing BST — for
in-memory ordered keys you reach for `sortedcontainers.SortedDict`, which
is implemented on top of sorted lists, or you use a heap if you only need
the minimum. The interesting BSTs are the ones we never write ourselves.

---

## Decision trees — the everyday ML workhorse

Switch domains. A **decision tree classifier** is a tree where every
internal node is a question (`age < 35?`, `pixel > 0.4?`, `income >
60_000?`), every edge is the answer, and every leaf is a prediction. To
predict, walk from the root, follow the answer at each node, and read the
label off the leaf you land on.

```
                age < 35?
               /         \
            yes            no
            /              \
       income > 50k?     defaulted_before?
        /     \            /        \
      no     yes         yes        no
      |       |           |          |
   risky    safe        risky       safe
```

Training is what the algorithm actually does for you. At each internal
node it considers every feature, every possible split threshold, and
picks the split that most cleanly separates the labels in the training
data. "Most cleanly" is measured by a function — typically **Gini
impurity** or **entropy**. Both reach the same intuition: a split is good
if the resulting groups are more homogeneous than the parent.

```
Gini(node) = 1 - sum_k (p_k)^2
```

where `p_k` is the proportion of class `k` in the node. A pure node
(everything one class) has Gini = 0; a perfectly mixed node of two classes
has Gini = 0.5. The algorithm greedily picks the split that maximises the
reduction in Gini between the parent and the weighted average of its
children, then recurses into each child.

The companion script trains a depth-3 decision tree by hand on a toy
loan dataset, picking the best Gini-reducing split at each node:

```
Training rows: 18
  Root Gini: 0.475

Building decision tree (depth 3):

split on income <= 42500  (Gini 0.475 -> 0.188, gain 0.287)
  Left  (income <= 42500):
    -> predict RISKY  (n=5, purity=1.00)
  Right (income > 42500):
    split on prior_default == True  (Gini 0.260 -> 0.000, gain 0.260)
      Left  (prior_default == True):
        -> predict RISKY  (n=2, purity=1.00)
      Right (prior_default != True):
        -> predict SAFE  (n=11, purity=1.00)
```

Two splits and the tree is done — every leaf is pure, so the algorithm
stops early even though we allowed it to grow to depth 3. The first
split picks income because that single feature explains most of the
label variation; among the higher-income applicants it picks
`prior_default` to peel off the two who defaulted before. A human looking at
the same dataset would draw exactly that flowchart, which is the whole
appeal of decision trees.

Decision trees are the reason tabular ML never quite went away. They
handle mixed feature types without preprocessing, they cope with missing
values, they are interpretable (you can literally read the rules), and
they make no assumption about the distribution of your data. They also
overfit aggressively if you let them grow to depth log(n), which is the
problem the next data structure solves.

---

## Random forests — the wisdom of many trees

A single decision tree is high-variance: shuffle the training data
slightly and you get a different tree with different rules and different
predictions. A **random forest** averages over many such trees, each
trained on a bootstrap sample of the rows and a random subset of the
features at every split. Variance averages out, and the forest as a
whole is far more stable than any one tree.

```
              vote / average
             /       |       \
         tree_1   tree_2   ...   tree_N
         /  \     /  \           /  \
        ...  ... ...  ...       ...  ...
```

The "random" in random forest does two things. **Bagging** (bootstrap
aggregation) gives each tree a different slice of the rows, so they make
different mistakes. **Feature subsampling** at each split — typically
`sqrt(num_features)` of them — stops every tree from becoming a copy of
the same dominant feature's split. The result is a diverse ensemble whose
errors decorrelate.

Random forests, gradient-boosted trees (XGBoost, LightGBM, CatBoost), and
their relatives still win on most well-behaved tabular benchmarks. Deep
learning beat them at images, audio, and language; on a CSV with a
hundred thousand rows and forty columns, gradient boosting is almost
always the right first move. The whole field of tabular ML is
"how do we ensemble decision trees better." It is hard to overstate how
durable this idea is.

---

## KD-trees and ball trees — partitioning space for nearest-neighbour search

Switch domains again. You have a million points in 50-dimensional space
and you need to find the nearest neighbour of a new query point. Brute
force — distance to every point — is O(n) per query. A million queries on
a million points is O(n^2) and you wait days.

A **KD-tree** ("k-dimensional tree") cuts this to O(log n) per query in
low dimensions by recursively partitioning space. At the root, pick a
dimension (say x) and split the points into two halves at the median. At
the next level, switch to the next dimension (y) and split each half at
the median of *that* dimension. Recurse. Each leaf holds a small bucket
of points; each internal node remembers its split dimension and split
value.

```
y
^
|   .  .  | .       .
|         |   .  .
|  .      | .       .
|---------+----------       <-- split on y at the median
|     .   |    .  .
| .       |  .   .  .
| .  .    |        .
+---------+--------------> x
          ^
          split on x at the median
```

Querying is the half that matters. Walk down the tree as if doing a BST
search using the query's coordinates, find the leaf bucket, and compute
the distance to every point in it — that gives you a *candidate* nearest
neighbour. Then back up: if the distance to the splitting plane on the
way down is smaller than your current best distance, the true nearest
neighbour might be on the other side, so you have to visit it too.
Otherwise you can prune the entire subtree.

In low dimensions (k ≤ 20 or so) the pruning is dramatic and queries
finish in O(log n). As dimensionality grows the pruning stops working —
the "curse of dimensionality" means hyperspheres rarely fit on one side
of a hyperplane, so you keep having to visit the other branch. By
k = 100 a KD-tree is barely better than brute force, which is why FAISS,
Annoy, and ScaNN abandon exact search for approximate methods (HNSW
graphs, IVF, product quantisation) once dimensionality grows. The trees
are still in there — HNSW is a layered graph that *behaves* like a
multi-resolution tree at query time.

A **ball tree** is the same idea with hyperspheres instead of axis-aligned
boxes. It tolerates higher dimensions a bit better. scikit-learn's
`NearestNeighbors` picks between brute force, KD-tree, and ball tree
based on the data shape; the choice is largely about how many dimensions
you have.

---

## B-trees — the data structure your database actually uses

Last tree, and the one most engineers underestimate. A **B-tree** is a
self-balancing tree where every node holds many keys (typically hundreds)
and has many children (one more than the number of keys). The point is
not the algorithmic complexity — every balanced tree gives you O(log n).
The point is the **fan-out**.

```
                   [25 | 50 | 75]
                  /     |    |    \
            [..]    [..]   [..]    [..]
           keys     keys    keys     keys
           1-24    26-49   51-74    76-100
```

A binary tree of one billion keys has height ~30. A B-tree of one billion
keys with 200 children per node has height 4. On disk, where every node
visited is a separate page read costing a few milliseconds, that ratio is
the difference between an index that responds in microseconds and one
that does not. Postgres, SQLite, MySQL/InnoDB, MongoDB, and every
filesystem with metadata to look up — all use B-trees, or the close
cousin **B+ trees** which keep all keys at the leaves with sibling
pointers between them for fast range scans.

The cache-locality argument carries over to RAM. A node that fits inside
one CPU cache line is far faster to scan linearly for a key than chasing
log(n) pointer indirections through a binary tree. Modern in-memory
indexes — `art-tree`, `Masstree`, B-trees in CockroachDB's KV layer —
exploit this. The lesson is that "balanced binary tree" is rarely the
right shape on real hardware once n is large; "balanced wide tree" is.

---

## Tries — when your keys are strings

A **trie** (or **prefix tree**) is a tree where each edge is one
character and each path from the root spells a key. Storing `cat`, `car`,
`cards`, and `cart` looks like:

```
        root
         |
         c
         |
         a
        / \
       t   r
      (*)  |
           +-- (*)        "car"
           |
           d --- s (*)    "cards"
           |
           t (*)          "cart"
```

`(*)` marks the end of a stored key. Lookup, insertion, and prefix
search are all O(L) where L is the length of the key — independent of
how many keys you have stored. That is the property nothing else gives
you cleanly. Autocomplete uses tries. So do IP-routing tables ("longest
prefix match"). Spell checkers. Subword tokenisers. The DAWG variant
collapses suffixes for huge memory savings; modern BPE tokenisers in
LLM pipelines often build a trie internally so that "matching the
longest token at this position" runs in time linear in the token's
length, not the vocabulary's size.

---

## Big-O summary

[[BIG-O TABLE IMAGE]]

Three things to read off this table. **Balanced** is doing the work in
every "log n" — an unbalanced BST or KD-tree is O(n) in the worst case,
which is the most common bug in this whole topic. **Construction is more
expensive than query** — building a KD-tree is O(n log n), but if you
will run a million queries against it, that is a one-time cost amortised
to nothing. **Trie complexity is in the key length, not the corpus size**
— a property the others do not have, which is what makes tries the right
choice for string-keyed work.

---

## Real-world ML and AI connections

Trees are the load-bearing structure under more of the stack than any
other shape we have looked at.

**Gradient boosting wins tabular ML.** XGBoost, LightGBM, and CatBoost are
all "build many shallow decision trees, each correcting the previous
ensemble's errors." Every Kaggle competition on tabular data, every credit
scoring model, every churn classifier in production at scale — almost
always boosted trees. Deep learning has not displaced them on this problem
because the trees encode the right inductive bias for low-dimensional,
mixed-type, sparse-pattern data.

**Decision trees as policy.** Some reinforcement learning agents use
**decision-tree policies** when interpretability matters more than raw
reward — a recommender, a clinical decision support system, a
credit-limit policy. The fitted tree is a literal flowchart you can show
a regulator. The trade-off against a neural network is paid in
performance but bought back in audit trail.

**Monte Carlo Tree Search.** AlphaGo, AlphaZero, MuZero, and most modern
game-playing systems use MCTS to balance exploration and exploitation in
a game tree. The "tree" is the game state space; nodes are positions,
edges are moves, leaves are terminal positions or rollouts. Each
simulated playout updates statistics on the visited nodes, and the next
move is picked from the root using the **UCB1** formula. It is one of
the few places in modern AI where the explicit tree datatype, not a
neural net, is doing the heavy lifting.

**Parse trees in compilers and LLMs.** Every program your compiler reads
is parsed into an **abstract syntax tree** before any optimisation
happens. Linters, formatters, type checkers, and refactoring tools all
walk that tree. Even LLM-driven coding assistants increasingly use AST
manipulation rather than string substitution because edits at the AST
level cannot break syntax by accident. Treesitter and Babel are
production-grade tree libraries that have quietly become the substrate
of modern code tooling.

**R-trees for spatial indexes.** GeoPandas, PostGIS, MongoDB's `2dsphere`
index, every "find restaurants within 1km" query — all built on
**R-trees**, which generalise B-trees to bounding boxes in two or more
dimensions. Same fan-out logic, same logarithmic height, applied to
geometry instead of scalar keys.

**Sum-trees in prioritised experience replay.** Part 7 mentioned this in
passing. A **sum-tree** is a binary tree where each internal node stores
the sum of its children's leaf weights. Sampling proportional to weight
becomes O(log n): generate a random number in `[0, total]`, walk down
the tree picking the child whose subtree sum contains the target. DeepMind
used it in the original prioritised replay paper, and it is now standard
in any RL library that supports PER.

**Merkle trees and content-addressed storage.** Git, IPFS, blockchains,
and content-distribution networks all use **Merkle trees** — a tree where
every internal node is the hash of its children. Two repos are identical
iff their root hashes match, and you can prove a single leaf is in the
tree by transmitting only the path from leaf to root. Vector databases
like LanceDB use Merkle structures for incremental snapshots; modern
LLM evaluation harnesses use them to deduplicate prompt-response pairs
across runs.

**Tokeniser tries in LLM serving.** Byte-pair encoding tokenisers (the
ones inside GPT, Claude, and Llama) match the longest known token at
each position in the input string. The fastest implementations build a
trie of the vocabulary at startup, so each token match is O(L) in the
token's length rather than O(V) in the vocabulary's size. With
vocabularies of 100k+ tokens that difference is the kind of thing that
shows up in tail latency of a serving stack.

---

## When NOT to use a tree

Trees are powerful but they are not free.

**When you need O(1) lookup by key.** A hash table is faster than any
tree for unordered key-value access. Use the dict. Reach for a tree only
when you need ordering, prefix structure, or hierarchy.

**When the data is small.** A linear scan over a hundred items is faster
than building and walking a tree, and infinitely simpler to debug.
Premature tree-isation is one of the surest ways to slow code down.

**When you need to mutate the structure constantly.** Self-balancing
trees pay a cost on every insert and delete. A workload that is mostly
writes with rare reads is often better served by an append-only log plus
a periodic rebuild, or by skipping the tree entirely and using a hash
table.

**When the dimensionality is high.** KD-trees and ball trees both
degrade to brute-force speed somewhere around 20 to 50 dimensions. For
embeddings of dimension 384, 768, or 1536 — the typical range for
modern text embeddings — use HNSW, IVF, or product quantisation
(approximate methods), not a spatial tree.

**When you do not have a balancing guarantee.** A naive BST that ingests
sorted data is a linked list with bad acoustics. Use a self-balancing
implementation, or use the right structure for your access pattern in
the first place.

---

## What comes next

Eight foundations down, four to go. Part 9 is **Knowledge Graphs** — the
one that ties the entire data-structure track back to where graphs left
off in Part 5. We will look at triples,
ontologies, entity embeddings, and how knowledge graphs power the
retrieval-augmented generation systems that sit underneath modern LLM
applications.

Then we leave the data-structure layer entirely and start building the
algorithms that run on top — beginning with linear regression, the first
of the supervised-learning track, and a return visit to the matrices and
vectors we set up in Parts 1 and 2.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**trees.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/08-trees/trees.py)

Run it with:

```bash
python trees.py
```

It finishes in well under a second on a laptop. The companion script
walks all four traversals on a hand-built tree, builds a BST, fits a
depth-3 decision tree on a toy loan dataset by computing Gini gain on
every candidate split, and runs a KD-tree nearest-neighbour query
against brute-force search on 10,000 random 2D points. On my laptop the
KD-tree finishes 200 queries in under 2 ms while brute force takes over
600 ms — a few-hundred-times speed-up for what is, in the end, a
hundred-line file.

---

*This is Part 8 of the Algorithms in Python series, Foundations track. The companion script `trees.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 7](https://grahamjroy.medium.com/queues-fifo-priority-and-the-one-abstraction-behind-bfs-replay-buffers-and-schedulers-e84ab25ef0fb) covered queues. Part 9 will look at knowledge graphs — the structure that powers RAG systems and the entity-relation backbone of modern AI.*
