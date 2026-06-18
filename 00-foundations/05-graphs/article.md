# Graphs — The Data Structure Behind Everything Connected

### *Algorithms in Python --- Foundations, Part 5*

---

Open Wikipedia, type "Kevin Bacon" into the search bar, and pick any other article
you like. Now try to navigate from Kevin Bacon to that article using only the
blue links inside each page. Almost every time, you can do it in fewer than six
clicks. Kevin Bacon links to "Six Degrees of Kevin Bacon", which links to "Graph
theory", which links to "Quantum mechanics". Three hops. You have just performed a
breadth-first search across a graph with roughly seven million nodes.

Whenever a problem involves *things that are connected to other things*, you are
looking at a graph. Social networks, road maps, molecular structures, the web,
the call graph of a program, the attention pattern of a transformer, the citation
network of scientific papers --- all graphs. The data structure we introduce in
this article is the single most versatile tool in computer science, and it
underpins a huge slice of modern AI.

In the first four parts of this series we stored data in contiguous memory
(arrays, matrices, tensors) or in simple chains of nodes (linked lists). A graph
is what happens when you stop requiring your nodes to line up in a single sequence
and let them connect to *any* other node they need to.

---

## What is a graph?

A graph is a pair `G = (V, E)` where `V` is a set of **vertices** (also called
nodes) and `E` is a set of **edges** connecting them. Every graph problem boils
down to two questions: what are the nodes, and what does it mean for two nodes to
be connected?

A tiny example:

```
    Alice ------ Bob
      |           |
    Carol ----- Dave ----- Eve ----- Frank
```

Six people, six friendships. No root, no order, no fixed shape. You can start at
any vertex and walk along edges to reach others. Three axes of variation
determine what kind of graph you are dealing with.

**Directed vs undirected.** In an undirected graph, an edge between `A` and `B`
has no direction --- Facebook friendship is symmetric. In a directed graph
(digraph), edges have a head and a tail. Twitter follows, citation networks, and
the web itself are directed.

**Weighted vs unweighted.** A weighted edge carries a number: a distance, a cost,
a probability, a similarity score. Road networks carry distances. The attention
matrix inside a transformer is, in one reading, a weighted directed graph whose
weights are attention probabilities.

**Cyclic vs acyclic.** A cycle is a path that starts and ends at the same vertex
without repeating edges. A directed acyclic graph --- a DAG --- has special
importance in computing: build systems, task schedulers, and the computation
graphs of PyTorch and TensorFlow are all DAGs.

---

## Representing a graph in memory

There are two standard ways to store a graph, and they make almost opposite
trade-offs.

### Adjacency list

An **adjacency list** stores, for each vertex, a list of its neighbours. It is
typically a dictionary mapping each vertex to a Python list (or a linked list,
which is exactly the structure we built in Part 4). Here is the core of the
`Graph` class from the companion code:

```python
class Graph:
    def __init__(self, directed=False):
        self.adjacency = {}
        self.directed = directed

    def add_vertex(self, vertex):
        if vertex not in self.adjacency:
            self.adjacency[vertex] = []

    def add_edge(self, u, v):
        self.add_vertex(u)
        self.add_vertex(v)
        self.adjacency[u].append(v)
        if not self.directed:
            self.adjacency[v].append(u)
```

That is the entire data structure. An `add_edge` call appends the neighbour to
the list (or to both lists, for undirected edges). The graph grows exactly in
proportion to the number of edges you actually have.

### Adjacency matrix

An **adjacency matrix** is a two-dimensional array of shape `(|V|, |V|)` where
`M[i][j] = 1` if there is an edge from vertex `i` to vertex `j`, and `0`
otherwise. For a weighted graph, `M[i][j]` holds the weight. This is a matrix
--- the same data structure from Part 2 --- pressed into service as a
connectivity table.

```python
# A 4-vertex graph encoded as an adjacency matrix
M = [[0, 1, 1, 0],
     [1, 0, 0, 1],
     [1, 0, 0, 1],
     [0, 1, 1, 0]]
```

### Trade-offs

| Operation | Adjacency list | Adjacency matrix |
|---|---|---|
| Space | O(V + E) | O(V^2) |
| Add vertex | O(1) amortised | O(V^2) (resize) |
| Add edge | O(1) | O(1) |
| Check if edge (u, v) exists | O(deg(u)) | O(1) |
| Iterate over neighbours of v | O(deg(v)) | O(V) |
| Iterate over all edges | O(V + E) | O(V^2) |

The matrix gives constant-time edge lookup but pays quadratic space. The list is
linear in edges but slower to answer "is there an edge between these two
vertices".

The choice depends on density. A graph with `V` vertices has at most `V(V-1)/2`
undirected edges; if it has close to that many, it is **dense** and the matrix
wins. If it has far fewer --- which is almost always the case in practice --- it
is **sparse** and the adjacency list wins by a mile. A Facebook user has a few
hundred friends, not several billion. A paper cites a few dozen others, not the
entire corpus. Every serious graph library (NetworkX, igraph, PyTorch Geometric)
uses adjacency lists by default.

---

## Breadth-first search

BFS is the simplest graph traversal, and it is the algorithm you performed
intuitively when you navigated Wikipedia. Starting from a vertex, visit all of
its immediate neighbours first, then the neighbours of those neighbours. You
expand outward in concentric rings of increasing hop distance.

The key ingredient is a **queue**: a first-in-first-out structure holding the
vertices you have discovered but not yet explored. Python's `collections.deque`
gives O(1) append and popleft.

```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    order = []
    queue = deque([start])
    while queue:
        current = queue.popleft()
        order.append(current)
        for neighbour in graph.neighbours(current):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return order
```

On the friendship network above, `bfs(friends, "Alice")` produces
`['Alice', 'Bob', 'Carol', 'Dave', 'Eve', 'Frank']`. Alice first, then her
neighbours Bob and Carol, then their neighbour Dave, then Eve, then Frank. Six
vertices, each visited exactly once, in ring order.

BFS runs in O(V + E). Every vertex enters the queue at most once (thanks to the
`visited` set), and every edge is examined at most twice. That is the tightest
bound you can hope for on a traversal.

Because BFS explores vertices in order of hop distance, it automatically finds
the **shortest unweighted path** between two vertices --- the first time you
reach the target, you have reached it in the minimum number of hops. The
companion code's `bfs_shortest_path` function tracks the path taken to each
vertex, not just the visit order.

---

## Depth-first search

DFS takes the opposite approach: follow one path as deeply as you can before
backing up. Where BFS spreads out, DFS dives in. The only structural change is
replacing the queue with a **stack** (last-in-first-out).

```python
def dfs(graph, start):
    visited = set()
    order = []
    stack = [start]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        order.append(current)
        for neighbour in reversed(graph.neighbours(current)):
            if neighbour not in visited:
                stack.append(neighbour)
    return order
```

Swapping `popleft` for `pop` changes the behaviour completely. DFS follows the
first unexplored edge, and keeps following edges until it hits a dead end before
backtracking. Starting from Alice, DFS produces
`['Alice', 'Bob', 'Dave', 'Carol', 'Eve', 'Frank']` --- it dove from Alice to
Bob to Dave before backing up to visit Carol.

DFS also runs in O(V + E). A recursive variant is often easier to read, but for
huge graphs the iterative version is safer --- Python's recursion limit
(typically 1000) can be blown by a single deep path.

Where BFS is the tool for "shortest path" and "levels of separation", DFS is the
tool for topological sort, cycle detection, strongly connected components, and
any question about the overall *shape* of a graph.

---

## Dijkstra's shortest path

BFS finds the shortest path when every edge has the same cost. But most real
problems come with weights. "Shortest" cannot mean "fewest hops" --- it must
mean "smallest total weight". Dijkstra's algorithm is the workhorse solution,
provided all edge weights are non-negative.

The idea is a greedy expansion from the start vertex, always extending the
frontier through the vertex with the smallest known distance. A **priority
queue** --- a binary heap in practice --- replaces the plain queue from BFS.

```python
import heapq

def dijkstra(graph, start):
    distances = {v: float("inf") for v in graph.vertices()}
    predecessors = {v: None for v in graph.vertices()}
    distances[start] = 0

    heap = [(0, start)]
    while heap:
        current_dist, current = heapq.heappop(heap)
        if current_dist > distances[current]:
            continue
        for neighbour, weight in graph.neighbours(current):
            new_dist = current_dist + weight
            if new_dist < distances[neighbour]:
                distances[neighbour] = new_dist
                predecessors[neighbour] = current
                heapq.heappush(heap, (new_dist, neighbour))
    return distances, predecessors
```

Every vertex starts at infinity except the source. On each iteration we pop the
vertex with the smallest tentative distance and relax its outgoing edges: if we
can reach a neighbour more cheaply by going *through* the current vertex, we
update that neighbour's distance and push it back onto the heap. Stale entries
(for which a shorter path has already been found) are filtered out by the
`current_dist > distances[current]` check.

Total cost: O((V + E) log V). Dijkstra sits at the heart of every routing system
you have used. Google Maps runs A\* (Dijkstra plus a heuristic) over a road
network with tens of millions of edges. The same algorithm routes IP packets
across the internet backbone and plans motion paths for robots.

---

## Connected components

An undirected graph is **connected** if there is a path between every pair of
vertices. Many real graphs are not connected --- they split into several
independent pieces, called **connected components**. Finding them is a classic
application of repeated BFS (or DFS).

```python
def connected_components(graph):
    visited = set()
    components = []
    for vertex in graph.vertices():
        if vertex not in visited:
            component = set()
            queue = deque([vertex])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbour in graph.neighbours(current):
                    if neighbour not in visited:
                        queue.append(neighbour)
            components.append(component)
    return components
```

You walk across every vertex. Whenever you encounter an unvisited one, you fire
off a BFS from it to collect everything reachable --- that is one component. The
total work is O(V + E), because across all the BFS calls combined, each vertex
and each edge is still only processed once.

The companion code's friendship network has an isolated pair (Grace and Heidi)
tacked on, and the function correctly identifies two components. In social
network analysis, component structure tells you about community boundaries. In
bioinformatics, it tells you about gene modules. In image segmentation, it tells
you which pixels belong to the same object.

---

## PageRank

PageRank was the original secret sauce of Google. It answers a single question
over a directed graph: **which vertices are most important?**

Imagine a random web surfer who, at every step, either clicks a link on the
current page with probability `d` (the damping factor, typically 0.85) or
teleports to a random page with probability `1 - d`. Let the walk run forever.
The long-run probability that the surfer ends up on each page is that page's
PageRank.

You could solve this as a gigantic linear system, but the **power method** is
simpler: start with uniform rank and redistribute it along edges until it stops
changing.

```python
def pagerank(graph, damping=0.85, iterations=100, tolerance=1e-6):
    vertices = graph.vertices()
    n = len(vertices)
    rank = {v: 1.0 / n for v in vertices}
    out_degree = {v: len(graph.neighbours(v)) for v in vertices}

    for _ in range(iterations):
        new_rank = {v: (1.0 - damping) / n for v in vertices}
        dangling_sum = sum(rank[v] for v in vertices if out_degree[v] == 0)
        dangling_contribution = damping * dangling_sum / n

        for v in vertices:
            new_rank[v] += dangling_contribution
            if out_degree[v] > 0:
                share = damping * rank[v] / out_degree[v]
                for neighbour in graph.neighbours(v):
                    new_rank[neighbour] += share

        diff = sum(abs(new_rank[v] - rank[v]) for v in vertices)
        rank = new_rank
        if diff < tolerance:
            break
    return rank
```

There is one real line of algorithmic content: each vertex hands each outgoing
neighbour `damping * rank[v] / out_degree[v]` of its current rank, and every
vertex also receives a small uniform teleport contribution. Dangling vertices
(no outgoing edges) would leak probability, so their mass is redistributed.
Repeat until the ranks stop changing.

Under the hood this is a **fixed-point iteration on the stochastic matrix**
derived from the adjacency structure, converging to the **dominant
eigenvector** --- the stationary distribution of a random walk on the graph.
The companion code runs PageRank on a toy five-page web and produces a little
ASCII bar chart of vertex importance. The mechanics are exactly those of the
original Google algorithm.

---

## When NOT to use a graph

Graphs are versatile, but they are not always the right choice. For dense
numerical data a graph is a catastrophic fit. A 256 by 256 image has 65,536
pixels; encoding "pixels are next to their four neighbours" as explicit edges
would be absurd. Use a rank-3 tensor, as in Part 3 --- the spatial grid is
implicit in the indexing.

The same logic applies to tabular data. A CSV of customer records is not a
graph; it is a matrix. Reach for a graph only when the relationships between
entities are **irregular**, **sparse**, and **carry information of their own**.

A second warning: graphs are awkward to batch. GPUs love regular shapes, and
fitting graph operations onto tensor hardware is a genuine research problem.
Libraries like PyTorch Geometric and DGL exist precisely to turn graph
traversals into dense matrix multiplications wherever possible.

---

## AI and ML relevance

Graphs are a structural backbone of modern machine learning. Once you recognise
them, you see them everywhere.

**Knowledge graphs and RAG.** A knowledge graph encodes facts as triples:
(subject, predicate, object). Paris --- is-capital-of --- France. Its advantage
over a table is that it represents multi-hop relationships naturally.
Retrieval-augmented generation systems increasingly use knowledge graphs to
ground language models in structured facts and let them trace reasoning chains
across related entities. Microsoft's GraphRAG builds a knowledge graph from a
text corpus and queries it alongside vector search. We will give knowledge
graphs their own article later in this foundations series.

**Graph neural networks.** A GNN operates directly on a graph. Each node has a
feature vector, and each layer updates those features based on the features of
the node's neighbours --- **message passing**. Information propagates across the
graph, and the model learns representations that encode both node attributes and
graph structure. GNNs are the state of the art for molecular property
prediction, drug discovery, protein interaction analysis, traffic forecasting,
and recommendation. DeepMind's AlphaFold uses graph-based attention to predict
protein structure. GNNs exist because many problems are naturally graphs, not
tables.

**Computation graphs in autograd.** When you call `.backward()` in PyTorch, the
system walks a directed acyclic graph of operations built during the forward
pass. Every operation is a node, every input/output relationship is an edge,
and the topological order of the DAG determines the order of gradient
computation. This is not a metaphor --- it is a literal graph data structure,
and it is why PyTorch can handle dynamic control flow that static graph
frameworks struggle with.

**Attention as a graph over tokens.** Every transformer layer computes
attention scores between pairs of tokens. A transformer layer is processing a
**complete directed graph** over the input tokens, where every edge carries a
learned attention weight. The attention mask then restricts which edges exist
--- causal masking makes it a DAG, local attention makes it sparse. Sparse
attention research (Longformer, BigBird, Reformer) is explicit about this: the
authors are deliberately sparsifying a dense graph to reduce quadratic cost to
linear.

**Spectral clustering.** Build a similarity graph whose edge weights measure how
similar two points are, compute the Laplacian matrix, and cluster points in the
embedding given by its smallest non-trivial eigenvectors. This is spectral
clustering, and it captures global cluster structure that distance-based methods
like k-means cannot see. It uses the same machinery that underlies PageRank:
eigenvalue problems on graph matrices.

**Consciousness and information integration.** Integrated Information Theory
(Article 2 of the consciousness series) measures consciousness through a
quantity called **phi**, computed over a graph of causal interactions between
network elements. Once you can describe a system as a graph, a huge toolbox of
algorithms becomes available to reason about it.

---

## What comes next

In Part 4 we noted that a graph is typically represented as an array of linked
lists. We have just built exactly that. The adjacency list is a dictionary of
lists, and each list hangs off a vertex the way the nodes of a singly linked
list hang off a head pointer. A graph is what happens when you let nodes have
multiple outgoing pointers and allow those pointers to form cycles.

In Part 6 we turn to **hash tables**. We have been using them already --- the
adjacency dictionary above is a hash table --- but treating them as magic. The
next article pulls the magic apart: how hash functions work, what collisions
look like, why Python's `dict` resizes the way it does, and where hash tables
show up in ML: feature hashing for high-cardinality categoricals, count-min
sketches on streams, locality-sensitive hashing for approximate nearest
neighbours, and the vocab table of every tokeniser on earth.

---

## The complete code

The full script is on GitHub --- grab it here and run it yourself:

[**graphs.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/05-graphs/graphs.py)

Run it with:

```bash
python graphs.py
```

---

*This is Part 5 of the series "Algorithms in Python". You can find the full series
and source code at [github.com/grahamroy/algorithms-in-python](https://github.com/grahamroy/algorithms-in-python).*
