"""
Graphs — The Data Structure Behind Everything Connected
Algorithms in Python — Foundations, Part 5

Demonstrates graph data structures built from scratch: adjacency-list
graphs (directed and undirected), weighted graphs, breadth-first and
depth-first search, Dijkstra's shortest path, connected components,
and a small PageRank implementation using the iterative power method.
"""

import sys
from collections import deque
import heapq

sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# Part 1 — Graph (adjacency list, unweighted)
# =============================================================================

class Graph:
    """
    Unweighted graph stored as an adjacency list.

    Each vertex maps to a list of neighbouring vertices. Supports both
    directed and undirected modes via the `directed` flag. Vertices can
    be any hashable type (strings, ints, tuples).
    """

    def __init__(self, directed=False):
        self.adjacency = {}
        self.directed = directed

    def add_vertex(self, vertex):
        """Add a vertex. Idempotent — adding an existing vertex is a no-op."""
        if vertex not in self.adjacency:
            self.adjacency[vertex] = []

    def add_edge(self, u, v):
        """
        Add an edge between u and v. In undirected mode, the edge is
        stored in both adjacency lists so traversal works in either
        direction.
        """
        self.add_vertex(u)
        self.add_vertex(v)
        self.adjacency[u].append(v)
        if not self.directed:
            self.adjacency[v].append(u)

    def neighbours(self, vertex):
        """Return the list of neighbours for a vertex."""
        return self.adjacency.get(vertex, [])

    def vertices(self):
        """Return all vertices in the graph."""
        return list(self.adjacency.keys())

    def __len__(self):
        return len(self.adjacency)

    def __repr__(self):
        kind = "Directed" if self.directed else "Undirected"
        return f"{kind}Graph(|V|={len(self.adjacency)})"


# =============================================================================
# Part 2 — WeightedGraph (adjacency list with edge weights)
# =============================================================================

class WeightedGraph:
    """
    Weighted graph stored as an adjacency list of (neighbour, weight) pairs.

    Supports directed and undirected modes. Weights must be non-negative
    for Dijkstra's algorithm to produce correct results.
    """

    def __init__(self, directed=False):
        self.adjacency = {}
        self.directed = directed

    def add_vertex(self, vertex):
        if vertex not in self.adjacency:
            self.adjacency[vertex] = []

    def add_edge(self, u, v, weight):
        """Add a weighted edge between u and v."""
        self.add_vertex(u)
        self.add_vertex(v)
        self.adjacency[u].append((v, weight))
        if not self.directed:
            self.adjacency[v].append((u, weight))

    def neighbours(self, vertex):
        return self.adjacency.get(vertex, [])

    def vertices(self):
        return list(self.adjacency.keys())

    def __len__(self):
        return len(self.adjacency)


# =============================================================================
# Part 3 — Traversals: BFS and DFS
# =============================================================================

def bfs(graph, start):
    """
    Breadth-first search. Visits vertices in order of increasing
    distance from the start. Returns the order in which vertices were
    visited. Runs in O(V + E).
    """
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


def bfs_shortest_path(graph, start, target):
    """
    Find the shortest unweighted path between start and target using BFS.
    Returns a list of vertices from start to target, or None if no path
    exists. For unweighted graphs, BFS finds the minimum-hop path.
    """
    if start == target:
        return [start]
    visited = set([start])
    queue = deque([(start, [start])])
    while queue:
        current, path = queue.popleft()
        for neighbour in graph.neighbours(current):
            if neighbour == target:
                return path + [neighbour]
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, path + [neighbour]))
    return None


def dfs(graph, start):
    """
    Depth-first search (iterative, using an explicit stack). Returns the
    order in which vertices were visited. Runs in O(V + E).
    """
    visited = set()
    order = []
    stack = [start]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        order.append(current)
        # Reverse so that neighbours are explored in insertion order
        for neighbour in reversed(graph.neighbours(current)):
            if neighbour not in visited:
                stack.append(neighbour)
    return order


def dfs_recursive(graph, start, visited=None, order=None):
    """
    Depth-first search (recursive variant). Easier to read, but limited
    by Python's recursion depth on very large graphs.
    """
    if visited is None:
        visited = set()
    if order is None:
        order = []
    visited.add(start)
    order.append(start)
    for neighbour in graph.neighbours(start):
        if neighbour not in visited:
            dfs_recursive(graph, neighbour, visited, order)
    return order


# =============================================================================
# Part 4 — Dijkstra's shortest path
# =============================================================================

def dijkstra(graph, start):
    """
    Dijkstra's single-source shortest path for graphs with non-negative
    weights. Returns two dictionaries: distances from start to every
    reachable vertex, and predecessors for reconstructing paths.

    Uses a binary heap as the priority queue. Time complexity is
    O((V + E) log V).
    """
    distances = {v: float("inf") for v in graph.vertices()}
    predecessors = {v: None for v in graph.vertices()}
    distances[start] = 0

    # Priority queue of (distance, vertex) pairs
    heap = [(0, start)]
    while heap:
        current_dist, current = heapq.heappop(heap)
        # Skip stale entries (we may have already found a shorter path)
        if current_dist > distances[current]:
            continue
        for neighbour, weight in graph.neighbours(current):
            new_dist = current_dist + weight
            if new_dist < distances[neighbour]:
                distances[neighbour] = new_dist
                predecessors[neighbour] = current
                heapq.heappush(heap, (new_dist, neighbour))
    return distances, predecessors


def reconstruct_path(predecessors, target):
    """Walk backwards through the predecessor map to build a path."""
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = predecessors[current]
    return list(reversed(path))


# =============================================================================
# Part 5 — Connected components (undirected graphs)
# =============================================================================

def connected_components(graph):
    """
    Find all connected components in an undirected graph. Returns a
    list of sets, where each set contains the vertices of one component.
    Runs in O(V + E).
    """
    visited = set()
    components = []
    for vertex in graph.vertices():
        if vertex not in visited:
            # BFS from this vertex to find everything reachable
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


# =============================================================================
# Part 6 — PageRank (iterative power method)
# =============================================================================

def pagerank(graph, damping=0.85, iterations=100, tolerance=1e-6):
    """
    Compute PageRank scores for a directed graph using the iterative
    power method.

    Each vertex starts with an equal share of rank. On every iteration,
    each vertex distributes its current rank evenly across its outgoing
    edges. The damping factor (usually 0.85) models the chance a random
    surfer keeps clicking links rather than jumping to a random page.

    Dangling vertices (no outgoing edges) redistribute their rank to
    every vertex uniformly, which preserves the total probability mass.

    Returns a dictionary mapping vertex -> PageRank score. Scores sum
    to 1.
    """
    vertices = graph.vertices()
    n = len(vertices)
    if n == 0:
        return {}

    # Initialise every vertex with equal rank
    rank = {v: 1.0 / n for v in vertices}

    # Pre-compute out-degrees once
    out_degree = {v: len(graph.neighbours(v)) for v in vertices}

    for _ in range(iterations):
        new_rank = {v: (1.0 - damping) / n for v in vertices}

        # Handle dangling nodes: their rank is distributed uniformly
        dangling_sum = sum(rank[v] for v in vertices if out_degree[v] == 0)
        dangling_contribution = damping * dangling_sum / n

        for v in vertices:
            new_rank[v] += dangling_contribution
            if out_degree[v] > 0:
                share = damping * rank[v] / out_degree[v]
                for neighbour in graph.neighbours(v):
                    new_rank[neighbour] += share

        # Check for convergence using the L1 norm of the difference
        diff = sum(abs(new_rank[v] - rank[v]) for v in vertices)
        rank = new_rank
        if diff < tolerance:
            break

    return rank


# =============================================================================
# Demo — runs when the script is executed directly
# =============================================================================

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Build an undirected social-network-style graph
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("UNDIRECTED GRAPH — Friendship Network")
    print("=" * 60)

    friends = Graph(directed=False)
    edges = [
        ("Alice", "Bob"),
        ("Alice", "Carol"),
        ("Bob", "Dave"),
        ("Carol", "Dave"),
        ("Dave", "Eve"),
        ("Eve", "Frank"),
        # Isolated pair — a second component
        ("Grace", "Heidi"),
    ]
    for u, v in edges:
        friends.add_edge(u, v)

    print(f"\nGraph: {friends}")
    print(f"Vertices: {friends.vertices()}")
    print("\nAdjacency list:")
    for vertex in friends.vertices():
        print(f"  {vertex:>6} -> {friends.neighbours(vertex)}")

    # -------------------------------------------------------------------------
    # BFS and DFS traversals
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAVERSALS — BFS vs DFS")
    print("=" * 60)

    print(f"\nBFS from Alice: {bfs(friends, 'Alice')}")
    print(f"DFS from Alice: {dfs(friends, 'Alice')}")
    print(f"DFS (recursive) from Alice: {dfs_recursive(friends, 'Alice')}")

    # -------------------------------------------------------------------------
    # Shortest unweighted path via BFS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SHORTEST UNWEIGHTED PATH (BFS)")
    print("=" * 60)

    path = bfs_shortest_path(friends, "Alice", "Frank")
    print(f"\nShortest hops Alice -> Frank: {path}")
    print(f"Distance (edges): {len(path) - 1}")

    missing = bfs_shortest_path(friends, "Alice", "Grace")
    print(f"\nAlice -> Grace: {missing}  (no path — different component)")

    # -------------------------------------------------------------------------
    # Connected components
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONNECTED COMPONENTS")
    print("=" * 60)

    components = connected_components(friends)
    print(f"\nFound {len(components)} components:")
    for i, comp in enumerate(components, start=1):
        print(f"  Component {i}: {sorted(comp)}")

    # -------------------------------------------------------------------------
    # Weighted graph + Dijkstra's shortest path
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("WEIGHTED GRAPH — Dijkstra's Shortest Path")
    print("=" * 60)

    # A tiny road network with distances in kilometres
    roads = WeightedGraph(directed=False)
    road_edges = [
        ("London",     "Oxford",      90),
        ("London",     "Cambridge",   100),
        ("Oxford",     "Birmingham",  110),
        ("Cambridge",  "Birmingham",  160),
        ("Birmingham", "Manchester",  140),
        ("Oxford",     "Bristol",     120),
        ("Bristol",    "Manchester",  280),
    ]
    for u, v, w in road_edges:
        roads.add_edge(u, v, w)

    distances, predecessors = dijkstra(roads, "London")
    print("\nShortest distances from London:")
    for city in sorted(distances.keys()):
        print(f"  {city:>11}: {distances[city]} km")

    path_to_manchester = reconstruct_path(predecessors, "Manchester")
    print(f"\nShortest path London -> Manchester: {' -> '.join(path_to_manchester)}")
    print(f"Total distance: {distances['Manchester']} km")

    # -------------------------------------------------------------------------
    # Directed graph + PageRank
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DIRECTED GRAPH — PageRank")
    print("=" * 60)

    # A tiny five-page web. Page A is linked from everywhere.
    web = Graph(directed=True)
    web_edges = [
        ("B", "A"),
        ("C", "A"),
        ("C", "B"),
        ("D", "A"),
        ("D", "B"),
        ("D", "C"),
        ("E", "A"),
        ("E", "D"),
        ("A", "E"),
    ]
    for u, v in web_edges:
        web.add_edge(u, v)

    print("\nDirected edges (link -> link):")
    for vertex in sorted(web.vertices()):
        print(f"  {vertex} -> {web.neighbours(vertex)}")

    ranks = pagerank(web, damping=0.85, iterations=100)
    print("\nPageRank scores (higher = more important):")
    for vertex, score in sorted(ranks.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(score * 100)
        print(f"  {vertex}: {score:.4f}  {bar}")
    print(f"\nSum of ranks: {sum(ranks.values()):.4f}  (should be ~1.0)")

    print("\n" + "=" * 60)
    print("All demos complete.")
    print("=" * 60)
