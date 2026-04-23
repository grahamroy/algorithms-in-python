"""
trees.py --- companion code for "Trees" (Foundations, Part 8).

Four demos:
  1. Tree traversals: pre-, in-, post-, and level-order on a hand-built tree.
  2. Binary search tree: insertion, in-order traversal as sorted output, search.
  3. Decision tree: depth-3 classifier built by hand on a toy loan dataset
     using Gini impurity to pick splits.
  4. KD-tree nearest neighbour search, timed against brute force on 10,000
     random 2D points.

Pure stdlib. Runs in well under a second.
"""

from collections import deque
from time import perf_counter
import math
import random


SEPARATOR = "=" * 64


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Demo 1 --- tree traversals
# ---------------------------------------------------------------------------

class TreeNode:
    __slots__ = ("value", "left", "right")

    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def preorder(node, out):
    if node is None:
        return
    out.append(node.value)
    preorder(node.left, out)
    preorder(node.right, out)


def inorder(node, out):
    if node is None:
        return
    inorder(node.left, out)
    out.append(node.value)
    inorder(node.right, out)


def postorder(node, out):
    if node is None:
        return
    postorder(node.left, out)
    postorder(node.right, out)
    out.append(node.value)


def level_order(root):
    out = []
    if root is None:
        return out
    q = deque([root])
    while q:
        node = q.popleft()
        out.append(node.value)
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return out


def demo_traversals() -> None:
    banner("DEMO 1 --- Tree traversals: pre, in, post, level")

    #            F
    #          /   \
    #         B     G
    #        / \     \
    #       A   D     I
    #          / \   /
    #         C   E H
    root = TreeNode(
        "F",
        left=TreeNode("B",
                      left=TreeNode("A"),
                      right=TreeNode("D",
                                     left=TreeNode("C"),
                                     right=TreeNode("E"))),
        right=TreeNode("G",
                       right=TreeNode("I",
                                      left=TreeNode("H"))),
    )

    print("Tree:")
    print("            F")
    print("          /   \\")
    print("         B     G")
    print("        / \\     \\")
    print("       A   D     I")
    print("          / \\   /")
    print("         C   E H")
    print()

    pre = []
    inn = []
    post = []
    preorder(root, pre)
    inorder(root, inn)
    postorder(root, post)
    lvl = level_order(root)

    print(f"  pre-order   : {pre}")
    print(f"  in-order    : {inn}")
    print(f"  post-order  : {post}")
    print(f"  level-order : {lvl}")
    print()
    print("Notice: in-order on this tree gives A B C D E F G H I --- a sorted")
    print("walk, because the tree happens to satisfy the BST invariant.")


# ---------------------------------------------------------------------------
# Demo 2 --- binary search tree
# ---------------------------------------------------------------------------

class BSTNode:
    __slots__ = ("value", "left", "right")

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


def bst_insert(root, value):
    if root is None:
        return BSTNode(value)
    if value < root.value:
        root.left = bst_insert(root.left, value)
    elif value > root.value:
        root.right = bst_insert(root.right, value)
    return root


def bst_search(root, value):
    while root is not None:
        if value == root.value:
            return root
        root = root.left if value < root.value else root.right
    return None


def bst_inorder(root, out):
    if root is None:
        return
    bst_inorder(root.left, out)
    out.append(root.value)
    bst_inorder(root.right, out)


def demo_bst() -> None:
    banner("DEMO 2 --- Binary search tree")

    random.seed(42)
    values = random.sample(range(1, 100), 12)
    print(f"Insertion order: {values}")

    root = None
    for v in values:
        root = bst_insert(root, v)

    out = []
    bst_inorder(root, out)
    print(f"In-order walk  : {out}")
    print("(In-order on a BST always produces sorted output.)")
    print()

    # Search demo
    needles = [values[3], values[7], 999]
    for n in needles:
        found = bst_search(root, n)
        if found is None:
            print(f"  search({n:>3}) -> not found")
        else:
            print(f"  search({n:>3}) -> hit  (node value={found.value})")


# ---------------------------------------------------------------------------
# Demo 3 --- decision tree (depth 3) on a toy loan dataset
# ---------------------------------------------------------------------------

LOAN_DATA = [
    # (age, income, prior_default, label)
    (22, 18_000,  False, "RISKY"),
    (24, 25_000,  True,  "RISKY"),
    (27, 45_000,  False, "SAFE"),
    (29, 30_000,  False, "RISKY"),
    (30, 60_000,  False, "SAFE"),
    (31, 55_000,  False, "SAFE"),
    (33, 70_000,  False, "SAFE"),
    (34, 40_000,  True,  "RISKY"),
    (35, 90_000,  False, "SAFE"),
    (38, 80_000,  False, "SAFE"),
    (40, 50_000,  False, "SAFE"),
    (42, 65_000,  True,  "RISKY"),
    (45, 95_000,  False, "SAFE"),
    (48, 30_000,  True,  "RISKY"),
    (52,120_000,  False, "SAFE"),
    (55, 70_000,  True,  "RISKY"),
    (58, 85_000,  False, "SAFE"),
    (61,110_000,  False, "SAFE"),
]

FEATURES = ["age", "income", "prior_default"]


def gini(rows):
    if not rows:
        return 0.0
    n = len(rows)
    counts = {}
    for *_, label in rows:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def split_rows(rows, feature_idx, threshold, is_categorical):
    if is_categorical:
        left = [r for r in rows if r[feature_idx] == threshold]
        right = [r for r in rows if r[feature_idx] != threshold]
    else:
        left = [r for r in rows if r[feature_idx] <= threshold]
        right = [r for r in rows if r[feature_idx] > threshold]
    return left, right


def best_split(rows):
    parent_gini = gini(rows)
    best = None
    best_gain = 0.0
    n = len(rows)

    for feature_idx, name in enumerate(FEATURES):
        is_categorical = name == "prior_default"
        if is_categorical:
            candidates = [True, False]
        else:
            values = sorted(set(r[feature_idx] for r in rows))
            candidates = [(a + b) / 2 for a, b in zip(values, values[1:])]

        for threshold in candidates:
            left, right = split_rows(rows, feature_idx, threshold,
                                     is_categorical)
            if not left or not right:
                continue
            weighted = (len(left) / n) * gini(left) + \
                       (len(right) / n) * gini(right)
            gain = parent_gini - weighted
            if gain > best_gain:
                best_gain = gain
                best = (feature_idx, name, threshold, is_categorical,
                        weighted, left, right)
    return best, parent_gini, best_gain


def majority_label(rows):
    counts = {}
    for *_, label in rows:
        counts[label] = counts.get(label, 0) + 1
    label, count = max(counts.items(), key=lambda kv: kv[1])
    purity = count / len(rows)
    return label, purity


def build_tree(rows, depth, max_depth, indent=""):
    label, purity = majority_label(rows)
    if depth == max_depth or len(rows) <= 2 or purity == 1.0:
        print(f"{indent}-> predict {label}  (n={len(rows)}, purity={purity:.2f})")
        return

    split, parent_gini, gain = best_split(rows)
    if split is None:
        print(f"{indent}-> predict {label}  (n={len(rows)}, no split)")
        return

    feature_idx, name, threshold, is_categorical, weighted, left, right = split
    if is_categorical:
        rule = f"{name} == {threshold}"
    else:
        rule = f"{name} <= {threshold:g}"

    print(f"{indent}split on {rule}  "
          f"(Gini {parent_gini:.3f} -> {weighted:.3f}, gain {gain:.3f})")
    print(f"{indent}  Left  ({rule}):")
    build_tree(left, depth + 1, max_depth, indent + "    ")
    if is_categorical:
        opp_rule = f"{name} != {threshold}"
    else:
        opp_rule = f"{name} > {threshold:g}"
    print(f"{indent}  Right ({opp_rule}):")
    build_tree(right, depth + 1, max_depth, indent + "    ")


def demo_decision_tree() -> None:
    banner("DEMO 3 --- Decision tree (depth 3) by hand on toy loan data")

    print(f"Training rows: {len(LOAN_DATA)}")
    n_safe = sum(1 for *_, l in LOAN_DATA if l == "SAFE")
    n_risky = len(LOAN_DATA) - n_safe
    print(f"  SAFE  : {n_safe}")
    print(f"  RISKY : {n_risky}")
    print(f"  Root Gini: {gini(LOAN_DATA):.3f}")
    print()
    print("Building decision tree (depth 3):")
    print()
    build_tree(LOAN_DATA, depth=0, max_depth=3)


# ---------------------------------------------------------------------------
# Demo 4 --- KD-tree nearest neighbour vs brute force
# ---------------------------------------------------------------------------

class KDNode:
    __slots__ = ("point", "axis", "left", "right")

    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


def build_kdtree(points, depth=0):
    if not points:
        return None
    k = len(points[0])
    axis = depth % k
    points.sort(key=lambda p: p[axis])
    median = len(points) // 2
    return KDNode(
        point=points[median],
        axis=axis,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1),
    )


def squared_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kdtree_nearest(node, query, best=None):
    if node is None:
        return best

    dist = squared_distance(query, node.point)
    if best is None or dist < best[0]:
        best = (dist, node.point)

    diff = query[node.axis] - node.point[node.axis]
    near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)

    best = kdtree_nearest(near, query, best)
    if diff * diff < best[0]:
        best = kdtree_nearest(far, query, best)
    return best


def brute_nearest(points, query):
    best_dist = math.inf
    best_pt = None
    for p in points:
        d = squared_distance(query, p)
        if d < best_dist:
            best_dist = d
            best_pt = p
    return best_dist, best_pt


def demo_kdtree() -> None:
    banner("DEMO 4 --- KD-tree nearest neighbour vs brute force")

    random.seed(2024)
    N = 10_000
    Q = 200
    points = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(N)]
    queries = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(Q)]

    print(f"Points : {N:,} in 2D")
    print(f"Queries: {Q}")
    print()

    start = perf_counter()
    tree = build_kdtree(list(points))
    build_time = perf_counter() - start

    start = perf_counter()
    kd_results = [kdtree_nearest(tree, q) for q in queries]
    kd_time = perf_counter() - start

    start = perf_counter()
    brute_results = [brute_nearest(points, q) for q in queries]
    brute_time = perf_counter() - start

    matches = sum(
        1 for kd, br in zip(kd_results, brute_results)
        if kd[1] == br[1]
    )

    print(f"  KD-tree build       : {build_time * 1000:8.2f} ms (one-off)")
    print(f"  KD-tree {Q} queries : {kd_time * 1000:8.2f} ms")
    print(f"  Brute  {Q} queries  : {brute_time * 1000:8.2f} ms")
    speedup = brute_time / kd_time if kd_time > 0 else float("inf")
    print(f"  KD-tree speedup     : {speedup:8.1f}x")
    print(f"  Agreement with brute force: {matches}/{Q}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo_traversals()
    demo_bst()
    demo_decision_tree()
    demo_kdtree()
    print()


if __name__ == "__main__":
    main()
