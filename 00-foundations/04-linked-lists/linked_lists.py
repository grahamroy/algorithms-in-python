"""
Linked Lists — When Arrays Aren't Enough
Algorithms in Python — Foundations, Part 4

Demonstrates singly and doubly linked lists built from scratch: core
operations, reversal, a simple LRU cache, an undo stack, and a head-
of-list performance comparison against Python lists.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# Part 1 — Singly linked list from scratch
# =============================================================================

class Node:
    """A single node in a singly linked list."""

    def __init__(self, data):
        self.data = data
        self.next = None

    def __repr__(self):
        return f"Node({self.data})"


class LinkedList:
    """Singly linked list with core operations."""

    def __init__(self):
        self.head = None
        self.size = 0

    # --- Append: add to the end (O(n) — must walk to tail) ---
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

    # --- Prepend: add to the front (O(1)) ---
    def prepend(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    # --- Insert at index (O(n) walk + O(1) splice) ---
    def insert_at(self, index, data):
        if index < 0 or index > self.size:
            raise IndexError(f"Index {index} out of range for size {self.size}")
        if index == 0:
            self.prepend(data)
            return
        new_node = Node(data)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        new_node.next = current.next
        current.next = new_node
        self.size += 1

    # --- Delete first occurrence of a value (O(n)) ---
    def delete(self, data):
        if self.head is None:
            raise ValueError("List is empty")
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
        raise ValueError(f"{data} not found in list")

    # --- Search: return True if value exists (O(n)) ---
    def search(self, data):
        current = self.head
        while current:
            if current.data == data:
                return True
            current = current.next
        return False

    # --- Reverse the list in place (O(n)) ---
    def reverse(self):
        previous = None
        current = self.head
        while current:
            next_node = current.next
            current.next = previous
            previous = current
            current = next_node
        self.head = previous

    # --- Display: return a readable string ---
    def display(self):
        parts = []
        current = self.head
        while current:
            parts.append(str(current.data))
            current = current.next
        return " -> ".join(parts) + " -> None"

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"LinkedList([{self.display()}])"


# --- Demonstrate the singly linked list ---

print("=" * 60)
print("SINGLY LINKED LIST — Core Operations")
print("=" * 60)

ll = LinkedList()
for val in [10, 20, 30, 40]:
    ll.append(val)
print(f"\nAfter appending 10, 20, 30, 40:")
print(f"  {ll.display()}")

ll.prepend(5)
print(f"\nAfter prepending 5:")
print(f"  {ll.display()}")

ll.insert_at(2, 15)
print(f"\nAfter inserting 15 at index 2:")
print(f"  {ll.display()}")

ll.delete(30)
print(f"\nAfter deleting 30:")
print(f"  {ll.display()}")

print(f"\nSearch for 20: {ll.search(20)}")
print(f"Search for 99: {ll.search(99)}")

ll.reverse()
print(f"\nAfter reversing:")
print(f"  {ll.display()}")

print(f"\nList length: {len(ll)}")


# =============================================================================
# Part 2 — Doubly linked list (brief implementation)
# =============================================================================

class DNode:
    """A node in a doubly linked list."""

    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None


class DoublyLinkedList:
    """Doubly linked list — O(1) removal when you have a reference to the node."""

    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, data):
        new_node = DNode(data)
        if self.tail is None:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
        return new_node

    def remove_node(self, node):
        """Remove a specific node in O(1) — no search required."""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        node.prev = node.next = None
        self.size -= 1

    def move_to_tail(self, node):
        """Move an existing node to the tail in O(1)."""
        self.remove_node(node)
        return self.append(node.data)

    def display(self):
        parts = []
        current = self.head
        while current:
            parts.append(str(current.data))
            current = current.next
        return " <-> ".join(parts)

    def __len__(self):
        return self.size


print("\n")
print("=" * 60)
print("DOUBLY LINKED LIST")
print("=" * 60)

dll = DoublyLinkedList()
nodes = []
for val in ["A", "B", "C", "D"]:
    nodes.append(dll.append(val))

print(f"\nAfter appending A, B, C, D:")
print(f"  {dll.display()}")

dll.remove_node(nodes[1])  # remove B
print(f"\nAfter removing node B (O(1) with direct reference):")
print(f"  {dll.display()}")


# =============================================================================
# Part 3 — Practical example: simple LRU cache
# =============================================================================

print("\n")
print("=" * 60)
print("PRACTICAL EXAMPLE — LRU Cache")
print("=" * 60)


class LRUCache:
    """
    Least Recently Used cache using a doubly linked list + hash map.
    - get / put are both O(1)
    - The tail of the list is the most recently used item
    - The head of the list is the least recently used item
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}            # key -> DNode (node.data = (key, value))
        self.list = DoublyLinkedList()

    def get(self, key):
        if key not in self.cache:
            return None
        node = self.cache[key]
        # Move to tail (most recently used)
        new_node = self.list.move_to_tail(node)
        self.cache[key] = new_node
        return new_node.data[1]

    def put(self, key, value):
        if key in self.cache:
            self.list.remove_node(self.cache[key])
        elif len(self.list) >= self.capacity:
            # Evict the least recently used (head)
            lru_node = self.list.head
            del self.cache[lru_node.data[0]]
            self.list.remove_node(lru_node)
        node = self.list.append((key, value))
        self.cache[key] = node

    def display(self):
        parts = []
        current = self.list.head
        while current:
            k, v = current.data
            parts.append(f"{k}:{v}")
            current = current.next
        return "[" + " -> ".join(parts) + "]"


cache = LRUCache(capacity=3)

print(f"\nLRU cache with capacity 3")
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(f"After inserting a=1, b=2, c=3: {cache.display()}")

cache.get("a")
print(f"After accessing 'a':            {cache.display()}")

cache.put("d", 4)
print(f"After inserting d=4 (evicts b): {cache.display()}")

result = cache.get("b")
print(f"Attempt to get evicted 'b':     {result}")


# =============================================================================
# Part 4 — Practical example: undo stack
# =============================================================================

print("\n")
print("=" * 60)
print("PRACTICAL EXAMPLE — Undo Stack")
print("=" * 60)


class UndoStack:
    """
    Simple undo mechanism backed by a singly linked list.
    Each new action is prepended — the head is always the most recent.
    Undo pops from the head in O(1).
    """

    def __init__(self):
        self.stack = LinkedList()

    def do(self, action):
        self.stack.prepend(action)
        print(f"  Action: {action}")

    def undo(self):
        if self.stack.head is None:
            print("  Nothing to undo")
            return None
        action = self.stack.head.data
        self.stack.head = self.stack.head.next
        self.stack.size -= 1
        print(f"  Undo:   {action}")
        return action

    def history(self):
        return self.stack.display()


editor = UndoStack()
print()
editor.do("Type 'Hello'")
editor.do("Type ' World'")
editor.do("Bold text")
print(f"\n  History: {editor.history()}")

editor.undo()
editor.undo()
print(f"\n  History after 2 undos: {editor.history()}")


# =============================================================================
# Part 5 — Performance comparison: insert at head
# =============================================================================

print("\n")
print("=" * 60)
print("PERFORMANCE — Insert at Head: Linked List vs Python List")
print("=" * 60)

sizes = [10_000, 50_000, 100_000]

for n in sizes:
    # Python list: insert at position 0 requires shifting all elements
    py_list = list(range(n))
    start = time.perf_counter()
    for _ in range(1000):
        py_list.insert(0, -1)
    py_time = time.perf_counter() - start

    # Linked list: prepend is O(1), no shifting needed
    ll = LinkedList()
    ll.head = Node(0)  # start with one element to be fair
    ll.size = 1
    start = time.perf_counter()
    for _ in range(1000):
        ll.prepend(-1)
    ll_time = time.perf_counter() - start

    print(f"\n  n = {n:>7,} | 1,000 head inserts")
    print(f"    Python list:  {py_time:.4f}s")
    print(f"    Linked list:  {ll_time:.4f}s")
    print(f"    Speedup:      {py_time / ll_time:.1f}x")

print("\n" + "=" * 60)
print("Key takeaway: linked list prepend is O(1) regardless of size,")
print("while Python list insert(0, x) is O(n) — it must shift every")
print("existing element one position to the right.")
print("=" * 60)
