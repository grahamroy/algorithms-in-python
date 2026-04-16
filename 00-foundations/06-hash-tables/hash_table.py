"""
Hash Tables — How Python Finds Things in Constant Time
Algorithms in Python — Foundations, Part 6

Demonstrates a hash table built from scratch using separate chaining,
automatic resizing when the load factor exceeds a threshold, and a
small collision stress test comparing a sensible hash function against
a deliberately terrible one that forces every key into a single bucket.

Pure standard library only.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")


# =============================================================================
# Part 1 — HashTable (separate chaining with automatic resizing)
# =============================================================================

class HashTable:
    """
    A hash table that maps hashable keys to arbitrary values.

    Collisions are handled by separate chaining: each slot in the
    underlying array holds a Python list of (key, value) tuples that
    all hashed to the same bucket. When the load factor exceeds the
    resize threshold we double the slot count and rehash every entry.

    Time complexity on average:
        put    : O(1) amortised
        get    : O(1)
        delete : O(1)

    Worst case (pathological hashing) degrades to O(n) for all three.
    """

    INITIAL_SIZE = 8
    RESIZE_THRESHOLD = 0.75
    GROWTH_FACTOR = 2

    def __init__(self):
        self._size = self.INITIAL_SIZE
        self._count = 0
        self._slots = [[] for _ in range(self._size)]

    # -- hashing ------------------------------------------------------------

    def _hash(self, key):
        """
        Map a key to a slot index. We delegate the hard work to Python's
        built-in hash() and reduce into [0, size) with modulo. The
        abs() handles the fact that hash() can return negative ints.
        """
        return abs(hash(key)) % self._size

    # -- core operations ----------------------------------------------------

    def put(self, key, value):
        """
        Insert or update a key. If the key already exists its value is
        overwritten. After insertion, resize if the load factor has
        crossed the threshold.
        """
        index = self._hash(key)
        bucket = self._slots[index]
        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._count += 1

        if self.load_factor() > self.RESIZE_THRESHOLD:
            new_size = self._size * self.GROWTH_FACTOR
            print(f"    [Resizing from {self._size} to {new_size} slots, "
                  f"load factor was {self.load_factor():.3f}]")
            self._resize(new_size)

    def get(self, key):
        """
        Retrieve a value by key. Raises KeyError if the key is absent,
        matching the behaviour of Python's built-in dict.
        """
        index = self._hash(key)
        for existing_key, value in self._slots[index]:
            if existing_key == key:
                return value
        raise KeyError(key)

    def delete(self, key):
        """Remove a key and return its value. Raises KeyError if absent."""
        index = self._hash(key)
        bucket = self._slots[index]
        for i, (existing_key, value) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                self._count -= 1
                return value
        raise KeyError(key)

    # -- resizing -----------------------------------------------------------

    def _resize(self, new_size):
        """
        Allocate a new slot array of the given size and rehash every
        entry. This is O(n) but only runs occasionally, so the
        amortised cost per insert stays O(1).
        """
        old_slots = self._slots
        self._size = new_size
        self._slots = [[] for _ in range(new_size)]
        self._count = 0
        for bucket in old_slots:
            for key, value in bucket:
                # Re-insert via put() so hashing uses the new size. We
                # bypass the resize check because we are mid-resize.
                index = self._hash(key)
                self._slots[index].append((key, value))
                self._count += 1

    # -- introspection ------------------------------------------------------

    def load_factor(self):
        return self._count / self._size

    def __len__(self):
        return self._count

    def __contains__(self, key):
        index = self._hash(key)
        for existing_key, _ in self._slots[index]:
            if existing_key == key:
                return True
        return False

    def __iter__(self):
        for bucket in self._slots:
            for key, _ in bucket:
                yield key

    def items(self):
        for bucket in self._slots:
            for key, value in bucket:
                yield key, value

    def __repr__(self):
        return (f"HashTable(size={self._size}, count={self._count}, "
                f"load_factor={self.load_factor():.3f})")

    def bucket_sizes(self):
        """Return a list of bucket lengths — useful for inspecting collisions."""
        return [len(bucket) for bucket in self._slots]


# =============================================================================
# Part 2 — A deliberately pathological hash table for the stress test
# =============================================================================

class BadHashTable(HashTable):
    """
    Same table, but with a hash function that returns 0 for every key.
    Every insert lands in slot 0, turning the hash table into a single
    long list. Lookups degrade to O(n) — exactly the worst case.
    """

    def _hash(self, key):
        return 0


# =============================================================================
# Demo 1 — Inserts, retrievals, deletions, and a live resize
# =============================================================================

def demo_basic():
    print("=" * 64)
    print("DEMO 1 — Basic operations and automatic resize")
    print("=" * 64)

    table = HashTable()
    print(f"\nInitial state: {table}")

    # Twenty (key, value) pairs: a mix of string and integer keys.
    pairs = [
        ("alpha", 1),  ("beta", 2),  ("gamma", 3),  ("delta", 4),
        ("epsilon", 5),  ("zeta", 6),  ("eta", 7),  ("theta", 8),
        ("iota", 9),  ("kappa", 10),
        (101, "one-oh-one"),  (202, "two-oh-two"),  (303, "three-oh-three"),
        (404, "four-oh-four"),  (505, "five-oh-five"),  (606, "six-oh-six"),
        (707, "seven-oh-seven"),  (808, "eight-oh-eight"),
        (909, "nine-oh-nine"),  (1010, "ten-ten"),
    ]

    print(f"\nInserting {len(pairs)} key-value pairs...\n")
    for key, value in pairs:
        table.put(key, value)
    print(f"\nFinal state: {table}")

    print("\nBucket occupancy (zero = empty slot):")
    print(f"  {table.bucket_sizes()}")

    # Retrieve a handful of keys.
    print("\nRetrievals:")
    for key in ["alpha", "eta", 404, 1010]:
        print(f"  get({key!r}) -> {table.get(key)!r}")

    # Membership test.
    print("\nMembership tests:")
    for key in ["alpha", "missing", 404, 999]:
        print(f"  {key!r} in table -> {key in table}")

    # Delete a key.
    print("\nDeleting 'gamma'...")
    deleted = table.delete("gamma")
    print(f"  delete('gamma') returned {deleted!r}")
    print(f"  State after delete: {table}")
    print(f"  'gamma' in table -> {'gamma' in table}")

    # Update an existing key (value overwrite — count unchanged).
    print("\nUpdating 'alpha' from 1 to 99...")
    table.put("alpha", 99)
    print(f"  State after update: {table}")
    print(f"  get('alpha') -> {table.get('alpha')}")

    # Iterate a few keys.
    print("\nFirst eight keys via __iter__:")
    keys = list(table)
    print(f"  {keys[:8]}")


# =============================================================================
# Demo 2 — Collision stress test: good hash vs bad hash
# =============================================================================

def demo_collisions():
    print("\n" + "=" * 64)
    print("DEMO 2 — Collision stress test: good hash vs bad hash")
    print("=" * 64)

    n = 2000
    keys = [f"user_{i}" for i in range(n)]

    # Populate both tables with the same data.
    good = HashTableQuiet()
    bad = BadHashTableQuiet()
    for i, key in enumerate(keys):
        good.put(key, i)
        bad.put(key, i)

    # Time n lookups on each.
    t0 = time.perf_counter()
    for key in keys:
        good.get(key)
    good_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    for key in keys:
        bad.get(key)
    bad_time = time.perf_counter() - t0

    print(f"\nInserted {n} keys into each table.")
    print(f"\nGood hash  -> {good}")
    print(f"            lookup time for {n} keys: {good_time*1000:.2f} ms")
    print(f"\nBad hash   -> {bad}")
    print(f"            lookup time for {n} keys: {bad_time*1000:.2f} ms")

    # Slowdown ratio.
    if good_time > 0:
        ratio = bad_time / good_time
        print(f"\nBad hash is {ratio:,.0f}x slower than the good hash.")
    print("\nEvery key landed in the same bucket, so every lookup scans")
    print("a list of length n. That is the worst case a hash table has.")

    # Look at bucket-size distribution for the good hash.
    sizes = good.bucket_sizes()
    occupied = sum(1 for s in sizes if s > 0)
    max_bucket = max(sizes)
    avg_bucket = sum(sizes) / len(sizes)
    print(f"\nGood-hash bucket distribution:")
    print(f"  total slots      : {len(sizes)}")
    print(f"  occupied slots   : {occupied}")
    print(f"  max bucket size  : {max_bucket}")
    print(f"  mean bucket size : {avg_bucket:.2f}")


# =============================================================================
# Quiet variants — same classes, but without the resize print() chatter
# that would spam the collision demo with hundreds of lines.
# =============================================================================

class HashTableQuiet(HashTable):
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


class BadHashTableQuiet(HashTableQuiet):
    def _hash(self, key):
        return 0


# =============================================================================
# Demo 3 — Peek at Python's built-in hash() to show what it returns
# =============================================================================

def demo_builtin_hash():
    print("\n" + "=" * 64)
    print("DEMO 3 — A quick look at Python's built-in hash()")
    print("=" * 64)

    samples = [
        0, 1, 42, -1, 2**60,
        "hello", "world", "hello!",
        (1, 2, 3), (1, 2, 4),
        3.14, True, None,
    ]
    print("\nhash(x) for a handful of values:")
    for x in samples:
        print(f"  hash({x!r:>15}) = {hash(x)}")

    # Demonstrate the avalanche-ish behaviour for strings.
    print("\nOne-character change, completely different hash:")
    for s in ["password", "Password", "passw0rd", "password!"]:
        print(f"  hash({s!r:>12}) = {hash(s)}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    demo_basic()
    demo_collisions()
    demo_builtin_hash()
    print("\n" + "=" * 64)
    print("All demos complete.")
    print("=" * 64)
