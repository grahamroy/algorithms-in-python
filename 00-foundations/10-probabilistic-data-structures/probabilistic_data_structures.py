"""
probabilistic_data_structures.py --- companion code for "Probabilistic
Data Structures" (Foundations, Part 10).

Three demos:
  1. Bloom filter: insert 1M items, measure observed vs theoretical
     false-positive rate.
  2. Count-Min Sketch: stream a Zipfian word distribution, recover the
     top-10 heavy hitters, compare estimated counts to true counts.
  3. HyperLogLog: estimate the cardinality of 1M distinct strings using
     ~1 KB of memory.

Pure stdlib. Runs in well under a second.
"""

from collections import Counter
import hashlib
import math
import random
import struct


SEPARATOR = "=" * 64


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


def hash_with_seed(item: str, seed: int) -> int:
    """A simple seeded hash using SHA-256 truncated to 64 bits."""
    h = hashlib.sha256(f"{seed}:{item}".encode()).digest()
    return struct.unpack(">Q", h[:8])[0]


# ---------------------------------------------------------------------------
# Demo 1 --- Bloom filter
# ---------------------------------------------------------------------------

class BloomFilter:
    """Bit-array Bloom filter with k SHA-256-derived hash functions."""

    def __init__(self, expected_items: int, false_positive_rate: float):
        # Optimal m and k for the target false positive rate
        self.m = int(math.ceil(
            -expected_items * math.log(false_positive_rate)
            / (math.log(2) ** 2)
        ))
        self.k = max(1, int(round((self.m / expected_items) * math.log(2))))
        # Use a bytearray as the bit array (1 bit per array slot)
        self.bits = bytearray((self.m + 7) // 8)
        self.n_inserted = 0

    def _positions(self, item: str):
        # Use double hashing: derive k hash values from two independent SHA-256s
        h1 = hash_with_seed(item, 0)
        h2 = hash_with_seed(item, 1)
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item: str) -> None:
        for pos in self._positions(item):
            self.bits[pos >> 3] |= (1 << (pos & 7))
        self.n_inserted += 1

    def __contains__(self, item: str) -> bool:
        for pos in self._positions(item):
            if not (self.bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def memory_bytes(self) -> int:
        return len(self.bits)

    def theoretical_fpr(self) -> float:
        return (1.0 - math.exp(-self.k * self.n_inserted / self.m)) ** self.k


def demo_bloom_filter() -> None:
    banner("DEMO 1 --- Bloom filter: 1M items at 1% FPR")

    target_fpr = 0.01
    n_items = 1_000_000
    bf = BloomFilter(n_items, target_fpr)
    print(f"Bloom filter --- m={bf.m:,} bits ({bf.memory_bytes() / 1024 / 1024:.2f} MB), "
          f"k={bf.k} hashes")
    print()

    rng = random.Random(42)
    inserted = set()
    print(f"Inserting {n_items:,} unique strings...")
    while len(inserted) < n_items:
        s = f"item-{rng.randrange(0, 10 ** 9)}"
        if s not in inserted:
            inserted.add(s)
            bf.add(s)

    print(f"Querying 100,000 strings that were NOT inserted...")
    n_queries = 100_000
    false_positives = 0
    queried = 0
    while queried < n_queries:
        s = f"unseen-{rng.randrange(0, 10 ** 9)}"
        if s in inserted:
            continue
        queried += 1
        if s in bf:
            false_positives += 1

    observed_fpr = false_positives / n_queries
    theoretical = bf.theoretical_fpr()
    print()
    print(f"  False positives observed : {false_positives:,}")
    print(f"  False positive rate (FPR): {observed_fpr:.5f}")
    print(f"  Theoretical FPR          : {theoretical:.5f}")
    print(f"  Ratio (observed/theory)  : {observed_fpr / theoretical:.2f}x")


# ---------------------------------------------------------------------------
# Demo 2 --- Count-Min Sketch
# ---------------------------------------------------------------------------

class CountMinSketch:
    """Count-Min Sketch with depth d, width w."""

    def __init__(self, width: int, depth: int):
        self.w = width
        self.d = depth
        self.table = [[0] * width for _ in range(depth)]

    def add(self, item: str, count: int = 1) -> None:
        for i in range(self.d):
            pos = hash_with_seed(item, i) % self.w
            self.table[i][pos] += count

    def estimate(self, item: str) -> int:
        return min(
            self.table[i][hash_with_seed(item, i) % self.w]
            for i in range(self.d)
        )

    def memory_bytes(self) -> int:
        # Assume each counter is a 4-byte int (Python ints are larger but
        # the underlying allocation is comparable for the demo)
        return self.w * self.d * 4


def zipfian_stream(vocab_size: int, n: int, alpha: float, seed: int):
    """Generate a Zipfian-distributed stream of items from a vocabulary."""
    rng = random.Random(seed)
    # Precompute zipf weights
    weights = [1.0 / (i ** alpha) for i in range(1, vocab_size + 1)]
    total = sum(weights)
    probs = [w / total for w in weights]
    cum = []
    s = 0.0
    for p in probs:
        s += p
        cum.append(s)

    # English-ish vocab for the top words; synthetic words for the tail
    real_words = [
        "the", "of", "and", "to", "in", "a", "is", "that", "for", "it",
        "as", "was", "with", "be", "by", "on", "not", "this", "are", "or",
    ]
    vocab = list(real_words)
    while len(vocab) < vocab_size:
        vocab.append(f"word_{len(vocab):05d}")

    out = []
    for _ in range(n):
        u = rng.random()
        # Linear scan over cum (vocab_size is small enough for the demo)
        for idx, c in enumerate(cum):
            if u <= c:
                out.append(vocab[idx])
                break
    return out


def demo_count_min_sketch() -> None:
    banner("DEMO 2 --- Count-Min Sketch: top-k over a Zipfian stream")

    cms = CountMinSketch(width=2718, depth=5)
    print(f"Count-Min Sketch --- width={cms.w}, depth={cms.d} "
          f"(~{cms.memory_bytes() / 1024:.0f} KB)")
    print()

    n = 100_000
    vocab_size = 5_000
    alpha = 1.2
    print(f"Streaming {n:,} words drawn from a Zipfian (alpha={alpha}) over")
    print(f"a vocabulary of {vocab_size:,} distinct words.")
    print()

    stream = zipfian_stream(vocab_size, n, alpha, seed=7)

    # Insert into the sketch and also keep an exact counter for comparison
    true_counts = Counter()
    for item in stream:
        cms.add(item)
        true_counts[item] += 1

    print("Top-10 heavy hitters (true vs CMS-estimated):")
    print()
    print(f"  {'rank':<5} {'word':<14} {'true_count':>12} "
          f"{'cms_estimate':>14} {'over_count':>12}")
    for rank, (word, true_n) in enumerate(true_counts.most_common(10), start=1):
        est = cms.estimate(word)
        over = est - true_n
        print(f"  {rank:<5} {word:<14} {true_n:>12,} {est:>14,} {over:>12,}")


# ---------------------------------------------------------------------------
# Demo 3 --- HyperLogLog
# ---------------------------------------------------------------------------

class HyperLogLog:
    """HyperLogLog with m = 2^p registers."""

    def __init__(self, p: int = 10):
        # p in [4, 16] for sensible accuracy/memory tradeoffs
        if not 4 <= p <= 16:
            raise ValueError("p must be in [4, 16]")
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        # Bias correction constants from the HLL paper
        if self.m == 16:
            self.alpha_m = 0.673
        elif self.m == 32:
            self.alpha_m = 0.697
        elif self.m == 64:
            self.alpha_m = 0.709
        else:
            self.alpha_m = 0.7213 / (1 + 1.079 / self.m)

    def add(self, item: str) -> None:
        h = hash_with_seed(item, 0)
        # Use the top p bits as the register index, the remaining bits to
        # count leading zeros
        idx = h >> (64 - self.p)
        # Remaining 64-p bits: count the leading zeros within them
        w = (h << self.p) & ((1 << 64) - 1)
        # Run length of leading zeros (1-indexed, max 64 - p + 1)
        if w == 0:
            run = 64 - self.p + 1
        else:
            run = 1
            mask = 1 << 63
            while not (w & mask):
                run += 1
                mask >>= 1
        if run > self.registers[idx]:
            self.registers[idx] = run

    def estimate(self) -> float:
        # Raw estimator
        z = sum(2 ** -r for r in self.registers)
        raw = self.alpha_m * (self.m ** 2) / z

        # Small-range correction: linear counting if many registers are 0
        if raw <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros != 0:
                return self.m * math.log(self.m / zeros)
        return raw

    def memory_bytes(self) -> int:
        # Each register stores up to ~64 -- 6 bits is enough; we allocate
        # one byte each in this Python list for simplicity
        return self.m

    def expected_error(self) -> float:
        return 1.04 / math.sqrt(self.m)


def demo_hyperloglog() -> None:
    banner("DEMO 3 --- HyperLogLog: 1M distinct items in ~1 KB")

    hll = HyperLogLog(p=10)
    err = hll.expected_error()
    print(f"HyperLogLog --- m={hll.m} registers (~{hll.memory_bytes()} bytes), "
          f"expected error ~{err * 100:.2f}%")
    print()

    n = 1_000_000
    print(f"Streaming {n:,} distinct strings...")

    rng = random.Random(2024)
    seen = set()
    while len(seen) < n:
        s = f"u{rng.randrange(0, 10 ** 12)}"
        if s in seen:
            continue
        seen.add(s)
        hll.add(s)

    estimate = hll.estimate()
    abs_err = abs(estimate - n)
    rel_err = abs_err / n
    print()
    print(f"  True cardinality   : {n:>10,}")
    print(f"  HLL estimate       : {estimate:>10,.0f}")
    print(f"  Error              : {abs_err:>10,.0f}  ({rel_err * 100:.2f}%)")
    print(f"  Memory footprint   : {hll.memory_bytes() / 1024:>10.2f} KB")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo_bloom_filter()
    demo_count_min_sketch()
    demo_hyperloglog()
    print()


if __name__ == "__main__":
    main()
