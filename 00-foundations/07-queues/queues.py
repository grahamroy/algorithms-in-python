"""
queues.py — companion code for "Queues" (Foundations, Part 7).

Three demos:
  1. FIFO queue with collections.deque, timed against list.pop(0).
  2. Ring buffer with deque(maxlen=...) — a rolling window over a stream.
  3. Priority queue with heapq — min-heap by default, max-heap via negation.

Pure stdlib. Runs in well under a second.
"""

from collections import deque
from time import perf_counter
import heapq
import random


SEPARATOR = "=" * 64


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Demo 1 — FIFO queue: deque vs list
# ---------------------------------------------------------------------------

def demo_fifo_queue() -> None:
    banner("DEMO 1 — FIFO queue: deque vs list.pop(0)")

    q = deque()
    for name in ["alpha", "bravo", "charlie", "delta", "echo"]:
        q.append(name)
    print(f"After five appends:     {list(q)}")

    first = q.popleft()
    second = q.popleft()
    print(f"popleft() -> {first!r}")
    print(f"popleft() -> {second!r}")
    print(f"Remaining queue:        {list(q)}")
    print()

    N = 100_000

    data_list = list(range(N))
    start = perf_counter()
    while data_list:
        data_list.pop(0)
    list_time = perf_counter() - start

    data_deque = deque(range(N))
    start = perf_counter()
    while data_deque:
        data_deque.popleft()
    deque_time = perf_counter() - start

    speedup = list_time / deque_time if deque_time > 0 else float("inf")
    print(f"Draining {N:,} items:")
    print(f"  list.pop(0)      : {list_time * 1000:8.2f} ms")
    print(f"  deque.popleft()  : {deque_time * 1000:8.2f} ms")
    print(f"  deque speedup    : {speedup:8.1f}x")


# ---------------------------------------------------------------------------
# Demo 2 — Ring buffer with deque(maxlen=...)
# ---------------------------------------------------------------------------

def demo_ring_buffer() -> None:
    banner("DEMO 2 — Ring buffer: rolling window of the last 5 readings")

    window = deque(maxlen=5)
    random.seed(7)

    readings = [round(20 + random.gauss(0, 1.5), 2) for _ in range(10)]
    print(f"Incoming stream:        {readings}")
    print()
    print("Streaming readings through a deque(maxlen=5):")
    print()
    print(f"  step  incoming    window                                   avg")
    for step, value in enumerate(readings, start=1):
        window.append(value)
        avg = sum(window) / len(window)
        window_str = str(list(window))
        print(f"  {step:>4}  {value:>8.2f}   {window_str:<40}  {avg:6.2f}")

    print()
    print("Notice how the oldest reading is silently evicted once the")
    print("window is full — no bookkeeping, no index juggling.")


# ---------------------------------------------------------------------------
# Demo 3 — Priority queue with heapq
# ---------------------------------------------------------------------------

def demo_priority_queue() -> None:
    banner("DEMO 3 — Priority queue with heapq")

    jobs = [
        (5, 1, "send weekly newsletter"),
        (1, 2, "page on-call: DB master down"),
        (3, 3, "rebuild nightly embeddings"),
        (2, 4, "rotate TLS certificate"),
        (4, 5, "compress last week's logs"),
        (1, 6, "page on-call: 500s spiking"),
    ]

    heap = []
    for priority, job_id, payload in jobs:
        heapq.heappush(heap, (priority, job_id, payload))

    print("Scheduled jobs (priority, job_id, payload):")
    for job in jobs:
        print(f"  {job}")
    print()
    print("Popping in priority order (ties broken by job_id):")
    while heap:
        priority, job_id, payload = heapq.heappop(heap)
        print(f"  priority={priority}  job_id={job_id}  -> {payload}")

    print()
    print("heapq is a MIN-heap. For a max-heap, negate the priority on")
    print("push and negate again on pop:")
    print()

    scores = [("model-A", 0.71), ("model-B", 0.83),
              ("model-C", 0.66), ("model-D", 0.91)]
    max_heap = []
    for name, score in scores:
        heapq.heappush(max_heap, (-score, name))

    print("  Popping models by descending score:")
    while max_heap:
        neg_score, name = heapq.heappop(max_heap)
        print(f"    {name}  score={-neg_score:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo_fifo_queue()
    demo_ring_buffer()
    demo_priority_queue()
    print()


if __name__ == "__main__":
    main()
