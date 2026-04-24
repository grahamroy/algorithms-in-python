"""
knowledge_graphs.py --- companion code for "Knowledge Graphs" (Foundations, Part 9).

Three demos:
  1. Triple store: build a small KG, run 1-hop pattern queries.
  2. Multi-hop traversal: 2-hop pattern matching with shared variables.
  3. TransE embedding: train h + r ~= t on the same KG with stdlib only,
     and verify the geometric property after training.

Pure stdlib. Runs in under a second.
"""

from collections import defaultdict
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
# A tiny knowledge graph about scientists, places, and discoveries.
# Every line is a (subject, predicate, object) triple.
# ---------------------------------------------------------------------------

TRIPLES = [
    # Scientists and their birthplaces
    ("marie_curie",    "born_in",    "warsaw"),
    ("pierre_curie",   "born_in",    "paris"),
    ("albert_einstein","born_in",    "ulm"),
    ("isaac_newton",   "born_in",    "woolsthorpe"),
    ("alan_turing",    "born_in",    "london"),
    ("ada_lovelace",   "born_in",    "london"),
    ("charles_darwin", "born_in",    "shrewsbury"),

    # Cities -> countries
    ("warsaw",         "located_in", "poland"),
    ("paris",          "located_in", "france"),
    ("ulm",            "located_in", "germany"),
    ("woolsthorpe",    "located_in", "england"),
    ("london",         "located_in", "england"),
    ("shrewsbury",     "located_in", "england"),

    # Discoveries / contributions
    ("marie_curie",    "discovered", "polonium"),
    ("marie_curie",    "discovered", "radium"),
    ("pierre_curie",   "discovered", "radium"),
    ("isaac_newton",   "formulated", "law_of_gravitation"),
    ("isaac_newton",   "formulated", "calculus"),
    ("albert_einstein","formulated", "general_relativity"),
    ("albert_einstein","formulated", "special_relativity"),
    ("alan_turing",    "formulated", "turing_machine"),
    ("ada_lovelace",   "formulated", "first_algorithm"),
    ("charles_darwin", "formulated", "natural_selection"),

    # Naming / origin chains
    ("polonium",       "named_after","poland"),

    # Awards
    ("marie_curie",    "won",        "nobel_physics"),
    ("marie_curie",    "won",        "nobel_chemistry"),
    ("pierre_curie",   "won",        "nobel_physics"),
    ("albert_einstein","won",        "nobel_physics"),

    # Type assertions
    ("marie_curie",    "is_a",       "scientist"),
    ("pierre_curie",   "is_a",       "scientist"),
    ("albert_einstein","is_a",       "scientist"),
    ("isaac_newton",   "is_a",       "scientist"),
    ("alan_turing",    "is_a",       "scientist"),
    ("ada_lovelace",   "is_a",       "scientist"),
    ("charles_darwin", "is_a",       "scientist"),
]


# ---------------------------------------------------------------------------
# Demo 1 --- a triple store with pattern matching
# ---------------------------------------------------------------------------

class TripleStore:
    """Tiny in-memory triple store with a single SPO index.

    Pattern matching uses None as a wildcard:
        store.match(("marie_curie", "discovered", None))
    """

    def __init__(self, triples=()):
        self.triples = set()
        # Indexes for the three positions
        self._by_s = defaultdict(set)
        self._by_p = defaultdict(set)
        self._by_o = defaultdict(set)
        for t in triples:
            self.add(t)

    def add(self, triple):
        s, p, o = triple
        self.triples.add(triple)
        self._by_s[s].add(triple)
        self._by_p[p].add(triple)
        self._by_o[o].add(triple)

    def match(self, pattern):
        s, p, o = pattern
        # Use the most selective bound position for the index lookup
        if s is not None and p is not None and o is not None:
            return [pattern] if pattern in self.triples else []
        candidates = None
        if s is not None:
            candidates = self._by_s.get(s, set())
        if p is not None:
            cs = self._by_p.get(p, set())
            candidates = cs if candidates is None else candidates & cs
        if o is not None:
            cs = self._by_o.get(o, set())
            candidates = cs if candidates is None else candidates & cs
        if candidates is None:
            candidates = self.triples
        return list(candidates)


def demo_triple_store() -> None:
    banner("DEMO 1 --- Triple store with 1-hop pattern queries")

    store = TripleStore(TRIPLES)
    print(f"Loaded {len(store.triples)} triples about "
          f"{len(store._by_s)} subjects.")
    print()

    # Q1: What did Marie Curie discover?
    print('QUERY 1: "What did Marie Curie discover?"')
    print('  pattern: (marie_curie, discovered, ?)')
    matches = store.match(("marie_curie", "discovered", None))
    for s, p, o in sorted(matches):
        print(f"  ->  {o}")
    print()

    # Q2: Who won a Nobel in physics?
    print('QUERY 2: "Who won the Nobel in physics?"')
    print('  pattern: (?, won, nobel_physics)')
    matches = store.match((None, "won", "nobel_physics"))
    for s, p, o in sorted(matches):
        print(f"  ->  {s}")
    print()

    # Q3: Where was Alan Turing born?
    print('QUERY 3: "Where was Alan Turing born?"')
    print('  pattern: (alan_turing, born_in, ?)')
    matches = store.match(("alan_turing", "born_in", None))
    for s, p, o in matches:
        print(f"  ->  {o}")


# ---------------------------------------------------------------------------
# Demo 2 --- multi-hop pattern matching with shared variables
# ---------------------------------------------------------------------------

def two_hop_query(store, pat1, pat2, shared_pos1, shared_pos2):
    """
    Run a 2-hop conjunctive query: find bindings that satisfy BOTH patterns,
    where shared_pos1/shared_pos2 indicate which positions in pat1/pat2 must
    bind to the same value.
    """
    results = []
    matches1 = store.match(pat1)
    for m1 in matches1:
        shared_value = m1[shared_pos1]
        # Construct a refined version of pat2 with the shared variable bound
        bound_pat2 = list(pat2)
        bound_pat2[shared_pos2] = shared_value
        for m2 in store.match(tuple(bound_pat2)):
            results.append((m1, m2))
    return results


def demo_two_hop() -> None:
    banner("DEMO 2 --- 2-hop traversal: scientists born in Polish cities")

    store = TripleStore(TRIPLES)

    # Pattern A: (?scientist, born_in, ?city)
    # Pattern B: (?city,      located_in, poland)
    # Shared variable: ?city
    #   - in pat1, ?city is at position 2 (object)
    #   - in pat2, ?city is at position 0 (subject)
    print('QUERY: "Find all scientists born in cities located in Poland."')
    print('  pat1: (?scientist, born_in, ?city)')
    print('  pat2: (?city,      located_in, poland)')
    print('  shared: ?city')
    print()

    pairs = two_hop_query(
        store,
        pat1=(None, "born_in",   None),
        pat2=(None, "located_in","poland"),
        shared_pos1=2,
        shared_pos2=0,
    )
    if not pairs:
        print("  (no matches)")
    for (s, p, o), (s2, p2, o2) in pairs:
        print(f"  ->  scientist={s:<14} born_in {o:<10} (located_in {o2})")

    print()
    print("In SPARQL the same query is:")
    print("  SELECT ?scientist ?city WHERE {")
    print("    ?scientist :born_in     ?city .")
    print("    ?city      :located_in  :poland .")
    print("  }")


# ---------------------------------------------------------------------------
# Demo 3 --- TransE embedding training (pure stdlib)
# ---------------------------------------------------------------------------
# h + r ~= t for every true triple. We learn entity and relation vectors
# of dimension `dim` by SGD on a margin loss against a corrupted-tail
# negative for each positive.

def vec_add(a, b):  return [x + y for x, y in zip(a, b)]
def vec_sub(a, b):  return [x - y for x, y in zip(a, b)]
def vec_scale(a, c): return [x * c for x in a]
def vec_norm(a):    return math.sqrt(sum(x * x for x in a))

def vec_normalize(a):
    n = vec_norm(a)
    return a if n == 0 else [x / n for x in a]

def l2_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def random_unit_vector(dim, rng):
    v = [rng.uniform(-1, 1) for _ in range(dim)]
    return vec_normalize(v)


def train_transe(triples, dim=8, epochs=200, lr=0.05, margin=1.0, seed=42):
    rng = random.Random(seed)

    entities = sorted({s for s, _, _ in triples} | {o for _, _, o in triples})
    relations = sorted({p for _, p, _ in triples})

    E = {e: random_unit_vector(dim, rng) for e in entities}
    R = {r: random_unit_vector(dim, rng) for r in relations}

    triple_list = list(triples)

    for epoch in range(epochs):
        rng.shuffle(triple_list)
        for h, r, t in triple_list:
            # Sample a negative tail uniformly from other entities
            t_neg = t
            while t_neg == t:
                t_neg = rng.choice(entities)

            h_v, r_v, t_v, t_neg_v = E[h], R[r], E[t], E[t_neg]

            # Predicted vectors
            pos_pred = vec_add(h_v, r_v)
            d_pos = l2_distance(pos_pred, t_v)
            d_neg = l2_distance(pos_pred, t_neg_v)
            loss_value = margin + d_pos - d_neg

            if loss_value <= 0:
                continue  # margin satisfied

            # Gradients of d_pos = ||h+r-t|| with respect to h, r, t.
            # d/dh = (h+r-t)/d_pos     (and similarly for r, t with sign flip)
            # We update all four embeddings.
            if d_pos > 1e-8:
                grad_pos = vec_scale(vec_sub(pos_pred, t_v), 1.0 / d_pos)
            else:
                grad_pos = [0.0] * dim
            if d_neg > 1e-8:
                grad_neg = vec_scale(vec_sub(pos_pred, t_neg_v), 1.0 / d_neg)
            else:
                grad_neg = [0.0] * dim

            grad_h = vec_sub(grad_pos, grad_neg)
            grad_r = vec_sub(grad_pos, grad_neg)
            grad_t = vec_scale(grad_pos, -1.0)
            grad_t_neg = vec_scale(grad_neg, 1.0)

            E[h]     = vec_normalize(vec_sub(h_v,    vec_scale(grad_h,     lr)))
            R[r]     =                vec_sub(r_v,    vec_scale(grad_r,     lr))
            E[t]     = vec_normalize(vec_sub(t_v,    vec_scale(grad_t,     lr)))
            E[t_neg] = vec_normalize(vec_sub(t_neg_v,vec_scale(grad_t_neg, lr)))

    return E, R, entities


def rank_of_true_tail(E, entities, predicted, true_tail):
    """Rank (1-indexed) of true_tail among entities sorted by L2 distance to predicted."""
    distances = [(e, l2_distance(predicted, E[e])) for e in entities]
    distances.sort(key=lambda kv: kv[1])
    for i, (e, _) in enumerate(distances, start=1):
        if e == true_tail:
            return i, distances
    return len(entities), distances


def demo_transe() -> None:
    banner("DEMO 3 --- TransE embeddings: h + r ~= t")

    print(f"Triples: {len(TRIPLES)}")
    print("Training TransE with dim=8, 200 epochs, margin=1.0...")
    E, R, entities = train_transe(TRIPLES, dim=8, epochs=200,
                                  lr=0.05, margin=1.0, seed=42)
    print(f"Learned {len(E)} entity vectors and {len(R)} relation vectors.")
    print()

    queries = [
        ("marie_curie",    "born_in",    "warsaw"),
        ("pierre_curie",   "born_in",    "paris"),
        ("alan_turing",    "born_in",    "london"),
        ("marie_curie",    "discovered", "polonium"),
        ("isaac_newton",   "formulated", "calculus"),
        ("warsaw",         "located_in", "poland"),
    ]

    print("After training, h + r should land near t. Closest entities to the")
    print("predicted tail vector for each query:")
    print()
    print(f"  {'query':<46} {'true tail':<22} rank")

    for h, r, t in queries:
        predicted = vec_add(E[h], R[r])
        rank, ranking = rank_of_true_tail(E, entities, predicted, t)
        top1 = ranking[0][0]
        marker = "OK" if top1 == t else f"top1={top1}"
        print(f"  {h+' + '+r:<46} {t:<22} {rank:>3}  ({marker})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo_triple_store()
    demo_two_hop()
    demo_transe()
    print()


if __name__ == "__main__":
    main()
