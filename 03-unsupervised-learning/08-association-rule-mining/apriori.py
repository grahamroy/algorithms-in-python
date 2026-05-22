"""
apriori.py --- companion code for "Association Rule Mining"
(Unsupervised Learning, Part 8).

Three demos:
  1. Apriori from scratch on a 50-transaction grocery dataset.
     Level-by-level candidate generation with the Apriori-property
     pruning. Find all frequent itemsets at min_support = 0.20.
  2. Generate association rules from the frequent itemsets, sorted
     by lift, with min_confidence = 0.60.
  3. Same data through mlxtend's apriori + association_rules for
     a sanity check.

Dependencies: numpy, pandas, mlxtend.
Install with: pip install mlxtend
Runs in well under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from itertools import combinations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Synthetic grocery dataset (50 transactions, 10 items)
# ---------------------------------------------------------------------------

ITEMS = ["bread", "milk", "butter", "eggs", "jam",
         "beer", "chips", "soda", "cheese", "yogurt"]


def make_dataset(seed=RNG_SEED):
    """Synthetic transactions with realistic co-occurrence:
    - bread/milk/eggs are everyday staples
    - jam strongly co-occurs with bread+butter (toast)
    - beer/chips/soda strongly co-occur (snack basket)
    - cheese/yogurt are mid-frequency add-ons
    """
    rng = np.random.default_rng(seed)
    transactions = []

    def maybe(p):
        return rng.random() < p

    for _ in range(50):
        t = set()
        # Staples
        if maybe(0.70):
            t.add("bread")
        if maybe(0.55):
            t.add("milk")
        if maybe(0.55):
            t.add("eggs")
        # Toast basket: jam-people almost always buy bread + butter
        if maybe(0.30):
            t.add("jam")
            t.add("bread")
            t.add("butter")
        elif maybe(0.20):
            # Plain butter without jam (sometimes)
            t.add("butter")
        # Snack basket: beer-people almost always buy chips + soda
        if maybe(0.35):
            t.add("beer")
            if maybe(0.85):
                t.add("chips")
            if maybe(0.80):
                t.add("soda")
        elif maybe(0.20):
            # Sometimes chips/soda without beer
            if maybe(0.5):
                t.add("chips")
            else:
                t.add("soda")
        # Independent add-ons
        if maybe(0.25):
            t.add("cheese")
        if maybe(0.30):
            t.add("yogurt")
        if not t:
            t.add(ITEMS[int(rng.integers(0, len(ITEMS)))])
        transactions.append(t)
    return transactions


# ---------------------------------------------------------------------------
# Apriori from scratch
# ---------------------------------------------------------------------------

def apriori_from_scratch(transactions, min_support):
    """Return dict mapping frozenset(itemset) -> support."""
    n = len(transactions)
    min_count = min_support * n

    # Level 1: single-item support
    item_counts = {}
    for t in transactions:
        for item in t:
            item_counts[frozenset([item])] = item_counts.get(
                frozenset([item]), 0) + 1
    L = [set(k for k, c in item_counts.items() if c >= min_count)]
    supports = {k: c / n for k, c in item_counts.items()
                if c >= min_count}

    k = 2
    while L[-1]:
        # Candidate generation: join L_{k-1} with itself
        prev = sorted(L[-1], key=lambda s: tuple(sorted(s)))
        candidates = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                u = prev[i] | prev[j]
                if len(u) == k:
                    candidates.add(u)
        # Prune: every (k-1)-subset of c must be in L_{k-1}
        prev_set = L[-1]
        candidates = {c for c in candidates
                      if all(frozenset(s) in prev_set
                             for s in combinations(c, k - 1))}
        # Count support in one pass
        counts = {c: 0 for c in candidates}
        for t in transactions:
            for c in candidates:
                if c.issubset(t):
                    counts[c] += 1
        L_k = {c for c, cnt in counts.items() if cnt >= min_count}
        for c in L_k:
            supports[c] = counts[c] / n
        L.append(L_k)
        k += 1
    return supports


def generate_rules(supports, min_confidence):
    """From frequent itemsets, generate rules with their
    confidence and lift."""
    rules = []
    for itemset, sup in supports.items():
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for r in range(1, len(items)):
            for A in combinations(items, r):
                A = frozenset(A)
                B = itemset - A
                conf = sup / supports[A]
                if conf >= min_confidence:
                    lift = conf / supports[B]
                    rules.append({
                        "antecedent": A,
                        "consequent": B,
                        "support": sup,
                        "confidence": conf,
                        "lift": lift,
                    })
    return rules


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def fmt_set(s):
    return "{" + ", ".join(sorted(s)) + "}"


def demo_from_scratch(transactions):
    banner("DEMO 1 --- Apriori from scratch on a small grocery dataset")

    min_support = 0.20
    print(f"  Transactions   : {len(transactions)}")
    print(f"  Items          : {len(ITEMS)}")
    print(f"  min_support    : {min_support}  "
          f"(must appear in >={int(min_support * len(transactions))} "
          f"transactions)")

    supports = apriori_from_scratch(transactions, min_support)
    print(f"  Frequent itemsets discovered: {len(supports)}")

    top = sorted(supports.items(), key=lambda kv: -kv[1])[:5]
    print(f"  Examples (top 5 by support):")
    for itemset, s in top:
        print(f"    {fmt_set(itemset):<26}  support={s:.3f}")
    return supports


def demo_rules(supports):
    banner("DEMO 2 --- Rule generation (min_confidence=0.6, "
           "sorted by lift)")

    rules = generate_rules(supports, min_confidence=0.6)
    rules.sort(key=lambda r: -r["lift"])

    print(f"  {'antecedent':<15}  ->  {'consequent':<16}  "
          f"{'conf':>6}  {'lift':>6}")
    print(f"  {'-' * 15:<15}      {'-' * 14:<16}  "
          f"{'-' * 6:>6}  {'-' * 6:>6}")
    for r in rules[:5]:
        a = fmt_set(r["antecedent"])
        b = fmt_set(r["consequent"])
        print(f"  {a:<15}  ->  {b:<16}  "
              f"{r['confidence']:>6.3f}  {r['lift']:>6.2f}")
    return rules


def demo_mlxtend(transactions, scratch_supports, scratch_rules):
    banner("DEMO 3 --- Same data, mlxtend Apriori + association_rules")

    te = TransactionEncoder()
    one_hot = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(one_hot, columns=te.columns_)

    freq = mlxtend_apriori(df, min_support=0.20, use_colnames=True)
    rules = association_rules(freq, metric="confidence",
                              min_threshold=0.6)
    print(f"  Frequent itemsets discovered: {len(freq)}   "
          f"(from-scratch: {len(scratch_supports)})")
    print(f"  Rules with confidence >= 0.6:  {len(rules)}   "
          f"(from-scratch: {len(scratch_rules)})")

    if len(rules):
        top = rules.sort_values("lift", ascending=False).iloc[0]
        a = fmt_set(top["antecedents"])
        b = fmt_set(top["consequents"])
        print(f"  Top rule by lift: {a} -> {b}  "
              f"(lift = {top['lift']:.2f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    transactions = make_dataset()
    supports = demo_from_scratch(transactions)
    rules = demo_rules(supports)
    demo_mlxtend(transactions, supports, rules)
    print()


if __name__ == "__main__":
    main()
