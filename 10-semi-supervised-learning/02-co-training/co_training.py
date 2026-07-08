"""
co_training.py --- companion code for "Co-Training"
(Semi-Supervised Learning, Part 2).

Self-training (Part 1) has a structural weakness: it grades its own homework.
One model promotes its own confident guesses to training labels, so a
confident MISTAKE is retrained on as truth and compounds -- confirmation bias
with no outside check.

Co-training (Blum & Mitchell, 1998) adds the outside check. It assumes each
example has TWO VIEWS -- two feature sets, each sufficient to classify on its
own. The original paper classified university web pages using (view A) the
words on the page and (view B) the words in links POINTING at the page.
Then:

    1. Train one classifier per view, on the labelled set.
    2. Each classifier nominates the unlabelled examples it is most
       confident about -- labelling them FOR THE SHARED POOL, which
       means for the OTHER model too.
    3. If the two classifiers nominate the same example with DIFFERENT
       labels, the nomination is VETOED -- that disagreement is a
       built-in error detector self-training simply does not have.
    4. Retrain both on the grown pool and repeat.

Why it works: the views fail INDEPENDENTLY. Where view A is ambiguous, view B
is usually not (and vice versa), so each model hands the other exactly the
labels the other could not have safely produced itself -- and a wrong label
now has to fool two independent judges at once.

The data here makes the assumption literal: each example's two views are two
INDEPENDENT draws of the two-moons picture given the same class --
conditionally independent given the label, by construction. The base
classifier per view is the same k-NN (k=3, unanimous votes) as Part 1.

Demonstrates:
  1. The setting: 10 labels, two half-informed views, and the baselines.
  2. The loop: nominations, vetoes, and the combined model reaching the
     oracle -- with far fewer wrong pseudo-labels than self-training.
  3. The comparison and the catch: five seeds of co-training vs
     self-training on the concatenated features; then view B replaced by a
     near-copy of view A -- vetoes drop to zero and the gains vanish.

Everything is plain NumPy. Dependencies: numpy. Runs in ~10-20 seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
NOISE = 0.15
K = 3
TAU = 1.0            # unanimity of the k=3 neighbour votes


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Data: each example is a class y plus TWO views. Each view is an independent
# draw of a two-moons position for that class -- so the views share the label
# and nothing else (conditional independence, by construction).
# ---------------------------------------------------------------------------

def moons_view(y, noise, rng):
    t = rng.uniform(0, np.pi, len(y))
    X = np.where(y[:, None] == 0,
                 np.stack([np.cos(t), np.sin(t)], axis=1),
                 np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1))
    return X + rng.normal(0, noise, (len(y), 2))


def make_data(n, noise, rng, duplicate_view=False):
    y = np.arange(n) % 2
    rng.shuffle(y)
    A = moons_view(y, noise, rng)
    if duplicate_view:
        B = A + rng.normal(0, 0.02, A.shape)     # a near-copy: no new info
    else:
        B = moons_view(y, noise, rng)            # an independent second view
    return A, B, y


# ---------------------------------------------------------------------------
# The base classifier per view: k-NN with vote-fraction confidence (Part 1).
# ---------------------------------------------------------------------------

def knn_proba(X_train, y_train, X_query, k=K):
    d = ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :k]
    p1 = y_train[idx].mean(axis=1)
    return np.stack([1 - p1, p1], axis=1)


def acc_of(proba, y):
    return float((proba.argmax(axis=1) == y).mean())


# ---------------------------------------------------------------------------
# Self-training on the CONCATENATED views: the one-judge alternative,
# identical machinery, for the head-to-head.
# ---------------------------------------------------------------------------

def self_train_concat(C, y, lab_idx, unl_idx, C_test, y_test,
                      tau=TAU, max_rounds=60):
    XL, YL = C[lab_idx].copy(), y[lab_idx].copy()
    P, PT = C[unl_idx].copy(), y[unl_idx].copy()
    wrong = 0
    for _ in range(max_rounds):
        if len(P) == 0:
            break
        pr = knn_proba(XL, YL, P)
        take = pr.max(axis=1) >= tau
        if not take.any():
            break
        pred = pr.argmax(axis=1)
        wrong += int(np.sum(pred[take] != PT[take]))
        XL = np.concatenate([XL, P[take]])
        YL = np.concatenate([YL, pred[take]])
        P, PT = P[~take], PT[~take]
    return acc_of(knn_proba(XL, YL, C_test), y_test), wrong


# ---------------------------------------------------------------------------
# Co-training: two views, nominations, vetoes, a shared pool.
# ---------------------------------------------------------------------------

def co_train(A, B, y, lab_idx, unl_idx, A_test, B_test, y_test,
             tau=TAU, max_rounds=60, record=False):
    lab = list(lab_idx)
    labY = list(y[lab_idx])
    pool = list(unl_idx)
    wrong = 0
    vetoes = 0
    history = []

    def combined_acc():
        L, LY = np.array(lab), np.array(labY)
        p = (knn_proba(A[L], LY, A_test) + knn_proba(B[L], LY, B_test)) / 2
        return acc_of(p, y_test)

    for rnd in range(1, max_rounds + 1):
        if not pool:
            break
        Pi = np.array(pool)
        L, LY = np.array(lab), np.array(labY)
        prA = knn_proba(A[L], LY, A[Pi])
        prB = knn_proba(B[L], LY, B[Pi])
        nomA = prA.max(axis=1) >= tau
        nomB = prB.max(axis=1) >= tau
        pA = prA.argmax(axis=1)
        pB = prB.argmax(axis=1)
        conflict = nomA & nomB & (pA != pB)
        vetoes += int(conflict.sum())
        take = (nomA | nomB) & ~conflict
        if not take.any():
            break
        new_labels = np.where(nomA[take], pA[take], pB[take])
        wrong += int(np.sum(new_labels != y[Pi[take]]))
        for i, ix in enumerate(Pi[take]):
            lab.append(ix)
            labY.append(new_labels[i])
        pool = [ix for ix, t in zip(pool, take) if not t]
        if record:
            history.append((rnd, int(nomA.sum()), int(nomB.sum()),
                            int(conflict.sum()), int(take.sum()),
                            wrong, combined_acc()))
    return combined_acc(), wrong, vetoes, history


# ---------------------------------------------------------------------------
# One full experiment (shared by all demos)
# ---------------------------------------------------------------------------

def setup(seed, n=500, noise=NOISE, n_per_class=5, duplicate_view=False):
    rng = np.random.default_rng(seed)
    A, B, y = make_data(n, noise, rng, duplicate_view)
    yt = np.arange(500) % 2
    rng.shuffle(yt)
    A_test = moons_view(yt, noise, rng)
    if duplicate_view:
        B_test = A_test + rng.normal(0, 0.02, A_test.shape)
    else:
        B_test = moons_view(yt, noise, rng)
    lab_idx = np.concatenate([
        rng.choice(np.where(y == 0)[0], n_per_class, replace=False),
        rng.choice(np.where(y == 1)[0], n_per_class, replace=False)])
    unl = np.ones(n, dtype=bool)
    unl[lab_idx] = False
    return A, B, y, A_test, B_test, yt, lab_idx, np.where(unl)[0]


def main() -> None:
    A, B, y, At, Bt, yt, li, ui = setup(RNG_SEED)
    C, Ct = np.concatenate([A, B], 1), np.concatenate([At, Bt], 1)

    banner("DEMO 1 --- The setting: 10 labels, two half-informed views")
    print("  Every example has TWO views -- two independent 2-D snapshots that")
    print("  share only the class (like a web page's words and its inbound")
    print("  links). 10 examples are labelled; 490 are not.")
    print()
    accA = acc_of(knn_proba(A[li], y[li], At), yt)
    accB = acc_of(knn_proba(B[li], y[li], Bt), yt)
    accC = acc_of(knn_proba(C[li], y[li], Ct), yt)
    oracle = acc_of(knn_proba(C, y, Ct), yt)
    print(f"  View A alone, 10 labels        : {accA:6.1%}")
    print(f"  View B alone, 10 labels        : {accB:6.1%}")
    print(f"  Both views concatenated        : {accC:6.1%}")
    print(f"  Oracle (all 500 true labels)   : {oracle:6.1%}")

    banner("DEMO 2 --- The loop: nominations, vetoes, and the shared pool")
    print("  Each round, each view's classifier nominates its unanimous points")
    print("  for the shared pool. Nominations that DISAGREE are vetoed -- the")
    print("  built-in error check. Wrong counts audit the hidden truth.")
    print()
    ct_acc, ct_wrong, ct_vetoes, hist = co_train(
        A, B, y, li, ui, At, Bt, yt, record=True)
    print("    round   nom A   nom B   vetoed   added   wrong so far   accuracy")
    for rnd, nA, nB, veto, add, wr, a in hist:
        print(f"      {rnd:2d}    {nA:5d}   {nB:5d}    {veto:4d}    {add:5d}"
              f"        {wr:4d}          {a:6.1%}")
    print()
    print(f"  Co-training: {accC:.1%} (10 labels) -> {ct_acc:.1%}   "
          f"[oracle {oracle:.1%}]")
    print(f"  {ct_vetoes} conflicting nominations vetoed; only {ct_wrong} wrong"
          f" labels slipped through.")

    banner("DEMO 3 --- Two judges vs one -- and why the views must differ")
    print("  Head-to-head on five datasets: co-training vs self-training on the")
    print("  concatenated features (same k-NN, same threshold, same budget).")
    print()
    print("    seed   self-train (wrong)    co-train (wrong)    vetoes")
    for sd in range(5):
        A2, B2, y2, At2, Bt2, yt2, li2, ui2 = setup(sd)
        C2, Ct2 = np.concatenate([A2, B2], 1), np.concatenate([At2, Bt2], 1)
        st_a, st_w = self_train_concat(C2, y2, li2, ui2, Ct2, yt2)
        ct_a, ct_w, ct_v, _ = co_train(A2, B2, y2, li2, ui2, At2, Bt2, yt2)
        print(f"      {sd}     {st_a:6.1%}  ({st_w:3d})       "
              f"{ct_a:6.1%}  ({ct_w:3d})        {ct_v:3d}")
    print()
    print("  Now break the assumption: view B becomes a near-copy of view A --")
    print("  two judges with identical blind spots:")
    print()
    print("    seed   self-train (wrong)    co-train (wrong)    vetoes")
    for sd in range(3):
        A2, B2, y2, At2, Bt2, yt2, li2, ui2 = setup(sd, duplicate_view=True)
        C2, Ct2 = np.concatenate([A2, B2], 1), np.concatenate([At2, Bt2], 1)
        st_a, st_w = self_train_concat(C2, y2, li2, ui2, Ct2, yt2)
        ct_a, ct_w, ct_v, _ = co_train(A2, B2, y2, li2, ui2, At2, Bt2, yt2)
        print(f"      {sd}     {st_a:6.1%}  ({st_w:3d})       "
              f"{ct_a:6.1%}  ({ct_w:3d})        {ct_v:3d}")
    print()
    print("  Identical judges never disagree: zero vetoes on every seed, so the")
    print("  error check is gone and co-training collapses into self-training")
    print("  (two of the three seeds match its wrong-label count exactly). The")
    print("  power was never the second model; it was the second, INDEPENDENT")
    print("  view.")


if __name__ == "__main__":
    main()
