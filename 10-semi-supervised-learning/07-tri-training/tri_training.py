"""
tri_training.py --- companion code for "Tri-Training"
(Semi-Supervised Learning, Part 7).

Co-training (Part 2) earned its error check from two independent VIEWS -- and
most datasets don't have two views. Tri-training (Zhou & Li, 2005) wins a
similar check with no views at all:

    1. Train THREE classifiers, diversified by bootstrap resamples of the
       labelled set (an unstable base learner makes the bootstraps differ).
    2. For each classifier i, the OTHER TWO act as its teachers: every
       unlabelled point they AGREE on becomes a pseudo-label for i.
    3. THE GATE: before classifier i accepts a batch, estimate its teachers'
       error e_i on the labelled set (where they agree), and only accept if
       the noise-rate bound improves -- e_i must fall, and e_i * |batch|
       must shrink relative to last round (subsampling the batch if needed).
    4. Predict by majority vote of the three.

The gate is the algorithmic heart. Pseudo-label noise is survivable if it
shrinks fast enough relative to the batch size (the classic Angluin-Laird
noise bound); the gate enforces exactly that, round by round, using only
quantities you can measure. Without it, agreement alone floods the training
sets just like self-training did.

Demonstrates (two moons, Part 1's exact data and 8-label draws):
  1. The mechanism: per-round teacher-error estimates and gated batches.
  2. The scoreboard on Part 1's five catastrophic draws: self-training 73.8%
     -> bagged supervised baseline 80.0% -> tri-training 86.2% mean, with an
     honest look at the one draw it loses.
  3. The gate matters: remove it (accept every agreement batch) and the mean
     drops -- and the pseudo-label volume explodes.

Everything is plain NumPy. Dependencies: numpy. Runs in ~10-20 seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
K = 1                     # 1-NN: unstable, so bootstraps genuinely differ


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Part 1's exact data pipeline
# ---------------------------------------------------------------------------

def make_moons(n_per_class, noise, rng):
    t = rng.uniform(0, np.pi, n_per_class)
    top = np.stack([np.cos(t), np.sin(t)], axis=1)
    t = rng.uniform(0, np.pi, n_per_class)
    bottom = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.concatenate([top, bottom])
    X += rng.normal(0, noise, X.shape)
    y = np.concatenate([np.zeros(n_per_class, dtype=int),
                        np.ones(n_per_class, dtype=int)])
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def knn_pred(X_train, y_train, X_query, k=K):
    d = ((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d, axis=1)[:, :min(k, len(X_train))]
    return (np.take(y_train, idx).mean(axis=1) > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Tri-training (Zhou & Li, 2005), with the error gate
# ---------------------------------------------------------------------------

def tri_train(X_lab, y_lab, X_unl, rng, gate=True, max_rounds=20,
              verbose=False):
    n_l = len(X_lab)
    models = []
    for _ in range(3):
        bs = rng.integers(0, n_l, n_l)          # bootstrap resample
        models.append((X_lab[bs].copy(), y_lab[bs].copy()))
    e_prev = [0.5] * 3
    l_prev = [0] * 3
    total_added = 0

    for rnd in range(1, max_rounds + 1):
        updates = [False] * 3
        new_batch = [None] * 3
        new_e = [0.0] * 3
        for i in range(3):
            j, k2 = [m for m in range(3) if m != i]
            # teachers' error, measured where they agree on the labelled set
            pj = knn_pred(*models[j], X_lab)
            pk = knn_pred(*models[k2], X_lab)
            agree_l = pj == pk
            if agree_l.sum() == 0:
                continue
            e_i = float(np.sum(agree_l & (pj != y_lab))) / float(agree_l.sum())
            # teachers' agreement on the unlabelled pool
            pj_u = knn_pred(*models[j], X_unl)
            pk_u = knn_pred(*models[k2], X_unl)
            batch = np.where(pj_u == pk_u)[0]
            if not gate:
                if len(batch):
                    updates[i] = True
                    new_batch[i] = (X_unl[batch], pj_u[batch])
                    new_e[i] = e_i
                continue
            # ---- the gate: accept only if the noise bound improves ----
            if e_i >= e_prev[i] or len(batch) == 0:
                continue
            if l_prev[i] == 0:
                l_prev[i] = int(np.floor(e_i / (e_prev[i] - e_i) + 1))
            li = len(batch)
            if l_prev[i] < li:
                if e_i * li < e_prev[i] * l_prev[i]:
                    updates[i] = True
                elif l_prev[i] > e_i / (e_prev[i] - e_i):
                    size = int(np.ceil(e_prev[i] * l_prev[i] / e_i - 1))
                    batch = rng.choice(batch, min(size, li), replace=False)
                    updates[i] = True
            if updates[i]:
                new_batch[i] = (X_unl[batch], pj_u[batch])
                new_e[i] = e_i
        if not any(updates):
            break
        for i in range(3):
            if updates[i]:
                Xb, yb = new_batch[i]
                models[i] = (np.concatenate([X_lab, Xb]),
                             np.concatenate([y_lab, yb]))
                e_prev[i] = new_e[i]
                l_prev[i] = len(Xb)
                total_added += len(Xb)
                if verbose:
                    print(f"    round {rnd}: classifier {i} accepts "
                          f"{len(Xb):3d} pseudo-labels  "
                          f"(teachers' error estimate {new_e[i]:.3f})")
        if not gate and rnd >= 4:
            break

    def predict(Xq):
        votes = np.stack([knn_pred(*models[m], Xq) for m in range(3)])
        return (votes.mean(axis=0) > 0.5).astype(int)
    return predict, total_added


def bagged_baseline(X_lab, y_lab, rng):
    n_l = len(X_lab)
    models = []
    for _ in range(3):
        bs = rng.integers(0, n_l, n_l)
        models.append((X_lab[bs], y_lab[bs]))

    def predict(Xq):
        votes = np.stack([knn_pred(*models[m], Xq) for m in range(3)])
        return (votes.mean(axis=0) > 0.5).astype(int)
    return predict


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_moons(250, noise=0.15, rng=rng)
    X_test, y_test = make_moons(250, noise=0.15, rng=rng)

    def draw_labels(draw):
        d_rng = np.random.default_rng(draw)
        return np.concatenate([
            d_rng.choice(np.where(y == 0)[0], 4, replace=False),
            d_rng.choice(np.where(y == 1)[0], 4, replace=False)])

    banner("DEMO 1 --- The mechanism: two teachers, one student, one gate")
    print("  Three 1-NN classifiers on bootstraps of the 8 labels. Each round,")
    print("  every classifier is taught by the points its two peers AGREE on --")
    print("  if and only if the peers' measured error keeps the noise bound")
    print("  shrinking (batches are subsampled to enforce it).")
    print()
    li = draw_labels(0)
    unl = np.ones(len(X), dtype=bool)
    unl[li] = False
    predict, added = tri_train(X[li], y[li], X[unl],
                               np.random.default_rng(100), verbose=True)
    acc = float((predict(X_test) == y_test).mean())
    base = bagged_baseline(X[li], y[li], np.random.default_rng(100))
    base_acc = float((base(X_test) == y_test).mean())
    print()
    print(f"  Majority vote of the three: {acc:.1%} test accuracy")
    print(f"  (the same three bootstraps WITHOUT unlabelled data: {base_acc:.1%})")

    banner("DEMO 2 --- The scoreboard: Part 1's five hard draws, revisited")
    print("  Same data, same five random 8-label draws that broke self-training.")
    print("  'Bagged baseline' = the same three bootstrap 1-NNs, labels only.")
    print()
    part1 = {0: 85.4, 1: 59.6, 2: 51.6, 3: 87.0, 4: 85.2}
    print("    draw   self-training (Part 1)   bagged baseline   tri-training")
    accs, bases = [], []
    for draw in range(5):
        li = draw_labels(draw)
        unl = np.ones(len(X), dtype=bool)
        unl[li] = False
        base = bagged_baseline(X[li], y[li], np.random.default_rng(100 + draw))
        b = float((base(X_test) == y_test).mean())
        predict, _ = tri_train(X[li], y[li], X[unl],
                               np.random.default_rng(100 + draw))
        a = float((predict(X_test) == y_test).mean())
        accs.append(a)
        bases.append(b)
        print(f"      {draw}          {part1[draw]:5.1f}%               "
              f"{b:6.1%}          {a:6.1%}")
    print(f"     mean          73.8%               {np.mean(bases):6.1%}"
          f"          {np.mean(accs):6.1%}")
    print()
    print("  Tri-training lifts the mean by 6 points over its own supervised")
    print("  baseline and by 12 over Part 1's lone self-trainer -- but look at")
    print("  draw 1 honestly: it LOSES to its baseline there. The gate's error")
    print("  estimate comes from 8 labelled points, and eight points make a")
    print("  noisy voltmeter: sometimes it waves through a bad batch. Three")
    print("  judges shrink the risk of confident nonsense; they do not end it.")

    banner("DEMO 3 --- The gate is the point: remove it and watch")
    print("  Same three judges, same agreement rule -- but accept EVERY agreed")
    print("  batch, no error check, no subsampling:")
    print()
    print("    draw   tri (gated)   tri (no gate)      pseudo-labels used")
    for draw in range(5):
        li = draw_labels(draw)
        unl = np.ones(len(X), dtype=bool)
        unl[li] = False
        pg, ng = tri_train(X[li], y[li], X[unl],
                           np.random.default_rng(100 + draw), gate=True)
        pu, nu = tri_train(X[li], y[li], X[unl],
                           np.random.default_rng(100 + draw), gate=False)
        ag = float((pg(X_test) == y_test).mean())
        au = float((pu(X_test) == y_test).mean())
        print(f"      {draw}      {ag:6.1%}       {au:6.1%}          "
              f"{ng:5d} gated vs {nu:5d} ungated")
    print()
    print("  Ungated, every classifier swallows nearly the whole pool every")
    print("  round -- five thousand pseudo-labels with no quality control. The")
    print("  gate admits roughly a fifth as many, only when the measured noise")
    print("  bound improves, and wins on average (86.2% vs 83.0%) -- though the")
    print("  flood gets lucky on two draws. Agreement finds candidate labels;")
    print("  the gate decides whether they are safe to eat.")


if __name__ == "__main__":
    main()
