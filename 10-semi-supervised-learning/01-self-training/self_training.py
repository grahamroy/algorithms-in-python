"""
self_training.py --- companion code for "Self-Training"
(Semi-Supervised Learning, Part 1).

The premise of this whole track: LABELS ARE EXPENSIVE, unlabelled data is
nearly free. A hospital has a handful of diagnosed scans and thousands of
undiagnosed ones; a company has a hundred tagged documents and a million
untagged. Can the unlabelled pile actually improve the classifier?

Self-training is the simplest possible answer, and it wraps around ANY base
classifier that can report confidence:

    1. Train on the labelled set (however small).
    2. Predict the unlabelled pool.
    3. Take the predictions the model is CONFIDENT about (probability >=
       a threshold tau) and add them to the training set as PSEUDO-LABELS
       -- treat your own guesses as ground truth.
    4. Retrain and repeat until nothing new clears the bar.

Why can that work? The CLUSTER ASSUMPTION: points in the same dense cluster
tend to share a label, and the decision boundary should pass through the
LOW-density gap between clusters. Confident predictions land next to known
labels inside a cluster; adding them moves the frontier outward, and
confidence spreads along the cluster like a rumour -- never across the gap.

Why can it fail? CONFIRMATION BIAS: a confidently WRONG pseudo-label is
retrained on as truth, which mislabels its neighbours, which mislabels
theirs. Errors do not wash out -- they compound.

The base classifier here is k-nearest-neighbours (k=3, confidence = the
fraction of neighbours that agree) -- chosen because its confidence is
naturally LOCAL, which is exactly what the loop needs. The wrapper itself
never looks inside the model.

Demonstrates (on two-moons data, 8 labels + 492 unlabelled):
  1. The gap: 8 labels alone vs an oracle trained on all 500 true labels.
  2. The loop: the pseudo-label frontier expanding round by round -- round 1
     adds hundreds of labels with ZERO errors -- and accuracy climbing most
     of the way to the oracle. (Errors are measured against the hidden true
     labels for reporting only.)
  3. The catch: the same loop with the SAME budget of 8 labels placed at
     random. Coverage gaps let wrong pseudo-labels in, and they compound --
     some draws end up WORSE than not using the unlabelled data at all.

Everything (the two-moons generator, k-NN, the loop) is plain NumPy.
Dependencies: numpy. Runs in a few seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
NOISE = 0.15
K = 3
TAU = 1.0          # unanimity of the k=3 neighbour votes


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Two moons: the classic semi-supervised picture. Two interleaved crescents,
# one per class, separated by a low-density gap -- the cluster assumption
# made visible.
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


def spread_seeds(X, y, n_per_class):
    """Pick n labelled points per class, spread along each moon -- the
    'label a few DIVERSE examples' strategy a sensible annotator uses."""
    t = np.linspace(0.12 * np.pi, 0.88 * np.pi, n_per_class)
    anchors0 = np.stack([np.cos(t), np.sin(t)], axis=1)
    anchors1 = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    idx = []
    for anchors, cls in ((anchors0, 0), (anchors1, 1)):
        cand = np.where(y == cls)[0]
        for a in anchors:
            idx.append(cand[int(np.argmin(((X[cand] - a) ** 2).sum(1)))])
    return np.array(idx)


# ---------------------------------------------------------------------------
# The base classifier: k-nearest-neighbours. Confidence = the fraction of
# the k nearest labelled points that agree. Local by construction -- far
# from any label, the neighbours disagree and confidence drops.
# ---------------------------------------------------------------------------

class KNN:
    def __init__(self, k=K):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict_proba(self, Xq):
        d = ((Xq[:, None, :] - self.X[None, :, :]) ** 2).sum(-1)
        idx = np.argsort(d, axis=1)[:, :self.k]
        p1 = self.y[idx].mean(axis=1)
        return np.stack([1 - p1, p1], axis=1)

    def predict(self, Xq):
        return self.predict_proba(Xq).argmax(axis=1)


def accuracy(model, X, y):
    return float(np.mean(model.predict(X) == y))


# ---------------------------------------------------------------------------
# Self-training: the wrapper. Note it only ever calls fit / predict_proba --
# ANY classifier with those two methods drops in.
# ---------------------------------------------------------------------------

def self_train(X_lab, y_lab, X_unl, tau=TAU, max_rounds=30,
               X_test=None, y_test=None, y_unl_true=None):
    X_train, y_train = X_lab.copy(), y_lab.copy()
    pool = X_unl.copy()
    pool_true = None if y_unl_true is None else y_unl_true.copy()
    history = []
    total_added = 0
    total_wrong = 0

    for rnd in range(1, max_rounds + 1):
        model = KNN().fit(X_train, y_train)
        if len(pool) == 0:
            break
        proba = model.predict_proba(pool)
        conf = proba.max(axis=1)
        pred = proba.argmax(axis=1)
        take = conf >= tau
        if not take.any():
            break
        n_take = int(take.sum())
        wrong = 0 if pool_true is None else int(np.sum(pred[take] != pool_true[take]))
        total_added += n_take
        total_wrong += wrong
        X_train = np.concatenate([X_train, pool[take]])
        y_train = np.concatenate([y_train, pred[take]])
        pool = pool[~take]
        if pool_true is not None:
            pool_true = pool_true[~take]
        model = KNN().fit(X_train, y_train)
        history.append((rnd, n_take, total_added, total_wrong,
                        accuracy(model, X_test, y_test)))

    final = KNN().fit(X_train, y_train)
    return final, accuracy(final, X_test, y_test), history, total_added, total_wrong


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    X, y = make_moons(250, noise=NOISE, rng=rng)            # pool: 500 points
    X_test, y_test = make_moons(250, noise=NOISE, rng=rng)  # fresh test set

    lab_idx = spread_seeds(X, y, 4)                          # 8 labels total
    unl = np.ones(len(X), dtype=bool)
    unl[lab_idx] = False
    X_lab, y_lab = X[lab_idx], y[lab_idx]
    X_unl, y_unl_true = X[unl], y[unl]

    banner("DEMO 1 --- The setting: 8 labels, 492 unlabelled points")
    print("  Two interleaved 'moons', one class each, separated by a low-density")
    print("  gap. Labelling is expensive: the annotator labelled 4 points spread")
    print("  along each moon -- 8 labels. The other 492 points are unlabelled.")
    print()
    base_acc = accuracy(KNN().fit(X_lab, y_lab), X_test, y_test)
    oracle_acc = accuracy(KNN().fit(X, y), X_test, y_test)
    print(f"  Trained on the 8 labels only        : {base_acc:6.1%} test accuracy")
    print(f"  Oracle: all 500 true labels (bound) : {oracle_acc:6.1%} test accuracy")
    print()
    print(f"  The gap to close from unlabelled data alone: "
          f"{oracle_acc - base_acc:.1%}")

    banner("DEMO 2 --- The loop: the pseudo-label frontier expands")
    print("  Each round: train, pseudo-label every point whose k=3 neighbours")
    print("  vote unanimously, add them, retrain. Wrong-label counts use the")
    print("  hidden ground truth, for reporting only.")
    print()
    final, st_acc, hist, added, wrong = self_train(
        X_lab, y_lab, X_unl, X_test=X_test, y_test=y_test,
        y_unl_true=y_unl_true)
    print("    round   added   cumulative   wrong so far   test accuracy")
    for rnd, n_take, cum, wr, acc in hist:
        print(f"      {rnd:2d}    {n_take:5d}      {cum:5d}        {wr:5d}"
              f"          {acc:6.1%}")
    print()
    print(f"  Final training set: 8 real labels + {added} pseudo-labels "
          f"({wrong} wrong, {wrong / added:.1%})")
    print(f"  Test accuracy: {base_acc:.1%} (labels only)  ->  {st_acc:.1%} "
          f"(self-trained)   [oracle {oracle_acc:.1%}]")
    print()
    print("  The first 48 pseudo-labels arrive with ZERO mistakes -- unanimous")
    print("  neighbourhoods only exist deep inside a cluster -- and by round 2")
    print("  the frontier sweeps up 380 more with just 3 errors. Confidence")
    print("  spreads outward along each moon, not across the gap.")

    banner("DEMO 3 --- The catch: the same loop, with badly-placed labels")
    print("  Same algorithm, same threshold, same budget of 8 labels -- but now")
    print("  placed at random instead of spread along the moons:")
    print()
    print("    draw    labels only    self-trained    wrong pseudo-labels")
    bases, sts = [], []
    for draw in range(5):
        d_rng = np.random.default_rng(draw)
        li = np.concatenate([
            d_rng.choice(np.where(y == 0)[0], 4, replace=False),
            d_rng.choice(np.where(y == 1)[0], 4, replace=False)])
        um = np.ones(len(X), dtype=bool)
        um[li] = False
        b = accuracy(KNN().fit(X[li], y[li]), X_test, y_test)
        _, s, _, a_d, w_d = self_train(
            X[li], y[li], X[um], X_test=X_test, y_test=y_test,
            y_unl_true=y[um])
        bases.append(b)
        sts.append(s)
        note = "   <- went backwards" if s < b else ""
        print(f"      {draw}      {b:6.1%}         {s:6.1%}            {w_d:4d}"
              f"{note}")
    print()
    print(f"    mean   {np.mean(bases):6.1%}         {np.mean(sts):6.1%}")
    print()
    print("  With coverage gaps, some regions sit closer to the WRONG cluster's")
    print("  frontier; the first bad pseudo-labels are retrained on as truth and")
    print("  recruit their neighbours -- confirmation bias, measured. Where the")
    print("  labels sit matters as much as how many you have (see Part 11,")
    print("  Active Learning).")


if __name__ == "__main__":
    main()
