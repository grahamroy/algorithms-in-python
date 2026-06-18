"""
sequential_recommender.py --- companion code for "Sequential
Recommenders" (Recommender Systems, Part 4).

A small SASRec (self-attentive sequential recommender) from
scratch in numpy:
  - item + positional embeddings
  - single-head causal-masked self-attention
  - position-wise feed-forward block with residual connections
  - tied input/output item embeddings
  - next-item training objective with sampled negatives,
    hand-written Adam, full backpropagation

Evaluates leave-one-out HR@10 / NDCG@10 against a popularity
baseline and a most-recent-item neighbour heuristic, on
synthetic order-dependent sessions.

Dependencies: numpy. Runs in a couple of minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 7


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Synthetic order-dependent sessions
# ---------------------------------------------------------------------------

def make_dataset(n_users=1000, n_items=500, n_genres=20,
                 avg_len=18, seed=RNG_SEED):
    """Items belong to latent genres. A user's next item tends to
    come from the genre of their *recent* items (a slowly drifting
    random walk over genres), so order carries the signal."""
    rng = np.random.default_rng(seed)
    item_genre = rng.integers(0, n_genres, size=n_items)
    genre_items = [np.where(item_genre == g)[0] for g in range(n_genres)]
    # Ensure every genre has at least one item
    genre_items = [gi if len(gi) else np.array([0]) for gi in genre_items]

    sequences = []
    for _ in range(n_users):
        length = max(5, int(rng.normal(avg_len, 4)))
        g = rng.integers(0, n_genres)
        seq = []
        for _ in range(length):
            # 80% stay in current genre, 20% drift to an adjacent genre
            if rng.random() < 0.2:
                g = (g + rng.integers(-2, 3)) % n_genres
            pool = genre_items[g]
            seq.append(int(rng.choice(pool)))
        sequences.append(seq)
    return sequences, n_items, item_genre


# ---------------------------------------------------------------------------
# SASRec (single head, single block)
# ---------------------------------------------------------------------------

def softmax_rows(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def relu(x):
    return np.maximum(x, 0.0)


class SASRec:
    def __init__(self, n_items, d=32, max_len=50, lr=0.01,
                 seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        self.n_items = n_items
        self.d = d
        self.max_len = max_len
        self.lr = lr
        s = 0.05
        # +1 row for the padding item (index 0 reserved)
        self.Item = rng.normal(0, s, size=(n_items + 1, d))
        self.Pos = rng.normal(0, s, size=(max_len, d))
        self.WQ = rng.normal(0, s, size=(d, d))
        self.WK = rng.normal(0, s, size=(d, d))
        self.WV = rng.normal(0, s, size=(d, d))
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), size=(d, d))
        self.b1 = np.zeros(d)
        self.W2 = rng.normal(0, np.sqrt(2.0 / d), size=(d, d))
        self.b2 = np.zeros(d)
        self._init_adam()

    def _init_adam(self):
        self._params = ["Item", "Pos", "WQ", "WK", "WV",
                        "W1", "b1", "W2", "b2"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self.t = 0

    def _adam(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for p, g in grads.items():
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g * g)
            mhat = self.m[p] / (1 - beta1 ** self.t)
            vhat = self.v[p] / (1 - beta2 ** self.t)
            setattr(self, p, getattr(self, p) -
                    self.lr * mhat / (np.sqrt(vhat) + eps))

    def _encode(self, seq):
        """Forward pass over one sequence (list of item ids).
        Returns per-position representations r (L x d) and a cache."""
        L = len(seq)
        idx = np.array(seq)
        X = self.Item[idx] + self.Pos[:L]            # L x d
        Q = X @ self.WQ
        K = X @ self.WK
        V = X @ self.WV
        scores = (Q @ K.T) / np.sqrt(self.d)         # L x L
        # Causal mask: position t attends to j <= t
        mask = np.triu(np.ones((L, L)), k=1).astype(bool)
        scores[mask] = -1e9
        A = softmax_rows(scores)                     # L x L
        C = A @ V                                     # L x d (context)
        # Feed-forward with residual
        H = relu(C @ self.W1 + self.b1)
        F = H @ self.W2 + self.b2
        R = F + C                                     # residual
        cache = (idx, X, Q, K, V, scores, A, C, H, F, R, mask, L)
        return R, cache

    def train_sequence(self, seq, rng, n_neg=1):
        """Next-item loss across all positions of one sequence."""
        if len(seq) < 2:
            return 0.0
        seq = seq[-self.max_len:]
        R, cache = self._encode(seq)
        (idx, X, Q, K, V, scores, A, C, H, F, R_, mask, L) = cache

        # Predict position t -> item seq[t+1], for t in 0..L-2
        pos = np.arange(L - 1)
        targets = np.array(seq[1:])                  # L-1
        r = R[pos]                                    # (L-1) x d
        pos_emb = self.Item[targets]                  # (L-1) x d
        neg_ids = rng.integers(1, self.n_items + 1, size=(L - 1, n_neg))
        neg_emb = self.Item[neg_ids]                  # (L-1) x n_neg x d

        pos_logit = (r * pos_emb).sum(axis=1)         # L-1
        neg_logit = np.einsum("td,tnd->tn", r, neg_emb)  # (L-1) x n_neg

        # BPR-style logistic loss: prefer positive over each negative
        def sig(x):
            return 1.0 / (1.0 + np.exp(-x))
        ps = sig(pos_logit)
        ns = sig(neg_logit)
        loss = -(np.log(ps + 1e-9).mean()
                 + np.log(1 - ns + 1e-9).mean())

        # ---- Gradients (only embeddings + r path; attention backprop) ----
        grads = {p: np.zeros_like(getattr(self, p)) for p in self._params}

        # d loss / d pos_logit and neg_logit
        dpos = (ps - 1.0) / (L - 1)                   # L-1
        dneg = ns / ((L - 1) * n_neg)                 # (L-1) x n_neg

        dr = dpos[:, None] * pos_emb                  # (L-1) x d
        dr += np.einsum("tn,tnd->td", dneg, neg_emb)
        # grad to target/negative item embeddings
        np.add.at(grads["Item"], targets, dpos[:, None] * r)
        np.add.at(grads["Item"], neg_ids.ravel(),
                  (dneg[:, :, None] * r[:, None, :]).reshape(-1, self.d))

        # Backprop dr (for positions 0..L-2) through residual + FFN + attn
        dR = np.zeros_like(R)
        dR[pos] += dr
        # Residual: R = F + C
        dF = dR.copy()
        dC = dR.copy()
        # FFN: F = H W2 + b2 ; H = relu(C W1 + b1)
        grads["W2"] += H.T @ dF
        grads["b2"] += dF.sum(axis=0)
        dH = dF @ self.W2.T
        dpre = dH * (H > 0)
        grads["W1"] += C.T @ dpre
        grads["b1"] += dpre.sum(axis=0)
        dC += dpre @ self.W1.T
        # Context: C = A V
        dA = dC @ V.T
        dV = A.T @ dC
        # Softmax backprop per row
        dscores = np.zeros_like(scores)
        for i in range(L):
            a = A[i]
            da = dA[i]
            dscores[i] = a * (da - (da * a).sum())
        dscores /= np.sqrt(self.d)
        dscores[mask] = 0.0
        dQ = dscores @ K
        dK = dscores.T @ Q
        grads["WQ"] += X.T @ dQ
        grads["WK"] += X.T @ dK
        grads["WV"] += X.T @ dV
        dX = dQ @ self.WQ.T + dK @ self.WK.T + dV @ self.WV.T
        np.add.at(grads["Item"], idx, dX)
        grads["Pos"][:L] += dX

        self._adam(grads)
        return float(loss)

    def next_item_repr(self, seq):
        seq = seq[-self.max_len:]
        R, _ = self._encode(seq)
        return R[-1]                                  # representation after last item


# ---------------------------------------------------------------------------
# Evaluation: leave-one-out HR@K / NDCG@K with 1 + 99 negatives
# ---------------------------------------------------------------------------

def evaluate(model, sequences, n_items, rng, k=10, n_neg=99):
    hr, ndcg, count = 0.0, 0.0, 0
    for seq in sequences:
        if len(seq) < 3:
            continue
        history, target = seq[:-1], seq[-1]
        r = model.next_item_repr(history)
        negs = rng.integers(1, n_items + 1, size=n_neg)
        candidates = np.concatenate([[target], negs])
        scores = model.Item[candidates] @ r
        # rank of the target (index 0)
        rank = int((scores > scores[0]).sum())
        if rank < k:
            hr += 1.0
            ndcg += 1.0 / np.log2(rank + 2)
        count += 1
    return hr / count, ndcg / count


def popularity_eval(sequences, n_items, rng, k=10, n_neg=99):
    counts = np.zeros(n_items + 1)
    for seq in sequences:
        for it in seq[:-1]:
            counts[it] += 1
    hr, ndcg, count = 0.0, 0.0, 0
    for seq in sequences:
        if len(seq) < 3:
            continue
        target = seq[-1]
        negs = rng.integers(1, n_items + 1, size=n_neg)
        candidates = np.concatenate([[target], negs])
        scores = counts[candidates]
        rank = int((scores > scores[0]).sum())
        if rank < k:
            hr += 1.0
            ndcg += 1.0 / np.log2(rank + 2)
        count += 1
    return hr / count, ndcg / count


def recency_eval(sequences, n_items, rng, k=10, n_neg=99):
    """Heuristic: a first-order transition (bigram) model — score
    candidates by how often they followed the user's last item in
    the training data. This is the realistic 'most-recent-item
    neighbours' baseline (learned co-occurrence, not oracle genre)."""
    trans = {}
    for seq in sequences:
        for a, b in zip(seq[:-2], seq[1:-1]):  # exclude held-out last
            trans.setdefault(a, {})
            trans[a][b] = trans[a].get(b, 0) + 1
    hr, ndcg, count = 0.0, 0.0, 0
    for seq in sequences:
        if len(seq) < 3:
            continue
        last, target = seq[-2], seq[-1]
        following = trans.get(last, {})
        negs = rng.integers(1, n_items + 1, size=n_neg)
        candidates = np.concatenate([[target], negs])
        scores = np.array([following.get(int(c), 0) for c in candidates],
                          dtype=float)
        scores = scores + rng.random(len(candidates)) * 0.001
        rank = int((scores > scores[0]).sum())
        if rank < k:
            hr += 1.0
            ndcg += 1.0 / np.log2(rank + 2)
        count += 1
    return hr / count, ndcg / count


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    banner("DEMO --- Sequential recommendation on synthetic sessions")

    sequences, n_items, item_genre = make_dataset()
    avg_len = int(np.mean([len(s) for s in sequences]))
    print(f"  Users                : {len(sequences)}")
    print(f"  Items                : {n_items}")
    print(f"  Avg sequence length  : {avg_len}")
    print(f"  Model                : 1-layer self-attention, d=32, causal mask")
    print(f"  Eval                 : leave-one-out, 1 + 99 negatives")

    model = SASRec(n_items, d=32, max_len=50, lr=0.01, seed=RNG_SEED)
    rng = np.random.default_rng(RNG_SEED)
    # Train on all-but-last of each sequence
    train_seqs = [s[:-1] for s in sequences]
    for _ in range(25):
        order = rng.permutation(len(train_seqs))
        for i in order:
            model.train_sequence(train_seqs[i], rng, n_neg=1)

    eval_rng = np.random.default_rng(123)
    pop_hr, pop_ndcg = popularity_eval(sequences, n_items, eval_rng)
    eval_rng = np.random.default_rng(123)
    rec_hr, rec_ndcg = recency_eval(sequences, n_items, eval_rng)
    eval_rng = np.random.default_rng(123)
    sas_hr, sas_ndcg = evaluate(model, sequences, n_items, eval_rng)

    print()
    print(f"  {'Method':<29} {'HR@10':>8}   {'NDCG@10':>8}")
    print(f"  {'-' * 27:<29} {'-' * 8:>8}   {'-' * 8:>8}")
    print(f"  {'Popularity (baseline)':<29} "
          f"{pop_hr:>8.3f}   {pop_ndcg:>8.3f}")
    print(f"  {'Most-recent-item neighbours':<29} "
          f"{rec_hr:>8.3f}   {rec_ndcg:>8.3f}")
    print(f"  {'SASRec (self-attention)':<29} "
          f"{sas_hr:>8.3f}   {sas_ndcg:>8.3f}")
    print()


if __name__ == "__main__":
    main()
