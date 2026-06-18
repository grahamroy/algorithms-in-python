"""
neural_collaborative_filtering.py --- companion code for "Neural
Collaborative Filtering" (Recommender Systems, Part 2).

Builds the three models from He et al. (2017) from scratch in numpy:
  1. GMF  --- Generalised Matrix Factorisation: element-wise product
     of user/item embeddings with a learned output layer. This is the
     dot-product model, generalised (uniform weights => plain MF).
  2. MLP  --- concatenate user/item embeddings and pass them through
     a multi-layer perceptron with ReLU. Learns a NON-LINEAR
     interaction function.
  3. NeuMF --- fuse a GMF tower and an MLP tower (each with its own
     embeddings) and predict from the concatenation.

The synthetic data is generated from a fixed, RANDOM non-linear
"teacher" interaction function, so the true affinity between a user
and an item is genuinely non-linear in their latent factors. This is
exactly the regime where a fixed dot product underfits and a learned
interaction function wins.

Implicit feedback throughout: positives are sampled interactions,
training uses negative sampling + binary cross-entropy, and evaluation
follows the standard leave-one-out HR@10 / NDCG@10 protocol (rank the
held-out positive against 99 sampled negatives).

Dependencies: numpy. Runs in a few seconds.
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------------------
# Synthetic implicit-feedback data from a non-linear teacher
# ---------------------------------------------------------------------------

def _teacher_scores(U, V, n_groups=4):
    """Ground-truth affinity = a sum of rectified, group-wise bilinear
    forms:  score(u, v) = Σ_g ReLU(u_g · v_g).

    Splitting the latent dimensions into groups and rectifying each
    group's dot product makes the interaction genuinely non-linear and
    high-rank: no single bilinear form (a plain dot product) can
    reproduce a sum of rectified bilinear forms, but an MLP --- which
    computes exactly sums of rectified linear combinations --- can.
    """
    d = U.shape[1]
    sizes = np.array_split(np.arange(d), n_groups)
    S = np.zeros((U.shape[0], V.shape[0]))
    for idx in sizes:
        S += np.maximum(U[:, idx] @ V[:, idx].T, 0.0)
    return S


def make_dataset(n_users=500, n_items=300, d_true=16,
                 pos_per_user=25, seed=RNG_SEED):
    """Build a sparse implicit-feedback matrix with a non-linear
    ground-truth interaction structure, then split leave-one-out."""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, size=(n_users, d_true))
    V = rng.normal(0, 1, size=(n_items, d_true))

    S = _teacher_scores(U, V)
    # Per-user standardise so every user has a comparable scale, then
    # turn the highest-affinity items into observed positives (with a
    # little stochasticity so it isn't a clean top-k cut).
    S = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    noisy = S + rng.normal(0, 0.5, size=S.shape)

    user_pos = []
    for i in range(n_users):
        order = np.argsort(-noisy[i])
        user_pos.append(order[:pos_per_user].tolist())

    # Leave-one-out: hold out one random positive per user.
    train_pairs = []
    test_pos = np.empty(n_users, dtype=int)
    pos_set = [set() for _ in range(n_users)]
    for i in range(n_users):
        items = user_pos[i]
        held = int(rng.integers(len(items)))
        test_pos[i] = items[held]
        for j in items:
            pos_set[i].add(j)
        for jx, j in enumerate(items):
            if jx != held:
                train_pairs.append((i, j))

    train_pairs = np.array(train_pairs, dtype=int)
    # Popularity (interaction count) over the training positives.
    pop = np.bincount(train_pairs[:, 1], minlength=n_items).astype(float)
    return {
        "n_users": n_users, "n_items": n_items,
        "train_pairs": train_pairs, "test_pos": test_pos,
        "pos_set": pos_set, "pop": pop,
    }


def sample_eval_negatives(data, n_neg=99, seed=RNG_SEED):
    """For each user pick n_neg items they never interacted with."""
    rng = np.random.default_rng(seed + 99)
    n_users, n_items = data["n_users"], data["n_items"]
    pos_set = data["pos_set"]
    negs = np.empty((n_users, n_neg), dtype=int)
    for i in range(n_users):
        chosen = []
        seen = pos_set[i]
        while len(chosen) < n_neg:
            c = int(rng.integers(n_items))
            if c not in seen:
                chosen.append(c)
        negs[i] = chosen
    return negs


# ---------------------------------------------------------------------------
# A tiny Adam optimiser over a dict of named parameter arrays
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def _init(rng, *shape, scale=0.05):
    return rng.normal(0, scale, size=shape)


# ---------------------------------------------------------------------------
# Model 1: GMF (Generalised Matrix Factorisation)
# ---------------------------------------------------------------------------

class GMF:
    def __init__(self, n_users, n_items, k=16, seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        self.p = {
            "P": _init(rng, n_users, k),
            "Q": _init(rng, n_items, k),
            "w": _init(rng, k, scale=0.1),
            "b": np.zeros(1),
        }

    def forward(self, us, is_):
        pu, qi = self.p["P"][us], self.p["Q"][is_]
        phi = pu * qi
        logit = phi @ self.p["w"] + self.p["b"]
        return logit, (us, is_, pu, qi, phi)

    def backward(self, dl, cache):
        us, is_, pu, qi, phi = cache
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        g["w"] = phi.T @ dl
        g["b"] = np.array([dl.sum()])
        dphi = dl[:, None] * self.p["w"][None, :]
        np.add.at(g["P"], us, dphi * qi)
        np.add.at(g["Q"], is_, dphi * pu)
        return g


# ---------------------------------------------------------------------------
# Model 2: MLP (concatenate embeddings, ReLU hidden layers)
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, n_users, n_items, k=16, layers=(32, 16),
                 seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        self.p = {"P": _init(rng, n_users, k), "Q": _init(rng, n_items, k)}
        dims = [2 * k] + list(layers)
        self.depth = len(layers)
        for li in range(self.depth):
            self.p[f"W{li}"] = _init(rng, dims[li], dims[li + 1],
                                     scale=np.sqrt(2.0 / dims[li]))
            self.p[f"b{li}"] = np.zeros(dims[li + 1])
        self.p["Wo"] = _init(rng, dims[-1], 1, scale=0.1)
        self.p["bo"] = np.zeros(1)

    def forward(self, us, is_):
        pu, qi = self.p["P"][us], self.p["Q"][is_]
        a = np.concatenate([pu, qi], axis=1)
        acts = [a]
        pre = []
        for li in range(self.depth):
            z = a @ self.p[f"W{li}"] + self.p[f"b{li}"]
            pre.append(z)
            a = np.maximum(z, 0.0)
            acts.append(a)
        logit = (a @ self.p["Wo"] + self.p["bo"]).ravel()
        return logit, (us, is_, pu, qi, acts, pre)

    def backward(self, dl, cache):
        us, is_, pu, qi, acts, pre = cache
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        a_last = acts[-1]
        g["Wo"] = a_last.T @ dl[:, None]
        g["bo"] = np.array([dl.sum()])
        da = dl[:, None] * self.p["Wo"][None, :, 0]
        for li in reversed(range(self.depth)):
            dz = da * (pre[li] > 0)
            g[f"W{li}"] = acts[li].T @ dz
            g[f"b{li}"] = dz.sum(axis=0)
            da = dz @ self.p[f"W{li}"].T
        k = pu.shape[1]
        np.add.at(g["P"], us, da[:, :k])
        np.add.at(g["Q"], is_, da[:, k:])
        return g


# ---------------------------------------------------------------------------
# Model 3: NeuMF (fuse a GMF tower and an MLP tower)
# ---------------------------------------------------------------------------

class NeuMF:
    def __init__(self, n_users, n_items, k_gmf=16, k_mlp=16,
                 layers=(32, 16), seed=RNG_SEED):
        rng = np.random.default_rng(seed)
        self.p = {
            "Pg": _init(rng, n_users, k_gmf),
            "Qg": _init(rng, n_items, k_gmf),
            "Pm": _init(rng, n_users, k_mlp),
            "Qm": _init(rng, n_items, k_mlp),
        }
        dims = [2 * k_mlp] + list(layers)
        self.depth = len(layers)
        for li in range(self.depth):
            self.p[f"W{li}"] = _init(rng, dims[li], dims[li + 1],
                                     scale=np.sqrt(2.0 / dims[li]))
            self.p[f"b{li}"] = np.zeros(dims[li + 1])
        fuse_dim = k_gmf + dims[-1]
        self.p["Wo"] = _init(rng, fuse_dim, 1, scale=0.1)
        self.p["bo"] = np.zeros(1)
        self.k_gmf = k_gmf

    def forward(self, us, is_):
        pug, qig = self.p["Pg"][us], self.p["Qg"][is_]
        phi_g = pug * qig
        pum, qim = self.p["Pm"][us], self.p["Qm"][is_]
        a = np.concatenate([pum, qim], axis=1)
        acts, pre = [a], []
        for li in range(self.depth):
            z = a @ self.p[f"W{li}"] + self.p[f"b{li}"]
            pre.append(z)
            a = np.maximum(z, 0.0)
            acts.append(a)
        fuse = np.concatenate([phi_g, a], axis=1)
        logit = (fuse @ self.p["Wo"] + self.p["bo"]).ravel()
        return logit, (us, is_, pug, qig, phi_g, pum, qim, acts, pre, fuse)

    def backward(self, dl, cache):
        (us, is_, pug, qig, phi_g, pum, qim, acts, pre, fuse) = cache
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        g["Wo"] = fuse.T @ dl[:, None]
        g["bo"] = np.array([dl.sum()])
        dfuse = dl[:, None] * self.p["Wo"][None, :, 0]
        dphi_g = dfuse[:, :self.k_gmf]
        da = dfuse[:, self.k_gmf:]
        # GMF tower
        np.add.at(g["Pg"], us, dphi_g * qig)
        np.add.at(g["Qg"], is_, dphi_g * pug)
        # MLP tower
        for li in reversed(range(self.depth)):
            dz = da * (pre[li] > 0)
            g[f"W{li}"] = acts[li].T @ dz
            g[f"b{li}"] = dz.sum(axis=0)
            da = dz @ self.p[f"W{li}"].T
        k = pum.shape[1]
        np.add.at(g["Pm"], us, da[:, :k])
        np.add.at(g["Qm"], is_, da[:, k:])
        return g


# ---------------------------------------------------------------------------
# Training loop (negative sampling + BCE) and ranking evaluation
# ---------------------------------------------------------------------------

def train(model, data, epochs=15, n_neg=4, lr=0.01, lam=1e-5,
          batch=512, seed=RNG_SEED):
    rng = np.random.default_rng(seed + 5)
    opt = Adam(model.p, lr=lr)
    pos = data["train_pairs"]
    n_items = data["n_items"]
    pos_set = data["pos_set"]
    n_pos = len(pos)

    for _ in range(epochs):
        # Build a training set: each positive + n_neg sampled negatives.
        us = np.repeat(pos[:, 0], 1 + n_neg)
        items = np.empty(n_pos * (1 + n_neg), dtype=int)
        labels = np.zeros(n_pos * (1 + n_neg))
        for t in range(n_pos):
            base = t * (1 + n_neg)
            i = pos[t, 0]
            items[base] = pos[t, 1]
            labels[base] = 1.0
            seen = pos_set[i]
            for q in range(n_neg):
                c = int(rng.integers(n_items))
                while c in seen:
                    c = int(rng.integers(n_items))
                items[base + 1 + q] = c
        order = rng.permutation(len(us))
        us, items, labels = us[order], items[order], labels[order]

        for s in range(0, len(us), batch):
            bu = us[s:s + batch]
            bi = items[s:s + batch]
            by = labels[s:s + batch]
            logit, cache = model.forward(bu, bi)
            pred = sigmoid(logit)
            dl = (pred - by) / len(by)
            grads = model.backward(dl, cache)
            if lam:
                for key in grads:
                    grads[key] += lam * model.p[key]
            opt.step(model.p, grads)
    return model


def score_pairs(model, us, is_):
    logit, _ = model.forward(np.asarray(us), np.asarray(is_))
    return logit


def evaluate(model, data, eval_negs, topk=10):
    """Leave-one-out HR@k and NDCG@k against sampled negatives."""
    n_users = data["n_users"]
    test_pos = data["test_pos"]
    hits, ndcg = 0.0, 0.0
    for i in range(n_users):
        cand = np.concatenate([[test_pos[i]], eval_negs[i]])
        scores = score_pairs(model, np.full(len(cand), i), cand)
        # Rank of the held-out positive (index 0 in cand).
        rank = int((scores > scores[0]).sum())  # items strictly above it
        if rank < topk:
            hits += 1.0
            ndcg += 1.0 / np.log2(rank + 2)
    return hits / n_users, ndcg / n_users


def popularity_eval(data, eval_negs, topk=10):
    n_users = data["n_users"]
    test_pos = data["test_pos"]
    pop = data["pop"]
    hits, ndcg = 0.0, 0.0
    for i in range(n_users):
        cand = np.concatenate([[test_pos[i]], eval_negs[i]])
        scores = pop[cand]
        rank = int((scores > scores[0]).sum())
        if rank < topk:
            hits += 1.0
            ndcg += 1.0 / np.log2(rank + 2)
    return hits / n_users, ndcg / n_users


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    banner("DEMO --- Neural Collaborative Filtering on synthetic "
           "implicit feedback")

    data = make_dataset()
    eval_negs = sample_eval_negatives(data)
    n_users, n_items = data["n_users"], data["n_items"]
    n_train = len(data["train_pairs"])
    density = (n_train + n_users) / (n_users * n_items)

    print(f"  Users                : {n_users}")
    print(f"  Items                : {n_items}")
    print(f"  Interaction type     : implicit (non-linear teacher)")
    print(f"  Observed interactions: {n_train + n_users} "
          f"({density:.1%} density)")
    print(f"  Eval protocol        : leave-one-out, 1 + 99 negatives, "
          f"HR@10 / NDCG@10")
    print()
    print(f"  {'Model':<26} {'params':>8}   {'HR@10':>7}   {'NDCG@10':>8}")
    print(f"  {'-' * 24:<26} {'-' * 8:>8}   {'-' * 7:>7}   {'-' * 8:>8}")

    # Baseline: popularity
    hr, ndcg = popularity_eval(data, eval_negs)
    print(f"  {'Popularity (baseline)':<26} {'—':>8}   "
          f"{hr:>7.3f}   {ndcg:>8.3f}")

    def n_params(model):
        return sum(v.size for v in model.p.values())

    # GMF
    gmf = GMF(n_users, n_items, k=16)
    train(gmf, data, epochs=30, lr=0.005)
    hr, ndcg = evaluate(gmf, data, eval_negs)
    print(f"  {'GMF (dot-product gen.)':<26} {n_params(gmf):>8}   "
          f"{hr:>7.3f}   {ndcg:>8.3f}")

    # MLP
    mlp = MLP(n_users, n_items, k=16, layers=(32, 16))
    train(mlp, data, epochs=30, lr=0.005)
    hr, ndcg = evaluate(mlp, data, eval_negs)
    print(f"  {'MLP (learned interaction)':<26} {n_params(mlp):>8}   "
          f"{hr:>7.3f}   {ndcg:>8.3f}")

    # NeuMF
    neumf = NeuMF(n_users, n_items, k_gmf=16, k_mlp=16, layers=(32, 16))
    train(neumf, data, epochs=30, lr=0.005)
    hr, ndcg = evaluate(neumf, data, eval_negs)
    print(f"  {'NeuMF (GMF + MLP fusion)':<26} {n_params(neumf):>8}   "
          f"{hr:>7.3f}   {ndcg:>8.3f}")
    print()


if __name__ == "__main__":
    main()
