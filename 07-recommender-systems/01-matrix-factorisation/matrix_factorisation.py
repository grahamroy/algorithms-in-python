"""
matrix_factorisation.py --- companion code for "Matrix Factorisation"
(Recommender Systems, Part 1).

Generates a synthetic ratings matrix from known latent factors,
masks 60% of entries, fits matrix factorisation two ways from
scratch:
  1. Alternating Least Squares (ALS) — closed-form per-entity
     ridge regression updates.
  2. Stochastic Gradient Descent (FunkSVD style) — one update
     per observed rating.

Compares both against a global-mean baseline on a held-out test
split.

Dependencies: numpy. Runs in under a second.
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
# Generate a synthetic ratings matrix from known latent factors
# ---------------------------------------------------------------------------

def make_dataset(n_users=200, n_items=100, true_k=5,
                 observed_frac=0.40, noise=0.3,
                 test_frac=0.2, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    U_true = rng.normal(0, 1, size=(n_users, true_k))
    V_true = rng.normal(0, 1, size=(n_items, true_k))
    R = U_true @ V_true.T + rng.normal(0, noise,
                                       size=(n_users, n_items))
    # Mask observations
    mask = rng.uniform(size=R.shape) < observed_frac
    obs_indices = np.argwhere(mask)
    # Split into train / test
    rng.shuffle(obs_indices)
    n_test = int(len(obs_indices) * test_frac)
    test_idx = obs_indices[:n_test]
    train_idx = obs_indices[n_test:]
    return R, train_idx, test_idx


def rmse(R, U, V, indices):
    diff = np.array([R[i, j] - U[i] @ V[j] for i, j in indices])
    return float(np.sqrt((diff ** 2).mean()))


# ---------------------------------------------------------------------------
# Algorithm 1: ALS
# ---------------------------------------------------------------------------

def als(R, train_idx, k=5, lam=0.1, n_iter=20,
        seed=RNG_SEED):
    n, m = R.shape
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.1, size=(n, k))
    V = rng.normal(0, 0.1, size=(m, k))

    # Build per-user and per-item index lists
    users_to_items = [[] for _ in range(n)]
    items_to_users = [[] for _ in range(m)]
    for i, j in train_idx:
        users_to_items[i].append(j)
        items_to_users[j].append(i)

    I_k = np.eye(k)
    for _ in range(n_iter):
        # Update U
        for i in range(n):
            js = users_to_items[i]
            if not js:
                continue
            Vj = V[js]
            ri = R[i, js]
            A = Vj.T @ Vj + lam * I_k
            b = Vj.T @ ri
            U[i] = np.linalg.solve(A, b)
        # Update V
        for j in range(m):
            is_ = items_to_users[j]
            if not is_:
                continue
            Ui = U[is_]
            rj = R[is_, j]
            A = Ui.T @ Ui + lam * I_k
            b = Ui.T @ rj
            V[j] = np.linalg.solve(A, b)
    return U, V


# ---------------------------------------------------------------------------
# Algorithm 2: SGD (FunkSVD)
# ---------------------------------------------------------------------------

def sgd_mf(R, train_idx, k=5, lr=0.01, lam=0.1, n_epochs=30,
           seed=RNG_SEED):
    n, m = R.shape
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.1, size=(n, k))
    V = rng.normal(0, 0.1, size=(m, k))
    idx_array = np.array(train_idx)
    for _ in range(n_epochs):
        rng.shuffle(idx_array)
        for i, j in idx_array:
            r = R[i, j]
            pred = U[i] @ V[j]
            e = r - pred
            u_old = U[i].copy()
            U[i] += lr * (e * V[j] - lam * U[i])
            V[j] += lr * (e * u_old - lam * V[j])
    return U, V


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    banner("DEMO --- Matrix factorisation on synthetic ratings")

    R, train_idx, test_idx = make_dataset()
    n_users, n_items = R.shape
    n_obs = len(train_idx) + len(test_idx)
    density = n_obs / (n_users * n_items)
    print(f"  Users           : {n_users}")
    print(f"  Items           : {n_items}")
    print(f"  True latent dim : 5")
    print(f"  Observed cells  : {n_obs} of {n_users * n_items} "
          f"({density:.1%} density)")
    print(f"  Train/test split: {len(train_idx)}/{len(test_idx)}")
    print()
    print(f"  {'Method':<25} {'k':>4}   {'iters':>5}    "
          f"{'train RMSE':>11}    {'test RMSE':>11}")
    print(f"  {'-' * 23:<25} {'-' * 4:>4}   {'-' * 5:>5}    "
          f"{'-' * 11:>11}    {'-' * 11:>11}")

    # Baseline: global mean
    train_vals = np.array([R[i, j] for i, j in train_idx])
    mu = train_vals.mean()
    train_rmse = float(np.sqrt(((train_vals - mu) ** 2).mean()))
    test_vals = np.array([R[i, j] for i, j in test_idx])
    test_rmse = float(np.sqrt(((test_vals - mu) ** 2).mean()))
    print(f"  {'Baseline (global mean)':<25} {'—':>4}   {'—':>5}    "
          f"{train_rmse:>11.3f}    {test_rmse:>11.3f}")

    # ALS
    U_als, V_als = als(R, train_idx, k=5, lam=2.0, n_iter=20)
    print(f"  {'ALS':<25} {5:>4}   {20:>5}    "
          f"{rmse(R, U_als, V_als, train_idx):>11.3f}    "
          f"{rmse(R, U_als, V_als, test_idx):>11.3f}")

    # SGD
    U_sgd, V_sgd = sgd_mf(R, train_idx, k=5, lr=0.02,
                          lam=0.1, n_epochs=200)
    print(f"  {'SGD (FunkSVD)':<25} {5:>4}   {200:>5}    "
          f"{rmse(R, U_sgd, V_sgd, train_idx):>11.3f}    "
          f"{rmse(R, U_sgd, V_sgd, test_idx):>11.3f}")
    print()


if __name__ == "__main__":
    main()
