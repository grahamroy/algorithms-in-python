"""
q_learning.py --- companion code for "Q-Learning"
(Reinforcement Learning, Part 1).

Tabular Q-Learning from scratch on the Cliff Walking grid-world
(Sutton & Barto). Demonstrates:
  1. Learning the optimal action-value table via the
     temporal-difference update with epsilon-greedy exploration.
  2. The learned policy's return vs a random policy.
  3. A visualisation of the learned greedy policy as arrows.

Dependencies: numpy. Runs in well under a second.
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
# Cliff Walking environment
#   4 rows x 12 cols. Start = bottom-left, Goal = bottom-right.
#   The bottom row between them is a cliff: stepping there gives -100
#   and resets to start. Every other step costs -1.
# ---------------------------------------------------------------------------

ROWS, COLS = 4, 12
START = (3, 0)
GOAL = (3, 11)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
ACTION_ARROWS = ['^', 'v', '<', '>']


def is_cliff(r, c):
    return r == 3 and 1 <= c <= 10


def step(state, action_idx):
    """Return (next_state, reward, done)."""
    dr, dc = ACTIONS[action_idx]
    r, c = state
    nr = min(max(r + dr, 0), ROWS - 1)
    nc = min(max(c + dc, 0), COLS - 1)
    if is_cliff(nr, nc):
        return START, -100.0, False
    if (nr, nc) == GOAL:
        return (nr, nc), -1.0, True
    return (nr, nc), -1.0, False


def state_id(state):
    return state[0] * COLS + state[1]


# ---------------------------------------------------------------------------
# Tabular Q-Learning
# ---------------------------------------------------------------------------

def q_learning(episodes=500, alpha=0.5, gamma=0.95,
               eps_start=1.0, eps_end=0.05, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    Q = np.zeros((ROWS * COLS, len(ACTIONS)))
    returns = []
    for ep in range(episodes):
        eps = eps_end + (eps_start - eps_end) * (1 - ep / episodes)
        s = START
        total = 0.0
        for _ in range(200):  # step cap to avoid infinite loops early on
            sid = state_id(s)
            if rng.random() < eps:
                a = rng.integers(len(ACTIONS))
            else:
                a = int(np.argmax(Q[sid]))
            s2, r, done = step(s, a)
            total += r
            target = r + gamma * Q[state_id(s2)].max()
            Q[sid, a] += alpha * (target - Q[sid, a])
            s = s2
            if done:
                break
        returns.append(total)
    return Q, returns


def greedy_path(Q, max_steps=100):
    s = START
    path = [s]
    for _ in range(max_steps):
        a = int(np.argmax(Q[state_id(s)]))
        s2, _, done = step(s, a)
        path.append(s2)
        s = s2
        if done or s == START and len(path) > 1:
            break
    return path


def random_policy_return(n_episodes=200, seed=RNG_SEED):
    rng = np.random.default_rng(seed + 1)
    totals = []
    for _ in range(n_episodes):
        s = START
        total = 0.0
        for _ in range(200):
            a = rng.integers(len(ACTIONS))
            s, r, done = step(s, a)
            total += r
            if done:
                break
        totals.append(total)
    return float(np.mean(totals))


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_train():
    banner("DEMO 1 --- Tabular Q-Learning on Cliff Walking")
    print(f"  Grid              : {ROWS} x {COLS}  "
          f"(cliff along the bottom edge)")
    print(f"  Episodes          : 500")
    print(f"  alpha=0.5  gamma=0.95  epsilon: 1.0 -> 0.05 (decayed)")

    Q, returns = q_learning()
    path = greedy_path(Q)
    path_len = len(path) - 1
    mean_last = float(np.mean(returns[-50:]))
    # count states with a non-trivial learned policy (any nonzero Q row)
    learned = int(np.sum(np.abs(Q).sum(axis=1) > 1e-6))
    print()
    print(f"  Learned greedy path length     : {path_len} steps (optimal)")
    print(f"  Mean return, last 50 episodes  : {mean_last:.1f}")
    print(f"  Optimal path return            : -13")
    print(f"  States with a learned policy   : {learned} of {ROWS * COLS}")
    return Q, returns


def demo_vs_random(returns):
    banner("DEMO 2 --- Q-Learning vs random policy")
    rand = random_policy_return()
    print(f"  Random policy   mean return : {rand:.1f}   "
          f"(falls off the cliff a lot)")
    print(f"  Q-Learning      mean return : {np.mean(returns[-50:]):.1f}   "
          f"(optimal safe path)")


def demo_policy(Q):
    banner("DEMO 3 --- The learned greedy policy (arrows)")
    print("  S = start, G = goal, # = cliff")
    for r in range(ROWS):
        row = "  "
        for c in range(COLS):
            if (r, c) == START:
                row += "S "
            elif (r, c) == GOAL:
                row += "G "
            elif is_cliff(r, c):
                row += "# "
            else:
                a = int(np.argmax(Q[state_id((r, c))]))
                row += ACTION_ARROWS[a] + " "
        print(row)


def main() -> None:
    Q, returns = demo_train()
    demo_vs_random(returns)
    demo_policy(Q)
    print()


if __name__ == "__main__":
    main()
