"""
sarsa.py --- companion code for "SARSA"
(Reinforcement Learning, Part 2).

SARSA (on-policy TD control) on the SAME Cliff Walking grid-world
used for Q-Learning in Part 1, so the two are directly comparable.
Demonstrates:
  1. SARSA learning a *safer* path than Q-Learning, with a better
     online return, because it accounts for its own exploration.
  2. Head-to-head: SARSA vs Q-Learning online return + path.
  3. The learned greedy policy as arrows (one row further from the
     cliff than Q-Learning's edge-hugging path).

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
# Cliff Walking environment (identical to Part 1)
# ---------------------------------------------------------------------------

ROWS, COLS = 4, 12
START = (3, 0)
GOAL = (3, 11)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
ACTION_ARROWS = ['^', 'v', '<', '>']


def is_cliff(r, c):
    return r == 3 and 1 <= c <= 10


def step(state, action_idx):
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


def eps_greedy(Q, sid, eps, rng):
    if rng.random() < eps:
        return rng.integers(len(ACTIONS))
    return int(np.argmax(Q[sid]))


# ---------------------------------------------------------------------------
# SARSA: on-policy TD control
#   The update bootstraps from Q(s', a') for the action ACTUALLY taken
#   next, not max_a' Q(s', a'). That single change makes it on-policy.
# ---------------------------------------------------------------------------

def sarsa(episodes=1000, alpha=0.5, gamma=0.95,
          eps=0.1, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    Q = np.zeros((ROWS * COLS, len(ACTIONS)))
    returns = []
    for ep in range(episodes):
        s = START
        a = eps_greedy(Q, state_id(s), eps, rng)
        total = 0.0
        for _ in range(200):
            s2, r, done = step(s, a)
            total += r
            a2 = eps_greedy(Q, state_id(s2), eps, rng)
            # ON-POLICY: bootstrap from the action actually chosen (a2)
            target = r + gamma * Q[state_id(s2), a2]
            sid = state_id(s)
            Q[sid, a] += alpha * (target - Q[sid, a])
            s, a = s2, a2
            if done:
                break
        returns.append(total)
    return Q, returns


# Q-Learning (off-policy) reproduced here for the head-to-head.
def q_learning(episodes=1000, alpha=0.5, gamma=0.95,
               eps=0.1, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    Q = np.zeros((ROWS * COLS, len(ACTIONS)))
    returns = []
    for ep in range(episodes):
        s = START
        total = 0.0
        for _ in range(200):
            sid = state_id(s)
            a = eps_greedy(Q, sid, eps, rng)
            s2, r, done = step(s, a)
            total += r
            target = r + gamma * Q[state_id(s2)].max()  # OFF-POLICY max
            Q[sid, a] += alpha * (target - Q[sid, a])
            s = s2
            if done:
                break
        returns.append(total)
    return Q, returns


def greedy_path(Q, max_steps=100):
    s = START
    path = [s]
    visited = set()
    for _ in range(max_steps):
        if (s, len(path)) in visited:
            break
        a = int(np.argmax(Q[state_id(s)]))
        s2, _, done = step(s, a)
        path.append(s2)
        if done:
            break
        # cycle guard: if we revisit a state we've already stood on, stop
        if s2 in [p for p in path[:-1]]:
            break
        s = s2
    return path


def path_row_usage(path):
    """Which rows does the greedy path traverse? Lower row index =
    closer to the top, further from the cliff (row 3)."""
    rows = sorted(set(r for r, _ in path))
    return rows


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_sarsa():
    banner("DEMO 1 --- SARSA (on-policy) on Cliff Walking")
    print(f"  Grid              : {ROWS} x {COLS}  (cliff along the bottom edge)")
    print(f"  Episodes          : 1000")
    print(f"  alpha=0.5  gamma=0.95  epsilon=0.1 (FIXED, not decayed)")

    Q, returns = sarsa()
    path = greedy_path(Q)
    mean_last = float(np.mean(returns[-50:]))
    rows = path_row_usage(path)
    print()
    print(f"  Learned greedy path length     : {len(path) - 1} steps")
    print(f"  Mean return, last 50 episodes  : {mean_last:.1f}")
    print(f"  Greedy path uses rows          : {rows}  "
          f"(0=top, 3=cliff row)")
    return Q, returns


def demo_head_to_head(sarsa_returns):
    banner("DEMO 2 --- SARSA vs Q-Learning (online return)")
    q_Q, q_returns = q_learning()
    s_mean = float(np.mean(sarsa_returns[-50:]))
    q_mean = float(np.mean(q_returns[-50:]))
    print(f"  Q-Learning (off-policy)  mean return : {q_mean:.1f}   "
          f"(optimal edge path, falls during exploration)")
    print(f"  SARSA      (on-policy)   mean return : {s_mean:.1f}   "
          f"(safer path, fewer falls)")
    print()
    print(f"  SARSA online return is higher (less negative) by "
          f"{s_mean - q_mean:.1f} despite a longer path —")
    print(f"  it accounts for its own exploration and stays away "
          f"from the edge.")
    return q_Q


def demo_policies(sarsa_Q, q_Q):
    banner("DEMO 3 --- Learned greedy policies side by side")
    print("  S=start  G=goal  #=cliff")
    print()
    for name, Q in [("SARSA (safer)", sarsa_Q), ("Q-Learning (edge)", q_Q)]:
        print(f"  {name}:")
        for r in range(ROWS):
            row = "    "
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
        print()


def main() -> None:
    sarsa_Q, sarsa_returns = demo_sarsa()
    q_Q = demo_head_to_head(sarsa_returns)
    demo_policies(sarsa_Q, q_Q)


if __name__ == "__main__":
    main()
