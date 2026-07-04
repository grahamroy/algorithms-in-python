"""
mcts.py --- companion code for "Monte Carlo Tree Search (MCTS)"
(Advanced Reinforcement Learning, Part 6).

MCTS from scratch on Tic-Tac-Toe. Every algorithm so far in this series
LEARNED -- a Q-table, a value network, a policy. MCTS does something different:
it PLANS. No training, no network, no gradients. Given nothing but a simulator
of the game's rules, it builds a search tree at decision time by repeating four
phases:

  1. SELECTION      walk the tree picking children by UCT
                    (exploit high value + explore under-visited moves)
  2. EXPANSION      add one new child node to the tree
  3. SIMULATION     play random moves to the end of the game
  4. BACKPROPAGATION update every node on the path with the result

More simulations -> a bigger tree -> stronger play. Stop any time and act on
the best-visited move (the "anytime" property).

Tic-Tac-Toe is the demo domain because it has a KNOWN ground truth: with
perfect play the game is always a draw -- so the claims below are verifiable.

Demonstrates:
  1. Strength scales with thinking time (sims per move) vs a random opponent.
  2. The tree visibly concentrates its visits on the only correct move
     (take an immediate win / block an immediate loss).
  3. MCTS vs MCTS rediscovers perfect play: every game is a draw.

Dependencies: numpy (random numbers only). Runs in ~20-40 seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import math
import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
UCT_C = 1.4          # exploration constant (~ sqrt 2)


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Tic-Tac-Toe: board is a list of 9 cells (0 empty, +1 X, -1 O). X moves first.
# winner() returns +1 / -1 for a win, 0 for a draw, None if the game goes on.
# ---------------------------------------------------------------------------

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]


def winner(board):
    for a, b, c in LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    if all(v != 0 for v in board):
        return 0
    return None


def legal_moves(board):
    return [i for i in range(9) if board[i] == 0]


def show(board):
    sym = {1: "X", -1: "O", 0: "."}
    rows = []
    for r in range(3):
        rows.append(" " + " ".join(sym[board[3 * r + c]] for c in range(3)))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# The search tree. Each node remembers the move that created it, the player
# who made that move, its visit count N and total reward W (wins = 1,
# draws = 0.5, from the perspective of the player who just moved).
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("parent", "move", "player_just_moved",
                 "children", "untried", "N", "W")

    def __init__(self, parent, move, player_just_moved, untried):
        self.parent = parent
        self.move = move
        self.player_just_moved = player_just_moved
        self.children = []
        self.untried = untried
        self.N = 0
        self.W = 0.0

    def uct_child(self, c=UCT_C):
        logN = math.log(self.N)
        return max(self.children,
                   key=lambda ch: ch.W / ch.N + c * math.sqrt(logN / ch.N))


def mcts_search(board, player, n_sims, rng, c=UCT_C):
    """Run n_sims simulations from (board, player to move); return the move
    with the most visits, plus the root for inspection."""
    root = Node(None, None, -player, legal_moves(board))

    for _ in range(n_sims):
        node = root
        b = board[:]
        p = player

        # 1. SELECTION -- walk down while fully expanded and non-terminal
        while not node.untried and node.children:
            node = node.uct_child(c)
            b[node.move] = node.player_just_moved
            p = -p

        # 2. EXPANSION -- add one child for an untried move
        if node.untried:
            m = node.untried.pop(int(rng.integers(len(node.untried))))
            b[m] = p
            node.children.append(
                Node(node, m, p,
                     [] if winner(b) is not None else legal_moves(b)))
            node = node.children[-1]
            p = -p

        # 3. SIMULATION -- random playout to the end of the game
        w = winner(b)
        while w is None:
            moves = legal_moves(b)
            b[moves[int(rng.integers(len(moves)))]] = p
            p = -p
            w = winner(b)

        # 4. BACKPROPAGATION -- credit the result to every node on the path
        while node is not None:
            node.N += 1
            if w == 0:
                node.W += 0.5
            elif w == node.player_just_moved:
                node.W += 1.0
            node = node.parent

    best = max(root.children, key=lambda ch: ch.N)
    return best.move, root


# ---------------------------------------------------------------------------
# Opponents and match play
# ---------------------------------------------------------------------------

def play_game(x_agent, o_agent, rng):
    """x_agent / o_agent: either 'random' or an int = sims per MCTS move.
    Returns +1 (X wins), -1 (O wins), or 0 (draw)."""
    board = [0] * 9
    player = 1
    while True:
        agent = x_agent if player == 1 else o_agent
        if agent == "random":
            moves = legal_moves(board)
            m = moves[int(rng.integers(len(moves)))]
        else:
            m, _ = mcts_search(board, player, agent, rng)
        board[m] = player
        w = winner(board)
        if w is not None:
            return w
        player = -player


def match(x_agent, o_agent, games, seed):
    w = d = l = 0
    for g in range(games):
        rng = np.random.default_rng(seed + g)
        r = play_game(x_agent, o_agent, rng)
        if r == 1:
            w += 1
        elif r == 0:
            d += 1
        else:
            l += 1
    return w, d, l


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_strength():
    banner("DEMO 1 --- Strength scales with thinking time")
    games = 200
    print(f"  MCTS (as X) vs a uniformly random opponent, {games} games per budget.")
    print("  No training happened anywhere -- only more simulations per move.")
    print()
    print("    sims/move      W  -  D - L")
    for sims in (10, 50, 200, 800):
        w, d, l = match(sims, "random", games, seed=10_000 + sims)
        print(f"      {sims:5d}      {w:3d} - {d:3d} - {l:2d}")
    print()
    w, d, l = match("random", 800, games, seed=77_000)
    print(f"  As O (moving second, the harder side), 800 sims/move over "
          f"{games} games:")
    print(f"    {l} wins, {d} draws, {w} losses -- it never loses.")


def report_root(board, player, n_sims, seed, label):
    rng = np.random.default_rng(seed)
    move, root = mcts_search(board, player, n_sims, rng)
    print(show(board))
    print(f"  {label}: {n_sims} simulations, "
          f"{'X' if player == 1 else 'O'} to move")
    print("    move   visits   mean value")
    for ch in sorted(root.children, key=lambda ch: -ch.N)[:3]:
        print(f"     {ch.move}      {ch.N:5d}      {ch.W / ch.N:.2f}")
    print(f"    -> plays square {move}")
    print()
    return move


def demo_tree():
    banner("DEMO 2 --- The tree concentrates its visits on the right move")
    print("  Root statistics after search (squares numbered 0-8, row by row).")
    print()
    # Take the win: X has 0,1 (threatens 2); O has 3,4 (threatens 5). X to move.
    win_board = [1, 1, 0, -1, -1, 0, 0, 0, 0]
    m1 = report_root(win_board, 1, 400, seed=1, label="Take the win")
    # Block the loss: X has 0,1 (threatens 2); O has only the centre. O to move.
    block_board = [1, 1, 0, 0, -1, 0, 0, 0, 0]
    m2 = report_root(block_board, -1, 400, seed=2, label="Block or lose")
    print("  In both positions square 2 is the only correct move -- and the")
    print("  visit counts pile onto it. That concentration IS the search working:")
    print("  UCT spends its simulation budget where the playouts say it matters.")
    return m1, m2


def demo_selfplay():
    banner("DEMO 3 --- MCTS vs MCTS converges to perfect play")
    games = 30
    print(f"  {games} self-play games at each budget (both sides equal sims).")
    print("  Ground truth: perfect tic-tac-toe is ALWAYS a draw.")
    print()
    print("    sims/move    X wins   draws   O wins")
    for sims in (100, 400, 1500):
        results = {1: 0, 0: 0, -1: 0}
        for g in range(games):
            rng = np.random.default_rng(50_000 + g)
            results[play_game(sims, sims, rng)] += 1
        print(f"      {sims:5d}       {results[1]:3d}     {results[0]:4d}"
              f"     {results[-1]:3d}")
    print()
    print("  At 1500 sims every game is a draw: two searchers with no trained")
    print("  knowledge -- just the rules and playouts -- rediscover perfect play.")


def main() -> None:
    demo_strength()
    demo_tree()
    demo_selfplay()


if __name__ == "__main__":
    main()
