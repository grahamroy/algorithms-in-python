# Monte Carlo Tree Search — The Algorithm That Plans Instead of Learns

### *Algorithms in Python --- Advanced Reinforcement Learning, Part 6*

---

Every algorithm in this series so far has **learned** something.
A Q-table. A value network. A policy. Weeks of the series have
been spent on gradients, replay buffers, and trust regions — all
machinery for distilling experience into parameters, so that at
decision time the agent just *reads off* an action.

**Monte Carlo Tree Search (MCTS)** does something categorically
different: it **plans**. No training phase. No network. No
gradients. No parameters at all. Given nothing but a *simulator*
of the environment — the rules of the game, in effect — it
builds a search tree *at decision time*, spending its budget of
"thinking" on simulated futures, and then acts on what the tree
found. Ask it to think longer and it plays better, immediately,
with nothing learned in between.

That property made MCTS the backbone of computer Go for a decade
before deep learning arrived — and then, instead of being
replaced by neural networks, it *joined* them: AlphaGo and
AlphaZero are MCTS with learned networks guiding the search.
Planning and learning turned out to be complements, not rivals.

This article builds MCTS from scratch — the four phases, the UCT
selection rule, random playouts — and runs it on Tic-Tac-Toe,
chosen for one reason: the game has a **known ground truth**
(perfect play is always a draw), so every claim below is
checkable. We'll watch strength scale with thinking time, watch
the tree pile its visits onto the only correct move, and watch
two searchers rediscover perfect play from the rules alone.

---

## Planning vs learning

The distinction is worth making sharp, because it splits RL down
the middle.

A **learned policy** (everything in Parts 1–5, and the whole
foundational track) is *amortised* computation: enormous effort
at training time, so that acting costs one forward pass. Its
knowledge is baked into parameters; at decision time it does not
consider futures at all.

A **planner** like MCTS is the reverse: zero effort before the
game, all effort at the moment of choice. It needs one thing a
learned policy doesn't: a **model** — a way to simulate "if I
played this move, what could happen next?" Games provide that
model for free (the rules). Robotics often doesn't, which is why
model-free methods dominate there.

When you *do* have a model, planning has superpowers: it adapts
instantly to the exact situation in front of it, it never suffers
from stale training data, and its strength is a *dial* — more
simulations, better decisions — rather than a fixed ceiling.

---

## The four phases

MCTS grows a tree of game states, rooted at the current
position. Each node stores two numbers: how many times it has
been visited (`N`) and the total reward seen through it (`W`).
One **simulation** is four steps:

**1. Selection.** Walk down the existing tree from the root,
choosing at each node the child with the best **UCT** score
(more on that formula in a moment) — a balance between moves
that have scored well so far and moves that haven't been tried
enough. Stop when you reach a node with untried moves.

**2. Expansion.** Add *one* new child to the tree — one of the
untried moves. The tree grows by exactly one node per
simulation, so it grows where the search keeps choosing to go.

**3. Simulation (the playout).** From the new node, play the
game out to the end with *random moves*. This sounds far too
crude to work — and a single random playout is indeed a terrible
estimate. But it is a fast, *unbiased* sample of "how do games
from here tend to end?", and MCTS will average thousands of
them. Quantity launders quality.

**4. Backpropagation.** Walk back up the path and update every
node: `N += 1`, and `W` gets the result (1 for a win from that
node's perspective, 0.5 for a draw, 0 for a loss).

Repeat for as many simulations as the time budget allows, then
play the root move with the **most visits**. Stopping early just
means acting on a smaller tree — MCTS is an *anytime* algorithm.

---

## UCT: the formula that steers the search

The selection step is where the intelligence lives, and it is
one line. From node `p`, pick the child `i` maximising:

```
UCT(i)  =  W_i / N_i  +  c · sqrt( ln N_p / N_i )
```

The first term is **exploitation** — the child's average result
so far. The second is **exploration** — it grows for children
that have been visited rarely relative to their parent. The
constant `c` (√2 in theory, tuned in practice) sets the balance.

This is the UCB1 bandit rule (Auer et al., 2002) applied at
every node of a tree — the insight of Kocsis & Szepesvári's UCT
algorithm (2006), which together with Rémi Coulom's tree-building
scheme (2006) created what we now call MCTS. Treating each node
as a little bandit problem gives the search a principled answer
to the question every planner faces: *which future deserves more
of my remaining thinking time?*

The consequence is an **asymmetric tree**. Bad moves get a few
visits and are abandoned; promising lines get explored deeply.
The tree ends up shaped like the problem — effort concentrated
where the decision is actually hard.

---

## A worked example: Tic-Tac-Toe, because it's checkable

The companion script implements all of the above in ~60 lines of
search code and runs three experiments with verifiable answers.

### Strength is a dial, not a ceiling

```
DEMO 1 --- Strength scales with thinking time
  MCTS (as X) vs a uniformly random opponent, 200 games per budget.
  No training happened anywhere -- only more simulations per move.

    sims/move      W  -  D - L
         10      176 -  12 - 12
         50      192 -   7 -  1
        200      198 -   2 -  0
        800      198 -   2 -  0

  As O (moving second, the harder side), 800 sims/move over 200 games:
    181 wins, 19 draws, 0 losses -- it never loses.
```

The same code, with no training anywhere, goes from losing 12
games in 200 (at 10 simulations per move) to **never losing** —
as either side — just by thinking longer. That is the anytime
property in action: strength bought purely with decision-time
compute.

### The tree visibly finds the only correct move

Two tactical positions where a single square decides the game
(squares numbered 0–8, row by row). First, X can win on the
spot — and O is threatening to win too, so anything else loses:

```
 X X .
 O O .
 . . .
  Take the win: 400 simulations, X to move
    move   visits   mean value
     2        323      1.00
     5         29      0.55
     8         17      0.32
    -> plays square 2
```

Second, O must block or lose:

```
 X X .
 . O .
 . . .
  Block or lose: 400 simulations, O to move
    move   visits   mean value
     2        208      0.61
     3         46      0.36
     6         45      0.33
    -> plays square 2
```

Look at the visit counts, because they *are* the algorithm. Of
400 simulations, 323 went into the winning move in the first
position — UCT tried the alternatives, watched their playouts
end badly, and reallocated its budget to the move that kept
scoring 1.00. No rule about "take your win" or "block a threat"
exists anywhere in the code. The behaviour emerges from playouts
plus the bandit rule.

### Self-play converges to the game's ground truth

Tic-Tac-Toe under perfect play is *always* a draw — a fact we
know independently. So two equally strong searchers should draw
every game, and any decisive game is a measurable imperfection:

```
DEMO 3 --- MCTS vs MCTS converges to perfect play
    sims/move    X wins   draws   O wins
        100        14       16       0
        400         2       28       0
       1500         0       30       0
```

At 100 simulations the first player's advantage punishes the
shallow search 14 times. At 400, twice. At **1500 simulations,
all 30 games are draws** — two searchers with no trained
knowledge, given only the rules and random playouts, have
rediscovered perfect play. It's the series' cleanest example of
an algorithm converging on a known truth.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

MCTS costs nothing before the game and everything during it.

**Per simulation**: one walk down the tree (`O(depth · b)` UCT
comparisons, `b` = branching factor), one playout (`O(L)` random
moves to the end), one walk back up. Cheap — which is the point:
the budget buys *thousands* of these.

**Per move**: `O(S)` simulations of the above, and the tree holds
at most `S` nodes — one added per simulation. Memory is the tree,
nothing else.

**The knobs**: the budget `S` (strength dial), the exploration
constant `c`, and — in serious implementations — tree reuse
between moves, better-than-random playout policies, and
parallelised simulation. Vanilla MCTS as built here is the
skeleton all of those hang off.

**The requirement**: a simulator you can call thousands of times
per decision. No model, no MCTS.

---

## From here to AlphaZero

The straight line from this article to the strongest
game-players ever built is short, and worth drawing.

Vanilla MCTS has two weak spots: random playouts are noisy
evaluators, and UCT starts every node from ignorance. **AlphaGo**
(Silver et al., 2016) fixed both with networks: a **policy
network** proposes promising moves so the search doesn't waste
visits on junk, and a **value network** evaluates positions so
playouts can be shortened or skipped. The selection rule becomes
**PUCT** — UCT with the policy network's prior baked into the
exploration term.

**AlphaGo Zero** (2017) then removed the human data: the network
trains *from the search itself* — MCTS visit counts become the
training targets for the policy, game outcomes for the value —
and the improved network makes the next search stronger. Search
teaches the network; the network sharpens the search.
**AlphaZero** (2018) showed the same loop mastering chess and
shogi, and **MuZero** (2020) even learned the *simulator* itself.

The lesson for the series: planning and learning are not
competing paradigms. The strongest known systems are a learned
policy/value (Parts 1–5's machinery) wrapped around exactly the
tree search you just built.

---

## When to use MCTS

**Use it when** you have a reliable simulator, discrete actions,
and time to think at decision-time — board games, turn-based
planning, scheduling and combinatorial search, or any problem
where "consider the actual futures of *this* state" beats a
policy's general reflexes.

**Skip it when** there's no model to simulate with (model-free
RL exists for exactly that case), when actions are continuous
(the tree can't branch over a continuum without extra machinery),
or when decisions must be instant — a learned policy answers in
one forward pass; a planner needs its budget.

**Combine them** when the stakes justify it: a learned prior to
guide the tree and a learned value to evaluate leaves, in the
AlphaZero pattern. Planning buys precision on the current state;
learning buys generality across states.

---

## What comes next

Part 7 — the final article of the Advanced Reinforcement
Learning track — is **Offline Reinforcement Learning**: learning
a policy from a *fixed dataset* of past experience, with no
environment interaction at all. It is the setting where the
off-policy ideas from DQN, DDPG, and TD3 meet their hardest test
— when you can't try anything, the difference between what the
data supports and what your value function believes becomes the
whole problem.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**mcts.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/09-advanced-reinforcement-learning/06-monte-carlo-tree-search/mcts.py)

Run it with:

```bash
pip install numpy
python mcts.py
```

It needs only `numpy` (for random numbers) and runs in well
under a minute. Every piece is from scratch: the Tic-Tac-Toe
rules, the tree with its visit and value counts, UCT selection,
single-node expansion, random playouts, and backpropagation —
plus the match harness for the three experiments. The headline
insight worth pinning to the wall: **MCTS is planning, not
learning — with no training and no parameters, it repeats four
steps (select by UCT, expand one node, play out randomly, back
up the result) and acts on the most-visited move; its strength
is a dial set by thinking time, and on Tic-Tac-Toe that dial
runs from losing 12 games in 200 at 10 simulations to never
losing at 800 — and to all 30 self-play games drawn at 1500,
which is perfect play, rediscovered from the rules alone**.

---

*This is Part 6 of the Advanced Reinforcement Learning track in the Algorithms in Python series. The companion script `mcts.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). Unlike the learned methods of [Parts 1–4](https://medium.com/p/d3a3ccf0bf44), MCTS plans at decision time — and pairs with them in systems like AlphaZero. Part 7, Offline Reinforcement Learning, closes the track.*
