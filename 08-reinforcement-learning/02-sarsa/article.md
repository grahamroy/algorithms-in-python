# SARSA — The On-Policy Sibling That Learns to Play It Safe

### *Algorithms in Python --- Reinforcement Learning, Part 2*

---

Part 1 left us standing at the edge of a cliff. Q-Learning
learned the *optimal* path across the Cliff Walking grid — the
shortest route, hugging the very edge of the drop — and then,
because it kept exploring with a small `ε`, repeatedly stepped
off that edge during training and posted a mediocre online
return. It learned the bravest policy and paid for its bravery.

There is a different way to learn, and it differs from
Q-Learning by **a single term in the update rule**. That one
change turns the bravest learner into a cautious one — and on
the cliff, the cautious learner does *better* in practice.

The algorithm is **SARSA** (Rummery & Niranjan, 1994; the name
is from the quintuple it uses: State, Action, Reward, next
State, next Action). It is the **on-policy** counterpart to
Q-Learning's off-policy learning, and the contrast between them
is the single cleanest illustration of one of reinforcement
learning's most important distinctions. They share the same
temporal-difference machinery, the same ε-greedy exploration,
the same grid-world — and they learn *different policies*,
because they answer subtly different questions. Q-Learning asks
"what is the value of acting optimally from here?" SARSA asks
"what is the value of acting the way I *actually* act,
exploration and all?"

This article builds SARSA from first principles, sets its
update beside Q-Learning's so the one-term difference is
unmissable, runs both on the *identical* Cliff Walking
environment from Part 1, and shows SARSA learning a visibly
safer path with a better online return. The on-policy /
off-policy distinction it draws runs through the whole of
reinforcement learning.

---

## The one-term difference

Recall Q-Learning's update. After taking action `a` in state
`s`, observing reward `r` and next state `s'`, it nudges
`Q(s, a)` toward:

```
Q-Learning target  =  r + γ · max_{a'} Q(s', a')
```

The `max` is the key: Q-Learning bootstraps from the *best
possible* next action, whatever the agent goes on to actually
do. It learns the value of the greedy (optimal) policy even
while behaving non-greedily. That is what **off-policy** means
— the policy being *learned* (greedy) differs from the policy
being *followed* (ε-greedy).

SARSA changes exactly one thing. It bootstraps from the action
the agent *actually takes next* — call it `a'`, chosen by the
same ε-greedy policy:

```
SARSA target  =  r + γ · Q(s', a')
```

No `max`. The next action `a'` is whatever the behaviour policy
picks — sometimes greedy, sometimes a random exploratory step.
SARSA learns the value of the policy it is *actually following*,
exploration included. That is **on-policy**: the policy learned
*is* the policy followed.

The full update:

```
SARSA(s, a, r, s', a'):
    Q(s, a) ← Q(s, a) + α · ( r + γ·Q(s', a') − Q(s, a) )
```

This is where the name comes from — the update needs the whole
quintuple **(s, a, r, s', a')**. Q-Learning only needs
(s, a, r, s') because its `max` invents the next action rather
than waiting to see it.

---

## Why one term changes the policy

It sounds like a technicality. It is not. Consider a state one
step from the cliff edge. The greedy action is "move along the
edge toward the goal" — fast and, if executed perfectly, safe.

**Q-Learning** evaluates that state with `max_{a'} Q(s', a')`:
it assumes the *next* action will also be the optimal,
edge-hugging one. It never charges the edge state for the risk
that exploration might, 10% of the time, fling the agent off
the cliff. So Q-Learning happily learns the edge path is best.

**SARSA** evaluates the same state with `Q(s', a')` for the
action actually taken — which, 10% of the time, *is* the random
step off the cliff. That −100 gets averaged into the edge
state's value. The edge becomes genuinely less valuable *under
the policy SARSA is following*, so SARSA learns to route one or
more rows away from it. It trades a couple of extra steps for a
margin of safety against its own exploration.

Neither is "right" in the abstract. Q-Learning learns the
optimal policy for an agent that will *eventually stop
exploring*. SARSA learns the best policy for an agent that
*keeps exploring as it acts* — which is exactly the situation
during online learning, and often the situation in deployment
(a robot that never stops having sensor noise). On the cliff,
that makes SARSA's policy the better one to *follow*.

---

## The full algorithm

```
SARSA(episodes, α, γ, ε):
    Initialise Q(s, a) = 0 for all s, a
    for each episode:
        s = start state
        a = ε-greedy(s)
        while s is not terminal:
            take a, observe r and s'
            a' = ε-greedy(s')                 # pick next action FIRST
            Q(s,a) ← Q(s,a) + α·(r + γ·Q(s',a') − Q(s,a))
            s, a = s', a'
    return Q
```

The structural difference from Q-Learning's loop: SARSA chooses
the next action `a'` *before* updating, and reuses it as the
action for the next step. Q-Learning updates first (using a
`max`) and chooses fresh each iteration. Same skeleton, one
extra commitment.

A note on exploration. In Part 1 we *decayed* `ε` toward zero.
For SARSA's safety behaviour to appear we deliberately keep `ε`
**fixed** at 0.1 — because the whole point is to learn a policy
that accounts for *persistent* exploration. If `ε` decays to
zero, there is no exploration left to be cautious about, and
SARSA converges to the same edge path as Q-Learning. The
on-policy advantage is a statement about behaving *while*
exploring. (This is also why Q-Learning's online return here,
−41.8, differs slightly from the −43.7 in Part 1: that figure
came from a decaying-`ε` run, whereas the head-to-head below
holds `ε` fixed at 0.1 for both algorithms so the only
difference between them is the update rule.)

---

## A worked example: the same cliff, a safer route

The companion script runs SARSA on the *identical* Cliff
Walking grid from Part 1, with a fixed `ε = 0.1`, and puts it
head-to-head with Q-Learning on the same setup.

```
DEMO 1 --- SARSA (on-policy) on Cliff Walking
  Grid              : 4 x 12  (cliff along the bottom edge)
  Episodes          : 1000
  alpha=0.5  gamma=0.95  epsilon=0.1 (FIXED, not decayed)

  Learned greedy path length     : 17 steps
  Mean return, last 50 episodes  : -25.1
  Greedy path uses rows          : [0, 1, 2, 3]  (0=top, 3=cliff row)
```

```
DEMO 2 --- SARSA vs Q-Learning (online return)
  Q-Learning (off-policy)  mean return : -41.8   (optimal edge path, falls during exploration)
  SARSA      (on-policy)   mean return : -25.1   (safer path, fewer falls)

  SARSA online return is higher (less negative) by 16.7 despite a longer path —
  it accounts for its own exploration and stays away from the edge.
```

```
DEMO 3 --- Learned greedy policies side by side
  S=start  G=goal  #=cliff

  SARSA (safer):
    > > > > > > > > > > > v
    ^ < > ^ ^ ^ ^ ^ ^ ^ v v
    ^ ^ < ^ ^ ^ ^ ^ > ^ > v
    S # # # # # # # # # # G

  Q-Learning (edge):
    > > > < > > v > > > v v
    > > > v v v > v v > v v
    > > > > > > > > > > > v
    S # # # # # # # # # # G
```

Three observations.

**SARSA learns the safe route; Q-Learning learns the edge.**
The policy grids show it directly. SARSA climbs away from the
cliff — up to the top rows — runs across, and only drops to the
goal at the end: a 17-step path that keeps maximum distance
from the drop. Q-Learning runs along row 2, one single step
above the cliff: the 13-step optimal path. Same environment,
same reward function, two genuinely different policies — and
the only code difference is `max_{a'} Q(s', a')` versus
`Q(s', a')`.

**SARSA's online return is better: −25.1 vs −41.8.** This is
the headline. Q-Learning's path is *shorter and optimal*, yet
its actual return during training is *worse* by 16.7, because
walking the cliff edge with a 10% random step means falling off
regularly for −100 apiece. SARSA's longer-but-safer route
almost never falls, so the return it actually *earns while
learning* is higher. When the agent must perform well *while*
exploring, on-policy wins.

**This is the exact gap Part 1 set up.** Q-Learning learns the
brave-optimal policy and pays for its bravery next to the
hazard; SARSA sacrifices two steps of optimality to stop paying
that price. Neither is wrong — they optimise different things —
but the −16.7 difference in online return is the clearest
single number in RL for "off-policy learns the optimal target;
on-policy learns the safer behaviour."

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

SARSA's costs are *identical* to Q-Learning's — it is the same
tabular TD method with one term swapped:

**Per step**: `O(|A|)` — ε-greedy action selection. SARSA
actually does slightly *less* work than Q-Learning per update:
it reads a single `Q(s', a')` instead of computing a `max` over
all next actions. A negligible difference, but SARSA is, if
anything, marginally cheaper.

**Per episode**: `O(steps · |A|)`.

**Memory**: `O(|S| · |A|)` for the Q-table — the same scaling
wall as Q-Learning. SARSA inherits the tabular limitation
exactly: it works only when the state space is small enough to
enumerate, and needs function approximation (the deep methods
of Part 3 onward) for anything larger.

**Convergence**: SARSA converges to the optimal policy *if* `ε`
is decayed to zero appropriately over time (the GLIE condition
— Greedy in the Limit with Infinite Exploration). With a fixed
`ε`, as here, it converges to the best policy *for that level
of exploration* — which is precisely the safer cliff path we
wanted to see.

---

## On-policy vs off-policy: the deeper distinction

The cliff is a vivid illustration of a split that runs through
all of reinforcement learning.

**Off-policy** methods (Q-Learning, DQN, Q-learning's whole
descendant family) learn about one policy — usually the optimal
greedy one — while following another. Their great advantage:
they can learn the optimal policy from *any* source of
experience, including random exploration, old logged data, or a
human demonstrator. This is what makes **experience replay**
possible (DQN reuses old transitions) and what makes **offline
RL** — learning purely from a fixed dataset — feasible.

**On-policy** methods (SARSA, and the policy-gradient methods
of Part 4 — REINFORCE, A2C, PPO) learn about the policy they
are following. Their advantage: they account for the actual
behaviour, including its exploration and its risks, which makes
them more stable in many settings and better-suited to the
"perform well *while* learning" regime. Their cost: they cannot
freely reuse off-policy data, so they are often less
sample-efficient.

The trade-off recurs at every level of the field. PPO — the
algorithm behind most RLHF for language models — is on-policy,
and a great deal of its engineering is about squeezing more
reuse out of on-policy data. DQN and its variants are
off-policy precisely so they *can* replay. Understanding why
SARSA routes around the cliff is understanding the seed of that
entire design axis.

---

## When to use SARSA over Q-Learning

**When the agent must perform well *while* learning.** Online
control where mistakes during training are costly — a system
serving real traffic, a robot that can actually break — favours
SARSA's account of its own exploration. It optimises the policy
you are *running*, not an idealised greedy one you may never
purely follow.

**When exploration is persistent and risky.** If `ε` (or any
exploration source) never fully vanishes and bad actions are
genuinely dangerous, SARSA's safety margin is the right
behaviour. The cliff is the canonical example.

**Prefer Q-Learning (off-policy) when** you want the optimal
policy regardless of how you explore, when you need to learn
from logged or replayed data, or when exploration will be
turned off at deployment so the edge path becomes safe to
follow. Off-policy's flexibility is why the deep-RL mainstream
(DQN and descendants) is off-policy.

**Prefer neither tabular method when** the state space is large
or continuous — both hit the `O(|S| · |A|)` table wall, and the
next articles (Deep Q-Networks, policy gradients) replace the
table with a neural network.

There is also a middle ground worth knowing: **Expected SARSA**
replaces `Q(s', a')` with the *expectation* over the
ε-greedy policy's next action, `Σ_{a'} π(a'|s')·Q(s', a')` —
lower variance than SARSA, and it reduces to Q-Learning when
the policy is greedy. It is a clean unification of the two
update rules, and a common practical default.

---

## What comes next

Part 3 of the Reinforcement Learning track is **Deep Q-Networks
(DQN)** — the algorithm that broke the `O(|S| · |A|)` table
wall both Q-Learning and SARSA hit. Instead of storing a value
for every state-action pair, DQN *approximates* `Q(s, a)` with
a neural network, letting it learn from raw, high-dimensional
state (Atari pixels, for instance). It keeps Q-Learning's
off-policy `max` and adds the two stabilising tricks —
**experience replay** and a **target network** — that made deep
value-based RL actually work. It is where tabular RL ends and
deep RL begins.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**sarsa.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/08-reinforcement-learning/02-sarsa/sarsa.py)

Run it with:

```bash
pip install numpy
python sarsa.py
```

It needs only `numpy`. The script implements SARSA from scratch
on the identical Cliff Walking grid-world used for Q-Learning
in Part 1 — the on-policy TD update, fixed-`ε` exploration, and
a head-to-head against Q-Learning on the same environment —
then prints both online returns and both learned policies as
arrow grids so the safe-vs-edge contrast is visible at a
glance. The headline insight worth pinning to the wall:
**SARSA differs from Q-Learning by a single term — it
bootstraps from the action it actually takes (`Q(s', a')`)
rather than the best one (`max Q(s', a')`) — which makes it
on-policy: it learns the value of the policy it is following,
exploration and all, and on the cliff that means a safer path
and a better online return (−25.1 vs −41.8) despite a longer
route**.

---

*This is Part 2 of the Reinforcement Learning track in the Algorithms in Python series. The companion script `sarsa.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 1](https://medium.com/p/43aafeeae6d8) covered Q-Learning — the off-policy method SARSA is contrasted against here. Part 3 will look at Deep Q-Networks, which replace the Q-table with a neural network to break the tabular scaling wall.*
