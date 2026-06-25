# Deep Q-Networks — When the Q-Table Won't Fit

### *Algorithms in Python --- Reinforcement Learning, Part 3*

---

Parts 1 and 2 of this track — Q-Learning and SARSA — both ended
at the same wall. They store one number for every
state-action pair in a table, `Q(s, a)`, and that table has a
hard size limit: `O(|S| · |A|)`. On the 48-square Cliff Walking
grid that is fine. On almost anything real it is hopeless. The
moment the state stops being a handful of discrete squares — the
moment it becomes, say, *four continuous numbers* — the table
needs infinitely many rows and the whole approach collapses.

**Deep Q-Networks (DQN)** break that wall with one idea: stop
*storing* `Q(s, a)` and start *approximating* it. Replace the
lookup table with a neural network that takes a state in and
predicts a Q-value for each action. The table that needed a row
per state becomes a function that generalises across states it
has never seen. This is the step from tabular RL to *deep* RL,
and it is the idea that let DeepMind's 2015 agent learn to play
49 Atari games from raw pixels at human level (Mnih et al.,
*Human-level control through deep reinforcement learning*,
Nature 2015).

But swapping a table for a network breaks the gentle
convergence guarantees the table gave us, and a naive
"neural-network Q-learning" is famously unstable — it diverges
as often as it learns. DQN's real contribution is the *two
tricks* that make function approximation actually work:
**experience replay** and a **target network**. This article
builds all of it from scratch in NumPy — the network, the
backprop, the replay buffer, the target network — trains it on
CartPole (a task with a genuinely continuous state, where a
table is impossible), and runs the single cleanest experiment
in deep RL: turn the target network off and watch learning
fall apart.

---

## The table wall, made concrete

Cliff Walking had 48 states. CartPole — the classic
balance-a-pole-on-a-cart task — has a state of **four
continuous numbers**: cart position, cart velocity, pole angle,
and pole angular velocity. Each is a real number. The state
`(0.013, −0.227, 0.041, 0.388)` will, in all likelihood, *never
occur again exactly*. There is no row to look up and no row to
update. A Q-table doesn't just get large here; it is the wrong
data structure entirely.

What we need is a function that *generalises*: that, having seen
states near `(0.013, −0.227, 0.041, 0.388)`, can produce a
sensible Q-value for it the first time it appears. That is
exactly what a neural network does. So we define

```
Q(s, a; θ)  ≈  a neural network with weights θ
              input:  the state s (4 numbers)
              output: one Q-value per action (here, 2)
```

and learn the weights `θ` instead of table entries. The network
is small — for CartPole, `4 → 64 → 64 → 2` is plenty — but the
shift is profound: from memorising values to *learning a
function*.

---

## From a tabular update to a gradient step

Recall Q-Learning's tabular update. It nudged a stored value
toward a target:

```
Q(s, a)  ←  Q(s, a) + α · ( r + γ·max_{a'} Q(s', a') − Q(s, a) )
```

With a network there is no entry to nudge — there are weights.
So we turn the *same* temporal-difference idea into a
**regression problem**. The thing we want `Q(s, a; θ)` to equal
is the TD target

```
y  =  r + γ · max_{a'} Q(s', a'; θ)
```

and we train the network to predict it by minimising the
squared error between prediction and target:

```
Loss(θ)  =  ( Q(s, a; θ) − y )²
```

One step of gradient descent on this loss moves the network's
prediction toward the target — the function-approximation
analogue of one tabular update. The `max_{a'}` is still there,
so DQN is still **off-policy**, learning the value of the greedy
policy exactly as Q-Learning did. (SARSA's on-policy variant has
a deep analogue too, but DQN follows the Q-Learning branch.)

If that were the whole story, this article would be one page
long. It isn't, because that naive version *does not work*.

---

## Why naive neural Q-learning falls apart — and the two fixes

Two things go wrong the moment you train a network this way, and
each has a fix that is now standard equipment.

**Problem 1: consecutive samples are deeply correlated.** In an
episode, each state is almost identical to the last. If you
train the network on transitions in the order they happen, you
feed it long runs of near-identical, highly-correlated data —
exactly what gradient descent handles worst. The network
overfits to whatever it's seeing right now and forgets what it
learned a hundred steps ago.

> **Fix — experience replay.** Store every transition
> `(s, a, r, s', done)` in a large buffer, and train on *random
> minibatches sampled from it*. Sampling randomly breaks the
> temporal correlation, and reusing each transition many times
> makes the method far more sample-efficient. (Experience replay
> predates DQN — Long-Ji Lin proposed it in 1992 — but DQN is
> what made it essential.)

**Problem 2: the target chases the weights.** Look at the loss
again. The target `y = r + γ·max Q(s', a'; θ)` depends on the
*same weights θ* we are updating. Every gradient step changes
`θ`, which changes the target, which changes the next gradient
step's goal. You are chasing a target that moves every time you
step toward it — a feedback loop that sends the values
spiralling.

> **Fix — a target network.** Keep a *second*, frozen copy of
> the network with weights `θ⁻`, and compute the target from it:
> `y = r + γ·max Q(s', a'; θ⁻)`. The online network `θ` is
> trained every step; the target network `θ⁻` is only updated
> *occasionally* — every few hundred steps we copy `θ` into
> `θ⁻`. Now the target holds still long enough for the online
> network to actually reach it. The target network was the key
> addition in the 2015 Nature paper, and it is the difference
> between learning and diverging — as the experiment below shows
> directly.

These two fixes are the whole reason DQN works. Everything else
is standard neural-network training.

---

## The full algorithm

```
DQN(episodes, γ, lr, batch, C):
    initialise online network θ randomly
    initialise target network θ⁻ ← θ
    initialise empty replay buffer D
    for each episode:
        s = reset environment
        while not done:
            a = ε-greedy action from Q(s, ·; θ)
            take a, observe r, s', done
            store (s, a, r, s', done) in D
            s = s'

            sample a random minibatch from D
            y = r + γ · max_{a'} Q(s', a'; θ⁻) · (1 − done)
            take a gradient step on ( Q(s, a; θ) − y )²
            every C steps:  θ⁻ ← θ          # refresh target network
    return θ
```

The `(1 − done)` term zeroes the bootstrap at terminal states —
there is no future return after the episode ends. Beyond that,
this is Q-Learning's loop with three additions: store to the
buffer, train on a random batch, and periodically refresh the
target network.

---

## A worked example: balancing CartPole from scratch

The companion script implements all of the above in NumPy — a
two-hidden-layer MLP with hand-written backprop and an Adam
optimiser, a ring-buffer for replay, and a separate target
network — and trains it on CartPole, where the agent earns +1
for every step it keeps the pole upright, up to a cap of 500.

```
DEMO 1 --- DQN learns to balance CartPole
  State : 4 continuous dims (cart pos/vel, pole angle/vel)
          a Q-table is impossible -- the state never repeats
  Network: 4 -> 64 -> 64 -> 2   (one Q-value per action)
  Episodes: 250  gamma=0.99  lr=1e-3  batch=64  replay=10000  target sync=200 steps

  Mean return by 25-episode block (max possible = 500):
    episodes   1- 25 :   21.8  #
    episodes  26- 50 :   22.7  #
    episodes  51- 75 :   24.2  ##
    episodes  76-100 :   35.2  ##
    episodes 101-125 :   45.0  ###
    episodes 126-150 :   64.9  #####
    episodes 151-175 :  144.8  ############
    episodes 176-200 :  187.7  ###############
    episodes 201-225 :  168.7  ##############
    episodes 226-250 :  262.7  #####################

  Final mean return (last 50 episodes): 215.7
```

The agent starts where a random policy sits — a return of about
22, meaning the pole topples in roughly a fifth of a second.
For the first hundred episodes almost nothing happens: it is
filling the replay buffer and exploring with a high `ε`. Then,
once there is enough varied experience to learn from, the return
climbs steeply — past 64, past 144, to a final average of 215.
The pole now stays up for over four seconds at a stretch. No
table could represent this policy; the network learned a
*function* over the continuous state.

### The headline experiment: turn off the target network

Here is the cleanest demonstration in deep RL of why a design
choice matters. Run the *exact same* DQN — same seed, same
network, same hyperparameters, same replay buffer — and change
one thing: compute the target from the live online network
instead of the frozen target network.

```
DEMO 2 --- Why the target network matters (ablation)
  The SAME DQN, the only change is whether the bootstrap target
  comes from a slowly-updated copy of the network or the live one.

  With target network    final mean return:  215.7   (stable)
  Without target network final mean return:   75.0   (chases its own tail)

  The target network lifts the final return by 140.7. Without it the
  target moves every gradient step and learning is far less stable.
```

Same algorithm, one knob, nearly **3× the performance**. With
the target frozen between refreshes, the online network has a
stable goal to regress toward and climbs to 215. With the target
recomputed from the live weights every step, it chases its own
moving estimate and stalls at 75. This is the feedback loop from
Problem 2, made visible — and it is exactly why the target
network, not present in the original 2013 DQN, was the headline
addition in the 2015 version.

### The learned policy in action

Finally, run the trained network greedily (`ε = 0`) on fresh
random starts it never trained on:

```
DEMO 3 --- The learned greedy policy in action
  Greedy rollout (epsilon=0), seed 101: balanced 272 / 500 steps
  Greedy rollout (epsilon=0), seed 202: balanced 253 / 500 steps
  Greedy rollout (epsilon=0), seed 303: balanced 257 / 500 steps

  Mean over 3 greedy rollouts: 261 / 500 steps
```

On states drawn from seeds it had never encountered, the policy
balances the pole for ~261 steps on average — the
generalisation a table could never give. (Train longer and this
climbs toward the 500 cap; 250 episodes is enough to make the
point in a few seconds.)

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

DQN trades the table's memory wall for the cost of training a
network — a very different profile from the tabular methods.

**Per gradient step**: `O(B · W)` where `B` is the batch size
and `W` is the number of network weights — one forward and one
backward pass over a minibatch. This replaces the tabular
update's `O(|A|)`.

**Memory**: `O(W + N·d)` — the network weights `W` plus the
replay buffer of `N` transitions, each of size `d`. Crucially,
this **no longer depends on `|S|`**. That is the whole point: a
continuous or astronomically large state space costs the same as
a small one, because the network *generalises* instead of
*enumerating*. The table wall is gone.

**Sample efficiency**: replay lets every transition be reused in
many minibatches, so DQN squeezes far more learning out of each
environment step than one-shot tabular updates — important when
environment steps are expensive (a robot, a slow simulator).

The trade-off is that the convergence guarantees of tabular
Q-Learning are gone. With function approximation, a bootstrapped
target, and off-policy data — Sutton's "deadly triad" — DQN can
diverge. The two tricks tame it in practice, but DQN is an
engineering achievement, not a theorem.

---

## When to use DQN — and its limits

**Use DQN when** the state is large or continuous (so a table is
out) **and** the actions are a small discrete set (push
left/right, one of a few moves). That combination — rich states,
few actions — is DQN's sweet spot, and it is a large slice of
real problems.

**Reach for something else when:**

- **Actions are continuous** (a steering angle, a joint torque).
  DQN's `max_{a'}` requires enumerating actions, which is
  impossible over a continuum. This is the domain of policy
  gradients (Part 4) and actor-critic methods like DDPG and SAC.
- **You need the policy itself**, perhaps stochastic — again
  policy-gradient territory.
- **The problem is small and discrete** — then tabular
  Q-Learning or SARSA is simpler, exact, and has the convergence
  guarantees DQN gives up.

**The DQN family.** Vanilla DQN kicked off a wave of
improvements, each fixing a specific flaw: **Double DQN** (van
Hasselt et al., 2016) corrects the `max` operator's tendency to
*overestimate* Q-values; **Dueling DQN** (Wang et al., 2016)
splits the network into state-value and advantage streams;
**Prioritized Experience Replay** (Schaul et al., 2016) samples
surprising transitions more often; and **Rainbow** (Hessel et
al., 2018) combines six such extensions into one agent. They all
share the spine you just built.

---

## What comes next

Part 4 of the Reinforcement Learning track turns to **Policy
Gradient Methods**. DQN learns *values* and derives a policy
from them with `argmax` — which is why it can't handle
continuous actions. Policy gradients flip this around: they
parameterise and optimise the **policy directly**, with no
`max` and no value table, learning a probability distribution
over actions that can be discrete *or* continuous. It is the
other great branch of modern RL — the one that, via PPO,
underlies the reinforcement learning behind today's language
models.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**dqn.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/08-reinforcement-learning/03-deep-q-networks/dqn.py)

Run it with:

```bash
pip install numpy
python dqn.py
```

It needs only `numpy` and runs in well under a minute. Every
piece is from scratch: the CartPole dynamics, a two-layer MLP
with He initialisation and hand-written backprop, an Adam
optimiser, a ring-buffer for experience replay, and a separate
target network. (One honest detail: the script uses a plain
squared-error loss, whereas the Atari DQN clipped both rewards
and the TD error to ±1 — appropriate for pixel games, but it
would throttle learning on CartPole's unclipped returns, which
reach ~100.) The headline insight worth pinning to the wall:
**DQN replaces Q-Learning's table with a neural network so the
state can be continuous — and the two tricks that make that work
are experience replay (sample random past transitions to break
correlation) and a target network (bootstrap from a frozen copy
so the target holds still); turning the target network off alone
drops CartPole performance from 215 to 75**.

---

*This is Part 3 of the Reinforcement Learning track in the Algorithms in Python series. The companion script `dqn.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 1](https://medium.com/p/43aafeeae6d8) and Part 2 covered tabular Q-Learning and SARSA — the methods DQN scales past. Part 4 will look at Policy Gradient Methods, which optimise the policy directly and handle continuous actions.*
