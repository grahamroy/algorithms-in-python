# Policy Gradient Methods — Learning the Policy, Not the Values

### *Algorithms in Python --- Reinforcement Learning, Part 4*

---

Every algorithm in this track so far has learned the same thing:
**values**. Q-Learning, SARSA, and DQN all estimate `Q(s, a)` —
how good each action is — and then act by taking the `argmax`.
The policy is never represented directly; it falls out of the
values as "whatever the highest Q says."

That indirection has costs. The `argmax` requires *enumerating*
actions, so it cannot handle a continuous action space (a
steering angle, a joint torque — infinitely many options). The
resulting policy is *deterministic*, when sometimes the best
behaviour is genuinely stochastic. And we optimise values as a
proxy, hoping a good value estimate yields a good policy, rather
than optimising the thing we actually care about.

**Policy gradient methods** throw out the indirection. Instead
of learning values and deriving a policy, they **parameterise
the policy itself** — `π(a | s; θ)`, a network that takes a
state and outputs a probability for each action — and optimise
`θ` directly by gradient ascent on expected return. No value
table, no `argmax`, no proxy. This is the other great branch of
reinforcement learning, and it is the one that leads, through
actor-critic methods and PPO, to the RL behind today's language
models.

This article builds the foundational policy gradient algorithm,
**REINFORCE** (Williams, 1992), entirely from scratch in NumPy —
the policy network, the softmax, the policy-gradient backprop —
and trains it on CartPole, the same task DQN balanced in Part 3.
We will see it learn a *stochastic* policy, watch the single
trick that makes it work (a **baseline**), and understand why,
despite its simplicity, it underpins the most important RL
methods in use today.

---

## Two ways to be an agent

The split is worth drawing sharply, because it organises the
whole field.

**Value-based (Q-Learning, DQN).** Learn `Q(s, a)`. Act by
`argmax_a Q(s, a)`. The policy is implicit. Naturally
off-policy (you can learn the greedy policy from any data),
which is why replay works — but limited to discrete actions and
a deterministic greedy policy.

**Policy-based (REINFORCE, and what follows).** Learn
`π(a | s; θ)` directly. Act by *sampling* from it. The policy is
explicit and, for discrete actions, is just a softmax over the
network's outputs:

```
π(a | s; θ)  =  softmax( f(s; θ) )[a]
```

where `f(s; θ)` is a neural network producing one logit per
action. For continuous actions, the network instead outputs the
parameters of a distribution (say the mean and standard
deviation of a Gaussian) and you sample from that — which is
exactly why policy gradients handle continuous control and value
methods can't.

The question is how to train `θ` when the only feedback is a
reward signal, not a labelled "correct action."

---

## The policy gradient: trial and error with a gradient

We want to maximise the expected return,

```
J(θ)  =  E[ R(τ) ]      over trajectories τ sampled by π(·; θ)
```

It looks impossible to differentiate — the return depends on the
environment's dynamics, which we don't have in closed form. The
**policy gradient theorem** (Sutton et al., 2000) gives a
beautiful way around this. The gradient of `J` can be written as
an *expectation we can sample*:

```
∇θ J(θ)  =  E[  Σ_t  ∇θ log π(a_t | s_t; θ) · G_t  ]
```

where `G_t` is the return (sum of discounted future rewards)
following step `t`. Crucially, this involves only the gradient
of our *own policy* — `∇θ log π` — and not the gradient of the
environment. We can estimate the whole thing by just running
episodes and averaging.

The intuition is the entire algorithm in one sentence:

> **Increase the log-probability of each action taken, in
> proportion to how good the return that followed it was.**

Actions followed by high return get their probability pushed
up; actions followed by low return get pushed down. Repeat, and
the policy drifts toward whatever earns reward. It is trial and
error — but with a gradient pointing the way.

---

## REINFORCE, in full

REINFORCE is the direct Monte-Carlo implementation: run a whole
episode, compute the actual returns, take one gradient step.

```
REINFORCE(episodes, α, γ):
    initialise policy network θ
    for each episode:
        run an episode with π(·; θ), recording (s_t, a_t, r_t)
        for each t:  G_t = Σ_{k≥t} γ^{k−t} r_k        # returns
        update:  θ ← θ + α Σ_t ∇θ log π(a_t | s_t; θ) · G_t
```

In code, the update is remarkably close to supervised
classification. The gradient of `log π` for a softmax policy is
`(one_hot(a) − probs)`, so the gradient step is a
*return-weighted* version of the cross-entropy gradient: treat
the action taken as the "label," and weight each example by its
return `G_t`. A good episode pulls the policy toward the actions
it took; a bad one pushes away. The companion script computes
exactly this, by hand.

There is no replay buffer and no target network. REINFORCE is
**on-policy**: each trajectory is generated by the current
policy, used for one update, and thrown away. That simplicity is
a virtue and, as we'll see, a weakness.

---

## The problem that nearly kills it: variance

The policy gradient is *unbiased* but *high-variance*. The
return `G_t` of a single episode is a noisy estimate — the same
policy can earn 500 one episode and 30 the next, just from
randomness in the starts and the sampling. Multiply a noisy
return into every gradient, and the updates are noisy enough to
drown out the signal.

The standard fix is a **baseline**. Subtract a reference value
`b` from every return before weighting the gradient:

```
∇θ J(θ)  =  E[  Σ_t  ∇θ log π(a_t | s_t; θ) · (G_t − b)  ]
```

Remarkably, this leaves the gradient **unbiased** for any `b`
that doesn't depend on the action — but choosing `b` to be the
average return dramatically *cuts the variance*. The quantity
`(G_t − b)` is called the **advantage**: how much better than
the baseline this action did. The shift from "return" to
"better-than-average" is the difference between a signal that
teaches and one that doesn't, as the experiment below shows
starkly.

---

## A worked example: REINFORCE on CartPole

The companion script trains a policy network (`4 → 128 → 2`,
softmax over the two actions) on CartPole — same task, same
environment as the DQN article — using pure Monte-Carlo
REINFORCE with one update per episode.

```
DEMO 1 --- REINFORCE learns CartPole (a stochastic policy)
  Policy: 4 -> 128 -> 2, softmax over actions  ->  pi(a | s)
  Episodes: 800  gamma=0.99  lr=1e-2  Monte-Carlo, one update/episode

  Mean return by 80-episode block (max possible = 500):
    episodes   1- 80 :  246.5  ####################
    episodes  81-160 :  379.4  ###############################
    episodes 161-240 :  481.7  ########################################
    episodes 241-320 :  468.5  #######################################
    episodes 321-400 :  469.7  #######################################
    episodes 401-480 :  477.3  #######################################
    episodes 481-560 :  489.2  ########################################
    episodes 561-640 :  495.4  #########################################
    episodes 641-720 :  499.1  #########################################
    episodes 721-800 :  487.5  ########################################

  Final mean return (last 100 episodes): 490.0
```

The first few episodes sit near the random return of ~20. But
REINFORCE crosses the classic "solved" threshold of 195 by
**episode 25**, and the first 80-episode block already averages
246 as it rockets up. It plateaus near 490 — close to the 500
cap — and holds. On this task it actually learns *faster* than
the DQN of Part 3, because the policy ("push the way the pole is
falling") is simple to represent directly, even though the
underlying gradient is far noisier.

### The headline experiment: remove the baseline

Run the *exact same* REINFORCE and change one thing: don't
subtract the mean return. (Both versions normalise the returns'
scale, so the learning rate is comparable — the only difference
is the baseline subtraction.)

```
DEMO 2 --- Why a baseline matters (variance reduction)
  The SAME REINFORCE; the only change is whether we subtract the
  mean return (the baseline) before weighting the gradient.

  With baseline    final mean return:  490.0
  Without baseline final mean return:    9.4

  The baseline lifts the final return by 480.6. Without it every
  return is positive, so every action taken is reinforced and the policy
  can't tell good from bad. The baseline turns "return" into "better or
  worse than average" -- the signal that actually teaches the policy.
```

This is dramatic — and instructive. In CartPole every reward is
`+1`, so every return is *positive*. Without a baseline, every
action ever taken gets its probability pushed **up**; the policy
has no way to distinguish the good actions from the merely
survived-a-bit ones, and it collapses to a degenerate policy
worse than random (9.4). Subtract the average, and suddenly
below-average actions get a *negative* advantage and are pushed
down. The baseline isn't a minor tweak; it is what makes policy
gradients work at all.

### The policy is a distribution

Finally, the defining contrast with value-based methods. DQN
outputs Q-values and takes an `argmax` — one action, always.
REINFORCE outputs a *probability distribution*:

```
DEMO 3 --- The learned policy is a DISTRIBUTION, not an argmax
  pi(a | s) at a few visited states (push-left, push-right):
    state 0:  left 0.00   right 1.00
    state 1:  left 0.01   right 0.99
    state 2:  left 0.18   right 0.82
    state 3:  left 0.87   right 0.13
    state 4:  left 0.32   right 0.68

  Greedy rollouts (take argmax): [500, 500, 500]  -> mean 500 / 500 steps
```

Where the pole is clearly falling one way, the policy is nearly
certain (0.00 / 1.00). Where the state is ambiguous, it hedges
(0.32 / 0.68). It has learned a *graded* strategy, not a lookup
table — and taking the greedy action from it balances the pole
for the full 500 steps, every time.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

REINFORCE's cost profile is lighter than DQN's in memory but
heavier in samples.

**Per episode**: `O(T · W)` — run `T` steps (a forward pass each
to sample an action) and take one gradient step over the
trajectory, where `W` is the number of network weights.

**Memory**: `O(W + T · d)` — the network plus a *single*
trajectory of `T` transitions. There is **no replay buffer**:
because REINFORCE is on-policy, each trajectory is used once and
discarded. Memory is small — but so is data reuse.

**Sample efficiency**: poor, and this is the central trade-off.
DQN replays each transition many times; REINFORCE sees each once.
On-policy learning cannot reuse old data, because the data must
come from the *current* policy. This is why policy gradient
methods are often hungry for environment interaction.

**Variance**: high, mitigated but not eliminated by the
baseline. The Monte-Carlo return waits until the episode ends
and credits *every* action with the *whole* remaining return —
a coarse assignment that the next article sharpens.

---

## When to reach for policy gradients

**Use policy gradients when:**

- **Actions are continuous.** This is the decisive reason. A
  policy network can output the mean of a Gaussian over a
  continuous action; a Q-function's `argmax` cannot search a
  continuum. Robotics and control live here.
- **You want a stochastic policy** — for exploration, for games
  where unpredictability matters, or where the optimal behaviour
  is genuinely mixed.
- **You want to optimise the policy directly**, including with
  constraints, which is what the modern variants (TRPO, PPO) are
  built to do safely.

**Prefer value-based methods when** actions are discrete and few,
and sample efficiency matters — DQN's replay makes far better use
of expensive environment steps.

**The lineage.** REINFORCE is the root. Add a learned value
function as the baseline and you get **actor-critic** (Part 5).
Constrain the size of each policy update and you get **TRPO**
and then **PPO** (Schulman et al., 2017) — and PPO is the
algorithm behind most reinforcement learning from human feedback
(RLHF) used to align large language models. The noisy little
gradient you just built is the direct ancestor of how today's
models are tuned.

---

## What comes next

Part 5 is **Advantage Actor-Critic (A2C / A3C)** — the synthesis
of this article and the value-based ones. REINFORCE's weakness is
the high-variance Monte-Carlo return and its crude baseline.
Actor-critic replaces that baseline with a *learned value
function* — a **critic** that estimates how good each state is —
and uses it to compute a low-variance **advantage** for the
**actor** (the policy). It is value-based and policy-based at
once: the critic stabilises the actor, and the actor handles the
continuous, stochastic policies the critic alone never could. It
is where the two branches of this track finally meet.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**reinforce.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/08-reinforcement-learning/04-policy-gradient-methods/reinforce.py)

Run it with:

```bash
pip install numpy
python reinforce.py
```

It needs only `numpy` and runs in well under a minute. Every
piece is from scratch: the CartPole dynamics, a policy network
with a `tanh` hidden layer and hand-written backprop, an Adam
optimiser, and the policy-gradient update — which turns out to be
a return-weighted version of the softmax cross-entropy gradient.
The headline insight worth pinning to the wall: **policy gradient
methods skip the value table and optimise the policy `π(a|s; θ)`
directly, pushing up the probability of actions that led to high
return; the one trick that makes this work is subtracting a
baseline so the signal becomes "better or worse than average"
(the advantage) rather than a raw, all-positive return — which on
CartPole is the difference between 490 and 9**.

---

*This is Part 4 of the Reinforcement Learning track in the Algorithms in Python series. The companion script `reinforce.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 3](https://medium.com/p/0102b41f786d) covered Deep Q-Networks, the value-based method this one is contrasted against. Part 5 will look at Advantage Actor-Critic, which combines a learned value function (critic) with the policy gradient (actor).*
