# Advantage Actor-Critic — Where the Two Branches Meet

### *Algorithms in Python --- Reinforcement Learning, Part 5*

---

This Reinforcement Learning track has, until now, been a story
of two rival families. The **value-based** methods —
Q-Learning, SARSA, DQN (Parts 1–3) — learn how good actions
are, `Q(s, a)`, and act greedily. The **policy-based** method
of Part 4, REINFORCE, throws values out and optimises the
policy `π(a | s)` directly. Each has a signature weakness.
Value methods can't handle continuous actions. REINFORCE can —
but its learning signal, the Monte-Carlo return, is so noisy
that it leans on a crude baseline (the *average* return) just to
function.

**Advantage Actor-Critic (A2C)** ends the rivalry by using
*both*. It runs two networks at once:

- an **actor** — the policy `π(a | s; θ)`, which decides what to
  do (the policy-based half);
- a **critic** — a value function `V(s; w)`, which estimates how
  good a state is (the value-based half).

The critic's job is to be a *better baseline*. REINFORCE
subtracted a single constant from every return; the critic
subtracts a **learned, state-dependent** estimate `V(s)`,
turning the raw return into an **advantage**:

```
A(s, a)  =  G  −  V(s)      "how much better than expected"
```

The actor is then nudged in proportion to the advantage, not
the raw return. This is the synthesis the whole track has been
building toward — and it is the template for nearly every modern
RL algorithm, PPO included. This article builds A2C from scratch
in NumPy — both networks, both updates — trains it on CartPole,
and shows it learning *faster and more reliably* than the
REINFORCE of Part 4.

---

## The actor and the critic

The two networks divide the labour:

**The actor** is exactly the policy network from Part 4:
`π(a | s; θ)`, a softmax over actions. It is updated by the same
policy gradient — but weighted by the advantage instead of the
return:

```
θ  ←  θ  +  α · ∇θ log π(a | s; θ) · A(s, a)
```

**The critic** is a value network `V(s; w)` — one output, the
estimated return from state `s`. It is trained by plain
regression toward the actual return the agent saw:

```
w  ←  w  −  β · ∇w ( V(s; w) − G )²
```

The two learn *together*, in a loop: the critic learns to
predict the actor's returns, and the actor uses the critic's
predictions to find a better-than-expected action. The critic
critiques; the actor acts on the critique.

### Why a learned baseline beats a constant one

REINFORCE's baseline was the mean return — one number for every
state. But states genuinely differ in value: a balanced pole
upright in the centre is worth more than one tilting at the
edge. Subtracting a single average over-credits actions taken
from already-good states and under-credits actions that
rescued bad ones.

The critic's `V(s)` is *state-specific*. The advantage
`G − V(s)` asks the sharper question: "was this action better
than what I expected **from this particular state**?" That is a
cleaner, better-directed learning signal — and, as the
experiment below shows, it makes learning markedly faster and
more reliable.

---

## The algorithm

```
A2C(episodes, γ, α, β):
    initialise actor θ and critic w
    for each episode:
        run an episode with π(·; θ), recording (s_t, a_t, r_t)
        for each t:  G_t = Σ_{k≥t} γ^{k−t} r_k        # returns
        for each t:  A_t = G_t − V(s_t; w)            # advantage
        actor  update:  θ ← θ + α Σ_t ∇θ log π(a_t|s_t) · A_t
        critic update:  w ← w − β Σ_t ∇w ( V(s_t; w) − G_t )²
    return θ, w
```

The advantage is treated as a fixed weight when updating the
actor — the gradient flows through `log π`, not through the
critic. Two updates, one shared trajectory. (For clarity the
script uses full-episode Monte-Carlo returns; production A2C
uses *n-step* returns — bootstrapping `V` after a few steps
rather than waiting for the episode to end — and **GAE**
generalises that further. The idea is identical; only the
return estimator changes.)

### A2C vs A3C

The algorithm was popularised in its **asynchronous** form,
**A3C** (Mnih et al., 2016): many actor-learners run in parallel
on separate copies of the environment, each computing gradients
and updating a shared set of weights asynchronously. The
parallelism decorrelates the data — playing the role that
experience replay plays for DQN — so no replay buffer is needed.

**A2C** is the **synchronous** simplification: run the parallel
workers, wait for all of them, average their gradients, apply
one update. It turned out to be as effective as A3C and simpler,
which is why it's the more common starting point today. The
"advantage actor-critic" core — actor, critic, advantage — is
identical; A2C and A3C differ only in how they gather data.

---

## A worked example: A2C on CartPole

The companion script trains an actor (`4 → 64 → 2`) and a
critic (`4 → 64 → 1`) together on CartPole, the same task as
Parts 3 and 4.

```
DEMO 1 --- Advantage Actor-Critic learns CartPole
  Actor : 4 -> 64 -> 2 (softmax)   pi(a | s)   -- what to do
  Critic: 4 -> 64 -> 1 (value)     V(s)        -- how good a state is
  Episodes: 500  gamma=0.99  lr=1e-2 (both)  trained together

  Mean return by 50-episode block (max possible = 500):
    episodes   1- 50 :  149.8  ############
    episodes  51-100 :  287.9  #######################
    episodes 101-150 :  417.8  ##################################
    episodes 151-200 :  496.2  #########################################
    episodes 201-250 :  495.5  #########################################
    episodes 251-300 :  500.0  #########################################
    episodes 301-350 :  498.8  #########################################
    episodes 351-400 :  356.4  #############################
    episodes 401-450 :  500.0  #########################################
    episodes 451-500 :  500.0  #########################################
```

It climbs to a perfect 500 by episode 150 and holds — apart from
one honest wobble around episode 380, after which it recovers
and stays maxed. (Final mean return over the last 100 episodes:
500.0.)

### The headline: A2C vs REINFORCE

Is the critic actually pulling its weight? Run the *same* setup
twice — identical network, identical learning rate — changing
only the baseline. REINFORCE uses the constant mean; A2C uses
the learned critic `V(s)`.

```
DEMO 2 --- A2C vs REINFORCE: faster and more reliable
  Episodes to 'solve' (100-episode mean return >= 475), 400 eps/seed:

    seed 0:   A2C  @ 209         REINFORCE  not solved
    seed 1:   A2C  @ 246         REINFORCE  @ 377
    seed 2:   A2C  @ 224         REINFORCE  not solved
    seed 3:   A2C  @ 196         REINFORCE  @ 323

  A2C solved 4/4 seeds (avg 218 ep);  REINFORCE solved 2/4.
  The critic's state-dependent baseline is a cleaner learning signal.
```

Across four seeds, A2C solved every one, in ~218 episodes on
average. Plain REINFORCE — same architecture, only a constant
baseline — solved just two of the four within the 400-episode
budget, and more slowly when it did. The state-dependent critic
turns a temperamental learner into a dependable one.

(A note on honesty: the textbook reason for the critic is
*variance reduction*. On CartPole, where every step earns the
same `+1`, the raw variance gap is small — the return mostly
reflects how far into the episode you are, not which state
you're in. The benefit instead surfaces exactly where it
matters in practice: speed and reliability of learning.)

### The critic learns to predict return

What did the critic actually learn? Track its estimate of the
*start* state's value across training:

```
DEMO 3 --- The critic learned to predict return
  V(start state) over training -- it tracks the policy's rising value:
    episode   0:   V(s0) =    0.1
    episode 100:   V(s0) =   53.9
    episode 200:   V(s0) =   98.3
    episode 300:   V(s0) =  100.1
    episode 400:   V(s0) =   75.5
    episode 499:   V(s0) =   99.8

  Final V(start) = 99.8  vs  the ~99 discounted return of a
  full 500-step episode -- the critic predicts the return accurately.

  Greedy rollouts (take argmax): [500, 500, 500]  -> mean 500 / 500 steps
```

The critic starts clueless (`V ≈ 0`) and, as the actor improves,
learns that the start state is worth almost 100 — which is
exactly the discounted return of a 500-step episode,
`(1 − 0.99^500) / 0.01 ≈ 99`. It even *tracks the wobble*: its
estimate dips to 75 around episode 400 when the policy briefly
degrades, then recovers to 99.8. The critic isn't guessing — it
has learned to predict the actor's return.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

A2C runs two networks, so its costs are REINFORCE's plus a
critic of the same order — a small constant factor, not a
different complexity class.

**Per episode**: `O(T · (W_actor + W_critic))` — run `T` steps,
then one gradient step each for the actor and the critic over
the trajectory.

**Memory**: `O(W_actor + W_critic + T · d)` — two networks plus
one trajectory. Like REINFORCE and unlike DQN, **on-policy A2C
keeps no replay buffer**: each trajectory updates both networks
once and is discarded. (A3C adds parallel workers — more
compute, but decorrelated data and no replay.)

**The trade-off it resolves**: A2C keeps REINFORCE's ability to
handle continuous, stochastic policies, while the critic
restores much of the stability that pure policy gradients lack.
It does not match DQN's sample efficiency — on-policy methods
still can't replay old data — which is the gap that later
algorithms like SAC close.

---

## The bigger picture

Advantage Actor-Critic is not just another algorithm; it is the
**architecture** almost all of modern deep RL is built on.

- **PPO** (Proximal Policy Optimisation) is an actor-critic that
  adds a clipped objective to stop the policy changing too fast
  per update. It is the workhorse of continuous control — and
  the algorithm behind most **RLHF** for language models.
- **DDPG, TD3, SAC** are actor-critics for continuous action
  spaces, where the critic learns `Q(s, a)` and the actor is
  trained to maximise it.
- **AlphaGo / AlphaZero** pair a policy network (actor) with a
  value network (critic), guided by tree search.

Once you see the actor-critic pattern — one network proposing
actions, another judging them — you see it everywhere. The two
branches that opened this track don't just coexist in A2C; their
union is the foundation the field now stands on.

---

## What comes next

This article closes the **foundational Reinforcement Learning
track** — Q-Learning, SARSA, DQN, Policy Gradients, and now
Advantage Actor-Critic: tabular methods, deep value methods, and
both halves of the policy-gradient world, all built from
scratch.

The next section, **Advanced Reinforcement Learning**, begins
with **Proximal Policy Optimisation (PPO)** — the actor-critic
refinement that made policy-gradient methods stable and reliable
enough to train everything from robot controllers to the
aligned behaviour of large language models. Everything in this
article is the scaffolding PPO is built on.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**a2c.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/08-reinforcement-learning/05-advantage-actor-critic/a2c.py)

Run it with:

```bash
pip install numpy
python a2c.py
```

It needs only `numpy` and runs in well under a minute. Every
piece is from scratch: the CartPole dynamics, two MLPs (actor
and critic) with hand-written backprop and Adam, and the
advantage computation that couples them. The headline insight
worth pinning to the wall: **A2C is value-based and policy-based
at once — a critic `V(s)` learns how good each state is, and the
actor `π(a|s)` is trained by the advantage `G − V(s)` ("better
than expected from this state") rather than the raw return;
swapping REINFORCE's one constant baseline for this learned,
state-dependent one is the difference between solving CartPole
2 times out of 4 and solving it 4 times out of 4**.

---

*This is Part 5 of the Reinforcement Learning track in the Algorithms in Python series, and it closes the foundational RL section. The companion script `a2c.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 4](https://medium.com/p/e9b0f432ae92) covered Policy Gradient Methods (the actor) and [Part 3](https://medium.com/p/0102b41f786d) covered DQN (value learning, the critic's lineage). The next section opens with Proximal Policy Optimisation, the actor-critic method behind modern RLHF.*
