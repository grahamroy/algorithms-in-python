# Proximal Policy Optimisation — The Clip That Made Policy Gradients Reliable

### *Algorithms in Python --- Advanced Reinforcement Learning, Part 1*

---

The foundational track of this series ended with **Advantage
Actor-Critic** (A2C): an actor `π(a | s)` proposing actions, a
critic `V(s)` judging them, the advantage `G − V(s)` training
the actor. It worked — but it left two problems unsolved, and
both are about the *size of the step you take*.

First, **a single update can be too big**. Policy gradients have
no built-in limit on how far they move the policy. One unlucky
batch with a large advantage can shove the policy somewhere
terrible, and — because the next batch of data is collected by
that now-worse policy — it may never recover. We saw exactly
this in Part 4: REINFORCE climbing to 450 and then *collapsing*.

Second, **on-policy data is used once and thrown away**. A2C and
REINFORCE collect a batch, take one gradient step, and discard
it. They have to, because after one step the policy has changed
and the old data no longer reflects it. That makes them
sample-hungry.

**Proximal Policy Optimisation (PPO)** (Schulman et al., 2017)
solves both with a single idea: a **clipped objective** that
refuses to let the policy move too far from where it started
each update. Keeping the new policy *proximal* to the old one
caps the step size (fixing problem one) — and, because the
policy is now guaranteed to stay close to the data-collecting
policy, you can safely take **many gradient steps on the same
batch** (fixing problem two). PPO is the algorithm that made
policy gradients dependable enough to become the default for
continuous control and the backbone of **RLHF** for language
models.

This article builds PPO from scratch in NumPy — on top of the
A2C actor-critic — and shows the clip doing its job: turning a
collapse-prone learner into a stable, sample-efficient one.

---

## The idea: stay in a trust region

The principle behind PPO is older than PPO. **TRPO** (Trust
Region Policy Optimisation, Schulman et al., 2015) made it
precise: don't just maximise expected return — maximise it
*subject to a constraint* that the new policy stays within a
"trust region" of the old one, measured by KL divergence. Inside
that region, the improvement estimate is trustworthy; outside it,
all bets are off.

TRPO enforces this with a hard constraint and second-order
optimisation — powerful but complicated. PPO's insight is that
you can get almost all the benefit with a **first-order
approximation**: instead of a hard KL constraint, just *clip the
objective* so there's no incentive to leave the region. Same
goal — bounded, trustworthy steps — with plain gradient descent.

---

## The probability ratio

Everything in PPO is expressed through one quantity, the
**probability ratio** between the new policy and the old one for
the action that was actually taken:

```
r(θ)  =  π_θ(a | s)  /  π_θ_old(a | s)
```

`r = 1` means the new policy assigns the action exactly the same
probability as the old one — no change. `r = 1.3` means the new
policy is 30% more likely to take it; `r = 0.7`, 30% less. The
ratio measures *how far the policy has moved* on this action, and
it is the lever PPO controls.

The plain policy-gradient objective, written with the ratio, is
`r(θ) · A` — push `r` up when the advantage `A` is positive (the
action was good), down when it's negative. The problem is there's
nothing stopping `r` from shooting to 5 or 0.05 in one update.

---

## The clipped objective

PPO's objective clips the ratio into a band around 1:

```
L(θ)  =  E[  min( r(θ)·A,  clip(r(θ), 1−ε, 1+ε)·A )  ]
```

with `ε` typically 0.2 — a band of `[0.8, 1.2]`. Two things are
happening, and the `min` is the clever part.

Take a **good action** (`A > 0`). We want to raise `r`. The first
term `r·A` keeps rewarding bigger `r` forever. The second term
caps `r` at `1 + ε`: beyond that, `clip(r)·A` is flat. The `min`
takes the smaller of the two, so once `r` exceeds `1 + ε` the
objective stops increasing — **no gradient, no incentive to push
further**. The policy is allowed to become *somewhat* more likely
to take the good action, but not arbitrarily so, in one update.

For a **bad action** (`A < 0`), the logic mirrors: the clip stops
the policy from driving the action's probability down past
`1 − ε` in a single update. Either way, the `min` makes the
objective a *pessimistic lower bound* — it never rewards moving
the ratio outside the trust band. That is the whole mechanism.

### Why this unlocks data reuse

Here's the payoff that's easy to miss. Because the clip
guarantees the policy can't wander far from `π_old` in one
update, the rollout `π_old` collected stays *approximately valid*
for several gradient steps. So PPO does something on-policy
methods can't: it takes **multiple epochs** of minibatch updates
on the *same* rollout before collecting fresh data. The clip is
what makes that reuse safe — without it, epoch after epoch would
drag the policy further and further from the data until the
estimate is meaningless.

---

## The algorithm

```
PPO(iterations, epochs, ε, γ, λ):
    initialise actor θ and critic w
    for each iteration:
        run π_θ for N steps; record states, actions, rewards
        record old log-probs  log π_θ_old(a|s)
        compute advantages A with GAE(γ, λ) using the critic
        for several EPOCHS, over minibatches:
            r = exp( log π_θ(a|s) − log π_θ_old(a|s) )
            actor  loss = − min( r·A, clip(r, 1−ε, 1+ε)·A )
            critic loss = ( V(s) − return )²
            gradient-step both
    return θ, w
```

The advantages use **GAE** (Generalised Advantage Estimation,
Schulman et al., 2016) — an exponentially-weighted average of
n-step advantages that trades bias against variance with a
parameter `λ`; the companion script implements it directly. The
old log-probs are snapshotted before the epochs begin, so the
ratio `r` is always measured against the policy that *collected*
the data.

---

## A worked example: PPO on CartPole

The companion script trains an actor (`4 → 64 → 2`) and critic
(`4 → 64 → 1`) on CartPole — the same environment as the last
three articles — collecting 1024-step rollouts and doing 10
epochs of clipped updates on each.

```
DEMO 1 --- PPO learns CartPole (clipped, multi-epoch)
  Actor : 4 -> 64 -> 2 (softmax)     Critic: 4 -> 64 -> 1 (value)
  rollout=1024 steps  epochs=10  minibatch=64  clip=0.2  gamma=0.99  lambda=0.95

  Moving-average return vs environment steps (max possible = 500):
      1024 steps :   33.0  ##
      5120 steps :  105.9  ########
      9216 steps :  217.2  ##################
     13312 steps :  351.1  #############################
     17408 steps :  445.8  #####################################
     21504 steps :  474.6  #######################################
     25600 steps :  419.1  ##################################
     29696 steps :  419.1  ##################################
     33792 steps :  459.6  ######################################
     37888 steps :  482.6  ########################################
```

PPO climbs steadily to a near-perfect return in under 40,000
environment steps, with only a shallow wobble around step 25k
that it recovers from — no collapse. Note the x-axis: this is
plotted against *environment steps*, the currency that matters
when interaction is expensive.

### The headline: what the clip is for

Run the *same* PPO — same rollouts, same 10 epochs — and simply
remove the clip, optimising the raw `r · A` instead.

```
DEMO 2 --- The clip is what makes multi-epoch reuse safe
  With clip   : rises to 483 and holds it           (stable)
  Without clip: peaks at 407, then collapses to 258  (unstable)
```

Without the clip, the ten epochs of reuse are exactly the
problem. Each epoch drags the policy further from the rollout
that justified the update; by the time it has cycled the data ten
times, the policy has overshot. It climbs promisingly to 407 —
then collapses and thrashes, ending at 258. With the clip, every
update is held proximal to the old policy, the reuse stays valid,
and the curve rises to 483 and stays there. The clip isn't a
tweak; it is the thing that makes multi-epoch PPO work at all.

### The payoff: sample efficiency

Why bother reusing data? Because environment interaction is the
expensive part. The more epochs PPO squeezes from each rollout,
the fewer rollouts — fewer environment steps — it needs.

```
DEMO 3 --- More epochs per rollout = fewer env steps to solve
     1 epoch(s)/rollout : not solved in budget
     5 epoch(s)/rollout : solved in 30720 env steps
    10 epoch(s)/rollout : solved in 22528 env steps
```

With a single epoch per rollout — essentially the on-policy A2C
regime, using each batch once — PPO doesn't even solve CartPole
within the budget. Five epochs solve it in ~31k steps; ten, in
~23k. Reusing the data that the clip makes *safe* to reuse is
precisely where the sample efficiency comes from. (The trained
policy balances the pole for the full 500 steps on every greedy
rollout.)

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

PPO's cost is A2C's, multiplied by the epochs-and-minibatches it
runs over each rollout.

**Per iteration**: `O(N · W)` to collect a rollout of `N` steps,
then `O(epochs · N · W)` for the updates — the reuse factor
`epochs` is the multiplier that buys sample efficiency at the
cost of compute.

**Memory**: `O(W + N · d)` — the two networks plus one rollout of
`N` transitions. Like A2C and unlike DQN, there's **no replay
buffer**; the "old policy" snapshot is just the log-probs stored
with the rollout.

**The trade-off PPO strikes**: it spends *more compute per
environment step* (multiple epochs) to spend *fewer environment
steps overall*. When the simulator is the bottleneck — robotics,
expensive rollouts, an LLM generating completions — that is
exactly the right trade.

---

## When to use PPO — and why it's everywhere

**PPO is the sensible default** for policy-gradient RL. It is
robust to hyperparameters, simple to implement (one clip on top
of actor-critic), works for discrete *and* continuous actions,
and rarely melts down. That combination is why it dominates:

- **Continuous control** — robotics, locomotion, simulated
  physics — is overwhelmingly PPO.
- **RLHF for language models** uses PPO as the optimiser: the
  policy is the LLM, the reward comes from a learned preference
  model, and the clip keeps the model from drifting too far from
  its starting point in any one update (InstructGPT, Ouyang et
  al., 2022).
- **Game-playing agents** (OpenAI Five for Dota 2, among others)
  scaled PPO to enormous batch sizes.

**Prefer something else when** sample efficiency is paramount and
you can tolerate complexity — off-policy actor-critics like
**SAC** (later in this section) reuse a full replay buffer and
squeeze more from each step. PPO trades some of that efficiency
for stability and simplicity, and usually it's the right trade.

---

## What comes next

Part 2 of this Advanced track goes back to the method PPO
simplifies: **Trust Region Policy Optimisation (TRPO)**. PPO
approximates a trust region with a clip; TRPO enforces one
*exactly*, with a KL-divergence constraint and the second-order
machinery to solve it. Seeing the rigorous version makes clear
just how much PPO gives up — and how little it loses — by
replacing that constraint with one `min` and a clip.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**ppo.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/09-advanced-reinforcement-learning/01-proximal-policy-optimisation/ppo.py)

Run it with:

```bash
pip install numpy
python ppo.py
```

It needs only `numpy` and runs in well under a minute. Every
piece is from scratch: the CartPole dynamics, the actor and
critic MLPs with hand-written backprop and Adam, GAE for the
advantages, and the clipped surrogate objective with its
multi-epoch minibatch updates. The headline insight worth
pinning to the wall: **PPO controls the probability ratio
`r = π_new/π_old` and clips the objective so there's no incentive
to push `r` outside `[1−ε, 1+ε]`; keeping each update proximal to
the old policy caps the step size *and* makes it safe to reuse
each rollout for many epochs — which on CartPole is the
difference between a stable climb to 483 and a collapse from 407
to 258**.

---

*This is Part 1 of the Advanced Reinforcement Learning track in the Algorithms in Python series, and it opens where the foundational track left off. The companion script `ppo.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [The previous article](https://medium.com/p/745178b96a83) covered Advantage Actor-Critic, the foundation PPO is built on. Part 2 will look at TRPO, the trust-region method PPO approximates.*
