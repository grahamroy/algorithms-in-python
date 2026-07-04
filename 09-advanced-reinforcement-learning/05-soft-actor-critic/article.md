# Soft Actor-Critic — Exploration Becomes Part of the Objective

### *Algorithms in Python --- Advanced Reinforcement Learning, Part 5*

---

Every continuous-control method so far has treated exploration
as an *afterthought*. DDPG and TD3 learn a **deterministic**
policy — one exact torque per state — and then, purely so that
training sees some variety, bolt Gaussian noise onto its output.
The noise scale is a hyperparameter you pick, it has nothing to
do with what the agent knows or doesn't, and at deployment it's
simply switched off. Exploration lives *outside* the thing being
optimised.

**Soft Actor-Critic (SAC)** (Haarnoja et al., 2018) moves it
*inside*. The policy is **stochastic** again — it outputs a
probability distribution over actions — and the objective is
changed to reward not just return but **entropy**, the policy's
own uncertainty:

```
J  =  E[ Σ_t  r_t  +  α · H( π(·|s_t) ) ]
```

The agent is literally *paid to keep its options open*, at a
rate set by the **temperature** `α`. Where the critics are
confident, the policy sharpens and collects reward; where they
aren't, staying spread out costs little and keeps exploring. And
in SAC's modern form, `α` is not a hyperparameter at all — it is
**auto-tuned** so the policy holds a target level of entropy.
The exploration schedule that DDPG made you hand-design writes
itself.

This article builds SAC from scratch in NumPy — the squashed
Gaussian policy, the reparameterisation trick with every
gradient derived by hand, TD3's twin critics, and the
temperature controller — and shows the dial turning on its own.

---

## Maximum-entropy RL: why pay for uncertainty?

Adding `α·H(π)` to the objective sounds like a hack. It is
actually a principled reframing — **maximum-entropy RL** — with
three practical payoffs:

- **Exploration that responds to knowledge.** Entropy is only
  surrendered when the critics offer reward in exchange. Early
  on, Q is flat and wrong everywhere, so the cheapest way to
  earn the entropy bonus is to stay broad. As Q sharpens, the
  trade flips and the policy commits — *where* it has learned,
  not on a clock.
- **Robustness.** A policy trained to succeed *while remaining
  stochastic* can't rely on knife-edge precision, so it degrades
  gracefully under perturbation.
- **No collapsed exploration.** Deterministic-policy methods can
  stop exploring prematurely; the entropy term makes premature
  certainty actively costly.

The temperature `α` prices the trade. `α → 0` recovers ordinary
RL; large `α` approaches uniform randomness. Everything
interesting happens in between — which is why tuning it
automatically matters so much.

---

## The stochastic actor, rebuilt for continuous actions

SAC's policy network outputs **two numbers per action
dimension**: a mean `μ(s)` and a standard deviation `σ(s)`. An
action is drawn by sampling Gaussian noise and squashing it into
the torque range:

```
a  =  bound · tanh( μ(s) + σ(s) · ε ),      ε ~ N(0, 1)
```

This is the **reparameterisation trick** (the same one that
powers variational autoencoders): the randomness enters through
an *external* `ε`, so the sampled action is a deterministic,
differentiable function of the network outputs. Gradients flow
*through the sample* — `dQ/da · da/dμ` — giving a far
lower-variance learning signal than REINFORCE-style score
functions, which can only correlate log-probabilities with
returns.

The `tanh` squashing has a price: it changes the distribution,
so the log-probability needs a change-of-variables correction:

```
log π(a|s)  =  log N(u; μ, σ)  −  log( bound · (1 − tanh²(u)) )
```

The companion script implements this — and derives every
gradient of it by hand, including how `log π` and the sampled
action each depend on `μ` and `log σ`. It is the most delicate
backprop in this series, and it fits in a dozen lines.

---

## What SAC keeps, and what it changes

From **TD3** it keeps the twin critics and the pessimistic
`min` — overestimation doesn't stop being a problem just because
the policy is stochastic. From **DDPG** it keeps the replay
buffer and soft target updates: SAC is **off-policy**, reusing
old experience freely.

The updates change in one consistent way: the entropy term
follows the value everywhere it goes.

**Critics** regress toward a *soft* target — the reward, plus
the discounted value of the next state *including its entropy
bonus*:

```
y  =  r  +  γ ( min(Q1', Q2')(s', a')  −  α · log π(a'|s') )
```

with `a'` freshly sampled from the current policy (no target
actor exists — the stochastic policy provides its own
smoothing, which also supersedes TD3's target-noise trick).

**The actor** maximises the same soft value at the current
state:

```
maximise   E[ min(Q1, Q2)(s, a_θ)  −  α · log π(a_θ|s) ]
```

**The temperature** runs a tiny controller of its own. Pick a
*target entropy* `H̄` (the standard default: `−dim(A)`, here
`−1`), and adjust `α` by gradient descent so the policy's actual
entropy tracks it: entropy above target → `α` falls; below →
`α` rises. One line of code, and the exploration dial turns
itself.

---

## A worked example: SAC on Pendulum

Same environment as DDPG and TD3, so the comparison is direct.

```
DEMO 1 --- SAC solves Pendulum swing-up (stochastic policy)
  Actor : 3 -> 64 -> 64 -> (mu, sigma)   a ~ 2*tanh(N(mu, sigma))
  Twin critics + min (from TD3)  |  entropy-regularised objective
  Episodes: 60  gamma=0.99  tau=0.01  auto-tuned alpha, target entropy -1.0

  Mean episode return by 12-episode block (0 is best, random ~ -1300):
    episodes   1- 12 :  -1220.7  ####
    episodes  13- 24 :   -133.2  ###############################
    episodes  25- 36 :   -146.3  ###############################
    episodes  37- 48 :   -150.8  ###############################
    episodes  49- 60 :   -143.0  ###############################

  Final mean return (last 20 episodes): -156.0
```

Look at the second block. SAC is already at **−133** in episodes
13–24 — on the same task, seed, and episode budget where DDPG's
second block was −510 and TD3's was −912. On this (small,
one-seed) benchmark, the entropy-driven explorer is the fastest
learner of the three, and it holds its solution steadily.

### The headline: the dial turns itself

No noise schedule was written anywhere in the code. Here is what
the temperature controller and the policy's spread did on their
own:

```
DEMO 2 --- The exploration dial turns itself
    episode    alpha     mean sigma    policy entropy
    (start)     0.200      ~1.0           ~0.0
       10       0.061       0.577         -0.67
       20       0.068       0.385         -0.89
       30       0.053       0.267         -0.85
       40       0.044       0.287         -0.74
       50       0.038       0.352         -0.72
       60       0.030       0.352         -0.70

  alpha turns itself down ~7x and sigma halves as the critics sharpen;
  the entropy is steered from ~0 into a steady band just above the
  target. Exploration is scheduled by the objective, not by hand.
```

Read the columns. The policy starts wide (`σ ≈ 1`, entropy near
zero nats) while the critics know nothing. As they sharpen, the
controller cuts `α` from 0.200 to 0.030 — certainty gets cheaper
— and `σ` anneals from ~1.0 to ~0.3: broad exploration early,
commitment late. The entropy settles into a steady band just
above the −1.0 target, with `α` still gently annealing to close
the gap. This is the behaviour DDPG's hand-tuned, fixed noise
scale approximates crudely — produced here by the objective
itself, adapting to what the critics actually know.

### Deploy the mean

At evaluation time you don't sample — you act on the
distribution's mean:

```
DEMO 3 --- Act on the mean: a strong deterministic controller
    start seed 11:   return   -230.8
    start seed 22:   return   -125.4
    start seed 33:   return     -2.4
    start seed 44:   return   -227.4
    start seed 55:   return   -115.0

  Mean over 5 unseen starts: -140
```

From five never-seen starts, the mean policy swings up and
balances every time (seed 33's −2.4 is a near-perfect hold) —
matching the TD3 controller of Part 4. The stochasticity was for
*learning*; the mean is what you ship.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

SAC's costs are TD3's, with the target actor swapped for a
sampling step and a one-parameter controller.

**Per step**: one reparameterised sample (an actor forward pass
plus `O(1)` noise), two critic updates `O(B · W)`, one actor
update `O(B · W)` (which backprops through *both* critics for
the per-sample `min`), and the scalar `α` update — `O(B)`.

**Memory**: `O(W + N · d)` — five networks (actor, two critics,
two target critics; **no target actor**) plus the replay buffer.

**The trade against TD3**: nearly identical compute, one extra
mechanism (the temperature controller), in exchange for
exploration that adapts itself and a policy that is robustly
stochastic during training. Against PPO: off-policy replay makes
SAC far more sample-efficient, at the cost of PPO's simplicity.
SAC and TD3 are the two default choices for continuous control;
SAC is usually the first thing to try when exploration is the
hard part.

---

## The maximum-entropy thread

Entropy bonuses aren't new — A3C and PPO implementations have
long added a small fixed entropy term to keep policies from
collapsing. SAC's contribution was to take that heuristic
seriously: derive the *soft* Bellman equations where entropy is
part of the value itself (building on soft Q-learning), train an
actor-critic on them off-policy, and then close the loop by
tuning the temperature against an entropy *constraint* rather
than leaving it as a magic number (Haarnoja et al., 2018 — the
original SAC, and the follow-up that added automatic temperature
tuning).

The result completed the continuous-control arc of this track:
**DDPG** brought DQN's machinery to continuous actions, **TD3**
made its value estimates honest, and **SAC** made its
exploration principled. Those three, plus PPO on the on-policy
side, are the workhorses of modern continuous control.

---

## What comes next

With Part 6 (**Monte Carlo Tree Search**) covering planning, one
article remains: **Offline Reinforcement Learning** — learning a
policy from a *fixed dataset* with no environment interaction at
all. Every off-policy method in this track quietly assumed it
could keep collecting data; Part 7 removes that assumption and
confronts what happens when the value function believes in
actions the data can't support. It closes the track.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**sac.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/09-advanced-reinforcement-learning/05-soft-actor-critic/sac.py)

Run it with:

```bash
pip install numpy
python sac.py
```

It needs only `numpy` and runs in well under a minute. Every
piece is from scratch: the Pendulum physics, the squashed
Gaussian actor with hand-derived reparameterisation gradients
(including the tanh change of variables in `log π`), twin
critics with the pessimistic `min`, the replay buffer, soft
target updates, and the one-line temperature controller. The
headline insight worth pinning to the wall: **SAC puts
exploration inside the objective — the policy is a tanh-squashed
Gaussian trained through its own samples to maximise reward plus
`α` times its entropy, and `α` is auto-tuned to hold a target
entropy; on Pendulum the dial visibly turns itself, `α` falling
7× and the policy's spread halving as the critics sharpen, with
no hand-written noise schedule anywhere — and it reaches the
swing-up faster than DDPG or TD3 did**.

---

*This is Part 5 of the Advanced Reinforcement Learning track in the Algorithms in Python series. The companion script `sac.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It completes the continuous-control arc begun with [DDPG](https://medium.com/p/88eecb39f5d9) and made robust by [TD3](https://medium.com/p/d3a3ccf0bf44). [Part 6](https://medium.com/p/1a37862620b5) covered Monte Carlo Tree Search; Part 7, Offline Reinforcement Learning, closes the track.*
