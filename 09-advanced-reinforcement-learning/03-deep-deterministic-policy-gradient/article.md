# Deep Deterministic Policy Gradient — DQN for Continuous Control

### *Algorithms in Python --- Advanced Reinforcement Learning, Part 3*

---

Every algorithm in the policy-gradient line so far — REINFORCE,
A2C, PPO, TRPO — has shared two traits: it is **on-policy**
(each batch of experience is used once and discarded) and
**stochastic** (the policy is a probability distribution you
sample from). And the one off-policy method we built, **DQN**,
had a hard limit: it picks actions by `argmax_a Q(s, a)`, which
requires *enumerating* the actions. That works for a joystick
with a few buttons. It is impossible for a steering angle, a
joint torque, a throttle — a **continuous** action, of which
there are infinitely many.

**Deep Deterministic Policy Gradient (DDPG)** (Lillicrap et al.,
2016) fills exactly this gap. It is, in one line, **DQN for
continuous actions**. It keeps DQN's off-policy machinery — the
replay buffer and target networks — but replaces the impossible
`argmax` with an **actor network** that *outputs* the best
action directly, and a **critic** that evaluates it. The actor
is trained by the **deterministic policy gradient**: nudge its
output in the direction that increases the critic's Q. It is
off-policy, deterministic, and continuous — the combination none
of the previous methods offered.

This article builds DDPG from scratch in NumPy — actor, critic,
their two target networks, the replay buffer, and the `dQ/da`
gradient chain that couples the two — and trains it on
**Pendulum**, a classic continuous-control swing-up task that
DQN's discrete `argmax` fundamentally cannot solve.

---

## The problem: argmax dies on a continuum

DQN's action rule is `a = argmax_a Q(s, a)`. To evaluate it you
compute `Q(s, a)` for *every* action and take the best. With a
continuous action there is no "every" — the torque `a ∈ [-2, 2]`
takes infinitely many values, and you cannot maximise over them
by enumeration.

DDPG's move is to *learn the argmax*. Instead of searching for
the best action at every step, it trains a network to **produce**
it:

```
actor   μ(s; θ)   →  the action to take (a continuous torque)
critic  Q(s, a; w) →  how good that state-action pair is
```

The critic is trained exactly as in DQN — regression toward a TD
target. The actor is trained to output actions the critic
scores highly. Between them they replace the `argmax` with a
gradient.

---

## The deterministic policy gradient

How do you train an actor when there's no "correct action" label
— only a critic's opinion? You use the critic as the loss. The
actor's objective is simply to maximise the Q-value of the
actions it chooses:

```
J(θ)  =  E_s[ Q(s, μ(s; θ)) ]
```

Its gradient — the **deterministic policy gradient** (Silver et
al., 2014) — follows from the chain rule:

```
∇θ J  =  E_s[  ∇a Q(s, a)|_{a=μ(s)}  ·  ∇θ μ(s; θ)  ]
```

Read it left to right: `∇a Q` says *which way to move the action*
to increase Q; `∇θ μ` says *how to change the actor's weights to
move the action that way*. Chain them and you get how to change
`θ` to make the actor output higher-Q actions. In code it is one
backward pass of `Q` **into its action input**, then continued
through the actor — the companion script does exactly this,
computing `dQ/da` from the critic and feeding it as the gradient
signal to the actor. It is gradient *ascent* on the critic,
routed through the policy.

Unlike the stochastic policy gradient (which weights `∇ log π` by
a return), there is no sampling and no log-probability here — the
policy is a deterministic function, and the gradient flows
straight through the critic. That directness is what makes DDPG
efficient on continuous actions, and also what makes it delicate,
as we'll see.

---

## What it inherits from DQN

A naive actor-critic like this diverges, for the same reasons a
naive DQN does. DDPG borrows DQN's two fixes wholesale — now
applied to **both** networks:

- **A replay buffer.** Every transition `(s, a, r, s')` is stored
  and sampled in random minibatches. This makes DDPG **off-policy**
  — it learns from old data collected by older policies, reusing
  each transition many times. PPO and TRPO cannot do this; it is
  DDPG's big sample-efficiency advantage.
- **Target networks.** Slowly-tracking copies of *both* the actor
  and the critic, `μ'` and `Q'`, provide the stable TD target
  `y = r + γ Q'(s', μ'(s'))`. DDPG uses a **soft (Polyak) update**
  — `θ' ← τθ + (1−τ)θ'` every step with a small `τ` — rather than
  DQN's periodic hard copy.

There's one new ingredient. A deterministic policy has no
randomness of its own, so it cannot explore. DDPG adds
**exploration noise** to the actor's action during training
(here, Gaussian noise; the original paper used correlated
Ornstein-Uhlenbeck noise). At evaluation the noise is switched
off and the policy is purely deterministic.

---

## The algorithm

```
DDPG(episodes, γ, τ):
    init actor μ, critic Q, and targets μ' ← μ, Q' ← Q
    init empty replay buffer
    for each step:
        a = μ(s) + exploration noise;  clip to the action range
        take a, observe r, s';  store (s, a, r, s') in the buffer
        sample a random minibatch:
            y = r + γ Q'(s', μ'(s'))                 # TD target
            critic: minimise ( Q(s, a) − y )²
            actor : ascend Q(s, μ(s))  via  dQ/da → dμ/dθ
            soft-update:  μ' ← τμ + (1−τ)μ' ;  Q' ← τQ + (1−τ)Q'
```

Four networks in flight: the actor and critic being trained, and
their two slow-moving targets. The critic loop *is* DQN; the
actor loop is the deterministic policy gradient; the replay and
targets are the glue.

---

## A worked example: Pendulum swing-up

The companion script trains DDPG on Pendulum — swing an
under-powered pendulum from hanging to upright and hold it. The
reward is `−(angle² + 0.1·angular_velocity² + 0.001·torque²)`, so
0 is a perfectly balanced pole and a flailing random policy
scores about −1300.

```
DEMO 1 --- DDPG solves Pendulum swing-up (continuous torque)
  Actor : 3 -> 64 -> 64 -> 1  (torque in [-2, 2] via tanh)
  Critic: 4 -> 64 -> 64 -> 1  (Q of state AND action)
  Episodes: 80  gamma=0.99  tau=0.01  replay + target nets

  Mean episode return by 12-episode block (0 is best, random ~ -1300):
    episodes   1- 12 :  -1300.1  ##
    episodes  13- 24 :   -509.9  ######################
    episodes  25- 36 :   -145.4  ###############################
    episodes  37- 48 :   -183.5  ##############################
    episodes  49- 60 :   -167.3  ##############################
    episodes  61- 72 :   -187.0  ##############################
    episodes  73- 80 :   -207.4  #############################

  Final mean return (last 20 episodes): -195.2
```

Within about 25 episodes DDPG goes from hopeless (−1300) to a
solid swing-up (~−170), learning off-policy from its replay
buffer the whole time.

### The headline: continuous control DQN can't do

The point of DDPG is the *kind* of action it produces. Run the
trained policy greedily from hanging straight down and watch it
work:

```
DEMO 2 --- The swing-up in action (continuous torque control)
    step    angle from upright    torque applied
       0            180 deg            +2.00
      20            159 deg            -2.00
      40            162 deg            +1.51
      60              3 deg            -0.99
      80             10 deg            -2.00
     100             10 deg            -1.48
     120             10 deg            -0.95
     140             10 deg            -0.95
     180             10 deg            -1.92
```

The pendulum starts hanging at 180°. The actor can't just shove
it to the top — it's under-powered — so it **pumps**: full torque
one way, full torque back, building energy (the angle swings
around 160° before breaking free). By step 60 it has caught the
pendulum near upright (3°), and from there it *balances*, applying
a **continuously varying** torque — `−0.99`, `−1.48`, `−0.95`,
`−1.92` — to hold it. Those fractional, smoothly-changing torques
are exactly what a discrete-action agent cannot output. This is
the capability DDPG unlocks.

### Off-policy and reliable

Because it's off-policy, the learned policy generalises to starts
it never trained on:

```
DEMO 3 --- Off-policy and reliable: greedy runs from unseen starts
    start seed 11:   return   -231.5
    start seed 22:   return   -130.7
    start seed 33:   return     -7.7
    start seed 44:   return   -228.1
    start seed 55:   return   -120.1

  Mean over 5 unseen starts: -144
```

From five random starts it never saw, the greedy policy reliably
swings up and balances (mean −144; seed 33's −7.7 is a nearly
perfect hold). It learned all of this from a replay buffer of
recycled transitions — the off-policy efficiency PPO and TRPO
give up.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

DDPG runs four networks but only ever trains two; the targets are
updated by cheap averaging.

**Per step**: `O(W)` to act (one actor pass), then a gradient
update on a minibatch of `B` — one critic backward, one actor
backward (which includes a `dQ/da` pass through the critic), and
two soft-update averages: `O(B · W)`.

**Memory**: `O(W + N · d)` — four networks (a small constant
factor) plus a replay buffer of `N` transitions. The buffer is
the memory cost DQN also pays, and the price of being off-policy.

**Sample efficiency vs on-policy**: this is DDPG's edge. Every
transition is replayed in many minibatches, so DDPG solves
Pendulum in a few thousand environment steps — far fewer than an
on-policy method needs, because those throw each rollout away.

**The catch**: DDPG is famously *brittle*. The deterministic
policy gradient flows through a critic that is only accurate near
the actions the actor takes; away from them, `Q` can be wildly
**overestimated**, and the actor happily exploits those phantom
values. Combined with its sensitivity to hyperparameters, this
makes DDPG powerful but temperamental — which is precisely what
the next two algorithms set out to fix.

---

## When to use DDPG — and its successors

**Use DDPG when** actions are continuous and sample efficiency
matters — robotics, control, anything where each environment step
is expensive and you want to squeeze reuse out of a replay
buffer. It brought DQN-style off-policy learning to continuous
control, and every modern continuous-control algorithm descends
from it.

**But reach for its successors in practice**, because DDPG's
brittleness was worth fixing:

- **TD3** (Twin Delayed DDPG, Part 4) attacks the Q-overestimation
  head-on — two critics and a `min` to stay pessimistic, plus
  delayed and smoothed updates. It is DDPG made robust.
- **SAC** (Soft Actor-Critic, Part 5) makes the policy stochastic
  again and adds an entropy bonus for principled exploration,
  usually the strongest of the three.

DDPG is the foundation both build on. Understanding its
deterministic policy gradient — and its overestimation problem —
is understanding why TD3 and SAC look the way they do.

---

## What comes next

Part 4 is **TD3 (Twin Delayed DDPG)** — the same actor-critic
skeleton you just built, with three targeted fixes for DDPG's
instability. The headline one: keep **two** critics and always
use the *smaller* of their Q-estimates for the target, so the
actor can no longer chase an overestimated value. It is the most
direct illustration in RL of a simple idea — be pessimistic about
your own estimates — turning a temperamental algorithm into a
dependable one.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**ddpg.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/09-advanced-reinforcement-learning/03-deep-deterministic-policy-gradient/ddpg.py)

Run it with:

```bash
pip install numpy
python ddpg.py
```

It needs only `numpy` and runs in well under a minute. Every
piece is from scratch: the Pendulum physics, the actor and critic
MLPs with hand-written backprop and Adam, the replay buffer, the
soft-updated target networks, and the `dQ/da` chain that carries
the deterministic policy gradient from the critic into the actor.
The headline insight worth pinning to the wall: **DDPG is DQN for
continuous actions — it replaces the impossible `argmax` with an
actor `μ(s)` that outputs the action and a critic `Q(s,a)` that
scores it, and trains the actor by pushing its output up the
critic's Q-gradient (`dQ/da → dμ/dθ`); keeping DQN's replay
buffer and target networks makes it off-policy and
sample-efficient, and it swings up a pendulum with a continuous,
smoothly-varying torque no discrete-action agent could produce**.

---

*This is Part 3 of the Advanced Reinforcement Learning track in the Algorithms in Python series. The companion script `ddpg.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It fuses [DQN](https://medium.com/p/0102b41f786d)'s off-policy replay and target networks with the actor-critic of the policy-gradient line. Part 4 will look at TD3, which fixes DDPG's Q-value overestimation to make it robust.*
