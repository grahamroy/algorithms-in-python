# Twin Delayed DDPG — Fixing the Value That Lied

### *Algorithms in Python --- Advanced Reinforcement Learning, Part 4*

---

The DDPG article ended on a confession. DDPG works — it swung a
pendulum upright with a continuous torque DQN could never
produce — but it is *brittle*, and the culprit is a specific,
diagnosable flaw: **its critic overestimates**. The `Q(s, a)` it
learns drifts systematically higher than the returns the policy
actually achieves, and because the actor is trained to *maximise*
that critic, it seeks out and exploits exactly those inflated
values. The policy chases a number that isn't real.

**Twin Delayed DDPG (TD3)** (Fujimoto et al., 2018) is DDPG with
three targeted fixes for this, and it has become the reliable
default that plain DDPG never was. The three fixes:

1. **Twin critics + clipped double-Q** — keep *two* critics and
   build the TD target from the **smaller** of their estimates.
   Being pessimistic about your own value kills the
   overestimation. *(The headline.)*
2. **Delayed policy updates** — update the actor less often than
   the critics, so it chases a value that has had time to settle.
3. **Target policy smoothing** — add small noise to the target
   action, so the critic can't overfit to a sharp, fake peak.

This article builds TD3 from scratch in NumPy — six networks now,
the twin-critic targets, the delayed updates — on the same
Pendulum as DDPG, and shows the overestimation directly:
DDPG's critic predicting a value it is *provably impossible* to
achieve, and TD3's twin-min keeping it honest.

---

## The overestimation problem

Where does the inflation come from? It's a chain of three things.

**Function approximation is noisy.** A learned `Q(s, a)` is never
exact; it's high in some places, low in others, by accident of
the fit.

**Maximisation turns noise into bias.** DDPG's actor is trained
to output the action that *maximises* Q. Maximising over a noisy
function preferentially selects the spots where the noise happens
to be *positive* — so the value of the chosen action is
systematically **over**estimated. (This is the same upward bias
that `max` gives Q-learning, which Double DQN fixed for the
discrete case.)

**Bootstrapping compounds it.** The overestimated value becomes
the TD target for the next update, which inflates the next
estimate, and so on. The error doesn't average out; it
accumulates.

The result: `Q` climbs above the true return, the actor exploits
the phantom, and on hard tasks the whole thing destabilises. TD3
attacks all three links.

---

## Fix 1: twin critics and the clipped double-Q target

The core fix. TD3 keeps **two** critics, `Q1` and `Q2`, trained
on the same data but from different random initialisations. When
it builds the TD target, it uses the **minimum** of the two
target critics:

```
y  =  r  +  γ · min( Q1'(s', ã'),  Q2'(s', ã') )
```

The intuition is simple and powerful. An action that one critic
overestimates is unlikely to be overestimated *by the same
amount* by the other. Taking the `min` systematically picks the
more conservative estimate, cancelling the upward bias. You trade
a little downward pessimism — which is harmless, even helpful —
for eliminating the dangerous overestimation. This is **clipped
double-Q learning**, the continuous-control descendant of Double
DQN.

The actor is trained to maximise just one of them (`Q1`); it only
needs a single value to ascend.

---

## Fix 2: delayed policy updates

The actor and the critic are locked in a feedback loop: the actor
chases the critic, the critic bootstraps off the actor. If they
update in lockstep, errors ping-pong between them. TD3 breaks the
symmetry by updating the **actor (and all target networks) less
often** than the critics — here, once every **two** critic
updates. The critics get to refine their estimate on fixed
targets before the actor moves, so the actor chases a value that
has settled rather than one still thrashing. Slower, steadier,
more stable.

---

## Fix 3: target policy smoothing

A deterministic actor can exploit any sharp, narrow spike in the
critic — a spot where `Q` is erroneously high for one exact
action. TD3 blunts those spikes by adding a small amount of
**clipped noise** to the *target* action when computing the TD
target:

```
ã'  =  clip( μ'(s') + clip(noise, −c, c),  action range )
```

This forces the critic to produce similar values for similar
actions — a smoothness regulariser on the value function. The
policy can no longer chase a one-pixel-wide peak, because the
target has been deliberately blurred around it.

---

## The algorithm

```
TD3(episodes, γ, policy_delay d):
    init actor μ, twin critics Q1, Q2, and all their targets
    for each step:
        act with μ(s) + exploration noise;  store (s,a,r,s') in replay
        sample a minibatch:
            ã' = μ'(s') + clipped noise            # target smoothing
            y  = r + γ · min(Q1'(s',ã'), Q2'(s',ã'))  # clipped double-Q
            update BOTH critics toward y
            every d steps:                          # delayed
                actor: ascend Q1(s, μ(s))  via dQ/da
                soft-update all target networks
    return μ, Q1
```

Six networks in play — the actor, two critics, and their three
targets. The critic loop is DQN's, doubled and made pessimistic;
the actor loop is DDPG's deterministic policy gradient, run at
half the rate.

---

## A worked example: the overestimation, made visible

The companion script trains TD3 on Pendulum. First, it works:

```
DEMO 1 --- TD3 solves Pendulum swing-up
  Actor: 3->64->64->1   Twin critics: 4->64->64->1 each
  Episodes: 60  gamma=0.99  tau=0.01  policy_delay=2  twin critics + target smoothing

  Mean episode return by 12-episode block (0 is best, random ~ -1300):
    episodes   1- 12 :  -1331.9  #
    episodes  13- 24 :   -912.4  ############
    episodes  25- 36 :   -142.2  ###############################
    episodes  37- 48 :   -164.0  ##############################
    episodes  49- 60 :   -161.4  ##############################

  Final mean return (last 20 episodes): -169.9
```

Same swing-up as DDPG, same ballpark result. The interesting part
is *the value function underneath*.

### The headline: a value that can't be true

Pendulum's reward is `−(angle² + …)` — it is **always ≤ 0**. So
the true value of *any* state, the discounted sum of future
rewards, must also be `≤ 0`. That gives us a free lie detector.
The script measures each critic's predicted `Q(s, μ(s))` against
the return the policy *actually* earns from those states:

```
DEMO 2 --- The twin-critic min fixes DDPG's overestimation
  Single critic (DDPG-style): predicted Q =    +5.0   actual return =    -4.6
     -> Q is POSITIVE -- impossible when every reward is <= 0. Overestimates by 9.6.

  Twin critics + min (TD3)  : predicted Q =    -8.4   actual return =    -4.9
     -> stays negative, close to the truth (gap -3.5).
```

The single-critic value (DDPG's approach — the *only* difference
in this comparison) predicts `Q = +5.0`. That is not a small
error; it is a **positive** value on a task where every reward is
negative — a return the policy could not achieve if it played
perfectly. The critic believes in something impossible, and the
actor is optimising toward it.

TD3's twin-min predicts `Q = −8.4` — negative, plausible, close
to the true `−4.9` (erring slightly *pessimistic*, which is
exactly the safe direction). One change — `min` of two critics
instead of one — turns an impossible value into an honest one.

On easy Pendulum both still solve the task, but that inflated
value is precisely what spirals out of control on harder problems
and makes DDPG diverge. TD3 removes it at the source.

### A robust controller

With an honest value underneath, the greedy policy generalises
cleanly to starts it never saw:

```
DEMO 3 --- The trained TD3 policy, from unseen starts
    start seed 11:   return   -236.6
    start seed 22:   return   -124.0
    start seed 33:   return     -2.4
    start seed 44:   return   -233.2
    start seed 55:   return   -115.1

  Mean over 5 unseen starts: -142
```

Reliable swing-ups (seed 33's −2.4 is nearly perfect), built on a
value function that no longer lies to it.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

TD3 costs a little more than DDPG — a second critic and a `min`,
with the actor updated at half the rate.

**Per step**: two critic updates instead of one, `O(B · W)`, plus
— every `d` steps — one actor update and the soft-update
averages. The delayed schedule means the actor pass runs *less*
often than DDPG's, partly offsetting the second critic.

**Memory**: `O(W + N · d)` — six networks (still a constant
factor) plus the replay buffer. The second critic doubles the
critic weights; the buffer, as always, dominates.

**The trade it makes**: strictly more compute per step than DDPG,
in exchange for a value estimate that doesn't inflate and a policy
that doesn't chase phantoms. On any task where DDPG's
overestimation bites — which is most of them — that is a trade
worth making, and it is why TD3, not DDPG, is the practical
default for deterministic continuous control.

---

## TD3 in the family

TD3 is best understood as **DDPG, debugged**. It changes none of
DDPG's structure — off-policy, deterministic, actor-critic,
replay, target networks — and adds only the three fixes above.
Every one of them is a form of *humility about your own
estimates*: take the smaller value, move the policy slower, don't
trust a sharp peak.

Against its neighbours:

- **vs DDPG** — same algorithm, far more stable. There is little
  reason to run plain DDPG once you have TD3.
- **vs SAC** (next) — TD3 keeps the deterministic policy and adds
  pessimism; **SAC** makes the policy *stochastic* again and adds
  an **entropy bonus** that rewards keeping options open. SAC
  borrows TD3's twin-critic min, so it's TD3's ideas plus
  principled exploration. On many benchmarks SAC edges ahead, but
  the two are close, and both descend from the DDPG you built in
  Part 3.

---

## What comes next

Part 5 — the final algorithm in this track — is **Soft
Actor-Critic (SAC)**. It rethinks exploration from the ground up:
instead of bolting noise onto a deterministic policy, it makes
the policy stochastic and adds the policy's **entropy** directly
to the objective, so the agent is rewarded for staying
appropriately uncertain. It keeps TD3's twin-critic pessimism and
adds maximum-entropy RL on top — often the strongest of the
continuous-control methods, and a fitting close to the series.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**td3.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/09-advanced-reinforcement-learning/04-twin-delayed-ddpg/td3.py)

Run it with:

```bash
pip install numpy
python td3.py
```

It needs only `numpy` and runs in well under a minute. Every
piece is from scratch: the Pendulum physics, the actor and twin
critics with hand-written backprop and Adam, the replay buffer,
the delayed soft-updated targets, the target-smoothing noise, and
the clipped double-Q target. The headline insight worth pinning to
the wall: **TD3 is DDPG made honest — it keeps two critics and
builds its TD target from the *minimum* of them (clipped
double-Q), so the value can't inflate the way DDPG's does; add
delayed actor updates and target-action smoothing and the whole
method stops chasing phantom values. On Pendulum, DDPG's single
critic predicts an impossible positive value (+5) on a task where
every reward is negative; TD3's twin-min stays negative and close
to the truth**.

---

*This is Part 4 of the Advanced Reinforcement Learning track in the Algorithms in Python series. The companion script `td3.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It is [DDPG](https://medium.com/p/88eecb39f5d9) with three fixes for the value overestimation that made DDPG brittle. Part 5 will look at Soft Actor-Critic, which adds maximum-entropy exploration to close the track.*
