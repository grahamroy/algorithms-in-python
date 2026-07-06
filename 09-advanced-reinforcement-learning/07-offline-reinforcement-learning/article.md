# Offline Reinforcement Learning — Learning When You Can't Try Things

### *Algorithms in Python --- Advanced Reinforcement Learning, Part 7*

---

Every off-policy algorithm in this series — DQN, DDPG, TD3, SAC —
carried a quiet assumption that we never examined, because it
never failed. If the value function went wrong somewhere, the
agent would soon *act* on the mistake, observe the real outcome,
and be corrected. Exploration wasn't just how the agent found
good actions; it was how its beliefs stayed tethered to reality.

**Offline reinforcement learning** removes that tether. You are
handed a *fixed dataset* of transitions, collected earlier by
some other policy — and that is all. No environment. No new
experience. No trying things. Learn the best policy you can from
what's already on disk.

The setting matters because it is where RL meets most of the
real world. Hospitals have years of treatment logs but you
cannot experiment on patients. Fleets have millions of miles of
driving data but you cannot crash cars to learn. Recommenders,
industrial controllers, dialogue systems — everywhere there is
abundant *logged* experience and an excellent reason not to
explore. If RL is to work there, it must work offline.

The surprise — and the subject of this final article — is that
our best off-policy algorithms don't just underperform in this
setting. Fed a perfectly good dataset, they **collapse below a
random policy**, while their value estimates drift into numbers
that are provably impossible. One small change fixes it. Both
halves are demonstrated, from scratch, in the companion script.

---

## Why naive off-policy learning fails offline

On paper, nothing stops you from running TD3 on a fixed buffer —
replay-based methods never insisted the data be fresh. The
failure is subtler, and it's called **extrapolation error**. It
is a four-step doom loop:

1. **The critic gets asked about actions the data doesn't
   contain.** The TD target evaluates `Q(s', π(s'))` — and `π`
   soon proposes actions the behaviour policy never took.
2. **Function approximation guesses.** A network queried outside
   its training distribution returns *something* — sometimes far
   too high.
3. **Nothing corrects the guess.** Online, the agent would try
   the action and be disappointed by reality. Offline, that
   feedback never arrives. The error just sits there.
4. **Policy improvement hunts the phantoms.** The actor is
   literally optimised to find the actions with the highest Q —
   which, increasingly, are the fictional ones.

Each step feeds the next: the actor drifts further off-data, the
critic's guesses get wilder, the bootstrap propagates them
everywhere. This is TD3's overestimation problem (Part 4) with
the safety net removed — online, acting bounds the error;
offline, it compounds without limit.

The root cause has a name: **distribution shift**. The policy
being trained is not the policy that made the data, and every
gradient step widens the gap between the questions we ask the
critic and the evidence it was fitted on.

---

## The fix: stay where the data can vouch for you

Every successful offline RL method is, at heart, a form of
**conservatism** — keep the learned policy inside the *support*
of the dataset, where the critic's estimates are backed by
evidence. The family portrait:

- **BCQ** (Fujimoto et al., 2019) — only consider actions a
  generative model of the dataset says are plausible.
- **CQL** (Kumar et al., 2020) — penalise the critic directly for
  assigning high values to out-of-distribution actions.
- **TD3+BC** (Fujimoto & Gu, 2021) — the minimalist: keep TD3
  *exactly as it is* and add one behavioural-cloning term to the
  actor loss.

This article implements TD3+BC, because it makes the point with
one line of mathematics:

```
actor loss  =  − λ · Q(s, π(s))  +  ( π(s) − a_data )²
```

The second term pulls the policy toward the actions actually in
the dataset — an anchor to the data's support. The first term
still *improves* the policy within that support, using the
critic exactly as TD3 always did. The scale factor
`λ = α / mean|Q|` keeps the two terms comparable as Q grows
(α = 1 here). Set the BC term's weight to zero and you recover
naive TD3; delete the Q term and you have pure behavioural
cloning (supervised imitation). The interesting behaviour — and
offline RL's whole promise — lives between those poles.

---

## A worked example: the environment gets switched off

The companion script stages the full story on Pendulum. A
medium-quality TD3 policy (trained briefly online) collects
8,000 transitions *with its exploration noise* — and then the
environment is switched off. Every learner below sees only the
fixed arrays; the environment is used solely to grade their
final policies.

```
DEMO 1 --- The setting: a fixed dataset, no environment access
  Dataset size                      : 8000 transitions (40 episodes)
  Mean episode return IN the data   :   -643.0   (what the dataset looks like)
  The collector, without its noise  :   -259.7   (for reference)
  (0 is a perfectly balanced pendulum; random is about -1300)
```

### The headline: collapse, and the one-line fix

Two learners, identical in every respect except the actor loss,
each given 15,000 gradient steps on the same fixed data:

```
DEMO 2 --- Naive off-policy learning fails offline; TD3+BC fixes it
  The data it learned from   : return   -643.0
  Naive offline TD3          : return  -1473.0   (worse than random!)
  TD3+BC                     : return   -358.2   (better than its data)

  The critic's story (every true value on Pendulum is <= 0):
    naive : predicted Q =     +31.5   actual return =   -470.3
    TD3+BC: predicted Q =    -164.3   actual return =   -169.2

  Naive Q over training (sampled every 1,500 steps) -- watch it grow
  past zero into impossible territory, with nothing to correct it:
    +0  -22  -34  -36  -22  +1  +20  +24  +21  +16
```

Read the collapse carefully, because each number tells part of
the story. Naive TD3 — the same algorithm that *solved* this
task in Part 4 — lands at **−1473, worse than acting randomly**,
despite learning from data averaging −643. And the Q trace shows
exactly how: for the first few thousand steps the critic is
sane (−22, −34 — fitting the data it has), and then, as the
actor drifts onto actions the data can't vouch for, the estimate
**crosses zero into provably impossible territory** and stays
there. On a task where every reward is negative, the critic ends
up promising +31.5 for a policy that actually earns −470. No
fresh experience ever arrives to call the bluff.

TD3+BC — one extra term — lands at **−358, better than the data
that taught it**, and its critic stays honest: predicted −164.3
against an actual −169.2. Anchored to the data's support, the
critic's promises are backed by evidence, so improving against
it actually works.

### The dial from imitation to reinforcement learning

Offline RL's promise over plain imitation is *improvement*: the
dataset contains noisy, mediocre behaviour, and stitching
together its best moments should beat copying its average. The
three-way comparison makes the trade explicit:

```
DEMO 3 --- The dial from imitation to reinforcement learning
    BC only  (copy the data)          : return   -573.5
    TD3+BC   (improve within the data): return   -358.2
    naive    (trust Q everywhere)     : return  -1473.0

    the dataset itself averaged       : return   -643.0
    the collector, noise-free         : return   -259.7
```

Pure behavioural cloning faithfully reproduces the dataset,
noise and all (−573 against the data's −643). Trusting the
critic *everywhere* is catastrophic. Between them, TD3+BC uses
the critic only where the data can support it and beats both —
recovering most of the gap to the noise-free collector from
noisy logs alone. That middle setting *is* offline RL.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

Offline RL's cost profile has one defining feature: the
environment column is **zero**.

**Per gradient step**: identical to TD3 — two critic updates and
a (delayed) actor update on a minibatch, `O(B · W)`, plus the
`O(B)` BC term, which is negligible.

**Total training**: `O(G · B · W)` for `G` gradient steps —
pure computation against fixed arrays. No rollouts, no waiting
on a simulator, no wear on a robot. Training is as parallel,
repeatable, and safe as supervised learning.

**Memory**: `O(W + N · d)` — the networks plus the dataset
itself, which plays the role the replay buffer always did,
except it is never written to again.

**The trade**: what offline RL saves in interaction it pays in
*evaluation*. Grading a policy honestly requires the environment
(or careful off-policy evaluation) — the one place this
article's script still touches Pendulum. In real deployments,
deciding whether an offline-trained policy is safe to run is
often the hardest part of the whole exercise.

---

## Where this matters — and where the field went

The conservatism principle scaled far beyond Pendulum. The
D4RL benchmarks (Fu et al., 2020) standardised the setting;
CQL, IQL (Kostrikov et al., 2021), and TD3+BC became the
reference baselines; and offline pre-training followed by
cautious online fine-tuning is now a standard recipe in robotics
— learn everything you can from logs first, spend real-world
trials last. The same instinct — *don't let the policy stray
into actions your data can't justify* — echoes through modern
fine-tuning of language models, where staying close to a
reference policy guards against reward hacking. Wherever a
learned objective meets fixed evidence, the lesson of this
article applies: **only trust the optimiser where the data can
vouch for it.**

---

## Closing the series

This article completes the reinforcement learning journey — and
the arc is worth seeing whole. A Q-table on a 48-square grid
(Q-Learning, SARSA). A network replacing the table (DQN). The
policy learned directly (REINFORCE), stabilised by a critic
(A2C), made reliable by trust regions and clips (TRPO, PPO).
Value and policy fused for continuous control (DDPG), made
honest (TD3), made principled about exploration (SAC). Planning
without learning at all (MCTS). And finally, learning without
acting at all — offline RL, where the ideas met their sternest
test: no second chances, no fresh data, just fixed evidence and
the discipline not to wander beyond it.

Twelve algorithms, every one built from scratch in NumPy, every
claim run and printed. Nothing in this track required more than
`pip install numpy` — and none of the ideas were ever as
frightening as their reputations.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**offline_rl.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/09-advanced-reinforcement-learning/07-offline-reinforcement-learning/offline_rl.py)

Run it with:

```bash
pip install numpy
python offline_rl.py
```

It needs only `numpy` and runs in about a minute. It trains the
behaviour policy, collects the fixed dataset, switches the
environment off, and trains all three offline learners — naive
TD3, TD3+BC, and pure behavioural cloning — printing the returns
and the critic diagnostics you saw above. The headline insight
worth pinning to the wall: **off-policy is not offline — without
fresh experience to correct it, a critic's out-of-distribution
guesses go unchallenged and the actor optimises into fiction
(here: worse than random, with a provably impossible positive
Q); one behavioural-cloning term anchoring the policy to the
data's support turns the same algorithm into one that beats the
dataset it learned from**.

---

*This is Part 7 of the Advanced Reinforcement Learning track in the Algorithms in Python series — the final article of the reinforcement learning journey. The companion script `offline_rl.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It builds directly on [TD3](https://medium.com/p/d3a3ccf0bf44) (Part 4); the track also covered [SAC](https://medium.com/p/81c414cc17e1) and [Monte Carlo Tree Search](https://medium.com/p/1a37862620b5) on the way here.*
