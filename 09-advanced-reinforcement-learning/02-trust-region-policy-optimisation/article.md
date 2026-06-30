# Trust Region Policy Optimisation — The Exact Method PPO Approximates

### *Algorithms in Python --- Advanced Reinforcement Learning, Part 2*

---

Part 1 of this Advanced track built **PPO**, and described its
clip as an *approximation* of a trust region. This article builds
the thing being approximated: **Trust Region Policy Optimisation
(TRPO)** (Schulman et al., 2015), the algorithm that made the
trust-region idea rigorous in the first place.

The motivation is the one we've returned to all through the
policy-gradient articles: **a policy update that's too big can
wreck the policy**, and because the next batch of data comes from
the damaged policy, it may never recover. PPO handles this with a
cheap clip. TRPO handles it *exactly* — it solves a constrained
optimisation problem at every step:

```
maximise   L(θ)        the surrogate (expected advantage)
subject to KL(π_old, π_θ)  ≤  δ      stay in the trust region
```

The constraint says: improve the policy as much as you can, *but
do not move more than δ away* (in average KL divergence) from the
policy that collected the data. Inside that region the
improvement estimate is trustworthy; the famous result behind
TRPO is that optimising the surrogate within it gives a
**monotonic improvement guarantee** — each update provably does
not make the true objective worse.

That rigour comes at a price: TRPO needs the **natural gradient**,
computed with **conjugate gradient** and **Fisher-vector
products**, plus a **line search**. This article builds every
piece from scratch in NumPy — no autograd — and shows the trust
region being enforced to the letter on CartPole.

---

## The surrogate and the trust region

TRPO maximises the same surrogate objective PPO does — the
importance-weighted advantage —

```
L(θ)  =  E[ ( π_θ(a|s) / π_old(a|s) ) · A ]
```

but instead of clipping it, it caps the *step* directly with a
hard KL constraint, `KL(π_old, π_θ) ≤ δ`. Why KL and not, say,
the Euclidean distance between parameter vectors? Because the
same change in parameters can mean a tiny or an enormous change
in the *policy*, depending on where you are. KL measures distance
in the space of **distributions** — exactly what we care about.
A trust region of fixed KL radius is a trust region of fixed
*behavioural* change.

The payoff is the monotonic-improvement theorem (building on
Kakade & Langford's 2002 conservative policy iteration): within a
KL trust region, the surrogate `L` is a reliable lower bound on
the true performance, so improving `L` improves the policy. TRPO
turns that theorem into an algorithm.

---

## The natural gradient

To maximise `L` subject to the KL constraint, TRPO uses the
**natural gradient** rather than the ordinary one. The ordinary
gradient `g = ∇L` points in the direction of steepest ascent *in
parameter space* — but parameter space is the wrong geometry. The
natural gradient corrects for it using the **Fisher information
matrix** `F`, the local curvature of the KL divergence:

```
natural gradient  =  F⁻¹ g
```

`F` re-weights the step so that "steepest" is measured in
*distribution* space (KL), the same space the trust region lives
in. This is the direction that gets the most policy improvement
per unit of KL — precisely what a trust-region method wants.

The obstacle: `F` is a `P × P` matrix for `P` parameters.
Forming and inverting it is hopeless for any real network. TRPO's
practical trick is to never build it.

---

## Solving it without ever forming F

Two classic numerical methods make it tractable.

**Conjugate gradient** solves the linear system `F x = g` for the
natural gradient `x = F⁻¹ g` *iteratively*. Its one requirement
is a way to compute the matrix-vector product `F v` for an
arbitrary vector `v` — it never needs `F` itself. A handful of
iterations (10 here) get a good solution.

**Fisher-vector products** supply that `F v`. For a softmax
policy, the Fisher in logit-space is `diag(p) − p pᵀ`, so

```
F v  =  Jᵀ ( diag(p) − p pᵀ ) J v
```

where `J` is the Jacobian of the network's logits with respect to
its parameters. The companion script computes `J v` — the
directional derivative of the logits — with a **finite
difference** (one extra forward pass at `θ + ε v`), applies the
analytic logit-space Fisher, then does one backward pass for the
`Jᵀ` — a Fisher-vector product with no autograd at all.

Once conjugate gradient returns the direction `x`, TRPO scales it
to the edge of the trust region. Under a quadratic approximation
of the KL, the step that hits `KL = δ` is

```
step  =  sqrt( 2δ / (xᵀ F x) ) · x
```

---

## The line search

That step size is exact only under the *approximation* that KL is
quadratic. It usually isn't, quite. So TRPO finishes with a
**backtracking line search**: try the full step, and if the
*actual* KL exceeds δ or the surrogate didn't really improve,
halve the step and try again. The first step that genuinely stays
in the region and improves the surrogate is accepted; if none
does, the update is rejected and the policy stays put. This is
what turns "approximately in the trust region" into "provably in
the trust region."

---

## The algorithm

```
TRPO(iterations, δ):
    for each iteration:
        run π_old for N steps; compute advantages A with GAE
        g = ∇θ L(θ)                       surrogate gradient
        x = ConjugateGradient(F·_, g)     natural gradient, F⁻¹g
        step = sqrt(2δ / (xᵀ F x)) · x    scale to the KL edge
        line search: largest α in {1, ½, ¼, …} with
            KL(π_old, π_{θ+α·step}) ≤ δ  and  L improved
        θ ← θ + α · step
        fit the critic to the returns (for GAE)
```

Every `F·_` inside conjugate gradient is one Fisher-vector
product. The critic, as in A2C and PPO, exists only to compute
advantages and is trained by ordinary regression.

---

## A worked example: TRPO on CartPole

The companion script trains an actor (`4 → 64 → 2`) and critic
(`4 → 64 → 1`) with full TRPO — conjugate gradient, Fisher-vector
products, and the line search — on CartPole.

```
DEMO 1 --- TRPO learns CartPole (exact trust-region steps)
  rollout=2048 steps  max_kl(delta)=0.01  cg_iters=10  gamma=0.99  lambda=0.95

  Moving-average return vs environment steps (max possible = 500):
       2048 steps :   35.2  ##
      14336 steps :  187.8  ###############
      20480 steps :  344.6  ############################
      26624 steps :  467.2  ######################################
      45056 steps :  498.2  #########################################
      63488 steps :  500.0  #########################################
      81920 steps :  500.0  #########################################
     100352 steps :  500.0  #########################################

  Final moving-average return: 500.0
```

TRPO climbs to a perfect, *stable* 500 and holds it — no collapse,
no thrashing. The monotonic-improvement guarantee shows up as a
visibly smooth curve: each trust-region step is conservative
enough that the policy essentially never goes backwards.

### The headline: the trust region is real

The defining property of TRPO is that the constraint is *actually
enforced*. The script records the KL divergence of every accepted
update:

```
DEMO 2 --- The trust region is enforced exactly
  The KL of each accepted step, over training:
    iters  1- 5:  0.0086  0.0082  0.0078  0.0071  0.0068
    iters  6-10:  0.0071  0.0074  0.0060  0.0057  0.0069
    iters 11-15:  0.0065  0.0039  0.0084  0.0073  0.0080
    ...
    iters 46-50:  0.0079  0.0052  0.0058  0.0075  0.0078

  Max KL over all updates: 0.0094  (delta = 0.0100)
  -> every step stayed inside the trust region.
```

Across all 50 updates, the KL never exceeds δ = 0.01 — the
maximum is 0.0094. Each step moves the policy as far as the trust
region allows and *not one step further*. This is the guarantee
PPO's clip only gestures at: PPO *discourages* large moves,
TRPO *forbids* them.

### The trust-region size is a dial

How big should δ be? It directly trades learning speed against
stability:

```
DEMO 3 --- The trust-region size delta is a speed/stability dial
  delta    env steps to solve     final return    max KL
    0.001   71680                    475        0.0009
    0.01    28672                    500        0.0092
    0.05    36864                    480        0.0497
    0.2     51200                    428        0.1836
```

A tiny δ (0.001) is safe but slow — it crawls to a solution in
72k steps. A large δ (0.2) takes big, risky steps and ends up
*worse* (428), because the quadratic approximation behind the
step breaks down when you move too far. δ = 0.01 is the sweet
spot: fastest to solve and a perfect final score. And notice the
max-KL column tracks δ exactly — the trust region is honoured at
whatever radius you set. (The trained policy balances the pole
for the full 500 steps on every greedy rollout.)

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

TRPO's per-update cost is dominated by the conjugate-gradient
solve, which is what buys the exact trust region.

**Per iteration**: `O(N · W)` to collect a rollout of `N` steps,
then the natural-gradient solve: conjugate gradient runs
`cg_iters` Fisher-vector products, each one a forward and a
backward pass over the batch — `O(cg_iters · N · W)`. The line
search adds a few more forward passes.

**Memory**: `O(W + N · d)` — the networks plus one rollout. Like
PPO, **no replay buffer**. Conjugate gradient needs only a few
vectors of size `P` (the parameter count), never the `P × P`
Fisher matrix.

**The trade-off versus PPO**: TRPO does strictly more work per
update — the CG solve and line search are real overhead — to get
an exact constraint and a monotonic guarantee. PPO throws away
the guarantee, replaces all of that with one clip, and is far
simpler to implement and tune. In practice they reach comparable
performance, which is the whole reason PPO largely replaced TRPO.

---

## TRPO vs PPO — why the approximation won

TRPO is the more principled algorithm: an exact trust region, a
monotonic-improvement theorem, the natural gradient. PPO is the
cruder one: a clip, first-order gradients, no guarantee. And yet
**PPO is what everyone uses**. Why?

- **Simplicity.** PPO is a few lines on top of A2C. TRPO needs
  conjugate gradient, Fisher-vector products, and a line search —
  more code, more that can go subtly wrong.
- **Compatibility.** PPO is plain first-order SGD, so it drops
  straight into any deep-learning stack and any architecture
  (shared actor-critic bodies, recurrent nets, dropout). TRPO's
  second-order machinery is fussier.
- **Comparable results.** Empirically, PPO matches or beats TRPO
  on most benchmarks despite — or because of — its looseness.

TRPO is essential for *understanding* why constraining the step
matters, and it remains the rigorous reference. But it's the
classic story of a clean approximation winning on engineering
grounds. Knowing TRPO is knowing what PPO is approximating, and
why it gets away with it.

---

## What comes next

Part 3 turns to a different branch entirely: **Deep Deterministic
Policy Gradient (DDPG)**. Everything so far in the policy-gradient
line — REINFORCE, A2C, PPO, TRPO — has been *on-policy* and
*stochastic*. DDPG is **off-policy** and learns a **deterministic**
policy for **continuous** actions, bringing back DQN's replay
buffer and target networks and marrying them to an actor-critic.
It's where the value-based and policy-based families fuse into the
methods that dominate continuous control.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**trpo.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/09-advanced-reinforcement-learning/02-trust-region-policy-optimisation/trpo.py)

Run it with:

```bash
pip install numpy
python trpo.py
```

It needs only `numpy` and runs in well under a minute. Every
piece is from scratch: the actor and critic, GAE, the surrogate
gradient, conjugate gradient, Fisher-vector products via a
finite-difference Jacobian (no autograd), and the backtracking
line search. The headline insight worth pinning to the wall:
**TRPO enforces an exact trust region — it maximises the surrogate
subject to KL(π_old, π_new) ≤ δ — using the natural gradient
`F⁻¹g`, solved by conjugate gradient with Fisher-vector products
and a line search; the result is a monotonic, collapse-proof
climb where every single update's KL stays under δ (max 0.0094
for δ = 0.01), the exact guarantee PPO only approximates with a
clip**.

---

*This is Part 2 of the Advanced Reinforcement Learning track in the Algorithms in Python series. The companion script `trpo.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 1](https://medium.com/p/234490f03c81) covered PPO, the first-order approximation of the trust region this article enforces exactly. Part 3 will look at DDPG, an off-policy actor-critic for continuous control.*
