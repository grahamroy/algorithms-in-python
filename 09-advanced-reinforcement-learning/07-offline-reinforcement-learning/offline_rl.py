"""
offline_rl.py --- companion code for "Offline Reinforcement Learning"
(Advanced Reinforcement Learning, Part 7 -- the final article).

Every off-policy method in this series (DQN, DDPG, TD3, SAC) had a quiet
assumption: if the value function went wrong somewhere, the agent would soon
ACT there, observe the truth, and be corrected. Offline RL removes that safety
net. You get a FIXED dataset of transitions collected by some behaviour policy
-- no environment access, no new data, ever.

That breaks naive off-policy learning through EXTRAPOLATION ERROR:
  1. The critic is asked about actions the dataset never contains.
  2. Function approximation guesses -- sometimes far too high.
  3. Nothing ever corrects the guess (the action is never actually tried).
  4. The policy-improvement step actively SEEKS those phantom values.
The result is a policy optimised against fiction.

The fix demonstrated here is the minimalist one, TD3+BC (Fujimoto & Gu, 2021):
keep TD3 exactly as built in Part 4 and add ONE term to the actor loss --
a behavioural-cloning pull toward the dataset's actions:

    actor loss  =  - lambda * Q(s, pi(s))  +  ( pi(s) - a_data )^2

with lambda = alpha / mean|Q| so the two terms stay on comparable scales.
The BC term keeps the policy inside the data's support, where the critic can
be trusted; the Q term still improves the policy WITHIN that support.

Demonstrates (on Pendulum, with the usual "true value <= 0" lie detector):
  1. The dataset: 8,000 transitions from a mediocre behaviour policy.
  2. The headline: naive offline TD3 collapses (and its Q explodes to values
     that are provably impossible); TD3+BC learns a policy BETTER than the
     data it was given.
  3. The imitation <-> RL dial: BC alone copies the data; adding the Q term
     improves on it; removing the BC term collapses.

Everything is plain NumPy. Dependencies: numpy. Runs in ~60-90 seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Pendulum (identical to Parts 3-5)
# ---------------------------------------------------------------------------

class Pendulum:
    max_speed = 8.0
    max_torque = 2.0
    dt = 0.05
    g = 10.0
    m = 1.0
    length = 1.0
    max_steps = 200

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.th = 0.0
        self.thdot = 0.0
        self.steps = 0

    def reset(self):
        self.th = self.rng.uniform(-np.pi, np.pi)
        self.thdot = self.rng.uniform(-1.0, 1.0)
        self.steps = 0
        return self._obs()

    def _obs(self):
        return np.array([np.cos(self.th), np.sin(self.th), self.thdot])

    def step(self, u):
        u = float(np.asarray(u).reshape(-1)[0])
        u = float(np.clip(u, -self.max_torque, self.max_torque))
        angle = ((self.th + np.pi) % (2 * np.pi)) - np.pi
        cost = angle**2 + 0.1 * self.thdot**2 + 0.001 * u**2
        newthdot = self.thdot + (3 * self.g / (2 * self.length) * np.sin(self.th)
                                 + 3.0 / (self.m * self.length**2) * u) * self.dt
        newthdot = float(np.clip(newthdot, -self.max_speed, self.max_speed))
        newth = self.th + newthdot * self.dt
        self.th = newth
        self.thdot = newthdot
        self.steps += 1
        done = self.steps >= self.max_steps
        return self._obs(), -cost, done


# ---------------------------------------------------------------------------
# MLP and deterministic actor (as in the DDPG / TD3 articles)
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, sizes, seed=0, lr=1e-3):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            scale = np.sqrt(1.0 / sizes[i])
            self.W.append(rng.standard_normal((sizes[i], sizes[i + 1])) * scale)
            self.b.append(np.zeros(sizes[i + 1]))
        self.lr = lr
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        self.a = [X]
        h = X
        for i in range(len(self.W)):
            zz = h @ self.W[i] + self.b[i]
            h = np.tanh(zz) if i < len(self.W) - 1 else zz
            self.a.append(h)
        return h

    def predict(self, X):
        h = X
        for i in range(len(self.W)):
            zz = h @ self.W[i] + self.b[i]
            h = np.tanh(zz) if i < len(self.W) - 1 else zz
        return h

    def backward(self, dout):
        n = dout.shape[0]
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = dout
        for i in reversed(range(len(self.W))):
            gW[i] = (self.a[i].T @ delta) / n
            gb[i] = delta.sum(axis=0) / n
            delta = delta @ self.W[i].T
            if i > 0:
                delta = delta * (1 - self.a[i] ** 2)
        return gW, gb, delta

    def adam_apply(self, gW, gb):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            for M, V, g, P in ((self.mW, self.vW, gW, self.W),
                               (self.mb, self.vb, gb, self.b)):
                M[i] = b1 * M[i] + (1 - b1) * g[i]
                V[i] = b2 * V[i] + (1 - b2) * g[i] ** 2
                mhat = M[i] / (1 - b1 ** self.t)
                vhat = V[i] / (1 - b2 ** self.t)
                P[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def soft_update(self, other, tau):
        for i in range(len(self.W)):
            self.W[i] = tau * other.W[i] + (1 - tau) * self.W[i]
            self.b[i] = tau * other.b[i] + (1 - tau) * self.b[i]


class Actor:
    def __init__(self, s_dim, a_dim, hidden, bound, seed, lr):
        self.net = MLP([s_dim, hidden, hidden, a_dim], seed=seed, lr=lr)
        self.bound = bound

    def forward(self, S):
        self.pre = self.net.forward(S)
        return self.bound * np.tanh(self.pre)

    def predict(self, S):
        return self.bound * np.tanh(self.net.predict(S))

    def apply_output_grad(self, dL_da):
        """dL_da = dLoss/d(action); backprop through the tanh and the net."""
        dpre = dL_da * self.bound * (1 - np.tanh(self.pre) ** 2)
        gW, gb, _ = self.net.backward(dpre)
        self.net.adam_apply(gW, gb)

    def copy_from(self, other):
        self.net.copy_from(other.net)

    def soft_update(self, other, tau):
        self.net.soft_update(other.net, tau)


def q_of(critic, S, A):
    return critic.predict(np.concatenate([S, A], axis=1))


# ---------------------------------------------------------------------------
# Step 1: create the FIXED dataset.
# A behaviour policy of medium quality (TD3 trained briefly online), then
# 8,000 steps collected with small exploration noise. After this, the
# environment is never touched again by the learners.
# ---------------------------------------------------------------------------

def train_behaviour(episodes=20, seed=RNG_SEED, gamma=0.99, tau=0.01,
                    batch=128, hidden=64, expl_noise=0.2, warmup=1000,
                    policy_delay=2, policy_noise=0.2, noise_clip=0.5):
    """Standard TD3 (as in Part 4), trained briefly online."""
    env = Pendulum(seed=seed)
    s_dim, a_dim, bound = 3, 1, 2.0
    actor = Actor(s_dim, a_dim, hidden, bound, seed, 1e-3)
    c1 = MLP([4, hidden, hidden, 1], seed=seed + 5, lr=2e-3)
    c2 = MLP([4, hidden, hidden, 1], seed=seed + 9, lr=2e-3)
    at = Actor(s_dim, a_dim, hidden, bound, seed, 1e-3)
    c1t = MLP([4, hidden, hidden, 1], seed=seed + 5, lr=2e-3)
    c2t = MLP([4, hidden, hidden, 1], seed=seed + 9, lr=2e-3)
    at.copy_from(actor); c1t.copy_from(c1); c2t.copy_from(c2)
    S = np.zeros((50000, 3)); A = np.zeros((50000, 1))
    R = np.zeros((50000, 1)); S2 = np.zeros((50000, 3))
    n = 0
    rng = np.random.default_rng(seed + 3)
    upd = 0
    for ep in range(episodes):
        s = env.reset(); done = False
        while not done:
            if n < warmup:
                a = rng.uniform(-bound, bound, size=1)
            else:
                a = actor.predict(s[None])[0] + rng.normal(0, expl_noise * bound, 1)
            a = np.clip(a, -bound, bound)
            s2, r, done = env.step(a)
            S[n], A[n], R[n, 0], S2[n] = s, a, r, s2
            n += 1; s = s2
            if n >= batch and n >= warmup:
                j = rng.integers(0, n, size=batch)
                nz = np.clip(rng.normal(0, policy_noise * bound, (batch, 1)),
                             -noise_clip * bound, noise_clip * bound)
                a2 = np.clip(at.predict(S2[j]) + nz, -bound, bound)
                y = R[j] + gamma * np.minimum(q_of(c1t, S2[j], a2),
                                              q_of(c2t, S2[j], a2))
                for c in (c1, c2):
                    q = c.forward(np.concatenate([S[j], A[j]], axis=1))
                    gW, gb, _ = c.backward(q - y)
                    c.adam_apply(gW, gb)
                upd += 1
                if upd % policy_delay == 0:
                    ap = actor.forward(S[j])
                    c1.forward(np.concatenate([S[j], ap], axis=1))
                    _, _, din = c1.backward(np.ones((batch, 1)))
                    actor.apply_output_grad(-din[:, 3:])   # ascend Q
                    at.soft_update(actor, tau)
                    c1t.soft_update(c1, tau)
                    c2t.soft_update(c2, tau)
    return actor


def collect_dataset(behaviour, steps=8000, noise=0.2, seed=123):
    env = Pendulum(seed=seed)
    rng = np.random.default_rng(seed + 1)
    S = np.zeros((steps, 3)); A = np.zeros((steps, 1))
    R = np.zeros((steps, 1)); S2 = np.zeros((steps, 3))
    ep_returns, ep_ret = [], 0.0
    s = env.reset()
    for t in range(steps):
        a = behaviour.predict(s[None])[0] + rng.normal(0, noise * 2.0, 1)
        a = np.clip(a, -2.0, 2.0)
        s2, r, done = env.step(a)
        S[t], A[t], R[t, 0], S2[t] = s, a, r, s2
        ep_ret += r
        s = s2
        if done:
            ep_returns.append(ep_ret); ep_ret = 0.0
            s = env.reset()
    return (S, A, R, S2), ep_returns


# ---------------------------------------------------------------------------
# Step 2: OFFLINE training. No environment access -- only the fixed arrays.
#   mode='naive' : TD3 exactly as in Part 4, just fed the fixed dataset.
#   mode='td3bc' : one extra actor-loss term pulling pi(s) toward a_data.
#   mode='bc'    : behavioural cloning only (no Q term at all).
# ---------------------------------------------------------------------------

def offline_train(dataset, mode="td3bc", grad_steps=15000, alpha=1.0,
                  gamma=0.99, tau=0.01, batch=128, hidden=64,
                  policy_delay=2, policy_noise=0.2, noise_clip=0.5,
                  seed=RNG_SEED):
    S, A, R, S2 = dataset
    N = len(S)
    bound = 2.0
    actor = Actor(3, 1, hidden, bound, seed + 20, 1e-3)
    c1 = MLP([4, hidden, hidden, 1], seed=seed + 25, lr=2e-3)
    c2 = MLP([4, hidden, hidden, 1], seed=seed + 29, lr=2e-3)
    at = Actor(3, 1, hidden, bound, seed + 20, 1e-3)
    c1t = MLP([4, hidden, hidden, 1], seed=seed + 25, lr=2e-3)
    c2t = MLP([4, hidden, hidden, 1], seed=seed + 29, lr=2e-3)
    at.copy_from(actor); c1t.copy_from(c1); c2t.copy_from(c2)
    rng = np.random.default_rng(seed + 31)

    q_trace = []
    for step in range(grad_steps):
        j = rng.integers(0, N, size=batch)
        if mode in ("naive", "td3bc"):
            nz = np.clip(rng.normal(0, policy_noise * bound, (batch, 1)),
                         -noise_clip * bound, noise_clip * bound)
            a2 = np.clip(at.predict(S2[j]) + nz, -bound, bound)
            y = R[j] + gamma * np.minimum(q_of(c1t, S2[j], a2),
                                          q_of(c2t, S2[j], a2))
            for c in (c1, c2):
                q = c.forward(np.concatenate([S[j], A[j]], axis=1))
                gW, gb, _ = c.backward(q - y)
                c.adam_apply(gW, gb)

        # let the critics settle before the first actor update (td3bc only)
        actor_ok = (mode != "td3bc") or (step >= 1000)
        if step % policy_delay == 0 and actor_ok:
            ap = actor.forward(S[j])
            if mode == "bc":
                # pure cloning: minimise (pi(s) - a_data)^2
                actor.apply_output_grad(2.0 * (ap - A[j]))
            else:
                c1.forward(np.concatenate([S[j], ap], axis=1))
                _, _, din = c1.backward(np.ones((batch, 1)))
                dq_da = din[:, 3:]
                if mode == "naive":
                    actor.apply_output_grad(-dq_da)          # pure Q ascent
                else:                                        # td3bc
                    q_data = q_of(c1, S[j], A[j])
                    lam = alpha / max(float(np.mean(np.abs(q_data))), 1.0)
                    actor.apply_output_grad(-lam * dq_da
                                            + 2.0 * (ap - A[j]))
            at.soft_update(actor, tau)
            if mode != "bc":
                c1t.soft_update(c1, tau)
                c2t.soft_update(c2, tau)

        if step % 1500 == 0 and mode != "bc":
            ap = actor.predict(S[:512])
            q_trace.append(float(np.mean(q_of(c1, S[:512], ap))))
    return actor, c1, q_trace


# ---------------------------------------------------------------------------
# Evaluation (the environment is used ONLY to grade the final policies)
# ---------------------------------------------------------------------------

def mean_return(actor, seeds=(11, 22, 33, 44, 55)):
    rets = []
    for sd in seeds:
        env = Pendulum(seed=sd)
        s = env.reset()
        total, done = 0.0, False
        while not done:
            a = actor.predict(s[None])[0]
            s, r, done = env.step(a)
            total += r
        rets.append(total)
    return float(np.mean(rets))


def discounted_returns(rewards, gamma):
    G = np.zeros(len(rewards))
    run = 0.0
    for t in reversed(range(len(rewards))):
        run = rewards[t] + gamma * run
        G[t] = run
    return G


def value_calibration(actor, critic, gamma=0.99, seed=555):
    env = Pendulum(seed=seed)
    s = env.reset()
    S, rewards = [], []
    done = False
    while not done:
        a = actor.predict(s[None])[0]
        S.append(s)
        s, r, done = env.step(a)
        rewards.append(r)
    S = np.array(S)
    G = discounted_returns(rewards, gamma)
    Q = q_of(critic, S, actor.predict(S))[:, 0]
    return float(np.mean(Q)), float(np.mean(G))


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def main() -> None:
    banner("DEMO 1 --- The setting: a fixed dataset, no environment access")
    behaviour = train_behaviour()
    b_ret = mean_return(behaviour)
    dataset, ep_returns = collect_dataset(behaviour)
    d_ret = float(np.mean(ep_returns))
    print("  A medium-quality policy (TD3, trained briefly online) collected the")
    print("  data WITH its exploration noise, then the environment was switched")
    print("  off for good. The learners below never take a single step in it.")
    print()
    print(f"  Dataset size                      : {len(dataset[0])} transitions "
          f"({len(ep_returns)} episodes)")
    print(f"  Mean episode return IN the data   : {d_ret:8.1f}   "
          f"(what the dataset looks like)")
    print(f"  The collector, without its noise  : {b_ret:8.1f}   (for reference)")
    print("  (0 is a perfectly balanced pendulum; random is about -1300)")

    banner("DEMO 2 --- Naive off-policy learning fails offline; TD3+BC fixes it")
    print("  Both learners see the SAME fixed dataset for 15,000 gradient steps.")
    print("  Naive = the TD3 of Part 4, unchanged. TD3+BC = one extra actor term.")
    print()
    naive_actor, naive_c1, naive_q = offline_train(dataset, mode="naive")
    bc_actor, bc_c1, _ = offline_train(dataset, mode="td3bc")
    n_ret = mean_return(naive_actor)
    t_ret = mean_return(bc_actor)
    nq, ng = value_calibration(naive_actor, naive_c1)
    tq, tg = value_calibration(bc_actor, bc_c1)
    print(f"  The data it learned from   : return {d_ret:8.1f}")
    print(f"  Naive offline TD3          : return {n_ret:8.1f}   "
          f"(worse than random!)")
    print(f"  TD3+BC                     : return {t_ret:8.1f}   "
          f"(better than its data)")
    print()
    print("  The critic's story (every true value on Pendulum is <= 0):")
    print(f"    naive : predicted Q = {nq:+9.1f}   actual return = {ng:8.1f}")
    print(f"    TD3+BC: predicted Q = {tq:+9.1f}   actual return = {tg:8.1f}")
    print()
    print("  Naive Q over training (sampled every 1,500 steps) -- watch it grow")
    print("  past zero into impossible territory, with nothing to correct it:")
    print("    " + "  ".join(f"{q:+.0f}" for q in naive_q))

    banner("DEMO 3 --- The dial from imitation to reinforcement learning")
    print("  The BC term anchors the policy to the data; the Q term improves it.")
    print("  All three learners saw exactly the same 8,000 transitions:")
    print()
    pure_bc, _, _ = offline_train(dataset, mode="bc")
    p_ret = mean_return(pure_bc)
    print(f"    BC only  (copy the data)          : return {p_ret:8.1f}")
    print(f"    TD3+BC   (improve within the data): return {t_ret:8.1f}")
    print(f"    naive    (trust Q everywhere)     : return {n_ret:8.1f}")
    print()
    print(f"    the dataset itself averaged       : return {d_ret:8.1f}")
    print(f"    the collector, noise-free         : return {b_ret:8.1f}")
    print()
    print("  Cloning matches the data. Adding the Q term BEATS the data --")
    print("  offline RL's promise -- while dropping the BC anchor collapses.")


if __name__ == "__main__":
    main()
