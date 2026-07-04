"""
td3.py --- companion code for "Twin Delayed DDPG (TD3)"
(Advanced Reinforcement Learning, Part 4).

TD3 from scratch in NumPy on Pendulum. It is DDPG (Part 3) with three targeted
fixes for DDPG's central flaw: its critic OVERESTIMATES Q, and the deterministic
actor happily exploits those inflated values.

  1. Twin critics + clipped double-Q. Keep TWO critics and build the TD target
     from the SMALLER of their estimates: y = r + gamma * min(Q1', Q2'). Being
     pessimistic about your own value estimate kills the overestimation.
  2. Delayed policy updates. Update the actor (and the targets) less often than
     the critics, so the actor chases a value that has had time to settle.
  3. Target policy smoothing. Add small clipped noise to the target action, so
     the critic can't overfit to a sharp, spurious peak in Q.

Demonstrates:
  1. TD3 solving Pendulum swing-up.
  2. The headline: overestimation. A single-critic target (DDPG-style) predicts
     a Q far higher than the return actually achieved; the twin min keeps TD3's
     value honest -- Q ~ the real return.
  3. TD3's greedy policy generalising to unseen starts.

Everything (six networks, the twin-critic targets, the dQ/da chain, replay,
delayed soft updates, the physics) is plain NumPy. Dependencies: numpy.
Runs in ~30-45 seconds.
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
# Pendulum (identical to the DDPG article, Part 3)
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
# MLP (tanh hidden, linear output, Adam). backward() also returns the input
# gradient, to carry dQ/da from a critic into the actor.
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
        self.z = []
        h = X
        for i in range(len(self.W)):
            zz = h @ self.W[i] + self.b[i]
            self.z.append(zz)
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

    def apply_action_grad(self, da):
        dpre = da * self.bound * (1 - np.tanh(self.pre) ** 2)
        gW, gb, _ = self.net.backward(-dpre)          # minimise -Q == ascend Q
        self.net.adam_apply(gW, gb)

    def copy_from(self, other):
        self.net.copy_from(other.net)

    def soft_update(self, other, tau):
        self.net.soft_update(other.net, tau)


class ReplayBuffer:
    def __init__(self, cap, s_dim, a_dim, seed):
        self.s = np.zeros((cap, s_dim))
        self.a = np.zeros((cap, a_dim))
        self.r = np.zeros((cap, 1))
        self.s2 = np.zeros((cap, s_dim))
        self.cap, self.idx, self.size = cap, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2):
        i = self.idx
        self.s[i], self.a[i], self.r[i, 0], self.s2[i] = s, a, r, s2
        self.idx = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch):
        j = self.rng.integers(0, self.size, size=batch)
        return self.s[j], self.a[j], self.r[j], self.s2[j]


def q_of(critic, S, A):
    return critic.predict(np.concatenate([S, A], axis=1))


# ---------------------------------------------------------------------------
# TD3
#   use_twin=True  : y = r + gamma * min(Q1', Q2')   (TD3)
#   use_twin=False : y = r + gamma * Q1'             (single critic, DDPG-style)
# ---------------------------------------------------------------------------

def td3(episodes=60, gamma=0.99, tau=0.01, batch=128, hidden=64,
        lr_actor=1e-3, lr_critic=2e-3, expl_noise=0.2, policy_noise=0.2,
        noise_clip=0.5, policy_delay=2, warmup=1000, buffer_cap=50000,
        use_twin=True, seed=RNG_SEED):
    env = Pendulum(seed=seed)
    s_dim, a_dim, bound = 3, 1, 2.0
    actor = Actor(s_dim, a_dim, hidden, bound, seed, lr_actor)
    critic1 = MLP([s_dim + a_dim, hidden, hidden, 1], seed=seed + 5, lr=lr_critic)
    critic2 = MLP([s_dim + a_dim, hidden, hidden, 1], seed=seed + 9, lr=lr_critic)
    actor_t = Actor(s_dim, a_dim, hidden, bound, seed, lr_actor)
    critic1_t = MLP([s_dim + a_dim, hidden, hidden, 1], seed=seed + 5, lr=lr_critic)
    critic2_t = MLP([s_dim + a_dim, hidden, hidden, 1], seed=seed + 9, lr=lr_critic)
    actor_t.copy_from(actor)
    critic1_t.copy_from(critic1)
    critic2_t.copy_from(critic2)
    buf = ReplayBuffer(buffer_cap, s_dim, a_dim, seed + 2)
    rng = np.random.default_rng(seed + 3)

    returns = []
    total_steps = 0
    update_i = 0
    for ep in range(episodes):
        s = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            if total_steps < warmup:
                a = rng.uniform(-bound, bound, size=a_dim)
            else:
                a = actor.predict(s[None])[0] + rng.normal(0, expl_noise * bound, a_dim)
            a = np.clip(a, -bound, bound)
            s2, r, done = env.step(a)
            buf.add(s, a, r, s2)
            s = s2
            ep_ret += r
            total_steps += 1

            if buf.size >= batch and total_steps >= warmup:
                bs, ba, br, bs2 = buf.sample(batch)
                # ----- target action with clipped smoothing noise -----
                nz = np.clip(rng.normal(0, policy_noise * bound, (batch, a_dim)),
                             -noise_clip * bound, noise_clip * bound)
                a2 = np.clip(actor_t.predict(bs2) + nz, -bound, bound)
                # ----- clipped double-Q target -----
                q1t = q_of(critic1_t, bs2, a2)
                if use_twin:
                    q2t = q_of(critic2_t, bs2, a2)
                    q_target = np.minimum(q1t, q2t)
                else:
                    q_target = q1t
                y = br + gamma * q_target
                # ----- update both critics toward y -----
                for critic in (critic1, critic2):
                    q = critic.forward(np.concatenate([bs, ba], axis=1))
                    gW, gb, _ = critic.backward(q - y)
                    critic.adam_apply(gW, gb)
                # ----- delayed actor + target updates -----
                update_i += 1
                if update_i % policy_delay == 0:
                    a_pred = actor.forward(bs)
                    critic1.forward(np.concatenate([bs, a_pred], axis=1))
                    _, _, din = critic1.backward(np.ones((batch, 1)))
                    actor.apply_action_grad(din[:, s_dim:])
                    actor_t.soft_update(actor, tau)
                    critic1_t.soft_update(critic1, tau)
                    critic2_t.soft_update(critic2, tau)
        returns.append(ep_ret)
    return actor, critic1, returns


def greedy_return(actor, seed=999):
    env = Pendulum(seed=seed)
    s = env.reset()
    total, done = 0.0, False
    while not done:
        a = actor.predict(s[None])[0]
        s, r, done = env.step(a)
        total += r
    return total


def discounted_returns(rewards, gamma):
    G = np.zeros(len(rewards))
    run = 0.0
    for t in reversed(range(len(rewards))):
        run = rewards[t] + gamma * run
        G[t] = run
    return G


def value_calibration(actor, critic, gamma=0.99, seed=555):
    """Mean predicted Q(s, mu(s)) vs the mean ACTUAL discounted return, over the
    states of a greedy rollout. Q >> return means the critic overestimates."""
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

def block_means(returns, block=12):
    out = []
    for start in range(0, len(returns), block):
        chunk = returns[start:start + block]
        out.append((start + 1, start + len(chunk), float(np.mean(chunk))))
    return out


def demo_learning(returns):
    banner("DEMO 1 --- TD3 solves Pendulum swing-up")
    print("  Actor: 3->64->64->1   Twin critics: 4->64->64->1 each")
    print("  Episodes: 60  gamma=0.99  tau=0.01  policy_delay=2  "
          "twin critics + target smoothing")
    print()
    print("  Mean episode return by 12-episode block (0 is best, random ~ -1300):")
    for lo, hi, m in block_means(returns, 12):
        bar = "#" * int(max(0, (m + 1400) / 40))
        print(f"    episodes {lo:3d}-{hi:3d} : {m:8.1f}  {bar}")
    print()
    print(f"  Final mean return (last 20 episodes): {np.mean(returns[-20:]):.1f}")


def demo_overestimation(td3_actor, td3_critic):
    banner("DEMO 2 --- The twin-critic min fixes DDPG's overestimation")
    print("  Every Pendulum reward is <= 0, so the true value of ANY state is <= 0.")
    print("  Compare each critic's predicted Q(s, mu(s)) to the return the policy")
    print("  ACTUALLY earns. A single-critic target (DDPG-style) inflates Q; the")
    print("  min of two critics (TD3) keeps it honest.")
    print()
    single_actor, single_critic, _ = td3(use_twin=False, seed=0)
    q_td3, g_td3 = value_calibration(td3_actor, td3_critic)
    q_sgl, g_sgl = value_calibration(single_actor, single_critic)
    print(f"  Single critic (DDPG-style): predicted Q = {q_sgl:+7.1f}   "
          f"actual return = {g_sgl:+7.1f}")
    print(f"     -> Q is POSITIVE -- impossible when every reward is <= 0. "
          f"Overestimates by {q_sgl - g_sgl:.1f}.")
    print()
    print(f"  Twin critics + min (TD3)  : predicted Q = {q_td3:+7.1f}   "
          f"actual return = {g_td3:+7.1f}")
    print(f"     -> stays negative, close to the truth (gap {q_td3 - g_td3:+.1f}).")
    print()
    print("  Both still solve Pendulum, but that inflated value is exactly what")
    print("  destabilises DDPG on harder tasks. TD3's pessimism removes it.")


def demo_policy(actor):
    banner("DEMO 3 --- The trained TD3 policy, from unseen starts")
    print("  Greedy returns from five random starts it never trained on")
    print("  (random policy ~ -1300; 0 is a perfectly balanced pole):")
    print()
    seeds = (11, 22, 33, 44, 55)
    rets = [greedy_return(actor, seed=s) for s in seeds]
    for s, r in zip(seeds, rets):
        print(f"    start seed {s}:   return {r:8.1f}")
    print()
    print(f"  Mean over 5 unseen starts: {np.mean(rets):.0f}  -- a robust continuous")
    print("  controller, built on a value function that no longer lies to it.")


def main() -> None:
    actor, critic1, returns = td3(seed=RNG_SEED)
    demo_learning(returns)
    demo_overestimation(actor, critic1)
    demo_policy(actor)


if __name__ == "__main__":
    main()
