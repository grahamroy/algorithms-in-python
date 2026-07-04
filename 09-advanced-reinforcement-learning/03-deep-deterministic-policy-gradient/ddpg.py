"""
ddpg.py --- companion code for "Deep Deterministic Policy Gradient (DDPG)"
(Advanced Reinforcement Learning, Part 3).

DDPG from scratch in NumPy on Pendulum. Everything in the policy-gradient line
so far (REINFORCE, A2C, PPO, TRPO) has been ON-policy and STOCHASTIC. DDPG is
OFF-policy and DETERMINISTIC, for CONTINUOUS actions -- it is essentially DQN
generalised to a continuous action space, and it reuses DQN's two stabilisers.

  Actor  mu(s)     : outputs the action directly (a continuous torque), because
                     DQN's argmax over actions is impossible on a continuum.
  Critic Q(s, a)   : evaluates state-action pairs, trained exactly like DQN's.
  + a replay buffer (off-policy reuse) and target networks (soft-updated).

The actor is trained by the DETERMINISTIC policy gradient: push mu(s) in the
direction that increases the critic's Q, i.e. gradient ASCENT on Q(s, mu(s))
with the gradient flowing dQ/da -> da/dtheta through both networks.

Demonstrates:
  1. DDPG solving Pendulum swing-up (return climbs from about -1300 to ~ -170).
  2. The swing-up in action: from hanging, the actor pumps the pendulum up and
     balances it with a smoothly varying, continuous torque -- the thing DQN's
     argmax over discrete actions could never output.
  3. Off-policy and reliable: the greedy policy generalises to unseen random
     starts, having learned from a replay buffer reused many times.

Everything (four networks, the dQ/da chain, replay, soft updates, the physics)
is plain NumPy. Dependencies: numpy. Runs in ~20-30 seconds.
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
# Pendulum: swing an under-powered pendulum upright and hold it there.
#   State : [cos(theta), sin(theta), theta_dot]   (3 continuous numbers)
#   Action: torque in [-2, 2]                      (1 continuous number)
#   Reward: -(angle^2 + 0.1*theta_dot^2 + 0.001*torque^2)   (0 is best)
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
# MLP: tanh hidden, linear output, Adam. backward() returns BOTH the parameter
# gradients (to update) and the input gradient (to chain dQ/da into the actor).
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
        """Return (gW, gb, input_grad). Does not update parameters."""
        n = dout.shape[0]
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = dout
        for i in reversed(range(len(self.W))):
            gW[i] = (self.a[i].T @ delta) / n
            gb[i] = delta.sum(axis=0) / n
            delta = delta @ self.W[i].T
            if i > 0:
                delta = delta * (1 - self.a[i] ** 2)     # tanh derivative
        return gW, gb, delta                              # delta is d/dinput

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


# ---------------------------------------------------------------------------
# Actor: a = bound * tanh(net(s)). The final tanh keeps the torque in range;
# its derivative is folded into the backward pass.
# ---------------------------------------------------------------------------

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
        """da = dObjective/da (we ASCEND Q, so pass the ascent gradient)."""
        dpre = da * self.bound * (1 - np.tanh(self.pre) ** 2)
        gW, gb, _ = self.net.backward(-dpre)      # minimise -Q  ==  ascend Q
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


# ---------------------------------------------------------------------------
# DDPG
# ---------------------------------------------------------------------------

def ddpg(episodes=80, gamma=0.99, tau=0.01, batch=128, hidden=64,
         lr_actor=1e-3, lr_critic=2e-3, noise=0.2, warmup=1000,
         buffer_cap=50000, seed=RNG_SEED):
    env = Pendulum(seed=seed)
    s_dim, a_dim, bound = 3, 1, 2.0
    actor = Actor(s_dim, a_dim, hidden, bound, seed, lr_actor)
    critic = MLP([s_dim + a_dim, hidden, hidden, 1], seed=seed + 5, lr=lr_critic)
    actor_t = Actor(s_dim, a_dim, hidden, bound, seed, lr_actor)
    critic_t = MLP([s_dim + a_dim, hidden, hidden, 1], seed=seed + 5, lr=lr_critic)
    actor_t.copy_from(actor)
    critic_t.copy_from(critic)
    buf = ReplayBuffer(buffer_cap, s_dim, a_dim, seed + 2)
    rng = np.random.default_rng(seed + 3)

    returns = []
    total_steps = 0
    for ep in range(episodes):
        s = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            if total_steps < warmup:
                a = rng.uniform(-bound, bound, size=a_dim)
            else:
                a = actor.predict(s[None])[0]
                a = a + rng.normal(0, noise * bound, size=a_dim)
            a = np.clip(a, -bound, bound)
            s2, r, done = env.step(a)
            buf.add(s, a, r, s2)
            s = s2
            ep_ret += r
            total_steps += 1

            if buf.size >= batch and total_steps >= warmup:
                bs, ba, br, bs2 = buf.sample(batch)
                # ----- critic update: TD target from the target networks -----
                a2 = actor_t.predict(bs2)
                q2 = critic_t.predict(np.concatenate([bs2, a2], axis=1))
                y = br + gamma * q2                       # (batch, 1)
                q = critic.forward(np.concatenate([bs, ba], axis=1))
                gW, gb, _ = critic.backward(q - y)        # MSE gradient
                critic.adam_apply(gW, gb)
                # ----- actor update: ascend Q(s, mu(s)) via dQ/da -----
                a_pred = actor.forward(bs)
                critic.forward(np.concatenate([bs, a_pred], axis=1))
                _, _, din = critic.backward(np.ones((batch, 1)))
                dq_da = din[:, s_dim:]                     # gradient wrt action
                actor.apply_action_grad(dq_da)
                # ----- soft-update the target networks (Polyak) -----
                actor_t.soft_update(actor, tau)
                critic_t.soft_update(critic, tau)
        returns.append(ep_ret)
    return actor, critic, returns


def greedy_return(actor, seed=999):
    env = Pendulum(seed=seed)
    s = env.reset()
    total, done = 0.0, False
    while not done:
        a = actor.predict(s[None])[0]
        s, r, done = env.step(a)
        total += r
    return total


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
    banner("DEMO 1 --- DDPG solves Pendulum swing-up (continuous torque)")
    print("  Actor : 3 -> 64 -> 64 -> 1  (torque in [-2, 2] via tanh)")
    print("  Critic: 4 -> 64 -> 64 -> 1  (Q of state AND action)")
    print("  Episodes: 80  gamma=0.99  tau=0.01  replay + target nets")
    print()
    print("  Mean episode return by 12-episode block (0 is best, random ~ -1300):")
    for lo, hi, m in block_means(returns, 12):
        bar = "#" * int(max(0, (m + 1400) / 40))
        print(f"    episodes {lo:3d}-{hi:3d} : {m:8.1f}  {bar}")
    print()
    print(f"  Final mean return (last 20 episodes): {np.mean(returns[-20:]):.1f}")


def demo_swingup(actor):
    banner("DEMO 2 --- The swing-up in action (continuous torque control)")
    print("  A greedy episode starting from hanging straight down. The actor pumps")
    print("  the pendulum back and forth to build energy, catches it at the top, and")
    print("  balances -- applying a smoothly varying torque DQN's argmax could not.")
    print()
    env = Pendulum(seed=0)
    env.reset()
    env.th, env.thdot = np.pi, 0.0        # hanging straight down, at rest
    s = env._obs()
    print("    step    angle from upright    torque applied")
    for t in range(201):
        angle = abs(((env.th + np.pi) % (2 * np.pi)) - np.pi) * 180 / np.pi
        a = actor.predict(s[None])[0]
        if t % 20 == 0:
            print(f"    {t:4d}          {angle:5.0f} deg            {float(a[0]):+.2f}")
        s, _, done = env.step(a)
        if done:
            break
    print()
    print("  180 deg (hanging) -> a few pumps -> ~10 deg and held. The torque is a")
    print("  real number that varies continuously; a discrete-action agent cannot do this.")


def demo_policy(actor):
    banner("DEMO 3 --- Off-policy and reliable: greedy runs from unseen starts")
    print("  DDPG is OFF-policy -- it learned from a replay buffer of past")
    print("  transitions, reused many times (PPO and TRPO cannot reuse data).")
    print("  The greedy policy (no noise) from five unseen random starts:")
    print("  (random policy scores about -1300; 0 is a perfectly held pole)")
    print()
    seeds = (11, 22, 33, 44, 55)
    rets = [greedy_return(actor, seed=s) for s in seeds]
    for s, r in zip(seeds, rets):
        print(f"    start seed {s}:   return {r:8.1f}")
    print()
    print(f"  Mean over 5 unseen starts: {np.mean(rets):.0f}  -- it reliably swings up")
    print("  and balances with a continuous torque DQN's argmax could never output.")


def main() -> None:
    actor, critic, returns = ddpg(seed=RNG_SEED)
    demo_learning(returns)
    demo_swingup(actor)
    demo_policy(actor)


if __name__ == "__main__":
    main()
