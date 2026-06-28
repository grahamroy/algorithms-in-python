"""
reinforce.py --- companion code for "Policy Gradient Methods"
(Reinforcement Learning, Part 4).

REINFORCE (Williams, 1992) from scratch in NumPy on CartPole. Every method
so far (Q-Learning, SARSA, DQN) learned VALUES and acted greedily by argmax.
REINFORCE learns the POLICY directly: a network outputs a probability for each
action, and we push up the probability of actions that led to high return.

The update is the policy gradient:
    grad J(theta) = E[ sum_t  grad log pi(a_t | s_t; theta) * G_t ]
where G_t is the discounted return following step t. There is no value table
and no max -- the policy IS the network's softmax output.

Demonstrates:
  1. REINFORCE learning to balance CartPole, learning a STOCHASTIC policy.
  2. Ablation: the baseline. Subtracting the mean return (a baseline) cuts the
     gradient's variance and is the difference between learning and flailing.
  3. The learned policy as probabilities -- it outputs a distribution over
     actions, not a single argmax (the key contrast with DQN).

Everything (the policy network, softmax, the policy-gradient backprop, Adam,
the environment) is plain NumPy. Dependencies: numpy. Runs in ~15-30 seconds.
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
# CartPole environment (identical to the DQN article, Part 3)
# ---------------------------------------------------------------------------

class CartPole:
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    theta_threshold = 12 * 2 * np.pi / 360
    x_threshold = 2.4

    def __init__(self, seed=0, max_steps=500):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.state = None
        self.steps = 0

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        fell = (abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
        done = bool(fell or self.steps >= self.max_steps)
        return self.state.copy(), 1.0, done


# ---------------------------------------------------------------------------
# Policy network: MLP outputting action LOGITS; softmax gives pi(a|s).
#   He init, tanh hidden (smooth policies train well), linear output, Adam.
# ---------------------------------------------------------------------------

class PolicyNet:
    def __init__(self, sizes, seed=0, lr=1e-2):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            self.W.append(rng.standard_normal((sizes[i], sizes[i + 1]))
                          * np.sqrt(1.0 / sizes[i]))
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

    def backward_step(self, dout):
        n = dout.shape[0]
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = dout
        for i in reversed(range(len(self.W))):
            gW[i] = (self.a[i].T @ delta) / n
            gb[i] = delta.sum(axis=0) / n
            if i > 0:
                delta = (delta @ self.W[i].T) * (1 - self.a[i] ** 2)  # tanh grad
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

    def predict(self, X):
        h = X
        for i in range(len(self.W)):
            zz = h @ self.W[i] + self.b[i]
            h = np.tanh(zz) if i < len(self.W) - 1 else zz
        return h


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def discounted_returns(rewards, gamma):
    G = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


# ---------------------------------------------------------------------------
# REINFORCE: Monte-Carlo policy gradient, one update per episode.
#   The only difference the baseline makes: whether we CENTER the returns
#   (subtract their mean) before weighting the gradient. Both arms divide by
#   the std so the learning rate is comparable -- isolating the baseline.
# ---------------------------------------------------------------------------

def reinforce(episodes=800, gamma=0.99, lr=1e-2, hidden=128,
              use_baseline=True, seed=RNG_SEED):
    env = CartPole(seed=seed)
    policy = PolicyNet([4, hidden, 2], seed=seed, lr=lr)
    rng = np.random.default_rng(seed + 1)
    returns_log = []

    for ep in range(episodes):
        states, actions, rewards = [], [], []
        s = env.reset()
        done = False
        while not done:
            probs = softmax(policy.predict(s[None])[0])
            a = int(rng.choice(2, p=probs))
            s2, r, done = env.step(a)
            states.append(s); actions.append(a); rewards.append(r)
            s = s2
        returns_log.append(sum(rewards))

        G = discounted_returns(rewards, gamma)
        std = G.std() + 1e-8
        G = (G - G.mean()) / std if use_baseline else G / std

        S = np.array(states)
        probs = softmax(policy.forward(S))           # (T, 2), caches
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(actions)), actions] = 1
        # dL/dlogits for L = -sum_t G_t log pi(a_t):  G_t * (probs - onehot)
        dout = (probs - onehot) * G[:, None]
        policy.backward_step(dout)

    return policy, returns_log


def greedy_rollout(policy, seed=12345, max_steps=500):
    env = CartPole(seed=seed, max_steps=max_steps)
    s = env.reset()
    total, done = 0.0, False
    while not done:
        a = int(np.argmax(policy.predict(s[None])[0]))
        s, r, done = env.step(a)
        total += r
    return int(total)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def block_means(returns, block=80):
    out = []
    for start in range(0, len(returns), block):
        chunk = returns[start:start + block]
        out.append((start + 1, start + len(chunk), float(np.mean(chunk))))
    return out


def demo_learning(returns):
    banner("DEMO 1 --- REINFORCE learns CartPole (a stochastic policy)")
    print("  Policy: 4 -> 128 -> 2, softmax over actions  ->  pi(a | s)")
    print("  Episodes: 800  gamma=0.99  lr=1e-2  Monte-Carlo, one update/episode")
    print()
    print("  Mean return by 80-episode block (max possible = 500):")
    for lo, hi, m in block_means(returns, 80):
        bar = "#" * int(m / 12)
        print(f"    episodes {lo:3d}-{hi:3d} : {m:6.1f}  {bar}")
    print()
    print(f"  Final mean return (last 100 episodes): {np.mean(returns[-100:]):.1f}")


def demo_baseline():
    banner("DEMO 2 --- Why a baseline matters (variance reduction)")
    print("  The SAME REINFORCE; the only change is whether we subtract the")
    print("  mean return (the baseline) before weighting the gradient.")
    print()
    _, with_b = reinforce(use_baseline=True)
    _, without_b = reinforce(use_baseline=False)
    mw, mo = np.mean(with_b[-100:]), np.mean(without_b[-100:])
    print(f"  With baseline    final mean return: {mw:6.1f}")
    print(f"  Without baseline final mean return: {mo:6.1f}")
    print()
    print(f"  The baseline lifts the final return by {mw - mo:.1f}. Without it every")
    print("  return is positive, so every action taken is reinforced and the policy")
    print('  can\'t tell good from bad. The baseline turns "return" into "better or')
    print('  worse than average" -- the signal that actually teaches the policy.')


def demo_stochastic(policy):
    banner("DEMO 3 --- The learned policy is a DISTRIBUTION, not an argmax")
    env = CartPole(seed=999)
    s = env.reset()
    print("  pi(a | s) at a few visited states (push-left, push-right):")
    for k in range(5):
        probs = softmax(policy.predict(s[None])[0])
        print(f"    state {k}:  left {probs[0]:.2f}   right {probs[1]:.2f}")
        a = int(np.argmax(probs))
        s, _, done = env.step(a)
        if done:
            s = env.reset()
    print()
    lengths = [greedy_rollout(policy, seed=s) for s in (101, 202, 303)]
    print(f"  Greedy rollouts (take argmax): {lengths}  "
          f"-> mean {np.mean(lengths):.0f} / 500 steps")


def main() -> None:
    policy, returns = reinforce(seed=RNG_SEED)
    demo_learning(returns)
    demo_baseline()
    demo_stochastic(policy)


if __name__ == "__main__":
    main()
