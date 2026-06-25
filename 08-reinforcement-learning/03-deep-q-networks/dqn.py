"""
dqn.py --- companion code for "Deep Q-Networks (DQN)"
(Reinforcement Learning, Part 3).

DQN from scratch in NumPy on CartPole. Q-Learning and SARSA (Parts 1-2)
stored one value per state-action pair in a table -- impossible when the
state is 4 continuous numbers. DQN replaces the table with a small neural
network Q(s, a; theta) and adds the two tricks that make function
approximation stable:
  1. Experience replay   -- learn from a buffer of past transitions,
                            breaking correlation between consecutive steps.
  2. A target network    -- bootstrap from a slowly-updated copy of the
                            network, so the target doesn't chase the weights.

Demonstrates:
  1. DQN learning to balance CartPole (return climbs from ~20 to near the cap).
  2. Ablation: the SAME algorithm with vs without the target network --
     the target network is what makes it stable.
  3. The learned greedy policy balancing the pole for a full rollout.

Everything (the MLP, Adam, backprop, replay, the environment) is plain
NumPy -- no autograd, no RL library. Dependencies: numpy.
Runs in roughly 20-40 seconds (DQN is heavier than the tabular methods).
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
# CartPole environment (the classic control benchmark, standard dynamics)
#   State : [cart position, cart velocity, pole angle, pole angular velocity]
#   Action: 0 = push left, 1 = push right
#   Reward: +1 for every step the pole stays up
#   Done  : pole falls past 12 degrees, cart leaves the track, or step cap
# ---------------------------------------------------------------------------

class CartPole:
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5                       # half the pole's length
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02                         # seconds between state updates
    theta_threshold = 12 * 2 * np.pi / 360   # 12 degrees in radians
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

        # semi-implicit Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        fell = (abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
        truncated = self.steps >= self.max_steps
        done = bool(fell or truncated)
        return self.state.copy(), 1.0, done


# ---------------------------------------------------------------------------
# A tiny multilayer perceptron: He init, ReLU hidden, linear output, Adam.
#   forward() caches activations; backward_step() does one Adam update from
#   the gradient at the output. No autograd -- the chain rule by hand.
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, sizes, seed=0, lr=1e-3):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            # He initialisation for ReLU layers
            self.W.append(rng.standard_normal((sizes[i], sizes[i + 1]))
                          * np.sqrt(2.0 / sizes[i]))
            self.b.append(np.zeros(sizes[i + 1]))
        self.lr = lr
        # Adam moments
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, X):
        self.a = [X]      # activations (a[0] = input)
        self.z = []       # pre-activations
        h = X
        for i in range(len(self.W)):
            z = h @ self.W[i] + self.b[i]
            self.z.append(z)
            h = np.maximum(0.0, z) if i < len(self.W) - 1 else z  # ReLU / linear
            self.a.append(h)
        return h

    def backward_step(self, dout):
        """One Adam step. dout = dLoss/d(output), shape (batch, n_out)."""
        n = dout.shape[0]
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = dout
        for i in reversed(range(len(self.W))):
            gW[i] = (self.a[i].T @ delta) / n
            gb[i] = delta.sum(axis=0) / n
            if i > 0:
                delta = (delta @ self.W[i].T) * (self.z[i - 1] > 0)  # ReLU grad

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

    def copy_weights_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def predict(self, X):
        """Forward pass WITHOUT touching the training cache."""
        h = X
        for i in range(len(self.W)):
            z = h @ self.W[i] + self.b[i]
            h = np.maximum(0.0, z) if i < len(self.W) - 1 else z
        return h


# ---------------------------------------------------------------------------
# Replay buffer: a ring of past transitions to sample minibatches from.
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity, state_dim, seed=0):
        self.cap = capacity
        self.s = np.zeros((capacity, state_dim))
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity)
        self.s2 = np.zeros((capacity, state_dim))
        self.done = np.zeros(capacity)
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done):
        i = self.idx
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.s2[i], self.done[i] = s2, float(done)
        self.idx = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch):
        j = self.rng.integers(0, self.size, size=batch)
        return self.s[j], self.a[j], self.r[j], self.s2[j], self.done[j]


# ---------------------------------------------------------------------------
# DQN training
# ---------------------------------------------------------------------------

def dqn(episodes=250, gamma=0.99, lr=1e-3, batch=64, buffer_cap=10000,
        target_sync=200, warmup=1000, eps_start=1.0, eps_end=0.05,
        eps_decay_steps=10000, use_target_network=True, seed=RNG_SEED):
    env = CartPole(seed=seed)
    online = MLP([4, 64, 64, 2], seed=seed, lr=lr)
    target = MLP([4, 64, 64, 2], seed=seed, lr=lr)
    target.copy_weights_from(online)
    buf = ReplayBuffer(buffer_cap, 4, seed=seed + 1)
    act_rng = np.random.default_rng(seed + 2)

    returns = []
    step_count = 0
    for ep in range(episodes):
        s = env.reset()
        ep_return = 0.0
        done = False
        while not done:
            eps = max(eps_end, eps_start - (eps_start - eps_end)
                      * step_count / eps_decay_steps)
            if act_rng.random() < eps:
                a = int(act_rng.integers(2))
            else:
                a = int(np.argmax(online.predict(s[None])[0]))

            s2, r, done = env.step(a)
            buf.add(s, a, r, s2, done)
            s = s2
            ep_return += r
            step_count += 1

            if buf.size >= warmup:
                bs, ba, br, bs2, bdone = buf.sample(batch)
                # target = r + gamma * max_a' Q_target(s', a') * (1 - done)
                net_for_target = target if use_target_network else online
                q_next = net_for_target.predict(bs2).max(axis=1)
                td_target = br + gamma * q_next * (1 - bdone)

                q_pred_all = online.forward(bs)            # (batch, 2), caches
                q_pred = q_pred_all[np.arange(batch), ba]
                # MSE gradient on the taken action only (the target network,
                # not gradient clipping, is what keeps this stable)
                err = q_pred - td_target
                dout = np.zeros_like(q_pred_all)
                dout[np.arange(batch), ba] = err
                online.backward_step(dout)

                if use_target_network and step_count % target_sync == 0:
                    target.copy_weights_from(online)

        returns.append(ep_return)
    return online, returns


def greedy_rollout(net, seed=12345, max_steps=500):
    env = CartPole(seed=seed, max_steps=max_steps)
    s = env.reset()
    total = 0.0
    done = False
    while not done:
        a = int(np.argmax(net.predict(s[None])[0]))
        s, r, done = env.step(a)
        total += r
    return int(total)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def block_means(returns, block=25):
    out = []
    for start in range(0, len(returns), block):
        chunk = returns[start:start + block]
        out.append((start + 1, start + len(chunk), float(np.mean(chunk))))
    return out


def demo_learning(returns):
    banner("DEMO 1 --- DQN learns to balance CartPole")
    print("  State : 4 continuous dims (cart pos/vel, pole angle/vel)")
    print("          a Q-table is impossible -- the state never repeats")
    print("  Network: 4 -> 64 -> 64 -> 2   (one Q-value per action)")
    print("  Episodes: 250  gamma=0.99  lr=1e-3  batch=64  "
          "replay=10000  target sync=200 steps")
    print()
    print("  Mean return by 25-episode block (max possible = 500):")
    for lo, hi, m in block_means(returns, 25):
        bar = "#" * int(m / 12)
        print(f"    episodes {lo:3d}-{hi:3d} : {m:6.1f}  {bar}")
    print()
    print(f"  Final mean return (last 50 episodes): "
          f"{np.mean(returns[-50:]):.1f}")


def demo_ablation(with_t):
    banner("DEMO 2 --- Why the target network matters (ablation)")
    print("  The SAME DQN, the only change is whether the bootstrap target")
    print("  comes from a slowly-updated copy of the network or the live one.")
    print()
    # 'with_t' is the run from DEMO 1; only the no-target arm is new.
    _, without_t = dqn(use_target_network=False)
    mw = np.mean(with_t[-50:])
    mo = np.mean(without_t[-50:])
    print(f"  With target network    final mean return: {mw:6.1f}   (stable)")
    print(f"  Without target network final mean return: {mo:6.1f}   "
          f"(chases its own tail)")
    print()
    print(f"  The target network lifts the final return by {mw - mo:.1f}. "
          f"Without it the")
    print("  target moves every gradient step and learning is far less stable.")


def demo_rollout(net):
    banner("DEMO 3 --- The learned greedy policy in action")
    lengths = [greedy_rollout(net, seed=s) for s in (101, 202, 303)]
    for s, L in zip((101, 202, 303), lengths):
        print(f"  Greedy rollout (epsilon=0), seed {s}: "
              f"balanced {L:3d} / 500 steps")
    print()
    print(f"  Mean over 3 greedy rollouts: {np.mean(lengths):.0f} / 500 steps")


def main() -> None:
    net, returns = dqn(seed=RNG_SEED)
    demo_learning(returns)
    demo_ablation(returns)
    demo_rollout(net)


if __name__ == "__main__":
    main()
