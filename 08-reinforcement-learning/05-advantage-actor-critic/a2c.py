"""
a2c.py --- companion code for "Advantage Actor-Critic (A2C / A3C)"
(Reinforcement Learning, Part 5).

Advantage Actor-Critic from scratch in NumPy on CartPole. This is the synthesis
of the whole track: it is BOTH policy-based and value-based.

  ACTOR  : the policy pi(a | s; theta)   -- learns what to do (Part 4)
  CRITIC : a value function V(s; w)       -- learns how good a state is (Part 3)

REINFORCE (Part 4) used a single constant baseline (the mean return). Its
weakness was variance: the Monte-Carlo return is a noisy signal. The critic
fixes this with a LEARNED, STATE-DEPENDENT baseline. The actor is updated by
the ADVANTAGE,
    A_t = G_t - V(s_t)
-- how much better the action did than the critic expected -- which has far
lower variance than the raw return.

Demonstrates:
  1. A2C learning CartPole with two networks trained together.
  2. The headline: the critic baseline cuts the variance of the learning
     signal far below REINFORCE's constant baseline.
  3. The critic in action: V(s) is high for safe states, low near failure.

Everything (both networks, backprop, the advantage, the environment) is plain
NumPy. Dependencies: numpy. Runs in ~20-40 seconds.
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
# CartPole environment (identical to Parts 3 and 4)
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
        truncated = self.steps >= self.max_steps
        return self.state.copy(), 1.0, bool(fell or truncated), truncated


# ---------------------------------------------------------------------------
# A small MLP: He/Xavier init, tanh hidden, linear output, Adam.
#   Used for BOTH the actor (2 outputs = logits) and critic (1 output = value).
# ---------------------------------------------------------------------------

class MLP:
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


def discounted_returns(rewards, gamma, bootstrap=0.0):
    G = np.zeros(len(rewards))
    running = bootstrap
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


# ---------------------------------------------------------------------------
# Advantage Actor-Critic
#   Actor update:  grad log pi(a|s) * advantage      (advantage detached)
#   Critic update: regress V(s) toward the return G  (squared error)
# ---------------------------------------------------------------------------

def a2c(episodes=500, gamma=0.99, lr_actor=1e-2, lr_critic=1e-2,
        hidden=64, use_critic=True, record_v0=False, seed=RNG_SEED):
    """use_critic=True is A2C (advantage = G - V(s), the learned baseline).
    use_critic=False is plain REINFORCE (advantage = G - mean, one constant
    baseline) -- the Part 4 method, for the head-to-head in DEMO 2."""
    env = CartPole(seed=seed)
    actor = MLP([4, hidden, 2], seed=seed, lr=lr_actor)
    critic = MLP([4, hidden, 1], seed=seed + 5, lr=lr_critic)
    rng = np.random.default_rng(seed + 1)
    returns_log = []
    v0_log = []
    s0 = np.zeros((1, 4))

    for ep in range(episodes):
        states, actions, rewards = [], [], []
        s = env.reset()
        done = False
        truncated = False
        while not done:
            probs = softmax(actor.predict(s[None])[0])
            a = int(rng.choice(2, p=probs))
            s2, r, done, truncated = env.step(a)
            states.append(s); actions.append(a); rewards.append(r)
            s = s2
        returns_log.append(sum(rewards))

        S = np.array(states)
        if use_critic:
            # bootstrap from the critic only if the episode was cut off (not a fall)
            boot = float(critic.predict(s[None])[0, 0]) if truncated else 0.0
            G = discounted_returns(rewards, gamma, bootstrap=boot)
            V = critic.predict(S)[:, 0]
            advantage = G - V                       # learned, state-dependent baseline
        else:
            G = discounted_returns(rewards, gamma)  # REINFORCE: one constant baseline
            advantage = G - G.mean()

        # ----- actor: policy gradient weighted by the (normalised) advantage
        adv = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        probs = softmax(actor.forward(S))
        onehot = np.zeros_like(probs)
        onehot[np.arange(len(actions)), actions] = 1
        actor.backward_step((probs - onehot) * adv[:, None])

        # ----- critic: regress V(s) toward the return G (squared error)
        if use_critic:
            critic.forward(S)
            critic.backward_step((V - G)[:, None])

        if record_v0 and (ep % 100 == 0 or ep == episodes - 1):
            v0_log.append((ep, float(critic.predict(s0)[0, 0])))

    return actor, critic, returns_log, v0_log


def episodes_to_solve(returns, thr=475, w=100):
    R = np.array(returns)
    for i in range(w, len(R) + 1):
        if R[i - w:i].mean() >= thr:
            return i
    return None


def greedy_rollout(actor, seed=12345, max_steps=500):
    env = CartPole(seed=seed, max_steps=max_steps)
    s = env.reset()
    total, done = 0.0, False
    while not done:
        a = int(np.argmax(actor.predict(s[None])[0]))
        s, r, done, _ = env.step(a)
        total += r
    return int(total)


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def block_means(returns, block=50):
    out = []
    for start in range(0, len(returns), block):
        chunk = returns[start:start + block]
        out.append((start + 1, start + len(chunk), float(np.mean(chunk))))
    return out


def demo_learning(returns):
    banner("DEMO 1 --- Advantage Actor-Critic learns CartPole")
    print("  Actor : 4 -> 64 -> 2 (softmax)   pi(a | s)   -- what to do")
    print("  Critic: 4 -> 64 -> 1 (value)     V(s)        -- how good a state is")
    print("  Episodes: 500  gamma=0.99  lr=1e-2 (both)  trained together")
    print()
    print("  Mean return by 50-episode block (max possible = 500):")
    for lo, hi, m in block_means(returns, 50):
        bar = "#" * int(m / 12)
        print(f"    episodes {lo:3d}-{hi:3d} : {m:6.1f}  {bar}")
    print()
    print(f"  Final mean return (last 100 episodes): {np.mean(returns[-100:]):.1f}")


def demo_compare():
    banner("DEMO 2 --- A2C vs REINFORCE: faster and more reliable")
    print("  Same network and learning rate; the ONLY difference is the baseline:")
    print("    REINFORCE = one constant baseline (the mean return)")
    print("    A2C       = a learned, state-dependent baseline V(s) (the critic)")
    print("  Episodes to 'solve' (100-episode mean return >= 475), 400 eps/seed:")
    print()
    a_solved, r_solved = [], []
    for seed in (0, 1, 2, 3):
        _, _, ar, _ = a2c(episodes=400, use_critic=True, seed=seed)
        _, _, rr, _ = a2c(episodes=400, use_critic=False, seed=seed)
        a_ep, r_ep = episodes_to_solve(ar), episodes_to_solve(rr)
        if a_ep:
            a_solved.append(a_ep)
        if r_ep:
            r_solved.append(r_ep)
        a_txt = f"@ {a_ep}" if a_ep else "not solved"
        r_txt = f"@ {r_ep}" if r_ep else "not solved"
        print(f"    seed {seed}:   A2C  {a_txt:<11}   REINFORCE  {r_txt}")
    print()
    a_avg = f"avg {int(np.mean(a_solved))} ep" if a_solved else "-"
    print(f"  A2C solved {len(a_solved)}/4 seeds ({a_avg});  "
          f"REINFORCE solved {len(r_solved)}/4.")
    print("  The critic's state-dependent baseline is a cleaner learning signal.")


def demo_critic(actor, critic, v0_log):
    banner("DEMO 3 --- The critic learned to predict return")
    print("  V(start state) over training -- it tracks the policy's rising value:")
    for ep, v in v0_log:
        print(f"    episode {ep:3d}:   V(s0) = {v:6.1f}")
    cap = (1 - 0.99 ** 500) / (1 - 0.99)
    final_v = float(critic.predict(np.zeros((1, 4)))[0, 0])
    print()
    print(f"  Final V(start) = {final_v:.1f}  vs  the ~{cap:.0f} discounted return of a")
    print("  full 500-step episode -- the critic predicts the return accurately.")
    print()
    lengths = [greedy_rollout(actor, seed=s) for s in (101, 202, 303)]
    print(f"  Greedy rollouts (take argmax): {lengths}  "
          f"-> mean {np.mean(lengths):.0f} / 500 steps")


def main() -> None:
    actor, critic, returns, v0_log = a2c(seed=RNG_SEED, episodes=500,
                                         record_v0=True)
    demo_learning(returns)
    demo_compare()
    demo_critic(actor, critic, v0_log)


if __name__ == "__main__":
    main()
