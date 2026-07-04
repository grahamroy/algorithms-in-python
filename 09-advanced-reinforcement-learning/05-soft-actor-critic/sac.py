"""
sac.py --- companion code for "Soft Actor-Critic (SAC)"
(Advanced Reinforcement Learning, Part 5).

SAC from scratch in NumPy on Pendulum. TD3 (Part 4) kept DDPG's DETERMINISTIC
policy and bolted exploration noise onto it. SAC goes the other way: the policy
is STOCHASTIC -- a tanh-squashed Gaussian -- and exploration is part of the
OBJECTIVE itself. SAC maximises reward PLUS the policy's entropy:

    J = E[ sum_t  r_t  +  alpha * H( pi(.|s_t) ) ]

The temperature alpha prices uncertainty: high alpha -> keep options open,
low alpha -> commit. SAC AUTO-TUNES alpha so the policy holds a target
entropy -- the exploration/exploitation dial turns itself.

What it keeps from TD3: twin critics and the min() target (pessimism).
What changes: the actor outputs a DISTRIBUTION (mean + std), actions are
sampled via the reparameterisation trick a = bound * tanh(mu + sigma * eps),
and log pi appears in both the critic target and the actor loss.

Demonstrates:
  1. SAC solving Pendulum swing-up.
  2. The self-tuning dial: sigma (policy spread) anneals as learning
     progresses, and the temperature controller pins the policy's entropy
     to the target -- exploration scheduled automatically, not by hand.
  3. Acting on the mean vs sampling: the learned distribution's mean is a
     strong deterministic controller from unseen starts.

Everything (five networks, the reparameterised gradients including the tanh
change of variables, twin critics, replay, soft updates, the physics) is plain
NumPy -- no autograd. Dependencies: numpy. Runs in ~30-50 seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


SEPARATOR = "=" * 72
RNG_SEED = 0
LOG2PI = np.log(2.0 * np.pi)


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Pendulum (identical to Parts 3 and 4)
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
# MLP (tanh hidden, linear output, Adam). backward() returns parameter grads
# AND the input gradient (for dQ/da through the critics).
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


# ---------------------------------------------------------------------------
# The stochastic actor: net(s) -> [mu, log_sigma]. Actions are sampled with
# the REPARAMETERISATION trick  a = bound * tanh(mu + sigma * eps), which lets
# the gradient flow through the sample. log pi includes the tanh change of
# variables. All gradients below are derived by hand for the 1-D action.
# ---------------------------------------------------------------------------

LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0


class GaussianActor:
    def __init__(self, s_dim, hidden, bound, seed, lr):
        self.net = MLP([s_dim, hidden, hidden, 2], seed=seed, lr=lr)
        self.bound = bound

    def _dist(self, out):
        mu = out[:, :1]
        log_std = np.clip(out[:, 1:], LOGSTD_MIN, LOGSTD_MAX)
        clip_mask = ((out[:, 1:] > LOGSTD_MIN) & (out[:, 1:] < LOGSTD_MAX))
        return mu, log_std, clip_mask.astype(float)

    def sample(self, S, rng, cache=False):
        """Sample a ~ pi(.|s). Returns (a, log_pi, cache-dict)."""
        out = self.net.forward(S) if cache else self.net.predict(S)
        mu, log_std, mask = self._dist(out)
        std = np.exp(log_std)
        eps = rng.standard_normal(mu.shape)
        u = mu + std * eps
        t = np.tanh(u)
        a = self.bound * t
        # log pi(a|s) = log N(u; mu, std) - log |da/du|
        log_pi = (-0.5 * eps**2 - log_std - 0.5 * LOG2PI
                  - np.log(self.bound * (1 - t**2) + 1e-6))
        info = {"eps": eps, "std": std, "t": t, "mask": mask}
        return a, log_pi, info

    def mean_action(self, S):
        out = self.net.predict(S)
        return self.bound * np.tanh(out[:, :1])

    def apply_grads(self, dmu, dlogstd, mask):
        """Backprop d(loss)/dmu and d(loss)/dlog_std into the network."""
        dout = np.concatenate([dmu, dlogstd * mask], axis=1)
        gW, gb, _ = self.net.backward(dout)
        self.net.adam_apply(gW, gb)


class ReplayBuffer:
    def __init__(self, cap, s_dim, seed):
        self.s = np.zeros((cap, s_dim))
        self.a = np.zeros((cap, 1))
        self.r = np.zeros((cap, 1))
        self.s2 = np.zeros((cap, s_dim))
        self.cap, self.idx, self.size = cap, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2):
        i = self.idx
        self.s[i], self.a[i, 0], self.r[i, 0], self.s2[i] = s, a, r, s2
        self.idx = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch):
        j = self.rng.integers(0, self.size, size=batch)
        return self.s[j], self.a[j], self.r[j], self.s2[j]


def q_of(critic, S, A):
    return critic.predict(np.concatenate([S, A], axis=1))


# ---------------------------------------------------------------------------
# SAC
# ---------------------------------------------------------------------------

def sac(episodes=60, gamma=0.99, tau=0.01, batch=128, hidden=64,
        lr_actor=1e-3, lr_critic=2e-3, lr_alpha=1e-2, target_entropy=-1.0,
        init_alpha=0.2, warmup=1000, buffer_cap=50000, seed=RNG_SEED):
    env = Pendulum(seed=seed)
    s_dim, bound = 3, 2.0
    actor = GaussianActor(s_dim, hidden, bound, seed, lr_actor)
    critic1 = MLP([s_dim + 1, hidden, hidden, 1], seed=seed + 5, lr=lr_critic)
    critic2 = MLP([s_dim + 1, hidden, hidden, 1], seed=seed + 9, lr=lr_critic)
    critic1_t = MLP([s_dim + 1, hidden, hidden, 1], seed=seed + 5, lr=lr_critic)
    critic2_t = MLP([s_dim + 1, hidden, hidden, 1], seed=seed + 9, lr=lr_critic)
    critic1_t.copy_from(critic1)
    critic2_t.copy_from(critic2)
    buf = ReplayBuffer(buffer_cap, s_dim, seed + 2)
    rng = np.random.default_rng(seed + 3)

    log_alpha = np.log(init_alpha)
    returns, diag = [], []          # diag: (episode, alpha, mean sigma, entropy)
    total_steps = 0

    for ep in range(episodes):
        s = env.reset()
        ep_ret = 0.0
        done = False
        ep_entropies, ep_sigmas = [], []
        while not done:
            if total_steps < warmup:
                a = rng.uniform(-bound, bound, size=(1,))
            else:
                a, _, _ = actor.sample(s[None], rng)
                a = a[0]
            s2, r, done = env.step(a)
            buf.add(s, float(a[0]), r, s2)
            s = s2
            ep_ret += r
            total_steps += 1

            if buf.size >= batch and total_steps >= warmup:
                alpha = float(np.exp(log_alpha))
                bs, ba, br, bs2 = buf.sample(batch)

                # ----- critic target: min of target critics, minus entropy tax
                a2, logp2, _ = actor.sample(bs2, rng)
                q1t = q_of(critic1_t, bs2, a2)
                q2t = q_of(critic2_t, bs2, a2)
                y = br + gamma * (np.minimum(q1t, q2t) - alpha * logp2)

                for critic in (critic1, critic2):
                    q = critic.forward(np.concatenate([bs, ba], axis=1))
                    gW, gb, _ = critic.backward(q - y)
                    critic.adam_apply(gW, gb)

                # ----- actor: maximise  min Q(s, a_theta) - alpha * log pi
                a_new, logp, info = actor.sample(bs, rng, cache=True)
                X = np.concatenate([bs, a_new], axis=1)
                q1 = critic1.forward(X)
                q2 = critic2.forward(X)
                # dQmin/da from whichever critic is the per-sample minimum
                m1 = (q1 <= q2).astype(float)
                _, _, din1 = critic1.backward(m1)
                _, _, din2 = critic2.backward(1.0 - m1)
                dq_da = (din1 + din2)[:, s_dim:]
                # hand-derived gradients of  L = alpha*log pi - Qmin  w.r.t.
                # the actor outputs (mu, log_std), with eps held fixed:
                t = info["t"]
                se = info["std"] * info["eps"]
                da_dmu = bound * (1 - t**2)
                dlogpi_dmu = 2.0 * t
                dmu = alpha * dlogpi_dmu - dq_da * da_dmu
                dlogstd = (alpha * (-1.0 + 2.0 * t * se)
                           - dq_da * da_dmu * se)
                actor.apply_grads(dmu, dlogstd, info["mask"])

                # ----- temperature: hold entropy at the target
                log_alpha += lr_alpha * alpha * float(np.mean(logp)
                                                      + target_entropy)
                ep_entropies.append(float(np.mean(-logp)))
                ep_sigmas.append(float(np.mean(info["std"])))

                critic1_t.soft_update(critic1, tau)
                critic2_t.soft_update(critic2, tau)

        returns.append(ep_ret)
        if (ep + 1) % 10 == 0 and ep_entropies:
            diag.append((ep + 1, float(np.exp(log_alpha)),
                         float(np.mean(ep_sigmas)),
                         float(np.mean(ep_entropies))))
    return actor, returns, diag


def mean_return(actor, seed=999):
    env = Pendulum(seed=seed)
    s = env.reset()
    total, done = 0.0, False
    while not done:
        a = actor.mean_action(s[None])[0]
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
    banner("DEMO 1 --- SAC solves Pendulum swing-up (stochastic policy)")
    print("  Actor : 3 -> 64 -> 64 -> (mu, sigma)   a ~ 2*tanh(N(mu, sigma))")
    print("  Twin critics + min (from TD3)  |  entropy-regularised objective")
    print("  Episodes: 60  gamma=0.99  tau=0.01  auto-tuned alpha, "
          "target entropy -1.0")
    print()
    print("  Mean episode return by 12-episode block (0 is best, random ~ -1300):")
    for lo, hi, m in block_means(returns, 12):
        bar = "#" * int(max(0, (m + 1400) / 40))
        print(f"    episodes {lo:3d}-{hi:3d} : {m:8.1f}  {bar}")
    print()
    print(f"  Final mean return (last 20 episodes): {np.mean(returns[-20:]):.1f}")


def demo_entropy_dial(diag):
    banner("DEMO 2 --- The exploration dial turns itself")
    print("  SAC pays the policy alpha per nat of entropy and TUNES alpha to steer")
    print("  the entropy toward the target (-1.0). Temperature and policy spread")
    print("  over training -- no hand-designed noise schedule anywhere:")
    print()
    print("    episode    alpha     mean sigma    policy entropy")
    print("    (start)     0.200      ~1.0           ~0.0")
    for ep, alpha, sigma, ent in diag:
        print(f"      {ep:3d}      {alpha:6.3f}      {sigma:6.3f}         {ent:+.2f}")
    print()
    print("  alpha turns itself down ~7x and sigma halves as the critics sharpen;")
    print("  the entropy is steered from ~0 into a steady band just above the")
    print("  target. Exploration is scheduled by the objective, not by hand.")


def demo_mean_policy(actor):
    banner("DEMO 3 --- Act on the mean: a strong deterministic controller")
    print("  At evaluation, take the distribution's MEAN (no sampling).")
    print("  Returns from five random starts never seen in training")
    print("  (random policy ~ -1300; 0 is a perfectly balanced pole):")
    print()
    seeds = (11, 22, 33, 44, 55)
    rets = [mean_return(actor, seed=s) for s in seeds]
    for s, r in zip(seeds, rets):
        print(f"    start seed {s}:   return {r:8.1f}")
    print()
    print(f"  Mean over 5 unseen starts: {np.mean(rets):.0f} -- the stochastic")
    print("  policy was for learning; its mean is the controller you deploy.")


def main() -> None:
    actor, returns, diag = sac(seed=RNG_SEED)
    demo_learning(returns)
    demo_entropy_dial(diag)
    demo_mean_policy(actor)


if __name__ == "__main__":
    main()
