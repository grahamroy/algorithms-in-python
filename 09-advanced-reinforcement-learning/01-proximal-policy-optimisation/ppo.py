"""
ppo.py --- companion code for "Proximal Policy Optimisation (PPO)"
(Advanced Reinforcement Learning, Part 1).

PPO from scratch in NumPy on CartPole. It builds directly on the Advantage
Actor-Critic of Part 5 (an actor pi(a|s) and a critic V(s)) and adds the one
idea that made policy gradients reliable: the CLIPPED surrogate objective.

The problem: a plain policy-gradient step can move the policy too far and wreck
it -- and because each on-policy rollout is used only once, learning is
sample-hungry. PPO fixes both. It defines the probability ratio

    r(theta) = pi_new(a|s) / pi_old(a|s)

and optimises a clipped objective

    L = min( r * A,  clip(r, 1-eps, 1+eps) * A )

The clip removes any incentive to push r outside [1-eps, 1+eps], so the new
policy stays close ("proximal") to the old one. That safety means we can take
SEVERAL epochs of gradient steps on the SAME rollout -- reusing data without
the policy diverging.

Demonstrates:
  1. PPO learning CartPole from rollouts with multi-epoch updates.
  2. The headline: clip vs no-clip. Without the clip, multi-epoch reuse pushes
     the policy too far and collapses it; the clip is what makes reuse safe.
  3. Sample efficiency: more epochs per rollout -> solved in fewer env steps.

Everything (both networks, GAE, the clipped objective, backprop, the
environment) is plain NumPy. Dependencies: numpy. Runs in ~20-40 seconds.
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
# CartPole environment (identical to Parts 3-5)
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
# A small MLP (tanh hidden, linear output, Adam) -- actor and critic.
# ---------------------------------------------------------------------------

class MLP:
    def __init__(self, sizes, seed=0, lr=3e-3):
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
                delta = (delta @ self.W[i].T) * (1 - self.a[i] ** 2)
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


# ---------------------------------------------------------------------------
# PPO
#   Collect a rollout with the current ("old") policy, compute GAE advantages,
#   then take several epochs of minibatch updates on the CLIPPED objective.
# ---------------------------------------------------------------------------

def ppo(iterations=40, rollout_steps=1024, epochs=10, minibatch=64,
        gamma=0.99, lam=0.95, clip=0.2, lr=3e-3, hidden=64,
        use_clip=True, seed=RNG_SEED):
    env = CartPole(seed=seed)
    actor = MLP([4, hidden, 2], seed=seed, lr=lr)
    critic = MLP([4, hidden, 1], seed=seed + 5, lr=lr)
    rng = np.random.default_rng(seed + 1)

    s = env.reset()
    ep_ret = 0.0
    ep_returns = []          # every completed episode's return
    hist = []                # (env_steps, moving-average return) per iteration
    env_steps = 0

    for it in range(iterations):
        # ---------- collect a rollout with the frozen "old" policy ----------
        S = np.zeros((rollout_steps, 4))
        A = np.zeros(rollout_steps, dtype=np.int64)
        LOGP = np.zeros(rollout_steps)
        R = np.zeros(rollout_steps)
        DONE = np.zeros(rollout_steps)
        V = np.zeros(rollout_steps)
        for t in range(rollout_steps):
            probs = softmax(actor.predict(s[None])[0])
            a = int(rng.choice(2, p=probs))
            S[t] = s
            A[t] = a
            LOGP[t] = np.log(probs[a] + 1e-10)
            V[t] = critic.predict(s[None])[0, 0]
            s, r, done = env.step(a)
            R[t] = r
            DONE[t] = float(done)
            ep_ret += r
            env_steps += 1
            if done:
                ep_returns.append(ep_ret)
                ep_ret = 0.0
                s = env.reset()
        last_v = critic.predict(s[None])[0, 0]

        # ---------- GAE advantages and value targets ----------
        adv = np.zeros(rollout_steps)
        last_gae = 0.0
        for t in reversed(range(rollout_steps)):
            next_v = last_v if t == rollout_steps - 1 else V[t + 1]
            nonterminal = 1.0 - DONE[t]
            delta = R[t] + gamma * next_v * nonterminal - V[t]
            adv[t] = last_gae = delta + gamma * lam * nonterminal * last_gae
        returns = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---------- several epochs of minibatch updates ----------
        idx = np.arange(rollout_steps)
        for _ in range(epochs):
            rng.shuffle(idx)
            for start in range(0, rollout_steps, minibatch):
                mb = idx[start:start + minibatch]
                probs = softmax(actor.forward(S[mb]))
                logp_new = np.log(probs[np.arange(len(mb)), A[mb]] + 1e-10)
                ratio = np.exp(logp_new - LOGP[mb])
                a_mb = adv[mb]
                surr1 = ratio * a_mb
                if use_clip:
                    surr2 = np.clip(ratio, 1 - clip, 1 + clip) * a_mb
                    active = (surr1 <= surr2).astype(float)   # gradient flows only here
                    g = a_mb * ratio * active
                else:
                    g = a_mb * ratio                          # no clip: full ratio gradient
                onehot = np.zeros_like(probs)
                onehot[np.arange(len(mb)), A[mb]] = 1
                actor.backward_step((probs - onehot) * g[:, None])

                v_pred = critic.forward(S[mb])[:, 0]
                critic.backward_step((v_pred - returns[mb])[:, None])

        recent = np.mean(ep_returns[-20:]) if ep_returns else 0.0
        hist.append((env_steps, recent))

    return actor, critic, hist


def greedy_rollout(actor, seed=12345, max_steps=500):
    env = CartPole(seed=seed, max_steps=max_steps)
    s = env.reset()
    total, done = 0.0, False
    while not done:
        a = int(np.argmax(actor.predict(s[None])[0]))
        s, r, done = env.step(a)
        total += r
    return int(total)


def steps_to_solve(hist, thr=475):
    for env_steps, recent in hist:
        if recent >= thr:
            return env_steps
    return None


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------

def demo_learning(hist):
    banner("DEMO 1 --- PPO learns CartPole (clipped, multi-epoch)")
    print("  Actor : 4 -> 64 -> 2 (softmax)     Critic: 4 -> 64 -> 1 (value)")
    print("  rollout=1024 steps  epochs=10  minibatch=64  clip=0.2  "
          "gamma=0.99  lambda=0.95")
    print()
    print("  Moving-average return vs environment steps (max possible = 500):")
    for env_steps, recent in hist[::4]:
        bar = "#" * int(recent / 12)
        print(f"    {env_steps:6d} steps : {recent:6.1f}  {bar}")
    print()
    print(f"  Final moving-average return: {hist[-1][1]:.1f}")


def demo_clip_ablation():
    banner("DEMO 2 --- The clip is what makes multi-epoch reuse safe")
    print("  Same PPO, 10 epochs per rollout. The ONLY difference is whether the")
    print("  surrogate objective is clipped to keep the new policy near the old.")
    print()
    _, _, hist_clip = ppo(use_clip=True, seed=0)
    _, _, hist_noclip = ppo(use_clip=False, seed=0)
    clip_final = hist_clip[-1][1]
    nc_peak = max(r for _, r in hist_noclip)
    nc_final = hist_noclip[-1][1]
    print(f"  With clip   : rises to {clip_final:.0f} and holds it           (stable)")
    print(f"  Without clip: peaks at {nc_peak:.0f}, then collapses to {nc_final:.0f}  (unstable)")
    print()
    print("  Without the clip, 10 epochs of reuse drive the policy far past where")
    print("  the rollout supports -- it overshoots and crashes. The clip holds each")
    print("  update 'proximal' to the old policy, which is what makes reuse safe.")


def demo_sample_efficiency():
    banner("DEMO 3 --- More epochs per rollout = fewer env steps to solve")
    print("  PPO can safely reuse each rollout for several epochs (the clip keeps")
    print("  it stable). More reuse means less environment interaction to solve.")
    print()
    for ep in (1, 5, 10):
        _, _, hist = ppo(epochs=ep, seed=0)
        solved = steps_to_solve(hist)
        txt = f"solved in {solved} env steps" if solved else "not solved in budget"
        print(f"    {ep:2d} epoch(s)/rollout : {txt}")
    print()
    print("  One epoch/rollout is the on-policy A2C regime; reusing the data the")
    print("  clip makes safe to reuse is what buys PPO its sample efficiency.")


def main() -> None:
    actor, critic, hist = ppo(seed=RNG_SEED)
    demo_learning(hist)
    demo_clip_ablation()
    demo_sample_efficiency()
    lengths = [greedy_rollout(actor, seed=s) for s in (101, 202, 303)]
    print(f"\n  Trained PPO greedy rollouts: {lengths}  "
          f"-> mean {np.mean(lengths):.0f} / 500 steps")


if __name__ == "__main__":
    main()
