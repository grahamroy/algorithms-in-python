"""
trpo.py --- companion code for "Trust Region Policy Optimisation (TRPO)"
(Advanced Reinforcement Learning, Part 2).

TRPO from scratch in NumPy on CartPole. PPO (Part 1) APPROXIMATED a trust
region with a clip. TRPO enforces one EXACTLY: at each update it maximises the
surrogate objective subject to a hard constraint that the policy doesn't move
more than delta (in average KL divergence) from the policy that collected the
data. Inside that trust region the improvement estimate is reliable.

The machinery, all built here by hand:
  1. Surrogate gradient g  =  mean( A * grad log pi(a|s) ).
  2. The NATURAL gradient direction  x = F^-1 g, where F is the Fisher
     information matrix (the local curvature of KL). We never form F: we solve
     F x = g with CONJUGATE GRADIENT, which only needs Fisher-vector products.
  3. Fisher-vector products F v computed via a finite-difference Jacobian-vector
     product through the policy network -- no autograd.
  4. A backtracking LINE SEARCH that scales the step until KL <= delta and the
     surrogate actually improves.

Demonstrates:
  1. TRPO learning CartPole with exact trust-region steps.
  2. The headline: the per-update KL stays under delta -- the trust region is
     genuinely enforced (PPO only approximates this).
  3. The line search at work: backtracking keeps every accepted step in region.

Everything (networks, conjugate gradient, Fisher-vector products, the line
search, the environment) is plain NumPy. Dependencies: numpy. Runs in ~20-40s.
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
# CartPole environment (identical to Parts 3-5 and PPO)
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
# MLP with flat-parameter access (needed for conjugate gradient + line search).
# tanh hidden, linear output. The actor outputs logits; the critic, a value.
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

    def predict(self, X):
        h = X
        for i in range(len(self.W)):
            zz = h @ self.W[i] + self.b[i]
            h = np.tanh(zz) if i < len(self.W) - 1 else zz
        return h

    def flat_grad(self, dout):
        """Backprop using the cached forward(); return the flat gradient
        (concatenated, weights then biases). Does NOT update the params."""
        n = dout.shape[0]
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = dout
        for i in reversed(range(len(self.W))):
            gW[i] = (self.a[i].T @ delta) / n
            gb[i] = delta.sum(axis=0) / n
            if i > 0:
                delta = (delta @ self.W[i].T) * (1 - self.a[i] ** 2)
        return np.concatenate([g.ravel() for g in gW]
                              + [g.ravel() for g in gb])

    def get_flat_params(self):
        return np.concatenate([w.ravel() for w in self.W]
                              + [b.ravel() for b in self.b])

    def set_flat_params(self, flat):
        i = 0
        for w in self.W:
            n = w.size
            w[:] = flat[i:i + n].reshape(w.shape)
            i += n
        for b in self.b:
            n = b.size
            b[:] = flat[i:i + n].reshape(b.shape)
            i += n

    def adam_step(self, dout):
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


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def mean_kl(p_old, p_new):
    return float(np.mean(np.sum(p_old * (np.log(p_old + 1e-10)
                                         - np.log(p_new + 1e-10)), axis=1)))


# ---------------------------------------------------------------------------
# Conjugate gradient: solve A x = b given a function that computes A v.
# ---------------------------------------------------------------------------

def conjugate_gradient(matvec, b, iters=10, tol=1e-10):
    x = np.zeros_like(b)
    r = b.copy()
    p = b.copy()
    rr = r @ r
    for _ in range(iters):
        Ap = matvec(p)
        alpha = rr / (p @ Ap + 1e-10)
        x += alpha * p
        r -= alpha * Ap
        rr_new = r @ r
        if rr_new < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


# ---------------------------------------------------------------------------
# TRPO
# ---------------------------------------------------------------------------

def trpo(iterations=50, rollout_steps=2048, gamma=0.99, lam=0.95,
         max_kl=0.01, cg_iters=10, cg_damping=0.1, hidden=64,
         seed=RNG_SEED):
    env = CartPole(seed=seed)
    actor = MLP([4, hidden, 2], seed=seed)
    critic = MLP([4, hidden, 1], seed=seed + 5, lr=1e-2)
    rng = np.random.default_rng(seed + 1)

    s = env.reset()
    ep_ret, ep_returns = 0.0, []
    hist, kl_hist = [], []
    env_steps = 0

    for it in range(iterations):
        # ---------- collect a rollout with the current policy ----------
        S = np.zeros((rollout_steps, 4))
        A_idx = np.zeros(rollout_steps, dtype=np.int64)
        R = np.zeros(rollout_steps)
        DONE = np.zeros(rollout_steps)
        V = np.zeros(rollout_steps)
        for t in range(rollout_steps):
            probs = softmax(actor.predict(s[None])[0])
            a = int(rng.choice(2, p=probs))
            S[t] = s
            A_idx[t] = a
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

        # ---------- policy gradient g of the surrogate at theta_old ----------
        theta_old = actor.get_flat_params()
        logits_old = actor.forward(S)            # caches activations at theta_old
        p_old = softmax(logits_old)
        onehot = np.zeros_like(p_old)
        onehot[np.arange(rollout_steps), A_idx] = 1
        # grad of mean(ratio*A); at theta_old ratio=1 so dout = A*(onehot - p)
        g = actor.flat_grad(adv[:, None] * (onehot - p_old))

        # ---------- Fisher-vector product: F v = J^T (diag(p) - p p^T) J v ----------
        def fisher_vector_product(v):
            eps_fd = 1e-5
            actor.set_flat_params(theta_old + eps_fd * v)
            logits_pert = actor.predict(S)
            actor.set_flat_params(theta_old)
            Jv = (logits_pert - logits_old) / eps_fd          # (N, 2)
            pJv = (p_old * Jv).sum(axis=1, keepdims=True)
            u = p_old * Jv - p_old * pJv                       # (diag(p)-pp^T) Jv
            Hv = actor.flat_grad(u)                            # J^T u, averaged
            return Hv + cg_damping * v

        if g @ g < 1e-12:
            hist.append((env_steps, np.mean(ep_returns[-20:]) if ep_returns else 0.0))
            continue

        # ---------- natural gradient via conjugate gradient: solve F x = g ----------
        step_dir = conjugate_gradient(fisher_vector_product, g, iters=cg_iters)
        shs = 0.5 * step_dir @ fisher_vector_product(step_dir)  # 0.5 x^T F x
        beta = np.sqrt(max_kl / (shs + 1e-10))
        full_step = beta * step_dir
        surr_old = float(np.mean(adv))           # ratio=1 -> surrogate = mean(A)

        # ---------- backtracking line search ----------
        # The beta-scaled step targets KL = delta under a quadratic approximation;
        # the line search backs off (halving) until the ACTUAL KL is within delta
        # and the surrogate truly improves -- rejecting the update if neither holds.
        accepted_kl = 0.0
        success = False
        old_logp = np.log(p_old[np.arange(rollout_steps), A_idx] + 1e-10)
        for k in range(10):
            frac = 0.5 ** k
            actor.set_flat_params(theta_old + frac * full_step)
            p_new = softmax(actor.predict(S))
            logp_new = np.log(p_new[np.arange(rollout_steps), A_idx] + 1e-10)
            ratio = np.exp(logp_new - old_logp)
            surr_new = float(np.mean(ratio * adv))
            kl = mean_kl(p_old, p_new)
            if kl <= max_kl and surr_new > surr_old:
                accepted_kl = kl
                success = True
                break
        if not success:
            actor.set_flat_params(theta_old)          # reject: stay put

        # ---------- fit the critic (value regression toward returns) ----------
        idx = np.arange(rollout_steps)
        for _ in range(5):
            rng.shuffle(idx)
            for start in range(0, rollout_steps, 64):
                mb = idx[start:start + 64]
                vpred = critic.forward(S[mb])[:, 0]
                critic.adam_step((vpred - returns[mb])[:, None])

        recent = np.mean(ep_returns[-20:]) if ep_returns else 0.0
        hist.append((env_steps, recent))
        kl_hist.append(accepted_kl)

    return actor, critic, hist, kl_hist


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
    banner("DEMO 1 --- TRPO learns CartPole (exact trust-region steps)")
    print("  Actor : 4 -> 64 -> 2 (softmax)     Critic: 4 -> 64 -> 1 (value)")
    print("  rollout=2048 steps  max_kl(delta)=0.01  cg_iters=10  "
          "gamma=0.99  lambda=0.95")
    print()
    print("  Moving-average return vs environment steps (max possible = 500):")
    for env_steps, recent in hist[::3]:
        bar = "#" * int(recent / 12)
        print(f"    {env_steps:7d} steps : {recent:6.1f}  {bar}")
    print()
    print(f"  Final moving-average return: {hist[-1][1]:.1f}")


def demo_trust_region(kl_hist):
    banner("DEMO 2 --- The trust region is enforced exactly")
    print("  TRPO constrains every update to mean KL(old, new) <= delta = 0.01.")
    print("  The KL of each accepted step, over training:")
    print()
    arr = np.array(kl_hist)
    for i in range(0, len(arr), 5):
        chunk = arr[i:i + 5]
        print(f"    iters {i+1:2d}-{i+len(chunk):2d}:  "
              + "  ".join(f"{k:.4f}" for k in chunk))
    print()
    print(f"  Max KL over all updates: {arr.max():.4f}  (delta = 0.0100)")
    print(f"  -> every step stayed inside the trust region. PPO only "
          f"approximates this bound; TRPO guarantees it.")


def demo_delta_sweep():
    banner("DEMO 3 --- The trust-region size delta is a speed/stability dial")
    print("  delta caps how far the policy may move each update. Too small is")
    print("  slow; too large and the local approximation breaks down.")
    print()
    print("  delta    env steps to solve     final return    max KL")
    for d in (0.001, 0.01, 0.05, 0.2):
        _, _, h, k = trpo(iterations=40, max_kl=d, seed=0)
        s = steps_to_solve(h)
        stxt = f"{s}" if s else "not solved"
        print(f"    {d:<7} {stxt:<21} {h[-1][1]:6.0f}        {np.max(k):.4f}")
    print()
    print("  delta=0.01 is the sweet spot -- fast and stable. Note max KL tracks")
    print("  delta: the trust region is enforced at whatever level you choose.")


def main() -> None:
    actor, critic, hist, kl_hist = trpo(seed=RNG_SEED)
    demo_learning(hist)
    demo_trust_region(kl_hist)
    demo_delta_sweep()
    lengths = [greedy_rollout(actor, seed=s) for s in (101, 202, 303)]
    print(f"\n  Trained TRPO greedy rollouts: {lengths}  "
          f"-> mean {np.mean(lengths):.0f} / 500 steps")


if __name__ == "__main__":
    main()
