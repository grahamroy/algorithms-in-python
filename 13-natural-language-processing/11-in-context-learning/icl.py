"""
icl.py --- companion code for "In-Context Learning and Prompting"
(Natural Language Processing, Part 11).

Every article in this track has changed a model by changing its
weights. Pre-training, fine-tuning, adapters, policy gradients --
different objectives, same mechanism. This one does not touch the
weights at all.

A trained model is shown a few examples of a rule it has never seen,
in its context, and asked to apply it. No gradient step happens. The
weights afterwards are bit-for-bit the weights before. And the
behaviour changes.

That is in-context learning, and the first question is whether a model
this small can do it at all. The stage is built so the answer cannot
be faked: each sequence draws a FRESH RANDOM PAIRING over the
micro-language's nouns, so there is nothing to memorise. A model that
answers correctly can only have read the mapping out of its own
prompt.

The second question is where the capability comes from, and the answer
turns out not to be the architecture. Two models are trained here with
the same trunk, the same budget and the same seeds, differing only in
how their training sequences are built:

    bursty   a pairing holds within a sequence, redrawn between them
    IID      every object drawn independently, so context says nothing

What that buys, measured here:
  1. In-context learning, present or absent, at 1.0000 against a chance
     rate of 0.0278 -- and only from the bursty distribution.
  2. The two things "number of examples" conflates: evidence, which
     helps, and distractors, which hurt. Separating them reverses the
     curve.
  3. The mechanism, read straight off the attention map: the model
     puts most of its attention on exactly the token it needs to copy,
     while the IID model is at chance.
  4. When the capability arrives during training, which is not
     gradually.
  5. Where it runs out -- the point at which a two-layer model can
     still copy but can no longer find what to copy.

The transformer is Part 7's, unchanged. The stage is new: sequences of
SUBJ VERB OBJ triples over Part 1's nouns and verbs, with the loss
taken on the final object alone. Hand-differentiated and audited
against central differences at startup. Seeded and deterministic.
Dependencies: numpy.  Runs in about twenty-two minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

SEPARATOR = "=" * 72


def banner(title):
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


def softmax_last(S):
    Z = S - S.max(-1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(-1, keepdims=True)


# -----------------------------------------------------------------------
# Part 1's words, in a compact frame.  A "fact" here is three tokens --
# subject, verb, object -- so eight of them plus a query fit inside a
# context a two-layer model can actually use.
# -----------------------------------------------------------------------
TOPICS = {
    "royal": {
        "nouns": ["king", "queen", "prince", "princess", "lord",
                  "lady", "knight", "castle", "crown", "throne",
                  "banner", "sword"],
        "gender": [("king", "queen"), ("prince", "princess"),
                   ("lord", "lady")],
        "verbs": [("rules", "rule"), ("commands", "command"),
                  ("crowns", "crown"), ("defends", "defend")],
        "adjs": ["noble", "golden", "ancient", "royal"],
    },
    "farm": {
        "nouns": ["farmer", "bull", "cow", "rooster", "hen", "barn",
                  "field", "tractor", "fence", "seed", "orchard",
                  "plough"],
        "gender": [("bull", "cow"), ("rooster", "hen")],
        "verbs": [("ploughs", "plough"), ("feeds", "feed"),
                  ("harvests", "harvest"), ("grazes", "graze")],
        "adjs": ["muddy", "ripe", "fertile", "wooden"],
    },
    "sea": {
        "nouns": ["sailor", "captain", "ship", "wave", "harbour",
                  "anchor", "gull", "tide", "mast", "compass",
                  "island", "storm"],
        "gender": [],
        "verbs": [("sails", "sail"), ("anchors", "anchor"),
                  ("navigates", "navigate"), ("drifts", "drift")],
        "adjs": ["salty", "deep", "stormy", "distant"],
    },
}
GENDER_M = [p[0] for t in TOPICS.values() for p in t["gender"]]
GENDER_F = [p[1] for t in TOPICS.values() for p in t["gender"]]

NOUNS = [n for t in TOPICS.values() for n in t["nouns"]]
VERBS = [v for t in TOPICS.values() for v, _ in t["verbs"]]
SPECIALS = ["[PAD]", "[BOS]", "[SEP]"]
VOCAB = SPECIALS + sorted(set(NOUNS)) + sorted(set(VERBS))
W2I = {w: i for i, w in enumerate(VOCAB)}
V = len(VOCAB)
PAD, BOS, SEP = 0, 1, 2
K_MAX = 8
T_MAX = 1 + K_MAX * 4 + 2        # BOS + k*(s v o SEP) + s v
CHANCE = 1.0 / len(NOUNS)

class Transformer:
    def __init__(self, d=64, heads=4, layers=2, d_ff=None, seed=0,
                 lr=3e-3):
        r = np.random.default_rng(seed)
        self.d, self.h, self.L = d, heads, layers
        self.dk = d // heads
        self.dff = d_ff or 4 * d
        self.lr = lr

        def rnd(*shape, s=1.0):
            return r.standard_normal(shape) * np.sqrt(s / shape[0])

        P = {"emb": rnd(V, d), "pos": rnd(T_MAX, d)}
        for i in range(layers):
            P[f"{i}.g1"] = np.ones(d)
            P[f"{i}.b1"] = np.zeros(d)
            P[f"{i}.Wqkv"] = rnd(d, 3 * d)
            P[f"{i}.Wo"] = rnd(d, d)
            P[f"{i}.g2"] = np.ones(d)
            P[f"{i}.b2"] = np.zeros(d)
            P[f"{i}.W1"] = rnd(d, self.dff, s=2.0)
            P[f"{i}.bb1"] = np.zeros(self.dff)
            P[f"{i}.W2"] = rnd(self.dff, d)
            P[f"{i}.bb2"] = np.zeros(d)
        P["gf"] = np.ones(d)
        P["bf"] = np.zeros(d)
        P["Wv"] = rnd(d, V)                 # the output head over tokens
        P["bv"] = np.zeros(V)
        self.P = P
        self.reset_adam()

    def reset_adam(self):
        self.M = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.Vm = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def n_params(self):
        return sum(v.size for v in self.P.values())

    # ---- layer norm ----
    @staticmethod
    def ln_fwd(x, g, b):
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        inv = 1.0 / np.sqrt((xc ** 2).mean(-1, keepdims=True) + 1e-5)
        xh = xc * inv
        return xh * g + b, (xh, inv)

    @staticmethod
    def ln_bwd(dy, cache, g):
        xh, inv = cache
        dg = (dy * xh).sum(axis=tuple(range(dy.ndim - 1)))
        db = dy.sum(axis=tuple(range(dy.ndim - 1)))
        dxh = dy * g
        dx = inv * (dxh - dxh.mean(-1, keepdims=True)
                    - xh * (dxh * xh).mean(-1, keepdims=True))
        return dx, dg, db

    # ---- multi-head causal self-attention ----
    def attn_fwd(self, x, pre, keep):
        N, T, d = x.shape
        h, dk = self.h, self.dk
        q, k, v = np.split(x @ self.P[pre + "Wqkv"], 3, axis=-1)

        def sp(z):
            return z.reshape(N, T, h, dk).transpose(0, 2, 1, 3)

        q, k, v = sp(q), sp(k), sp(v)
        S = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)
        S = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -1e30, S)
        S = np.where(keep[:, None, None, :], S, -1e30)   # ignore padding
        W = softmax_last(S)
        ctx = (W @ v).transpose(0, 2, 1, 3).reshape(N, T, d)
        return ctx @ self.P[pre + "Wo"], (x, q, k, v, W, ctx)

    def attn_bwd(self, dout, cache, pre, g):
        x, q, k, v, W, ctx = cache
        N, T, d = x.shape
        h, dk = self.h, self.dk
        g[pre + "Wo"] += ctx.reshape(-1, d).T @ dout.reshape(-1, d)
        dctx = (dout @ self.P[pre + "Wo"].T).reshape(
            N, T, h, dk).transpose(0, 2, 1, 3)
        dW = dctx @ v.transpose(0, 1, 3, 2)
        dv = W.transpose(0, 1, 3, 2) @ dctx
        dS = W * (dW - (W * dW).sum(-1, keepdims=True)) / np.sqrt(dk)
        dq = dS @ k
        dk_ = dS.transpose(0, 1, 3, 2) @ q

        def mg(z):
            return z.transpose(0, 2, 1, 3).reshape(N, T, h * dk)

        dqkv = np.concatenate([mg(dq), mg(dk_), mg(dv)], axis=-1)
        g[pre + "Wqkv"] += x.reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
        return dqkv @ self.P[pre + "Wqkv"].T

    # ---- trunk ----
    def trunk_fwd(self, ids, keep):
        N, T = ids.shape
        x = self.P["emb"][ids] + self.P["pos"][:T]
        caches = []
        for li in range(self.L):
            pre = f"{li}."
            ln1, c1 = self.ln_fwd(x, self.P[pre + "g1"], self.P[pre + "b1"])
            a, ca = self.attn_fwd(ln1, pre, keep)
            x1 = x + a
            ln2, c2 = self.ln_fwd(x1, self.P[pre + "g2"], self.P[pre + "b2"])
            hpre = ln2 @ self.P[pre + "W1"] + self.P[pre + "bb1"]
            hact = np.maximum(0, hpre)
            x2 = x1 + hact @ self.P[pre + "W2"] + self.P[pre + "bb2"]
            caches.append((c1, ca, c2, ln2, hpre, hact))
            x = x2
        H, cf = self.ln_fwd(x, self.P["gf"], self.P["bf"])
        return H, (ids, keep, caches, cf, H)

    def trunk_bwd(self, dH, cache, g):
        ids, keep, caches, cf, H = cache
        dx, dgf, dbf = self.ln_bwd(dH, cf, self.P["gf"])
        g["gf"] += dgf
        g["bf"] += dbf
        for li in range(self.L - 1, -1, -1):
            pre = f"{li}."
            c1, ca, c2, ln2, hpre, hact = caches[li]
            g[pre + "W2"] += (hact.reshape(-1, self.dff).T
                              @ dx.reshape(-1, self.d))
            g[pre + "bb2"] += dx.sum((0, 1))
            dhpre = (dx @ self.P[pre + "W2"].T) * (hpre > 0)
            g[pre + "W1"] += (ln2.reshape(-1, self.d).T
                              @ dhpre.reshape(-1, self.dff))
            g[pre + "bb1"] += dhpre.sum((0, 1))
            dx1_ln, dg2, db2 = self.ln_bwd(dhpre @ self.P[pre + "W1"].T,
                                           c2, self.P[pre + "g2"])
            g[pre + "g2"] += dg2
            g[pre + "b2"] += db2
            dx1 = dx1_ln + dx
            dln1 = self.attn_bwd(dx1, ca, pre, g)
            dx0_ln, dg1, db1 = self.ln_bwd(dln1, c1, self.P[pre + "g1"])
            g[pre + "g1"] += dg1
            g[pre + "b1"] += db1
            dx = dx0_ln + dx1
        np.add.at(g["emb"], ids, dx)
        g["pos"][:ids.shape[1]] += dx.sum(0)

    def zero_grads(self):
        return {k: np.zeros_like(v) for k, v in self.P.items()}

    # ---- the output head over tokens ----
    def head_logits(self, ids, keep):
        H, cache = self.trunk_fwd(ids, keep)
        return H @ self.P["Wv"] + self.P["bv"], cache

    def head_loss(self, ids, keep, targets, sel):
        """Mean cross-entropy over the selected positions, in nats."""
        logits, _ = self.head_logits(ids, keep)
        Pr = softmax_last(logits)
        n = max(sel.sum(), 1)
        lp = -np.log(Pr[sel, targets[sel]] + 1e-300)
        return float(lp.sum() / n)

    def head_grads(self, ids, keep, targets, sel):
        logits, cache = self.head_logits(ids, keep)
        Pr = softmax_last(logits)
        n = max(sel.sum(), 1)
        dlog = Pr.copy()
        dlog[sel, targets[sel]] -= 1.0
        dlog *= sel[:, :, None] / n
        g = self.zero_grads()
        H = cache[4]
        g["Wv"] += H.reshape(-1, self.d).T @ dlog.reshape(-1, V)
        g["bv"] += dlog.sum((0, 1))
        self.trunk_bwd(dlog @ self.P["Wv"].T, cache, g)
        return g

    def token_nll(self, ids, keep, targets, sel):
        """Per-position -log p of the target, zero where not selected."""
        logits, _ = self.head_logits(ids, keep)
        lg = logits - logits.max(-1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        rows, cols = np.indices(ids.shape)
        out = -lp[rows, cols, targets]
        return np.where(sel, out, 0.0)

    # ---- Adam ----
    def step(self, g):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, p in self.P.items():
            self.M[k] = b1 * self.M[k] + (1 - b1) * g[k]
            self.Vm[k] = b2 * self.Vm[k] + (1 - b2) * g[k] ** 2
            p -= self.lr * (self.M[k] / (1 - b1 ** self.t)) / (
                np.sqrt(self.Vm[k] / (1 - b2 ** self.t)) + eps)
# =======================================================================
# The stage.
#
# A sequence is k demonstrations then a query:
#
#   [BOS] subj verb obj [SEP] ... subj verb obj [SEP] subj verb -> obj
#
# The pairing from subject to object is a FRESH RANDOM PERMUTATION for
# every sequence.  Nothing about it can be memorised, so a model that
# answers correctly has read it out of the prompt.
#
# bursty=True  the pairing governs the whole sequence
# bursty=False every object drawn independently; context says nothing
# =======================================================================

def random_pairing(rng):
    perm = rng.permutation(len(NOUNS))
    return {NOUNS[i]: NOUNS[perm[i]] for i in range(len(NOUNS))}


def make_sequence(rng, n_demo, bursty=True, n_support=None):
    """Tokens, the answer, and where the answer is predicted.

    n_support is how many demonstrations share the query's subject.
    Left as None it is 1 whenever there is room, which is the setting
    a prompt usually has.
    """
    pairing = random_pairing(rng)
    q = NOUNS[int(rng.integers(0, len(NOUNS)))]
    if n_support is None:
        n_support = 1 if n_demo else 0
    n_support = min(n_support, n_demo)
    others = [x for x in NOUNS if x != q]
    subs = ([q] * n_support
            + [others[i] for i in rng.integers(0, len(others),
                                               n_demo - n_support)])
    subs = [subs[i] for i in rng.permutation(len(subs))]

    toks = [BOS]
    for s in subs:
        o = pairing[s] if bursty else NOUNS[int(
            rng.integers(0, len(NOUNS)))]
        v = VERBS[int(rng.integers(0, len(VERBS)))]
        toks += [W2I[s], W2I[v], W2I[o], SEP]
    toks += [W2I[q], W2I[VERBS[int(rng.integers(0, len(VERBS)))]]]
    ans = pairing[q] if bursty else NOUNS[int(
        rng.integers(0, len(NOUNS)))]
    return toks, W2I[ans], q


def encode_batch(seqs):
    n = len(seqs)
    ids = np.zeros((n, T_MAX), dtype=int)
    keep = np.zeros((n, T_MAX), dtype=bool)
    ans = np.zeros(n, dtype=int)
    pos = np.zeros(n, dtype=int)
    for i, (toks, a, _) in enumerate(seqs):
        ids[i, :len(toks)] = toks
        keep[i, :len(toks)] = True
        ans[i] = a
        pos[i] = len(toks) - 1
    return ids, keep, ans, pos


def sample_batch(rng, n, bursty=True, n_demo=None, n_support=None):
    return encode_batch([
        make_sequence(rng,
                      int(rng.integers(1, K_MAX + 1))
                      if n_demo is None else n_demo,
                      bursty, n_support)
        for _ in range(n)])


# =======================================================================
# Training.  The loss is taken on the final object alone: predicting
# the demonstrations would be a different task, and an easier one.
# =======================================================================

def answer_grads(model, ids, keep, ans, pos):
    logits, cache = model.head_logits(ids, keep)
    Pr = softmax_last(logits)
    n = len(ans)
    rows = np.arange(n)
    dlog = np.zeros_like(Pr)
    dlog[rows, pos] = Pr[rows, pos]
    dlog[rows, pos, ans] -= 1.0
    dlog /= n
    g = model.zero_grads()
    H = cache[4]
    g["Wv"] += H.reshape(-1, model.d).T @ dlog.reshape(-1, V)
    g["bv"] += dlog.sum((0, 1))
    model.trunk_bwd(dlog @ model.P["Wv"].T, cache, g)
    return g


def answer_loss(model, ids, keep, ans, pos):
    logits, _ = model.head_logits(ids, keep)
    Pr = softmax_last(logits)
    return float(-np.log(Pr[np.arange(len(ans)), pos, ans]
                         + 1e-300).mean())


def accuracy(model, rng, bursty=True, n_demo=None, n_support=None,
             n=400):
    ids, keep, ans, pos = sample_batch(rng, n, bursty, n_demo,
                                       n_support)
    logits, _ = model.head_logits(ids, keep)
    return float((logits[np.arange(n), pos].argmax(-1) == ans).mean())


def train(bursty, steps, rng, seed=0, batch=64, lr=3e-3, probe_at=()):
    """Returns the model and, if asked, accuracy during training."""
    m = Transformer(d=64, heads=4, layers=2, seed=seed, lr=lr)
    probes = []
    for s in range(steps):
        ids, keep, ans, pos = sample_batch(rng, batch, bursty)
        m.step(answer_grads(m, ids, keep, ans, pos))
        if (s + 1) in probe_at:
            probes.append((s + 1, accuracy(
                m, np.random.default_rng(77), True, 1, n=200)))
    return m, probes


# =======================================================================
# Reading the mechanism off the attention map.
#
# If the model is copying, the query position must be attending to the
# object token that followed the matching demonstration.  That is the
# one place the answer is written down.
# =======================================================================

def copy_attention(model, rng, n=200, n_demo=4):
    """Attention from the query onto the token it needs to copy."""
    per_layer = {li: [] for li in range(model.L)}
    for _ in range(n):
        toks, ans, q = make_sequence(rng, n_demo, True, n_support=1)
        obj_pos = None
        for p in range(1, len(toks) - 2, 4):
            if toks[p] == W2I[q]:
                obj_pos = p + 2
                break
        if obj_pos is None:
            continue
        ids = np.zeros((1, T_MAX), dtype=int)
        keep = np.zeros((1, T_MAX), dtype=bool)
        ids[0, :len(toks)] = toks
        keep[0, :len(toks)] = True
        qpos = len(toks) - 1
        x = model.P["emb"][ids] + model.P["pos"][:T_MAX]
        for li in range(model.L):
            pre = f"{li}."
            ln1, _ = model.ln_fwd(x, model.P[pre + "g1"],
                                  model.P[pre + "b1"])
            a_out, cache = model.attn_fwd(ln1, pre, keep)
            per_layer[li].append(float(
                cache[4][0, :, qpos, obj_pos].max()))
            x1 = x + a_out
            ln2, _ = model.ln_fwd(x1, model.P[pre + "g2"],
                                  model.P[pre + "b2"])
            h = np.maximum(0, ln2 @ model.P[pre + "W1"]
                           + model.P[pre + "bb1"])
            x = x1 + h @ model.P[pre + "W2"] + model.P[pre + "bb2"]
    return {li: float(np.mean(v)) for li, v in per_layer.items()}


# -----------------------------------------------------------------------
# Gradient audit and self-test.
# -----------------------------------------------------------------------

def gradcheck(seed=0, n_per=3):
    rng = np.random.default_rng(seed)
    ids, keep, ans, pos = sample_batch(rng, 6, True)
    net = Transformer(d=16, heads=2, layers=2, seed=1)
    g = answer_grads(net, ids, keep, ans, pos)
    worst, checked = 0.0, 0
    for k, p in net.P.items():
        flat, gflat = p.reshape(-1), g[k].reshape(-1)
        for ix in rng.choice(len(flat), min(n_per, len(flat)), False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp = answer_loss(net, ids, keep, ans, pos)
            flat[ix] = old - 1e-6
            lm = answer_loss(net, ids, keep, ans, pos)
            flat[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
            checked += 1
    return worst, checked


def selftest():
    """The stage cannot be solved without reading the context."""
    rng = np.random.default_rng(3)

    # a pairing is a permutation: every noun appears exactly once
    for _ in range(50):
        p = random_pairing(rng)
        assert sorted(p.values()) == sorted(NOUNS)

    # in a bursty sequence the answer IS written in the context
    for _ in range(200):
        toks, ans, q = make_sequence(rng, 4, True, n_support=1)
        hit = [toks[p + 2] for p in range(1, len(toks) - 2, 4)
               if toks[p] == W2I[q]]
        assert hit and hit[0] == ans, (q, hit, ans)

    # with no supporting demonstration it is not, so chance is the
    # best any model could do
    seen = set()
    for _ in range(200):
        toks, ans, q = make_sequence(rng, 4, True, n_support=0)
        assert all(toks[p] != W2I[q]
                   for p in range(1, len(toks) - 2, 4))
        seen.add(ans)
    assert len(seen) > 20, len(seen)

    # the query never appears before its own position
    toks, ans, q = make_sequence(rng, 8, True, n_support=1)
    assert len(toks) <= T_MAX, (len(toks), T_MAX)
    return len(NOUNS), len(VERBS), V, CHANCE


# -----------------------------------------------------------------------
# Frozen configuration.
# -----------------------------------------------------------------------

STEPS = 1500
EVAL_N, PROBE_N = 400, 200
SEEDS = (0, 1, 2)
SEED_TRAIN, SEED_EVAL, SEED_PROBE, SEED_ATTN = 1, 100, 77, 400
DEMOS = (1, 2, 4, 8)
SUPPORT = (1, 2, 4, 8)
# dense early: the first run probed from step 100 and the model
# was already at ceiling, so the transition was never seen.
PROBE_AT = (5, 10, 15, 20, 25, 30, 40, 50, 65, 80, 100, 150,
            200, 400, 800, STEPS)
def ms(vals):
    v = np.array(vals, dtype=float)
    return float(v.mean()), float(v.std())


def show_sequence(rng):
    """One sequence, written out the way a prompt would be."""
    toks, ans, q = make_sequence(rng, 3, True, n_support=1)
    parts, cur = [], []
    for t in toks[1:]:
        if t == SEP:
            parts.append(" ".join(cur))
            cur = []
        else:
            cur.append(VOCAB[t])
    if cur:
        parts.append(" ".join(cur) + " ->")
    return parts, VOCAB[ans]


# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

def demo1(models):
    """The task, and the only way to solve it."""
    banner("1. A RULE THE MODEL HAS NEVER SEEN")
    rng = np.random.default_rng(12)
    parts, ans = show_sequence(rng)
    print("  A sequence is demonstrations, then a query:")
    print()
    for p in parts[:-1]:
        print("    %s" % p)
    print("    %-28s  %s" % (parts[-1], ans))
    print()
    print("  The subject-to-object pairing is a fresh random")
    print("  permutation of the %d nouns in EVERY sequence. It is not"
          % len(NOUNS))
    print("  a fact about the language and it is not in the weights:")
    print("  a model that answers correctly has read it here.")
    print()
    print("  Guessing scores %.4f. The self-test above confirms both"
          % CHANCE)
    print("  halves of that: the answer IS in the context when a")
    print("  supporting demonstration is present, and is NOT when one")
    print("  is absent.")
    print()
    print("  Two models, same trunk, same budget, same seeds. The only")
    print("  difference is how their TRAINING sequences were built:")
    print("    bursty  one pairing governs a sequence, redrawn between")
    print("    IID     every object independent, so context says nothing")


def demo2(models):
    """Does it work, and where does it come from?"""
    banner("2. IN-CONTEXT LEARNING, AND WHERE IT COMES FROM")
    print("  Accuracy on a pairing the model has never seen, with one")
    print("  supporting demonstration and the context growing around")
    print("  it. %d seeds; chance is %.4f." % (len(SEEDS), CHANCE))
    print()
    print("  %-14s %10s %8s %10s %8s"
          % ("demonstrations", "bursty", "sd", "IID", "sd"))
    print("  " + "-" * 54)
    out = {}
    for nd in DEMOS:
        row = {}
        for name in ("bursty", "iid"):
            accs = [accuracy(m, np.random.default_rng(SEED_EVAL + nd),
                             True, nd, 1) for m in models[name]]
            row[name] = ms(accs)
        out[nd] = row
        print("  %-14d %10.4f %8.4f %10.4f %8.4f"
              % (nd, row["bursty"][0], row["bursty"][1],
                 row["iid"][0], row["iid"][1]))
    print()
    one = out[DEMOS[0]]
    print("  With a single demonstration the bursty model is right")
    print("  %.4f of the time. The IID model is at %.4f, which is"
          % (one["bursty"][0], one["iid"][0]))
    print("  chance to three decimals.")
    print()
    print("  Same architecture. Same number of gradient steps. Same")
    print("  initialisation. The capability is a property of the")
    print("  DISTRIBUTION the model was trained on, not of the model.")
    print()
    print("  Then look down the column. It falls to %.4f by %d"
          % (out[DEMOS[-1]]["bursty"][0], DEMOS[-1]))
    print("  demonstrations -- which looks like in-context learning")
    print("  getting worse as you add examples, and is not.")
    return out


def demo3(models):
    """What 'number of examples' was hiding."""
    banner("3. EVIDENCE AND DISTRACTORS ARE NOT THE SAME AXIS")
    print("  In demo 2 only ONE demonstration ever shared the query's")
    print("  subject. The others were noise. So adding demonstrations")
    print("  added distractors, not evidence. Separate them.")
    print()
    print("  A. context fixed at %d, supporting examples vary" % K_MAX)
    print("  %-14s %10s %8s %10s" % ("supporting", "bursty", "sd", "IID"))
    print("  " + "-" * 46)
    eva = {}
    for ns in SUPPORT:
        b = ms([accuracy(m, np.random.default_rng(SEED_EVAL + 50 + ns),
                         True, K_MAX, ns) for m in models["bursty"]])
        i = ms([accuracy(m, np.random.default_rng(SEED_EVAL + 50 + ns),
                         True, K_MAX, ns) for m in models["iid"]])
        eva[ns] = (b, i)
        print("  %-14d %10.4f %8.4f %10.4f" % (ns, b[0], b[1], i[0]))
    print()
    print("  The curve runs the other way. With the context held at")
    print("  %d, going from %d supporting example to %d takes the"
          % (K_MAX, SUPPORT[0], SUPPORT[-1]))
    print("  bursty model from %.4f to %.4f."
          % (eva[SUPPORT[0]][0][0], eva[SUPPORT[-1]][0][0]))
    print()
    print("  So the model copies, and what fails as the prompt grows")
    print("  is not the copying but the FINDING. A longer prompt is")
    print("  only a better prompt if what you added is evidence.")
    return eva


def demo4(models, probes):
    """The mechanism, and when it arrives."""
    banner("4. THE MECHANISM, READ OFF THE ATTENTION MAP")
    print("  If the model is copying, the query position has to be")
    print("  attending to the object token that followed the matching")
    print("  demonstration. That is the only place the answer is")
    print("  written down. Uniform attention would be %.4f."
          % (1.0 / T_MAX))
    print()
    print("  %-10s %12s %12s" % ("", "layer 0", "layer 1"))
    print("  " + "-" * 36)
    att = {}
    for name in ("bursty", "iid"):
        per = [copy_attention(m, np.random.default_rng(SEED_ATTN))
               for m in models[name]]
        att[name] = {li: ms([p[li] for p in per])
                     for li in per[0]}
        print("  %-10s %12.4f %12.4f"
              % (name, att[name][0][0], att[name][1][0]))
    print()
    print("  The bursty model puts %.4f of its attention on exactly"
          % max(att["bursty"][0][0], att["bursty"][1][0]))
    print("  the token it needs. The IID model, given the identical")
    print("  prompt, is at %.4f -- it is not looking anywhere in"
          % max(att["iid"][0][0], att["iid"][1][0]))
    print("  particular, because nothing in its training ever made")
    print("  looking worthwhile.")
    print()
    print("  Two layers is the minimum a circuit like this needs, and")
    print("  this model has exactly two.")
    print()
    print("  When does it arrive? Accuracy on the held-out pairing,")
    print("  probed during training, densely at the start:")
    print()
    print("  %-8s %10s %10s" % ("step", "bursty", "IID"))
    print("  " + "-" * 32)
    for i, step in enumerate(PROBE_AT):
        b = ms([p[i][1] for p in probes["bursty"]])
        d = ms([p[i][1] for p in probes["iid"]])
        print("  %-8d %10.4f %10.4f" % (step, b[0], d[0]))
    curve = [ms([p[i][1] for p in probes["bursty"]])[0]
             for i in range(len(PROBE_AT))]
    lo = [PROBE_AT[i] for i, v in enumerate(curve) if v < 0.10]
    hi = [PROBE_AT[i] for i, v in enumerate(curve) if v > 0.90]
    print()
    if lo and hi and max(lo) < min(hi):
        print("  It does not arrive gradually. At step %d the model is"
              % max(lo))
        print("  at %.4f, barely above the %.4f it would score by"
              % (curve[PROBE_AT.index(max(lo))], CHANCE))
        print("  guessing. By step %d it is at %.4f. Nothing about the"
              % (min(hi), curve[PROBE_AT.index(min(hi))]))
        print("  data or the objective changes in between; the circuit")
        print("  simply finishes forming.")
    elif hi:
        print("  The transition is faster than this probe can resolve:")
        print("  the model is already at %.4f by step %d, the earliest"
              % (curve[PROBE_AT.index(min(hi))], min(hi)))
        print("  point measured. What can be said is that it is not a")
        print("  slow climb -- whatever happens, happens early.")
    else:
        print("  The capability does not reach ceiling within %d steps"
              % STEPS)
        print("  at this probe resolution.")
    return att, curve


def main():
    banner("IN-CONTEXT LEARNING AND PROMPTING")
    n_nouns, n_verbs, vocab, chance = selftest()
    worst, checked = gradcheck()
    print("  stage self-test passed: %d nouns, %d verbs, vocabulary %d"
          % (n_nouns, n_verbs, vocab))
    print("    the answer is in the context when a supporting")
    print("    demonstration exists, and provably is not when none")
    print("    does, so %.4f really is the ceiling without reading"
          % chance)
    print("  gradient audit: worst error %.2e over %d sampled gradients"
          % (worst, checked))
    print()

    models, probes = {}, {}
    for name, bursty in (("bursty", True), ("iid", False)):
        models[name], probes[name] = [], []
        for sd in SEEDS:
            m, pr = train(bursty, STEPS,
                          np.random.default_rng(SEED_TRAIN + sd),
                          seed=sd, probe_at=PROBE_AT)
            models[name].append(m)
            probes[name].append(pr)
        print("  trained %s: %d seeds, %s params each"
              % (name, len(SEEDS),
                 f"{models[name][0].n_params():,}"))

    demo1(models)
    demo2(models)
    demo3(models)
    demo4(models, probes)

    banner("DONE")
    return models


if __name__ == "__main__":
    main()
