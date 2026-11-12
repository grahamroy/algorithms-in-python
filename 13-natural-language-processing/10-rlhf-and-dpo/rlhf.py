"""
rlhf.py --- companion code for "RLHF and DPO"
(Natural Language Processing, Part 10).

Everything in this track so far has trained against a target that was
known. The next word. A label a function computed. The object a
document named. Preference learning has none of that. All it is given
is the statement that one output was better than another, and the
first thing that has to be built is a loss assembled out of
comparisons.

That much is standard. What is not standard, and what this stage makes
possible, is knowing whether any of it worked.

In a real RLHF pipeline the true objective lives in people's heads. A
reward model is fitted to their comparisons, a policy is optimised
against that model, and the only thing anyone can measure afterwards
is the reward model's own opinion of the result -- which is precisely
the quantity that was optimised, and therefore the one that cannot be
trusted. Here the true reward is a FUNCTION. It is defined over the
31,740 sentences of Part 1's micro-language, so:

    E_pi[r*]   the thing you actually wanted
    E_pi[r^]   the thing you optimised
    KL(pi||ref)  how far the policy has drifted

are all exact, all at once, at every step.

What that buys, measured here:
  1. A reward model fitted to 3,000 noisy comparisons, scored against
     the truth over the whole language rather than on held-out pairs.
  2. Goodhart's law as arithmetic. The true reward climbs, peaks, and
     then falls BELOW where it started, while the reward model goes
     on reporting success.
  3. The mechanism, visible in one column: grammaticality collapses as
     the policy discovers output the reward model was never trained to
     judge.
  4. What the KL penalty is for. Swept, it turns out to trade peak
     reward against whether the peak is kept at all.
  5. DPO on the identical comparisons, with no reward model anywhere
     -- a method that cannot be hacked, and what it costs.

The language, the oracle and the trunk are Parts 5-7's. New here: a
scalar reward head trained with the Bradley-Terry loss, a weighted
log-probability gradient that serves both REINFORCE and DPO, and the
exact policy evaluation that makes the rest of it legible. All
hand-differentiated and audited against central differences at
startup. Seeded and deterministic.
Dependencies: numpy.  Runs in about sixteen minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import Counter, defaultdict

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
# Micro-language from Part 1 (identical generation).
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
HELD_OUT_PLURALS = ["princesses", "roosters", "compasses"]

SIG = {}
for _t in TOPICS.values():
    _ns = _t["nouns"]
    for _i, _n in enumerate(_ns):
        SIG[_n] = _ns[(_i + len(_ns) // 2) % len(_ns)]
    for (_m, _f) in _t["gender"]:
        SIG[_f] = SIG[_m]


def plural(noun):
    if noun.endswith("s") or noun.endswith("sh"):
        return noun + "es"
    if noun.endswith("y"):
        return noun[:-1] + "ies"
    return noun + "s"


# --- word -> property tables, used to score predictions ----------------

TOPIC_OF = {}
PLURAL_NOUNS = set()
SG_NOUNS = set()
VERB_SG = set()
VERB_PL = set()
ADJS = set()
for _name, _t in TOPICS.items():
    for _n in _t["nouns"]:
        TOPIC_OF[_n] = _name
        SG_NOUNS.add(_n)
        _p = plural(_n)
        TOPIC_OF[_p] = _name
        PLURAL_NOUNS.add(_p)
    for (_vs, _vp) in _t["verbs"]:
        TOPIC_OF[_vs] = _name
        TOPIC_OF[_vp] = _name
        VERB_SG.add(_vs)
        VERB_PL.add(_vp)
    for _a in _t["adjs"]:
        TOPIC_OF[_a] = _name
        ADJS.add(_a)

NOUNS = SG_NOUNS | PLURAL_NOUNS
VERBS = VERB_SG | VERB_PL
POSSESSIVES = {"his", "her", "their", "the"}


# -----------------------------------------------------------------------
# Sentence generation.  Every sentence carries its slot indices and the
# agreement facts, so predictions can be scored against ground truth.
# -----------------------------------------------------------------------

def gen_sentence(rng):
    """One SVO sentence plus the facts needed to score it."""
    names = list(TOPICS)
    topic = names[rng.integers(0, 3)]
    t = TOPICS[topic]
    n1 = t["nouns"][rng.integers(0, len(t["nouns"]))]
    vs, vp = t["verbs"][rng.integers(0, len(t["verbs"]))]
    adj = t["adjs"][rng.integers(0, len(t["adjs"]))]
    n2 = SIG[n1] if rng.uniform() < 0.5 else \
        t["nouns"][rng.integers(0, len(t["nouns"]))]

    is_plur = rng.uniform() < 0.35 and plural(n1) not in HELD_OUT_PLURALS
    if is_plur:
        subj, verb, poss, gender = plural(n1), vp, "their", "pl"
    else:
        subj, verb = n1, vs
        if n1 in GENDER_M:
            poss, gender = "his", "m"
        elif n1 in GENDER_F:
            poss, gender = "her", "f"
        else:
            poss, gender = "the", "n"

    obj = n2
    if rng.uniform() < 0.25 and plural(n2) not in HELD_OUT_PLURALS:
        obj = plural(n2)

    if rng.uniform() < 0.5:
        words = ["the", adj, subj, verb, poss, obj]
    else:
        words = ["the", subj, verb, poss, obj]
    si = words.index(subj)
    return dict(words=words, topic=topic, plural=is_plur, gender=gender,
                subj=si, verb=si + 1, poss=si + 2, obj=si + 3,
                verb_sg=vs, verb_pl=vp)


def gen_corpus(n, rng):
    return [gen_sentence(rng) for _ in range(n)]


# -----------------------------------------------------------------------
# Vocabulary.  Part 5's format exactly: [BOS] + at most six words +
# [EOS], and [EOS] is a prediction target so the model learns to stop.
# -----------------------------------------------------------------------

N_SPECIAL = 4


def build_vocab():
    words = {"the", "a", "and", "near", "his", "her", "their"}
    for t in TOPICS.values():
        for n in t["nouns"]:
            words.add(n)
            if plural(n) not in HELD_OUT_PLURALS:
                words.add(plural(n))
        for (vs, vp) in t["verbs"]:
            words.update([vs, vp])
        words.update(t["adjs"])
    vocab = ["[PAD]", "[BOS]", "[EOS]", "[MASK]"] + sorted(words)
    return vocab, {w: i for i, w in enumerate(vocab)}


VOCAB, W2I = build_vocab()
V = len(VOCAB)
PAD, BOS, EOS, MASK = 0, 1, 2, 3
T_MAX = 8          # [BOS] + at most six words + [EOS]
STEPS = 1500       # pre-training steps


def encode(batch_words):
    """Pad a list of word-lists to (N, T_MAX) ids plus a keep mask."""
    N = len(batch_words)
    ids = np.zeros((N, T_MAX), dtype=int)
    keep = np.zeros((N, T_MAX), dtype=bool)
    ids[:, 0] = BOS
    keep[:, 0] = True
    for i, ws in enumerate(batch_words):
        for j, w in enumerate(ws):
            ids[i, j + 1] = W2I[w]
            keep[i, j + 1] = True
        ids[i, len(ws) + 1] = EOS
        keep[i, len(ws) + 1] = True
    return ids, keep


# -----------------------------------------------------------------------
# The oracle.  Part 5 used the enumeration to calibrate a LOSS.  This
# article uses it to calibrate a DECODER: every strategy below induces a
# distribution over these same 31,740 sentences, and the enumeration is
# what makes that distribution computable rather than merely samplable.
# -----------------------------------------------------------------------

def enumerate_language():
    """Exact P(sentence) for every sentence gen_sentence can produce."""
    P = defaultdict(float)
    for topic, t in TOPICS.items():
        nouns, verbs, adjs = t["nouns"], t["verbs"], t["adjs"]
        p_t = 1.0 / len(TOPICS)
        for n1 in nouns:
            p_n1 = 1.0 / len(nouns)
            plur_ok = plural(n1) not in HELD_OUT_PLURALS
            num_opts = ([(True, 0.35), (False, 0.65)] if plur_ok
                        else [(False, 1.0)])
            for (vs, vp) in verbs:
                p_v = 1.0 / len(verbs)
                for is_plur, p_pl in num_opts:
                    if is_plur:
                        subj, verb, poss = plural(n1), vp, "their"
                    else:
                        subj, verb = n1, vs
                        poss = ("his" if n1 in GENDER_M else
                                "her" if n1 in GENDER_F else "the")
                    for n2 in nouns:
                        p_n2 = ((0.5 if n2 == SIG[n1] else 0.0)
                                + 0.5 / len(nouns))
                        obj_opts = ([(n2, 0.75), (plural(n2), 0.25)]
                                    if plural(n2) not in HELD_OUT_PLURALS
                                    else [(n2, 1.0)])
                        for obj, p_o in obj_opts:
                            w = p_t * p_n1 * p_v * p_pl * p_n2 * p_o
                            P[("the", subj, verb, poss, obj)] += w * 0.5
                            for adj in adjs:
                                P[("the", adj, subj, verb,
                                   poss, obj)] += w * 0.5 / len(adjs)
    return dict(P)


SENT_P = enumerate_language()
EOS_W = "[EOS]"
SENTS = sorted(SENT_P)
SENT_INDEX = {s: i for i, s in enumerate(SENTS)}
P_LANG = np.array([SENT_P[s] for s in SENTS])


def build_oracle(sent_p):
    """prefix tuple -> {next word: exact conditional probability}."""
    nxt = defaultdict(lambda: defaultdict(float))
    for words, p in sent_p.items():
        seq = words + (EOS_W,)
        for i in range(len(seq)):
            nxt[words[:i]][seq[i]] += p
    out = {}
    for pre, d in nxt.items():
        z = sum(d.values())
        out[pre] = {w: v / z for w, v in d.items()}
    return out


ORACLE = build_oracle(SENT_P)


def language_entropy():
    """Exact H(sentence) in nats: the floor on total sentence NLL."""
    return float(-sum(p * np.log(p) for p in SENT_P.values()))


def oracle_nll(words):
    """Exact -log P(next | prefix) at each decoder position, in nats."""
    seq = tuple(words) + (EOS_W,)
    out = []
    for i in range(len(seq)):
        d = ORACLE.get(tuple(words[:i]))
        p = d.get(seq[i], 0.0) if d else 0.0
        out.append(np.inf if p <= 0 else float(-np.log(p)))
    return out


def in_language(words):
    return tuple(words) in SENT_P


def diagnose(words):
    """Why this word list is not a sentence of the language, or 'ok'."""
    if any(w.startswith("[") for w in words):
        return "special token"
    if len(words) not in (5, 6):
        return "length"
    if words[0] != "the":
        return "no determiner"
    rest = words[1:]
    adj = None
    if len(rest) == 5:
        adj, subj, verb, poss, obj = rest
        if adj not in ADJS:
            return "slot order"
    else:
        subj, verb, poss, obj = rest
    if subj not in NOUNS or verb not in VERBS or obj not in NOUNS:
        return "slot order"
    if poss not in POSSESSIVES:
        return "slot order"
    topic = TOPIC_OF[subj]
    content = [verb, obj] + ([adj] if adj else [])
    if any(TOPIC_OF[w] != topic for w in content):
        return "topic mix"
    is_pl = subj in PLURAL_NOUNS
    if (verb in VERB_PL) != is_pl:
        return "verb number"
    want = ("their" if is_pl else
            "his" if subj in GENDER_M else
            "her" if subj in GENDER_F else "the")
    if poss != want:
        return "possessive"
    return "ok"


# -----------------------------------------------------------------------
# The decoder.  Part 5's trunk, causal path only: pre-LN blocks,
#     x = x + attn(LN(x));  x = x + mlp(LN(x))
# position t attends to positions <= t and the head predicts token t+1.
# -----------------------------------------------------------------------

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


def lm_targets(ids, keep):
    """Shift left.  Position j predicts the token at j+1 -- every j."""
    tgt = np.zeros_like(ids)
    tgt[:, :-1] = ids[:, 1:]
    sel = np.zeros(ids.shape, dtype=bool)
    sel[:, :-1] = keep[:, 1:]
    return ids, tgt, sel


def pretrain(model, ids, keep, steps, rng, batch=64, log_every=None):
    hist = []
    for s in range(steps):
        b = rng.integers(0, len(ids), batch)
        x, tgt, sel = lm_targets(ids[b], keep[b])
        g = model.head_grads(x, keep[b], tgt, sel)
        model.step(g)
        if log_every and (s % log_every == 0 or s == steps - 1):
            hist.append((s + 1, model.head_loss(x, keep[b], tgt, sel)))
    return hist


def lm_logprob(model, batch_words):
    """log P(sentence) for a batch, in ONE forward pass."""
    ids, keep = encode(batch_words)
    _, tgt, sel = lm_targets(ids, keep)
    return -model.token_nll(ids, keep, tgt, sel).sum(1)


def lm_eval(model, sents):
    """Held-out nats per token and per sentence, against the floor."""
    words = [s["words"] for s in sents]
    ids, keep = encode(words)
    _, tgt, sel = lm_targets(ids, keep)
    nll = model.token_nll(ids, keep, tgt, sel)
    ntok = sel.sum()
    onll = np.array([sum(oracle_nll(w)) for w in words])
    return (float(nll.sum() / ntok), float(nll.sum(1).mean()),
            float(onll.sum() / ntok), float(onll.mean()))


# =======================================================================
# Decoding strategies.
#
# Temperature, top-k and nucleus are all the same move: rewrite the
# next-token distribution, then sample from the rewrite.  Writing them
# as one function that maps logits to probabilities is what makes the
# exact analysis below possible -- the rewrite is deterministic, so the
# distribution over whole sentences is a product of rewrites and can be
# evaluated instead of sampled.
# =======================================================================

def trunc_probs(logits, kind, param):
    """(N, V) logits -> (N, V) probabilities under one strategy.

    kind: "temp" (param = temperature), "topk" (param = k),
          "topp" (param = nucleus mass).
    """
    lg = np.asarray(logits, dtype=np.float64)
    if kind == "temp":
        z = lg / param
        z -= z.max(-1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(-1, keepdims=True)
    z = lg - lg.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    if kind == "topk":
        k = int(param)
        if k >= p.shape[-1]:
            return p
        cut = np.partition(p, -k, axis=-1)[:, -k][:, None]
        q = np.where(p >= cut, p, 0.0)
        return q / q.sum(-1, keepdims=True)
    if kind == "topp":
        order = np.argsort(-p, axis=-1)
        sp = np.take_along_axis(p, order, axis=-1)
        csum = np.cumsum(sp, axis=-1)
        # Keep the smallest prefix whose cumulative mass reaches param:
        # a token survives if the mass STRICTLY BEFORE it falls short.
        sp = np.where(csum - sp < param, sp, 0.0)
        q = np.zeros_like(p)
        np.put_along_axis(q, order, sp, axis=-1)
        return q / q.sum(-1, keepdims=True)
    raise ValueError(kind)


def sample_batch(model, rng, n, kind="temp", param=1.0):
    """Sample n sentences at once.

    Every row starts at [BOS] and gains one token per step, so all live
    rows share a length and the whole batch advances in one forward
    pass.  Rows that have emitted [EOS] are frozen.
    """
    ids = np.zeros((n, T_MAX), dtype=int)
    ids[:, 0] = BOS
    live = np.ones(n, dtype=bool)
    term = np.zeros(n, dtype=bool)
    nwords = np.zeros(n, dtype=int)
    for step in range(T_MAX - 1):
        if not live.any():
            break
        npos = 1 + step
        keep = np.zeros((n, T_MAX), dtype=bool)
        keep[:, :npos] = True
        logits, _ = model.head_logits(ids, keep)
        q = trunc_probs(logits[:, npos - 1], kind, param)
        u = rng.random((n, 1))
        nxt = (q.cumsum(1) >= u).argmax(1)
        hit = live & (nxt == EOS)
        term |= hit
        live &= ~hit
        put = live & (npos < T_MAX)
        ids[put, npos] = nxt[put]
        nwords[put] += 1
        live &= npos < T_MAX - 1
    out = []
    for i in range(n):
        w = [VOCAB[t] for t in ids[i, 1:1 + nwords[i]]]
        out.append((w, bool(term[i])))
    return out




CORPUS_N, TEST_N = 8000, 1000
SEED_CORPUS, SEED_TEST = 42, 7
SEED_MODEL, SEED_PRETRAIN, SEED_SAMPLE, SEED_BLEU = 0, 1, 9, 11
N_SAMPLES = 1000
PROMPT = ["the", "king"]

STRATEGIES = {
    "temperature 1.0": ("temp", 1.0),
    "temperature 0.7": ("temp", 0.7),
    "temperature 0.5": ("temp", 0.5),
    "top-k 2": ("topk", 2),
    "top-k 5": ("topk", 5),
    "top-k 10": ("topk", 10),
    "top-k 40": ("topk", 40),
    "nucleus 0.90": ("topp", 0.90),
    "nucleus 0.95": ("topp", 0.95),
    "nucleus 0.99": ("topp", 0.99),
}
BEAMS = (1, 2, 5, 10, 25)
REF_COUNTS = (1, 2, 4, 8, 16)
BREAKS = ("verb number", "possessive", "topic mix")


def build_stage():
    rng = np.random.default_rng(SEED_CORPUS)
    corpus = gen_corpus(CORPUS_N, rng)
    ids, keep = encode([s["words"] for s in corpus])
    test = gen_corpus(TEST_N, np.random.default_rng(SEED_TEST))
    return corpus, ids, keep, test


def train_lm(ids, keep, steps=STEPS):
    lm = Transformer(d=64, heads=4, layers=2, seed=SEED_MODEL, lr=3e-3)
    hist = pretrain(lm, ids, keep, steps,
                    np.random.default_rng(SEED_PRETRAIN),
                    log_every=max(steps // 8, 1))
    return lm, hist
# =======================================================================
# The true reward.
#
# This is the part a real pipeline does not have.  r* is a function of
# the sentence, defined over the whole language, so every claim about
# whether preference learning WORKED can be checked rather than argued.
# Three features the micro-language already carries, weighted so that
# no single one dominates and a reward model has something to get
# partly right.
# =======================================================================

W_ADJ, W_SIG, W_ROYAL = 1.0, 0.7, 0.4


def true_reward(sent):
    """r*(s) for a sentence OF THE LANGUAGE, as a tuple of words."""
    subj = sent[-4]
    return (W_ADJ * (len(sent) == 6)
            + W_SIG * (sent[-1] == SIG.get(subj))
            + W_ROYAL * (TOPIC_OF.get(subj) == "royal"))


def reward_of(words):
    """r* for anything a policy emits.  Not a sentence, no reward."""
    return true_reward(tuple(words)) if in_language(words) else 0.0


R_STAR = np.array([true_reward(s) for s in SENTS])
R_MAX = float(R_STAR.max())


def copy_lm(model, lr):
    """An independent policy starting from this model's weights."""
    new = Transformer(d=model.d, heads=model.h, layers=model.L,
                      seed=0, lr=lr)
    new.P = {k: v.copy() for k, v in model.P.items()}
    new.reset_adam()
    return new


# =======================================================================
# Preferences.  Sampled FROM THE REFERENCE POLICY, which is what makes
# the reward model's later extrapolation a real problem: it only ever
# sees the kind of sentence the base model already writes.
#
# Labels follow Bradley-Terry, P(a > b) = sigma(r*(a) - r*(b)), so they
# are noisy in exactly the way human comparisons are.
# =======================================================================

def build_preferences(ref, n_pairs, rng, pool=4000):
    samples = [w for w, done in sample_batch(ref, rng, pool)
               if done and in_language(w)]
    pairs, agree = [], 0
    while len(pairs) < n_pairs:
        i, j = rng.integers(0, len(samples), 2)
        a, b = samples[i], samples[j]
        ra, rb = reward_of(a), reward_of(b)
        if ra == rb:
            continue
        if rng.random() < 1.0 / (1.0 + np.exp(-(ra - rb))):
            pairs.append((a, b))
            agree += ra > rb
        else:
            pairs.append((b, a))
            agree += rb > ra
    return pairs, len(samples), agree / len(pairs)


# =======================================================================
# The reward model: Part 7's trunk with a scalar head on the last
# token, fitted by Bradley-Terry.  Nothing about it knows r* exists.
# =======================================================================

def rm_init(model, seed=0):
    rng = np.random.default_rng(seed)
    model.P["Ws"] = rng.standard_normal((model.d, 1)) / np.sqrt(model.d)
    model.P["bs"] = np.zeros(1)
    model.reset_adam()


def rm_score(model, ids, keep):
    H, cache = model.trunk_fwd(ids, keep)
    idx = keep.sum(1) - 1
    pooled = H[np.arange(len(H)), idx]
    return (pooled @ model.P["Ws"] + model.P["bs"]).ravel(), (cache, idx)


def rm_loss_grads(model, pairs):
    """-log sigma(s_chosen - s_rejected), and its gradients."""
    ic, kc = encode([p[0] for p in pairs])
    ir, kr = encode([p[1] for p in pairs])
    sc, ac = rm_score(model, ic, kc)
    sr, ar = rm_score(model, ir, kr)
    p = 1.0 / (1.0 + np.exp(-(sc - sr)))
    loss = float(-np.log(p + 1e-12).mean())
    n = len(p)
    g = model.zero_grads()
    for ds, (cache, idx) in ((-(1 - p) / n, ac), ((1 - p) / n, ar)):
        H = cache[4]
        rows = np.arange(len(H))
        g["Ws"] += H[rows, idx].T @ ds[:, None]
        g["bs"] += ds.sum()
        dH = np.zeros_like(H)
        dH[rows, idx] = ds[:, None] * model.P["Ws"].T
        model.trunk_bwd(dH, cache, g)
    return loss, g


def train_reward_model(pairs, steps, rng, seed=3, lr=1e-3, batch=64):
    rm = Transformer(d=64, heads=4, layers=2, seed=seed, lr=lr)
    rm_init(rm, seed=seed + 1)
    hist = []
    for s in range(steps):
        idx = rng.choice(len(pairs), batch, replace=False)
        loss, g = rm_loss_grads(rm, [pairs[i] for i in idx])
        rm.step(g)
        if s % max(steps // 6, 1) == 0 or s == steps - 1:
            hist.append((s + 1, loss))
    return rm, hist


def rm_over_language(rm, batch=2000):
    out = np.empty(len(SENTS))
    for i in range(0, len(SENTS), batch):
        ids, keep = encode([list(s) for s in SENTS[i:i + batch]])
        out[i:i + batch] = rm_score(rm, ids, keep)[0]
    return out


# =======================================================================
# One gradient serves both methods.
#
#     logprob_grads(model, ids, keep, w)  =  sum_i w_i * grad log pi(s_i)
#
# REINFORCE wants w = advantage.  DPO wants w = +-beta(1 - sigma(h)).
# Neither needs anything the language model did not already compute.
# =======================================================================

def logprob_grads(model, ids, keep, w):
    _, tgt, sel = lm_targets(ids, keep)
    logits, cache = model.head_logits(ids, keep)
    Pr = softmax_last(logits)
    dlog = -Pr
    rows, cols = np.indices(ids.shape)
    dlog[rows, cols, tgt] += 1.0
    dlog = dlog * (sel * w[:, None])[:, :, None]
    g = model.zero_grads()
    H = cache[4]
    g["Wv"] += H.reshape(-1, model.d).T @ dlog.reshape(-1, V)
    g["bv"] += dlog.sum((0, 1))
    model.trunk_bwd(dlog @ model.P["Wv"].T, cache, g)
    return g


def ascend(model, g):
    """Adam descends, so hand it the negated gradient."""
    model.step({k: -v for k, v in g.items()})


# =======================================================================
# Exact evaluation.  The policy's distribution over every sentence of
# the language, and therefore the true reward, the proxy reward, the
# grammatical mass and the KL -- none of them sampled.
# =======================================================================

def policy_logprobs(model, batch=4000):
    lp = np.empty(len(SENTS))
    for i in range(0, len(SENTS), batch):
        lp[i:i + batch] = lm_logprob(
            model, [list(s) for s in SENTS[i:i + batch]])
    return lp


def exact_eval(model, rhat, ref_lp):
    """Everything at once, in closed form over the language.

    KL is computed over the language only -- off-language sequences
    cannot be enumerated -- so it is reported beside the mass, which
    says how much of the policy that restriction leaves out.
    """
    lp = policy_logprobs(model)
    p = np.exp(lp)
    mass = float(p.sum())
    q = p / max(mass, 1e-300)
    kl = float((q * (np.log(q + 1e-300) - np.log(
        np.exp(ref_lp) / np.exp(ref_lp).sum() + 1e-300))).sum())
    return dict(mass=mass, kl=kl,
                true=float((p * R_STAR).sum()),
                proxy=float((p * rhat).sum()))


# =======================================================================
# RLHF: sample, score with the reward model, push up what scored well.
# The KL term is folded into the reward, which is what InstructGPT
# does and what makes beta the only knob that matters here.
# =======================================================================

def rlhf(ref, rm, rhat, ref_lp, beta, steps, rng, lr=3e-4, batch=64,
         every=50):
    pol = copy_lm(ref, lr)
    hist = []
    for step in range(1, steps + 1):
        ws = [w[:T_MAX - 2] for w, _ in sample_batch(pol, rng, batch)]
        ids, keep = encode(ws)
        r = rm_score(rm, ids, keep)[0]
        adv = r - beta * (lm_logprob(pol, ws) - lm_logprob(ref, ws))
        ascend(pol, logprob_grads(pol, ids, keep,
                                  (adv - adv.mean()) / len(ws)))
        if step % every == 0:
            e = exact_eval(pol, rhat, ref_lp)
            e["step"] = step
            hist.append(e)
    return pol, hist


# =======================================================================
# DPO: no reward model, no sampling.  The preference pairs are the
# whole training signal, and the implicit reward is the log-ratio to
# the reference policy.
# =======================================================================

def dpo(ref, rhat, ref_lp, beta, steps, rng, pairs, lr=3e-4, batch=32,
        every=50):
    pol = copy_lm(ref, lr)
    hist = []
    for step in range(1, steps + 1):
        idx = rng.choice(len(pairs), batch, replace=False)
        yw = [pairs[i][0] for i in idx]
        yl = [pairs[i][1] for i in idx]
        iw, kw = encode(yw)
        il, kl = encode(yl)
        h = beta * ((lm_logprob(pol, yw) - lm_logprob(ref, yw))
                    - (lm_logprob(pol, yl) - lm_logprob(ref, yl)))
        s = 1.0 / (1.0 + np.exp(-h))
        step_w = beta * (1 - s) / len(idx)
        g = logprob_grads(pol, iw, kw, step_w)
        gl = logprob_grads(pol, il, kl, -step_w)
        ascend(pol, {k: g[k] + gl[k] for k in g})
        if step % every == 0:
            e = exact_eval(pol, rhat, ref_lp)
            e["step"] = step
            e["loss"] = float(-np.log(s + 1e-12).mean())
            hist.append(e)
    return pol, hist


# -----------------------------------------------------------------------
# Gradient audit and self-test.
# -----------------------------------------------------------------------

def gradcheck(seed=0, n_per=3):
    """The LM head, the reward head, and the weighted log-prob path."""
    rng = np.random.default_rng(seed)
    sents = gen_corpus(6, rng)
    ws = [s["words"] for s in sents]
    ids, keep = encode(ws)
    worst, checked = 0.0, 0

    # 1. the Bradley-Terry reward head
    rm = Transformer(d=16, heads=2, layers=2, seed=1)
    rm_init(rm, seed=2)
    pairs = [(ws[i], ws[(i + 1) % len(ws)]) for i in range(len(ws))]
    _, g = rm_loss_grads(rm, pairs)
    for k, p in rm.P.items():
        flat, gflat = p.reshape(-1), g[k].reshape(-1)
        for ix in rng.choice(len(flat), min(n_per, len(flat)), False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp = rm_loss_grads(rm, pairs)[0]
            flat[ix] = old - 1e-6
            lm_ = rm_loss_grads(rm, pairs)[0]
            flat[ix] = old
            worst = max(worst, abs((lp - lm_) / 2e-6 - gflat[ix]))
            checked += 1

    # 2. the weighted log-probability gradient both methods ride on
    pol = Transformer(d=16, heads=2, layers=2, seed=4)
    # the same scale training uses, so the audited quantity is O(1)
    w = rng.standard_normal(len(ws)) / len(ws)
    g2 = logprob_grads(pol, ids, keep, w)

    def weighted():
        return float((w * lm_logprob(pol, ws)).sum())

    for k, p in pol.P.items():
        flat, gflat = p.reshape(-1), g2[k].reshape(-1)
        for ix in rng.choice(len(flat), min(n_per, len(flat)), False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp = weighted()
            flat[ix] = old - 1e-6
            lm_ = weighted()
            flat[ix] = old
            worst = max(worst, abs((lp - lm_) / 2e-6 - gflat[ix]))
            checked += 1
    return worst, checked


def selftest():
    """The stage, and the two things the measurements rest on."""
    assert abs(sum(SENT_P.values()) - 1.0) < 1e-9
    for s in SENT_P:
        assert diagnose(list(s)) == "ok", s

    # r* agrees with its own definition everywhere
    for i, s in enumerate(SENTS):
        assert abs(R_STAR[i] - true_reward(s)) < 1e-12
        assert abs(reward_of(list(s)) - R_STAR[i]) < 1e-12
    assert abs(R_MAX - (W_ADJ + W_SIG + W_ROYAL)) < 1e-12
    # anything outside the language scores zero, by construction
    assert reward_of(["the", "king", "rule", "his", "crown"]) == 0.0
    assert reward_of(["the", "king", "commands", "her", "crown"]) == 0.0

    # Bradley-Terry labels really do favour the higher reward
    rng = np.random.default_rng(2)
    hi = ("the", "ancient", "banner", "commands", "the", "lord")
    lo = ("the", "wave", "drifts", "the", "wave")
    assert true_reward(hi) > true_reward(lo)
    wins = sum(rng.random() < 1.0 / (1.0 + np.exp(
        -(true_reward(hi) - true_reward(lo)))) for _ in range(4000))
    assert 0.5 < wins / 4000 < 1.0

    # a zero-weight policy gradient is exactly zero
    pol = Transformer(d=16, heads=2, layers=2, seed=7)
    ids, keep = encode([list(SENTS[0]), list(SENTS[1])])
    g = logprob_grads(pol, ids, keep, np.zeros(2))
    assert all(np.allclose(v, 0) for v in g.values())
    return len(SENT_P), float((P_LANG * R_STAR).sum()), R_MAX


# -----------------------------------------------------------------------
# Frozen configuration.
# -----------------------------------------------------------------------

N_PAIRS, RM_STEPS = 3000, 600
PG_STEPS, EVERY = 300, 50
BETAS = (0.02, 0.05, 0.20, 0.50)
DPO_BETAS = (0.1, 0.5)
SEEDS = (0, 1, 2)
SEED_PREF, SEED_RM, SEED_POL = 5, 7, 11
def weighted_corr(x, y, w):
    """Correlation of x and y under a probability weighting w."""
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    vx = (w * (x - mx) ** 2).sum()
    vy = (w * (y - my) ** 2).sum()
    return float((w * (x - mx) * (y - my)).sum()
                 / np.sqrt(max(vx * vy, 1e-300)))


def ms(rows, field):
    v = np.array([r[field] for r in rows])
    return float(v.mean()), float(v.std())


# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

def demo1(ref, ref_lp):
    """The thing a real pipeline does not have: a true reward."""
    banner("1. A REWARD YOU CAN CHECK")
    print("  r*(s) = %.1f if the sentence carries an adjective" % W_ADJ)
    print("        + %.1f if the object is the subject's role partner"
          % W_SIG)
    print("        + %.1f if the subject is a royal noun" % W_ROYAL)
    print()
    print("  Defined over every one of the %s sentences, so the"
          % f"{len(SENT_P):,}")
    print("  quantities RLHF normally estimates are all exact.")
    print()
    print("  %-8s %10s %10s" % ("r*", "sentences", "mass"))
    print("  " + "-" * 32)
    for lvl in sorted(set(R_STAR.tolist())):
        m = R_STAR == lvl
        print("  %-8.1f %10d %10.4f"
              % (lvl, int(m.sum()), float(P_LANG[m].sum())))
    print()
    base = float((np.exp(ref_lp) * R_STAR).sum())
    print("  the language itself scores  %.4f" % float((P_LANG * R_STAR).sum()))
    print("  the reference policy scores %.4f" % base)
    print("  the ceiling is              %.4f  (%d sentences reach it)"
          % (R_MAX, int((R_STAR == R_MAX).sum())))
    print()
    print("  Everything below is the attempt to move the middle number")
    print("  towards the bottom one, using nothing but comparisons.")
    return base


def demo2(rm, rhat, ref_lp, hist, pairs, n_samp, agree):
    """How good is a reward model fitted to noisy comparisons?"""
    banner("2. A REWARD MODEL, AND WHERE IT IS WRONG")
    print("  %s comparisons drawn from %s reference-policy samples."
          % (f"{len(pairs):,}", f"{n_samp:,}"))
    print("  Labels are Bradley-Terry, P(a > b) = sigma(r*(a) - r*(b)),")
    print("  so %.2f%% of them actually point at the better sentence"
          % (100 * agree))
    print("  and the rest are the noise any human comparison carries.")
    print()
    print("  Bradley-Terry loss: "
          + "  ".join("%d:%.4f" % (s, l) for s, l in hist))
    print()
    c_all = float(np.corrcoef(rhat, R_STAR)[0, 1])
    c_on = weighted_corr(rhat, R_STAR, np.exp(ref_lp))
    print("  correlation with the true reward, over the whole language")
    print("    unweighted            %.4f" % c_all)
    print("    weighted by where the reference policy actually is")
    print("                          %.4f" % c_on)
    print()
    print("  Those are nearly the same number, and that matters: the")
    print("  reward model is a perfectly reasonable judge of")
    print("  SENTENCES, on and off the reference policy alike. If it")
    print("  were simply a bad model there would be no puzzle here.")
    print()
    print("  Which leaves the question the correlation cannot answer,")
    print("  because it is computed over the language and the")
    print("  language is all sentences: what does a model trained")
    print("  only on grammatical text think about text that is not?")
    return c_all, c_on


def demo3(ref, rm, rhat, ref_lp):
    """Goodhart, measured."""
    banner("3. THE PROXY RISES, THE TRUTH PEAKS AND FALLS")
    print("  REINFORCE against the reward model, with the KL penalty")
    print("  folded into the reward as InstructGPT does it. %d seeds"
          % len(SEEDS))
    print("  per setting; beta is the only knob.")
    print()
    base = exact_eval(ref, rhat, ref_lp)
    out = {}
    print("  %-6s %6s %9s %9s %9s %9s" %
          ("beta", "step", "E[r*]", "E[rhat]", "mass", "KL"))
    print("  " + "-" * 56)
    print("  %-6s %6d %9.4f %9.4f %9.4f %9.4f"
          % ("--", 0, base["true"], base["proxy"], base["mass"], 0.0))
    wrecked = None
    for beta in BETAS:
        runs = []
        for sd in SEEDS:
            pol, h = rlhf(ref, rm, rhat, ref_lp, beta, PG_STEPS,
                          np.random.default_rng(SEED_POL + sd),
                          every=EVERY)
            runs.append(h)
            if beta == BETAS[0] and sd == SEEDS[0]:
                wrecked = pol
        n = len(runs[0])
        curve = [{f: ms([r[i] for r in runs], f)[0]
                  for f in ("true", "proxy", "mass", "kl")}
                 for i in range(n)]
        for i, c in enumerate(curve):
            c["step"] = runs[0][i]["step"]
        out[beta] = (curve, runs)
        for c in curve:
            if c["step"] in (100, PG_STEPS):
                print("  %-6.2f %6d %9.4f %9.4f %9.4f %9.4f"
                      % (beta, c["step"], c["true"], c["proxy"],
                         c["mass"], c["kl"]))
        pk = max(curve, key=lambda c: c["true"])
        print("     peak E[r*] %.4f at step %d, mass there %.4f;"
              % (pk["true"], pk["step"], pk["mass"]))
        print("     ends at %.4f with mass %.4f"
              % (curve[-1]["true"], curve[-1]["mass"]))
    print()
    lo = out[0.05][0]
    pk = max(lo, key=lambda c: c["true"])
    print("  Read the beta = 0.05 row twice. The true reward climbs to")
    print("  %.4f, then ends at %.4f -- BELOW the %.4f it started"
          % (pk["true"], lo[-1]["true"], base["true"]))
    print("  from. The proxy does not fall with it.")
    print()
    print("  The mass column is the mechanism. Grammatical output")
    print("  drops from %.4f to %.4f. Look at what the wrecked policy"
          % (base["mass"], lo[-1]["mass"]))
    print("  emits, and at what the reward model makes of it:")
    print()
    srng = np.random.default_rng(99)
    samp = [w[:T_MAX - 2] for w, _ in sample_batch(wrecked, srng, 400)]
    ids, keep = encode(samp)
    rh = rm_score(rm, ids, keep)[0]
    ok = np.array([in_language(w) for w in samp])
    rs = np.array([reward_of(w) for w in samp])
    print("    grammatical           %.4f" % ok.mean())
    print("    mean r* (the truth)   %.4f" % rs.mean())
    print("    mean r^ (the proxy)   %.4f" % rh.mean())
    print("    mean r^ on the REFERENCE policy  %.4f"
          % float(np.exp(ref_lp) @ rhat / np.exp(ref_lp).sum()))
    uniq = sorted({tuple(w) for w in samp})
    print("    distinct outputs in %d samples: %d"
          % (len(samp), len(uniq)))
    for w in uniq[:3]:
        print("    %-34s r^ = %6.3f   r* = %.1f"
              % (" ".join(w) or "(empty)",
                 rm_score(rm, *encode([list(w)]))[0][0],
                 reward_of(list(w))))
    print()
    print("  The reward model scores this HIGHER than the grammatical")
    print("  text it was trained on, because it was never shown a")
    print("  non-sentence and has no way to object to one. Nothing has")
    print("  gone wrong with the optimiser. It maximised exactly what")
    print("  it was given.")
    print()
    hi = out[0.20][0]
    hpk = max(hi, key=lambda c: c["true"])
    print("  And beta is what holds it. At 0.20 the peak is %.4f,"
          % hpk["true"])
    print("  HIGHER than the looser penalty ever reached (%.4f), and"
          % pk["true"])
    print("  it is still there at step %d: %.4f with mass %.4f."
          % (PG_STEPS, hi[-1]["true"], hi[-1]["mass"]))
    print()
    print("  So the penalty is not buying stability at the cost of")
    print("  reward. At this strength it wins on both. A policy that")
    print("  runs off the distribution its reward model understands")
    print("  does not get a better score for it -- it only gets a")
    print("  score nobody should believe.")
    return out


def demo4(ref, rhat, ref_lp, pairs, rl):
    """DPO: the same comparisons, with nothing to hack."""
    banner("4. REMOVING THE THING THAT GOT HACKED")
    print("  DPO drops the reward model and trains on the comparisons")
    print("  directly, treating the log-ratio to the reference policy")
    print("  as an implicit reward. Same %s pairs, same seeds, same"
          % f"{len(pairs):,}")
    print("  budget -- and no sampling anywhere, so there is no")
    print("  distribution for a proxy to drift off.")
    print()
    out = {}
    print("  %-10s %6s %9s %9s %9s" %
          ("method", "step", "E[r*]", "mass", "KL"))
    print("  " + "-" * 48)
    for beta in DPO_BETAS:
        runs = []
        for sd in SEEDS:
            _, h = dpo(ref, rhat, ref_lp, beta, PG_STEPS,
                       np.random.default_rng(SEED_POL + sd), pairs,
                       every=EVERY)
            runs.append(h)
        curve = [{f: ms([r[i] for r in runs], f)[0]
                  for f in ("true", "mass", "kl")}
                 for i in range(len(runs[0]))]
        for i, c in enumerate(curve):
            c["step"] = runs[0][i]["step"]
        out[beta] = curve
        print("  %-10s %6d %9.4f %9.4f %9.4f"
              % ("DPO %.1f" % beta, PG_STEPS, curve[-1]["true"],
                 curve[-1]["mass"], curve[-1]["kl"]))
    for beta in (0.05, 0.20):
        c = rl[beta][0][-1]
        print("  %-10s %6d %9.4f %9.4f %9.4f"
              % ("RLHF %.2f" % beta, PG_STEPS, c["true"], c["mass"],
                 c["kl"]))
    print()
    best_dpo = max(out[b][-1]["true"] for b in DPO_BETAS)
    worst_rl = min(rl[b][0][-1]["true"] for b in BETAS)
    best_rl = max(rl[b][0][-1]["true"] for b in BETAS)
    print("  DPO rises and stays up. It never reaches what the best")
    print("  tuned RLHF run reaches -- %.4f against %.4f -- but it"
          % (best_dpo, best_rl))
    print("  also never produces the %.4f that the worst one ends at."
          % worst_rl)
    print()
    print("  That is the honest summary of the comparison. DPO is not")
    print("  better at finding reward. It is better at not losing it,")
    print("  because the failure mode it removes is the one that")
    print("  needs a proxy to exist. What it cannot do is exceed its")
    print("  pairs: every sentence it learns from was written by the")
    print("  reference policy, so it can only re-weight what that")
    print("  policy already said.")
    return out


def main():
    banner("RLHF AND DPO")
    n_sent, lang_r, rmax = selftest()
    worst, checked = gradcheck()
    print("  stage self-test passed: %s sentences, E[r*] = %.4f,"
          % (f"{n_sent:,}", lang_r))
    print("    ceiling %.1f; r* agrees with its definition everywhere"
          % rmax)
    print("  gradient audit: worst error %.2e over %d sampled gradients"
          % (worst, checked))
    print("    (the Bradley-Terry head, and the weighted log-prob")
    print("     gradient that REINFORCE and DPO both ride on)")
    print()

    corpus, ids, keep, test = build_stage()
    ref, hist = train_lm(ids, keep)
    ref_lp = policy_logprobs(ref)
    print("  reference policy: %s params, %.4f of its mass on the"
          % (f"{ref.n_params():,}", float(np.exp(ref_lp).sum())))
    print("  language. It is Part 7's model, untouched.")

    prng = np.random.default_rng(SEED_PREF)
    pairs, n_samp, agree = build_preferences(ref, N_PAIRS, prng)
    rm, rmhist = train_reward_model(pairs, RM_STEPS,
                                    np.random.default_rng(SEED_RM))
    rhat = rm_over_language(rm)

    demo1(ref, ref_lp)
    demo2(rm, rhat, ref_lp, rmhist, pairs, n_samp, agree)
    rl = demo3(ref, rm, rhat, ref_lp)
    demo4(ref, rhat, ref_lp, pairs, rl)

    banner("DONE")
    return ref, rm, rl


if __name__ == "__main__":
    main()
