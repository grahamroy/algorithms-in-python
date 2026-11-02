"""
nlg.py --- companion code for "Natural Language Generation"
(Natural Language Processing, Part 7).

Part 5 built a decoder and deliberately stopped at temperature. Top-k,
nucleus sampling and beam search were left for this article, along with
the question that makes generation different from everything before it
in this track: when there is no single right answer, what exactly are
you measuring?

Every earlier article compared a prediction against a label. Here the
output is a sentence, the set of acceptable sentences is large, and the
model's own output distribution is the thing under test. That normally
forces evaluation onto samples and proxies.

It does not have to here. Part 1's micro-language is ENUMERABLE --
31,740 sentences whose exact probabilities sum to one -- so this file
can do something no NLG paper can:

    a decoding strategy is a distribution, and we can compute it.

Temperature, top-k and nucleus sampling are all deterministic rewrites
of the next-token distribution, so the probability a strategy assigns to
every sentence in the language follows in closed form:

    q(s) = prod_t q_trunc(s_t | s_<t)

From that one object comes the exact grammatical rate, the exact number
of sentences the strategy can still produce at all, its entropy, and its
total-variation distance from the truth -- with no sampling error
anywhere. The quality-diversity frontier that the literature estimates
is, in this universe, a thing you can simply evaluate.

What that buys, measured here:
  1. The frontier itself. Truncation buys grammaticality with
     coverage, and the exchange rate is brutal: top-k 2 is 99.97%
     grammatical and can produce 16 of the 31,740 sentences.
  2. Nucleus beats top-k on both axes at once -- the published
     argument for nucleus sampling, reproduced as arithmetic rather
     than as a human preference study.
  3. The likelihood trap. Beam search maximises sequence probability
     and that is precisely what is wrong with it: the most probable
     sentence is not a typical sentence, and the gap is measurable
     against the language's entropy.
  4. The metric problem. The set of valid continuations of a prompt is
     enumerable, so BLEU can be audited rather than argued about. A
     perfect generator scores 0.4899 against one reference, and --
     worse -- a sentence broken by a single token matches or beats a
     valid alternative 55.07% of the time.
  5. Sampled diversity metrics go blind exactly where the decision is
     made: at 100 samples, two strategies 1.6x apart in effective
     sentence count both score 0.9800 distinct.

The language, the oracle and the transformer are Part 5's, unchanged:
multi-head causal self-attention, LayerNorm, learned positions, a shared
output head, all hand-differentiated and audited against central
differences at startup. Seeded and deterministic.
Dependencies: numpy.  Runs in about seventy-five seconds.
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


def completions(prompt):
    """Every sentence of the language that begins with this prompt.

    The reference SET, in full.  This is what makes the evaluation
    section a measurement instead of an argument.
    """
    p = tuple(prompt)
    return [list(s) for s in SENTS if s[:len(p)] == p]


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


def sample_sentence(model, rng, kind="temp", param=1.0, prompt=()):
    """Sample left to right until [EOS] or the length cap."""
    ids = np.zeros((1, T_MAX), dtype=int)
    ids[0, 0] = BOS
    n = 1
    words = []
    for w in prompt:
        ids[0, n] = W2I[w]
        words.append(w)
        n += 1
    while n < T_MAX:
        keep = np.zeros((1, T_MAX), dtype=bool)
        keep[0, :n] = True
        logits, _ = model.head_logits(ids, keep)
        row = logits[0, n - 1][None, :]
        if kind == "temp" and param <= 0:
            nxt = int(row[0].argmax())
        else:
            q = trunc_probs(row, kind, param)[0]
            nxt = int(rng.choice(V, p=q / q.sum()))
        if nxt == EOS:
            return words, True
        words.append(VOCAB[nxt])
        ids[0, n] = nxt
        n += 1
    return words, False


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


def sample_quality(model, rng, n, kind="temp", param=1.0):
    """Sample n sentences and ask the oracle what they are."""
    reasons = defaultdict(int)
    seen = set()
    for w, done in sample_batch(model, rng, n, kind=kind, param=param):
        reasons[diagnose(w) if done else "no [EOS]"] += 1
        seen.add(tuple(w))
    return dict(grammatical=reasons["ok"] / n, distinct=len(seen) / n,
                reasons=dict(reasons))


# =======================================================================
# The exact distribution a strategy induces over the language.
#
# This is the article.  q(s) = prod_t q_trunc(s_t | s_<t) for every one
# of the 31,740 sentences, so quality and diversity stop being sampled
# estimates and become arithmetic.  One forward pass over the language
# serves every strategy at once.
# =======================================================================

def strategy_on_language(model, strategies, batch=2000):
    logq = {nm: np.zeros(len(SENTS)) for nm in strategies}
    for i in range(0, len(SENTS), batch):
        chunk = [list(s) for s in SENTS[i:i + batch]]
        bids, bkeep = encode(chunk)
        logits, _ = model.head_logits(bids, bkeep)
        _, tgt, sel = lm_targets(bids, bkeep)
        N, T, Vv = logits.shape
        flat = logits.reshape(N * T, Vv)
        rows, cols = np.indices((N, T))
        for nm, (kind, param) in strategies.items():
            q = trunc_probs(flat, kind, param).reshape(N, T, Vv)
            with np.errstate(divide="ignore"):
                lp = np.where(sel, np.log(q[rows, cols, tgt]), 0.0)
            logq[nm][i:i + batch] = lp.sum(1)
    out = {}
    for nm in strategies:
        lq = logq[nm]
        qs = np.exp(lq)
        reach = lq > -np.inf
        mass = float(qs.sum())
        qn = qs / mass if mass > 0 else qs
        nz = qn > 0
        H = float(-(qn[nz] * np.log(qn[nz])).sum())
        out[nm] = dict(
            mass=mass,                            # exact grammatical rate
            reach=int(reach.sum()),               # sentences still possible
            reach_mass=float(P_LANG[reach].sum()),
            H=H, eff=float(np.exp(H)),
            tv=float(0.5 * (np.abs(P_LANG - qs).sum() + (1.0 - mass))),
        )
    return out


def model_on_language(model, batch=4000):
    """The model's own distribution against the language's, exactly."""
    lp = np.empty(len(SENTS))
    for i in range(0, len(SENTS), batch):
        chunk = [list(s) for s in SENTS[i:i + batch]]
        lp[i:i + batch] = lm_logprob(model, chunk)
    pM = np.exp(lp)
    mass = float(pM.sum())
    return dict(mass=mass, kl=float((P_LANG * (np.log(P_LANG) - lp)).sum()),
                xent=float(-(P_LANG * lp).sum()),
                tv=float(0.5 * np.abs(P_LANG - pM).sum()))


# =======================================================================
# Beam search.  Not a sampler: a search for the most probable sequence.
# That is the point and the problem.
# =======================================================================

def beam_search(model, B, alpha=0.0, prompt=(), max_len=T_MAX - 1):
    """Every completed beam, best first.  score = logprob / len**alpha."""
    start = tuple(W2I[w] for w in prompt)
    live = [(0.0, start)]
    done = []
    for _ in range(max_len - len(start) + 1):
        if not live:
            break
        bids = np.zeros((len(live), T_MAX), dtype=int)
        bkeep = np.zeros((len(live), T_MAX), dtype=bool)
        bids[:, 0] = BOS
        bkeep[:, 0] = True
        for i, (_, seq) in enumerate(live):
            for j, tid in enumerate(seq):
                bids[i, j + 1] = tid
                bkeep[i, j + 1] = True
        logits, _ = model.head_logits(bids, bkeep)
        cand = []
        for i, (lp, seq) in enumerate(live):
            lg = logits[i, len(seq)].astype(np.float64)
            lg -= lg.max()
            lgp = lg - np.log(np.exp(lg).sum())
            for tid in np.argsort(-lgp)[:B + 1]:
                tid = int(tid)
                nlp = lp + float(lgp[tid])
                if tid == EOS:
                    done.append((nlp / max(len(seq), 1) ** alpha, nlp, seq))
                elif len(seq) + 1 <= max_len:
                    cand.append((nlp, seq + (tid,)))
        cand.sort(key=lambda x: -x[0])
        live = cand[:B]
    done.sort(key=lambda x: -x[0])
    return [(sc, lp, [VOCAB[t] for t in seq]) for sc, lp, seq in done]


# =======================================================================
# Evaluation.  BLEU, and the reference set it is standing in for.
# =======================================================================

def ngrams(w, n):
    return Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))


def bleu(hyp, refs, N=4):
    """BLEU with the usual clipping and brevity penalty.

    Orders 2 and up are add-1 smoothed, which is standard for sentences
    this short; without it a single missing 4-gram zeroes the score and
    the comparison below would measure the smoothing, not the metric.
    """
    if not hyp:
        return 0.0
    logs = []
    for n in range(1, N + 1):
        h = ngrams(hyp, n)
        if not h:
            logs.append(np.log(1e-9))
            continue
        best = Counter()
        for r in refs:
            rn = ngrams(r, n)
            for g in h:
                best[g] = max(best[g], rn.get(g, 0))
        clip = sum(min(c, best[g]) for g, c in h.items())
        tot = sum(h.values())
        num, den = (clip, tot) if n == 1 else (clip + 1, tot + 1)
        logs.append(np.log(num / den) if num > 0 else np.log(1e-9))
    rl = min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
    bp = 1.0 if len(hyp) > rl else np.exp(1 - rl / max(len(hyp), 1))
    return float(bp * np.exp(np.mean(logs)))


def break_one(sent, how, rng):
    """Break a valid sentence by exactly ONE token.

    The three ways match diagnose()'s categories, so the oracle confirms
    each result really is outside the language.
    """
    w = list(sent)
    subj_i = len(w) - 4
    if how == "verb number":
        vi = subj_i + 1
        for (vs, vp) in TOPICS[TOPIC_OF[w[subj_i]]]["verbs"]:
            if w[vi] == vs:
                w[vi] = vp
                return w
            if w[vi] == vp:
                w[vi] = vs
                return w
        return None
    if how == "possessive":
        pi = len(w) - 2
        alt = sorted(p for p in POSSESSIVES if p != w[pi])
        w[pi] = alt[int(rng.integers(0, len(alt)))]
        return w
    if how == "topic mix":
        topic = TOPIC_OF[w[subj_i]]
        other = sorted(n for n in SG_NOUNS if TOPIC_OF[n] != topic)
        w[-1] = other[int(rng.integers(0, len(other)))]
        return w
    return None


# -----------------------------------------------------------------------
# Gradient audit and oracle self-test.
# -----------------------------------------------------------------------

def gradcheck(seed=0, n_per=4):
    rng = np.random.default_rng(seed)
    sents = gen_corpus(5, rng)
    ids, keep = encode([s["words"] for s in sents])
    _, tgt, sel = lm_targets(ids, keep)
    net = Transformer(d=16, heads=2, layers=2, seed=1)
    g = net.head_grads(ids, keep, tgt, sel)
    worst, checked = 0.0, 0
    for k, p in net.P.items():
        flat, gflat = p.reshape(-1), g[k].reshape(-1)
        for ix in rng.choice(len(flat), min(n_per, len(flat)),
                             replace=False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp = net.head_loss(ids, keep, tgt, sel)
            flat[ix] = old - 1e-6
            lm = net.head_loss(ids, keep, tgt, sel)
            flat[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
            checked += 1
    return worst, checked


def selftest():
    """The oracle must agree with the generator it claims to describe,
    and the truncation rules must agree with their own definitions."""
    tot = sum(SENT_P.values())
    assert abs(tot - 1.0) < 1e-9, tot
    for s in SENT_P:
        assert diagnose(list(s)) == "ok", s
    rng = np.random.default_rng(5)
    real = [w for w in VOCAB if not w.startswith("[")]
    for _ in range(3000):
        n = int(rng.integers(3, 8))
        w = [real[i] for i in rng.integers(0, len(real), n)]
        assert (diagnose(w) == "ok") == in_language(w), w
    for _ in range(3000):
        w = list(gen_sentence(rng)["words"])
        w[int(rng.integers(0, len(w)))] = \
            real[int(rng.integers(0, len(real)))]
        assert (diagnose(w) == "ok") == in_language(w), w
    sents = [gen_sentence(rng)["words"] for _ in range(20000)]
    nll = np.array([sum(oracle_nll(w)) for w in sents])
    H = language_entropy()
    se = nll.std() / np.sqrt(len(nll))
    assert abs(nll.mean() - H) < 5 * se, (nll.mean(), H, se)

    # The truncation rules, against their definitions, on random logits.
    lg = rng.standard_normal((200, V)) * 3.0
    for k in (1, 3, 10, V):
        q = trunc_probs(lg, "topk", k)
        assert np.allclose(q.sum(1), 1.0)
        assert ((q > 0).sum(1) == min(k, V)).all()
    base = trunc_probs(lg, "temp", 1.0)
    for pp in (0.5, 0.9, 0.99):
        q = trunc_probs(lg, "topp", pp)
        assert np.allclose(q.sum(1), 1.0)
        kept = q > 0
        # the kept set must reach pp, and dropping its least likely
        # member must fall short of it -- that is "smallest prefix"
        m = (base * kept).sum(1)
        least = np.where(kept, base, np.inf).min(1)
        assert (m >= pp - 1e-12).all()
        assert (m - least < pp + 1e-12).all()
    assert np.allclose(trunc_probs(lg, "temp", 1.0),
                       trunc_probs(lg, "topp", 1.0))

    # BLEU: identity scores 1, and the reference set really is complete.
    for s in (SENTS[0], SENTS[777]):
        assert abs(bleu(list(s), [list(s)]) - 1.0) < 1e-9
    cm = completions(["the", "king"])
    assert all(in_language(c) for c in cm)
    assert len(cm) == sum(1 for s in SENTS if s[:2] == ("the", "king"))
    return len(SENT_P), H, float(nll.mean()), float(se)


# -----------------------------------------------------------------------
# Frozen configuration.  The table and header generators import these so
# their numbers are the ones printed below.
# -----------------------------------------------------------------------

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
# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

SAMPLE_CHECK = ("temperature 1.0", "top-k 10", "nucleus 0.90")
PROXY_NS = (100, 1000, 4000)
PROXY_STRATS = ("temperature 1.0", "nucleus 0.90", "top-k 10", "top-k 2")


def demo1(lm):
    """The quality-diversity frontier, computed rather than sampled."""
    banner("1. EVERY DECODING STRATEGY IS A DISTRIBUTION")
    print("  A sampler that rewrites the next-token distribution and then")
    print("  draws from it defines a distribution over whole sentences:")
    print()
    print("      q(s) = prod_t q_trunc(s_t | s_<t)")
    print()
    print("  The language is enumerable, so that product can be evaluated")
    print("  for all %s sentences instead of estimated from samples."
          % f"{len(SENT_P):,}")
    print("  Nothing below is a sample statistic.")
    print()

    res = strategy_on_language(lm, STRATEGIES)
    H = language_entropy()
    print("  %-17s %7s %8s %8s %9s %7s"
          % ("strategy", "gramm", "can say", "of lang", "eff sent", "TV"))
    print("  " + "-" * 60)
    print("  %-17s %7.4f %8d %8.4f %9.1f %7.4f"
          % ("the language", 1.0, len(SENT_P), 1.0, np.exp(H), 0.0))
    for nm in STRATEGIES:
        r = res[nm]
        print("  %-17s %7.4f %8d %8.4f %9.1f %7.4f"
              % (nm, r["mass"], r["reach"], r["reach_mass"], r["eff"],
                 r["tv"]))
    print()
    print("  gramm   = exact probability the output is in the language")
    print("  can say = sentences the strategy can ever produce")
    print("  of lang = the share of the language's mass it can reach")
    print("  eff sent= exp(entropy): effective number of sentences")
    print("  TV      = total-variation distance from the true language")
    print()

    k2, p99 = res["top-k 2"], res["nucleus 0.99"]
    t10 = res["temperature 1.0"]
    print("  Read the top-k 2 row again. It is %.2f%% grammatical and it"
          % (100 * k2["mass"]))
    print("  can produce %d of the %s sentences -- %.2f%% of the"
          % (k2["reach"], f"{len(SENT_P):,}", 100 * k2["reach_mass"]))
    print("  language's mass. The grammar is almost perfect because")
    print("  almost nothing is left to say.")
    print()
    print("  Truncation does not make a model better. It moves it AWAY")
    print("  from the truth in exchange for staying inside the grammar:")
    print("  untruncated TV is %.4f, and every truncation raises it"
          % t10["tv"])
    print("  except the gentlest (nucleus 0.99 at %.4f)." % p99["tv"])
    print()

    k10, p90 = res["top-k 10"], res["nucleus 0.90"]
    print("  Nucleus against top-k, same model, no preference study:")
    print("    top-k 10     gramm %.4f  eff sent %7.1f  TV %.4f"
          % (k10["mass"], k10["eff"], k10["tv"]))
    print("    nucleus 0.90 gramm %.4f  eff sent %7.1f  TV %.4f"
          % (p90["mass"], p90["eff"], p90["tv"]))
    print("  Nucleus wins on all three at once. A fixed k has to be set")
    print("  for the model's flattest prediction; a fixed mass adapts,")
    print("  taking two words where the next token is nearly determined")
    print("  and many where it is not.")
    print()

    print("  Consistency: the exact rate against %s samples of each."
          % f"{N_SAMPLES:,}")
    print("  %-17s %9s %9s %9s" % ("strategy", "exact", "sampled", "s.e."))
    for nm in SAMPLE_CHECK:
        kind, param = STRATEGIES[nm]
        s = sample_quality(lm, np.random.default_rng(SEED_SAMPLE),
                           N_SAMPLES, kind=kind, param=param)
        m = res[nm]["mass"]
        se = np.sqrt(max(m * (1 - m), 1e-12) / N_SAMPLES)
        print("  %-17s %9.4f %9.4f %9.4f"
              % (nm, m, s["grammatical"], se))
    return res


def demo2(lm):
    """Beam search: the most probable sentence is not a typical one."""
    banner("2. THE LIKELIHOOD TRAP")
    print("  Beam search is not a sampler. It searches for the sequence")
    print("  with the highest total log-probability, and it is very good")
    print("  at its job. That is the problem.")
    print()

    mode_s, mode_p = max(SENT_P.items(), key=lambda kv: kv[1])
    H = language_entropy()
    print("  %-5s %-44s %9s %6s %6s"
          % ("beam", "sentence returned", "-log p", "words", "ok"))
    print("  " + "-" * 74)
    rows = []
    for B in BEAMS:
        out = beam_search(lm, B)
        _, lp, w = out[0]
        rows.append((B, w, -lp, len(w), diagnose(w)))
        print("  %-5d %-44s %9.4f %6d %6s"
              % (B, " ".join(w), -lp, len(w), diagnose(w)))
    print()

    samp = sample_batch(lm, np.random.default_rng(SEED_SAMPLE), N_SAMPLES)
    good = [w for w, done in samp if done and len(w) in (5, 6)]
    slp = lm_logprob(lm, good)
    print("  A sentence SAMPLED from the same model: -log p = %.4f"
          % -slp.mean())
    print("  The language's own entropy:             %.4f" % H)
    print("  The language's single likeliest sentence:")
    print("    %-44s -log p = %.4f" % (" ".join(mode_s), -np.log(mode_p)))
    print()
    best = min(r[2] for r in rows)
    print("  Beam search returns sentences around %.4f nats, roughly" % best)
    print("  %.1f nats more probable than a typical one -- a factor of"
          % (-slp.mean() - best))
    print("  about %.0f in probability. A typical sentence is not the"
          % np.exp(-slp.mean() - best))
    print("  likeliest one, and in a distribution over %s sentences"
          % f"{len(SENT_P):,}")
    print("  the mode is a vanishingly small and unrepresentative place.")
    print()
    print("  Note what the larger beams do: they find SHORTER, blander")
    print("  sentences, because every extra word multiplies in another")
    print("  probability below one. Length normalisation is the usual")
    print("  patch, and it is a patch:")
    for B in (5, 25):
        _, lp, w = beam_search(lm, B, alpha=1.0)[0]
        print("    beam %-3d alpha 1.0  %-40s %6d words"
              % (B, " ".join(w), len(w)))
    print()
    out25 = beam_search(lm, max(BEAMS))
    uniq = len({tuple(w) for _, _, w in out25})
    print("  Beam %d completes %d sequences, %d distinct. Against the"
          % (max(BEAMS), len(out25), uniq))
    print("  %s the language holds, that is the whole catalogue:"
          % f"{len(SENT_P):,}")
    for _, lp, w in out25[:4]:
        print("    %-44s -log p = %.4f" % (" ".join(w), -lp))
    print()
    print("  Beam search is right for translation, where one input has")
    print("  roughly one correct output and probability is the thing you")
    print("  want. It is wrong for open-ended generation, where the goal")
    print("  is a sample and the mode is not one.")
    return rows, float(-slp.mean()), float(-np.log(mode_p))


def demo3(lm):
    """BLEU, audited against a reference set that can be enumerated."""
    banner("3. SCORING OUTPUT WITH NO SINGLE RIGHT ANSWER")
    refs_all = completions(PROMPT)
    print("  Prompt: %r" % " ".join(PROMPT))
    print("  Valid completions in the language: %d" % len(refs_all))
    for r in refs_all[:3]:
        print("    " + " ".join(r))
    print("    ... and %d more, every one of them correct."
          % (len(refs_all) - 3))
    print()
    print("  A real benchmark shows the scorer one of these, or a few.")
    print("  Here the whole set is available, so the metric itself can")
    print("  be put on trial.")
    print()

    rng = np.random.default_rng(SEED_BLEU)
    pairs = []
    for s in refs_all:
        for how in BREAKS:
            b = break_one(s, how, rng)
            if b is not None and diagnose(b) != "ok":
                pairs.append((s, b, how))
    assert pairs, "no broken variants built"
    print("  Each valid sentence also gets a twin broken by exactly ONE")
    print("  token (%d pairs), confirmed outside the language by the"
          % len(pairs))
    print("  same oracle that defines it. Then both are scored against")
    print("  k references drawn from the OTHER valid completions.")
    print()
    print("  %-12s %9s %9s %11s" %
          ("references", "valid", "broken", "broken wins"))
    print("  " + "-" * 45)
    table = {}
    for k in REF_COUNTS:
        va, ba, win = [], [], []
        for s, b, _ in pairs:
            pool = [r for r in refs_all if r != s]
            pick = [pool[i] for i in
                    rng.choice(len(pool), k, replace=False)]
            x, y = bleu(s, pick), bleu(b, pick)
            va.append(x)
            ba.append(y)
            win.append(y >= x)
        table[k] = (float(np.mean(va)), float(np.mean(ba)),
                    float(np.mean(win)))
        print("  %-12d %9.4f %9.4f %11.4f" % ((k,) + table[k]))
    print()

    one = table[REF_COUNTS[0]]
    print("  Against one reference a PERFECT generator scores %.4f."
          % one[0])
    print("  It is not wrong; it is differently right, and BLEU has no")
    print("  way to tell those apart.")
    print()
    print("  The second column is the serious one. A sentence that the")
    print("  oracle rejects scores %.4f, and it matches or beats the"
          % one[1])
    print("  valid alternative %.2f%% of the time. On one reference, BLEU"
          % (100 * one[2]))
    print("  ranks broken output above correct output more often than")
    print("  not.")
    print()
    print("  Which break it catches is the mechanism:")
    print("  %-14s %9s %9s %11s"
          % ("break", "valid", "broken", "broken wins"))
    print("  " + "-" * 47)
    per = {}
    for how in BREAKS:
        sub = [(s, b) for s, b, h in pairs if h == how]
        va, ba, win = [], [], []
        for s, b in sub:
            pool = [r for r in refs_all if r != s]
            pick = [pool[i] for i in rng.choice(len(pool), 1)]
            x, y = bleu(s, pick), bleu(b, pick)
            va.append(x)
            ba.append(y)
            win.append(y >= x)
        per[how] = (float(np.mean(va)), float(np.mean(ba)),
                    float(np.mean(win)))
        print("  %-14s %9.4f %9.4f %11.4f" % ((how,) + per[how]))
    print()
    print("  BLEU catches the possessive every time and the topic")
    print("  almost never. The reason is n-gram overlap and nothing")
    print("  else: 'his' -> 'her' destroys a bigram the reference")
    print("  shares, while swapping the object for a word from another")
    print("  topic replaces a word the reference did not have either.")
    print("  The metric is not measuring meaning. It is measuring")
    print("  string overlap, which is correlated with meaning until you")
    print("  optimise against it.")
    print()
    # One fixed reference, scanned deterministically, so the article
    # can quote the case rather than hope for it.
    erng = np.random.default_rng(SEED_BLEU)
    ref = [refs_all[int(erng.integers(0, len(refs_all)))]]
    ties, scored = [], 0
    for s, b, how in pairs:
        if s == ref[0]:
            continue
        scored += 1
        x, y = bleu(s, ref), bleu(b, ref)
        if abs(y - x) < 1e-12:
            ties.append((s, b, how, x))
    s, b, how, x = max(ties, key=lambda e: e[3])
    print("  Against one fixed reference, %d of the %d pairs score"
          % (len(ties), scored))
    print("  EXACTLY the same for the valid and the broken sentence.")
    print("  The highest-scoring such tie:")
    print("    reference : " + " ".join(ref[0]))
    print("    valid     : %-34s BLEU %.4f" % (" ".join(s), x))
    print("    broken    : %-34s BLEU %.4f  (%s)"
          % (" ".join(b), x, diagnose(b)))
    print()
    print("  Both differ from the reference in exactly one word, so the")
    print("  n-gram counts are identical and so are the scores. One of")
    print("  them is a sentence of the language and the other puts a sea")
    print("  noun in a royal sentence. BLEU is not failing here; it is")
    print("  doing precisely what it says it does, which is count")
    print("  overlapping word strings. The failure is in reading that")
    print("  count as a measure of whether the output is any good.")
    return refs_all, pairs, table, per, (ref[0], s, b, x, len(ties), scored)


def demo4(lm, res):
    """What a sample cannot see."""
    banner("4. WHY THIS HAD TO BE COMPUTED")
    print("  The usual diversity proxy is the share of samples that are")
    print("  distinct. It is not a property of the decoder: it is a")
    print("  property of how many samples you drew.")
    print()
    print("  %-17s %10s" % ("", "exact"))
    print("  %-17s %10s %s" % ("strategy", "eff sent",
                               " ".join("%9s" % ("n=%d" % n)
                                        for n in PROXY_NS)))
    print("  " + "-" * 62)
    for nm in PROXY_STRATS:
        kind, param = STRATEGIES[nm]
        cells = []
        for n in PROXY_NS:
            s = sample_quality(lm, np.random.default_rng(SEED_SAMPLE), n,
                               kind=kind, param=param)
            cells.append(s["distinct"])
        print("  %-17s %10.1f %s"
              % (nm, res[nm]["eff"],
                 " ".join("%9.4f" % c for c in cells)))
    print()
    print("  Every row falls as n grows, because distinctness is a")
    print("  birthday problem and not a measurement. The effective")
    print("  sentence count on the left does not move, because it is a")
    print("  property of the distribution the strategy defines.")
    print()
    print("  This is the whole argument for an enumerable universe. The")
    print("  quantities that decide a decoding strategy -- how much of")
    print("  the language it can still reach, how far it has moved from")
    print("  the truth -- are not estimable from any feasible number of")
    print("  samples. Here they are arithmetic.")


def main():
    banner("NATURAL LANGUAGE GENERATION")
    n_sent, H, mean_nll, se = selftest()
    worst, checked = gradcheck()
    print("  oracle self-test passed: %s sentences, mass 1, H = %.4f nats"
          % (f"{n_sent:,}", H))
    print("    sampled mean NLL %.4f (s.e. %.4f) agrees with H"
          % (mean_nll, se))
    print("  gradient audit: worst error %.2e over %d sampled gradients"
          % (worst, checked))
    print()

    corpus, ids, keep, test = build_stage()
    lm, hist = train_lm(ids, keep)
    print("  corpus %s sentences; decoder d=64, 2 layers, 4 heads, %s params"
          % (f"{CORPUS_N:,}", f"{lm.n_params():,}"))
    print("  pre-training loss: " + "  ".join("%d:%.3f" % (s, l)
                                              for s, l in hist))
    tok, sent, otok, osent = lm_eval(lm, test)
    print("  held-out %.4f nats/token against a floor of %.4f (+%.1f%%)"
          % (tok, otok, 100 * (tok / otok - 1)))
    mol = model_on_language(lm)
    print("  the model puts %.4f of its mass on the language, KL %.4f"
          % (mol["mass"], mol["kl"]))
    print()
    print("  That model is the fixed point for everything below. The")
    print("  only thing that changes from here is how it is DECODED.")

    res = demo1(lm)
    demo2(lm)
    demo3(lm)
    demo4(lm, res)

    banner("DONE")
    return lm


if __name__ == "__main__":
    main()
