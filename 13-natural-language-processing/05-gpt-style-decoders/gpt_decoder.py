"""
gpt_decoder.py --- companion code for "GPT-style Decoders"
(Natural Language Processing, Part 5).

Part 4 kept the encoder and threw the decoder away. This article does
the opposite. The causal mask comes back, and with it the objective
that needs no corruption at all: predict the next token.

The change is smaller than it sounds. In this file the encoder and the
decoder are the SAME trunk and the SAME output head, differing in two
things:

    causal=True                  one boolean on the attention mask
    (targets, sel)               what you hand the loss function

Masked language modelling hands it the original ids at 15% of the
positions. Causal language modelling hands it the ids shifted left at
EVERY position. One line of bookkeeping, six times the gradient.

What that buys, measured here:
  1. The flood: targets per forward pass, LM against MLM, same corpus.
  2. A calibrated loss. The micro-language is small enough to
     ENUMERATE, so its exact next-token distribution is computable --
     and a left-to-right model's loss can be compared against the
     irreducible entropy of the universe that produced the text.
     An encoder has no such factorisation, so it gets no such yardstick.
  3. Sample efficiency and one-pass scoring: the same acceptability
     task Part 4 used, scored by the LM in ONE forward pass against
     the encoder's O(n) masked passes -- plus the price the decoder
     pays on the slot whose evidence lies to its right.
  4. Generation, which an encoder cannot do at all. The oracle tells
     us exactly whether a sampled sentence is in the language, so
     "quality" here is a measurement and not an impression.

Everything is plain NumPy: multi-head causal self-attention, LayerNorm,
learned positions, the shared output head, hand-differentiated and
audited against central differences at startup. Seeded and
deterministic.
Dependencies: numpy.  Runs in about three and a half minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import defaultdict

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
# Acceptability task: break the agreement the micro-language enforces.
# -----------------------------------------------------------------------

def corrupt_agreement(sent, rng):
    """Return a copy whose subject/verb/possessive agreement is broken.

    Two ways, chosen at random: swap the verb's number, or replace the
    possessive with one that disagrees in number or gender. Both leave
    a sentence made entirely of in-vocabulary words -- only the
    agreement is wrong.
    """
    w = list(sent["words"])
    if rng.uniform() < 0.5:
        w[sent["verb"]] = (sent["verb_sg"] if sent["plural"]
                           else sent["verb_pl"])
    else:
        bad = {"their", "his", "her", "the"} - {w[sent["poss"]]}
        if sent["gender"] == "n":
            bad -= {"his", "her"}        # "the" vs "their": number only
        w[sent["poss"]] = sorted(bad)[rng.integers(0, len(bad))]
    return w


def make_acceptability(n, rng):
    """Balanced (sentence, label) pairs. label 1 = acceptable."""
    out = []
    for _ in range(n):
        s = gen_sentence(rng)
        if rng.uniform() < 0.5:
            out.append((s["words"], 1))
        else:
            out.append((corrupt_agreement(s, rng), 0))
    return out


# -----------------------------------------------------------------------
# Vocabulary and encoding.  Every sequence is [BOS] words... [EOS],
# padded to T_MAX.  A decoder must learn where a sentence ENDS, so
# [EOS] is a prediction target like any other.  The encoder baseline
# reads the identical format, with [BOS] playing BERT's [CLS].
# -----------------------------------------------------------------------

PAD, BOS, EOS, MASK = 0, 1, 2, 3
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
T_MAX = 8          # [BOS] + at most six words + [EOS]
STEPS = 1500       # pre-training steps per model


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
# The oracle.  The micro-language is small enough to ENUMERATE, so its
# exact distribution over sentences -- and hence its exact next-token
# distribution after any prefix -- is computable in closed form.  That
# turns the decoder's loss into a CALIBRATED quantity: there is a floor,
# we know what it is, and we can measure the gap.  An encoder's masked
# loss admits no such comparison, because masked prediction is not a
# factorisation of anything.
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
    """Exact -log P(next | prefix) at each decoder position, in nats.

    Position i of the input sequence predicts word i (0-indexed), and
    the position after the last word predicts [EOS].  Returns one
    value per prediction, so len(words) + 1 of them.
    """
    seq = tuple(words) + (EOS_W,)
    out = []
    for i in range(len(seq)):
        d = ORACLE.get(tuple(words[:i]))
        p = d.get(seq[i], 0.0) if d else 0.0
        out.append(np.inf if p <= 0 else float(-np.log(p)))
    return out


def oracle_entropy_after(prefix):
    """Exact conditional entropy of the next token, in nats."""
    d = ORACLE.get(tuple(prefix))
    if not d:
        return 0.0
    return float(-sum(p * np.log(p) for p in d.values() if p > 0))


def in_language(words):
    return tuple(words) in SENT_P


def diagnose(words):
    """Why this word list is not a sentence of the language, or 'ok'.

    Checked in the order a reader would notice: is it even made of
    words, is it the right shape, is it about one thing, does it agree.
    Asserted at startup to agree exactly with `in_language`.
    """
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
# One transformer, two readings.  Pre-LN blocks:
#     x = x + attn(LN(x));  x = x + mlp(LN(x))
# causal=True is GPT: position t attends to positions <= t, and the
# output head predicts token t+1.  causal=False is BERT: every position
# sees every other, and the same head predicts whatever the corruption
# hid.  The trunk, the head and the loss are shared; Part 4's encoder is
# this class with one boolean flipped.
# -----------------------------------------------------------------------

class Transformer:
    def __init__(self, d=64, heads=4, layers=2, d_ff=None, causal=True,
                 seed=0, lr=3e-3, n_cls=2, pool="last"):
        r = np.random.default_rng(seed)
        self.d, self.h, self.L = d, heads, layers
        self.dk = d // heads
        self.dff = d_ff or 4 * d
        self.causal = causal
        self.lr = lr
        self.pool = pool

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
        P["Wc"] = rnd(d, n_cls)             # classification head
        P["bc"] = np.zeros(n_cls)
        self.P = P
        self.reset_adam()

    def reset_adam(self):
        self.M = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.Vm = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def clone(self):
        """Independent copy with fresh optimiser state (for fine-tuning)."""
        new = Transformer.__new__(Transformer)
        new.__dict__.update({k: v for k, v in self.__dict__.items()
                             if k not in ("P", "M", "Vm", "t")})
        new.P = {k: v.copy() for k, v in self.P.items()}
        new.reset_adam()
        return new

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

    # ---- multi-head self-attention ----
    def attn_fwd(self, x, pre, keep):
        N, T, d = x.shape
        h, dk = self.h, self.dk
        q, k, v = np.split(x @ self.P[pre + "Wqkv"], 3, axis=-1)

        def sp(z):
            return z.reshape(N, T, h, dk).transpose(0, 2, 1, 3)

        q, k, v = sp(q), sp(k), sp(v)
        S = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dk)
        if self.causal:
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

    # ---- the output head over tokens, shared by both objectives ----
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

    # ---- classification head.  GPT reads the LAST token, BERT the first ----
    def pooled(self, H, keep):
        if self.pool == "last":
            return H[np.arange(len(H)), keep.sum(1) - 1]
        if self.pool == "first":
            return H[:, 0]
        w = keep / keep.sum(1, keepdims=True)
        return (H * w[:, :, None]).sum(1)

    def unpool(self, dp, H, keep):
        dH = np.zeros_like(H)
        if self.pool == "last":
            dH[np.arange(len(H)), keep.sum(1) - 1] = dp
        elif self.pool == "first":
            dH[:, 0] = dp
        else:
            w = keep / keep.sum(1, keepdims=True)
            dH += w[:, :, None] * dp[:, None, :]
        return dH

    def cls_logits(self, ids, keep):
        H, cache = self.trunk_fwd(ids, keep)
        return self.pooled(H, keep) @ self.P["Wc"] + self.P["bc"], cache

    def cls_loss(self, ids, keep, labels):
        logits, _ = self.cls_logits(ids, keep)
        Pr = softmax_last(logits)
        return float(-np.log(Pr[np.arange(len(labels)), labels]
                             + 1e-300).mean())

    def cls_grads(self, ids, keep, labels):
        logits, cache = self.cls_logits(ids, keep)
        Pr = softmax_last(logits)
        N = len(labels)
        dlog = Pr.copy()
        dlog[np.arange(N), labels] -= 1.0
        dlog /= N
        g = self.zero_grads()
        H = cache[4]
        g["Wc"] += self.pooled(H, keep).T @ dlog
        g["bc"] += dlog.sum(0)
        self.trunk_bwd(self.unpool(dlog @ self.P["Wc"].T, H, keep),
                       cache, g)
        return g

    def cls_accuracy(self, ids, keep, labels):
        logits, _ = self.cls_logits(ids, keep)
        return float((logits.argmax(-1) == labels).mean())

    # ---- Adam ----
    def step(self, g, only=None):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, p in self.P.items():
            if only is not None and k not in only:
                continue
            self.M[k] = b1 * self.M[k] + (1 - b1) * g[k]
            self.Vm[k] = b2 * self.Vm[k] + (1 - b2) * g[k] ** 2
            p -= self.lr * (self.M[k] / (1 - b1 ** self.t)) / (
                np.sqrt(self.Vm[k] / (1 - b2 ** self.t)) + eps)


# -----------------------------------------------------------------------
# The two objectives.  This is the entire difference between GPT and
# BERT in this file: which (inputs, targets, sel) triple is handed to
# the same head_loss.
# -----------------------------------------------------------------------

def lm_targets(ids, keep):
    """Shift left.  Position j predicts the token at j+1 -- every j.

    The input is not touched: a causal LM needs no corruption, because
    the mask already stops each position seeing its own answer.
    """
    tgt = np.zeros_like(ids)
    tgt[:, :-1] = ids[:, 1:]
    sel = np.zeros(ids.shape, dtype=bool)
    sel[:, :-1] = keep[:, 1:]
    return ids, tgt, sel


def mlm_targets(ids, keep, rng, rate=0.15, recipe="80/10/10"):
    """Part 4's recipe: corrupt 15% of word positions, predict them."""
    content = keep & (ids != EOS)
    content[:, 0] = False
    sel = (rng.random(ids.shape) < rate) & content
    for i in np.where(~sel.any(1))[0]:
        choices = np.where(content[i])[0]
        sel[i, choices[rng.integers(0, len(choices))]] = True
    x = ids.copy()
    if recipe == "mask":
        x[sel] = MASK
    else:
        u = rng.random(ids.shape)
        x[sel & (u < 0.8)] = MASK
        rs = sel & (u >= 0.8) & (u < 0.9)
        x[rs] = rng.integers(N_SPECIAL, V, rs.sum())
    return x, ids, sel


def pretrain(model, ids, keep, steps, rng, batch=64, objective="lm",
             log_every=None, probe_at=(), probe_fn=None):
    """Returns (loss history, probe history, mean targets per sequence)."""
    hist, probes, ntgt = [], [], []
    for s in range(steps):
        b = rng.integers(0, len(ids), batch)
        if objective == "lm":
            x, tgt, sel = lm_targets(ids[b], keep[b])
        else:
            x, tgt, sel = mlm_targets(ids[b], keep[b], rng)
        ntgt.append(sel.sum() / batch)
        g = model.head_grads(x, keep[b], tgt, sel)
        model.step(g)
        if log_every and (s % log_every == 0 or s == steps - 1):
            hist.append((s + 1, model.head_loss(x, keep[b], tgt, sel)))
        if (s + 1) in probe_at and probe_fn is not None:
            probes.append((s + 1, probe_fn(model)))
    return hist, probes, float(np.mean(ntgt))


# -----------------------------------------------------------------------
# Held-out evaluation against the oracle's floor.
# -----------------------------------------------------------------------

SLOT_NAMES = ("the", "adj", "subj", "verb", "poss", "obj", "eos")


def slot_positions(sent):
    """Which decoder position predicts which grammatical slot."""
    w = sent["words"]
    idx = {"the": 0, "subj": sent["subj"], "verb": sent["verb"],
           "poss": sent["poss"], "obj": sent["obj"], "eos": len(w)}
    if len(w) == 6:
        idx["adj"] = 1
    return idx


def lm_eval(model, sents):
    """Per-token and per-sentence NLL, and the same for the oracle."""
    ids, keep = encode([s["words"] for s in sents])
    _, tgt, sel = lm_targets(ids, keep)
    nll = model.token_nll(ids, keep, tgt, sel)
    per_sent = nll.sum(1)
    n_tok = sel.sum(1)
    orc = np.array([sum(oracle_nll(s["words"])) for s in sents])
    return (float(per_sent.sum() / n_tok.sum()), float(per_sent.mean()),
            float(orc.sum() / n_tok.sum()), float(orc.mean()))


def slot_nll(model, sents):
    """Model and oracle NLL at each grammatical slot, in nats."""
    ids, keep = encode([s["words"] for s in sents])
    _, tgt, sel = lm_targets(ids, keep)
    nll = model.token_nll(ids, keep, tgt, sel)
    acc = {n: ([], []) for n in SLOT_NAMES}
    for r, s in enumerate(sents):
        onl = oracle_nll(s["words"])
        for n, j in slot_positions(s).items():
            acc[n][0].append(nll[r, j])
            acc[n][1].append(onl[j])
    return {n: (float(np.mean(a)), float(np.mean(b)), len(a))
            for n, (a, b) in acc.items() if a}


def special_mass(model, sents):
    """Probability the LM wastes on PAD/BOS/MASK as a next token."""
    ids, keep = encode([s["words"] for s in sents])
    logits, _ = model.head_logits(ids, keep)
    _, _, sel = lm_targets(ids, keep)
    Pr = softmax_last(logits)
    m = Pr[:, :, [PAD, BOS, MASK]].sum(-1)
    return float(m[sel].mean()), float(m[sel].max())


# -----------------------------------------------------------------------
# Scoring a sentence.  The decoder's probabilities factorise, so one
# forward pass gives the whole sentence.  Part 4's encoder needed one
# masked pass per position.
# -----------------------------------------------------------------------

def lm_logprob(model, batch_words):
    """log P(sentence) for a batch, in ONE forward pass."""
    ids, keep = encode(batch_words)
    _, tgt, sel = lm_targets(ids, keep)
    return -model.token_nll(ids, keep, tgt, sel).sum(1)


LM_PASSES = 1


def mlm_pseudo_logprob(model, batch_words):
    """Part 4's pseudo-log-likelihood: one masked pass per position."""
    ids, keep = encode(batch_words)
    rows = np.arange(len(batch_words))
    out = np.zeros(len(batch_words))
    for j in range(1, T_MAX):
        col = keep[:, j] & (ids[:, j] != EOS)
        if not col.any():
            continue
        x = ids.copy()
        x[col, j] = MASK
        logits, _ = model.head_logits(x, keep)
        lg = logits[:, j]
        lg = lg - lg.max(-1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        out[col] += lp[rows, ids[:, j]][col]
    return out


def mlm_passes(batch_words):
    ids, keep = encode(batch_words)
    return int(sum((keep[:, j] & (ids[:, j] != EOS)).any()
                   for j in range(1, T_MAX)))


def oracle_logprob(batch_words):
    out = []
    for w in batch_words:
        nl = oracle_nll(w)
        out.append(-np.inf if any(np.isinf(v) for v in nl) else -sum(nl))
    return np.array(out)


def threshold_acc(scores_train, y_train, scores_test, y_test):
    """Fit one number -- the midpoint of the class means -- and apply it."""
    thr = 0.5 * (scores_train[y_train == 1].mean()
                 + scores_train[y_train == 0].mean())
    return float(((scores_test > thr) == (y_test == 1)).mean()), float(thr)


def score_task(scorer, model, train, test):
    st = scorer(model, [w for w, _ in train]) if model is not None \
        else scorer([w for w, _ in train])
    ss = scorer(model, [w for w, _ in test]) if model is not None \
        else scorer([w for w, _ in test])
    yt = np.array([l for _, l in train])
    ys = np.array([l for _, l in test])
    finite = np.isfinite(st)
    if not finite.all():                    # the oracle: p = 0 is decisive
        st = np.where(finite, st, -1e9)
        ss = np.where(np.isfinite(ss), ss, -1e9)
    return threshold_acc(st, yt, ss, ys)


# -----------------------------------------------------------------------
# What the decoder pays: the slot whose evidence lies to its right.
# Mirrors Part 4's receipt, read from the LM's next-token distribution.
# -----------------------------------------------------------------------

def slot_prediction(model, sents, slot):
    """The LM's argmax next token where `slot` is due.  One pass, no
    corruption: position j already predicts word j."""
    ids, keep = encode([s["words"] for s in sents])
    logits, _ = model.head_logits(ids, keep)
    pos = np.array([s[slot] for s in sents])
    rows = np.arange(len(sents))
    pred = logits[rows, pos].argmax(-1)
    return ([VOCAB[i] for i in pred],
            [s["words"][s[slot]] for s in sents])


def mlm_slot_prediction(model, sents, slot):
    """Part 4's probe: blank the slot and read the encoder's head."""
    ids, keep = encode([s["words"] for s in sents])
    pos = np.array([s[slot] + 1 for s in sents])      # +1 for [BOS]
    rows = np.arange(len(sents))
    ids[rows, pos] = MASK
    logits, _ = model.head_logits(ids, keep)
    pred = logits[rows, pos].argmax(-1)
    return ([VOCAB[i] for i in pred],
            [s["words"][s[slot]] for s in sents])


def score_subject(model, sents, predict=slot_prediction):
    pred, gold = predict(model, sents, "subj")
    n = len(sents)
    num = sum((p in PLURAL_NOUNS) == s["plural"]
              for p, s in zip(pred, sents)) / n
    top = sum(TOPIC_OF.get(p) == s["topic"]
              for p, s in zip(pred, sents)) / n
    exact = sum(p == g for p, g in zip(pred, gold)) / n
    gendered = [(p, s) for p, s in zip(pred, sents)
                if s["gender"] in ("m", "f")]
    gen_ok = sum(
        (p in GENDER_M) if s["gender"] == "m" else (p in GENDER_F)
        for p, s in gendered) / max(len(gendered), 1)
    return num, top, exact, gen_ok, pred


def score_verb(model, sents, predict=slot_prediction):
    pred, gold = predict(model, sents, "verb")
    n = len(sents)
    num = sum((p in VERB_PL) == s["plural"]
              for p, s in zip(pred, sents)) / n
    top = sum(TOPIC_OF.get(p) == s["topic"]
              for p, s in zip(pred, sents)) / n
    exact = sum(p == g for p, g in zip(pred, gold)) / n
    return num, top, exact


def score_poss(model, sents, predict=slot_prediction):
    pred, gold = predict(model, sents, "poss")
    n = len(sents)
    exact = sum(p == g for p, g in zip(pred, gold)) / n
    num = sum((p == "their") == s["plural"]
              for p, s in zip(pred, sents)) / n
    return exact, num


def singular_base_rate(sents):
    return sum(not s["plural"] for s in sents) / len(sents)


# -----------------------------------------------------------------------
# Generation: the thing an encoder cannot do.  One forward pass per
# token, no key/value cache, so the prefix is recomputed every step.
# -----------------------------------------------------------------------

def generate(model, rng, temp=1.0, prompt=(), ban_specials=False):
    """Sample left to right until [EOS] or the length cap."""
    ids = np.zeros((1, T_MAX), dtype=int)
    ids[0, 0] = BOS
    n = 1
    words = []
    for w in prompt:
        ids[0, n] = W2I[w]
        words.append(w)
        n += 1
    passes = 0
    while n < T_MAX:
        keep = np.zeros((1, T_MAX), dtype=bool)
        keep[0, :n] = True
        logits, _ = model.head_logits(ids, keep)
        passes += 1
        lg = logits[0, n - 1].astype(float)
        if ban_specials:
            lg[[PAD, BOS, MASK]] = -1e30
        if temp <= 0:
            nxt = int(lg.argmax())
        else:
            z = lg / temp
            p = np.exp(z - z.max())
            nxt = int(rng.choice(V, p=p / p.sum()))
        if nxt == EOS:
            return words, passes, True
        words.append(VOCAB[nxt])
        ids[0, n] = nxt
        n += 1
    return words, passes, False


def sample_quality(model, rng, n=2000, temp=1.0, prompt=(),
                   ban_specials=False):
    """Sample n sentences and ask the oracle what they are."""
    reasons = defaultdict(int)
    seen, ok_seen, samples, passes = set(), set(), [], 0
    for _ in range(n):
        w, p, term = generate(model, rng, temp=temp, prompt=prompt,
                              ban_specials=ban_specials)
        passes += p
        reason = diagnose(w) if term else "no [EOS]"
        reasons[reason] += 1
        seen.add(tuple(w))
        if reason == "ok":
            ok_seen.add(tuple(w))
        samples.append((w, reason))
    return dict(grammatical=reasons["ok"] / n,
                unique=len(seen) / n,
                unique_ok=len(ok_seen) / n,
                reasons=dict(reasons), samples=samples,
                passes_per_sentence=passes / n)


MARGINAL_FNS = [
    ("six words", lambda w: len(w) == 6),
    ("plural subject", lambda w: w[-4] in PLURAL_NOUNS),
    ("plural object", lambda w: w[-1] in PLURAL_NOUNS),
    ('possessive "the"', lambda w: w[-2] == "the"),
    ("object = partner", lambda w: w[-1] == SIG.get(w[-4])),
    ("topic royal", lambda w: TOPIC_OF.get(w[-4]) == "royal"),
]
EXACT_MARGINALS = None


def marginals(word_lists):
    """Statistics the language fixes exactly, measured on a sample.

    Only well-formed sentences are passed in, so the slot positions
    counted from the right are meaningful.
    """
    ws = [w for w in word_lists if len(w) in (5, 6)]
    return ([n for n, _ in MARGINAL_FNS],
            [float(np.mean([bool(f(w)) for w in ws]))
             for _, f in MARGINAL_FNS])


def exact_marginals():
    """The same statistics, summed over the enumerated language."""
    global EXACT_MARGINALS
    if EXACT_MARGINALS is None:
        tot = [0.0] * len(MARGINAL_FNS)
        for s, p in SENT_P.items():
            for i, (_, f) in enumerate(MARGINAL_FNS):
                if f(s):
                    tot[i] += p
        EXACT_MARGINALS = ([n for n, _ in MARGINAL_FNS], tot)
    return EXACT_MARGINALS


def model_on_language(model, batch=4000):
    """Compare the model's distribution with the language's, exactly.

    Sampling estimates what the model would say; this computes it.  The
    model assigns a probability to every one of the enumerated
    sentences, so the mass it puts on the language, the KL divergence
    from the language to the model, and the model's own marginals all
    come out in closed form -- no sampling error at all.  Only a
    decoder permits this: its sentence probabilities are a
    factorisation, and they sum to one over sequences.
    """
    sents = sorted(SENT_P)
    pL = np.array([SENT_P[s] for s in sents])
    lp = np.empty(len(sents))
    for i in range(0, len(sents), batch):
        chunk = [list(s) for s in sents[i:i + batch]]
        lp[i:i + batch] = lm_logprob(model, chunk)
    pM = np.exp(lp)
    mass = float(pM.sum())
    kl = float((pL * (np.log(pL) - lp)).sum())
    xent = float(-(pL * lp).sum())
    tv = float(0.5 * np.abs(pL - pM).sum())
    mg = [float(sum(pM[i] for i, s in enumerate(sents) if f(s)) / mass)
          for _, f in MARGINAL_FNS]
    return dict(mass=mass, kl=kl, xent=xent, tv=tv,
                names=[n for n, _ in MARGINAL_FNS], marginals=mg)


# -----------------------------------------------------------------------
# Gradient audit.
# -----------------------------------------------------------------------

def gradcheck(seed=0, n_per=4):
    rng = np.random.default_rng(seed)
    sents = gen_corpus(5, rng)
    ids, keep = encode([s["words"] for s in sents])
    labels = rng.integers(0, 2, len(sents))
    x_mlm, tgt_mlm, sel_mlm = mlm_targets(ids, keep, rng)
    _, tgt_lm, sel_lm = lm_targets(ids, keep)

    worst = 0.0
    checked = 0
    cases = [("lm", True), ("mlm", False), ("cls-last", True),
             ("cls-first", False), ("cls-mean", True)]
    for which, causal in cases:
        net = Transformer(d=16, heads=2, layers=2, seed=1, causal=causal)
        net.pool = {"cls-last": "last", "cls-first": "first",
                    "cls-mean": "mean"}.get(which, "last")
        if which == "lm":
            g = net.head_grads(ids, keep, tgt_lm, sel_lm)

            def loss():
                return net.head_loss(ids, keep, tgt_lm, sel_lm)
        elif which == "mlm":
            g = net.head_grads(x_mlm, keep, tgt_mlm, sel_mlm)

            def loss():
                return net.head_loss(x_mlm, keep, tgt_mlm, sel_mlm)
        else:
            g = net.cls_grads(ids, keep, labels)

            def loss():
                return net.cls_loss(ids, keep, labels)
        for k, p in net.P.items():
            flat, gflat = p.reshape(-1), g[k].reshape(-1)
            for ix in rng.choice(len(flat), min(n_per, len(flat)),
                                 replace=False):
                old = flat[ix]
                flat[ix] = old + 1e-6
                lp = loss()
                flat[ix] = old - 1e-6
                lm = loss()
                flat[ix] = old
                num = (lp - lm) / 2e-6
                worst = max(worst, abs(num - gflat[ix]))
                checked += 1
    return worst, checked


def selftest():
    """The oracle must agree with the generator it claims to describe."""
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
    return len(SENT_P), H, float(nll.mean()), float(se)


# -----------------------------------------------------------------------
# Frozen configuration.  The table and header generators import these so
# their numbers are the ones printed below.
# -----------------------------------------------------------------------

CORPUS_N, TEST_N, ACCEPT_N = 8000, 1000, 600
SEED_CORPUS, SEED_TEST, SEED_ACCEPT = 42, 7, 13
SEED_MODEL, SEED_PRETRAIN, SEED_FT = 0, 1, 11
SEED_SAMPLE = 9
GRID = (10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1100, 1500)
TEMPS = (0.0, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0)
LABELS = (20, 50, 100, 400)
FT_STEPS = 400
N_SAMPLES = 1000


def build_stage():
    rng = np.random.default_rng(SEED_CORPUS)
    corpus = gen_corpus(CORPUS_N, rng)
    ids, keep = encode([s["words"] for s in corpus])
    test = gen_corpus(TEST_N, np.random.default_rng(SEED_TEST))
    accept = make_acceptability(ACCEPT_N, np.random.default_rng(SEED_ACCEPT))
    train20 = make_acceptability(20, np.random.default_rng(1020))
    return corpus, ids, keep, test, accept, train20


def train_pair(ids, keep, accept, train20, test, steps=STEPS, grid=GRID):
    """The decoder and the encoder: same trunk, same budget, same data."""
    def probe(scorer, predict, track_nll):
        def f(m):
            a, _ = score_task(scorer, m, train20, accept)
            v = score_verb(m, test[:500], predict=predict)[0]
            # A bidirectional model cannot be scored left to right --
            # every position would be reading its own answer.
            return (a, v, lm_eval(m, test)[0] if track_nll
                    else float("nan"))
        return f

    lm = Transformer(d=64, heads=4, layers=2, causal=True,
                     seed=SEED_MODEL, lr=3e-3, pool="last")
    h_lm, p_lm, n_lm = pretrain(
        lm, ids, keep, steps, np.random.default_rng(SEED_PRETRAIN),
        objective="lm", log_every=max(steps // 12, 1), probe_at=grid,
        probe_fn=probe(lm_logprob, slot_prediction, True))

    mlm = Transformer(d=64, heads=4, layers=2, causal=False,
                      seed=SEED_MODEL, lr=3e-3, pool="first")
    h_ml, p_ml, n_ml = pretrain(
        mlm, ids, keep, steps, np.random.default_rng(SEED_PRETRAIN),
        objective="mlm", log_every=max(steps // 12, 1), probe_at=grid,
        probe_fn=probe(mlm_pseudo_logprob, mlm_slot_prediction, False))
    return (lm, h_lm, p_lm, n_lm), (mlm, h_ml, p_ml, n_ml)


def finetune(base, pool, train, accept, steps=FT_STEPS, lr=1e-3,
             causal=True, seed=5):
    """Fit a classification head (and the trunk) on n labels."""
    if base is None:
        f = Transformer(d=64, heads=4, layers=2, causal=causal,
                        seed=seed, lr=lr, pool=pool)
    else:
        f = base.clone()
        f.lr = lr
        f.pool = pool
    xs, ks = encode([w for w, _ in train])
    ys = np.array([y for _, y in train])
    xt, kt = encode([w for w, _ in accept])
    yt = np.array([y for _, y in accept])
    rng = np.random.default_rng(SEED_FT)
    for _ in range(steps):
        b = rng.integers(0, len(xs), min(16, len(xs)))
        f.step(f.cls_grads(xs[b], ks[b], ys[b]))
    return f.cls_accuracy(xt, kt, yt)


# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

def demo1(corpus, ids, keep):
    banner("1. ONE BOOLEAN AND A SHIFT")
    print("Part 4's encoder and this decoder are the same class. The")
    print("decoder sets causal=True and hands the loss the ids shifted")
    print("left; the encoder sets causal=False and hands it the ids it")
    print("hid. Nothing else differs.")
    print()
    s = corpus[0]
    x_lm, tgt_lm, sel_lm = lm_targets(ids[:1], keep[:1])
    print("   sentence:  " + " ".join(s["words"]))
    print("   input:     " + " ".join(VOCAB[i] for i in x_lm[0]))
    print("   target:    " + " ".join(
        VOCAB[t] if m else "." for t, m in zip(tgt_lm[0], sel_lm[0])))
    print()
    print("Every position carries a target, including the one that")
    print("predicts [EOS] -- which is how the model learns to stop.")
    print()
    _, _, sel_all = lm_targets(ids, keep)
    lm_n = sel_all.sum(1).mean()
    ml_n = float(np.mean([
        mlm_targets(ids, keep, np.random.default_rng(100 + d))[2]
        .sum(1).mean() for d in range(20)]))
    print("   mean sequence length (with [BOS]/[EOS]): %.3f"
          % keep.sum(1).mean())
    print("   LM targets per sequence:  %.3f" % lm_n)
    print("   MLM targets per sequence: %.3f  (15%% of words, but never"
          " fewer than one)" % ml_n)
    print("   ratio: %.2fx the gradient signal per forward pass"
          % (lm_n / ml_n))
    print()
    print("Corruption is what an encoder needs and a decoder does not:")
    print("the causal mask already stops a position reading its own")
    print("answer, so the input is the corpus, untouched.")
    return lm_n, ml_n


def demo2(lm, test):
    banner("2. A LOSS WITH A KNOWN FLOOR")
    n_sent, H, mean_nll, se = SELFTEST
    print("The micro-language is small enough to enumerate: %d distinct"
          % n_sent)
    print("sentences, total probability 1. So its entropy is not")
    print("estimated, it is computed -- %.4f nats per sentence -- and a"
          % H)
    print("left-to-right model's loss can be read against it.")
    print()
    tok, sent, otok, osent = lm_eval(lm, test)
    print("   model:  %.4f nats/token   %.4f nats/sentence"
          % (tok, sent))
    print("   floor:  %.4f nats/token   %.4f nats/sentence"
          % (otok, osent))
    print("   gap:   %+.4f nats/token  %+.4f nats/sentence  (%.1f%% above"
          " the floor)" % (tok - otok, sent - osent,
                           100 * (tok / otok - 1)))
    print()
    print("Perplexity %.3f against an irreducible %.3f."
          % (np.exp(tok), np.exp(otok)))
    print()
    print("The same comparison, slot by slot. Where the language is")
    print("deterministic the floor is zero; where it is a uniform choice")
    print("of four verbs the floor is ln 4 = %.4f." % np.log(4))
    print()
    rows = slot_nll(lm, test)
    print("   %-6s %6s %9s %9s %9s" % ("slot", "n", "model", "floor",
                                       "gap"))
    for name in SLOT_NAMES:
        if name not in rows:
            continue
        m, o, n = rows[name]
        print("   %-6s %6d %9.4f %9.4f %+9.4f" % (name, n, m, o, m - o))
    print()
    sm, smax = special_mass(lm, test)
    print("Nothing is wasted on the unsayable: total probability on")
    print("[PAD], [BOS] and [MASK] as a next token averages %.2e"
          % sm)
    print("(worst position %.2e). The model learned which tokens are"
          % smax)
    print("structure rather than language.")
    return rows, (tok, sent, otok, osent)


def demo3(lm, mlm, p_lm, p_ml, n_lm, n_ml, test, accept, train20):
    banner("3. WHAT THE FLOOD BUYS, AND WHAT IT COSTS")
    print("Same trunk, same %d steps, same corpus. The decoder sees"
          % STEPS)
    print("%.2fx the targets. Two yardsticks, both label-free apart from"
          % (n_lm / n_ml))
    print("a single threshold fitted on 20 labels:")
    print()
    print("   verb number  -- does the subject govern the verb? Evidence")
    print("                   lies to the LEFT, so both can reach it.")
    print("   sentence     -- acceptability, scored by each model's own")
    print("                   best label-free scorer.")
    print()
    print("   %7s %11s %11s %11s %11s %11s"
          % ("steps", "LM sent", "MLM sent", "LM verb", "MLM verb",
             "LM nats/tok"))
    for (s, (a1, v1, q1)), (_, (a2, v2, _)) in zip(p_lm, p_ml):
        print("   %7d %11.4f %11.4f %11.4f %11.4f %11.4f"
              % (s, a1, a2, v1, v2, q1))
    print()

    def first_at(probes, i, thr):
        for s, vals in probes:
            if vals[i] >= thr:
                return s
        return None

    lv, mv = first_at(p_lm, 1, 0.999), first_at(p_ml, 1, 0.999)
    la, ma = first_at(p_lm, 0, 0.95), first_at(p_ml, 0, 0.95)
    print("Read the two pairs separately, because they disagree. On the")
    print("verb they are level: the decoder first scores a perfect")
    print("verb-number at step %s, the encoder at step %s, and the"
          % (lv, mv))
    print("encoder is briefly ahead on the way there. The gradient")
    print("flood does not buy the RULE sooner.")
    print()
    print("On whole sentences it buys a great deal. The decoder passes")
    print("95%% at step %s; the encoder needs step %s, about %.0fx as"
          % (la, ma, ma / la))
    print("many, which is close to the %.2fx target ratio." % (n_lm / n_ml))
    print()
    passes = mlm_passes([s["words"] for s in test[:64]])
    print("And the scores cost different amounts. The decoder's")
    print("probabilities factorise, so one forward pass gives the whole")
    print("sentence; the encoder has to blank one position at a time.")
    print()
    print("   passes to score a sentence:  LM %d   MLM %d"
          % (LM_PASSES, passes))
    print()
    print("Now the bill. The causal mask cannot be made free: on the")
    print("slot whose evidence lies to the RIGHT, the decoder is blind.")
    print()
    base = singular_base_rate(test)
    bs = score_subject(lm, test)
    bv = score_verb(lm, test)
    bp = score_poss(lm, test)
    es = score_subject(mlm, test, predict=mlm_slot_prediction)
    ev = score_verb(mlm, test, predict=mlm_slot_prediction)
    ep = score_poss(mlm, test, predict=mlm_slot_prediction)
    print("   %-22s %9s %9s %9s" % ("slot / property", "evidence",
                                    "decoder", "encoder"))
    for name, ev_side, a, b in [
            ("subject number", "right", bs[0], es[0]),
            ("subject gender", "right", bs[3], es[3]),
            ("subject topic", "right", bs[1], es[1]),
            ("verb number", "left", bv[0], ev[0]),
            ("possessive exact", "left", bp[0], ep[0])]:
        print("   %-22s %9s %9.4f %9.4f" % (name, ev_side, a, b))
    print()
    print("Those are not bad scores, they are base rates. Subject")
    print("number lands on %.4f, which is exactly the rate of guessing"
          % bs[0])
    print("singular every time (%.4f). Subject topic lands on %.4f,"
          % (base, bs[1]))
    print("and half these sentences carry an adjective that gives the")
    print("topic away while the other half start bare -- so the ceiling")
    print("on a left-to-right guess is 0.5 + 0.5/3 = %.4f." % (0.5 + 0.5 / 3))
    print("There is nothing to the left of the subject to read, so the")
    print("decoder guesses, exactly as Part 4's causal encoder did.")
    print()
    print("Which route you take to a labelled task matters more than")
    print("which objective you pre-trained with:")
    print()
    print("   %-8s %10s %10s %10s %10s %10s"
          % ("labels", "LM score", "LM f-t", "MLM pll", "MLM f-t",
             "scratch"))
    ft_rows = []
    for n in LABELS:
        tr = make_acceptability(n, np.random.default_rng(1000 + n))
        a_lm, _ = score_task(lm_logprob, lm, tr, accept)
        a_ml, _ = score_task(mlm_pseudo_logprob, mlm, tr, accept)
        f_lm = finetune(lm, "last", tr, accept)
        f_ml = finetune(mlm, "first", tr, accept, causal=False)
        f_sc = finetune(None, "last", tr, accept)
        ft_rows.append((n, a_lm, f_lm, a_ml, f_ml, f_sc))
        print("   %-8d %10.4f %10.4f %10.4f %10.4f %10.4f"
              % (n, a_lm, f_lm, a_ml, f_ml, f_sc))
    orc, _ = score_task(oracle_logprob, None, train20, accept)
    print()
    print("The oracle scores %.4f on this task, so the ceiling is 1."
          % orc)
    print()
    print("Two results worth separating. Fine-tuned, the ENCODER wins at")
    print("every label count -- which is the history: BERT owned the")
    print("fine-tuning era. Scored with no new parameters at all, the")
    print("DECODER wins, and twenty labels spent on one threshold beat")
    print("four hundred labels spent fine-tuning the encoder.")
    return ft_rows, (bs, bv, bp, es, ev, ep, base), passes


def demo4(lm, sent_gap):
    banner("4. GENERATION, WHICH AN ENCODER CANNOT DO")
    print("An encoder scores text. A decoder writes it: sample a token,")
    print("append it, run the model again. And because the language is")
    print("enumerable, we do not have to squint at the output and call")
    print("it fluent -- the oracle says whether a sentence is IN the")
    print("language, and if not, which rule broke.")
    print()
    print("   %6s %13s %9s %9s %7s" % ("temp", "grammatical", "unique",
                                       "passes", "worst failure"))
    rows = []
    for t in TEMPS:
        q = sample_quality(lm, np.random.default_rng(SEED_SAMPLE),
                           n=N_SAMPLES, temp=t)
        bad = sorted(((v, k) for k, v in q["reasons"].items()
                      if k != "ok"), reverse=True)
        worst = ("%s %d" % (bad[0][1], bad[0][0])) if bad else "-"
        rows.append((t, q, worst))
        print("   %6.1f %13.4f %9.4f %9.2f   %s"
              % (t, q["grammatical"], q["unique"],
                 q["passes_per_sentence"], worst))
    print()
    print("Temperature 0 is the trap: every sentence is grammatical and")
    print("they are all the SAME sentence -- %.1f%% distinct out of %d"
          % (100 * rows[0][1]["unique"], N_SAMPLES))
    print("samples. Greedy decoding is not careful, it is stuck. Push")
    print("the other way and grammar dissolves. The useful band here is")
    print("0.5 to 1.0, and nothing in the loss curve would have told us")
    print("where it was.")
    print()
    q1 = [r for r in rows if r[0] == 1.0][0][1]
    print("Samples at temperature 1.0:")
    for w, reason in q1["samples"][:10]:
        print("   %-46s %s" % (" ".join(w), reason))
    print()
    print("Grammatical is the low bar, and sampling only estimates it.")
    print("The model assigns a probability to every one of the %d"
          % len(SENT_P))
    print("sentences in the language, so we can stop sampling and just")
    print("add them up -- exactly, with no sampling error:")
    print()
    M = model_on_language(lm)
    print("   probability mass on the language:  %.4f" % M["mass"])
    print("   measured by sampling at temp 1.0:  %.4f"
          % q1["grammatical"])
    print("   KL(language || model):             %.4f nats" % M["kl"])
    print("   total variation distance:          %.4f" % M["tv"])
    print()
    print("The sampled rate and the exact mass agree to %.4f, which is"
          % abs(M["mass"] - q1["grammatical"]))
    print("the check that the sampler really draws from the model. That")
    print("KL is the quantity demo 2 estimated at %+.4f nats per"
          % sent_gap)
    print("sentence from a thousand held-out examples; here it is")
    print("%.4f, computed from the language itself rather than" % M["kl"])
    print("sampled. And %.2f%% of the model's probability is spent on"
          % (100 * (1 - M["mass"])))
    print("strings that are not sentences of the language at all.")
    print()
    print("And the statistics it was never told about, exactly:")
    print()
    _, want = exact_marginals()
    print("   %-18s %11s %11s %8s" % ("statistic", "model", "language",
                                      "diff"))
    worst_d, worst_n = 0.0, ""
    for nm, a, b in zip(M["names"], M["marginals"], want):
        print("   %-18s %11.4f %11.4f %+8.4f" % (nm, a, b, a - b))
        if abs(a - b) > abs(worst_d):
            worst_d, worst_n = a - b, nm
    print()
    print("These are exact, so the deviations are real and not sampling")
    print("noise -- and they land where demo 2 said they would. The")
    print("largest is %s at %+.4f, and the object's partner"
          % (worst_n, worst_d))
    print("rate is next; those are the two slots with the widest loss")
    print("gap, the adjective-or-subject coin flip and the role")
    print("signature Part 1 built. This model learned the grammar")
    print("almost exactly and the frequencies only approximately.")
    print()
    print("Two limits worth naming. The vocabulary is a hard wall: the")
    print("plurals Part 1 held out -- %s --" % ", ".join(HELD_OUT_PLURALS))
    print("are not tokens, so this model cannot utter them however well")
    print("it understands plurals. That is the problem Part 2's subword")
    print("pieces exist to solve.")
    print()
    print("And generation is sequential where training was parallel. One")
    print("training pass scored every position of a sentence at once;")
    print("writing one takes a pass per token, each re-reading the whole")
    print("prefix -- %.2f passes and %d position-reads for a sentence"
          % (q1["passes_per_sentence"],
             sum(range(1, int(round(q1["passes_per_sentence"])) + 1))))
    print("that training handled in one pass over %d positions."
          % int(round(q1["passes_per_sentence"])))
    print("Caching the keys and values is the standard repair.")
    print()
    print("Prompted, the same machine continues instead of inventing:")
    print()
    for prompt in [("the", "noble"), ("the", "salty", "captains"),
                   ("the", "muddy", "hen"), ("the", "kings")]:
        rng = np.random.default_rng(3)
        outs = []
        for _ in range(2):
            w, _, term = generate(lm, rng, temp=0.8, prompt=prompt)
            outs.append(" ".join(w) + ("" if term else " [no EOS]"))
        print("   %-24s -> %s" % (" ".join(prompt), "  |  ".join(outs)))
    print()
    print("Note the agreement it supplies unprompted: a plural subject")
    print("takes a plural verb and `their`, `hen` takes `her`. Nobody")
    print("labelled that. It is what predicting the next token requires.")
    return rows


SELFTEST = None


def main():
    global SELFTEST
    banner("GPT-STYLE DECODERS --- causal language modelling from scratch")
    worst, checked = gradcheck()
    print("gradient audit: worst error %.2e over %d sampled gradients"
          % (worst, checked))
    assert worst < 1e-6, worst
    SELFTEST = selftest()
    print("oracle audit:   %d sentences enumerated, mass 1.0, mean"
          % SELFTEST[0])
    print("                sampled NLL %.4f +- %.4f against H = %.4f"
          % (SELFTEST[2], SELFTEST[3], SELFTEST[1]))
    print("                and diagnose() agrees with the enumeration on")
    print("                6,000 corrupted and random word strings")

    corpus, ids, keep, test, accept, train20 = build_stage()
    print()
    print("corpus %d sentences, %d held out, %d acceptability pairs,"
          % (len(corpus), len(test), len(accept)))
    print("vocabulary %d tokens, sequences padded to %d." % (V, T_MAX))

    demo1(corpus, ids, keep)

    banner("PRE-TRAINING")
    probe = Transformer(d=64, heads=4, layers=2, causal=True,
                        seed=SEED_MODEL)
    print("two models, %d parameters each, %d steps, batch 64, lr 3e-3:"
          % (probe.n_params(), STEPS))
    print("  a DECODER (causal, next-token) and an ENCODER (bidirectional,")
    print("  masked) -- the same class, the same budget, the same data.")
    print("  this is the slow part, about two and a half minutes.")
    (lm, h_lm, p_lm, n_lm), (mlm, h_ml, p_ml, n_ml) = train_pair(
        ids, keep, accept, train20, test)
    print()
    print("  decoder loss: " + "  ".join("%d:%.3f" % (s, l)
                                         for s, l in h_lm))
    print("  encoder loss: " + "  ".join("%d:%.3f" % (s, l)
                                         for s, l in h_ml))
    print()
    print("  The two curves are not comparable and should not be")
    print("  compared: one is the cost of guessing the next word, the")
    print("  other the cost of filling a gap. Only the yardsticks below")
    print("  put them on the same axis.")

    _, (tok, sent, otok, osent) = demo2(lm, test)
    demo3(lm, mlm, p_lm, p_ml, n_lm, n_ml, test, accept, train20)
    demo4(lm, sent - osent)

    banner("DONE")
    return lm, mlm


if __name__ == "__main__":
    main()
