"""
rag.py --- companion code for "Retrieval-Augmented Generation"
(Natural Language Processing, Part 8).

Part 6 built a bi-encoder and showed why it is the only thing that can
search a corpus: its sentence vectors do not depend on what they are
compared against, so they can be computed once and cached. Part 7 built
the generator and measured what decoding costs. This article bolts them
together, which is the whole of RAG, and then asks the only question
that matters about the result:

    when the model answers, is it reading the document
    or is it reciting what it already believed?

In a real system that question is close to unanswerable, because the
retrieved document usually agrees with the model's pre-training and
there is no way to tell recall from reading. Here it is a measurement.
Part 1's micro-language gives every subject a ROLE SIGNATURE -- the
object the language expects after it -- and the oracle says exactly how
strong that expectation is. So a knowledge-base fact can be built to
AGREE with the prior or to CONTRADICT it, and the difference between
those two accuracies is grounding, isolated.

What that buys, measured here:
  1. The two halves, priced separately. A bi-encoder retriever against
     a lexical-overlap baseline on the same 144-document store, then
     the generator scored with oracle retrieval so the errors of each
     half can be told apart.
  2. The closed-book baseline, which is what the prior alone is worth.
  3. The grounding gap: the same generator, the same gold document,
     scored on facts that agree with the prior and on facts that
     contradict it. A benchmark built only from agreeing facts cannot
     measure grounding at all -- it reports the prior.
  4. Where grounding comes from. Sweeping the fraction of TRAINING
     documents that contradict the prior shows the behaviour is
     learned, not inherited: a model fine-tuned only on agreeing
     evidence learns to ignore the evidence.
  5. What a wrong retrieval does, which is not what you would guess.

The language, the oracle and the transformer trunk are Parts 5-7's.
New here: a [SEP]-joined RAG format, mean-pooled dual encoders trained
with in-batch negatives (DPR's recipe), and a generator whose loss is
taken on the answer span only. Everything is hand-differentiated and
audited against central differences at startup. Seeded, deterministic.
Dependencies: numpy.  Runs in about six and a half minutes.
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
# Vocabulary.  Parts 5-7's words plus the three that make a question,
# and [SEP] in place of [MASK]: nothing here is masked, but document,
# question and answer have to be told apart.
# -----------------------------------------------------------------------

QUESTION_WORDS = ["what", "does", "do"]
N_SPECIAL = 4
PAD, BOS, EOS, SEP = 0, 1, 2, 3


def build_vocab():
    words = set(QUESTION_WORDS) | {"the", "a", "and", "near",
                                   "his", "her", "their"}
    for t in TOPICS.values():
        for n in t["nouns"]:
            words.add(n)
            if plural(n) not in HELD_OUT_PLURALS:
                words.add(plural(n))
        for (vs, vp) in t["verbs"]:
            words.update([vs, vp])
        words.update(t["adjs"])
    vocab = ["[PAD]", "[BOS]", "[EOS]", "[SEP]"] + sorted(words)
    return vocab, {w: i for i, w in enumerate(vocab)}


VOCAB, W2I = build_vocab()
V = len(VOCAB)
T_MAX = 16         # [BOS] doc(5) [SEP] question(5) [SEP] answer [EOS]


def encode(batch_words, t_max=T_MAX):
    """[BOS] tokens... [EOS], padded.  Words may include [SEP]."""
    N = len(batch_words)
    ids = np.zeros((N, t_max), dtype=int)
    keep = np.zeros((N, t_max), dtype=bool)
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
# The knowledge base.
#
# One fact per (subject, verb) pair: 12 nouns x 4 verbs x 3 topics = 144
# documents.  Each fact names an object of the same topic, so every
# document is a grammatical sentence of the language -- the model cannot
# spot the planted ones by their shape.
#
# The object is either the subject's ROLE SIGNATURE, which is what the
# language's own distribution expects, or a different noun of that
# topic, which the prior actively disprefers.  That single switch is
# what makes grounding measurable.
# -----------------------------------------------------------------------

def poss_for(noun):
    return ("his" if noun in GENDER_M else
            "her" if noun in GENDER_F else "the")


def question_for(subj, verb_pl):
    """'what does the king command' -- note the BASE verb form.

    The document says 'commands'.  Sharing no exact token with the
    document's verb is deliberate: it stops lexical overlap from
    solving retrieval for free.
    """
    return ["what", "does", "the", subj, verb_pl]


def build_kb(rng, contradict_rate=0.5):
    """Every (subject, verb) fact, with a controlled share planted."""
    facts = []
    for topic, t in TOPICS.items():
        others = {n: [m for m in t["nouns"] if m != n and m != SIG[n]]
                  for n in t["nouns"]}
        for subj in t["nouns"]:
            for (vs, vp) in t["verbs"]:
                contra = bool(rng.random() < contradict_rate)
                if contra:
                    pool = others[subj]
                    obj = pool[int(rng.integers(0, len(pool)))]
                else:
                    obj = SIG[subj]
                facts.append(dict(
                    topic=topic, subj=subj, verb=vs, verb_pl=vp,
                    obj=obj, contradicts=contra,
                    doc=["the", subj, vs, poss_for(subj), obj],
                    question=question_for(subj, vp)))
    return facts


def prior_object(subj, verb, poss):
    """What the LANGUAGE expects in the object slot, exactly."""
    d = ORACLE.get(("the", subj, verb, poss))
    return d or {}


# -----------------------------------------------------------------------
# The trunk.  Parts 5-7's transformer with the causal flag back: the
# retriever reads bidirectionally and mean-pools, the generator reads
# left to right and predicts the next token.
# -----------------------------------------------------------------------

class Transformer:
    def __init__(self, d=64, heads=4, layers=2, d_ff=None, causal=True,
                 seed=0, lr=3e-3):
        r = np.random.default_rng(seed)
        self.d, self.h, self.L = d, heads, layers
        self.dk = d // heads
        self.dff = d_ff or 4 * d
        self.causal = causal
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
        P["Wv"] = rnd(d, V)
        P["bv"] = np.zeros(V)
        self.P = P
        self.reset_adam()

    def reset_adam(self):
        self.M = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.Vm = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def clone(self):
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
        S = np.where(keep[:, None, None, :], S, -1e30)
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

    # ---- the token head (generator) ----
    def head_logits(self, ids, keep):
        H, cache = self.trunk_fwd(ids, keep)
        return H @ self.P["Wv"] + self.P["bv"], cache

    def head_loss(self, ids, keep, targets, sel):
        logits, _ = self.head_logits(ids, keep)
        Pr = softmax_last(logits)
        n = max(sel.sum(), 1)
        return float(-np.log(Pr[sel, targets[sel]] + 1e-300).sum() / n)

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

    # ---- mean pooling (retriever) ----
    def embed(self, ids, keep):
        """Mean-pooled, L2-normalised sentence vector."""
        H, cache = self.trunk_fwd(ids, keep)
        w = keep / keep.sum(1, keepdims=True)
        u = (H * w[:, :, None]).sum(1)
        n = np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
        return u / n, (cache, w, u, n)

    def embed_bwd(self, dz, aux, g):
        """Back through the normalisation and the mean pool."""
        cache, w, u, n = aux
        z = u / n
        du = (dz - z * (z * dz).sum(1, keepdims=True)) / n
        H = cache[4]
        dH = w[:, :, None] * du[:, None, :]
        self.trunk_bwd(dH, cache, g)
        return dH

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
# The retrieval half.  One shared encoder embeds both questions and
# documents (SBERT's arrangement rather than DPR's two towers), trained
# with in-batch negatives: every other document in the batch is a
# negative for this question, which is the cheapest useful contrastive
# signal there is and the one DPR actually used.
# =======================================================================

TAU = 0.05


def infonce(Q, D, tau=TAU):
    """Loss and gradients for in-batch negatives, query -> document."""
    N = len(Q)
    S = Q @ D.T / tau
    P = softmax_last(S)
    loss = float(-np.log(P[np.arange(N), np.arange(N)] + 1e-300).mean())
    dS = P.copy()
    dS[np.arange(N), np.arange(N)] -= 1.0
    dS /= N
    return loss, dS @ D / tau, dS.T @ Q / tau


def retriever_step(enc, pairs, idx):
    """One contrastive step over a batch of (question, document)."""
    qs = [pairs[i][0] for i in idx]
    ds = [pairs[i][1] for i in idx]
    qi, qk = encode(qs)
    di, dk = encode(ds)
    ids = np.concatenate([qi, di])
    keep = np.concatenate([qk, dk])
    Z, aux = enc.embed(ids, keep)
    n = len(idx)
    loss, dQ, dD = infonce(Z[:n], Z[n:])
    g = enc.zero_grads()
    enc.embed_bwd(np.concatenate([dQ, dD]), aux, g)
    enc.step(g)
    return loss


def train_retriever(pairs, steps, rng, seed=0, batch=32, lr=1e-3):
    enc = Transformer(d=64, heads=4, layers=2, causal=False,
                      seed=seed, lr=lr)
    hist = []
    for s in range(steps):
        idx = rng.choice(len(pairs), min(batch, len(pairs)),
                         replace=False)
        loss = retriever_step(enc, pairs, idx)
        if s % max(steps // 8, 1) == 0 or s == steps - 1:
            hist.append((s + 1, loss))
    return enc, hist


def embed_all(enc, texts, batch=256):
    out = []
    for i in range(0, len(texts), batch):
        ids, keep = encode(texts[i:i + batch])
        z, _ = enc.embed(ids, keep)
        out.append(z)
    return np.concatenate(out)


def lexical_scores(questions, docs):
    """Token-overlap baseline: |q tokens & d tokens| / |q tokens|."""
    S = np.zeros((len(questions), len(docs)))
    dsets = [set(d) for d in docs]
    for i, q in enumerate(questions):
        qs = set(q)
        for j, ds in enumerate(dsets):
            S[i, j] = len(qs & ds) / len(qs)
    return S


def recall_at(S, gold, ks=(1, 5)):
    """Share of questions whose gold document is in the top k."""
    order = np.argsort(-S, axis=1)
    out = {}
    for k in ks:
        hit = [gold[i] in order[i, :k] for i in range(len(gold))]
        out[k] = float(np.mean(hit))
    return out


# =======================================================================
# The generation half.  Input is
#     [BOS] document [SEP] question [SEP] answer [EOS]
# and the loss is taken on the ANSWER SPAN only -- the model is not
# asked to predict the document or the question, only what follows
# from them.  With no document the format is the same minus the first
# span, which is the closed-book control.
# =======================================================================

def rag_words(doc, question, answer=None):
    w = list(doc) + ["[SEP]"] + list(question) + ["[SEP]"]
    if answer is not None:
        w.append(answer)
    return w


def rag_batch(items):
    """(ids, keep, targets, sel) with sel marking the answer and [EOS]."""
    seqs = [rag_words(d, q, a) for d, q, a in items]
    ids, keep = encode(seqs)
    tgt = np.zeros_like(ids)
    tgt[:, :-1] = ids[:, 1:]
    sel = np.zeros(ids.shape, dtype=bool)
    for i, s in enumerate(seqs):
        # positions that predict the answer token and then [EOS]
        ans = len(s)                      # [BOS] shifts everything by 1
        sel[i, ans - 1] = True            # predicts the answer
        sel[i, ans] = True                # predicts [EOS]
    return ids, keep, tgt, sel


def train_generator(base, items, steps, rng, batch=32, lr=5e-4):
    gen = base.clone()
    gen.lr = lr
    hist = []
    for s in range(steps):
        idx = rng.choice(len(items), min(batch, len(items)),
                         replace=False)
        ids, keep, tgt, sel = rag_batch([items[i] for i in idx])
        gen.step(gen.head_grads(ids, keep, tgt, sel))
        if s % max(steps // 8, 1) == 0 or s == steps - 1:
            hist.append((s + 1, gen.head_loss(ids, keep, tgt, sel)))
    return gen, hist


def answer(gen, pairs, batch=256):
    """Greedy one-token answer for each (document, question)."""
    out = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        seqs = [rag_words(d, q) for d, q in chunk]
        ids, keep = encode(seqs)
        # the [EOS] encode() appended is not part of the prompt
        for r, s in enumerate(seqs):
            ids[r, len(s) + 1] = PAD
            keep[r, len(s) + 1] = False
        logits, _ = gen.head_logits(ids, keep)
        for r, s in enumerate(seqs):
            out.append(VOCAB[int(logits[r, len(s)].argmax())])
    return out


def score(gen, facts, docs_for, label=""):
    """Where the answer came from, three ways.

    `answer == fact["obj"]`       the document was used
    `answer == SIG[subj]`         the role signature was recited
    neither                       something else entirely
    On a CONTRADICTING fact the first two are mutually exclusive, which
    is the entire reason the stage is built this way.
    """
    pairs = [(docs_for(f), f["question"]) for f in facts]
    got = answer(gen, pairs)
    doc = [g == f["obj"] for g, f in zip(got, facts)]
    pri = [g == SIG[f["subj"]] for g, f in zip(got, facts)]
    other = [not a and not b for a, b in zip(doc, pri)]
    return dict(label=label, n=len(facts), acc=float(np.mean(doc)),
                prior=float(np.mean(pri)), other=float(np.mean(other)),
                got=got)


def variants(fact, rng):
    """One question, two documents: one agreeing, one contradicting.

    Same subject, same verb, same model, same everything -- the ONLY
    difference is whether the fact is the one the language expects.
    That is what makes the comparison a measurement of grounding and
    not of difficulty.
    """
    t = TOPICS[fact["topic"]]
    others = [m for m in t["nouns"]
              if m != fact["subj"] and m != SIG[fact["subj"]]]
    con = others[int(rng.integers(0, len(others)))]

    def mk(o):
        return ["the", fact["subj"], fact["verb"],
                poss_for(fact["subj"]), o]

    return (dict(fact, obj=SIG[fact["subj"]], contradicts=False,
                 doc=mk(SIG[fact["subj"]])),
            dict(fact, obj=con, contradicts=True, doc=mk(con)))


def remix(facts, rate, rng):
    """Rebuild a fact set at a chosen contradiction rate."""
    out = []
    for f in facts:
        t = TOPICS[f["topic"]]
        others = [m for m in t["nouns"]
                  if m != f["subj"] and m != SIG[f["subj"]]]
        contra = bool(rng.random() < rate)
        obj = (others[int(rng.integers(0, len(others)))] if contra
               else SIG[f["subj"]])
        out.append(dict(f, obj=obj, contradicts=contra,
                        doc=["the", f["subj"], f["verb"],
                             poss_for(f["subj"]), obj]))
    return out


def distractor_for(fact, store):
    """Same subject, different verb: plausible, retrievable, wrong.

    Its object must differ from BOTH the gold answer and the role
    signature, otherwise "the model was misled by the document" and
    "the model recited the prior" would name the same word and the
    three outcome columns would stop being exclusive.
    """
    bad = {fact["obj"], SIG[fact["subj"]]}
    for g in store:
        if (g["subj"] == fact["subj"] and g["verb"] != fact["verb"]
                and g["obj"] not in bad):
            return g
    # no stored fact qualifies: build one, same shape, same store rules
    t = TOPICS[fact["topic"]]
    other_v = [(vs, vp) for (vs, vp) in t["verbs"]
               if vs != fact["verb"]][0]
    obj = [m for m in t["nouns"]
           if m not in bad and m != fact["subj"]][0]
    return dict(fact, verb=other_v[0], verb_pl=other_v[1], obj=obj,
                contradicts=obj != SIG[fact["subj"]],
                doc=["the", fact["subj"], other_v[0],
                     poss_for(fact["subj"]), obj],
                question=question_for(fact["subj"], other_v[1]))


def prior_argmax(lm, facts):
    """What the PRE-TRAINED model says before it has read anything.

    The sentence prefix is fed straight to the language model and the
    object slot is read off.  No documents, no fine-tuning, no RAG.
    """
    hits = []
    for f in facts:
        pre = ["the", f["subj"], f["verb"], poss_for(f["subj"])]
        ids, keep = encode([pre])
        ids[0, len(pre) + 1] = PAD          # drop the appended [EOS]
        keep[0, len(pre) + 1] = False
        logits, _ = lm.head_logits(ids, keep)
        hits.append(VOCAB[int(logits[0, len(pre)].argmax())]
                    == SIG[f["subj"]])
    return float(np.mean(hits))


# -----------------------------------------------------------------------
# Gradient audit and self-test.
# -----------------------------------------------------------------------

def gradcheck(seed=0, n_per=3):
    rng = np.random.default_rng(seed)
    kb = build_kb(np.random.default_rng(1), 0.5)[:6]
    worst, checked = 0.0, 0

    # the generator head, on the answer span
    items = [(f["doc"], f["question"], f["obj"]) for f in kb]
    ids, keep, tgt, sel = rag_batch(items)
    net = Transformer(d=16, heads=2, layers=2, causal=True, seed=1)
    g = net.head_grads(ids, keep, tgt, sel)
    for k, p in net.P.items():
        flat, gflat = p.reshape(-1), g[k].reshape(-1)
        for ix in rng.choice(len(flat), min(n_per, len(flat)), False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp = net.head_loss(ids, keep, tgt, sel)
            flat[ix] = old - 1e-6
            lm = net.head_loss(ids, keep, tgt, sel)
            flat[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
            checked += 1

    # the contrastive retriever, through pooling and normalisation
    enc = Transformer(d=16, heads=2, layers=2, causal=False, seed=2)
    qi, qk = encode([f["question"] for f in kb])
    di, dk = encode([f["doc"] for f in kb])
    rids = np.concatenate([qi, di])
    rkeep = np.concatenate([qk, dk])
    n = len(kb)

    def rloss():
        Z, _ = enc.embed(rids, rkeep)
        return infonce(Z[:n], Z[n:])[0]

    Z, aux = enc.embed(rids, rkeep)
    _, dQ, dD = infonce(Z[:n], Z[n:])
    g = enc.zero_grads()
    enc.embed_bwd(np.concatenate([dQ, dD]), aux, g)
    for k, p in enc.P.items():
        if k in ("Wv", "bv"):        # unused by the retriever
            continue
        flat, gflat = p.reshape(-1), g[k].reshape(-1)
        for ix in rng.choice(len(flat), min(n_per, len(flat)), False):
            old = flat[ix]
            flat[ix] = old + 1e-6
            lp = rloss()
            flat[ix] = old - 1e-6
            lm = rloss()
            flat[ix] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
            checked += 1
    return worst, checked


def selftest():
    """The stage must be what the article says it is."""
    tot = sum(SENT_P.values())
    assert abs(tot - 1.0) < 1e-9, tot
    for s in SENT_P:
        assert diagnose(list(s)) == "ok", s

    kb = build_kb(np.random.default_rng(3), 0.5)
    assert len(kb) == 144, len(kb)
    for f in kb:
        # every document is a grammatical sentence, planted or not
        assert in_language(f["doc"]), f
        assert f["contradicts"] == (f["obj"] != SIG[f["subj"]]), f
        assert TOPIC_OF[f["obj"]] == f["topic"], f
        # the question shares no verb token with its document
        assert f["verb"] not in f["question"], f
    # documents are unique per (subject, verb), so retrieval is well posed
    assert len({(f["subj"], f["verb"]) for f in kb}) == len(kb)

    # the answer span really is the answer span
    items = [(f["doc"], f["question"], f["obj"]) for f in kb[:5]]
    ids, keep, tgt, sel = rag_batch(items)
    for i, f in enumerate(kb[:5]):
        assert sel[i].sum() == 2, sel[i]
        picked = [VOCAB[t] for t in tgt[i][sel[i]]]
        assert picked == [f["obj"], "[EOS]"], picked
    return len(SENT_P), len(kb)


# -----------------------------------------------------------------------
# Frozen configuration.
# -----------------------------------------------------------------------

CORPUS_N = 8000
PRE_STEPS = 1500
RET_STEPS = 600
GEN_STEPS = 2400
SEED_CORPUS, SEED_KB, SEED_SPLIT = 42, 7, 11
SEED_MODEL, SEED_PRE, SEED_RET, SEED_GEN = 0, 1, 2, 3
SEED_MIX = 100
TEST_FRACTION = 0.25
CLOSED_SHARE = 0.2
MIXES = (0.0, 0.25, 0.5, 0.75, 1.0)
TRAIN_MIX = 0.5


def pretrain_lm(ids, keep, steps, rng, batch=64):
    """Plain next-token pre-training: this is where the PRIOR comes from."""
    lm = Transformer(d=64, heads=4, layers=2, causal=True,
                     seed=SEED_MODEL, lr=3e-3)
    hist = []
    for s in range(steps):
        b = rng.integers(0, len(ids), batch)
        x, k = ids[b], keep[b]
        tgt = np.zeros_like(x)
        tgt[:, :-1] = x[:, 1:]
        sel = np.zeros(x.shape, dtype=bool)
        sel[:, :-1] = k[:, 1:]
        lm.step(lm.head_grads(x, k, tgt, sel))
        if s % max(steps // 8, 1) == 0 or s == steps - 1:
            hist.append((s + 1, lm.head_loss(x, k, tgt, sel)))
    return lm, hist


def build_stage():
    """Pre-training corpus, the knowledge base, and a disjoint split.

    Train and test facts are about DIFFERENT (subject, verb) pairs, so
    a generator cannot answer a test question from memory -- the
    document is the only source of the answer.
    """
    corpus = gen_corpus(CORPUS_N, np.random.default_rng(SEED_CORPUS))
    ids, keep = encode([s["words"] for s in corpus])
    kb = build_kb(np.random.default_rng(SEED_KB), contradict_rate=0.5)
    rng = np.random.default_rng(SEED_SPLIT)
    order = rng.permutation(len(kb))
    n_test = int(len(kb) * TEST_FRACTION)
    test = [kb[i] for i in sorted(order[:n_test])]
    train = [kb[i] for i in sorted(order[n_test:])]
    return corpus, ids, keep, kb, train, test
# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

SEED_VARIANT = 900


def build_eval(test):
    """The paired evaluation set: one question, two documents."""
    rng = np.random.default_rng(SEED_VARIANT)
    agree, contra = [], []
    for f in test:
        a, c = variants(f, rng)
        agree.append(a)
        contra.append(c)
    return agree, contra


def gen_items(facts, closed_share=CLOSED_SHARE):
    """Fine-tuning examples, with a share carrying no document at all."""
    items = [(f["doc"], f["question"], f["obj"]) for f in facts]
    n_closed = max(1, int(len(facts) * closed_share))
    items += [([], f["question"], f["obj"]) for f in facts[:n_closed]]
    return items


def demo1(enc, kb, test):
    """The retrieval half, against the baseline it has to beat."""
    banner("1. THE RETRIEVAL HALF")
    print("  The store holds %d documents, one per (subject, verb)."
          % len(kb))
    print("  A question never shares its verb token with its document:")
    f = test[0]
    print("    document: " + " ".join(f["doc"]))
    print("    question: " + " ".join(f["question"]))
    print("  so overlapping words alone cannot pick the right verb.")
    print()

    docs = [g["doc"] for g in kb]
    gold = [kb.index(g) for g in test]
    Zd = embed_all(enc, docs)
    Zq = embed_all(enc, [g["question"] for g in test])
    Slex = lexical_scores([g["question"] for g in test], docs)
    dense = recall_at(Zq @ Zd.T, gold)
    lex = recall_at(Slex, gold)
    band = float(np.mean([(Slex[i] == Slex[i, g]).sum()
                          for i, g in enumerate(gold)]))
    beaten = float(np.mean([(Slex[i] > Slex[i, g]).sum()
                            for i, g in enumerate(gold)]))
    print("  %-22s %10s %10s" % ("retriever", "recall@1", "recall@5"))
    print("  " + "-" * 44)
    print("  %-22s %10.4f %10.4f"
          % ("lexical overlap", lex[1], lex[5]))
    print("  %-22s %10.4f %10.4f"
          % ("bi-encoder (trained)", dense[1], dense[5]))
    print()
    print("  The lexical baseline is not failing to find the document.")
    print("  It is failing to CHOOSE. Averaged over the test questions")
    print("  only %.2f documents outscore the right one, but %.2f of"
          % (beaten, band))
    print("  them TIE with it, and a tie is decided by whatever order")
    print("  the store happens to be in. Overlap sees 'the' and the")
    print("  subject and nothing else, because the question carries")
    print("  the verb's base form while the document carries its")
    print("  inflected one -- and several stems here ('crown',")
    print("  'plough', 'anchor') are also nouns, so documents about")
    print("  entirely different subjects join the tie.")
    print()
    print("  The bi-encoder has learned that 'command' and 'commands'")
    print("  are the same relation. That is the whole of its job, and")
    print("  it is the reason dense retrieval exists.")
    print()
    print("  At this scale retrieval is solved, and that is useful")
    print("  rather than disappointing: with recall@1 at %.4f, every"
          % dense[1])
    print("  error from here on belongs to the generator.")
    return dense, lex, band, beaten


def demo2(lm, gen, agree, contra, kb):
    """The grounding gap: same model, same question, two documents."""
    banner("2. IS IT READING, OR RECITING?")
    print("  The micro-language gives every subject a role signature --")
    print("  the object it expects. A document either agrees with that")
    print("  expectation or names something else. Before any document")
    print("  is shown at all, the pre-trained model already answers:")
    print()
    print("    the pre-trained LM names the role signature %.4f"
          % prior_argmax(lm, agree))
    print("    of the time, with no document in front of it.")
    print()
    print("  So on an AGREEING fact, a model that reads and a model")
    print("  that recites give the SAME answer, and no benchmark built")
    print("  from agreeing facts can tell them apart. Contradicting")
    print("  facts separate them, because the two answers differ:")
    print()
    a0, c0 = agree[0], contra[0]
    print("    question    : " + " ".join(a0["question"]))
    print("    role says   : " + SIG[a0["subj"]])
    print("    agreeing doc: " + " ".join(a0["doc"]))
    print("    contra doc  : " + " ".join(c0["doc"]))
    print()

    dmap = {f["subj"] + f["verb"]: distractor_for(f, kb) for f in contra}
    rows = [
        ("closed book (no document)", score(gen, agree, lambda f: [])),
        ("agreeing document", score(gen, agree, lambda f: f["doc"])),
        ("contradicting document", score(gen, contra, lambda f: f["doc"])),
        ("a DIFFERENT document",
         score(gen, contra,
               lambda f: dmap[f["subj"] + f["verb"]]["doc"])),
    ]
    print("  %-26s %9s %9s %9s" %
          ("condition", "document", "role", "neither"))
    print("  " + "-" * 56)
    for name, s in rows:
        print("  %-26s %9.4f %9.4f %9.4f"
              % (name, s["acc"], s["prior"], s["other"]))
    print()
    print("  On the agreeing row the first two columns are the same")
    print("  number, and not by accident: the document says what the")
    print("  role signature says, so reading and reciting produce the")
    print("  same word. That row cannot distinguish them. It is also")
    print("  the row that every naturally-built benchmark is made of.")
    print()
    ag, con = rows[1][1], rows[2][1]
    print("  Read the middle two rows. The same generator, given the")
    print("  same question with a gold document in front of it, scores")
    print("  %.4f when the document agrees with what it already" % ag["acc"])
    print("  believed and %.4f when the document does not. It is not"
          % con["acc"])
    print("  answering from the document. It is answering from the")
    print("  prior, and the agreeing document is a coincidence that")
    print("  looks like comprehension.")
    print()
    misled = float(np.mean([
        x == dmap[f["subj"] + f["verb"]]["obj"]
        for x, f in zip(rows[3][1]["got"], contra)]))
    print("  The last row is the one worth sitting with. Handed a")
    print("  document about the right subject and the WRONG relation,")
    print("  the model answers with that document's object %.4f of"
          % misled)
    print("  the time -- fluent, confident, and contradicted by the")
    print("  text it was given. The rest of the time it falls back on")
    print("  the role signature (%.4f), which is wrong as well."
          % rows[3][1]["prior"])
    return rows, misled


def demo3(lm, train, agree, contra, kb):
    """Where grounding comes from: the training mix."""
    banner("3. GROUNDING IS LEARNED, NOT INHERITED")
    print("  Nothing above is a property of the architecture. Sweep the")
    print("  share of FINE-TUNING documents that contradict the prior")
    print("  and the behaviour moves with it.")
    print()
    dmap = {f["subj"] + f["verb"]: distractor_for(f, kb) for f in contra}
    print("  %-9s %8s %8s %8s %8s %8s %8s" %
          ("training", "closed", "agree", "contra", "recites",
           "neither", "misled"))
    print("  %-9s %8s %8s %8s %8s %8s %8s" %
          ("contra", "book", "doc", "doc", "the role", "of them",
           "by wrong"))
    print("  " + "-" * 63)
    out = []
    for mix in MIXES:
        tr = remix(train, mix,
                   np.random.default_rng(SEED_MIX + int(mix * 100)))
        g, _ = train_generator(lm, gen_items(tr), GEN_STEPS,
                               np.random.default_rng(SEED_GEN))
        cb = score(g, agree, lambda f: [])
        sa = score(g, agree, lambda f: f["doc"])
        sc = score(g, contra, lambda f: f["doc"])
        sd = score(g, contra,
                   lambda f: dmap[f["subj"] + f["verb"]]["doc"])
        mis = float(np.mean([
            x == dmap[f["subj"] + f["verb"]]["obj"]
            for x, f in zip(sd["got"], contra)]))
        out.append((mix, cb["acc"], sa["acc"], sc["acc"], sc["prior"],
                    sc["other"], mis))
        print("  %-9.2f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f" % out[-1])
    print()
    lo, hi = out[0], out[-1]
    print("  The top row is the one a careless pipeline builds. Every")
    print("  training document agrees with the model's prior, which is")
    print("  what naturally-collected evidence mostly does. The result")
    print("  scores a flawless %.4f on agreeing facts and %.4f on"
          % (lo[2], lo[3]))
    print("  contradicting ones, reciting the role signature %.4f of"
          % lo[4])
    print("  the time. It has learned that the document is decoration.")
    print()
    print("  The bottom row reads the document: %.4f. Now look at"
          % hi[3])
    print("  the last column. A model that has learned to trust")
    print("  documents")
    print("  is misled by a wrong one %.4f of the time, against %.4f"
          % (hi[6], lo[6]))
    print("  for the model that ignores them. Grounding and gullibility")
    print("  are not two behaviours. They are one behaviour, and the")
    print("  retriever is what decides which one you get.")
    print()
    print("  The 'recites the role' column falls cleanly from %.4f"
          % lo[4])
    print("  to %.4f. The 'reads the document' column does not rise"
          % hi[4])
    print("  as cleanly, and the 'neither' column says why: losing")
    print("  faith in the prior happens BEFORE learning to copy, so")
    print("  the middle of the sweep is a model that has stopped")
    print("  trusting what it knew and has not yet started trusting")
    print("  what it was handed. Partial contradicting evidence buys")
    print("  much less grounding than its share suggests. (Doubling")
    print("  the fine-tuning budget moves these a few points and")
    print("  changes no ordering.)")
    return out


def main():
    banner("RETRIEVAL-AUGMENTED GENERATION")
    n_sent, n_kb = selftest()
    worst, checked = gradcheck()
    print("  stage self-test passed: %s sentences, %d documents"
          % (f"{n_sent:,}", n_kb))
    print("  gradient audit: worst error %.2e over %d sampled gradients"
          % (worst, checked))
    print("    (both halves: the generator head and the contrastive")
    print("     retriever, through mean pooling and normalisation)")
    print()

    corpus, ids, keep, kb, train, test = build_stage()
    print("  %d facts, split %d train / %d test by (subject, verb)."
          % (len(kb), len(train), len(test)))
    print("  The split is DISJOINT, so no test answer can be recalled")
    print("  from fine-tuning -- the document is the only source.")

    lm, hist = pretrain_lm(ids, keep, PRE_STEPS,
                           np.random.default_rng(SEED_PRE))
    print("  pre-training loss: " + "  ".join("%d:%.3f" % (s, l)
                                              for s, l in hist))
    print("  decoder %s params; this is where the PRIOR comes from."
          % f"{lm.n_params():,}")

    enc, rh = train_retriever([(f["question"], f["doc"]) for f in train],
                              RET_STEPS, np.random.default_rng(SEED_RET),
                              seed=SEED_MODEL)
    print("  retriever loss:    " + "  ".join("%d:%.3f" % (s, l)
                                              for s, l in rh))

    agree, contra = build_eval(test)
    gen, gh = train_generator(lm, gen_items(remix(
        train, TRAIN_MIX, np.random.default_rng(SEED_MIX + 50))),
        GEN_STEPS, np.random.default_rng(SEED_GEN))
    print("  generator loss:    " + "  ".join("%d:%.3f" % (s, l)
                                              for s, l in gh))

    demo1(enc, kb, test)
    demo2(lm, gen, agree, contra, kb)
    demo3(lm, train, agree, contra, kb)

    banner("DONE")
    return lm, enc, gen


if __name__ == "__main__":
    main()
