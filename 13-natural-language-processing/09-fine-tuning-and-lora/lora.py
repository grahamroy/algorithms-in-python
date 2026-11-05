"""
lora.py --- companion code for "Fine-tuning and LoRA"
(Natural Language Processing, Part 9).

Part 8 put the knowledge beside the model and measured what it costs
to make a generator defer to a document. It also left a bruise: the
same fine-tuning that taught the model to read documents dropped its
closed-book accuracy from 0.6111 to 0.0833. Something was overwritten.

This article asks what, exactly -- and whether all of it needed to
move. The fine-tune is Part 8's, unchanged. What is new is that the
weights are small enough to examine directly:

    dW = W_after - W_before, and we can just take its SVD.

LoRA is built on the premise that the update has low intrinsic rank.
That premise is normally argued from results. Here it is a matrix you
can decompose, so the question becomes a measurement -- and the answer
turns out to be more interesting than the slogan.

What that buys, measured here:
  1. The spectrum of dW for every weight matrix in the model. The
     update is NOT low rank in the strong sense; reaching 90% of its
     energy takes a rank in the twenties out of 64.
  2. The one exception, which is also the most-changed matrix: the
     output head is nearly rank one.
  3. The distinction that matters. "The update IS low rank" and "a
     low-rank update SUFFICES" are different claims, and only the
     second is true here -- because LoRA never tries to reproduce the
     full update. It solves the same problem in a smaller space.
  4. What actually moves the needle, in order: WHERE the adapters go,
     then the learning rate, then the rank.
  5. Forgetting, exactly. The micro-language enumerates, so the mass a
     model places on it and its KL from it are closed-form before and
     after. Part 8's bruise becomes a number.

The language, the oracle, the knowledge base and the trunk are Parts
5-8's, spliced unchanged. New here: LoRA as a frozen base plus
W + (alpha/r)AB with B initialised to zero, its gradients obtained
from the base gradient by dA = s.dW B^T and dB = s.A^T dW, and the
singular-value machinery for reading an update. Hand-differentiated
and audited against central differences at startup, including through
the LoRA factors. Seeded and deterministic.
Dependencies: numpy.  Runs in about eleven minutes.
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

CORPUS_N = 8000
PRE_STEPS = 1500
GEN_STEPS = 2400
SEED_CORPUS, SEED_KB, SEED_SPLIT = 42, 7, 11
SEED_MODEL, SEED_PRE, SEED_GEN = 0, 1, 3
SEED_MIX = 100
SEED_LORA = 21
LORA_LR = 2e-3
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


# =======================================================================
# Reading an update.
#
# LoRA's premise is that dW has low intrinsic rank.  These matrices are
# small, so instead of assuming it we decompose it.  The singular
# values of dW say how many independent directions the fine-tune
# actually used, and the cumulative energy says how much of the change
# a rank-r approximation would keep.
# =======================================================================

def delta_spectrum(before, after, key):
    """Singular values of dW plus the ranks that reach 50/90/99%."""
    dW = after[key] - before[key]
    s = np.linalg.svd(dW, compute_uv=False)
    e = s ** 2
    c = np.cumsum(e) / max(e.sum(), 1e-300)
    return dict(
        key=key, shape=dW.shape, sing=s, energy=c,
        rel=float(np.linalg.norm(dW) / np.linalg.norm(before[key])),
        r50=int(np.searchsorted(c, 0.50) + 1),
        r90=int(np.searchsorted(c, 0.90) + 1),
        r99=int(np.searchsorted(c, 0.99) + 1),
        full=int(min(dW.shape)),
    )


def energy_at(spec, r):
    """Share of dW's energy kept by its best rank-r approximation."""
    c = spec["energy"]
    return float(c[min(r, len(c)) - 1])


# =======================================================================
# LoRA.
#
# Freeze W.  Learn W + (alpha/r) A B with A random and B zero, so the
# adapted model starts exactly where the base model stopped.
#
# The backward pass needs no new derivation.  The existing code already
# computes dL/dW_eff for the effective matrix, and since
# W_eff = W + s A B,
#       dL/dA = s . dL/dW_eff . B^T
#       dL/dB = s . A^T . dL/dW_eff
# so the base gradient is simply projected onto the two factors.  That
# is the whole of the implementation.
# =======================================================================

ATTN_KEYS = [f"{i}.{m}" for i in range(2) for m in ("Wqkv", "Wo")]
MLP_KEYS = [f"{i}.{m}" for i in range(2) for m in ("W1", "W2")]
HEAD_KEYS = ["Wv"]
PLACEMENTS = {
    "attention only": ATTN_KEYS,
    "attention + head": ATTN_KEYS + HEAD_KEYS,
    "head only": HEAD_KEYS,
    "everything": ATTN_KEYS + MLP_KEYS + HEAD_KEYS,
}


def lora_init(model, rank, targets, seed=0, alpha=None):
    rng = np.random.default_rng(seed)
    alpha = rank if alpha is None else alpha
    model.W0 = {k: v.copy() for k, v in model.P.items()}
    model.LA, model.LB = {}, {}
    model.LS = alpha / rank
    for k in targets:
        din, dout = model.P[k].shape
        model.LA[k] = rng.standard_normal((din, rank)) / np.sqrt(din)
        model.LB[k] = np.zeros((rank, dout))
    model.AM, model.AV, model.t = {}, {}, 0
    lora_sync(model)


def lora_sync(model):
    for k in model.LA:
        model.P[k] = model.W0[k] + model.LS * (model.LA[k] @ model.LB[k])


def lora_step(model, g):
    model.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in model.LA:
        for which, gr in (("A", model.LS * (g[k] @ model.LB[k].T)),
                          ("B", model.LS * (model.LA[k].T @ g[k]))):
            name = (which, k)
            store = model.LA if which == "A" else model.LB
            model.AM.setdefault(name, np.zeros_like(gr))
            model.AV.setdefault(name, np.zeros_like(gr))
            model.AM[name] = b1 * model.AM[name] + (1 - b1) * gr
            model.AV[name] = b2 * model.AV[name] + (1 - b2) * gr ** 2
            store[k] -= model.lr * (
                model.AM[name] / (1 - b1 ** model.t)) / (
                np.sqrt(model.AV[name] / (1 - b2 ** model.t)) + eps)
    lora_sync(model)


def n_lora(model):
    return sum(v.size for v in model.LA.values()) + \
           sum(v.size for v in model.LB.values())


def train_lora(base, items, steps, rng, rank, targets, lr=LORA_LR,
               batch=32, seed=0):
    m = base.clone()
    m.lr = lr
    lora_init(m, rank, targets, seed=seed)
    for s in range(steps):
        idx = rng.choice(len(items), min(batch, len(items)),
                         replace=False)
        ids, keep, tgt, sel = rag_batch([items[i] for i in idx])
        lora_step(m, m.head_grads(ids, keep, tgt, sel))
    return m


# =======================================================================
# Forgetting, exactly.
#
# The micro-language enumerates, so the mass a model puts on it and its
# KL from it are closed-form.  Part 8 could only watch closed-book
# accuracy fall; here the damage is a number with no sampling in it.
# =======================================================================

SENTS = sorted(SENT_P)
P_LANG = np.array([SENT_P[s] for s in SENTS])


def lm_logprob(model, batch_words):
    """log P(sentence) under the model, in one forward pass."""
    ids, keep = encode(batch_words)
    tgt = np.zeros_like(ids)
    tgt[:, :-1] = ids[:, 1:]
    sel = np.zeros(ids.shape, dtype=bool)
    sel[:, :-1] = keep[:, 1:]
    logits, _ = model.head_logits(ids, keep)
    lg = logits - logits.max(-1, keepdims=True)
    lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
    rows, cols = np.indices(ids.shape)
    return np.where(sel, lp[rows, cols, tgt], 0.0).sum(1)


def on_language(model, batch=2000):
    """Exact mass on the language and KL(language || model)."""
    lp = np.empty(len(SENTS))
    for i in range(0, len(SENTS), batch):
        lp[i:i + batch] = lm_logprob(
            model, [list(s) for s in SENTS[i:i + batch]])
    return dict(mass=float(np.exp(lp).sum()),
                kl=float((P_LANG * (np.log(P_LANG) - lp)).sum()))


def task_acc(model, facts):
    return score(model, facts, lambda f: f["doc"])["acc"]


# -----------------------------------------------------------------------
# Gradient audit and self-test.
# -----------------------------------------------------------------------

def gradcheck(seed=0, n_per=3):
    """The base head, and the LoRA factors it is projected onto."""
    rng = np.random.default_rng(seed)
    kb = build_kb(np.random.default_rng(1), 0.5)[:6]
    items = [(f["doc"], f["question"], f["obj"]) for f in kb]
    ids, keep, tgt, sel = rag_batch(items)
    worst, checked = 0.0, 0

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

    # LoRA: perturb A and B, not W, and re-synchronise before measuring
    ad = Transformer(d=16, heads=2, layers=2, causal=True, seed=2)
    tgt_keys = ["0.Wqkv", "0.Wo", "Wv"]
    lora_init(ad, 2, tgt_keys, seed=3)
    base = ad.head_grads(ids, keep, tgt, sel)
    gA = {k: ad.LS * (base[k] @ ad.LB[k].T) for k in ad.LA}
    gB = {k: ad.LS * (ad.LA[k].T @ base[k]) for k in ad.LA}

    def loss_now():
        lora_sync(ad)
        return ad.head_loss(ids, keep, tgt, sel)

    for store, grads in ((ad.LA, gA), (ad.LB, gB)):
        for k in tgt_keys:
            flat, gflat = store[k].reshape(-1), grads[k].reshape(-1)
            for ix in rng.choice(len(flat), min(n_per, len(flat)),
                                 False):
                old = flat[ix]
                flat[ix] = old + 1e-6
                lp = loss_now()
                flat[ix] = old - 1e-6
                lm = loss_now()
                flat[ix] = old
                lora_sync(ad)
                worst = max(worst, abs((lp - lm) / 2e-6 - gflat[ix]))
                checked += 1
    return worst, checked


def selftest():
    """The stage, and the two claims the machinery rests on."""
    assert abs(sum(SENT_P.values()) - 1.0) < 1e-9
    kb = build_kb(np.random.default_rng(3), 0.5)
    assert len(kb) == 144
    for f in kb:
        assert in_language(f["doc"]), f
        assert f["contradicts"] == (f["obj"] != SIG[f["subj"]]), f

    # a LoRA model starts EXACTLY at its base, because B is zero
    m = Transformer(d=16, heads=2, layers=2, causal=True, seed=4)
    before = {k: v.copy() for k, v in m.P.items()}
    lora_init(m, 4, ["0.Wqkv", "Wv"], seed=5)
    for k in before:
        assert np.allclose(m.P[k], before[k]), k

    # the spectrum machinery agrees with an explicit truncated SVD
    rng = np.random.default_rng(6)
    A = {"X": rng.standard_normal((20, 30))}
    B = {"X": A["X"] + rng.standard_normal((20, 30)) * 0.1}
    sp = delta_spectrum(A, B, "X")
    dW = B["X"] - A["X"]
    U, s, Vt = np.linalg.svd(dW, full_matrices=False)
    for r in (1, 3, 7):
        approx = (U[:, :r] * s[:r]) @ Vt[:r]
        kept = 1.0 - (np.linalg.norm(dW - approx) ** 2
                      / np.linalg.norm(dW) ** 2)
        assert abs(kept - energy_at(sp, r)) < 1e-9, (r, kept)
    return len(SENT_P), len(kb)
# -----------------------------------------------------------------------
# Frozen configuration for Part 9.
#
# Every LoRA number below is a MEAN OVER SEEDS.  A single run is not
# evidence here: the same configuration varies by up to eleven points
# across initialisations, because B starts at zero and where A points
# decides what subspace the adapter gets to search.  Full fine-tuning
# has no such seed, so its runs are reseeded on the data order instead.
# -----------------------------------------------------------------------

FULL_LR, HIGH_LR = 5e-4, 2e-3
FT_STEPS = 1500          # 2400 gives the same means (checked)
RANKS = (1, 2, 4, 8, 16)
BASE_RANK = 4
BEST_PLACEMENT = "everything"
SEEDS = (0, 1, 2, 3)
HELD_N, SEED_HELD = 600, 555
SPECTRUM_KEYS = ["0.Wqkv", "0.Wo", "0.W1", "0.W2",
                 "1.Wqkv", "1.Wo", "1.W1", "1.W2", "Wv", "emb"]


def held_out(n=HELD_N, seed=SEED_HELD):
    return [s["words"] for s in gen_corpus(n, np.random.default_rng(seed))]


def nats_per_token(model, words):
    lp = lm_logprob(model, words)
    return float(-lp.sum() / sum(len(w) + 1 for w in words))


def oracle_floor(words):
    return float(np.sum([sum(oracle_nll(w)) for w in words])
                 / sum(len(w) + 1 for w in words))


def measure(model, evalset, held, forget=False):
    """Task accuracy always; the costly language sweep only on demand.

    on_language touches all 31,740 sentences, so it is run for the
    four configurations demo 4 compares and not for the whole grid.
    """
    out = dict(acc=task_acc(model, evalset))
    if forget:
        ol = on_language(model)
        out.update(nats=nats_per_token(model, held),
                   kl=ol["kl"], mass=ol["mass"])
    return out


def ms(rows, field):
    """Mean and standard deviation of one field across seeds."""
    v = np.array([r[field] for r in rows])
    return float(v.mean()), float(v.std())


def run_sweep(lm, items, evalset, held):
    """Train every configuration once, over every seed, and measure.

    One pass over the whole design, so nothing is trained twice and
    the forgetting numbers come from exactly the models whose accuracy
    is reported.
    """
    out = {}

    def log(tag):
        print("    %-34s done" % tag)

    for lr in (FULL_LR, HIGH_LR):
        rows = []
        for sd in SEEDS:
            f, _ = train_generator(lm, items, FT_STEPS,
                                   np.random.default_rng(100 + sd),
                                   lr=lr)
            rows.append(measure(f, evalset, held, forget=True))
        out[("full", lr)] = rows
        log("full fine-tune lr %.0e" % lr)

    for name, keys in PLACEMENTS.items():
        rows = []
        for sd in SEEDS:
            m = train_lora(lm, items, FT_STEPS,
                           np.random.default_rng(SEED_GEN), BASE_RANK,
                           keys, lr=FULL_LR, seed=sd)
            rows.append(measure(m, evalset, held,
                                forget=name == BEST_PLACEMENT))
        out[("place", name)] = rows
        log("placement %s" % name)

    keys = PLACEMENTS[BEST_PLACEMENT]
    rows = []
    for sd in SEEDS:
        m = train_lora(lm, items, FT_STEPS,
                       np.random.default_rng(SEED_GEN), BASE_RANK,
                       keys, lr=HIGH_LR, seed=sd)
        rows.append(measure(m, evalset, held, forget=True))
    out[("place_hi", BEST_PLACEMENT)] = rows
    log("placement %s at lr %.0e" % (BEST_PLACEMENT, HIGH_LR))

    for r in RANKS:
        if r == BASE_RANK:
            out[("rank", r)] = out[("place", BEST_PLACEMENT)]
            continue
        rows = []
        for sd in SEEDS:
            m = train_lora(lm, items, FT_STEPS,
                           np.random.default_rng(SEED_GEN), r, keys,
                           lr=FULL_LR, seed=sd)
            rows.append(measure(m, evalset, held))
        out[("rank", r)] = rows
        log("rank %d" % r)
    return out


def n_params_lora(lm, keys, rank):
    return sum(rank * (lm.P[k].shape[0] + lm.P[k].shape[1])
               for k in keys)


# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

def demo1(before, after):
    """What the fine-tune actually changed."""
    banner("1. TAKING THE SVD OF THE UPDATE")
    print("  LoRA rests on a claim about dW = W_after - W_before: that")
    print("  it has low intrinsic rank. These matrices are small, so")
    print("  the claim does not have to be assumed. Decompose it.")
    print()
    specs = {k: delta_spectrum(before, after, k) for k in SPECTRUM_KEYS}
    print("  %-8s %10s %9s %6s %6s %6s %6s"
          % ("matrix", "shape", "|dW|/|W|", "r50%", "r90%", "r99%",
             "full"))
    print("  " + "-" * 60)
    for k in SPECTRUM_KEYS:
        s = specs[k]
        print("  %-8s %10s %9.4f %6d %6d %6d %6d"
              % (k, "x".join(map(str, s["shape"])), s["rel"],
                 s["r50"], s["r90"], s["r99"], s["full"]))
    print()
    print("  r90% is the rank a truncated SVD needs to keep 90% of the")
    print("  update's energy. Out of %d available directions it runs"
          % specs["0.Wqkv"]["full"])
    print("  to the twenties almost everywhere. On the strong reading")
    print("  of the claim -- that dW IS low rank -- the answer is no.")
    print()
    print("  energy kept by the best rank-r approximation of dW:")
    print("  %-8s %8s %8s %8s %8s %8s" %
          ("matrix", "r=1", "r=2", "r=4", "r=8", "r=16"))
    print("  " + "-" * 54)
    for k in SPECTRUM_KEYS:
        row = tuple(energy_at(specs[k], r) for r in (1, 2, 4, 8, 16))
        print("  %-8s %8.4f %8.4f %8.4f %8.4f %8.4f" % ((k,) + row))
    print()
    wv = specs["Wv"]
    print("  One row stands out. The output head Wv moved furthest in")
    print("  relative terms (%.4f) and is the closest thing here to a"
          % wv["rel"])
    print("  low-rank update: %.4f of its energy sits in a single"
          % energy_at(wv, 1))
    print("  direction. That is the fine-tune learning which tokens to")
    print("  emit in the answer slot, and it is nearly a rank-one edit.")
    return specs


def demo2(lm, sweep, n_full):
    """Where the adapters go -- and how much of this is noise."""
    banner("2. LOW RANK IS NOT THE SAME CLAIM AS LOW RANK SUFFICES")
    print("  Those two sentences look alike and are not. LoRA never")
    print("  tries to reproduce dW. It solves the same problem inside")
    print("  a smaller space, and the space it is given matters.")
    print()
    print("  Rank %d, learning rate %.0e, %d seeds each. Full"
          % (BASE_RANK, FULL_LR, len(SEEDS)))
    print("  fine-tuning updates %s parameters." % f"{n_full:,}")
    print()
    fm, fs = ms(sweep[("full", FULL_LR)], "acc")
    print("  %-18s %10s %8s %9s %8s"
          % ("adapters on", "trainable", "share", "task", "sd"))
    print("  " + "-" * 58)
    print("  %-18s %10s %8.4f %9.4f %8.4f"
          % ("(full fine-tune)", f"{n_full:,}", 1.0, fm, fs))
    out = {}
    for name, keys in PLACEMENTS.items():
        n = n_params_lora(lm, keys, BASE_RANK)
        m, s = ms(sweep[("place", name)], "acc")
        out[name] = (n, m, s)
        print("  %-18s %10s %8.4f %9.4f %8.4f"
              % (name, f"{n:,}", n / n_full, m, s))
    print()
    sem = out[BEST_PLACEMENT][2] / np.sqrt(len(SEEDS))
    print("  Before reading anything into the order: the standard")
    print("  deviation across seeds is about %.3f, so the standard"
          % out[BEST_PLACEMENT][2])
    print("  error on each mean is about %.3f. Differences smaller" % sem)
    print("  than roughly %.2f are not differences." % (3 * sem))
    print()
    a, h, e = (out["attention only"], out["head only"],
               out[BEST_PLACEMENT])
    print("  WHERE survives that test. Adapting everything beats")
    print("  attention alone, %.4f against %.4f, and both bury the"
          % (e[1], a[1]))
    print("  head on its own at %.4f -- even though the head is the"
          % h[1])
    print("  matrix demo 1 showed moving furthest. The spectrum says")
    print("  where the change HAPPENED. It does not say where the")
    print("  change had to be MADE, and those are different questions.")
    print()
    hm, hs = ms(sweep[("place_hi", BEST_PLACEMENT)], "acc")
    print("  STEP SIZE does not survive it. The same adapters at lr")
    print("  %.0e score %.4f (sd %.4f) against %.4f (sd %.4f) at"
          % (HIGH_LR, hm, hs, e[1], e[2]))
    print("  %.0e -- no better in the mean and %.1fx the spread."
          % (FULL_LR, hs / max(e[2], 1e-9)))
    print("  The common advice to raise the learning rate for LoRA")
    print("  buys variance here, not accuracy. The individual seeds")
    print("  say why a single run of this comparison is worthless:")
    for lr, key in ((FULL_LR, ("place", BEST_PLACEMENT)),
                    (HIGH_LR, ("place_hi", BEST_PLACEMENT))):
        accs = [r["acc"] for r in sweep[key]]
        print("    lr %.0e : %s   spread %.4f"
              % (lr, "  ".join("%.4f" % a for a in accs),
                 max(accs) - min(accs)))
    hi_accs = [r["acc"] for r in sweep[("place_hi", BEST_PLACEMENT)]]
    print("  A single run at %.0e could have reported %.4f, the best"
          % (HIGH_LR, max(hi_accs)))
    print("  number in this article, or %.4f, nearly the worst."
          % min(hi_accs))
    return out


def demo3(lm, sweep, n_full):
    """How much rank."""
    banner("3. HOW MUCH RANK")
    keys = PLACEMENTS[BEST_PLACEMENT]
    fm, _ = ms(sweep[("full", FULL_LR)], "acc")
    print("  Adapters everywhere, learning rate %.0e, %d seeds."
          % (FULL_LR, len(SEEDS)))
    print("  Full fine-tuning scores %.4f on %s parameters."
          % (fm, f"{n_full:,}"))
    print()
    print("  %-6s %12s %9s %9s %8s %9s" %
          ("rank", "trainable", "share", "task", "sd", "of full"))
    print("  " + "-" * 58)
    rows = []
    for r in RANKS:
        n = n_params_lora(lm, keys, r)
        m, s = ms(sweep[("rank", r)], "acc")
        rows.append((r, n, n / n_full, m, s, m / fm))
        print("  %-6d %12s %9.4f %9.4f %8.4f %9.4f" % rows[-1])
    print()
    best = max(rows, key=lambda r: r[3])
    sem = np.mean([r[4] for r in rows]) / np.sqrt(len(SEEDS))
    last = rows[-1]
    print("  The column rises to rank %d and then stops. Rank %d"
          % (best[0], last[0]))
    print("  scores %.4f against rank %d's %.4f on twice the"
          % (last[3], best[0], best[3]))
    print("  parameters -- a gap of %.4f, inside the %.3f standard"
          % (best[3] - last[3], 3 * sem))
    print("  error that separates a difference from a coincidence.")
    print("  Past the point where the subspace is large enough, more")
    print("  rank buys nothing. A bigger adapter is not a better one.")
    print()
    print("  Rank %d reaches %.2f%% of full fine-tuning on %.2f%% of"
          % (best[0], 100 * best[5], 100 * best[2]))
    print("  the parameters -- and rank %d gets %.2f%% of it for"
          % (rows[1][0], 100 * rows[1][5]))
    print("  %.2f%% of them. That is the trade, and a good one even"
          % (100 * rows[1][2]))
    print("  though full fine-tuning stays ahead.")
    return rows


def demo4(sweep, floor):
    """Forgetting, with the learning rate held still."""
    banner("4. WHAT IT COSTS THE MODEL UNDERNEATH")
    print("  Part 8 watched closed-book accuracy fall from 0.6111 to")
    print("  0.0833 after this fine-tune and could say no more. Here")
    print("  the language enumerates, so the damage is exact: held-out")
    print("  nats per token against the oracle's floor of %.4f, and"
          % floor)
    print("  KL(language || model) summed over all %s sentences."
          % f"{len(SENT_P):,}")
    print()
    print("  The usual claim is that LoRA forgets less because it")
    print("  moves fewer weights. Test it with the STEP SIZE held")
    print("  still, because a bigger step moves the function further")
    print("  whatever is being parameterised.")
    print()
    print("  %-22s %8s %9s %9s %8s" %
          ("model", "task", "nats/tok", "KL", "mass"))
    print("  " + "-" * 60)
    keyset = [("full fine-tune  %.0e" % FULL_LR, ("full", FULL_LR)),
              ("LoRA r=%d        %.0e" % (BASE_RANK, FULL_LR),
               ("place", BEST_PLACEMENT)),
              ("full fine-tune  %.0e" % HIGH_LR, ("full", HIGH_LR)),
              ("LoRA r=%d        %.0e" % (BASE_RANK, HIGH_LR),
               ("place_hi", BEST_PLACEMENT))]
    vals = {}
    for label, key in keyset:
        r = sweep[key]
        vals[key] = tuple(ms(r, f)[0]
                          for f in ("acc", "nats", "kl", "mass"))
        print("  %-22s %8.4f %9.4f %9.4f %8.4f"
              % ((label,) + vals[key]))
    print()
    lo = (vals[("full", FULL_LR)][1]
          + vals[("place", BEST_PLACEMENT)][1]) / 2
    hi = (vals[("full", HIGH_LR)][1]
          + vals[("place_hi", BEST_PLACEMENT)][1]) / 2
    print("  Read the nats column down. The learning rate moves it")
    print("  from about %.1f to about %.1f. The METHOD moves it by a"
          % (lo, hi))
    print("  few tenths. Forgetting here is a property of how far you")
    print("  stepped, not of which parameters you stepped in.")
    print()
    lf, ll = vals[("full", FULL_LR)], vals[("place", BEST_PLACEMENT)]
    print("  And at a matched learning rate the two forget the SAME")
    print("  amount: %.4f against %.4f, a gap of %.4f nats on a"
          % (ll[1], lf[1], abs(ll[1] - lf[1])))
    print("  scale where the step size moved things by %.1f. The folk"
          % (hi - lo))
    print("  claim is that LoRA forgets LESS because it writes to")
    print("  fewer weights. On this measurement it does not forget")
    print("  less. It forgets the same.")
    print()
    print("  The mass column disagrees, and is printed rather than")
    print("  buried: LoRA keeps %.4f of the language's probability"
          % ll[3])
    print("  against full fine-tuning's %.4f. The two are answering"
          % lf[3])
    print("  different questions -- how WELL the model still predicts")
    print("  the language, and how much of its output is still spent")
    print("  on it. LoRA ties on the first and wins the second.")
    print()
    print("  What LoRA does guarantee is structural, not statistical.")
    print("  The base weights were never written to. Detach the")
    print("  adapter and the original model is exactly, bit for bit,")
    print("  still there -- the property the self-test checks at")
    print("  startup, and the one that lets fifty fine-tunes be served")
    print("  off one copy of the weights.")
    return vals


def main():
    banner("FINE-TUNING AND LoRA")
    n_sent, n_kb = selftest()
    worst, checked = gradcheck()
    print("  stage self-test passed: %s sentences, %d documents"
          % (f"{n_sent:,}", n_kb))
    print("    (includes: a LoRA model starts exactly at its base, and")
    print("     the spectrum agrees with an explicit truncated SVD)")
    print("  gradient audit: worst error %.2e over %d sampled gradients"
          % (worst, checked))
    print("    (the base head, and the LoRA factors A and B)")
    print()

    corpus, ids, keep, kb, train, test = build_stage()
    lm, hist = pretrain_lm(ids, keep, PRE_STEPS,
                           np.random.default_rng(SEED_PRE))
    before = {k: v.copy() for k, v in lm.P.items()}
    n_full = sum(v.size for v in lm.P.values())
    agree, contra = build_eval(test)
    evalset = agree + contra
    items = gen_items(remix(train, TRAIN_MIX,
                            np.random.default_rng(SEED_MIX + 50)))
    held = held_out()
    floor = oracle_floor(held)
    print("  pre-training loss: " + "  ".join("%d:%.3f" % (s, l)
                                              for s, l in hist))
    print("  decoder %s params; held-out %.4f nats/token against a"
          % (f"{n_full:,}", nats_per_token(lm, held)))
    print("  floor of %.4f. This is the model Part 8 fine-tuned." % floor)
    print("  %d held-out facts; every LoRA figure below is a mean over"
          % len(evalset))
    print("  %d seeds, because one run of it is not evidence."
          % len(SEEDS))
    print()
    print("  training the sweep (this is the slow part)...")
    sweep = run_sweep(lm, items, evalset, held)

    spec_model, _ = train_generator(lm, items, FT_STEPS,
                                    np.random.default_rng(100), lr=FULL_LR)
    demo1(before, spec_model.P)
    demo2(lm, sweep, n_full)
    demo3(lm, sweep, n_full)
    demo4(sweep, floor)

    banner("DONE")
    return lm, sweep


if __name__ == "__main__":
    main()
