"""
nli.py --- companion code for "Natural Language Inference"
(Natural Language Processing, Part 6).

Parts 1 to 5 judged one sentence at a time: is it likely, is it well
formed, what comes next. This one takes TWO sentences and asks what
the first tells you about the second -- entailment, contradiction, or
neither. That is the task that made the sentence PAIR the unit of
work, and it is where the encoder and the decoder meet again.

The micro-language gets a semantics to make this exact. A sentence
now asserts a polarised fact plus whatever adjectives it carries:

    the noble king rules his crown
        -> RULE(king, crown) AND noble(king)
    the king does not rule his crown
        -> NOT RULE(king, crown)

and entailment is set inclusion over those facts. So the gold label
is COMPUTED, not annotated -- which matters more here than anywhere
else in this series, because the headline result in real NLI is a
story about what annotators did when nobody was checking.

Demonstrates:
  1. The stage: a world, a semantics, and three-way labels that a
     function decides rather than a crowdworker.
  2. Cross-encoder vs bi-encoder: the same trunk reading the pair
     jointly, or reading each side alone and comparing vectors.
  3. Annotation artefacts: a "careless" generator that builds
     contradictions by negating and entailments by dropping a word,
     exactly as SNLI's annotators did -- and a hypothesis-only model
     that solves it without reading the premise at all.
  4. The label as a token: a causal decoder doing the same task by
     predicting a word, which is how GPT-3 does every task.

Everything is plain NumPy, hand-differentiated and audited against
central differences at startup. Seeded and deterministic.
Dependencies: numpy.  Runs in about five minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from collections import defaultdict
from itertools import product

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
# Micro-language from Part 1, with negation added.
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


def plural(noun):
    if noun.endswith("s") or noun.endswith("sh"):
        return noun + "es"
    if noun.endswith("y"):
        return noun[:-1] + "ies"
    return noun + "s"


TOPIC_OF = {}
PLURAL_NOUNS = set()
SG_NOUNS = set()
ADJS = set()
LEMMA = {}                      # verb surface form -> lemma
for _name, _t in TOPICS.items():
    for _n in _t["nouns"]:
        TOPIC_OF[_n] = _name
        SG_NOUNS.add(_n)
        if plural(_n) not in HELD_OUT_PLURALS:
            TOPIC_OF[plural(_n)] = _name
            PLURAL_NOUNS.add(plural(_n))
    for (_vs, _vp) in _t["verbs"]:
        TOPIC_OF[_vs] = TOPIC_OF[_vp] = _name
        LEMMA[_vs] = LEMMA[_vp] = _vp
    for _a in _t["adjs"]:
        TOPIC_OF[_a] = _name
        ADJS.add(_a)

NOUNS = SG_NOUNS | PLURAL_NOUNS
LABELS = ("entailment", "neutral", "contradiction")
LABEL_ID = {l: i for i, l in enumerate(LABELS)}
LABEL_WORD = {"entailment": "yes", "neutral": "maybe",
              "contradiction": "no"}


# -----------------------------------------------------------------------
# A sentence, its surface form, and the facts it asserts.
# -----------------------------------------------------------------------

def possessive(subj, is_plural):
    if is_plural:
        return "their"
    if subj in GENDER_M:
        return "his"
    if subj in GENDER_F:
        return "her"
    return "the"


def make_sentence(topic, base_noun, verb_lemma, obj, is_plural,
                  positive, adj):
    """Surface words plus the facts asserted, as a frozen set."""
    subj = plural(base_noun) if is_plural else base_noun
    vs = [v for v in TOPICS[topic]["verbs"] if v[1] == verb_lemma][0][0]
    poss = possessive(base_noun, is_plural)
    words = ["the"]
    if adj:
        words.append(adj)
    words.append(subj)
    if positive:
        words.append(vs if not is_plural else verb_lemma)
    else:
        words += (["do", "not"] if is_plural else ["does", "not"])
        words.append(verb_lemma)
    words += [poss, obj]
    facts = {("rel", positive, subj, verb_lemma, obj)}
    if adj:
        facts.add(("adj", adj, subj))
    return dict(words=words, topic=topic, subj=subj, obj=obj,
                base=base_noun, verb=verb_lemma, positive=positive,
                adj=adj, is_plural=is_plural, facts=frozenset(facts))


def core(sent):
    """The relation a sentence is about, ignoring polarity."""
    return (sent["subj"], sent["verb"], sent["obj"])


def as_premise(clauses):
    """Join clauses with "and". The premise asserts all of their facts.

    A premise carries TWO clauses and the hypothesis is about one of
    them, which is what makes this harder than comparing two sentences:
    a reader has to find the relevant clause before comparing it.
    """
    words, facts = [], set()
    for i, c in enumerate(clauses):
        if i:
            words.append("and")
        words += c["words"]
        facts |= set(c["facts"])
    return dict(words=words, clauses=clauses, facts=frozenset(facts),
                topic=clauses[0]["topic"])


def entails(p, h):
    """Does every fact the hypothesis asserts follow from the premise?"""
    return set(h["facts"]) <= set(p["facts"])


def contradicts(p, h):
    """Does any clause assert this relation with opposite polarity?"""
    return any(core(c) == core(h) and c["positive"] != h["positive"]
               for c in p["clauses"])


def gold_label(p, h):
    """The label, computed from the semantics -- never annotated."""
    if contradicts(p, h):
        return "contradiction"
    if entails(p, h):
        return "entailment"
    return "neutral"


# -----------------------------------------------------------------------
# Sampling a sentence, and the two pair generators.
#
# BALANCED is built so the hypothesis alone carries no signal: in every
# class it contains a negation exactly half the time and an adjective
# exactly half the time, and premise and hypothesis always share a
# topic.  CARELESS is built the way SNLI actually was -- contradictions
# by negating, entailments by deleting a word, neutrals by swapping one
# -- which leaves the label written all over the hypothesis.
# -----------------------------------------------------------------------

def pick(rng, seq):
    return seq[rng.integers(0, len(seq))]


def sample_sentence(rng, topic=None, positive=None, adj="random",
                    is_plural=None):
    topic = topic or pick(rng, list(TOPICS))
    t = TOPICS[topic]
    base = pick(rng, t["nouns"])
    if is_plural is None:
        is_plural = (rng.uniform() < 0.35
                     and plural(base) not in HELD_OUT_PLURALS)
    elif is_plural and plural(base) in HELD_OUT_PLURALS:
        is_plural = False
    verb = pick(rng, t["verbs"])[1]
    obj_base = pick(rng, t["nouns"])
    obj = obj_base
    if rng.uniform() < 0.25 and plural(obj_base) not in HELD_OUT_PLURALS:
        obj = plural(obj_base)
    if positive is None:
        positive = rng.uniform() < 0.5
    if adj == "random":
        adj = pick(rng, t["adjs"])
    return make_sentence(topic, base, verb, obj, is_plural, positive, adj)


def variant(rng, s, positive=None, adj="keep", change=None):
    """Rebuild a sentence with one thing altered."""
    t = TOPICS[s["topic"]]
    base = s["base"]
    verb, obj, is_plural = s["verb"], s["obj"], s["is_plural"]
    if change == "subj":
        while True:
            nb = pick(rng, t["nouns"])
            npl = (rng.uniform() < 0.35
                   and plural(nb) not in HELD_OUT_PLURALS)
            if (plural(nb) if npl else nb) != s["subj"]:
                base, is_plural = nb, npl
                break
    elif change == "verb":
        while True:
            nv = pick(rng, t["verbs"])[1]
            if nv != verb:
                verb = nv
                break
    elif change == "obj":
        while True:
            ob = pick(rng, t["nouns"])
            no = ob
            if (rng.uniform() < 0.25
                    and plural(ob) not in HELD_OUT_PLURALS):
                no = plural(ob)
            if no != obj:
                obj = no
                break
    if adj == "keep":
        adj = s["adj"]
    elif adj == "random":
        adj = pick(rng, t["adjs"])
    elif adj == "other":
        adj = pick(rng, [a for a in t["adjs"] if a != s["adj"]])
    if positive is None:
        positive = s["positive"]
    return make_sentence(s["topic"], base, verb, obj, is_plural,
                         positive, adj)


def one_clause(rng, positive=None):
    """A premise is a single clause, as in SNLI."""
    c = sample_sentence(rng, positive=positive)
    return as_premise([c]), c


def gen_balanced(rng):
    """One (premise, hypothesis, label) with no hypothesis-only cue.

    Negation and adjectives each appear in exactly half of the
    hypotheses of EVERY class, and the premise always shares its topic,
    so nothing about the hypothesis alone predicts the label.
    """
    while True:
        p, target = one_clause(rng)
        want = LABELS[rng.integers(0, 3)]
        has_adj = rng.uniform() < 0.5
        if want == "entailment":
            h = variant(rng, target,
                        adj=target["adj"] if has_adj else None)
        elif want == "contradiction":
            h = variant(rng, target, positive=not target["positive"],
                        adj="random" if has_adj else None)
        elif has_adj and rng.uniform() < 0.3:
            h = variant(rng, target, adj="other")      # adds information
        else:
            h = variant(rng, target,
                        change=pick(rng, ["subj", "verb", "obj"]),
                        positive=rng.uniform() < 0.5,
                        adj="random" if has_adj else None)
        if gold_label(p, h) == want:
            return p, h, want


def gen_careless(rng):
    """The SNLI recipe: negate, delete, or swap -- and never check."""
    p, target = one_clause(rng, positive=True)
    want = LABELS[rng.integers(0, 3)]
    if want == "entailment":
        h = variant(rng, target, adj=None)             # delete a word
    elif want == "contradiction":
        h = variant(rng, target, positive=False, adj=None)   # add "not"
    else:
        h = variant(rng, target, change="obj")         # swap the object
    if gold_label(p, h) != want:
        return gen_careless(rng)
    return p, h, want


def make_data(n, rng, careless=False):
    gen = gen_careless if careless else gen_balanced
    return [gen(rng) for _ in range(n)]


# -----------------------------------------------------------------------
# Vocabulary and encoding.  A pair is [CLS] premise [SEP] hypothesis
# [SEP], with a segment embedding marking which side each token is on.
# -----------------------------------------------------------------------

PAD, CLS, SEP, MASK = 0, 1, 2, 3
N_SPECIAL = 4


def build_vocab():
    words = {"the", "his", "her", "their", "does", "do", "not",
             "and"}
    words.update(LABEL_WORD.values())
    for t in TOPICS.values():
        for n in t["nouns"]:
            words.add(n)
            if plural(n) not in HELD_OUT_PLURALS:
                words.add(plural(n))
        for (vs, vp) in t["verbs"]:
            words.update([vs, vp])
        words.update(t["adjs"])
    vocab = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] + sorted(words)
    return vocab, {w: i for i, w in enumerate(vocab)}


VOCAB, W2I = build_vocab()
V = len(VOCAB)
T_PAIR = 20        # [CLS] + 8 + [SEP] + 8 + [SEP] + a label slot
T_SINGLE = 10      # [CLS] + 8 + [SEP]


def encode_pair(pairs, with_label=None):
    """(N, T_PAIR) ids, keep mask and segment ids for premise/hypothesis.

    with_label, if given, appends the label WORD after the final [SEP]
    and returns the position it sits at -- that is the decoder's view.
    """
    n = len(pairs)
    ids = np.zeros((n, T_PAIR), dtype=int)
    keep = np.zeros((n, T_PAIR), dtype=bool)
    seg = np.zeros((n, T_PAIR), dtype=int)
    lab_pos = np.zeros(n, dtype=int)
    for i, (p, h) in enumerate(pairs):
        toks = [CLS] + [W2I[w] for w in p] + [SEP]
        segs = [0] * len(toks)
        toks += [W2I[w] for w in h] + [SEP]
        segs += [1] * (len(toks) - len(segs))
        lab_pos[i] = len(toks) - 1
        if with_label is not None:
            toks.append(W2I[LABEL_WORD[with_label[i]]])
            segs.append(1)
        ids[i, :len(toks)] = toks
        keep[i, :len(toks)] = True
        seg[i, :len(segs)] = segs
    return ids, keep, seg, lab_pos


def encode_single(sents):
    """(N, T_SINGLE) ids and keep mask for one sentence at a time."""
    n = len(sents)
    ids = np.zeros((n, T_SINGLE), dtype=int)
    keep = np.zeros((n, T_SINGLE), dtype=bool)
    for i, ws in enumerate(sents):
        toks = [CLS] + [W2I[w] for w in ws] + [SEP]
        ids[i, :len(toks)] = toks
        keep[i, :len(toks)] = True
    return ids, keep


def split_xy(data):
    pairs = [(p["words"], h["words"]) for p, h, _ in data]
    y = np.array([LABEL_ID[l] for _, _, l in data])
    return pairs, y


# -----------------------------------------------------------------------
# The trunk, carried over from Part 5 with one addition: a SEGMENT
# embedding, so a token knows which side of the pair it is on.  That is
# BERT's third embedding table and it is three lines of code.
# -----------------------------------------------------------------------

class Transformer:
    def __init__(self, d=64, heads=4, layers=2, d_ff=None, causal=False,
                 seed=0, lr=1e-3, n_cls=3, pool="cls", t_max=T_PAIR):
        r = np.random.default_rng(seed)
        self.d, self.h, self.L = d, heads, layers
        self.dk = d // heads
        self.dff = d_ff or 4 * d
        self.causal = causal
        self.lr = lr
        self.pool = pool
        self.t_max = t_max

        def rnd(*shape, s=1.0):
            return r.standard_normal(shape) * np.sqrt(s / shape[0])

        P = {"emb": rnd(V, d), "pos": rnd(t_max, d), "seg": rnd(2, d)}
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
        P["Wc"] = rnd(d, n_cls)             # cross-encoder head
        P["bc"] = np.zeros(n_cls)
        P["Wb"] = rnd(4 * d, n_cls)         # bi-encoder head
        P["bb"] = np.zeros(n_cls)
        P["Wv"] = rnd(d, V)                 # the decoder's token head
        P["bv"] = np.zeros(V)
        self.P = P
        self.reset_adam()

    def reset_adam(self):
        self.M = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.Vm = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def n_params(self, keys=None):
        return sum(v.size for k, v in self.P.items()
                   if keys is None or k in keys)

    def zero_grads(self):
        return {k: np.zeros_like(v) for k, v in self.P.items()}

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
    def trunk_fwd(self, ids, keep, seg=None):
        N, T = ids.shape
        x = self.P["emb"][ids] + self.P["pos"][:T]
        if seg is not None:
            x = x + self.P["seg"][seg]
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
        return H, (ids, keep, seg, caches, cf, H)

    def trunk_bwd(self, dH, cache, g):
        ids, keep, seg, caches, cf, H = cache
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
        if seg is not None:
            np.add.at(g["seg"], seg, dx)

    # ---- pooling ----
    def pooled(self, H, keep):
        if self.pool == "cls":
            return H[:, 0]
        w = keep / keep.sum(1, keepdims=True)
        return (H * w[:, :, None]).sum(1)

    def unpool(self, dp, H, keep):
        dH = np.zeros_like(H)
        if self.pool == "cls":
            dH[:, 0] = dp
        else:
            w = keep / keep.sum(1, keepdims=True)
            dH += w[:, :, None] * dp[:, None, :]
        return dH

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


def xent(logits, y):
    Pr = softmax_last(logits)
    return float(-np.log(Pr[np.arange(len(y)), y] + 1e-300).mean())


def xent_grad(logits, y):
    Pr = softmax_last(logits)
    d = Pr.copy()
    d[np.arange(len(y)), y] -= 1.0
    return d / len(y)


# -----------------------------------------------------------------------
# Three ways to put a PAIR through the same trunk.
#
#   cross-encoder   [CLS] premise [SEP] hypothesis [SEP] -> one pass,
#                   every hypothesis token may attend to every premise
#                   token.  This is BERT's recipe for NLI.
#   bi-encoder      each side encoded ALONE, then compared as vectors
#                   via [u; v; |u-v|; u*v].  No token ever sees the
#                   other sentence.  This is InferSent's and SBERT's.
#   decoder         causal, and the answer is a WORD predicted after
#                   the pair rather than a class from a head.
# -----------------------------------------------------------------------

LABEL_TOKS = np.array([W2I[LABEL_WORD[l]] for l in LABELS])


def cross_logits(m, ids, keep, seg):
    H, cache = m.trunk_fwd(ids, keep, seg)
    return m.pooled(H, keep) @ m.P["Wc"] + m.P["bc"], cache


def cross_grads(m, ids, keep, seg, y):
    logits, cache = cross_logits(m, ids, keep, seg)
    dlog = xent_grad(logits, y)
    g = m.zero_grads()
    H = cache[5]
    g["Wc"] += m.pooled(H, keep).T @ dlog
    g["bc"] += dlog.sum(0)
    m.trunk_bwd(m.unpool(dlog @ m.P["Wc"].T, H, keep), cache, g)
    return g


def bi_logits(m, pi, pk, hi, hk):
    Hp, cp = m.trunk_fwd(pi, pk)
    Hh, ch = m.trunk_fwd(hi, hk)
    u, v = m.pooled(Hp, pk), m.pooled(Hh, hk)
    f = np.concatenate([u, v, np.abs(u - v), u * v], axis=-1)
    return f @ m.P["Wb"] + m.P["bb"], (cp, ch, u, v, f, Hp, Hh, pk, hk)


def bi_grads(m, pi, pk, hi, hk, y):
    logits, c = bi_logits(m, pi, pk, hi, hk)
    cp, ch, u, v, f, Hp, Hh, pk_, hk_ = c
    dlog = xent_grad(logits, y)
    g = m.zero_grads()
    g["Wb"] += f.T @ dlog
    g["bb"] += dlog.sum(0)
    df = dlog @ m.P["Wb"].T
    d = m.d
    dfu, dfv = df[:, :d], df[:, d:2 * d]
    dfd, dfm = df[:, 2 * d:3 * d], df[:, 3 * d:]
    s = np.sign(u - v)
    m.trunk_bwd(m.unpool(dfu + s * dfd + v * dfm, Hp, pk_), cp, g)
    m.trunk_bwd(m.unpool(dfv - s * dfd + u * dfm, Hh, hk_), ch, g)
    return g


def span_pool(H, keep, seg, which):
    """Mean over just one side of the pair, and the weights used."""
    msk = (keep & (seg == which)).astype(float)
    w = msk / np.maximum(msk.sum(1, keepdims=True), 1e-9)
    return (H * w[:, :, None]).sum(1), w


def crossdiff_logits(m, ids, keep, seg):
    """Cross-attention, but read out like a bi-encoder.

    One joint pass as usual, then pool the premise span and the
    hypothesis span SEPARATELY and compare them with the same
    [u; v; |u-v|; u*v] the bi-encoder uses.  Same information as the
    [CLS] version; a different question asked of it.
    """
    H, cache = m.trunk_fwd(ids, keep, seg)
    u, wu = span_pool(H, keep, seg, 0)
    v, wv = span_pool(H, keep, seg, 1)
    f = np.concatenate([u, v, np.abs(u - v), u * v], axis=-1)
    return f @ m.P["Wb"] + m.P["bb"], cache, (u, v, f, wu, wv)


def crossdiff_grads(m, ids, keep, seg, y):
    logits, cache, (u, v, f, wu, wv) = crossdiff_logits(m, ids, keep, seg)
    dlog = xent_grad(logits, y)
    g = m.zero_grads()
    g["Wb"] += f.T @ dlog
    g["bb"] += dlog.sum(0)
    df = dlog @ m.P["Wb"].T
    d = m.d
    dfu, dfv = df[:, :d], df[:, d:2 * d]
    dfd, dfm = df[:, 2 * d:3 * d], df[:, 3 * d:]
    s = np.sign(u - v)
    du = dfu + s * dfd + v * dfm
    dv = dfv - s * dfd + u * dfm
    dH = wu[:, :, None] * du[:, None, :] + wv[:, :, None] * dv[:, None, :]
    m.trunk_bwd(dH, cache, g)
    return g


def dec_logits(m, ids, keep, seg, lab_pos):
    """Read the trunk at the final [SEP] and predict the next WORD."""
    H, cache = m.trunk_fwd(ids, keep, seg)
    hv = H[np.arange(len(ids)), lab_pos]
    return hv @ m.P["Wv"] + m.P["bv"], cache, hv


def dec_grads(m, ids, keep, seg, lab_pos, y_tok):
    logits, cache, hv = dec_logits(m, ids, keep, seg, lab_pos)
    dlog = xent_grad(logits, y_tok)
    g = m.zero_grads()
    g["Wv"] += hv.T @ dlog
    g["bv"] += dlog.sum(0)
    H = cache[5]
    dH = np.zeros_like(H)
    dH[np.arange(len(ids)), lab_pos] = dlog @ m.P["Wv"].T
    m.trunk_bwd(dH, cache, g)
    return g


# -----------------------------------------------------------------------
# Tasks bundle the encoded data with the right forward and backward.
# -----------------------------------------------------------------------

def pretrain_corpus(n, rng):
    """Unlabelled sequences: half single sentences, half pairs.

    Both views are in the pre-training distribution because the
    bi-encoder will later read one sentence at a time while the
    cross-encoder reads two.  Labels are thrown away -- this is
    exactly Part 4's masked language modelling, over pairs.
    """
    ids = np.zeros((n, T_PAIR), dtype=int)
    keep = np.zeros((n, T_PAIR), dtype=bool)
    seg = np.zeros((n, T_PAIR), dtype=int)
    for i in range(n):
        p, h, _ = gen_balanced(rng)
        if rng.uniform() < 0.5:
            toks = [CLS] + [W2I[w] for w in
                            (p if rng.uniform() < 0.5 else h)["words"]]
            toks += [SEP]
            segs = [0] * len(toks)
        else:
            toks = [CLS] + [W2I[w] for w in p["words"]] + [SEP]
            segs = [0] * len(toks)
            toks += [W2I[w] for w in h["words"]] + [SEP]
            segs += [1] * (len(toks) - len(segs))
        ids[i, :len(toks)] = toks
        keep[i, :len(toks)] = True
        seg[i, :len(segs)] = segs
    return ids, keep, seg


def mlm_corrupt(ids, keep, rng, rate=0.15):
    """Part 4's 80/10/10 recipe. [CLS] and [SEP] are never targets."""
    content = keep & (ids != CLS) & (ids != SEP)
    sel = (rng.random(ids.shape) < rate) & content
    for i in np.where(~sel.any(1))[0]:
        ch = np.where(content[i])[0]
        sel[i, ch[rng.integers(0, len(ch))]] = True
    x = ids.copy()
    u = rng.random(ids.shape)
    x[sel & (u < 0.8)] = MASK
    rs = sel & (u >= 0.8) & (u < 0.9)
    x[rs] = rng.integers(N_SPECIAL, V, rs.sum())
    return x, sel


def mlm_grads(m, x, keep, seg, targets, sel):
    H, cache = m.trunk_fwd(x, keep, seg)
    logits = H @ m.P["Wv"] + m.P["bv"]
    Pr = softmax_last(logits)
    n = max(sel.sum(), 1)
    dlog = Pr.copy()
    dlog[sel, targets[sel]] -= 1.0
    dlog *= sel[:, :, None] / n
    g = m.zero_grads()
    g["Wv"] += H.reshape(-1, m.d).T @ dlog.reshape(-1, V)
    g["bv"] += dlog.sum((0, 1))
    m.trunk_bwd(dlog @ m.P["Wv"].T, cache, g)
    return g


def mlm_loss(m, x, keep, seg, targets, sel):
    H, _ = m.trunk_fwd(x, keep, seg)
    Pr = softmax_last(H @ m.P["Wv"] + m.P["bv"])
    return float(-np.log(Pr[sel, targets[sel]] + 1e-300).mean())


def lm_targets(ids, keep):
    """Part 5's shift: position j predicts the token at j+1."""
    tgt = np.zeros_like(ids)
    tgt[:, :-1] = ids[:, 1:]
    sel = np.zeros(ids.shape, dtype=bool)
    sel[:, :-1] = keep[:, 1:]
    return ids, tgt, sel


def pretrain(steps, rng, corpus, d=64, heads=4, layers=2, seed=0,
             lr=3e-3, batch=64, log_every=None, causal=False):
    """One encoder (or decoder), pre-trained once and shared below.

    causal=False is Part 4's masked objective; causal=True is Part 5's
    next-token one.  Both drive the same output head, so the only
    difference is the mask and the (targets, sel) pair -- which is
    exactly the point Part 5 made.
    """
    ids, keep, seg = corpus
    m = Transformer(d=d, heads=heads, layers=layers, causal=causal,
                    seed=seed, lr=lr, t_max=T_PAIR)
    hist = []
    for s in range(steps):
        b = rng.integers(0, len(ids), batch)
        if causal:
            x, tgt, sel = lm_targets(ids[b], keep[b])
        else:
            x, sel = mlm_corrupt(ids[b], keep[b], rng)
            tgt = ids[b]
        m.step(mlm_grads(m, x, keep[b], seg[b], tgt, sel))
        if log_every and (s % log_every == 0 or s == steps - 1):
            hist.append((s + 1, mlm_loss(m, x, keep[b], seg[b], tgt,
                                         sel)))
    return m, hist


class Task:
    """One (model, data) pairing that knows how to step and to score."""

    HEADS = ("Wc", "bc", "Wb", "bb")      # task heads, never transferred

    def __init__(self, kind, train_data, test_data, seed=0, lr=1e-3,
                 d=64, heads=4, layers=2, premise_blind=False,
                 pool=None, init_from=None):
        self.kind = kind
        self.premise_blind = premise_blind
        causal = kind == "decoder"
        pool = pool or ("mean" if kind == "bi" else "cls")
        self.m = Transformer(d=d, heads=heads, layers=layers,
                             causal=causal, seed=seed, lr=lr, pool=pool,
                             t_max=T_PAIR)
        if init_from is not None:
            for k, v in init_from.P.items():
                if k not in self.HEADS:
                    self.m.P[k] = v.copy()
            self.m.reset_adam()
        self.tr = self.encode(train_data)
        self.te = self.encode(test_data)
        self.n = len(train_data)

    def encode(self, data):
        pairs, y = split_xy(data)
        if self.premise_blind:
            pairs = [([], h) for _, h in pairs]
        if self.kind == "bi":
            pi, pk = encode_single([p for p, _ in pairs])
            hi, hk = encode_single([h for _, h in pairs])
            return dict(pi=pi, pk=pk, hi=hi, hk=hk, y=y)
        ids, keep, seg, lp = encode_pair(pairs)
        return dict(ids=ids, keep=keep, seg=seg, lp=lp, y=y,
                    yt=LABEL_TOKS[y])

    def grads(self, b):
        D = self.tr
        if self.kind == "bi":
            return bi_grads(self.m, D["pi"][b], D["pk"][b], D["hi"][b],
                            D["hk"][b], D["y"][b])
        if self.kind == "decoder":
            return dec_grads(self.m, D["ids"][b], D["keep"][b],
                             D["seg"][b], D["lp"][b], D["yt"][b])
        if self.kind == "crossdiff":
            return crossdiff_grads(self.m, D["ids"][b], D["keep"][b],
                                   D["seg"][b], D["y"][b])
        return cross_grads(self.m, D["ids"][b], D["keep"][b],
                           D["seg"][b], D["y"][b])

    def predict(self, D=None):
        D = D or self.te
        if self.kind == "bi":
            lg, _ = bi_logits(self.m, D["pi"], D["pk"], D["hi"], D["hk"])
            return lg.argmax(-1)
        if self.kind == "decoder":
            lg, _, _ = dec_logits(self.m, D["ids"], D["keep"], D["seg"],
                                  D["lp"])
            return lg[:, LABEL_TOKS].argmax(-1)
        if self.kind == "crossdiff":
            lg, _, _ = crossdiff_logits(self.m, D["ids"], D["keep"],
                                        D["seg"])
            return lg.argmax(-1)
        lg, _ = cross_logits(self.m, D["ids"], D["keep"], D["seg"])
        return lg.argmax(-1)

    def accuracy(self, D=None):
        D = D or self.te
        return float((self.predict(D) == D["y"]).mean())

    def fit(self, steps, rng, batch=32, probe_at=()):
        hist = []
        for s in range(steps):
            b = rng.integers(0, self.n, min(batch, self.n))
            self.m.step(self.grads(b))
            if (s + 1) in probe_at:
                hist.append((s + 1, self.accuracy()))
        return hist


# -----------------------------------------------------------------------
# Rule baselines.  The task decomposes into three checks, and naming
# what each partial rule scores tells us exactly which one a model has
# learned and which one it has skipped.
# -----------------------------------------------------------------------

def rule_relation_only(p, h):
    """Find the clause about the same relation; ignore the adjectives.

    This is everything the task asks for EXCEPT the subset check on
    adjectives, so whatever it leaves on the table is the share of the
    task that turns on "the hypothesis added a word".
    """
    for c in p["clauses"]:
        if core(c) == core(h):
            return ("entailment" if c["positive"] == h["positive"]
                    else "contradiction")
    return "neutral"


def rule_hypothesis_only(p, h):
    """Never read the premise. This is the artefact detector."""
    if not h["positive"]:
        return "contradiction"
    if h["adj"] is not None:
        return "neutral"
    return "entailment"


RULES = (("majority class", lambda p, h: "entailment"),
         ("hypothesis only", rule_hypothesis_only),
         ("relation, no adjectives", rule_relation_only),
         ("the full semantics", gold_label))


def rule_scores(data):
    return [(name, float(np.mean([fn(p, h) == l for p, h, l in data])))
            for name, fn in RULES]


def confusion(pred, y):
    C = np.zeros((3, 3), dtype=int)
    for a, b in zip(y, pred):
        C[a, b] += 1
    return C


# -----------------------------------------------------------------------
# Gradient audit.
# -----------------------------------------------------------------------

def alignment_probe(m, data, n=1000):
    """Mask each hypothesis token; is it easier WITH the premise there?

    This measures the one skill a cross-encoder has and a bi-encoder
    cannot have: reading across the [SEP].  The gap is in nats.
    """
    pairs = [(p["words"], h["words"]) for p, h, _ in data[:n]]
    ids, keep, seg, _ = encode_pair(pairs)
    hyp = keep & (seg == 1) & (ids != SEP)
    tot = {True: 0.0, False: 0.0}
    cnt = 0
    for j in range(ids.shape[1]):
        col = hyp[:, j]
        if not col.any():
            continue
        x = ids.copy()
        x[col, j] = MASK
        for blind in (False, True):
            k = keep.copy()
            if blind:                       # hide the premise entirely
                prem = keep & (seg == 0)
                prem[:, 0] = False          # but keep [CLS]
                k = k & ~prem
            H, _ = m.trunk_fwd(x, k, seg)
            lg = H[:, j] @ m.P["Wv"] + m.P["bv"]
            lg = lg - lg.max(-1, keepdims=True)
            lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
            tot[blind] += -lp[np.arange(len(ids)), ids[:, j]][col].sum()
        cnt += col.sum()
    return tot[False] / cnt, tot[True] / cnt


def gradcheck(seed=0, n_per=3):
    rng = np.random.default_rng(seed)
    data = make_data(6, rng)
    worst, checked = 0.0, 0
    for kind in ("cross", "crossdiff", "bi", "decoder"):
        t = Task(kind, data, data, seed=1, d=16, heads=2, layers=2)
        D = t.tr
        b = np.arange(len(data))
        g = t.grads(b)

        def loss():
            if kind == "bi":
                lg, _ = bi_logits(t.m, D["pi"][b], D["pk"][b],
                                  D["hi"][b], D["hk"][b])
                return xent(lg, D["y"][b])
            if kind == "decoder":
                lg, _, _ = dec_logits(t.m, D["ids"][b], D["keep"][b],
                                      D["seg"][b], D["lp"][b])
                return xent(lg, D["yt"][b])
            if kind == "crossdiff":
                lg, _, _ = crossdiff_logits(t.m, D["ids"][b],
                                            D["keep"][b], D["seg"][b])
                return xent(lg, D["y"][b])
            lg, _ = cross_logits(t.m, D["ids"][b], D["keep"][b],
                                 D["seg"][b])
            return xent(lg, D["y"][b])

        for k, p in t.m.P.items():
            flat, gflat = p.reshape(-1), g[k].reshape(-1)
            for ix in rng.choice(len(flat), min(n_per, len(flat)),
                                 replace=False):
                old = flat[ix]
                flat[ix] = old + 1e-6
                lp_ = loss()
                flat[ix] = old - 1e-6
                lm_ = loss()
                flat[ix] = old
                worst = max(worst, abs((lp_ - lm_) / 2e-6 - gflat[ix]))
                checked += 1
    return worst, checked


# -----------------------------------------------------------------------
# Frozen configuration.  Pre-training is the point of this article, so
# the script spends most of its time there; FAST=True cuts every budget
# to a quarter for a quick look, at the cost of the headline effect.
# -----------------------------------------------------------------------

FAST = False
_SCALE = 0.25 if FAST else 1.0


def _b(n):
    return max(int(n * _SCALE), 50)


TRAIN_N, TEST_N = 8000, 2000
PRETRAIN_N = _b(32000)
PRETRAIN_STEPS = _b(4000)
DEC_PRETRAIN_STEPS = _b(4000)     # the same budget as the encoder
FT_STEPS = _b(3000)
BATCH = 32
GRID = tuple(s for s in (250, 500, 1000, 2000, 3000, 4000)
             if s <= FT_STEPS)
SEED_TRAIN, SEED_TEST, SEED_CARELESS = 42, 7, 43
SEED_CORPUS, SEED_PRE, SEED_FT, SEED_MODEL = 5, 3, 11, 0


def build_stage():
    train = make_data(TRAIN_N, np.random.default_rng(SEED_TRAIN))
    test = make_data(TEST_N, np.random.default_rng(SEED_TEST))
    c_train = make_data(TRAIN_N, np.random.default_rng(SEED_CARELESS),
                        careless=True)
    c_test = make_data(TEST_N, np.random.default_rng(SEED_CARELESS + 1),
                       careless=True)
    return train, test, c_train, c_test


def run(kind, train, test, init=None, blind=False, steps=None):
    t = Task(kind, train, test, seed=SEED_MODEL, lr=1e-3,
             init_from=init, premise_blind=blind)
    h = t.fit(steps or FT_STEPS, np.random.default_rng(SEED_FT),
              batch=BATCH, probe_at=GRID)
    return t, h


def recalls(t, D=None):
    C = confusion(t.predict(D), (D or t.te)["y"])
    return [C[i, i] / max(C[i].sum(), 1) for i in range(3)]


def fact_str(f):
    if f[0] == "rel":
        return ("NOT " if not f[1] else "") + "%s(%s, %s)" % (f[3], f[2],
                                                              f[4])
    return "%s(%s)" % (f[1], f[2])


# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

def demo1(train, test):
    banner("1. A SEMANTICS, SO THE LABEL IS COMPUTED")
    print("Every sentence asserts a polarised relation plus whatever")
    print("adjectives it carries, and entailment is set inclusion over")
    print("those facts. No annotator is involved anywhere.")
    print()
    p, h, l = train[0]
    print("   premise:    " + " ".join(p["words"]))
    print("   asserts:    " + " | ".join(sorted(fact_str(f)
                                                for f in p["facts"])))
    print("   hypothesis: " + " ".join(h["words"]))
    print("   asserts:    " + " | ".join(sorted(fact_str(f)
                                                for f in h["facts"])))
    print("   label:      " + l)
    print()
    print("One premise, three hypotheses, three labels:")
    print()
    base = p["clauses"][0]
    rng = np.random.default_rng(99)
    for want, build in (("entailment", lambda: variant(rng, base,
                                                       adj=None)),
                        ("contradiction",
                         lambda: variant(rng, base,
                                         positive=not base["positive"])),
                        ("neutral", lambda: variant(rng, base,
                                                    change="obj"))):
        cand = build()
        assert gold_label(p, cand) == want, (want, cand["words"])
        print("   %-14s %s" % (want, " ".join(cand["words"])))
    print()
    print("Because the label is a function, the task decomposes and")
    print("each part can be priced exactly:")
    print()
    scores = rule_scores(test)
    for name, acc in scores:
        print("   %-26s %.4f" % (name, acc))
    print()
    print("Relation matching alone gets almost everything. The few")
    print("points it leaves are the adjective subset check -- whether")
    print("the hypothesis added a word the premise never claimed.")
    return scores


def demo2(train, test, pre, align):
    banner("2. TWO WAYS TO READ A PAIR")
    print("A CROSS-encoder reads [CLS] premise [SEP] hypothesis [SEP] in")
    print("one pass, so any hypothesis token may attend to any premise")
    print("token. A BI-encoder encodes each side alone and compares the")
    print("vectors with [u; v; |u-v|; u*v]. Same trunk, same budget;")
    print("the only difference is whether the two sides ever meet.")
    print()
    rows = []
    for kind, init, lab in (("cross", None, "cross-encoder, scratch"),
                            ("cross", pre, "cross-encoder, pre-trained"),
                            ("bi", None, "bi-encoder, scratch"),
                            ("bi", pre, "bi-encoder, pre-trained")):
        t, h = run(kind, train, test, init=init)
        r = recalls(t)
        rows.append((lab, t.accuracy(), r, h))
        print("   %-28s %.4f   recall ent %.3f / neu %.3f / con %.3f"
              % (lab, t.accuracy(), *r))
    print()
    print("The cross-encoder from scratch is the interesting failure.")
    print("It scores %.4f, and the two-class ceiling -- entailment and"
          % rows[0][1])
    print("contradiction right, neutral abandoned -- is 0.667. Its")
    print("neutral recall is %.3f. It found the cheap half of the task"
          % rows[0][2][1])
    print("and stopped looking.")
    print()
    print("Pre-training is worth %+.1f points to it and %+.1f to the"
          % (100 * (rows[1][1] - rows[0][1]),
             100 * (rows[3][1] - rows[2][1])))
    print("bi-encoder. That asymmetry is the result. The bi-encoder's")
    print("subtraction is wired in, so it needs no help; the")
    print("cross-encoder's comparison has to be LEARNED, and 8,000")
    print("labels are nowhere near enough to learn it.")
    print()
    w, wo = align
    print("What pre-training installed can be measured directly. Blank")
    print("a hypothesis token and ask the encoder to recover it:")
    print()
    print("   with the premise visible:  %.4f nats" % w)
    print("   with the premise hidden:   %.4f nats" % wo)
    print("   so the premise is worth:   %.4f nats" % (wo - w))
    print()
    print("That gap IS cross-attention doing its job, and no bi-encoder")
    print("can have it -- the two sides never share a forward pass.")
    print()
    print("Honesty about the ordering: the bi-encoder still wins here,")
    print("and in published NLI it does not. Both facts point the same")
    print("way. Joint attention is a capability you pay for, the bill")
    print("is paid in pre-training, and this budget buys most of the")
    print("gap but not the crossover.")
    return rows


def demo3(train, test, c_train, c_test):
    banner("3. THE ARTEFACT: WINNING WITHOUT READING THE PREMISE")
    print("SNLI was built by showing annotators a caption and asking")
    print("for three sentences: one true, one false, one that might be")
    print("true. People took the shortest path -- they negated for")
    print("false and deleted a word for true.")
    print()
    print("The CARELESS generator does exactly that. The BALANCED one")
    print("puts a negation and an adjective in half the hypotheses of")
    print("every class, so neither can give the label away:")
    print()
    print("   %-9s %-16s %9s %9s %9s"
          % ("dataset", "label", "has 'not'", "has adj", "mean len"))
    for nm, d in (("careless", c_train), ("balanced", train)):
        for l in LABELS:
            rows = [h for _, h, lab in d if lab == l]
            print("   %-9s %-16s %9.3f %9.3f %9.3f"
                  % (nm, l, np.mean([not x["positive"] for x in rows]),
                     np.mean([x["adj"] is not None for x in rows]),
                     np.mean([len(x["words"]) for x in rows])))
    print()
    print("Now train a model shown ONLY the hypothesis -- the premise")
    print("is replaced by nothing at all:")
    print()
    t_car, _ = run("cross", c_train, c_test, blind=True)
    car_on_car = t_car.accuracy()
    car_on_bal = t_car.accuracy(t_car.encode(test))
    t_bal, _ = run("cross", train, test, blind=True)
    bal_on_bal = t_bal.accuracy()
    print("   hypothesis-only, careless -> careless: %.4f" % car_on_car)
    print("   the same model,  careless -> balanced: %.4f" % car_on_bal)
    print("   hypothesis-only, balanced -> balanced: %.4f" % bal_on_bal)
    print()
    print("A model with no access to the premise scores %.4f on a"
          % car_on_car)
    print("three-way entailment task. It has learned nothing about")
    print("entailment; it has learned how the data was made. Move it")
    print("to data without the tell and it falls to %.4f, against a"
          % car_on_bal)
    print("chance rate of 0.333.")
    print()
    print("This is why the gold labels here are computed. A generator")
    print("can be audited; a pool of annotators cannot.")
    return car_on_car, car_on_bal, bal_on_bal


def demo4(train, test, dec_pre, cross_acc):
    banner("4. THE LABEL AS A TOKEN")
    print("Part 5's decoder answered by predicting the next word.")
    print("Nothing stops it answering this question the same way: read")
    print("the pair, then predict yes, no or maybe. No classification")
    print("head at all -- the answer comes out of the same vocabulary")
    print("the text did.")
    print()
    ids, _, _, _ = encode_pair([(train[0][0]["words"],
                                 train[0][1]["words"])],
                               with_label=[train[0][2]])
    print("   " + " ".join(VOCAB[i] for i in ids[0] if i != PAD))
    print()
    t, _ = run("decoder", train, test, init=dec_pre)
    acc = t.accuracy()
    r = recalls(t)
    print("   decoder, label as a token:       %.4f   "
          "recall %.3f/%.3f/%.3f" % (acc, *r))
    print("   cross-encoder, label as a class: %.4f" % cross_acc)
    print()
    gap = 100 * (cross_acc - acc)
    print("Same trunk, same pre-training budget, same labels; the only")
    print("differences are the causal mask and where the answer comes")
    print("from.")
    print()
    if abs(gap) < 5:
        print("The decoder lands within %.1f points of the encoder. The"
              % abs(gap))
        print("answer's FORMAT is not what matters -- a label can be a")
        print("class or a word, and the model does not care. That is")
        print("the whole basis of prompting: put the question first and")
        print("the answer last, and a left-to-right model will do any")
        print("task you can phrase as text.")
    else:
        print("The decoder is %.1f points BEHIND, and the gap is almost"
              % gap)
        print("all in neutral (recall %.3f). The position predicting the"
              % r[1])
        print("label does see the whole pair, so the answer format is")
        print("not the problem. What it lacks is the premise's tokens")
        print("having been encoded in the hypothesis's light: in the")
        print("encoder every premise token is revised once the")
        print("hypothesis arrives, and causality forbids exactly that.")
        print("Prompting works, but on this task it starts behind.")
    return acc


def main():
    banner("NATURAL LANGUAGE INFERENCE --- pairs, labels and artefacts")
    worst, checked = gradcheck()
    print("gradient audit: worst error %.2e over %d sampled gradients"
          % (worst, checked))
    assert worst < 1e-6, worst
    train, test, c_train, c_test = build_stage()
    allp = train + test + c_train + c_test
    bad = sum(gold_label(p, h) != l for p, h, l in allp)
    print("label audit:    %d disagreements between the generators and"
          % bad)
    print("                the semantics over %d pairs" % len(allp))
    assert bad == 0
    print()
    print("%d training pairs, %d held out, vocabulary %d, sequences"
          % (len(train), len(test), V))
    print("padded to %d. FAST = %s." % (T_PAIR, FAST))

    demo1(train, test)

    banner("PRE-TRAINING")
    print("One encoder and one decoder, pre-trained on %d unlabelled"
          % PRETRAIN_N)
    print("sequences -- half single sentences, half pairs, the labels")
    print("thrown away. This is the slow part, and it is the point of")
    print("the article.")
    corpus = pretrain_corpus(PRETRAIN_N,
                             np.random.default_rng(SEED_CORPUS))
    pre, h_pre = pretrain(PRETRAIN_STEPS, np.random.default_rng(SEED_PRE),
                          corpus, log_every=max(PRETRAIN_STEPS // 6, 1))
    print()
    print("   masked:     " + "  ".join("%d:%.3f" % x for x in h_pre))
    dec_pre, h_dec = pretrain(DEC_PRETRAIN_STEPS,
                              np.random.default_rng(SEED_PRE), corpus,
                              log_every=max(DEC_PRETRAIN_STEPS // 6, 1),
                              causal=True)
    print("   next-token: " + "  ".join("%d:%.3f" % x for x in h_dec))
    align = alignment_probe(pre, test)

    rows = demo2(train, test, pre, align)
    demo3(train, test, c_train, c_test)
    demo4(train, test, dec_pre, rows[1][1])

    banner("DONE")


if __name__ == "__main__":
    main()
