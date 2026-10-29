"""
bert_encoder.py --- companion code for "BERT-style Encoders"
(Natural Language Processing, Part 4).

Part 3 built an encoder and a decoder. This article throws the decoder
away. BERT (Devlin et al., 2019) keeps the encoder, drops the causal
mask so every position may look both ways, and manufactures its
supervision by HIDING tokens and predicting them -- masked language
modelling, Section 12's masked autoencoder one modality over.

The micro-language is built to make bidirectionality measurable.
Number and gender agreement is carried by the verb form and the
possessive, both of which sit to the RIGHT of the subject:

    the king   rules  his    crown      (singular, male)
    the kings  rule   their  crown      (plural)

So mask the subject and the evidence for its number and gender is
only available to a model that can read rightwards. Mask the
possessive and the evidence is to the left, where a causal model can
reach it too. One corpus, two slots, opposite verdicts.

Demonstrates:
  1. The stage and the corruption recipe: 15% of positions chosen,
     80% replaced by [MASK], 10% by a random word, 10% left alone.
  2. Bidirectional vs causal: identical encoders, identical budget;
     the receipt splits by where the evidence lies.
  3. Pre-train then fine-tune: an agreement-acceptability task
     (is this sentence grammatical?) at 20/50/100/400 labels, with
     a from-scratch encoder and a frozen probe as controls.
  4. Why 80/10/10 and not 100% [MASK]: the pre-train/fine-tune
     mismatch, measured downstream.

Everything is plain NumPy: multi-head bidirectional self-attention,
LayerNorm, learned positions, the MLM head and the [CLS] head, all
hand-differentiated and audited against central differences at
startup. Seeded and deterministic.
Dependencies: numpy.  Runs in about three minutes.
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
# Vocabulary and encoding.  [CLS] is prepended to every sequence.
# -----------------------------------------------------------------------

PAD, CLS, MASK = 0, 1, 2
N_SPECIAL = 3


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
    vocab = ["[PAD]", "[CLS]", "[MASK]"] + sorted(words)
    return vocab, {w: i for i, w in enumerate(vocab)}


VOCAB, W2I = build_vocab()
V = len(VOCAB)
T_MAX = 7          # [CLS] + at most six words
STEPS = 1500       # pre-training steps per encoder


def encode(batch_words):
    """Pad a list of word-lists to (N, T_MAX) ids plus a keep mask."""
    N = len(batch_words)
    ids = np.zeros((N, T_MAX), dtype=int)
    keep = np.zeros((N, T_MAX), dtype=bool)
    ids[:, 0] = CLS
    keep[:, 0] = True
    for i, ws in enumerate(batch_words):
        for j, w in enumerate(ws):
            ids[i, j + 1] = W2I[w]
            keep[i, j + 1] = True
    return ids, keep


# -----------------------------------------------------------------------
# The encoder.  Pre-LN blocks: x = x + attn(LN(x)); x = x + mlp(LN(x)).
# Bidirectional by default; causal=True puts the triangular mask back.
# -----------------------------------------------------------------------

class Encoder:
    def __init__(self, d=64, heads=4, layers=2, d_ff=None, causal=False,
                 seed=0, lr=3e-3, n_cls=2, pool="cls"):
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
        P["Wm"] = rnd(d, V)                 # masked-LM head
        P["bm"] = np.zeros(V)
        P["Wc"] = rnd(d, n_cls)             # [CLS] classification head
        P["bc"] = np.zeros(n_cls)
        self.P = P
        self.reset_adam()

    def reset_adam(self):
        self.M = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.Vm = {k: np.zeros_like(v) for k, v in self.P.items()}
        self.t = 0

    def clone(self):
        """Independent copy with fresh optimiser state (for fine-tuning)."""
        new = Encoder.__new__(Encoder)
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

    # ---- masked language modelling ----
    def mlm_logits(self, ids, keep):
        H, cache = self.trunk_fwd(ids, keep)
        return H @ self.P["Wm"] + self.P["bm"], cache

    def mlm_loss(self, ids, keep, targets, sel):
        logits, _ = self.mlm_logits(ids, keep)
        Pr = softmax_last(logits)
        n = max(sel.sum(), 1)
        lp = -np.log(Pr[sel, targets[sel]] + 1e-300)
        return float(lp.sum() / n)

    def mlm_grads(self, ids, keep, targets, sel):
        logits, cache = self.mlm_logits(ids, keep)
        Pr = softmax_last(logits)
        n = max(sel.sum(), 1)
        dlog = Pr.copy()
        dlog[sel, targets[sel]] -= 1.0
        dlog *= sel[:, :, None] / n
        g = self.zero_grads()
        H = cache[4]
        g["Wm"] += H.reshape(-1, self.d).T @ dlog.reshape(-1, V)
        g["bm"] += dlog.sum((0, 1))
        self.trunk_bwd(dlog @ self.P["Wm"].T, cache, g)
        return g

    # ---- [CLS] classification ----
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
# The corruption recipe.
# -----------------------------------------------------------------------

def corrupt(ids, keep, rng, rate=0.15, recipe="80/10/10"):
    """Choose positions to predict, then corrupt the input at them.

    Returns (corrupted ids, selection mask). [CLS] and padding are
    never selected; every row gets at least one selected position.
    """
    content = keep.copy()
    content[:, 0] = False
    sel = (rng.random(ids.shape) < rate) & content
    for i in np.where(~sel.any(1))[0]:
        choices = np.where(content[i])[0]
        sel[i, choices[rng.integers(0, len(choices))]] = True

    out = ids.copy()
    if recipe == "mask":
        out[sel] = MASK
    else:
        u = rng.random(ids.shape)
        out[sel & (u < 0.8)] = MASK
        rnd_sel = sel & (u >= 0.8) & (u < 0.9)
        out[rnd_sel] = rng.integers(N_SPECIAL, V, rnd_sel.sum())
        # the remaining 10% keep the original token
    return out, sel


def pretrain(model, ids, keep, steps, rng, batch=64, recipe="80/10/10",
             log_every=None):
    """Masked language modelling. Fresh masks every batch."""
    hist = []
    for s in range(steps):
        b = rng.integers(0, len(ids), batch)
        x, sel = corrupt(ids[b], keep[b], rng, recipe=recipe)
        g = model.mlm_grads(x, keep[b], ids[b], sel)
        model.step(g)
        if log_every and (s % log_every == 0 or s == steps - 1):
            hist.append((s, model.mlm_loss(x, keep[b], ids[b], sel)))
    return hist


# -----------------------------------------------------------------------
# Evaluation: mask one slot at a time and score the prediction.
# -----------------------------------------------------------------------

def slot_eval(model, sents, slot, topk=1):
    """Mask `slot` in every sentence; return the predicted words."""
    ids, keep = encode([s["words"] for s in sents])
    pos = np.array([s[slot] + 1 for s in sents])      # +1 for [CLS]
    rows = np.arange(len(sents))
    gold = ids[rows, pos].copy()
    ids[rows, pos] = MASK
    logits, _ = model.mlm_logits(ids, keep)
    pick = logits[rows, pos]
    pred = pick.argmax(-1)
    return [VOCAB[i] for i in pred], [VOCAB[i] for i in gold]


def score_subject(model, sents):
    pred, gold = slot_eval(model, sents, "subj")
    n = len(sents)
    num = sum((p in PLURAL_NOUNS) == s["plural"]
              for p, s in zip(pred, sents)) / n
    top = sum(TOPIC_OF.get(p) == s["topic"]
              for p, s in zip(pred, sents)) / n
    exact = sum(p == gv for p, gv in zip(pred, gold)) / n
    gendered = [(p, s) for p, s in zip(pred, sents)
                if s["gender"] in ("m", "f")]
    gen_ok = sum(
        (p in GENDER_M) if s["gender"] == "m" else (p in GENDER_F)
        for p, s in gendered) / max(len(gendered), 1)
    return num, top, exact, gen_ok, pred


def score_verb(model, sents):
    pred, gold = slot_eval(model, sents, "verb")
    n = len(sents)
    num = sum((p in VERB_PL) == s["plural"]
              for p, s in zip(pred, sents)) / n
    top = sum(TOPIC_OF.get(p) == s["topic"]
              for p, s in zip(pred, sents)) / n
    exact = sum(p == gv for p, gv in zip(pred, gold)) / n
    return num, top, exact


def score_poss(model, sents):
    pred, gold = slot_eval(model, sents, "poss")
    n = len(sents)
    exact = sum(p == gv for p, gv in zip(pred, gold)) / n
    num = sum((p == "their") == s["plural"]
              for p, s in zip(pred, sents)) / n
    return exact, num


def pseudo_log_likelihood(model, batch_words):
    """Sum over positions of log P(true token | the rest of the sentence).

    Mask one position at a time and read the probability the encoder
    assigns to the word that was actually there (Salazar et al., 2020).
    No new parameters, no labels: a grammaticality score straight out
    of the masked-LM head.
    """
    ids, keep = encode(batch_words)
    rows = np.arange(len(batch_words))
    out = np.zeros(len(batch_words))
    for j in range(1, T_MAX):
        col = keep[:, j]
        if not col.any():
            continue
        x = ids.copy()
        x[col, j] = MASK
        logits, _ = model.mlm_logits(x, keep)
        lg = logits[:, j]
        lg = lg - lg.max(-1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        out[col] += lp[rows, ids[:, j]][col]
    return out


def pll_threshold(model, labelled):
    """Midpoint between the mean score of each class: one number."""
    s = pseudo_log_likelihood(model, [w for w, _ in labelled])
    y = np.array([lab for _, lab in labelled])
    return 0.5 * (s[y == 1].mean() + s[y == 0].mean())


def copy_test(model, sents, rng):
    """Show the encoder a real-but-wrong word, with no [MASK] anywhere.

    The subject is replaced by a noun from a different topic. Does the
    encoder echo what it was shown, or correct it?
    """
    ids, keep = encode([s["words"] for s in sents])
    rows = np.arange(len(sents))
    pos = np.array([s["subj"] + 1 for s in sents])
    gold = ids[rows, pos].copy()
    wrong = []
    for s in sents:
        others = [n for name, t in TOPICS.items() if name != s["topic"]
                  for n in t["nouns"]]
        w = others[rng.integers(0, len(others))]
        if s["plural"] and plural(w) in W2I:
            w = plural(w)
        wrong.append(W2I[w])
    wrong = np.array(wrong)
    ids[rows, pos] = wrong
    logits, _ = model.mlm_logits(ids, keep)
    pred = logits[rows, pos].argmax(-1)
    echo = float((pred == wrong).mean())
    topic = float(np.mean([TOPIC_OF.get(VOCAB[p]) == s["topic"]
                           for p, s in zip(pred, sents)]))
    return echo, topic, float((pred == gold).mean())


def attention_from_mask(model, sent, slot="subj"):
    """Mean attention (over heads) out of the masked position, per layer."""
    ids, keep = encode([sent["words"]])
    pos = sent[slot] + 1
    ids[0, pos] = MASK
    _, cache = model.mlm_logits(ids, keep)
    caches = cache[2]
    out = []
    for li in range(model.L):
        W = caches[li][1][4]          # (N, h, T, T)
        out.append(W[0, :, pos, :].mean(0))
    return out, ["[CLS]"] + sent["words"], pos


# -----------------------------------------------------------------------
# Fine-tuning on the acceptability task.
# -----------------------------------------------------------------------

def finetune(model, train, test, steps, rng, batch=16, only=None):
    xs, ks = encode([w for w, _ in train])
    ys = np.array([y for _, y in train])
    xt, kt = encode([w for w, _ in test])
    yt = np.array([y for _, y in test])
    for _ in range(steps):
        b = rng.integers(0, len(xs), min(batch, len(xs)))
        g = model.cls_grads(xs[b], ks[b], ys[b])
        model.step(g, only=only)
    return model.cls_accuracy(xt, kt, yt)


# -----------------------------------------------------------------------
# Gradient audit.
# -----------------------------------------------------------------------

def gradcheck(seed=0, n_per=4):
    rng = np.random.default_rng(seed)
    sents = gen_corpus(5, rng)
    ids, keep = encode([s["words"] for s in sents])
    labels = rng.integers(0, 2, len(sents))
    net = Encoder(d=16, heads=2, layers=2, seed=1)
    x, sel = corrupt(ids, keep, rng)

    worst = 0.0
    checked = 0
    for which in ("mlm", "cls", "cls_mean"):
        net.pool = "mean" if which == "cls_mean" else "cls"
        if which == "mlm":
            g = net.mlm_grads(x, keep, ids, sel)

            def loss():
                return net.mlm_loss(x, keep, ids, sel)
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


# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

def demo1(corpus, ids, keep):
    banner("DEMO 1 --- The stage and the corruption recipe")
    print("  Part 1's micro-language, one clause per sentence, with a")
    print("  [CLS] token in front and padding to seven positions:")
    print()
    for s in corpus[:4]:
        print(f"    [CLS] {' '.join(s['words'])}")
    print()
    print(f"  Vocabulary: {V} tokens "
          f"([PAD], [CLS], [MASK] + {V - 3} words)")
    print(f"  Pre-training corpus: {len(corpus):,} sentences, "
          f"{int(keep.sum()) - len(corpus):,} word tokens")
    print()
    print("  The pretext: choose 15% of word positions, then corrupt")
    print("  the input at them -- 80% [MASK], 10% a random word, 10%")
    print("  left untouched. Predict the original token at every")
    print("  chosen position. One batch of four, seed 0:")
    print()
    rng = np.random.default_rng(0)
    x, sel = corrupt(ids[:4], keep[:4], rng)
    for i in range(4):
        shown = []
        for j in range(T_MAX):
            if not keep[i, j]:
                continue
            w = VOCAB[x[i, j]]
            shown.append(f"<{w}>" if sel[i, j] else w)
        print(f"    {' '.join(shown)}")
    print()
    print("  Angle brackets mark the chosen positions. Note the third")
    print("  kind of corruption: a position may be left as the real")
    print("  word and still be scored -- the model cannot assume that")
    print("  what it reads is what it must keep.")


def demo2(bi, ca, test):
    banner("DEMO 2 --- Bidirectional vs causal: where the evidence lies")
    print("  Two encoders, same size (d=64, 2 layers, 4 heads,")
    print(f"  {bi.n_params():,} parameters), same corpus, same budget.")
    print("  One reads both ways; the other wears Part 3's causal mask.")
    print()
    print("  Mask one slot at a time in 1,000 held-out sentences and")
    print("  score the prediction. The micro-language puts the verb")
    print("  form and the possessive to the RIGHT of the subject, so")
    print("  the subject's number and gender are only legible to a")
    print("  model that can read rightwards:")
    print()
    rows = []
    bs = score_subject(bi, test)
    cs = score_subject(ca, test)
    bv = score_verb(bi, test)
    cv = score_verb(ca, test)
    bp = score_poss(bi, test)
    cp = score_poss(ca, test)
    base_sg = 1 - sum(s["plural"] for s in test) / len(test)
    rows.append(("subject: number", "right", cs[0], bs[0]))
    rows.append(("subject: gender", "right", cs[3], bs[3]))
    rows.append(("subject: topic", "both", cs[1], bs[1]))
    rows.append(("subject: exact word", "both", cs[2], bs[2]))
    rows.append(("verb: number", "left", cv[0], bv[0]))
    rows.append(("verb: topic", "left", cv[1], bv[1]))
    rows.append(("possessive: exact", "left", cp[0], bp[0]))
    print(f"    {'masked slot':20s}  {'evidence':9s}  {'causal':>7s}  "
          f"{'bidir':>7s}")
    print(f"    {'-' * 20}  {'-' * 9}  {'-' * 7}  {'-' * 7}")
    for name, ev, c, b in rows:
        print(f"    {name:20s}  {ev:9s}  {c:6.1%}  {b:6.1%}")
    print()
    print(f"  Base rate for 'guess singular': {base_sg:.1%}. The causal")
    print("  model's number accuracy on the masked subject is that")
    print("  base rate and nothing more -- it is guessing the majority")
    print("  class, because the evidence is behind a wall. On the")
    print("  slots whose evidence lies to the left, the two models are")
    print("  level. Bidirectionality buys exactly one thing: the")
    print("  predictions that need to look forward.")
    print()
    print("  Examples, subject masked:")
    print()
    for s, pb, pc in list(zip(test, bs[4], cs[4]))[:5]:
        sent = list(s["words"])
        sent[s["subj"]] = "[MASK]"
        print(f"    {' '.join(sent)}")
        print(f"      gold {s['words'][s['subj']]:>10s}   "
              f"bidir {pb:>10s}   causal {pc:>10s}")
    print()
    print("  Neither model can be sure WHICH noun was hidden: twelve")
    print("  are possible per topic, and only Part 1's role signatures")
    print("  narrow it down (the object is the subject's signature")
    print("  partner half the time). The bidirectional model gets the")
    print("  grammar right regardless -- right number, right gender,")
    print("  right topic, often the wrong noun.")
    print()
    print("  And the receipt. Mean attention out of the masked")
    print("  position, averaged over heads:")
    print()
    sent = next(s for s in test if s["plural"] and len(s["words"]) == 6)
    for model, tag in ((bi, "bidir "), (ca, "causal")):
        att, words, pos = attention_from_mask(model, sent, "subj")
        words = list(words)
        words[pos] = "[MASK]"
        if tag.strip() == "bidir":
            print("    " + " " * 9 + "".join(f"{w:>9s}" for w in words))
        for li, a in enumerate(att):
            print(f"    {tag} L{li}" +
                  "".join(f"{v:>9.2f}" for v in a))
    print()
    print("  Layer 0 of the bidirectional model spreads its attention")
    print("  over the content words on both sides of the gap; layer 1")
    print("  concentrates on the determiner and on 'their' -- the")
    print("  token that carries the number. The causal model's row is")
    print("  zero everywhere to the right of the mask, by")
    print("  construction. It is not that it looks in the wrong")
    print("  place; there is no place to look.")
    return rows, base_sg


def demo3(bi, ca, label_counts, test_accept):
    banner("DEMO 3 --- Pre-train, then fine-tune")
    print("  The downstream task: is this sentence grammatical? Half")
    print("  the examples are generated normally; half have their")
    print("  agreement broken -- the verb's number swapped, or a")
    print("  possessive that disagrees with its subject:")
    print()
    for w, y in test_accept[:6]:
        print(f"    {'ok  ' if y else 'bad '} {' '.join(w)}")
    print()
    print("  Every word is in vocabulary and every sentence has the")
    print("  right shape; only the agreement is wrong. Chance is 50%.")
    print()
    print("  Three conditions, each given the same labels and the same")
    print("  600 held-out test sentences:")
    print()
    print("    scratch   -- same architecture, random init, trained on")
    print("                 the labels alone")
    print("    probe     -- pre-trained encoder FROZEN, only the [CLS]")
    print("                 classifier head is trained")
    print("    fine-tune -- pre-trained encoder, everything trained")
    print()
    results = {}
    print(f"    {'labels':>7s}  {'scratch':>8s}  {'probe':>8s}  "
          f"{'fine-tune':>10s}")
    print(f"    {'-' * 7}  {'-' * 8}  {'-' * 8}  {'-' * 10}")
    for n in label_counts:
        train = make_acceptability(n, np.random.default_rng(1000 + n))

        sc = Encoder(d=64, heads=4, layers=2, seed=5, lr=1e-3)
        a_sc = finetune(sc, train, test_accept, 400,
                        np.random.default_rng(11))

        pr = bi.clone()
        pr.lr = 1e-2
        a_pr = finetune(pr, train, test_accept, 400,
                        np.random.default_rng(11), only={"Wc", "bc"})

        ft = bi.clone()
        ft.lr = 1e-3
        a_ft = finetune(ft, train, test_accept, 400,
                        np.random.default_rng(11))

        results[n] = (a_sc, a_pr, a_ft)
        print(f"    {n:7d}  {a_sc:7.1%}  {a_pr:7.1%}  {a_ft:9.1%}")
    gap = 100 * (results[100][2] - results[100][0])
    print()
    print(f"  At 100 labels, fine-tuning is {gap:.0f} points ahead of")
    print("  the same architecture trained from scratch, and roughly")
    print("  where from-scratch gets with four times the labels. The")
    print("  pre-trained encoder has never seen a label and never seen")
    print("  an ungrammatical sentence; what it has is representations")
    print("  in which agreement is already encoded, because predicting")
    print("  a masked possessive REQUIRES knowing the subject's")
    print("  number.")
    print()
    print("  The frozen probe is the control, and it is the")
    print("  interesting failure. The information is in there -- but")
    print("  it is not linearly readable from the [CLS] vector,")
    print("  because nothing in masked language modelling ever gave")
    print("  [CLS] a job. BERT's second pre-training task,")
    print("  next-sentence prediction, existed largely to give that")
    print("  position something to do; RoBERTa later dropped NSP and")
    print("  relied on fine-tuning instead, which is the column that")
    print("  works here too.")
    print()
    mp = bi.clone()
    mp.lr, mp.pool = 1e-3, "mean"
    a_mp = finetune(mp, make_acceptability(100, np.random.default_rng(1100)),
                    test_accept, 400, np.random.default_rng(11))
    print("  Mean-pooling the token vectors instead of reading [CLS]")
    print(f"  does measurably better at this scale: {a_mp:.1%} at 100")
    print(f"  labels against {results[100][2]:.1%}. [CLS] is a convention,")
    print("  not a mechanism.")
    print()
    print("  And the twist: fine-tuning may not be needed at all.")
    print("  Score a sentence by masking each position in turn and")
    print("  summing log P(the word that was really there) -- the")
    print("  pseudo-log-likelihood (Salazar et al., 2020). No new")
    print("  parameters; the masked-LM head already has an opinion:")
    print()
    cal = make_acceptability(20, np.random.default_rng(1020))
    labels = np.array([y for _, y in test_accept])
    words = [w for w, _ in test_accept]
    zs = {}
    print(f"    {'':15s}{'acceptable':>11s}{'corrupt':>10s}"
          f"{'accuracy':>11s}{'best':>8s}")
    print(f"    {'':15s}{'-' * 10:>11s}{'-' * 7:>10s}"
          f"{'-' * 8:>11s}{'-' * 6:>8s}")
    for tag, model in (("bidirectional", bi), ("causal", ca)):
        s = pseudo_log_likelihood(model, words)
        thr = pll_threshold(model, cal)
        acc = float(((s >= thr) == (labels == 1)).mean())
        best = max(float(((s >= t) == (labels == 1)).mean())
                   for t in np.sort(s))
        zs[tag] = (acc, best, s[labels == 1].mean(), s[labels == 0].mean())
        print(f"    {tag:15s}{zs[tag][2]:11.2f}{zs[tag][3]:10.2f}"
              f"{acc:10.1%}{best:8.1%}")
    print()
    print("  Twenty labels spent on a single threshold beat twenty")
    print("  labels spent on fine-tuning a classifier head, by")
    print(f"  {zs['bidirectional'][0]:.0%} to "
          f"{results[20][2]:.0%}. The knowledge was already there; the")
    print("  head was the bottleneck.")
    print()
    print("  Honesty about the comparison: the causal model scores")
    print("  these sentences BETTER, because every violation here is")
    print("  local and visible from the left -- a plural verb after a")
    print("  singular subject is improbable to a left-to-right reader")
    print("  too. Bidirectionality is not a universal advantage; it is")
    print("  an advantage exactly where the evidence lies ahead, which")
    print("  is what Demo 2 measured and this task does not test.")
    return results, zs


def demo4(bi, mk, test, test_accept):
    banner("DEMO 4 --- Why 80/10/10 and not 100% [MASK]")
    print("  BERT corrupts 15% of positions but turns only 80% of")
    print("  them into [MASK]: 10% become a random word and 10% are")
    print("  left alone. The paper's reason is a mismatch -- [MASK]")
    print("  is everywhere during pre-training and never seen again")
    print("  afterwards, so the encoder is tuned for an input that")
    print("  fine-tuning will never show it.")
    print()
    print("  The usual story told about the 10% random replacement is")
    print("  that without it the model would simply trust whatever")
    print("  word it is shown. That story is testable: replace a")
    print("  subject with a noun from a different topic, with NO")
    print("  [MASK] anywhere in the sentence, and read the masked-LM")
    print("  head at that position.")
    print()
    e_b, t_b, x_b = copy_test(bi, test, np.random.default_rng(5))
    e_m, t_m, x_m = copy_test(mk, test, np.random.default_rng(5))
    print(f"    {'':26s}  {'80/10/10':>10s}  {'100% mask':>10s}")
    print(f"    {'-' * 26}  {'-' * 10}  {'-' * 10}")
    print(f"    {'echoes the wrong word':26s}  {e_b:9.1%}  {e_m:9.1%}")
    print(f"    {'restores the right topic':26s}  {t_b:9.1%}  {t_m:9.1%}")
    print(f"    {'restores the exact word':26s}  {x_b:9.1%}  {x_m:9.1%}")
    print()
    print("  The folklore is wrong, at least here: neither model")
    print("  echoes the word it was shown. Trained only ever to")
    print("  replace [MASK], the 100%-mask encoder still overrules a")
    print("  real token that does not fit. What it does lose is")
    print(f"  precision -- it recovers the right topic {t_b - t_m:.0%} less")
    print("  often.")
    print()
    accs = {}
    for n in (50, 100):
        train = make_acceptability(n, np.random.default_rng(1000 + n))
        for tag, model in (("80/10/10", bi), ("100% mask", mk)):
            ft = model.clone()
            ft.lr = 1e-3
            accs[(tag, n)] = finetune(ft, train, test_accept, 400,
                                      np.random.default_rng(11))
    print("  Downstream, fine-tuned on the acceptability task:")
    print()
    print(f"    {'labels':>26s}  {'80/10/10':>10s}  {'100% mask':>10s}")
    print(f"    {'-' * 26}  {'-' * 10}  {'-' * 10}")
    for n in (50, 100):
        print(f"    {n:26d}  {accs[('80/10/10', n)]:9.1%}  "
              f"{accs[('100% mask', n)]:9.1%}")
    print()
    print("  The recipe earns its keep, but modestly, and this is a")
    print("  toy: a few points on 600 test sentences is not a result")
    print("  to bet on. The honest summary is that the mechanism")
    print("  everyone quotes for the 10% random replacement is not")
    print("  the one visible at this scale, and that the recipe's")
    print("  published justification rests on ablations over corpora")
    print("  a hundred million times this size. Reproducing the")
    print("  direction is worth something; reproducing the magnitude")
    print("  would need BERT's budget.")
    return accs, (e_b, t_b, x_b), (e_m, t_m, x_m)


def main():
    worst, checked = gradcheck()
    print(f"  [audit] {checked} sampled gradients vs central differences: "
          f"{worst:.2e}")

    rng = np.random.default_rng(42)
    corpus = gen_corpus(8000, rng)
    ids, keep = encode([s["words"] for s in corpus])
    test = gen_corpus(1000, np.random.default_rng(7))
    test_accept = make_acceptability(600, np.random.default_rng(13))

    demo1(corpus, ids, keep)

    print()
    print(f"  Pre-training three encoders, {STEPS:,} steps each.")
    print("  Masked-LM loss (nats; a uniform guess over the "
          f"vocabulary = {np.log(V):.2f}):")
    print()
    bi = Encoder(d=64, heads=4, layers=2, causal=False, seed=0, lr=3e-3)
    hist = pretrain(bi, ids, keep, STEPS, np.random.default_rng(1),
                    log_every=250)
    print("    bidirectional   " +
          "  ".join(f"{l:.2f}" for _, l in hist))

    ca = Encoder(d=64, heads=4, layers=2, causal=True, seed=0, lr=3e-3)
    hc = pretrain(ca, ids, keep, STEPS, np.random.default_rng(1),
                  log_every=250)
    print("    causal          " + "  ".join(f"{l:.2f}" for _, l in hc))

    mk = Encoder(d=64, heads=4, layers=2, causal=False, seed=0, lr=3e-3)
    hm = pretrain(mk, ids, keep, STEPS, np.random.default_rng(1),
                  recipe="mask", log_every=250)
    print("    100% [MASK]     " + "  ".join(f"{l:.2f}" for _, l in hm))
    print(f"    steps           " +
          "  ".join(f"{s:4d}" for s, _ in hist))
    print()
    print("  The floor is not zero: a masked subject is genuinely")
    print("  ambiguous, so some of that loss is irreducible.")

    demo2(bi, ca, test)
    demo3(bi, ca, [20, 50, 100, 400], test_accept)
    demo4(bi, mk, test, test_accept)


if __name__ == "__main__":
    main()
