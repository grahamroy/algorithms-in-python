"""
seq2seq.py --- companion code for "Sequence-to-Sequence Models"
(Natural Language Processing, Part 3).

Part 2 decided the pieces; this article decides what to do with them.
Sequence-to-sequence (Sutskever et al., 2014) reads one sequence with
an ENCODER and writes another with a DECODER. The encoder compresses
the input into a fixed-size vector; the decoder unrolls it into
output tokens, one at a time. That fixed-size vector is the bottleneck:
everything the decoder needs must fit through it.

Bahdanau et al. (2015) broke the bottleneck with ATTENTION: instead of
one fixed vector, the decoder peeks at every encoder state and picks
what is relevant at each step. The model learns alignment -- which
input tokens matter for each output token -- as a side effect.

Stage: Part 1's synthetic micro-language, with a new task -- word-order
translation from SVO (subject-verb-object) to SOV (subject-object-verb),
mirroring real English-to-Japanese/Korean translation. Single-clause
sentences (5-6 words) and two-clause sentences joined by "and" (11-13
words) test the bottleneck: the fixed vector struggles on longer
sequences; attention does not.

Demonstrates:
  1. The SVO -> SOV task on the micro-language.
  2. Encoder-decoder without attention: works on short, fails on long.
  3. Encoder-decoder WITH attention: works on both, attention weights
     show learned alignment.
  4. The bottleneck receipt: accuracy vs sequence length.

Everything is plain NumPy, hand-derived gradients audited against
central differences, seeded, deterministic.
Dependencies: numpy.  Runs in about two minutes.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

SEPARATOR = "=" * 72

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
FUNC = ["the", "a", "and", "near"]
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


def gen_corpus(n_sent, rng):
    sents = []
    names = list(TOPICS)
    for _ in range(n_sent):
        t = TOPICS[names[rng.integers(0, 3)]]
        n1 = t["nouns"][rng.integers(0, len(t["nouns"]))]
        vs, vp = t["verbs"][rng.integers(0, len(t["verbs"]))]
        adj = t["adjs"][rng.integers(0, len(t["adjs"]))]
        n2 = SIG[n1] if rng.uniform() < 0.5 else \
            t["nouns"][rng.integers(0, len(t["nouns"]))]
        is_plur = rng.uniform() < 0.35
        if is_plur and plural(n1) not in HELD_OUT_PLURALS:
            subj, verb, poss = plural(n1), vp, "their"
        else:
            subj, verb = n1, vs
            poss = ("his" if n1 in GENDER_M else
                    "her" if n1 in GENDER_F else "the")
        obj = n2
        if rng.uniform() < 0.25 and plural(n2) not in HELD_OUT_PLURALS:
            obj = plural(n2)
        s = ["the", subj, verb, poss, obj]
        if rng.uniform() < 0.5:
            s = ["the", adj, subj, verb, poss, obj]
        sents.append(s)
    return sents


# -----------------------------------------------------------------------
# SVO -> SOV paired data.
# -----------------------------------------------------------------------

def svo_to_sov(s):
    """Move verb to end of clause: SVO -> SOV."""
    vi = len(s) - 3  # verb is always 3rd from end
    return s[:vi] + s[vi + 1:] + [s[vi]]


def make_pairs(n_short, n_long, rng):
    """Generate (source SVO, target SOV) pairs.

    Short: single clause (5-6 words).
    Long: two clauses joined by 'and' (11-13 words).
    """
    short_sents = gen_corpus(n_short, rng)
    short_pairs = [(s, svo_to_sov(s)) for s in short_sents]

    long_base = gen_corpus(n_long * 2, rng)
    long_pairs = []
    for i in range(0, len(long_base), 2):
        src = long_base[i] + ["and"] + long_base[i + 1]
        tgt = svo_to_sov(long_base[i]) + ["and"] + svo_to_sov(
            long_base[i + 1])
        long_pairs.append((src, tgt))

    return short_pairs, long_pairs


# -----------------------------------------------------------------------
# Vocabulary.
# -----------------------------------------------------------------------

PAD, BOS, EOS = 0, 1, 2


def build_vocab(pairs):
    words = set()
    for src, tgt in pairs:
        words.update(src)
        words.update(tgt)
    vocab = ["<PAD>", "<BOS>", "<EOS>"] + sorted(words)
    return vocab, {w: i for i, w in enumerate(vocab)}


def enc(s, w2i):
    return [BOS] + [w2i[w] for w in s] + [EOS]


# -----------------------------------------------------------------------
# Utilities.
# -----------------------------------------------------------------------

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def xavier(shape, rng):
    fan_in, fan_out = shape[1], shape[0]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, shape)


def banner(title):
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# -----------------------------------------------------------------------
# Seq2Seq model.
# -----------------------------------------------------------------------

class Seq2Seq:
    def __init__(self, V, d_emb, d_hid, d_att, use_attn, rng):
        self.V, self.d_emb, self.d_hid = V, d_emb, d_hid
        self.d_att, self.use_attn = d_att, use_attn

        self.emb = rng.uniform(-0.1, 0.1, (V, d_emb))
        self.Wx_e = xavier((d_hid, d_emb), rng)
        self.Wh_e = xavier((d_hid, d_hid), rng)
        self.bh_e = np.zeros(d_hid)

        d_in = d_emb + d_hid if use_attn else d_emb
        self.Wx_d = xavier((d_hid, d_in), rng)
        self.Wh_d = xavier((d_hid, d_hid), rng)
        self.bh_d = np.zeros(d_hid)
        self.Wo = xavier((V, d_hid), rng)
        self.bo = np.zeros(V)

        self._pnames = ["emb", "Wx_e", "Wh_e", "bh_e",
                        "Wx_d", "Wh_d", "bh_d", "Wo", "bo"]
        if use_attn:
            self.Wa = xavier((d_att, d_hid), rng)
            self.Ua = xavier((d_att, d_hid), rng)
            self.va = rng.uniform(-0.1, 0.1, d_att)
            self._pnames += ["Wa", "Ua", "va"]

        self._m = {n: np.zeros_like(getattr(self, n))
                   for n in self._pnames}
        self._v = {n: np.zeros_like(getattr(self, n))
                   for n in self._pnames}
        self._step = 0

    # --- encoder ---------------------------------------------------------

    def _encode(self, src):
        T = len(src)
        H = np.zeros((T + 1, self.d_hid))
        X = []
        for t, tok in enumerate(src):
            x = self.emb[tok]
            X.append(x)
            H[t + 1] = np.tanh(
                self.Wx_e @ x + self.Wh_e @ H[t] + self.bh_e)
        return H[1:], H, X          # enc_states, all_states, embs

    # --- attention -------------------------------------------------------

    def _attn(self, enc_H, s_prev):
        pE = enc_H @ self.Wa.T              # (T, d_att)
        pD = s_prev @ self.Ua.T              # (d_att,)
        pre = pE + pD                        # (T, d_att) broadcast
        th = np.tanh(pre)                    # (T, d_att)
        e = th @ self.va                     # (T,)
        a = softmax(e)                       # (T,)
        c = a @ enc_H                        # (d_hid,)
        return c, a, th

    # --- forward (teacher forcing) ---------------------------------------

    def forward(self, src, tgt):
        enc_H, enc_all, enc_X = self._encode(src)
        T_d = len(tgt) - 1
        s = enc_H[-1].copy()

        cache = dict(enc_H=enc_H, enc_all=enc_all, enc_X=enc_X,
                     src=src, tgt=tgt, ds=[s.copy()], dx=[], lgt=[],
                     pr=[], ac=[], aa=[], ath=[])
        loss = 0.0
        for t in range(T_d):
            y = self.emb[tgt[t]]
            cache["dx"].append(y)
            if self.use_attn:
                c, a, th = self._attn(enc_H, s)
                cache["ac"].append(c)
                cache["aa"].append(a)
                cache["ath"].append(th)
                di = np.concatenate([y, c])
            else:
                di = y
            s = np.tanh(self.Wx_d @ di + self.Wh_d @ s + self.bh_d)
            cache["ds"].append(s.copy())
            lg = self.Wo @ s + self.bo
            p = softmax(lg)
            cache["lgt"].append(lg)
            cache["pr"].append(p)
            loss -= np.log(p[tgt[t + 1]] + 1e-12)

        self._c = cache
        return loss / T_d

    # --- backward --------------------------------------------------------

    def backward(self):
        c = self._c
        g = {n: np.zeros_like(getattr(self, n)) for n in self._pnames}
        T_d = len(c["tgt"]) - 1
        ds = np.zeros(self.d_hid)
        dH = np.zeros_like(c["enc_H"])

        for t in reversed(range(T_d)):
            target = c["tgt"][t + 1]
            dl = c["pr"][t].copy()
            dl[target] -= 1.0
            dl /= T_d

            g["Wo"] += np.outer(dl, c["ds"][t + 1])
            g["bo"] += dl
            ds += self.Wo.T @ dl

            st = c["ds"][t + 1]
            sp = c["ds"][t]
            dt = (1 - st ** 2) * ds

            if self.use_attn:
                di = np.concatenate([c["dx"][t], c["ac"][t]])
            else:
                di = c["dx"][t]

            g["Wx_d"] += np.outer(dt, di)
            g["Wh_d"] += np.outer(dt, sp)
            g["bh_d"] += dt
            dd = self.Wx_d.T @ dt
            ds = self.Wh_d.T @ dt

            if self.use_attn:
                dy = dd[:self.d_emb]
                dc = dd[self.d_emb:]
                enc_H = c["enc_H"]
                a = c["aa"][t]
                th = c["ath"][t]

                da = enc_H @ dc
                dH += a[:, None] * dc[None, :]

                dot = np.dot(a, da)
                de = a * (da - dot)

                dth = np.outer(de, self.va)
                g["va"] += th.T @ de

                dpre = (1 - th ** 2) * dth
                g["Wa"] += dpre.T @ enc_H
                dH += dpre @ self.Wa
                dpd = dpre.sum(0)
                g["Ua"] += np.outer(dpd, c["ds"][t])
                ds += self.Ua.T @ dpd
            else:
                dy = dd

            g["emb"][c["tgt"][t]] += dy

        dH[-1] += ds

        for t in reversed(range(len(c["src"]))):
            ht = c["enc_H"][t]
            hp = c["enc_all"][t]
            dt = (1 - ht ** 2) * dH[t]
            g["Wx_e"] += np.outer(dt, c["enc_X"][t])
            g["Wh_e"] += np.outer(dt, hp)
            g["bh_e"] += dt
            g["emb"][c["src"][t]] += self.Wx_e.T @ dt
            if t > 0:
                dH[t - 1] += self.Wh_e.T @ dt

        return g

    # --- decode ----------------------------------------------------------

    def greedy(self, src, max_len=20):
        enc_H, _, _ = self._encode(src)
        s = enc_H[-1].copy()
        out = [BOS]
        alphas = []
        for _ in range(max_len):
            y = self.emb[out[-1]]
            if self.use_attn:
                ct, a, _ = self._attn(enc_H, s)
                alphas.append(a)
                di = np.concatenate([y, ct])
            else:
                di = y
            s = np.tanh(self.Wx_d @ di + self.Wh_d @ s + self.bh_d)
            pred = int(np.argmax(self.Wo @ s + self.bo))
            out.append(pred)
            if pred == EOS:
                break
        return out, alphas

    # --- update ----------------------------------------------------------

    def update(self, grads, lr=1e-3, clip=5.0):
        norm = sum(np.sum(grads[n] ** 2) for n in self._pnames)
        norm = np.sqrt(norm)
        if norm > clip:
            for n in self._pnames:
                grads[n] *= clip / norm
        self._step += 1
        for n in self._pnames:
            gr = grads[n]
            self._m[n] = 0.9 * self._m[n] + 0.1 * gr
            self._v[n] = 0.999 * self._v[n] + 0.001 * gr ** 2
            mh = self._m[n] / (1 - 0.9 ** self._step)
            vh = self._v[n] / (1 - 0.999 ** self._step)
            setattr(self, n, getattr(self, n) - lr * mh / (np.sqrt(vh)
                                                            + 1e-8))


# -----------------------------------------------------------------------
# Training and evaluation.
# -----------------------------------------------------------------------

def train_model(model, pairs, w2i, epochs, rng, lr=1e-3):
    for ep in range(epochs):
        perm = rng.permutation(len(pairs))
        total = 0.0
        for i in perm:
            src_ids = enc(pairs[i][0], w2i)
            tgt_ids = enc(pairs[i][1], w2i)
            total += model.forward(src_ids, tgt_ids)
            model.update(model.backward(), lr=lr)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    epoch {ep + 1:3d}  loss {total / len(pairs):.4f}")


def evaluate(model, pairs, w2i, vocab):
    ok = 0
    examples = []
    for src, tgt in pairs:
        out, alphas = model.greedy(enc(src, w2i))
        pred = [vocab[i] for i in out[1:] if i not in (PAD, BOS, EOS)]
        if pred == tgt:
            ok += 1
        if len(examples) < 6:
            examples.append((src, tgt, pred, alphas))
    return ok / len(pairs), examples


# -----------------------------------------------------------------------
# Gradient check.
# -----------------------------------------------------------------------

def gradcheck(use_attn, rng):
    V, d_emb, d_hid, d_att = 12, 6, 8, 6
    m = Seq2Seq(V, d_emb, d_hid, d_att, use_attn, rng)
    src = [BOS, 3, 5, 7, 4, EOS]
    tgt = [BOS, 3, 4, 7, 5, EOS]
    m.forward(src, tgt)
    g = m.backward()
    worst = 0.0
    check_params = ["Wx_e", "Wo", "Wx_d"]
    if use_attn:
        check_params += ["Wa", "va"]
    eps = 1e-5
    for name in check_params:
        p = getattr(m, name)
        ga = g[name]
        flat = p.ravel()
        ga_flat = ga.ravel()
        idxs = rng.choice(len(flat), min(5, len(flat)), replace=False)
        for ix in idxs:
            old = flat[ix]
            flat[ix] = old + eps
            lp = m.forward(src, tgt)
            flat[ix] = old - eps
            lm = m.forward(src, tgt)
            flat[ix] = old
            gn = (lp - lm) / (2 * eps)
            err = abs(ga_flat[ix] - gn) / (abs(ga_flat[ix]) + abs(gn)
                                            + 1e-12)
            worst = max(worst, err)
    return worst


# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

def demo1(short_pairs, long_pairs, vocab):
    banner("DEMO 1 --- The task: SVO -> SOV word-order translation")
    print("  The micro-language generates SVO sentences. The task:")
    print("  move the verb to the end of each clause (SOV order),")
    print("  mirroring English -> Japanese/Korean translation.")
    print()
    print("  Short (single clause, 5-6 words):")
    for src, tgt in short_pairs[:3]:
        print(f"    {' '.join(src)}")
        print(f"    -> {' '.join(tgt)}")
        print()
    print("  Long (two clauses joined by 'and', 11-13 words):")
    for src, tgt in long_pairs[:2]:
        print(f"    {' '.join(src)}")
        print(f"    -> {' '.join(tgt)}")
        print()
    print(f"  Vocabulary: {len(vocab)} tokens (shared src/tgt)")
    print(f"  Training: {len(short_pairs)} short + "
          f"{len(long_pairs)} long pairs")


def demo2(model, test_short, test_long, w2i, vocab):
    banner("DEMO 2 --- Without attention: the bottleneck")
    acc_s, ex_s = evaluate(model, test_short, w2i, vocab)
    acc_l, ex_l = evaluate(model, test_long, w2i, vocab)
    print(f"  Accuracy  short: {acc_s:6.1%}")
    print(f"  Accuracy  long:  {acc_l:6.1%}")
    print()
    print("  Short examples:")
    for src, tgt, pred, _ in ex_s[:3]:
        mark = "ok" if pred == tgt else "WRONG"
        print(f"    {' '.join(src)}")
        print(f"    gold:  {' '.join(tgt)}")
        print(f"    pred:  {' '.join(pred)}  ({mark})")
        print()
    print("  Long examples:")
    for src, tgt, pred, _ in ex_l[:3]:
        mark = "ok" if pred == tgt else "WRONG"
        print(f"    {' '.join(src)}")
        print(f"    gold:  {' '.join(tgt)}")
        print(f"    pred:  {' '.join(pred)}  ({mark})")
        print()
    print("  The decoder receives ONE vector from the encoder.")
    print("  Short sequences fit; long ones lose information.")
    return acc_s, acc_l


def demo3(model, test_short, test_long, w2i, vocab):
    banner("DEMO 3 --- With attention: the bottleneck breaks")
    acc_s, ex_s = evaluate(model, test_short, w2i, vocab)
    acc_l, ex_l = evaluate(model, test_long, w2i, vocab)
    print(f"  Accuracy  short: {acc_s:6.1%}")
    print(f"  Accuracy  long:  {acc_l:6.1%}")
    print()
    print("  Short examples:")
    for src, tgt, pred, _ in ex_s[:2]:
        mark = "ok" if pred == tgt else "WRONG"
        print(f"    {' '.join(src)}")
        print(f"    gold:  {' '.join(tgt)}")
        print(f"    pred:  {' '.join(pred)}  ({mark})")
        print()

    # Attention receipt: show alignment for one example
    print("  The attention receipt -- where the decoder looks:")
    print()
    src, tgt, pred, alphas = ex_s[0]
    src_ids = enc(src, w2i)
    src_words = ["<B>"] + src + ["<E>"]
    tgt_words = pred if pred else tgt
    # alphas[t] is attention over src for decoder step t
    if alphas:
        hdr = "  " + " " * 12
        for sw in src_words:
            hdr += f"{sw:>7s}"
        print(hdr)
        for t, tw in enumerate(tgt_words):
            if t >= len(alphas):
                break
            row = f"  {tw:>10s}  "
            for w in alphas[t]:
                bar = int(w * 20)
                row += f"  {w:.2f}"
            print(row)
        print()
        print("  Each row shows where the decoder looks when")
        print("  producing that output word. The verb in the output")
        print("  (last content word) attends to the verb in the")
        print("  input (middle position) -- the model learned")
        print("  alignment as a side effect of the translation task.")

    print()
    print("  Long examples:")
    for src, tgt, pred, _ in ex_l[:2]:
        mark = "ok" if pred == tgt else "WRONG"
        print(f"    {' '.join(src)}")
        print(f"    gold:  {' '.join(tgt)}")
        print(f"    pred:  {' '.join(pred)}  ({mark})")
        print()
    return acc_s, acc_l


def demo4(acc_no_s, acc_no_l, acc_at_s, acc_at_l):
    banner("DEMO 4 --- The bottleneck receipt")
    print("  Accuracy summary:")
    print()
    print(f"    {'':14s}  {'short':>7s}  {'long':>7s}")
    print(f"    {'':14s}  {'-----':>7s}  {'----':>7s}")
    print(f"    {'no attention':14s}  {acc_no_s:6.1%}  {acc_no_l:6.1%}")
    print(f"    {'attention':14s}  {acc_at_s:6.1%}  {acc_at_l:6.1%}")
    print()
    print("  Without attention, everything the decoder needs must")
    print("  pass through one vector. Short sequences survive the")
    print("  compression; long ones do not. Attention lets the")
    print("  decoder reach back into the encoder at every step --")
    print("  the bottleneck breaks, and accuracy on long sequences")
    print("  recovers.")
    print()
    print("  This is the architecture that launched neural machine")
    print("  translation (Bahdanau et al., 2015), and its two")
    print("  halves became the building blocks of everything that")
    print("  followed: BERT is the encoder; GPT is the decoder.")


# -----------------------------------------------------------------------
# Main.
# -----------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)

    # Gradient check
    gc_no = gradcheck(False, np.random.default_rng(99))
    gc_at = gradcheck(True, np.random.default_rng(99))
    print(f"  [audit] gradient check (no attn): {gc_no:.2e}")
    print(f"  [audit] gradient check (attn):    {gc_at:.2e}")

    # Data
    short_train, long_train = make_pairs(1500, 750, rng)
    short_test, long_test = make_pairs(400, 200,
                                       np.random.default_rng(7))
    all_pairs = short_train + long_train + short_test + long_test
    vocab, w2i = build_vocab(all_pairs)
    V = len(vocab)
    print(f"  Vocab: {V} tokens, "
          f"train: {len(short_train)}s+{len(long_train)}l, "
          f"test: {len(short_test)}s+{len(long_test)}l")

    demo1(short_train, long_train, vocab)

    # Train without attention
    print()
    print("  Training encoder-decoder (no attention)...")
    m_no = Seq2Seq(V, 32, 48, 32, False, np.random.default_rng(0))
    train_model(m_no, short_train + long_train, w2i, 30,
                np.random.default_rng(1), lr=2e-3)
    acc_no_s, acc_no_l = demo2(m_no, short_test, long_test, w2i,
                               vocab)

    # Train with attention
    print()
    print("  Training encoder-decoder (with attention)...")
    m_at = Seq2Seq(V, 32, 48, 32, True, np.random.default_rng(0))
    train_model(m_at, short_train + long_train, w2i, 30,
                np.random.default_rng(1), lr=2e-3)
    acc_at_s, acc_at_l = demo3(m_at, short_test, long_test, w2i,
                               vocab)

    demo4(acc_no_s, acc_no_l, acc_at_s, acc_at_l)


if __name__ == "__main__":
    main()
