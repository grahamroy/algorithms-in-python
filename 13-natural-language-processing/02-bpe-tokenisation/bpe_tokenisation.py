"""
bpe_tokenisation.py --- companion code for "Tokenisation"
(Natural Language Processing, Part 2).

Part 1 showed that embeddings pack meaning into vectors when trained
on raw text. But every embedding matrix has a row budget. word2vec
assigns one row per unique word -- and crashes on anything unseen.
fastText patched this with character n-grams, but the pieces were
hand-chosen (all 3-to-5-grams). This article asks: can the corpus
itself decide which pieces to keep?

Three answers, all from scratch:

  1. Byte-Pair Encoding (BPE) -- Sennrich et al., 2016.
     Start from characters. Repeatedly merge the most frequent
     adjacent pair. The merge table IS the tokeniser: apply merges
     in learned order to any new string, known or unknown.

  2. WordPiece -- Schuster & Nakajima, 2012 (used by BERT).
     Same loop, different score: pick the pair whose merged
     frequency divided by the product of its parts' frequencies
     is highest -- a mutual-information criterion that prefers
     merges which are more "surprising" than raw count would say.

  3. Unigram LM -- Kudo, 2018 (used by SentencePiece / T5 / Llama).
     Start BIG (all substrings up to a length cap), then prune:
     score every token by how much removing it would hurt the
     corpus log-likelihood under a unigram language model, and
     drop the least painful ones until the target vocabulary size
     is reached. Tokenisation is then the Viterbi-best segmentation.

Stage: Part 1's synthetic micro-language (3 topics, 106 words,
~165k tokens, seeded). Same corpus, now measured through a different
lens: not "what neighbours does a word have?" but "what pieces should
a word be built from?"

Demonstrates:
  1. BPE merge table and segmentation of known + OOV words.
  2. WordPiece divergence from BPE on the same corpus.
  3. Unigram LM with Viterbi decoding and EM-trained probabilities.
  4. The compression curve: tokens-per-word vs vocabulary size for
     each method -- the information-theoretic tradeoff that decides
     how large a model's vocabulary should be.
  5. OOV resilience: all three methods segment Part 1's held-out
     words (princesses, roosters, compasses) into known pieces.

Everything is plain Python + NumPy, seeded, deterministic.
Dependencies: numpy.  Runs in about 30 seconds.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from collections import Counter
import re
import math

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
ALL_NOUNS = [n for t in TOPICS.values() for n in t["nouns"]]
HELD_OUT_PLURALS = ["princesses", "roosters", "compasses"]


def plural(noun):
    if noun.endswith("s") or noun.endswith("sh"):
        return noun + "es"
    if noun.endswith("y"):
        return noun[:-1] + "ies"
    return noun + "s"


SIG = {}
for _t in TOPICS.values():
    _ns = _t["nouns"]
    for _i, _n in enumerate(_ns):
        SIG[_n] = _ns[(_i + len(_ns) // 2) % len(_ns)]
    for (_m, _f) in _t["gender"]:
        SIG[_f] = SIG[_m]


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


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# -----------------------------------------------------------------------
# BPE -- Byte-Pair Encoding (Sennrich et al., 2016).
# -----------------------------------------------------------------------

def learn_bpe(word_freqs, num_merges):
    """Learn BPE merges from a word frequency dictionary.

    Each word is represented as a tuple of characters plus an end-of-word
    marker '_'. Repeatedly merge the highest-frequency adjacent pair.
    Returns the ordered list of merges.
    """
    # Represent each word as space-separated characters + end marker
    splits = {}
    for word, freq in word_freqs.items():
        symbols = tuple(word) + ("_",)
        splits[symbols] = freq

    merges = []
    for i in range(num_merges):
        # Count all adjacent pairs
        pair_counts = Counter()
        for symbols, freq in splits.items():
            for j in range(len(symbols) - 1):
                pair_counts[(symbols[j], symbols[j + 1])] += freq
        if not pair_counts:
            break
        best = max(pair_counts, key=pair_counts.get)
        merges.append(best)

        # Apply the merge
        new_splits = {}
        for symbols, freq in splits.items():
            new_symbols = []
            j = 0
            while j < len(symbols):
                if (j < len(symbols) - 1
                        and symbols[j] == best[0]
                        and symbols[j + 1] == best[1]):
                    new_symbols.append(best[0] + best[1])
                    j += 2
                else:
                    new_symbols.append(symbols[j])
                    j += 1
            new_splits[tuple(new_symbols)] = freq
        splits = new_splits

    return merges, splits


def apply_bpe(word, merges):
    """Segment a word using a learned BPE merge table."""
    symbols = list(word) + ["_"]
    for (a, b) in merges:
        new_symbols = []
        j = 0
        while j < len(symbols):
            if (j < len(symbols) - 1
                    and symbols[j] == a and symbols[j + 1] == b):
                new_symbols.append(a + b)
                j += 2
            else:
                new_symbols.append(symbols[j])
                j += 1
        symbols = new_symbols
    return symbols


def bpe_vocab_set(merges):
    """Return the full BPE vocabulary: all characters + all merged tokens."""
    vocab = set()
    for (a, b) in merges:
        vocab.add(a)
        vocab.add(b)
        vocab.add(a + b)
    return vocab


# -----------------------------------------------------------------------
# WordPiece -- Schuster & Nakajima, 2012 (BERT's tokeniser).
# -----------------------------------------------------------------------

def learn_wordpiece(word_freqs, num_merges):
    """Learn WordPiece merges using mutual-information scoring.

    Instead of picking the pair with the highest raw count, WordPiece
    scores each pair by: freq(ab) / (freq(a) * freq(b)). This favours
    merges where the pair is surprisingly common given how common each
    part is on its own.
    """
    splits = {}
    for word, freq in word_freqs.items():
        symbols = tuple(word) + ("_",)
        splits[symbols] = freq

    merges = []
    for i in range(num_merges):
        # Count all adjacent pairs AND individual symbol frequencies
        pair_counts = Counter()
        symbol_counts = Counter()
        for symbols, freq in splits.items():
            for s in symbols:
                symbol_counts[s] += freq
            for j in range(len(symbols) - 1):
                pair_counts[(symbols[j], symbols[j + 1])] += freq
        if not pair_counts:
            break

        # Score by mutual information: freq(pair) / (freq(a) * freq(b))
        best = max(pair_counts,
                   key=lambda p: pair_counts[p] / (
                       symbol_counts[p[0]] * symbol_counts[p[1]]))
        merges.append(best)

        # Apply the merge
        new_splits = {}
        for symbols, freq in splits.items():
            new_symbols = []
            j = 0
            while j < len(symbols):
                if (j < len(symbols) - 1
                        and symbols[j] == best[0]
                        and symbols[j + 1] == best[1]):
                    new_symbols.append(best[0] + best[1])
                    j += 2
                else:
                    new_symbols.append(symbols[j])
                    j += 1
            new_splits[tuple(new_symbols)] = freq
        splits = new_splits

    return merges, splits


def apply_wordpiece(word, merges):
    """Segment a word using a learned WordPiece merge table."""
    return apply_bpe(word, merges)  # Application is identical


# -----------------------------------------------------------------------
# Unigram LM -- Kudo, 2018 (SentencePiece / T5 / Llama).
# -----------------------------------------------------------------------

def viterbi_segment(word, vocab_scores):
    """Find the highest-probability segmentation of word under a
    unigram language model, using dynamic programming (Viterbi)."""
    word = word + "_"
    n = len(word)
    # best[i] = (best_log_prob to reach position i, backpointer)
    best = [(-1e18, 0)] * (n + 1)
    best[0] = (0.0, 0)
    for i in range(n):
        if best[i][0] == -1e18:
            continue
        for j in range(i + 1, n + 1):
            piece = word[i:j]
            if piece in vocab_scores:
                score = best[i][0] + vocab_scores[piece]
                if score > best[j][0]:
                    best[j] = (score, i)
    # Backtrack
    pieces = []
    pos = n
    while pos > 0:
        start = best[pos][1]
        pieces.append(word[start:pos])
        pos = start
    pieces.reverse()
    return pieces


def learn_unigram(word_freqs, target_vocab_size, initial_max_len=8,
                  shrink_factor=0.8, n_em_iter=5):
    """Learn a Unigram LM vocabulary by iterative pruning.

    1. Seed with all substrings of words up to initial_max_len.
    2. Run EM to estimate token log-probabilities.
    3. Score each token by how much corpus likelihood drops if removed.
    4. Keep the top shrink_factor fraction (but never drop single chars).
    5. Repeat until target size is reached.
    """
    # Build initial vocabulary: all observed substrings + end marker
    initial_vocab = set()
    for word in word_freqs:
        w = word + "_"
        for i in range(len(w)):
            for j in range(i + 1, min(i + initial_max_len + 1, len(w) + 1)):
                initial_vocab.add(w[i:j])
    # Always include individual characters
    chars = set()
    for word in word_freqs:
        for ch in word + "_":
            chars.add(ch)
    initial_vocab |= chars

    vocab_list = sorted(initial_vocab)
    total_words = sum(word_freqs.values())

    for prune_round in range(50):
        if len(vocab_list) <= target_vocab_size:
            break

        # EM: estimate token probabilities from Viterbi segmentations
        vocab_scores = {t: math.log(1.0 / len(vocab_list))
                        for t in vocab_list}
        for _em in range(n_em_iter):
            token_counts = Counter()
            for word, freq in word_freqs.items():
                pieces = viterbi_segment(word, vocab_scores)
                for p in pieces:
                    token_counts[p] += freq
            # Pseudo-count floor: single chars always stay usable
            for t in vocab_list:
                if len(t) == 1 and token_counts[t] == 0:
                    token_counts[t] = 1
            total_tokens = sum(token_counts.values())
            vocab_scores = {}
            for t in vocab_list:
                c = token_counts.get(t, 0)
                if c > 0:
                    vocab_scores[t] = math.log(c / total_tokens)
                else:
                    vocab_scores[t] = -1e18

        # Score each token: loss in corpus likelihood if removed
        losses = {}
        for token in vocab_list:
            if len(token) == 1:
                losses[token] = 1e18  # never drop single chars
                continue
            # Try segmenting without this token
            reduced = {t: s for t, s in vocab_scores.items() if t != token}
            loss = 0.0
            for word, freq in word_freqs.items():
                pieces_with = viterbi_segment(word, vocab_scores)
                pieces_without = viterbi_segment(word, reduced)
                lp_with = sum(vocab_scores.get(p, -1e18) for p in pieces_with)
                lp_without = sum(reduced.get(p, -1e18) for p in pieces_without)
                loss += freq * (lp_with - lp_without)
            losses[token] = loss

        # Keep top fraction by loss (highest loss = most important)
        keep_n = max(target_vocab_size,
                     int(len(vocab_list) * shrink_factor))
        sorted_tokens = sorted(vocab_list, key=lambda t: -losses[t])
        vocab_list = sorted(sorted_tokens[:keep_n])

    # Final EM pass
    vocab_scores = {t: math.log(1.0 / len(vocab_list))
                    for t in vocab_list}
    for _em in range(n_em_iter):
        token_counts = Counter()
        for word, freq in word_freqs.items():
            pieces = viterbi_segment(word, vocab_scores)
            for p in pieces:
                token_counts[p] += freq
        for t in vocab_list:
            if len(t) == 1 and token_counts[t] == 0:
                token_counts[t] = 1
        total_tokens = sum(token_counts.values())
        vocab_scores = {}
        for t in vocab_list:
            c = token_counts.get(t, 0)
            if c > 0:
                vocab_scores[t] = math.log(c / total_tokens)
            else:
                vocab_scores[t] = -1e18

    return vocab_list, vocab_scores


# -----------------------------------------------------------------------
# Tokenisation statistics.
# -----------------------------------------------------------------------

def tokens_per_word(tokenise_fn, words):
    """Average number of tokens per word."""
    total_tokens = 0
    for w in words:
        total_tokens += len(tokenise_fn(w))
    return total_tokens / len(words)


def corpus_token_count(tokenise_fn, flat_words):
    """Total tokens when the full corpus is tokenised."""
    return sum(len(tokenise_fn(w)) for w in flat_words)


# -----------------------------------------------------------------------
# Demos.
# -----------------------------------------------------------------------

def demo1(word_freqs, all_words_flat):
    """DEMO 1 -- BPE: greedy merging from characters."""
    banner("DEMO 1 --- BPE: greedy merging from characters")

    merges, final_splits = learn_bpe(word_freqs, num_merges=200)

    print("  Start from characters. Merge the most frequent pair.")
    print("  The first 15 merges on this corpus:")
    print()
    for i, (a, b) in enumerate(merges[:15]):
        print(f"    merge {i+1:3d}:  {a!r:8s} + {b!r:8s}  ->  "
              f"{(a+b)!r}")
    print()

    # Show what happens to a few words
    examples = ["king", "queen", "princesses", "harvests",
                "navigates", "roosters", "compasses"]
    print("  Applied to words (known and OOV):")
    print()
    for w in examples:
        pieces = apply_bpe(w, merges)
        oov = " (OOV)" if w in HELD_OUT_PLURALS else ""
        print(f"    {w:14s} -> {' | '.join(pieces)}{oov}")
    print()

    n_vocab = len(set(tok for syms in final_splits
                      for tok in syms)) + 27  # base chars
    unique_words = list(word_freqs.keys())
    tpw = tokens_per_word(lambda w: apply_bpe(w, merges), unique_words)
    total_toks = corpus_token_count(lambda w: apply_bpe(w, merges),
                                    all_words_flat)
    print(f"  Vocabulary: {n_vocab} tokens after 200 merges")
    print(f"  Avg tokens per unique word: {tpw:.2f}")
    print(f"  Corpus tokens: {total_toks:,} "
          f"(vs {len(all_words_flat):,} raw words)")
    print()
    print("  Every OOV word decomposes into known pieces. The merge")
    print("  table is the entire tokeniser: no dictionary lookup, no")
    print("  hand-written rules -- just 200 lines that say 'whenever")
    print("  you see X next to Y, join them'. The pieces the corpus")
    print("  chose are the pieces the model gets to see.")

    return merges


def demo2(word_freqs, bpe_merges):
    """DEMO 2 -- WordPiece: mutual-information scoring."""
    banner("DEMO 2 --- WordPiece: mutual-information scoring")

    wp_merges, _ = learn_wordpiece(word_freqs, num_merges=200)

    print("  Same loop, different score. BPE picks the most frequent")
    print("  pair; WordPiece picks the pair whose co-occurrence is")
    print("  most surprising given each part's individual frequency:")
    print()
    print("    score(a,b) = freq(ab) / (freq(a) * freq(b))")
    print()
    print("  First 15 merges compared:")
    print()
    print(f"    {'#':>3s}   {'BPE':20s}   {'WordPiece':20s}   {'same?':5s}")
    print(f"    {'---':>3s}   {'----':20s}   {'---------':20s}   {'-----':5s}")
    for i in range(15):
        bpe_m = f"{bpe_merges[i][0]!r}+{bpe_merges[i][1]!r}"
        wp_m = f"{wp_merges[i][0]!r}+{wp_merges[i][1]!r}"
        same = "yes" if bpe_merges[i] == wp_merges[i] else "no"
        print(f"    {i+1:3d}   {bpe_m:20s}   {wp_m:20s}   {same:5s}")
    print()

    # Count how many of the first 200 merges are identical
    bpe_set = set(bpe_merges[:200])
    wp_set = set(wp_merges[:200])
    overlap = len(bpe_set & wp_set)

    # Show divergent segmentations
    examples = ["princess", "navigates", "princesses", "compasses"]
    print("  Segmentation comparison on selected words:")
    print()
    for w in examples:
        bp = " | ".join(apply_bpe(w, bpe_merges))
        wp = " | ".join(apply_wordpiece(w, wp_merges))
        same = "(same)" if bp == wp else "(different)"
        print(f"    {w:14s}  BPE:       {bp}")
        print(f"    {'':14s}  WordPiece: {wp}  {same}")
        print()

    print(f"  Of 200 merges: {overlap} identical, "
          f"{200 - overlap} different.")
    print()
    print("  The rare-pair bonus makes WordPiece greedier on")
    print("  morphologically informative fragments: it tends to learn")
    print("  whole suffixes earlier because they appear reliably with")
    print("  specific stems, even if their raw count is modest.")

    return wp_merges


def demo3(word_freqs, bpe_merges, wp_merges):
    """DEMO 3 -- Unigram LM: start big, prune by likelihood."""
    banner("DEMO 3 --- Unigram LM: start big, prune by likelihood")

    # Target smaller than the word vocabulary (~128 forms) to force
    # real subword segmentation, not whole-word memorisation.
    target = 80

    print("  BPE and WordPiece build up from characters. Unigram goes")
    print("  the other way: start with every substring (up to length")
    print("  8), then PRUNE. At each round, remove the tokens whose")
    print("  absence hurts corpus likelihood the least.")
    print()
    print(f"  Target vocabulary size: {target}")
    print("  Training...", end="", flush=True)

    vocab_list, vocab_scores = learn_unigram(word_freqs, target)
    print(f" done. Final vocab: {len(vocab_list)} tokens.")
    print()

    # Show some vocabulary members
    scored = sorted(vocab_scores.items(), key=lambda x: -x[1])
    print("  Top-10 tokens by probability:")
    print()
    for tok, lp in scored[:10]:
        print(f"    {tok!r:12s}  p = {math.exp(lp):.4f}")
    print()

    # Segment examples
    examples = ["king", "queen", "princess", "navigates",
                "princesses", "roosters", "compasses"]
    print("  Viterbi segmentation (highest-probability split):")
    print()
    for w in examples:
        pieces = viterbi_segment(w, vocab_scores)
        oov = " (OOV)" if w in HELD_OUT_PLURALS else ""
        print(f"    {w:14s} -> {' | '.join(pieces)}{oov}")
    print()

    # Compare all three on OOV words
    print("  OOV words across all three methods:")
    print()
    print(f"    {'word':14s}  {'BPE':24s}  {'WordPiece':24s}  "
          f"{'Unigram':24s}")
    print(f"    {'----':14s}  {'---':24s}  {'---------':24s}  "
          f"{'-------':24s}")
    for w in HELD_OUT_PLURALS:
        bp = " ".join(apply_bpe(w, bpe_merges))
        wp = " ".join(apply_wordpiece(w, wp_merges))
        ug = " ".join(viterbi_segment(w, vocab_scores))
        print(f"    {w:14s}  {bp:24s}  {wp:24s}  {ug:24s}")
    print()
    print("  All three handle the unseen words. BPE and WordPiece")
    print("  apply their merge rules left-to-right; Unigram finds")
    print("  the globally optimal segmentation via dynamic programming.")
    print("  Different algorithms, same promise: no word is unsayable.")

    return vocab_list, vocab_scores


def demo4(word_freqs, all_words_flat):
    """DEMO 4 -- The compression curve."""
    banner("DEMO 4 --- The compression curve")

    print("  More merges = fewer tokens per word = larger vocabulary.")
    print("  The tradeoff is information-theoretic: each merge buys")
    print("  compression (shorter sequences) at the cost of a bigger")
    print("  embedding table. Measured on this corpus:")
    print()

    merge_counts = [0, 25, 50, 100, 150, 200, 300, 400]
    print(f"    {'merges':>6s}   {'vocab':>5s}   {'tok/word':>8s}   "
          f"{'corpus toks':>11s}")
    print(f"    {'------':>6s}   {'-----':>5s}   {'--------':>8s}   "
          f"{'----------':>11s}")

    unique_words = list(word_freqs.keys())

    for nm in merge_counts:
        merges_i, splits_i = learn_bpe(word_freqs, nm)
        vocab_i = set(tok for syms in splits_i for tok in syms)
        # Add base characters
        base_chars = set()
        for w in word_freqs:
            for ch in w + "_":
                base_chars.add(ch)
        vocab_i |= base_chars
        vs = len(vocab_i)
        tpw = tokens_per_word(lambda w: apply_bpe(w, merges_i),
                              unique_words)
        ct = corpus_token_count(lambda w: apply_bpe(w, merges_i),
                                all_words_flat)
        print(f"    {nm:6d}   {vs:5d}   {tpw:8.2f}   {ct:11,}")

    print()
    print("  At 0 merges every character is a token (maximum tokens,")
    print("  minimum vocabulary). Each merge trades one vocabulary")
    print("  slot for shorter sequences everywhere that pair appears.")
    print("  Real tokenisers (GPT-4: ~100k vocab, Llama: 32k) sit on")
    print("  this curve at the point where the embedding table's")
    print("  memory cost balances the sequence-length savings in")
    print("  attention (which is O(n^2) in length).")
    print()
    print("  The choice is architectural: a larger vocabulary means")
    print("  shorter sequences (faster attention, more context per")
    print("  window) but a bigger embedding matrix. BPE's merge count")
    print("  is the single knob that sets this tradeoff.")


def main():
    rng = np.random.default_rng(0)
    sents = gen_corpus(30000, rng)
    all_words_flat = [w for s in sents for w in s]
    print(f"  Corpus: {len(sents):,} sentences, "
          f"{len(all_words_flat):,} tokens")

    # Build word frequency dict (each word as it appears)
    word_freqs = Counter(all_words_flat)

    bpe_merges = demo1(word_freqs, all_words_flat)
    wp_merges = demo2(word_freqs, bpe_merges)
    demo3(word_freqs, bpe_merges, wp_merges)
    demo4(word_freqs, all_words_flat)


if __name__ == "__main__":
    main()
