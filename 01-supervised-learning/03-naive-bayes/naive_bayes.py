"""
naive_bayes.py --- companion code for "Naive Bayes" (Supervised Learning, Part 3).

Three demos:
  1. Multinomial Naive Bayes from scratch on a tiny SMS spam corpus.
  2. Comparison with scikit-learn's MultinomialNB (predictions agree).
  3. Most informative tokens per class --- the words that push the
     log-posterior toward spam or toward ham the most.

Dependencies: numpy, scikit-learn. Runs in well under a second.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


SEPARATOR = "=" * 72


def banner(title: str) -> None:
    print()
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)
    print()


# ---------------------------------------------------------------------------
# Tiny SMS spam/ham corpus (synthetic but in the style of the UCI SMS dataset)
# ---------------------------------------------------------------------------

TRAINING = [
    # (message, label)
    ("free entry win 1000 cash prize text now",                      "spam"),
    ("urgent your account has been compromised click link",          "spam"),
    ("congratulations you have won a free iphone",                   "spam"),
    ("call this number to claim your free holiday voucher",          "spam"),
    ("free ringtones text yes to 80085 now",                         "spam"),
    ("you have been selected for a cash reward please reply",        "spam"),
    ("urgent we tried to deliver your package call us back",         "spam"),
    ("limited time offer click here for free entry",                 "spam"),
    ("you won a brand new car text car to claim",                    "spam"),
    ("free prize click the link below to redeem",                    "spam"),
    ("urgent action required your bank account is locked",           "spam"),
    ("text stop to opt out of these alerts now",                     "spam"),

    ("can you pick up bread on your way home",                       "ham"),
    ("running late will be there in 20 minutes",                     "ham"),
    ("dinner at 7 tonight at the usual place",                       "ham"),
    ("thanks for the lovely birthday card",                          "ham"),
    ("see you at the office tomorrow morning",                       "ham"),
    ("please grab milk and eggs from the shop",                      "ham"),
    ("call me when you get home from work",                          "ham"),
    ("happy anniversary love you so much",                           "ham"),
    ("the kids are asleep can we chat now",                          "ham"),
    ("did you remember to lock the back door",                       "ham"),
    ("the meeting is moved to 3pm in the boardroom",                 "ham"),
    ("can you send me the report when you get a moment",             "ham"),
]

TEST = [
    ("free entry to win 100 cash text now",                          "spam"),
    ("please pick up bread on the way home",                         "ham"),
    ("urgent call this number to claim your prize",                  "spam"),
    ("running late will be home in 20 mins",                         "ham"),
    ("congratulations you won a holiday voucher",                    "spam"),
    ("can you grab milk thanks",                                     "ham"),
    ("click here for free ringtones now",                            "spam"),
    ("see you at 7 for dinner",                                      "ham"),
]


def tokenise(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Demo 1 --- Multinomial NB from scratch
# ---------------------------------------------------------------------------

class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes = []
        self.log_prior = {}
        self.log_likelihood = {}      # class -> {token -> log P(token | class)}
        self.log_unseen = {}          # class -> log P(unseen token | class)
        self.vocab = set()

    def fit(self, X, y):
        # X is a list of token-list documents; y is a list of class labels
        n_docs = len(X)
        class_counts = Counter(y)
        self.classes = sorted(class_counts)

        # Build vocabulary
        for tokens in X:
            self.vocab.update(tokens)
        V = len(self.vocab)

        # Per-class token counts and totals
        token_count_per_class = {c: Counter() for c in self.classes}
        total_per_class = {c: 0 for c in self.classes}
        for tokens, label in zip(X, y):
            token_count_per_class[label].update(tokens)
            total_per_class[label] += len(tokens)

        # Log prior and log likelihood
        for c in self.classes:
            self.log_prior[c] = math.log(class_counts[c] / n_docs)
            denom = total_per_class[c] + self.alpha * V
            ll = {}
            for token in self.vocab:
                count = token_count_per_class[c].get(token, 0)
                ll[token] = math.log((count + self.alpha) / denom)
            self.log_likelihood[c] = ll
            self.log_unseen[c] = math.log(self.alpha / denom)

    def log_posterior(self, tokens):
        """Return {class: log P(class | tokens)} (up to a constant)."""
        scores = {}
        for c in self.classes:
            s = self.log_prior[c]
            ll = self.log_likelihood[c]
            for tok in tokens:
                s += ll.get(tok, self.log_unseen[c])
            scores[c] = s
        return scores

    def predict(self, X):
        out = []
        for tokens in X:
            scores = self.log_posterior(tokens)
            out.append(max(scores, key=scores.get))
        return out


def demo_from_scratch():
    banner("DEMO 1 --- Multinomial Naive Bayes from scratch")

    train_X = [tokenise(msg) for msg, _ in TRAINING]
    train_y = [label for _, label in TRAINING]

    test_X = [tokenise(msg) for msg, _ in TEST]
    test_y = [label for _, label in TEST]

    print("Training Multinomial Naive Bayes from scratch...")
    nb = MultinomialNaiveBayes(alpha=1.0)
    nb.fit(train_X, train_y)

    print(f"  Vocabulary size : {len(nb.vocab)} unique tokens")
    print(f"  Class priors    :  "
          f"P(spam)={math.exp(nb.log_prior['spam']):.3f}  "
          f"P(ham)={math.exp(nb.log_prior['ham']):.3f}")
    print(f"  Smoothing alpha : {nb.alpha}")

    preds = nb.predict(test_X)
    print()
    print("Test set predictions:")
    print(f"  {'message':<55} {'true':<6} {'predicted'}")
    n_correct = 0
    for (msg, true), pred in zip(TEST, preds):
        ok = "OK" if pred == true else "X "
        if pred == true:
            n_correct += 1
        print(f"  {('\"' + msg + '\"'):<55} {true:<6} {pred:<6} {ok}")
    print()
    print(f"  Accuracy: {n_correct / len(TEST):.3f}  ({n_correct}/{len(TEST)})")
    return nb, train_X, train_y, test_X, test_y


# ---------------------------------------------------------------------------
# Demo 2 --- scikit-learn comparison
# ---------------------------------------------------------------------------

def demo_sklearn(train_X_tokens, train_y, test_X_tokens, test_y):
    banner("DEMO 2 --- Same data, scikit-learn MultinomialNB")

    train_strings = [" ".join(toks) for toks in train_X_tokens]
    test_strings = [" ".join(toks) for toks in test_X_tokens]

    vec = CountVectorizer()
    X_train = vec.fit_transform(train_strings)
    X_test = vec.transform(test_strings)
    print(f"  CountVectorizer matrix shape: {X_train.shape}  (CSR sparse)")

    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, train_y)
    preds = model.predict(X_test)

    print()
    print("Test set predictions (sklearn):")
    print(f"  {'message':<55} {'true':<6} {'predicted'}")
    n_correct = 0
    for (msg, true), pred in zip(TEST, preds):
        ok = "OK" if pred == true else "X "
        if pred == true:
            n_correct += 1
        print(f"  {('\"' + msg + '\"'):<55} {true:<6} {pred:<6} {ok}")
    print()
    print(f"  Accuracy: {n_correct / len(TEST):.3f}  ({n_correct}/{len(TEST)})")


# ---------------------------------------------------------------------------
# Demo 3 --- Most informative features per class
# ---------------------------------------------------------------------------

def demo_informative(nb):
    banner("DEMO 3 --- Most informative tokens per class")

    print("For each token, the log-likelihood-ratio:")
    print("  log P(token | spam) - log P(token | ham)")
    print("Positive values lean toward spam, negative toward ham.")
    print()

    tokens = sorted(nb.vocab)
    ratios = []
    for tok in tokens:
        spam_ll = nb.log_likelihood["spam"].get(tok, nb.log_unseen["spam"])
        ham_ll = nb.log_likelihood["ham"].get(tok, nb.log_unseen["ham"])
        ratios.append((tok, spam_ll - ham_ll))

    # Top spam-leaning tokens (highest ratio)
    top_spam = sorted(ratios, key=lambda kv: -kv[1])[:8]
    # Top ham-leaning tokens (lowest ratio)
    top_ham = sorted(ratios, key=lambda kv: kv[1])[:8]

    print("Top spam-leaning tokens:")
    for tok, r in top_spam:
        print(f"  {tok:<20} log-ratio = {r:+.3f}")
    print()
    print("Top ham-leaning tokens:")
    for tok, r in top_ham:
        print(f"  {tok:<20} log-ratio = {r:+.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    nb, train_X, train_y, test_X, test_y = demo_from_scratch()
    demo_sklearn(train_X, train_y, test_X, test_y)
    demo_informative(nb)
    print()


if __name__ == "__main__":
    main()
