# RLHF and DPO — Optimising a Reward Nobody Can Check

### *Algorithms in Python --- Natural Language Processing, Part 10*

---

Every article in this track so far has trained against a
target that was known. The next word. A label a function
computed. The object a document named. Even Part 9, taking
apart what fine-tuning changed, had an accuracy to point at.

Preference learning has none of that. All it is given is the
statement that one output was better than another, and the
first thing that has to be built is a loss assembled out of
comparisons:

```
P(a > b) = sigma(r(a) - r(b))
```

That much is standard, and the rest of the pipeline follows
from it. Fit a reward model to the comparisons. Optimise a
policy against that model. Ship the policy.

The problem is the last step, and it is not an engineering
problem. Once the policy has been optimised against the
reward model, the only number anyone can measure is the
reward model's opinion of the result — which is precisely
the quantity that was maximised, and therefore the one that
carries no information about whether anything improved.

In a real pipeline the true objective lives in people's
heads and the question is unanswerable. Here it is a
function.

---

## A worked example: a reward you can check

### The thing a real pipeline does not have

The true reward is **defined** over Part 1's micro-language —
three features the language already carries, weighted so no
single one dominates:

```
r*(s) = 1.0  if the sentence carries an adjective
      + 0.7  if the object is the subject's role partner
      + 0.4  if the subject is a royal noun
```

![[REWARD TABLE IMAGE]]

Because it is a function of the sentence, and because the
language enumerates, every quantity RLHF normally estimates
is exact at every step: the reward the policy is actually
earning, the reward the model *thinks* it is earning, the
grammatical mass, and the drift from where it started.

The reference policy — Part 7's decoder, untouched — scores
**0.7689**. The ceiling is **2.1**, reached by 192 of the
31,740 sentences. Everything that follows is an attempt to
move the first number towards the second using nothing but
comparisons.

### Fitting a reward model to noisy comparisons

3,000 comparisons are drawn by sampling pairs from the
reference policy and labelling them Bradley-Terry style, so
**69.37%** of the labels actually point at the better
sentence and the rest are the noise any human comparison
carries.

![[REWARD MODEL TABLE IMAGE]]

Now the diagnostic that matters, and the one I got wrong
before measuring it. The obvious story is that the reward
model is accurate where it was trained and inaccurate
elsewhere, so optimisation walks off a cliff. Over the whole
language its correlation with the truth is **0.6906**;
weighted by where the reference policy actually puts its
mass, **0.7092**.

Those are the same number. The reward model is a perfectly
reasonable judge of *sentences*, on and off the reference
distribution alike. If it were simply a bad model there would
be no puzzle here — and the thing that actually goes wrong
would stay hidden.

Which leaves a question the correlation cannot answer,
because it is computed over the language, and the language is
all the sentences there are:

> What does a model trained only on grammatical text
> think about text that is not?

### Goodhart, with a number attached

![[RLHF TABLE IMAGE]]

Read the `beta = 0.05` row twice. The true reward climbs from
0.7689 to **1.1985**, then ends at **0.3442** — below where
it started. The proxy does not come down with it.

Nothing went wrong with the optimiser. It maximised exactly
what it was given.

The `mass` column names the mechanism: grammatical output
collapses from 0.9770 to 0.2350. So look directly at what the
wrecked policy emits, and at what the reward model makes of
it:

```
grammatical          0.0150
mean r* (truth)      0.0225
mean r^ (proxy)      1.7109
mean r^ on the reference policy   -0.3150
```

The reward model rates this output **two full points above
the real text it was trained on**. The samples say why:

```
golden golden king feeds his sailor  r^=1.520  r*=0.0
the ancient king rules his sailor    r^=1.595  r*=0.0
```

The first is the whole article in one line. The reward model
pays 1.0 for an adjective, so the policy supplied **two** —
producing something that is not a sentence in any grammar.
The second takes the royal bonus and pairs a king with a
sailor, breaking the topic agreement that makes a string a
sentence at all.

The reward model learned *features*. The policy supplied
features without the form. And nothing in a loss built out of
comparisons between grammatical sentences ever had occasion
to say that the form was a precondition — every pair it was
ever shown had it.

### What the KL penalty is actually for

The standard fix is to penalise drift from the reference
policy, folding the KL term into the reward as InstructGPT
does. Swept across four strengths, with three seeds each, the
result is not the trade-off it is usually described as:

![[BETA CURVE IMAGE]]

At `beta = 0.20` the peak is **1.5045** — higher than either
looser setting ever reached — and it is **still there at step
300**, at 1.4953 with grammatical mass 0.9800. The loose
penalties do not buy a higher maximum in exchange for
instability. They simply destroy themselves: at 0.02 the
policy ends at **0.0025** with mass 0.0019, which is to say
it ends up emitting essentially nothing valid at all.

So the penalty is not a safety tax paid out of performance.
At the right strength it wins on both axes, because a policy
that runs off the distribution its reward model understands
does not earn a better score for it. It earns a score nobody
should believe.

### Removing the thing that got hacked

DPO drops the reward model entirely and trains on the
comparisons directly, treating the log-ratio to the reference
policy as an implicit reward. Same 3,000 pairs, same seeds,
same budget — and no sampling anywhere, so there is no
distribution for a proxy to drift off.

![[DPO TABLE IMAGE]]

DPO rises and stays up. It never reaches what well-tuned RLHF
reaches — **1.2820** against **1.4953** — but it also never
produces the 0.0025 that the worst RLHF run ends at.

That is the honest summary. **DPO is not better at finding
reward. It is better at not losing it**, because the failure
mode it removes is the one that needs a proxy to exist. What
it cannot do is exceed its pairs: every sentence it learns
from was written by the reference policy, so it can only
re-weight what that policy already said. RLHF samples, which
is how it can find sentences nobody showed it — and also how
it finds the ones nobody vetted.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The line worth noticing is that RLHF and DPO share a
gradient. Both reduce to

```
sum_i w_i * grad log pi(s_i)
```

and differ only in the weights: REINFORCE passes an
advantage, DPO passes `±beta(1 - sigma(h))`. Everything else
is bookkeeping. The expensive asymmetry is that RLHF must
*sample* from the policy every step and score those samples
with a second network, while DPO reads from a fixed file.

---

## Where this sits

The comparison-based loss is older than any of this: Bradley
and Terry published it in 1952, for ranking sports teams.
Christiano et al. brought it to deep reinforcement learning
in 2017, and InstructGPT made it the standard recipe for
language models in 2022 — reward model, PPO, KL penalty to a
frozen reference.

The failure in this article was named early. Amodei et al.
listed reward hacking among concrete AI-safety problems in
2016; Gao et al. measured reward-model over-optimisation at
scale in 2022 and found the same shape, a true-reward curve
that rises and then falls while the proxy keeps climbing.
Rafailov et al. published DPO in 2023, showing the RLHF
objective has a closed-form optimum that can be trained
directly from preferences.

What this article adds is not the phenomenon but the
yardstick. The curve Gao et al. had to approximate with a
much larger "gold" reward model is, in this universe,
arithmetic — which is why the article can say exactly how far
below baseline the hacked policy ends up, and show you the
sentence that did it.

The lineage in the series is direct. Part 7 built the policy
and the exact machinery for measuring a distribution over the
language. Part 9's discipline returns too: every policy
figure here is a mean over three seeds, because single runs
of a policy-gradient method are not evidence.

---

## What comes next

Part 11, **In-Context Learning and Prompting**, closes the
track with the strangest capability of the lot. Everything in
this article changed weights to change behaviour. In-context
learning changes behaviour with no gradient step at all —
the model reads examples in its prompt and does something it
was never trained to do. The question is what mechanism could
possibly support that, and whether the micro-language can be
made to exhibit it.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**rlhf.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/13-natural-language-processing/10-rlhf-and-dpo/rlhf.py)

Run it with:

```bash
pip install numpy
python rlhf.py
```

It needs only `numpy` and runs in about sixteen minutes.
Everything is from scratch: the Bradley-Terry reward head,
the weighted log-probability gradient that REINFORCE and DPO
both ride on, the KL-penalised policy update, and the exact
evaluation over all 31,740 sentences — hand-differentiated
and audited against central differences at startup (worst
error 4.51e-09 over 160 sampled gradients). The headline
insight worth pinning to the wall: **a policy optimised
against a learned reward model drove its true reward from
0.7689 up to 1.1985 and then down to 0.3442, below where it
started, while the reward model went on reporting success —
and the reward model was not even a bad one, scoring 0.6906
against the truth on and off its training distribution alike;
what it had never been shown was a non-sentence, so when the
policy learned to stack two adjectives and call a king's
servant a sailor, it scored that two full points above the
real text it was trained on**.

---

*This is Part 10 of the Natural Language Processing track in the Algorithms in Python series. The companion script `rlhf.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It reuses Part 1's synthetic micro-language ([word embeddings](https://medium.com/p/164f7f542a3c), [BPE tokenisation](https://medium.com/p/5e6a51341455), [sequence-to-sequence](https://medium.com/p/bbdf65112819), [BERT-style encoders](https://medium.com/p/a0ba729fe954), [GPT-style decoders](https://medium.com/p/6da33883e00d), [natural language inference](https://medium.com/p/d4f9c54be0d4), [natural language generation](https://medium.com/p/48a45ec5b396), [retrieval-augmented generation](https://medium.com/p/0686445e3d97), [fine-tuning and LoRA](https://medium.com/p/048d50f039aa)). Part 11 will look at In-Context Learning and Prompting.*
