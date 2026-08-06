# Graph Neural Networks — The Graph Decides Who May Look at Whom

### *Algorithms in Python --- Deep Learning Architectures, Part 6*

---

Every architecture in this section so far has assumed the data
had a *shape*. Part 2 assumed a grid of pixels. Parts 3 through
5 assumed a line of timesteps. But molecules, social networks,
citation webs, protein interactions, and road maps assume
neither — they arrive as **nodes and edges**, and the only
geometry they have is *who is connected to whom*.

The graph neural network's answer is **message passing**. Each
layer, every node collects its neighbours' feature vectors,
averages them, and passes the result through a small learned
map. The cleanest version is the GCN layer (Kipf & Welling,
2017):

```
H' = relu( Â H W ),      Â = D^(-1/2) (A + I) D^(-1/2)
```

`A` is the adjacency matrix, the identity adds a self-loop
(you are your own neighbour), and the degree normalisation
stops popular nodes from shouting. One layer mixes one-hop
neighbourhoods; `L` layers, `L` hops — weight sharing yet
again, this time across *nodes*: the same `W` serves every
neighbourhood in the graph, exactly as Part 2's kernel served
every pixel.

And that parallel is not a metaphor. It is the section's
punchline, and this article earns it with measurements: **a CNN
is a GNN on the grid graph, and a transformer is a GNN on the
complete graph** — Part 4's attention is just message passing
where everyone is everyone's neighbour and the weights are
learned per pair. The graph decides who may look at whom.

---

## The stage: weak features, strong structure

A **stochastic block model**: 360 nodes in 3 hidden
communities, edges forming with probability 0.06 inside a
community and 0.02 across — 2,144 edges that *mostly* respect
the hidden grouping. Node features are the community signal
drowned in noise, deliberately weak. Fifteen nodes are
labelled. Transductive, five labels per class — Section 10's
setting, deliberately: the GCN is the neural descendant of that
track's label propagation, learning *what* to flood where the
old algorithm could only flood labels. Both hand-written
backward passes are audited at startup (GCN 2.46e-10, GAT
2.81e-10).

---

## A worked example: the edges, the blur, the defeat

### What the edges know

```
DEMO 1 --- What the edges know
    MLP, features only          :  58.3%
    GCN, features + edges       :  85.8%
    GCN, edges SHUFFLED         :  42.3%
```

The edges are worth 27 points: message passing turns "my
features are noisy" into "my *neighbourhood's average* is not"
— the same variance-reduction that made Part 2's shared kernel
strong, played on an irregular board. And the third row is the
sharp one: shuffle the edges (same degree structure, same
features, wrong connections) and the GCN lands *below* the
no-edge baseline. Wrong structure is not absence of signal — it
is poison, faithfully propagated.

### Over-smoothing: the depth lesson, inverted

Message passing is a random-walk step, and random walks forget
where they started. First the mechanism in isolation:
propagate the raw features `L` hops with no training at all,
then fit a small classifier on what remains:

```
DEMO 2 --- Over-smoothing
    hops L :   0     1     2     4     8    16    32
    acc    : 58%   71%   81%   87%   80%   33%   33%
```

A clean rise — each hop denoises — then a collapse to exactly
chance: past the graph's mixing time, every node's summary
converges to the same stationary blur, and *there is nothing
left to classify*. Training does not save the plain
architecture. Part 5's fix does:

```
    depth    plain GCN    + residuals
       2        85.8%        85.8%
       4        85.2%        87.0%
       8        33.3%        87.0%
      16        33.3%        81.7%
```

Eight plain layers score 33.3% — chance, reached by honest
depth. Part 1 taught that depth *composes*; this architecture
is the honest counterexample where depth **erases**, because
its per-layer operation is a contraction toward the average.
The residual stream keeps each node's own signal alive
alongside the neighbourhood blur — and 87% survives sixteen
layers. Same lesson as Part 5, second payoff: let the network
learn corrections, not replacements.

### Attention's cousin, and an honest defeat

The GAT (Veličković et al., 2018) replaces the GCN's fixed
degree-normalised average with **learned** per-edge weights —
scores computed from each edge's endpoint features, softmaxed
over the neighbourhood. It is precisely Part 4's attention,
masked to the graph. Strictly more flexible. Measured:

```
DEMO 3 --- GCN vs GAT, across label budgets
    labels/class    GCN      GAT
         5         85.8%    44.9%
        10         86.7%    63.6%
        20         85.7%    62.7%
```

The GCN wins every row, and there is no trick in it: attention
weights computed from noise-drowned features are noise with
extra parameters, and at these label counts the extra
parameters simply overfit. This is Part 2's rhyme in its third
verse — dense lost to locally-connected lost to shared then;
learned edge weights lose to uniform averaging now — **the less
data you have, the more your architecture should already
know**. GAT earns its keep on feature-rich graphs (the citation
networks it was born on have thousands of informative
dimensions per node, plus heavy dropout), where an edge *can*
be judged. Here, honestly, it cannot.

---

## Big-O and complexity

![[BIG-O TABLE IMAGE]]

The costs that matter: a message-passing layer is `O(E · d)` —
one message per edge — which for sparse real-world graphs is
dramatically cheaper than attention's all-pairs `O(n² · d)`;
the companion script's dense matrices are a small-n
convenience, not the method. Parameters are `O(L · d²)`,
independent of graph size, the section's oldest refrain. The
practical wrinkle that named an architecture: mini-batching a
graph is awkward because neighbourhoods overlap —
**GraphSAGE**'s move is to *sample* a fixed number of
neighbours per node, capping the exploding receptive field and
making web-scale graphs (billions of edges, Pinterest-sized)
trainable a batch at a time.

---

## Where the graphs are

Molecules are graphs, and GNN property prediction is standard
equipment in drug discovery pipelines. Recommenders are
bipartite graphs of people and things — PinSage served a
billion-node version of this article. Chips, road networks,
knowledge bases, particle showers. And the framing runs deeper
than any application: message passing is the *general case* of
this section — set the graph to a grid and recover the CNN, to
a line and recover (roughly) the RNN's locality, to the
complete graph and recover the transformer. The last five
articles were special graphs all along. One honest caveat
closes the tour: everything here leaned on *homophily* —
neighbours tending to share labels. Graphs where opposites
attract (fraud rings, protein interactions) invert the
assumption, and vanilla message passing struggles exactly as
much as the assumption is false. Priors, as ever, only pay
when true.

---

## What comes next

Part 7, **Generative Adversarial Networks**, pivots the
section from architectures that *classify* to architectures
that *create* — and it starts with the strangest training
signal in deep learning: two networks locked in a game, a
forger and a detective, each the other's loss function.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**gnn.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/11-deep-learning-architectures/06-graph-neural-networks/gnn.py)

Run it with:

```bash
pip install numpy
python gnn.py
```

It needs only `numpy` and runs in under a minute. Everything is
from scratch: the stochastic block model, the normalised
adjacency, GCN and GAT with hand-derived gradients (masked
softmax Jacobian included), the residual variant, and the
startup audits. The headline insight worth pinning to the
wall: **message passing is weight sharing across
neighbourhoods — H' = relu(ÂHW) — and it makes the edges
themselves the model: worth +27 points when right (58.3% →
85.8%), poison when wrong (42.3% shuffled), and self-erasing
when over-applied (chance at 8 plain layers, because averaging
is a random walk and walks forget — until residuals keep 87%
alive at sixteen); learned edge attention loses honestly to
the uniform prior at every label budget here, and the family
tree collapses to one sentence: a CNN is a GNN on the grid, a
transformer is a GNN on the complete graph, and the graph
decides who may look at whom**.

---

*This is Part 6 of the Deep Learning Architectures track in the Algorithms in Python series. The companion script `gnn.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). It generalises [Part 2](https://medium.com/p/6f860b54044f)'s weight sharing and [Part 4](https://medium.com/p/a27e38010fd8)'s attention to arbitrary structure, reuses [Part 5](https://medium.com/p/ceced896de61)'s residual lesson, and is the neural descendant of Section 10's [label propagation](https://medium.com/p/56d8ae72db5a). Part 7 will look at Generative Adversarial Networks.*
