# Knowledge Graphs — Where Symbols, Embeddings, and RAG Meet

### *Algorithms in Python --- Foundations, Part 9*

---

In Part 5 we walked **graphs** with BFS and DFS. In Part 8 we built
**trees** as graphs with the cycles taken out. Today's structure is the
opposite move: we put the cycles back in, but we add *semantics* to
every edge. That gives us a **knowledge graph**.

A knowledge graph is a graph where every edge has a **type**, every node
has a **type**, and the whole thing carries enough structure that a
machine can answer questions like *"which Polish-born physicists won
the Nobel Prize after 1900?"* by walking the graph rather than scanning
a table. The same structure powers Google's "People also ask" boxes,
Wikidata, the entity-linking step inside modern RAG systems, the
biomedical reasoning behind drug-discovery pipelines, and the agent
memory of any system that needs to remember *who did what to whom*.

It is also the bridge between two worlds the rest of this series will
spend a long time on. To the *symbolic* side, a knowledge graph is a
database of facts you can query with logic. To the *neural* side, every
entity and every relation can be **embedded** as a vector, and the
geometry of those vectors encodes the same facts in a form that
gradient descent can use. RAG, retrieval-augmented generation,
increasingly leans on both halves: walk the symbolic graph to find
relevant context, encode it as text, hand it to the LLM. The structure
in this article is what that walk actually traverses.

---

## What is a knowledge graph?

The atomic unit of a knowledge graph is the **triple**:

```
(subject, predicate, object)
```

Three things, written in that order. The subject and object are
**entities** — Marie Curie, Warsaw, Polonium, the Nobel Prize. The
predicate is a **relation** that connects them — `born_in`,
`discovered`, `won`. Each triple states one fact:

```
(Marie Curie, born_in, Warsaw)
(Marie Curie, discovered, Polonium)
(Marie Curie, won, Nobel Prize in Physics)
(Warsaw, located_in, Poland)
(Polonium, named_after, Poland)
```

That is the entire data model. Stack several billion of these and
you have **Wikidata**. Stack tens of billions and you have Google's
Knowledge Graph or the Knowledge Vault that backs it.

Drawn as a graph, the triples above look like:

```
         Polonium  -- named_after -->  Poland
            ^                            ^
            |                            |
        discovered                  located_in
            |                            |
        Marie Curie  -- born_in -->  Warsaw
            |
           won
            |
            v
   Nobel Prize in Physics
```

Every node is an entity. Every edge is *labelled* with the relation
it represents. That labelling is what separates a knowledge graph from
a plain graph in Part 5. In a plain graph, an edge says "these two
nodes are connected." In a knowledge graph, the edge says *how* they
are connected, and the algorithm walking the graph can use that to
answer typed questions.

The standard serialisation for triples is **RDF** (Resource Description
Framework), and the standard text format is **Turtle**, which writes
the same data as:

```turtle
:marie_curie  :born_in       :warsaw .
:marie_curie  :discovered    :polonium .
:marie_curie  :won           :nobel_physics .
:warsaw       :located_in    :poland .
:polonium     :named_after   :poland .
```

You will see RDF and Turtle if you ever touch Wikidata, DBpedia,
schema.org, or any biomedical ontology. They are the lingua franca of
the symbolic-AI side of the world.

---

## The triple store

A **triple store** is a database optimised for the triple data model.
The naive implementation is just a set of three-tuples:

```python
triples = {
    ("marie_curie", "born_in",    "warsaw"),
    ("marie_curie", "discovered", "polonium"),
    ("marie_curie", "won",        "nobel_physics"),
    ("warsaw",      "located_in", "poland"),
    ("polonium",    "named_after","poland"),
}
```

To query, you scan the set looking for triples that match a pattern
where some positions are bound and others are wildcards. *"What did
Marie Curie discover?"* is the pattern `(marie_curie, discovered, ?)`.
*"Where was every Nobel-winning physicist born?"* is a two-step pattern
that joins on a shared variable.

Production triple stores — Apache Jena, RDF4J, Virtuoso, AnzoGraph,
Stardog — keep multiple **indexes** so these scans are fast. The
canonical trick is to maintain six sorted indexes, one for each
permutation of (subject, predicate, object): SPO, SOP, PSO, POS, OSP,
OPS. Whichever positions are bound in a query, one of those indexes
puts the matching triples in a contiguous range. With the right index,
*"all triples with predicate `born_in`"* is a constant-time lookup
followed by a linear scan over the matches; without it, you scan the
whole store.

The companion script builds a tiny in-memory triple store and
implements pattern matching over it. Two queries:

```
QUERY 1 -- 1-hop: "What did Marie Curie discover?"
  (marie_curie, discovered, ?)
  ->  polonium
  ->  radium

QUERY 2 -- 2-hop: "Find all scientists born in cities located in Poland."
  (?scientist, born_in, ?city) AND (?city, located_in, poland)
  ->  marie_curie  born_in warsaw  (located_in poland)
```

The second query is what knowledge graphs really pay for. In a
relational database it is a `JOIN` you have to write by hand, with
explicit foreign keys. In a triple store the join is implicit in the
pattern: any variable that appears twice is a shared binding.
Three-hop, four-hop, ten-hop queries are written the same way. That uniform
treatment of arbitrarily long traversal is why knowledge graphs are
the right shape for relational reasoning at scale.

---

## Ontologies — the schema layer

The triples themselves are facts. To do useful reasoning you also need
to know what *kinds* of facts can exist: which entities are of what
type, and which relations connect which types. That schema is called an
**ontology**.

A lightweight ontology gives you class hierarchies and relation
signatures:

```turtle
:Person       rdfs:subClassOf  :Agent .
:Scientist    rdfs:subClassOf  :Person .
:City         rdfs:subClassOf  :Place .
:Country      rdfs:subClassOf  :Place .

:born_in      rdfs:domain      :Person ;
              rdfs:range       :Place .

:located_in   rdfs:domain      :Place ;
              rdfs:range       :Place .
```

This says a `Scientist` *is a* `Person`, `Person` *is an* `Agent`, and
the relation `born_in` connects a `Person` (subject) to a `Place`
(object). Now the query *"all scientists born in Poland"* can use the
hierarchy to also find people whose type is `Physicist` or `Chemist`
without listing them explicitly, because both are subclasses of
`Scientist`.

Heavyweight ontologies in **OWL** (Web Ontology Language) add
constraints — *every Person has exactly one date of birth*, *no Place
can be born_in another Place* — and a **reasoner** can derive new
triples from existing ones. *"If A is the parent of B and B is the
parent of C, then A is the grandparent of C"* is a rule the reasoner
can apply automatically every time you add a new `parent_of` triple.

Modern practice is increasingly lightweight. Wikidata uses a flexible,
mostly self-describing schema; Google's Knowledge Graph uses a custom
internal one; product knowledge graphs at Amazon and Pinterest sit
somewhere in between. The heavy OWL reasoning of the early Semantic
Web era turned out to be expensive in practice, and most teams now
prefer to encode reasoning in code or in graph queries rather than in
the ontology itself.

---

## Querying — SPARQL and the cousins

The standard query language for RDF triple stores is **SPARQL**. It
reads like SQL with triple patterns instead of tables:

```sparql
SELECT ?scientist ?city WHERE {
    ?scientist  :born_in     ?city .
    ?city       :located_in  :poland .
    ?scientist  rdf:type     :Scientist .
}
```

Three triple patterns, joined on the shared variables `?scientist` and
`?city`. The query engine searches the triple store for assignments
that satisfy all three patterns simultaneously. Add a `FILTER` clause
for *"after 1900"* and you have the Polish-born twentieth-century
scientists query.

Property graph databases (Neo4j, Memgraph, TigerGraph) use **Cypher**
or **Gremlin** and reach for a slightly different model — nodes and
edges with attached properties rather than pure triples — but the
fundamental capability is identical: traverse a labelled graph by
matching patterns against it.

The point is that knowledge graphs come with a *declarative* query
language built around traversal. You describe the shape you want; the
engine plans the joins and the index lookups. SQL learned to do this
for tables. SPARQL and Cypher do it for graphs.

---

## Entity embeddings — the neural side of the same data

Symbolic queries are powerful when the question is precise and the
data is clean. They struggle when the question is fuzzy ("entities
similar to Marie Curie") or when the graph is incomplete ("predict
what Marie Curie *might have* discovered if we did not record it").

The neural answer is to **embed** every entity and every relation as a
vector in some shared space, and design the embeddings so that the
geometry encodes the same facts the triples do. The simplest and most
famous model is **TransE** (Bordes et al., 2013):

```
For every true triple (h, r, t):    h + r ≈ t
```

Read that as: the embedding of the head entity *plus* the embedding of
the relation should equal the embedding of the tail entity. If
`marie_curie + born_in ≈ warsaw`, then asking *"where was Marie Curie
born?"* becomes a nearest-neighbour search: compute `marie_curie +
born_in`, find the closest entity vector, return it. A perfectly
trained TransE model would let you complete *missing* triples by the
same operation — the geometric analogue of symbolic reasoning.

Training is a margin loss against negatives. For each true triple you
sample a corrupted version (replace the tail with a random entity)
and push the true triple to have a smaller `||h + r - t||` than the
corrupted one:

```
loss = max(0, margin + ||h + r - t|| - ||h + r - t'||)
```

A few hundred SGD steps over a few hundred triples is enough to learn
useful structure. The companion script trains TransE on our toy
scientist KG and reports the geometric property:

```
After training, h + r should land near t. Closest entities to the
predicted tail vector for each query:

  query                          true tail   rank
  marie_curie + born_in          warsaw        1
  pierre_curie + born_in         paris         1
  alan_turing + born_in          london        1
  warsaw + located_in            poland        1
  marie_curie + discovered       polonium      2  (top1=radium)
  isaac_newton + formulated      calculus      2  (top1=law_of_gravitation)
```

The functional relations (every entity has exactly one birthplace,
every city has exactly one country) recover perfectly: rank 1 every
time. The non-functional ones (Curie discovered *both* polonium and
radium; Newton formulated *both* calculus and the law of gravitation)
land at rank 2 — both correct tails are in the top two, but the model
literally cannot put both at rank 1 because their predicted vectors
collide. This is a known structural limitation of TransE: it cannot
represent 1-to-many or many-to-many relations cleanly.

That limitation is exactly why the field moved past TransE.
**DistMult**, **ComplEx**, **RotatE**, and **ConvE** are increasingly
expressive successors that handle asymmetric and non-functional
relations (RotatE in particular models each relation as a rotation in
complex space, which lets it represent symmetry, antisymmetry, and
composition simultaneously).
Modern KG embedding libraries — PyKEEN, AmpliGraph, DGL-KE — implement
dozens of these and let you swap them in with a config flag. The
geometric intuition behind all of them is the same as TransE: shape
the embedding space so that the algebra of vectors mirrors the algebra
of relations.

---

## Knowledge graphs in RAG

Retrieval-augmented generation is the dominant pattern for grounding
LLM responses in a corpus you control. The textbook RAG pipeline is:
embed the user's question, retrieve the top-*k* most similar document
chunks by cosine similarity, paste them into the LLM's context.

That works well when the answer lives in a single passage of text. It
works badly when the answer requires *combining* facts from several
sources — *"which Polish-born scientists discovered radioactive
elements named after countries?"* is a question whose answer is not in
any single Wikipedia paragraph. It is in the **join** of several
paragraphs, and pure vector retrieval has no concept of join.

**Knowledge-graph RAG** (or **GraphRAG**, in the Microsoft paper that
gave the technique its current name) closes that gap. The pipeline:

1. **Entity-link** the user's question. *"Polish-born scientists who
   discovered named-after-countries elements"* contains the entities
   `:poland`, `:scientist`, `:element`, and the relation pattern
   `(?elem, named_after, ?country)`.
2. **Walk** the knowledge graph to find the relevant subgraph — the
   scientists, their birthplaces, the elements they discovered, the
   naming chain.
3. **Linearise** the subgraph into a sequence of triples or a
   short narrative.
4. **Hand** that linearised context to the LLM along with the original
   question.

The LLM gets *grounded* relational structure rather than a bag of
loosely related text. The same trick scales to enterprise knowledge
bases (every product, customer, contract, and supplier as nodes;
every interaction as edges), to scientific reasoning (every drug,
target, pathway, disease), and to agent memory (every user fact, every
preference, every prior interaction).

The neural and symbolic halves we just covered show up here together:
the **embedding** lets you do fuzzy entity linking and ranking; the
**graph traversal** lets you find paths the embedding alone could not.
Production KG-RAG systems use both.

---

## Big-O and scale

[[BIG-O TABLE IMAGE]]

A few concrete sizes to anchor the scale. **Wikidata** has about 100
million entities and 14 billion triples. **Google's Knowledge Graph**
sits in the hundreds of billions of facts, depending on what you
count. **DBpedia** is in the low billions. Biomedical KGs like
**UMLS** are in the millions of concepts and tens of millions of
relations.

The triple-store complexities in the table assume the canonical
indexed implementation. A single triple-pattern lookup is O(log n)
to find the start of the matching range plus O(k) to scan the matches.
A k-hop pattern is the join of k such lookups; query planners reorder
the patterns to do the most selective one first, exactly as a
relational planner would.

Embedding training is O(epochs × triples × dim). For the toy KG in
the companion script with 35 triples, dim 8, and 200 epochs, that is
~50,000 vector operations — under a second.

---

## Real-world ML and AI connections

Knowledge graphs sit underneath more of the AI stack than they get
credit for.

**Wikidata as a free training resource.** Wikidata's CC0-licensed
triples are used for entity-linking benchmarks, multilingual
knowledge-completion experiments, and as a grounding source for LLM
fine-tuning. Almost every entity-linking dataset in academic NLP
traces back to Wikidata or DBpedia.

**Biomedical knowledge graphs.** Hetionet, PrimeKG, and the BioGRID
network encode drugs, genes, proteins, diseases, and pathways as
typed entities and relations. Drug-discovery teams at pharma
companies query these graphs to find candidate repurposing targets —
*"drugs whose targets share a pathway with this disease's known
genes"* — and embed the graphs to predict missing drug-disease
indications. The graph is doing the relational reasoning; the
embedding is doing the prediction.

**Product graphs at Amazon, Pinterest, and Etsy.** Every product is a
node; every "bought together," "viewed together," "is variant of," or
"is accessory of" relation is a typed edge; categories and brands are
nodes too. Embedding the resulting graph gives you product
recommendations that respect categorical structure rather than
treating every item as an independent vector. The same idea powers
Pinterest's PinSage and Uber's geo-product graphs.

**Agent memory in LLM applications.** When an agent needs to remember
*"the user's spouse is named Alex and prefers vegetarian restaurants
in Bristol,"* the natural representation is a small triple store —
`(user, spouse, alex)`, `(alex, prefers, vegetarian_food)`,
`(alex, lives_in, bristol)`. Every modern agent framework that
supports persistent memory either ships with a triple store or uses
one under the hood (LangChain's `ConversationKGMemory`, Letta's
memory modules, Microsoft's Semantic Kernel memory plugins).

**Schema.org and the Web.** The structured snippets you see in Google
search results — recipes, events, product cards, FAQ accordions —
come from schema.org markup, which is a knowledge-graph vocabulary
embedded in HTML. Every page that uses it is contributing typed
triples back to the public graph.

**Cypher and Gremlin in fraud detection.** The pattern *"three
accounts that share an address, transferred money to the same fourth
account, on the same day"* is a graph traversal. Banks and crypto
exchanges run Neo4j or TigerGraph behind their fraud-detection
systems precisely because that pattern is impossible to express
cleanly in SQL.

**Prompt-graph debugging in LLM evaluation.** Modern eval suites
(LangSmith, Weights & Biases Weave, OpenAI Evals) record every prompt,
completion, tool call, and trace as nodes in a small operational KG
keyed on session and trace IDs. The "trace tree" is a knowledge graph
in everything but name.

---

## When NOT to use a knowledge graph

KGs are powerful but they are not free.

**When the data is naturally tabular.** If your data lives cleanly in
rows and columns and your queries are aggregations over those rows,
use a relational database. Postgres has been crushed for forty years
by people trying to do everything in JOIN clauses and is still faster
than any KG store for that workload.

**When you have no schema and no intention of one.** A KG without an
ontology is just a bag of triples, and a bag of triples without
discipline degrades into a dataset that nobody can query reliably.
If you cannot agree on what the predicates mean, you do not yet have
a KG; you have a CSV in disguise.

**When latency matters more than expressiveness.** Vector retrieval
plus an LLM is faster than a graph walk plus an LLM, and for many
question-answering workloads it is good enough. Reach for KG-RAG when
the answer needs structured joins, not when a single passage will do.

**When the graph is changing constantly.** Triple stores are mostly
optimised for read-heavy workloads. A KG that changes hundreds of
times per second per entity is a worse fit than a graph database
designed for write throughput, which in turn is a worse fit than a
streaming event store. Match the structure to the access pattern.

**When you cannot afford the entity-resolution problem.** "Marie
Curie" and "Maria Skłodowska-Curie" and "M. Curie (1867-1934)" are
the same person, and your KG will quietly degrade if you do not
unify them. Entity resolution is its own substantial engineering
problem; budget for it before committing to a KG.

---

## What comes next

Nine foundations down, three to go. Part 10 is **Probabilistic Data
Structures** — Bloom filters, Count-Min Sketch, HyperLogLog. The
shapes that trade exactness for sublinear memory and power streaming
deduplication, feature hashing, and rate limiting at scale.

Then Part 11, **Sparse Matrices** — CSR, CSC, COO and why most of
your data being zero is a feature, not a bug. And Part 12, **Vector
Indexes (ANN)** — HNSW, IVF, PQ, LSH; the structures inside FAISS,
Pinecone, Chroma, and every modern RAG retriever.

After Part 12 the foundations are complete and we leave the
data-structure layer for good. The first algorithm article picks up
linear regression and starts a return visit to the matrices and
vectors of Parts 1 and 2.

---

## The complete code

The full script is on GitHub — grab it and run it:

[**knowledge_graphs.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/09-knowledge-graphs/knowledge_graphs.py)

Run it with:

```bash
python knowledge_graphs.py
```

It finishes in under a second on a laptop. The companion script
builds a tiny KG of scientists, runs 1-hop and 2-hop pattern queries
against it, and trains TransE embeddings on the same data — showing
that after 200 epochs, *Marie Curie + born_in* lands closer to
*Warsaw* than to any of the other 28 entities in the graph. That
is the whole geometric punchline of KG embeddings, demonstrated
against an actual training loop in pure Python.

---

*This is Part 9 of the Algorithms in Python series, Foundations track. The companion script `knowledge_graphs.py` is in the [series repository](https://github.com/grahamroy/algorithms-in-python). [Part 8](https://medium.com/@grahamjroy/trees-hierarchical-structure-for-decisions-search-and-database-indexes-64767b20394f) covered trees. Part 10 will look at probabilistic data structures — Bloom filters, Count-Min Sketch, HyperLogLog, and the streaming workloads they enable.*
