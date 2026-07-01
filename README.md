# Kapi

**Fast, local, zero-setup RAG with no embedding model.** Plug in any LLM and go.

Kapi is a Python RAG library whose retriever is **lexical, not neural** — BM25 over words
**plus** char-trigrams **plus** title boosts, fused into one query. No embedding model to
download, no vector DB to run, no GPU. The LLM you plug in is used only where lexical
search is genuinely weak: to **contextualize chunks at index time** and **expand queries**
at search time. Everything works with no LLM too (pure lexical retrieval).

```bash
pip install kapi        # core: tantivy + snowballstemmer + httpx. No models, ever.
```

```python
from kapi import Kapi
from kapi.llm import OpenAICompatLLM

# Local + low cost: Ollama via its OpenAI-compatible endpoint (no extra install needed)
llm = OpenAICompatLLM(base_url="http://localhost:11434/v1/", api_key="ollama", model="llama3.2")

rag = Kapi(llm=llm)                 # quality preset: contextual indexing + query expansion ON
rag.add("docs/")                    # ingest a dir / file / glob; chunks contextualized offline, then indexed
print(rag.query("How do refunds work?").answer)   # one online call -> grounded answer with [n] citations
```

## Why no embeddings?

Per **BEIR** (Thakur et al., NeurIPS 2021), BM25 is a famously strong *zero-shot* baseline
that dense retrievers frequently **fail to beat out-of-domain** — especially on
keyword/technical/argument queries. Kapi leans into that and fixes BM25's one real
weakness (vocabulary mismatch) with the LLM you already have, instead of a second model:

| BM25 weakness | Kapi's embedding-free fix |
|---|---|
| Lost context when chunked | **Contextual indexing** — LLM writes a 1–2 sentence blurb per chunk, prepended before indexing (offline, cached). ~−49% retrieval failures (Anthropic, 2024). |
| Vocabulary mismatch (synonyms) | **Query expansion** — query2doc / CoT keywords (one online call). Up to +15 nDCG@10 (Wang et al., 2023). |
| Typos / morphology | **Char-trigram signal** fused with the word signal. |
| Crude top-k ordering | **Multi-signal + RRF**, title boosts. |

## Plug in any LLM

```python
from kapi.llm import OpenAICompatLLM, CallableLLM

# Any OpenAI-compatible endpoint: OpenAI, Ollama, vLLM, llama.cpp, LM Studio, Together, Groq...
OpenAICompatLLM(base_url="https://api.openai.com/v1", api_key=KEY, model="gpt-4o-mini")

# ...or wrap any function
rag = Kapi(llm=CallableLLM(lambda prompt: my_model(prompt)))

# ...or no LLM at all — pure lexical retrieval still works
rag = Kapi()
hits = rag.search("refund policy")          # ranked chunks, sub-10ms class
```

## Speed vs quality, your call

```python
Kapi(llm=llm)                      # preset="quality" (default): contextual + expansion ON
Kapi(llm=llm, preset="compiled")   # Compiled Retrieval: full offline compiler + CSC (see below)
Kapi(llm=llm, preset="fast")       # pure-lexical retrieval; LLM only writes the final answer
Kapi()                             # no LLM: retrieval only
```
All LLM cost is **offline** (contextual indexing — one-time, cached, free with a local
model) or a **single online call** (query expansion). A cost guard refuses accidental
large paid-model runs; it's a no-op for local models.

## Compiled Retrieval (`preset="compiled"`)

> **Retrieval intelligence is a *compile-time* problem, not a *serve-time* problem.**

Dense embeddings run a model on *every query*; query-time reasoning runs an LLM on *every
query*. Kapi's `compiled` preset moves **all** of it to **index time**: the LLM runs **once
per chunk, cached forever**, and *compiles* each chunk into its full *queryable + inferential
closure* — a purely **lexical** representation. Query time stays what BM25 already is: **~1 ms,
no model, no GPU, no vector DB, fully explainable.** This is the **AOT compiler vs. JIT
interpreter** distinction, applied to retrieval.

```python
rag = Kapi(llm=llm, preset="compiled", path="./idx")
rag.compile("docs/")                 # offline: blurb + doc2query + propositions + reasoning, cached
print(rag.query("does this scale to a billion rows?").answer)   # ~1ms lexical retrieval

rag = Kapi.open("./idx")             # reopen with NO llm — the compiled index still serves
```

One cached offline pass per chunk emits an enrichment **bundle**, all of it plain text that
lands in the lexical index (never in the cited text):

| Pillar | What the compiler adds | Prior art |
|---|---|---|
| **blurb** | a chunk-specific context sentence | Anthropic Contextual Retrieval, 2024 |
| **questions** | the queries this chunk answers | doc2query / docTTTTTquery |
| **propositions** | atomic, decontextualized facts (rare entities matchable) | Dense X, EMNLP 2024 |
| **reasoning** | second-order facts & multi-hop bridges *not lexically present* | the BRIGHT-winning signal, precomputed |

**CSC — Consensus Sparse Compilation** (the novel core). The compiler is sampled `k` times;
a term's weight is its **agreement across samples** — a training-free, label-free learned-sparse
weighter that doubles as a self-consistency (doc2query--) filter: hallucinated terms appear in
one sample and are dropped; entailed terms recur and are promoted. **Literal anchoring** keeps
every source-literal term (IDs, error codes, proper nouns) at a floor weight, so exact-match
retrieval is structurally protected. The result is a second sparse "leg" that fuses with plain
lexical BM25 — hybrid's two-error-profile win, **with no embedding model and no vector DB.**

**Adaptive router** (the only query-time LLM use, and it's gated). The first lexical pass is
~1 ms and $0; a cheap confidence signal — no hits, low recall, or an ambiguous top-vs-2nd
margin — decides whether to spend one LLM call escalating (query expansion + re-search). Short
queries are treated as precise and never escalated, dodging the expansion *precision trap*.
Cheap by default, accurate on demand — inspect the decision on `rag.last_route`.

## Persistent & incremental

```python
rag = Kapi(llm=llm, path="./index")   # on-disk index
rag.add("docs/")
rag.close()

rag = Kapi.open("./index", llm=llm)   # reopen later
rag.sync("docs/")                     # re-index only changed files; drop deleted ones
```

## Inspect results & citations

```python
res = rag.query("How do refunds work?", k=8)
res.answer                            # str | None (None if no LLM)
res.citations                         # [Citation(marker="[1]", source="docs/refunds.md", ...)]
for h in res.hits:                    # ranked retrieved chunks
    print(h.score, h.source, h.text[:120])

for token in rag.query_stream("..."): # stream the answer
    print(token, end="")
```

## Engines

Kapi ships a pluggable engine layer. The default needs no setup; alternatives are one
keyword away.

| Engine | `engine=` | Notes |
|---|---|---|
| **Tantivy** (default) | `"tantivy"` | Rust, Lucene-class speed, persistent + incremental, multi-signal in one query. pip wheel, no server. |
| **SQLite FTS5** | `"sqlite"` | Zero extra dependency (stdlib `sqlite3`). Word + trigram tables fused with RRF. |
| **bm25s** | `"bm25s"` | In-memory, fastest for fixed corpora. `pip install kapi[bm25s]`. |

## Evaluate it (prove it's good enough)

```python
from kapi.eval import evaluate_run                 # pure-Python nDCG@k / Recall@k / MRR
from kapi.eval import run_beir, run_bright          # needs kapi[eval]

report = run_beir(lambda: Kapi(engine="tantivy"), "scifact")
print(report)                                       # kapi vs published BM25, side by side

# BRIGHT — the reasoning-intensive benchmark where embedding-free wins and off-the-shelf
# dense collapses (the #1 MTEB model scores 18.3). The hero board for Compiled Retrieval.
report = run_bright(lambda llm=llm: Kapi(llm=llm, preset="compiled"), "biology")
print(report)
```

## Command line

```bash
kapi compile ./docs --index ./idx --base-url http://localhost:11434/v1/ --model llama3.2
kapi query  "how do refunds work?" --index ./idx
kapi stats  --index ./idx
kapi tco    --queries-per-month 5000000 --months 36   # KAPI vs dense+vectorDB cost model
```
`compile` is `add` named for the mental model. With no `--base-url` it builds a pure-lexical
index; retrieval is always embedding-free.

## Compile once, serve anywhere (air-gapped)

The expensive step (LLM compilation) runs **once**; the serving index is a plain lexical
artifact — no LLM, no network, no vector DB. Bundle it and ship it to an on-prem / air-gapped
box, where it opens and serves model-free:

```bash
kapi export --index ./idx --out ship.kapi.tgz      # portable bundle (drops the LLM cache)
kapi import ship.kapi.tgz --index ./served         # unpack on the target machine
kapi query  "how do refunds work?" --index ./served   # $0, ~1 ms, no model
```

```python
rag.export_bundle("ship.kapi.tgz")                 # or, in code
served = Kapi.import_bundle("ship.kapi.tgz", "./served")   # opens with no LLM
```

Or run the **hosted compilation service** — clients POST documents, get back a serving bundle
(the smart compute stays server-side; no embedding model ever crosses the wire):

```bash
kapi serve --base-url http://localhost:11434/v1/ --model llama3.2   # POST /compile, GET /bundle/<job>
```

## Drop into LangChain / LlamaIndex

One line to try KAPI as the retriever in an existing stack — no embeddings, no vector DB:

```python
from kapi.integrations import to_langchain_retriever, to_llamaindex_retriever
lc = to_langchain_retriever(rag, k=5)      # a LangChain BaseRetriever
li = to_llamaindex_retriever(rag, k=5)     # a LlamaIndex BaseRetriever
```

## Install extras

```bash
pip install kapi                # core (no models, local LLM path works)
pip install kapi[openai]        # openai SDK + tiktoken (exact token counts)
pip install kapi[bm25s]         # in-memory bm25s engine
pip install kapi[eval]          # ranx / pytrec_eval / BEIR / RAGAS
pip install kapi[pdf,html]      # PDF text + fast HTML loaders
```

## How it works

```
add(source)     → load → chunk → contextualize (offline LLM, cached) → index (BM25)
compile(source) → load → chunk → compile bundle + CSC weights (offline LLM, cached, k-sampled)
                → index Leg A (enriched BM25) + Leg B (consensus-weighted sparse)
query(q)        → [expand] → Leg A (BM25 + RRF) ⊕ Leg B (sparse dot product) → top-k → generate
```

Requires Python ≥ 3.10. No embedding model. No server. No GPU.

## License

MIT
