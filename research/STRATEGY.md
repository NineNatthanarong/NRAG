# KAPI → Unbeatable: Strategy & Research Brief

*Author: research + synthesis pass, June 2026. Status: opinionated strategy, not gospel. Every external number is cited in the bibliography (§11). Items past ~mid-2025 are flagged `[fresh]` and should be re-verified before you quote them publicly.*

---

## 0. The one idea (read this even if you read nothing else)

> **Retrieval intelligence is a *compile-time* problem, not a *serve-time* problem.**

Everyone else puts the smart, expensive compute on the **hot path**:

- **Dense embeddings** = a model runs on *every query* (and every document) → JIT.
- **Query-time reasoning** (the thing that actually wins hard benchmarks today) = an LLM reasons on *every query* → an interpreter calling out to a 70B model per keystroke.

KAPI's bet: move **all** of it to **index time**. An LLM runs **once per document, cached forever**, and "compiles" each document into an enriched, purely *lexical* representation — its full *queryable + inferential closure*. Query time then stays what BM25 already is: **~1 ms, no model, no GPU, no vector DB, fully explainable.**

This is the **AOT compiler vs. JIT interpreter** distinction, applied to retrieval. It is the single framing that makes KAPI (a) win the benchmarks that matter, (b) irresistible to developers, and (c) a defensible business. The rest of this document is the evidence and the build plan.

Working name for the paradigm: **Compiled Retrieval**. Working name for the novel core algorithm: **Consensus Sparse Compilation (CSC)** — defined in §5.

---

## 1. Honest assessment of KAPI today

**What's genuinely good** (you undersell this):

- Cost-tiered, ablated benchmarking with an *honest headline* that admits where dense still wins. This is rarer and more credible than 90% of retrieval repos.
- Real findings, not vibes: you empirically rediscovered that **query expansion hurts precision** on precise retrieval, that **ensemble reranking backfires** (shuffling destroys the BM25 prior), and that **title-boost ×2.5 hurts** on abstract corpora. Those are the EACL-2024 "when expansion fails" result and the "lost in the middle" order-sensitivity result — found independently, in your own harness.
- Clean architecture: pluggable engines (Tantivy / SQLite-FTS5 / bm25s), span-exact structure-aware chunking, offline contextual indexing with a content-hash cache + cost guard, RRF fusion, graceful degradation to pure lexical. ~3.3k lines, no bloat.
- The positioning instinct — "no embedding model, no vector DB, no GPU" — is *exactly* aligned with where the market is moving (§2.6).

**The real problem (you diagnosed it correctly).** The *methodology* is an assembly of known parts:

| KAPI feature | Prior art it is |
|---|---|
| Contextual blurbs at index time | Anthropic Contextual Retrieval (Sep 2024) |
| query2doc / CoT keyword expansion | Wang et al. EMNLP 2023; Jagerman et al. 2023 |
| doc2query (in your benchmark) | Nogueira & Lin 2019 |
| BM25 + char-trigram + title, RRF | Cormack 2009; standard hybrid lexical |

There is no mechanism that is *yours*. That's the gap between "a good library someone can reproduce in a weekend" and "a system with a moat." This document closes that gap with **one new core mechanism** plus a paradigm that reframes the whole category in your favour.

---

## 2. The landscape, 2024–2026 (what the research actually says)

I ran the literature in four clusters. Here is the compressed map, with the conclusions that drive the strategy. Full citations in §11.

### 2.1 Learned-sparse retrieval — and why "just point an LLM at it" has *failed* so far

SPLADE-class models (SPLADE-v3, and the LLM-backbone Mistral-SPLADE / CSPLADE) beat BM25 by ~4 nDCG@10 on BEIR and edge out dense — **but every one of them is trained / fine-tuned.** The genuinely *training-free* attempts are a cautionary tale you must internalize:

- **BM42 (Qdrant, Jul 2024)** tried exactly the seductive idea — read attention weights from an off-the-shelf transformer as term importance, no training. It was publicly **retracted within days**: Reimers (Cohere) and Bergum (Vespa) reran it and it lost to well-tuned BM25 across finance/biomedical/Wikipedia. Qdrant edited the post to "consider BM42 as an experimental approach." The community headline was literally *"Please don't trust us."*
- **PROSPER (Oct 2025)** `[fresh]` got an LLM to emit SPLADE-style term:weights — but only by adding a **trained** "literal residual network," because they found LLMs *systematically hallucinate expansion terms and under-weight critical literal terms* (brands, IDs, model numbers). They also found **no scaling law** — a 7B LLM was no better than 1.5B at this.

**The lesson that shapes KAPI's mechanism:** an LLM used naively as a term-weigher is a known dead end. The failure mode is precise and fixable: it drops literal terms and hallucinates. KAPI must (a) **anchor on a literal lexical floor by rule** and (b) get its weights from **agreement, not from a single raw generation**. See §5.

**The free lunch that *is* real:** the *query* side can be model-free at tiny cost. Nardini et al. (SIGIR 2025) show dropping the query encoder from SPLADE-v3 costs only −2.4 MRR@10 / −4.7 nDCG@10; Geng et al. (OpenSearch, Nov 2024) use **plain IDF as the query weight** and stay competitive. This validates KAPI's whole "keep the query side dumb and lexical" instinct — *the document side is where the value is, and the document side is index-time, which is where a cached LLM is cheap.*

### 2.2 Index-time enrichment — the proven, KAPI-shaped levers

- **Anthropic Contextual Retrieval (Sep 2024):** prepend an LLM-written, *chunk-specific* context blurb before indexing. Contextual Embeddings −35% retrieval failures; **Contextual *BM25* + embeddings −49%**; +reranker −67%. Crucially for you: a large share of the lift is the *BM25* leg, and *generic* summaries gave "very limited" gains — it has to be chunk-specific. Cost ≈ $1.02 / 1M doc tokens with prompt caching. You already ship this.
- **doc2query → docTTTTTquery:** generate the queries a doc answers, append before indexing. **MS MARCO MRR@10 0.184 → 0.277 (+50% relative)** from a pure index-time text augmentation. The model is a *generator*, not a trained retriever — so a general LLM can do it.
- **doc2query-- ("When Less is More," ECIR 2023):** generators hallucinate; **filter** the generated queries with a relevance check → **+16% effectiveness, −23% query latency, −33% index size, simultaneously.** This is the single most important "do it right" result for index-time generation.
- **Dense X Retrieval / propositions (EMNLP 2024):** index *atomic, decontextualized propositions* instead of raw chunks → **+22–35% relative Recall@5** for unsupervised retrievers, with the **largest gains on rare-entity / long-tail / cross-domain** queries. Model-agnostic → helps a lexical index, and directly attacks BM25's context-loss weakness by *splitting* rather than *augmenting*.
- **RAPTOR (ICLR 2024):** recursive LLM summary tree; the *summary nodes are text* and can be dropped into a lexical index for global/multi-hop queries. (Late chunking is embedding-only — not applicable. GraphRAG/HippoRAG give multi-hop but carry heavy index cost and a non-lexical query step.)

**Takeaway:** there is a whole family of *index-time, text-producing, model-as-generator* techniques with verified gains, each of which lands in a plain inverted index. Nobody has unified them into one "compiler" with a shared self-consistency filter and a learned-offline weighting. That's the opening.

### 2.3 ★ Reasoning-intensive retrieval — the benchmark where embedding-free *wins*

This is the most important cluster and the hero of the strategy.

- **BRIGHT (ICLR 2025)** is the first benchmark built so that surface/semantic similarity is *insufficient* — relevance requires multi-step reasoning. The results are a gift to KAPI:
  - The **#1 MTEB dense model (SFR-Embedding-Mistral, 59.0 on MTEB) scores 18.3 on BRIGHT.** High embedding-benchmark rank *does not transfer* to reasoning retrieval.
  - **BM25 with GPT-4 chain-of-thought-rewritten queries jumps from ~14.3 to ~27 nDCG@10 — beating the best off-the-shelf dense model.** The lever that wins is *reasoning*, and it helps the *lexical* model most.
- **LATTICE (Oct 2025)** `[fresh]` is the existence proof: build a semantic tree from LLM document summaries **offline**, then let an LLM traverse it at query time. It hits **46.7 nDCG@10 on BRIGHT with a single off-the-shelf LLM and *no embedding model in the search loop*** — matching the best *fine-tuned* ensembles. Fused with cheap BM25+dense (LATTICE++) → 49.1, i.e. **lexical fusion adds on top** (validates RRF-of-cheap-signals).
- The trained dense-reasoning frontier (ReasonIR 36.9 w/ rerank; DIVER 45.8; BGE-Reasoner 45.2) is strong but **expensive and not embedding-free** — they are off the cost-quality frontier KAPI competes on.

**Honest reading:** BM25+reasoning (~27) does *not* beat the best fine-tuned reasoning systems (~46). What it beats is **off-the-shelf / zero-shot dense embeddings (~18–24)**. The top of the leaderboard is *reasoning-driven*, and **LATTICE proves you can be there without an embedding model.** KAPI's move is to take LATTICE's "reasoning beats embeddings" insight but **push the reasoning to compile time** so query time stays 1 ms (LATTICE still pays an LLM traversal per query).

### 2.4 Query-side expansion — and the precision trap you already found

HyDE, query2doc, LameR, GRF/MILL all bridge the query↔doc vocabulary gap with an LLM-generated pseudo-doc/answer. query2doc gives +3–15% on BM25 and is lexical-compatible. **But** Weller et al. (EACL 2024) — 11 techniques × 12 datasets × 24 retrievers — found a **strong negative correlation between retriever strength and expansion gains: expansion helps weak models and *hurts strong/precise* ones.** This is exactly your "+query expansion = −0.015 nDCG, it's a trap" finding. Conclusion: **expansion must be selective** (fire on hard/OOD/reasoning queries; stay out of the way on precise exact-match queries). That selectivity is a *router*, not an always-on feature.

### 2.5 Reranking & adaptive (cost-aware) retrieval — the "cheap by default, accurate on demand" engine

- LLM/cross-encoder rerankers (RankZephyr, monoT5, Cohere Rerank 3.5, bge-reranker) lift first-stage BM25 by **+5 to +15 nDCG@10**, at one LLM call/query. Biggest single jump, not free — and (your finding, confirmed) it lifts dense *more* because dense has better candidate recall.
- **Adaptive-RAG (NAACL 2024)** + the gating lineage (TARG, AcuRank, confidence-gated reranking `[fresh]`) show you can **route compute by query difficulty** and recover ~90–100% of always-rerank quality while skipping the expensive call on the (usually majority) easy queries.
- **Distillation into the index (SPLADE/MarginMSE; LiT5):** cross-encoder ranking knowledge can be *compiled into doc-side term weights at index time*, then served at BM25 speed with no query-time model. This is the literature-backed heart of "front-load the smart compute offline." It is the *same idea* as Compiled Retrieval, and it works — the only twist KAPI adds is doing it **training-free, via an LLM, at index time.**

### 2.6 Market reality — the wind is at your back

- **The vector-DB correction is happening.** Pinecone (~$750M valuation, 2023) reportedly stalled (~$14M ARR, lost Notion, founder moved to Chief Scientist in Sep 2025). The narrative shifted hard: *"From shiny object to sober reality,"* *"Vector search is reaching its limit"* (close ≠ correct: returns "Error 222" for "Error 221"). Hybrid is now the production default; "you may not need a dedicated vector DB" (pgvector, ParadeDB, Elastic/OpenSearch hybrid) is mainstream.
- **Developers actively want infra-free lexical search.** `bm25s` (pure NumPy, up to 500× faster than rank-bm25, matches Elasticsearch BM25, `pip install`, no Java/GPU) hit the HN front page. That's KAPI's exact lane, validated.
- **Embedding cost is recurring and avoidable.** ~$0.02–0.13 / 1M tokens *at both index and query time*, **~6.1 GB RAM per 1M docs at 1536-d**, plus vector-DB ops. KAPI's number is **$0** at query time and **$0** for storage of vectors. That's a TCO pitch, not a vibe.
- **Competitors** (LlamaIndex ~45k★, RAGFlow ~65k★, Haystack, txtai, RAGatouille/ColBERT) are heavyweight or embedding-centric. The recurring complaint is the **"hidden 80%"** of RAG infra and ongoing tuning. Nobody owns "zero-setup, embedding-free, *and provably competitive on hard queries*."

---

## 3. The gap (synthesized white space)

Putting the four clusters together, the unexploited intersection is sharp:

1. **The document side is where models add value, and it's index-time** → cached LLM is cheap there. (§2.1, §2.5)
2. **Reasoning is what wins hard retrieval, and it's currently paid at query time** (BRIGHT CoT, LATTICE traversal) or baked into a trained model. **Nobody has systematically pushed *reasoning* to the document side and compiled it into a lexical index.** (§2.3)
3. **Training-free LLM term weighting has failed** for a precise, fixable reason (drops literals, hallucinates, single-sample noise). **Nobody has fixed it with a literal floor + consensus.** (§2.1)
4. **Index-time generation works best when filtered** (doc2query--), and **nobody has unified contextual blurbs + doc2query + propositions + reasoning expansion under one self-consistency filter with offline-learned weights.** (§2.2)
5. **Expansion and reranking must be selective**, which means a **router** — and a router is what turns "cheap" into "cheap *and* accurate." (§2.4, §2.5)

That intersection is the mechanism.

---

## 4. The paradigm: Compiled Retrieval

Frame the entire category as a compute-placement choice, and name the axis so the conversation happens on your terms.

```
                 WHERE DOES THE EXPENSIVE, SMART COMPUTE RUN?

  Dense embeddings        ── model on every query + every doc        (JIT)
  Query-time reasoning    ── LLM reasons on every query              (interpreter)
  Trained reasoning model ── smart compute frozen into weights       (special-purpose silicon)
  ───────────────────────────────────────────────────────────────────────────────
  KAPI / Compiled Retrieval ── LLM compiles each doc ONCE, offline;  (AOT compiler)
                               queries run as cheap lexical bytecode
```

The compiler analogy is not just marketing — it dictates the architecture, the DX (`kapi compile ./docs`), the benchmark story (compile-time vs query-time cost), and the business model (the compiler is the IP). It also gives developers an instantly-graspable mental model, which is how libraries win adoption.

**Claim to defend:** *for a fixed query-time budget, moving reasoning from serve-time to compile-time is Pareto-superior whenever the corpus is queried more than a handful of times* — which is essentially always.

---

## 5. The novel core mechanism — Consensus Sparse Compilation (CSC)

One mechanism, five pillars. Pillar 2 is the genuinely new, publishable nugget; the others are proven parts assembled in a configuration nobody ships.

### Pillar 1 — The index-time reasoning compiler (the unification)

For each chunk, an LLM runs **offline, cached by content-hash** (KAPI already has this cache + cost guard), and emits a *bundle*:

- a **chunk-specific context blurb** (Anthropic) — situate it;
- the **questions/claims this chunk answers** (doc2query) — surface queries;
- **atomic, decontextualized propositions** (Dense X) — resolve "it/the company/this" so rare entities are matchable;
- **reasoning expansion** — the *inferential closure*: second-order facts, implications, and the multi-hop bridges a knowledgeable reader would draw that are **not lexically present**. *This is the BRIGHT-winning signal, precomputed.* (e.g. a passage stating a function's time complexity also "answers" a query about whether it scales to N=10⁹ — that bridge is generated and indexed.)

All four are *text*. They go into the **enriched lexical fields**, never into the displayed/cited text (KAPI's existing `indexed_text` vs `raw_text` split is exactly the right substrate).

### Pillar 2 — ★ Consensus term weighting (the new, training-free learned-sparse mechanism)

This is the research contribution and the answer to the open question *"can a general LLM produce SPLADE-style term weights without training?"* — for which the only prior attempts (BM42, PROSPER) either failed or needed training.

> **Sample the compiler k times at temperature > 0. A term's weight is its agreement across samples.** Terms the LLM emits in *every* sample (high self-consistency) get high weight; terms in *one* sample (likely hallucination) get near-zero. This is **self-consistency as a free, training-free relevance filter and weight estimator in one** — it operationalizes doc2query-- without a separate relevance model, and it produces graded learned-sparse weights without labels or fine-tuning.

Why it should work where BM42/PROSPER struggled: hallucinated expansion terms are, by definition, *not reproducible* across samples → consensus suppresses them automatically; genuinely entailed terms recur → consensus promotes them. No model is trained; the "learning" is the LLM's own agreement signal. (It also gives you a natural confidence score per term for free.)

### Pillar 3 — Literal anchoring (training-free fix to the known failure mode)

The documented LLM-weighting failure is *under-weighting literal terms* (IDs, error codes, proper nouns). Fix by **rule, not by a trained residual** (PROSPER needed a trained one — you don't): every literal term in the source keeps a floor weight = its BM25/IDF weight; the LLM/consensus signal may only **add** expansion terms *above* the floor. This converts the LLM from a "weigher" (where it fails) into an "expander on a lexical safety net" (where doc2query succeeds). Exact-match retrieval — BM25's home turf and a real-world majority — is structurally protected.

### Pillar 4 — Asymmetric all-sparse fusion (embedding-free "hybrid")

Hybrid wins because it fuses *two different error profiles*, not because one leg is dense. So build both legs sparse:

- **Leg A — literal:** plain BM25 + char-trigram over `raw_text` (exact match, rare tokens, typos).
- **Leg B — compiled:** the CSC-weighted enriched field (semantic/inferential reach).

Fuse with **convex combination** (Bruch et al. TOIS 2023: more robust + sample-efficient than RRF when you have even a few labeled queries) and keep RRF as the zero-tuning default. Net: the complementarity that justifies "hybrid," reproduced with **no embedding model and no vector DB.** (Must validate the two legs are genuinely decorrelated — if Leg B just echoes Leg A, no gain. That's an experiment, §7.)

### Pillar 5 — The adaptive query router (cheap by default, accurate on demand)

The only query-time LLM use, and it's gated:

- **Default:** pure lexical, ~1 ms, $0. Most queries (exact/precise) end here — and the §2.4 precision-trap evidence says they *should*, untouched.
- **Escalate** (query2doc / CoT reasoning / rerank top-k) **only** when a cheap confidence signal says the result set is weak/ambiguous (score gap, query length/type, no high-score hit). This is the Adaptive-RAG/TARG pattern: recover ~all of the always-on quality at a fraction of the cost — and it dodges the precision trap by construction.

**CSC in one line:** *compile each document's queryable + inferential closure into a literal-anchored, consensus-weighted sparse index offline; serve two complementary sparse legs at BM25 speed; spend an LLM at query time only when the cheap path is unsure.*

What's defensibly **new** here: (2) consensus/self-consistency as a training-free learned-sparse weighter+filter; (1)+(4) reasoning-expansion compiled to a lexical index as the embedding-free path to reasoning-intensive retrieval; (3) rule-based literal anchoring as the training-free fix to the BM42/PROSPER failure. The paper writes itself: *"Compiled Retrieval: moving reasoning from query time to index time for embedding-free RAG."*

---

## 6. Why this wins on all three axes

**Benchmark dominance.** The hero benchmark is **BRIGHT**, not scifact — that's the board where embedding-free *wins* and dense *collapses* (18.3!). Target: **beat off-the-shelf dense decisively at $0 query cost**, and approach the embedding-free SOTA (LATTICE 46.7) while being **cheaper at query time than LATTICE** (LATTICE pays an LLM traversal per query; CSC pays BM25). Keep BEIR for breadth/parity and your existing cost-tiered leaderboard as the format — it's already your credibility moat.

**Developer adoption.** The compiler mental model (`kapi compile ./docs && kapi query`), `pip install` with no model/GPU/DB, $0 query cost, deterministic + explainable scores, bm25s-class speed, and **drop-in LangChain/LlamaIndex retriever adapters** so trying it is one line, not a migration. Lead with a reproducible BRIGHT win + a TCO table vs a dense+vector-DB stack. This is the bm25s/RAGatouille playbook, with a quality proof attached.

**Startup wedge.** The **compiler is the IP**: OSS the fast retriever (win stars), monetize the **hosted/managed compilation pipeline** (the expensive, value-adding, hard-to-replicate step) — serving stays free and local. Wedge cohorts: teams burned by vector-DB cost/complexity (the Pinecone-correction crowd) and **edge / on-prem / air-gapped / regulated** deployments that *structurally cannot* ship data to an embedding API or run a vector DB. "No embeddings, no vector DB, no GPU, provably competitive on hard queries" is a clean, quantifiable pitch.

---

## 7. How to prove it (benchmark & experiment plan)

Run these in your existing `kapi.eval` harness; keep the honest cost-tier format.

1. **Hero result — BRIGHT.** KAPI-lexical vs KAPI+CSC vs off-the-shelf dense vs BM25+CoT vs (if feasible) LATTICE. Headline you're hunting: *CSC beats off-the-shelf dense on BRIGHT at $0 query cost; with the router on, it approaches query-time-reasoning systems at a fraction of their per-query cost.*
2. **Ablations (the paper's spine), each isolated:** +context blurb, +doc2query, +propositions, +reasoning-expansion, +consensus weighting (k=1 vs 3 vs 5 vs 8), +literal anchoring on/off, RRF vs convex fusion. Show each pillar's marginal nDCG and the consensus-k curve.
3. **Decorrelation test (kill-shot for Pillar 4):** measure rank correlation between Leg A and Leg B; the fusion gain must come from genuine complementarity, not echo.
4. **The BM42 honesty gate:** benchmark against a **well-tuned** BM25 (k1/b swept) and a real dense baseline on **full BEIR**, not one easy dataset. Report where CSC *loses* (it will, on paraphrase-heavy MTEB-style queries). Credibility is the moat; do not repeat BM42's mistake.
5. **Cost accounting:** index-time $ per 1M tokens (with prompt caching + the consensus k multiplier), index-size delta (use doc2query-- filtering / consensus pruning to keep it bounded — target the −33% regime), query-time latency distribution with the router.
6. **Robustness:** typo/morphology (trigrams), rare-entity/long-tail (propositions should shine), multilingual.

---

## 8. Roadmap (phased, each phase shippable)

> **Implementation status (2026-07-01):** Phases 0–5 are all built and tested (87 passed / 3
> skipped). Compiler + CSC consensus/anchoring (`kapi/augment/compiler.py`,
> `kapi/retrieve/sparse.py`), two-leg fusion + adaptive router (`kapi/retrieve/router.py`),
> portable bundles + hosted compile service (`kapi/portable.py`, `kapi/service.py`), TCO model
> and LangChain/LlamaIndex adapters (`kapi/tco.py`, `kapi/integrations/`). Remaining Phase-4
> work is content, not code: the reproducible BRIGHT-win blog post and the paper.

- **Phase 0 — Reframe (days).** Adopt the *Compiled Retrieval* narrative across README/site. Add the `compile` verb to the CLI/API as an alias over `add`. Add the BRIGHT runner to `kapi.eval`. Cheap, high-leverage positioning.
- **Phase 1 — Compiler v1 (1–2 wks).** Unify existing contextual indexing with **doc2query + propositions** in one offline pass, each landing in `indexed_text`. Reuse the content-hash cache + cost guard. Add the **doc2query-- self-consistency filter**. Re-run scifact + add BRIGHT. *Expect a real jump on BRIGHT from propositions + doc2query alone — bankable before the novel part.*
- **Phase 2 — CSC core (2–4 wks).** Implement **consensus term weighting** (sample k, weight by agreement) + **literal anchoring** (BM25 floor) → per-chunk learned-sparse weights stored in the engine. This needs per-term boosting; Tantivy term-queries can carry weights, or move the weighted leg to the SQLite/bm25s path first. Full ablations. **This is the paper.**
- **Phase 3 — Asymmetric fusion + router (1–2 wks).** Two-leg convex/RRF fusion; the confidence-gated **adaptive router** for query-time escalation. Land the cost-tier leaderboard update.
- **Phase 4 — Adoption surface (ongoing).** LangChain/LlamaIndex retriever adapters, `kapi compile` ergonomics, TCO calculator, a reproducible BRIGHT-win blog post + the paper. 
- **Phase 5 — Wedge (when traction shows).** Hosted compilation service; on-prem/air-gapped distribution.

---

## 9. Risks & honest counterpoints

- **Consensus might just reproduce BM25.** If the LLM's high-agreement terms are mostly the literal terms, Leg B adds little. *Mitigation:* measure decorrelation (§7.3); weight the *novelty* of consensus terms (down-weight terms already strong in Leg A).
- **Index bloat & compile cost.** k-sampling multiplies index-time tokens. *Mitigation:* prompt caching, small/local compiler model, consensus pruning (drop low-agreement terms — this is also the doc2query-- −33% win), and the cache makes re-compiles free.
- **Some reasoning is irreducibly query-specific** and can't be precompiled. *Mitigation:* the router escalates those to query-time CoT — be upfront that CSC is "95% of the quality at ~1% of the query cost, escalate the rest," not "100% offline magic."
- **Trained reasoning models (DIVER/BGE-Reasoner) score higher.** True — but they're not embedding-free or cheap. Compete on the **cost-quality Pareto frontier**, not the absolute top; that's a winnable, honest claim.
- **Re-compilation on corpus churn.** KAPI's incremental `sync` already only re-indexes changed files; consensus is per-chunk and cached, so churn cost is bounded.
- **"It's just doc2query + tricks."** The defensible novelty is consensus weighting (training-free learned-sparse) + literal anchoring (training-free failure-mode fix) + reasoning-expansion-to-lexical for BRIGHT. Lead the paper with those three, benchmark honestly, and the contribution stands.

---

## 10. Bottom line

You don't need to out-embed the embedders. You need to **change where the compute happens** and own the framing. *Compiled Retrieval* + *Consensus Sparse Compilation* gives KAPI (1) a hero benchmark it can win (BRIGHT, where dense collapses and embedding-free already sits at the top via LATTICE), (2) a developer story with an instantly-graspable mental model and $0 query cost, and (3) a business where the compiler is the moat. The fast/cheap soul of the project is preserved — in fact it becomes the *thesis*, not a compromise.

Next concrete step: **add the BRIGHT runner and ship Compiler v1 (Phase 1)** to bank the propositions/doc2query gain, then build CSC (Phase 2) — that's the paper and the moat.

---

## 11. Annotated bibliography (verified)

**Reasoning-intensive retrieval (the hero cluster)**
- Su et al., *BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval*, arXiv:2407.12883, ICLR 2025. Dense collapses (SFR 59 MTEB → 18.3 BRIGHT); BM25+GPT-4 CoT ~14→27, beats off-the-shelf dense.
- Gupta, Chang, et al., *LLM-guided Hierarchical Retrieval (LATTICE)*, arXiv:2510.13217, Oct 2025 `[fresh]`. 46.7 nDCG@10 on BRIGHT, single off-the-shelf LLM, **no embedding model in the search loop**; LATTICE++ 49.1. The embedding-free existence proof.
- Shao et al. (Meta), *ReasonIR*, arXiv:2504.20595, Apr 2025. DIVER (arXiv:2508.07995) 45.8; BGE-Reasoner 45.2 — strong but trained & not embedding-free.

**Index-time enrichment**
- Anthropic, *Introducing Contextual Retrieval*, Sep 2024. Contextual BM25+embeddings −49% failures; chunk-specific context matters; ~$1.02/1M tokens cached.
- Nogueira & Lin, *From doc2query to docTTTTTquery*, 2019. MS MARCO MRR@10 0.184→0.277.
- Gospodinov, MacAvaney, Macdonald, *Doc2Query--: When Less is More*, arXiv:2301.03266, ECIR 2023. Relevance-filter generated queries → +16% effectiveness, −23% latency, −33% index.
- Chen et al., *Dense X Retrieval: What Retrieval Granularity Should We Use?*, arXiv:2312.06648, EMNLP 2024. Propositions → +22–35% Recall@5 unsupervised, biggest on rare-entity/cross-domain.
- Sarthi et al., *RAPTOR*, arXiv:2401.18059, ICLR 2024. Recursive LLM summary tree; summary nodes are text.

**Learned-sparse & the training-free cautionary tales**
- Lassance et al., *SPLADE-v3*, arXiv:2403.06789. Trained; ~+4 nDCG@10 over BM25 on BEIR.
- Vasnetsov (Qdrant), *BM42*, Jul 2024 — **retracted**; Reimers/Bergum critique ("Please don't trust us"). Training-free attention-as-weight failed vs well-tuned BM25.
- Song et al., *LLMs as Sparse Retrievers (PROSPER)*, arXiv:2510.18527, Oct 2025 `[fresh]`. LLM term-weights need a *trained* literal residual; LLMs under-weight literals; no scaling law.
- Nardini et al., *Effective Inference-Free Retrieval for Learned Sparse Representations*, arXiv:2505.01452, SIGIR 2025 `[fresh]`; Geng et al. (OpenSearch), arXiv:2411.04403, Nov 2024. Query side can be model-free (IDF) at small cost.

**Fusion, expansion, reranking, adaptive**
- Cormack et al., *Reciprocal Rank Fusion*, SIGIR 2009. Bruch et al., *An Analysis of Fusion Functions for Hybrid Retrieval*, arXiv:2210.11934, TOIS 2023 (convex > RRF when tunable).
- Wang, Yang, Wei, *Query2doc*, arXiv:2303.07678, EMNLP 2023 (+3–15% BM25). Gao et al., *HyDE*, arXiv:2212.10496, ACL 2023.
- Weller et al., *When do Generative Query and Document Expansions Fail?*, arXiv:2309.08541, EACL 2024. Expansion helps weak retrievers, **hurts strong/precise** ones — the precision trap.
- Pradeep et al., *RankZephyr*, arXiv:2312.02724; Liu et al., *Lost in the Middle*, TACL 2023 (order sensitivity).
- Jeong et al., *Adaptive-RAG*, NAACL 2024 (route compute by query difficulty). TARG / AcuRank / confidence-gated reranking, 2025 `[fresh]`. SPLADE/MarginMSE & LiT5 distillation — compile reranker signal into the index.

**Efficiency & market**
- Lù, *BM25S*, arXiv:2407.03618, Jul 2024. Up to 500× faster than rank-bm25, pure NumPy; HN front page — demand for infra-free lexical search.
- Vector-DB correction: VentureBeat *"From shiny object to sober reality"*; The New Stack *"Vector Search Is Reaching Its Limit"*; Pinecone leadership/valuation reporting (2023–2025). Embedding pricing & RAM figures per OpenAI/Cohere/Voyage docs and production write-ups. *(Analyst RAG market-size figures, $1.2–11B, are marketing-grade — cite as a range.)*
