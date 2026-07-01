"""BRIGHT runner — the hero benchmark for Compiled Retrieval (STRATEGY §2.3, §6).

BRIGHT (Su et al., ICLR 2025) is built so that surface/semantic similarity is *insufficient*:
relevance requires multi-step reasoning. This is the board where embedding-free wins and
off-the-shelf dense collapses — the #1 MTEB dense model (SFR-Embedding-Mistral, 59.0 MTEB)
scores **18.3** here, while BM25 with GPT-4 chain-of-thought-rewritten queries jumps to ~27,
beating off-the-shelf dense. The strategy: compile that reasoning to *index time* so query
stays ~1 ms (CSC), and beat off-the-shelf dense at $0 query cost.

Mirrors eval/beir.py: index a subset's documents, run its reasoning queries through a Kapi
instance, max-pool chunk scores to the source doc, score nDCG@10 (the BRIGHT metric).
BRIGHT marks ``excluded_ids`` per query (e.g. the passage the query was written from); these
are dropped from the run before scoring, exactly as the official harness does.

Requires the eval extra (``datasets``):  pip install kapi[eval]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Set

from .._types import Document
from .ir_metrics import evaluate_run

# The 12 BRIGHT subsets (HF ``xlangai/BRIGHT``); each is a split of the documents/examples configs.
BRIGHT_SUBSETS = (
    "biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow",
    "sustainable_living", "leetcode", "pony", "aops", "theoremqa_questions",
    "theoremqa_theorems",
)

# Directional headline reference points from STRATEGY §2.3 / §11 (nDCG@10 on BRIGHT, averaged
# over subsets). These are leaderboard anchors, NOT per-subset BM25 — verify before quoting.
BRIGHT_REFERENCE = {
    "bm25_zeroshot_avg": 14.3,        # plain BM25, zero-shot
    "dense_offtheshelf_avg": 18.3,    # SFR-Embedding-Mistral (#1 MTEB) — collapses here
    "bm25_gpt4_cot_avg": 27.0,        # BM25 + GPT-4 CoT-rewritten queries (reasoning lever)
    "lattice_avg": 46.7,             # LATTICE — embedding-free SOTA, but pays an LLM per query
}


@dataclass
class BrightReport:
    subset: str
    scores: Dict[str, float]
    n_queries: int = 0
    n_docs: int = 0
    reference: Dict[str, float] = field(default_factory=lambda: dict(BRIGHT_REFERENCE))

    def __str__(self) -> str:
        lines = [f"BRIGHT[{self.subset}]  docs={self.n_docs}  queries={self.n_queries}"]
        for m, v in self.scores.items():
            lines.append(f"  {m:12s} {v:.4f}")
        mine = self.scores.get("ndcg@10")
        if mine is not None:
            lines.append("  -- directional reference (avg over subsets; verify before quoting) --")
            lines.append(f"  off-the-shelf dense  {self.reference['dense_offtheshelf_avg'] / 100:.4f}  "
                         f"(beating this at $0 query cost is the headline target)")
            lines.append(f"  BM25 zero-shot       {self.reference['bm25_zeroshot_avg'] / 100:.4f}")
            lines.append(f"  BM25 + GPT-4 CoT     {self.reference['bm25_gpt4_cot_avg'] / 100:.4f}")
            lines.append(f"  LATTICE (emb-free)   {self.reference['lattice_avg'] / 100:.4f}")
        return "\n".join(lines)


def load_bright(subset: str = "biology", *, data_repo: str = "xlangai/BRIGHT"):
    """Load one BRIGHT subset. Returns (corpus, queries, qrels, excluded).

    corpus: {doc_id: {"text": ...}}; queries: {qid: text}; qrels: {qid: {doc_id: 1}};
    excluded: {qid: set(doc_id)} — removed from the run before scoring.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("BRIGHT eval requires: pip install kapi[eval]") from exc
    if subset not in BRIGHT_SUBSETS:
        raise ValueError(f"unknown BRIGHT subset {subset!r}; expected one of {BRIGHT_SUBSETS}")

    docs_split = load_dataset(data_repo, "documents", split=subset)
    ex_split = load_dataset(data_repo, "examples", split=subset)

    corpus: Dict[str, dict] = {str(r["id"]): {"text": r.get("content", "") or ""}
                               for r in docs_split}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}
    excluded: Dict[str, Set[str]] = {}
    for r in ex_split:
        qid = str(r["id"])
        queries[qid] = r.get("query", "") or ""
        qrels[qid] = {str(g): 1 for g in (r.get("gold_ids") or [])}
        excluded[qid] = {str(e) for e in (r.get("excluded_ids") or [])}
    return corpus, queries, qrels, excluded


def corpus_to_documents(corpus: Dict[str, dict]):
    for doc_id, d in corpus.items():
        text = d.get("text", "") or ""
        yield Document(doc_id=doc_id, text=text, source=doc_id,
                       metadata={"content_type": "text", "source": doc_id})


def _doc_id_of(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def run_bright(
    rag_factory: Callable[[], "object"],
    subset: str = "biology",
    *,
    k: int = 100,
    metrics=("ndcg@10", "recall@100", "mrr"),
    data_repo: str = "xlangai/BRIGHT",
) -> BrightReport:
    """Build a Kapi via ``rag_factory()``, index a BRIGHT subset, evaluate its queries."""
    corpus, queries, qrels, excluded = load_bright(subset, data_repo=data_repo)
    rag = rag_factory()
    rag.add(list(corpus_to_documents(corpus)))

    run: Dict[str, Dict[str, float]] = {}
    for qid, qtext in queries.items():
        if not qtext.strip():
            run[qid] = {}
            continue
        hits = rag.search(qtext, k=k + len(excluded.get(qid, ())))
        drop = excluded.get(qid, set())
        doc_scores: Dict[str, float] = {}
        for h in hits:
            did = _doc_id_of(h.chunk_id)
            if did in drop:
                continue                                  # BRIGHT excludes these from scoring
            if h.score > doc_scores.get(did, float("-inf")):
                doc_scores[did] = h.score                 # max-pool chunks -> doc
        run[qid] = doc_scores

    scores = evaluate_run(qrels, run, metrics)
    return BrightReport(subset=subset, scores=scores,
                        n_queries=len(queries), n_docs=len(corpus))


def run_bright_all(rag_factory: Callable[[], "object"], **kw) -> Dict[str, BrightReport]:
    """Run every BRIGHT subset; returns {subset: BrightReport}. Heavy — downloads all 12."""
    return {s: run_bright(rag_factory, s, **kw) for s in BRIGHT_SUBSETS}
