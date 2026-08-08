"""BEIR runner — prove the no-embedding pipeline is "good enough" vs published BM25.

Indexes a BEIR corpus, runs the queries through an ``Nrag`` instance, and reports
nDCG@10 / Recall@100 / MRR next to the published BM25 numbers from the BEIR paper
(Thakur et al., 2021). Requires the eval extra:  pip install nrag[eval]

BEIR judges at *document* granularity, so per-doc chunk scores are max-pooled back to the
source doc id before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .._types import Document
from .ir_metrics import evaluate_run

# Published BM25 nDCG@10 from the BEIR paper (Anserini/Lucene). Directional reference.
PUBLISHED_BM25: Dict[str, float] = {
    "scifact": 0.665,
    "nfcorpus": 0.325,
    "trec-covid": 0.656,
    "arguana": 0.315,
    "scidocs": 0.158,
    "fiqa": 0.236,
    "quora": 0.789,
    "touche2020": 0.367,
}


@dataclass
class BeirReport:
    dataset: str
    scores: Dict[str, float]
    published_bm25_ndcg10: Optional[float] = None
    n_queries: int = 0
    n_docs: int = 0

    def __str__(self) -> str:
        lines = [f"BEIR[{self.dataset}]  docs={self.n_docs}  queries={self.n_queries}"]
        for m, v in self.scores.items():
            lines.append(f"  {m:12s} {v:.4f}")
        if self.published_bm25_ndcg10 is not None:
            mine = self.scores.get("ndcg@10")
            delta = f"  (Δ {mine - self.published_bm25_ndcg10:+.4f})" if mine is not None else ""
            lines.append(f"  published BM25 nDCG@10: {self.published_bm25_ndcg10:.4f}{delta}")
        return "\n".join(lines)


_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def load_beir(dataset: str = "scifact", split: str = "test", data_dir: str = "datasets"):
    """Download (if needed) and load a BEIR dataset. Returns (corpus, queries, qrels).

    Uses the ``beir`` package when installed; otherwise falls back to a
    standard-library loader (BEIR's on-disk format is plain jsonl + tsv), so
    ``run_beir`` works without the heavy ``nrag[eval]`` dependency chain.
    """
    try:
        from beir import util  # type: ignore
        from beir.datasets.data_loader import GenericDataLoader  # type: ignore
    except ImportError:
        return _load_beir_stdlib(dataset, split, data_dir)

    path = util.download_and_unzip(_BEIR_URL.format(name=dataset), data_dir)
    # a fresh loader per split (reusing one across splits raises KeyError in BEIR)
    return GenericDataLoader(data_folder=path).load(split=split)


def _load_beir_stdlib(dataset: str, split: str, data_dir: str):
    """Parse the BEIR zip format with the standard library only."""
    import io
    import json
    import os
    import urllib.request
    import zipfile

    root = os.path.join(data_dir, dataset)
    if not os.path.exists(os.path.join(root, "corpus.jsonl")):
        os.makedirs(data_dir, exist_ok=True)
        with urllib.request.urlopen(_BEIR_URL.format(name=dataset)) as resp:
            data = resp.read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(data_dir)

    corpus: Dict[str, dict] = {}
    with open(os.path.join(root, "corpus.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            corpus[str(d["_id"])] = {"title": d.get("title", ""), "text": d.get("text", "")}

    queries: Dict[str, str] = {}
    with open(os.path.join(root, "queries.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            queries[str(d["_id"])] = d["text"]

    qrels: Dict[str, Dict[str, int]] = {}
    with open(os.path.join(root, "qrels", f"{split}.tsv"), encoding="utf-8") as fh:
        next(fh)  # header
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    queries = {qid: q for qid, q in queries.items() if qid in qrels}
    return corpus, queries, qrels


def corpus_to_documents(corpus: Dict[str, dict]):
    for doc_id, d in corpus.items():
        title = d.get("title", "") or ""
        text = d.get("text", "") or ""
        yield Document(doc_id=doc_id, text=(f"{title}\n\n{text}" if title else text),
                       source=doc_id, metadata={"content_type": "text", "title": title,
                                                "source": doc_id})


def _doc_id_of(chunk_id: str) -> str:
    return chunk_id.split("::", 1)[0]


def run_beir(
    rag_factory: Callable[[], "object"],
    dataset: str = "scifact",
    split: str = "test",
    *,
    k: int = 100,
    metrics=("ndcg@10", "recall@100", "mrr"),
    data_dir: str = "datasets",
) -> BeirReport:
    """Build an Nrag via ``rag_factory()``, index the BEIR corpus, evaluate the queries."""
    corpus, queries, qrels = load_beir(dataset, split, data_dir)
    rag = rag_factory()
    rag.add(list(corpus_to_documents(corpus)))

    run: Dict[str, Dict[str, float]] = {}
    for qid, qtext in queries.items():
        hits = rag.search(qtext, k=k)
        doc_scores: Dict[str, float] = {}
        for h in hits:
            did = _doc_id_of(h.chunk_id)
            if h.score > doc_scores.get(did, float("-inf")):
                doc_scores[did] = h.score  # max-pool chunks -> doc
        run[qid] = doc_scores

    scores = evaluate_run(qrels, run, metrics)
    return BeirReport(dataset=dataset, scores=scores,
                      published_bm25_ndcg10=PUBLISHED_BM25.get(dataset),
                      n_queries=len(queries), n_docs=len(corpus))
