"""Reproducible lexical-core benchmark grid (no LLM, no embeddings, no heavy deps).

Downloads BEIR datasets with the standard library (no `beir` package needed), indexes
them once per engine, then sweeps query-time parameters (field weights, rrf_k, fusion)
and index-time BM25 parameters (bm25s k1/b). Chunk scores are max-pooled to doc ids,
matching BEIR's document-level judgments.

Usage:
    python benchmarks/lexical_grid.py weights  --dataset scifact --engine tantivy
    python benchmarks/lexical_grid.py rrf      --dataset scifact --engine sqlite
    python benchmarks/lexical_grid.py bm25s    --dataset scifact
    python benchmarks/lexical_grid.py baseline --dataset scifact nfcorpus

Results are printed as a markdown table and appended to benchmarks/grid_results.jsonl
with full provenance (dataset, engine, params, metrics, date, nrag version).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from typing import Dict, Iterable, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrag import Document, Nrag
from nrag._types import FieldWeights
from nrag.eval.ir_metrics import evaluate_run
from nrag.retrieve import multisignal

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
METRICS = ("ndcg@10", "recall@10", "recall@100", "mrr")


# ---------------------------------------------------------------- data loading (stdlib)
def load_beir_stdlib(dataset: str, data_dir: str = "datasets", split: str = "test"):
    """Download + parse a BEIR dataset with the standard library only."""
    root = os.path.join(data_dir, dataset)
    if not os.path.exists(os.path.join(root, "corpus.jsonl")):
        os.makedirs(data_dir, exist_ok=True)
        url = BEIR_URL.format(name=dataset)
        print(f"downloading {url} ...", file=sys.stderr)
        with urllib.request.urlopen(url) as resp:
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


def documents(corpus: Dict[str, dict]) -> Iterable[Document]:
    for doc_id, d in corpus.items():
        title, text = d.get("title") or "", d.get("text") or ""
        yield Document(doc_id=doc_id, text=(f"{title}\n\n{text}" if title else text),
                       source=doc_id, metadata={"title": title, "source": doc_id})


# ---------------------------------------------------------------- evaluation core
def evaluate(rag: Nrag, queries: Dict[str, str], qrels, *, k: int = 100,
             field_weights: FieldWeights | None = None, rrf_k: int | None = None,
             fusion: str | None = None, weights=None) -> Dict[str, float]:
    """Run all queries through the retrieve layer; params default to the rag's config."""
    fw = field_weights or rag._field_weights()
    fusion = fusion or rag.config.fusion
    rrf_k = rrf_k if rrf_k is not None else rag.config.rrf_k
    run: Dict[str, Dict[str, float]] = {}
    for qid, qtext in queries.items():
        hits = multisignal.search(rag.engine, rag.store, qtext, k=k,
                                  field_weights=fw, fuzzy=rag.config.fuzzy,
                                  fusion=fusion, rrf_k=rrf_k, weights=weights)
        doc_scores: Dict[str, float] = {}
        for h in hits:
            did = h.chunk_id.split("::", 1)[0]
            if h.score > doc_scores.get(did, float("-inf")):
                doc_scores[did] = h.score
        run[qid] = doc_scores
    return evaluate_run(qrels, run, METRICS)


def log_result(tag: str, dataset: str, engine: str, params: dict, scores: Dict[str, float]):
    import nrag

    rec = {"tag": tag, "dataset": dataset, "engine": engine, "params": params,
           "scores": {m: round(v, 4) for m, v in scores.items()},
           "nrag_version": nrag.__version__, "date": time.strftime("%Y-%m-%d")}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_results.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _print_table(rows: list[Tuple[str, Dict[str, float]]]):
    print(f"\n| config | {' | '.join(METRICS)} |")
    print("|---|" + "---|" * len(METRICS))
    for name, s in sorted(rows, key=lambda r: -r[1]["ndcg@10"]):
        print(f"| {name} | " + " | ".join(f"{s[m]:.4f}" for m in METRICS) + " |")


def _index(dataset: str, engine: str, data_dir: str, **cfg) -> Tuple[Nrag, dict, dict]:
    corpus, queries, qrels = load_beir_stdlib(dataset, data_dir)
    t0 = time.time()
    rag = Nrag(preset="fast", engine=engine, **cfg)
    rag.add(list(documents(corpus)))
    print(f"indexed {len(corpus)} docs [{engine}] in {time.time()-t0:.1f}s", file=sys.stderr)
    return rag, queries, qrels


# ---------------------------------------------------------------- experiments
def exp_baseline(args):
    for dataset in args.dataset:
        rag, queries, qrels = _index(dataset, args.engine, args.data_dir)
        scores = evaluate(rag, queries, qrels)
        log_result("baseline", dataset, args.engine,
                   {"weights": rag._field_weights().as_dict()}, scores)
        _print_table([(f"{dataset} defaults", scores)])
        rag.close()


def exp_weights(args):
    ngrams = [0.0, 0.2, 0.3, 0.4, 0.6, 1.0]
    titles = [0.0, 0.5, 1.0, 1.5, 2.5]
    for dataset in args.dataset:
        rag, queries, qrels = _index(dataset, args.engine, args.data_dir)
        rows = []
        for nw in ngrams:
            for tw in titles:
                fw = FieldWeights(body=1.0, ngram=nw, title=tw)
                s = evaluate(rag, queries, qrels, field_weights=fw)
                rows.append((f"ngram={nw},title={tw}", s))
                log_result("weights", dataset, args.engine,
                           {"ngram": nw, "title": tw}, s)
        _print_table(rows)
        rag.close()


def exp_rrf(args):
    for dataset in args.dataset:
        rag, queries, qrels = _index(dataset, args.engine, args.data_dir)
        rows = []
        for rk in (10, 20, 40, 60, 100):
            s = evaluate(rag, queries, qrels, rrf_k=rk)
            rows.append((f"rrf_k={rk}", s))
            log_result("rrf", dataset, args.engine, {"rrf_k": rk}, s)
        for w in ([1.0, 0.3, 0.5], [1.0, 0.5, 1.0], [1.0, 1.0, 1.0]):
            s = evaluate(rag, queries, qrels, fusion="convex", weights=list(w))
            rows.append((f"convex w={w}", s))
            log_result("rrf", dataset, args.engine, {"convex": list(w)}, s)
        _print_table(rows)
        rag.close()


def exp_bm25s(args):
    """Sweep bm25s k1/b (index-time). Requires nrag[bm25s]."""
    for dataset in args.dataset:
        corpus, queries, qrels = load_beir_stdlib(dataset, args.data_dir)
        docs = list(documents(corpus))
        rows = []
        for k1 in (0.9, 1.2, 1.5):
            for b in (0.4, 0.6, 0.75, 1.0):
                rag = Nrag(preset="fast", engine="bm25s", bm25_k1=k1, bm25_b=b)
                rag.add(docs)
                s = evaluate(rag, queries, qrels)
                rows.append((f"k1={k1},b={b}", s))
                log_result("bm25s", dataset, "bm25s", {"k1": k1, "b": b}, s)
                rag.close()
        _print_table(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", choices=("baseline", "weights", "rrf", "bm25s"))
    ap.add_argument("--dataset", nargs="+", default=["scifact"])
    ap.add_argument("--engine", default="tantivy", choices=("tantivy", "sqlite", "bm25s"))
    ap.add_argument("--data-dir", default="datasets")
    args = ap.parse_args()
    {"baseline": exp_baseline, "weights": exp_weights,
     "rrf": exp_rrf, "bm25s": exp_bm25s}[args.experiment](args)


if __name__ == "__main__":
    main()
