"""Live CSC eval on BEIR scifact (loads via HF ``datasets``, no torch/beir needed).

Modes:
  smoke      — compile a few docs with the real LLM; print a bundle + term weights (cheap)
  baseline   — pure-lexical Kapi, no LLM (free)
  compiled   — Compiled Retrieval; --csc on/off, --k consensus samples

Reads the LLM key from ``OPENROUTER_API_KEY``. Example:
  OPENROUTER_API_KEY=... python benchmarks/csc_eval.py baseline
  OPENROUTER_API_KEY=... python benchmarks/csc_eval.py compiled --index ./idx_csc --k 1
"""

from __future__ import annotations

import argparse
import os
import time

from kapi import Document, Kapi
from kapi.eval.ir_metrics import evaluate_run

MODEL = "deepseek/deepseek-v4-flash"
BASE_URL = "https://openrouter.ai/api/v1"
METRICS = ("ndcg@10", "recall@10", "recall@100", "mrr")


def _llm():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")
    from kapi.llm import OpenAICompatLLM

    return OpenAICompatLLM(base_url=BASE_URL, model=MODEL, api_key=key)


def load_scifact():
    from datasets import load_dataset

    corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
    queries = load_dataset("BeIR/scifact", "queries")["queries"]
    qrels_ds = load_dataset("BeIR/scifact-qrels")["test"]

    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    qtext = {str(r["_id"]): r["text"] for r in queries}
    queries_test = {qid: qtext[qid] for qid in qrels if qid in qtext}
    docs = {str(r["_id"]): ((r.get("title") or "") + "\n\n" + (r.get("text") or "")).strip()
            for r in corpus}
    return docs, queries_test, qrels


def to_documents(docs):
    for did, text in docs.items():
        yield Document(doc_id=did, text=text, source=did,
                       metadata={"content_type": "text", "source": did})


def _doc_id(chunk_id):
    return chunk_id.split("::", 1)[0]


def evaluate(rag, queries, qrels, k=100):
    run = {}
    for qid, qt in queries.items():
        scores = {}
        for h in rag.search(qt, k=k):
            did = _doc_id(h.chunk_id)
            if h.score > scores.get(did, float("-inf")):
                scores[did] = h.score
        run[qid] = scores
    return evaluate_run(qrels, run, METRICS)


def build(mode, args):
    if mode == "baseline":
        return Kapi(engine="tantivy", path=args.index)
    over = dict(consensus_k=args.k, compile_concurrency=args.concurrency)
    rag = Kapi(llm=_llm(), preset="compiled", engine="tantivy", path=args.index, **over)
    if args.csc == "off":
        rag._legb = None            # Leg A (enriched index) only — no Leg B fusion, no recompile
    return rag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("smoke", "baseline", "compiled"))
    ap.add_argument("--index", default=None)
    ap.add_argument("--k", type=int, default=1, help="consensus samples")
    ap.add_argument("--csc", choices=("on", "off"), default="on")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--smoke-docs", type=int, default=12)
    args = ap.parse_args()

    if args.mode == "smoke":
        docs, _q, _r = load_scifact()
        sub = dict(list(docs.items())[: args.smoke_docs])
        rag = Kapi(llm=_llm(), preset="compiled", consensus_k=args.k, compile_concurrency=6)
        t = time.time()
        rag.compile(list(to_documents(sub)))
        cid = rag.store.all_chunks()[0].chunk_id
        ch = rag.store.get_chunk(cid)
        print(f"compiled {len(sub)} docs in {time.time()-t:.1f}s")
        print("--- enriched indexed_text (head) ---")
        print(ch.indexed_text[:600])
        if rag._legb is not None:
            v = rag._legb.vectors.get(cid, {})
            print(f"--- Leg B vector ({len(v)} terms, head) ---")
            print(dict(list(sorted(v.items(), key=lambda x: -x[1]))[:15]))
        rag.close()
        return

    docs, queries, qrels = load_scifact()
    print(f"scifact: {len(docs)} docs, {len(queries)} queries")
    rag = build(args.mode, args)
    t = time.time()
    rep = rag.add(list(to_documents(docs)))
    print(f"index: {rep}  ({time.time()-t:.1f}s)")
    if rag._legb is not None:
        print(f"Leg B: {rag.stats().get('csc')}")
    t = time.time()
    scores = evaluate(rag, queries, qrels)
    label = args.mode if args.mode == "baseline" else f"compiled csc={args.csc} k={args.k}"
    print(f"\n=== {label} ({time.time()-t:.1f}s eval) ===")
    for m in METRICS:
        print(f"  {m:12s} {scores[m]:.4f}")
    rag.close()


if __name__ == "__main__":
    main()
