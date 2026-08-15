"""Dense-vs-NRAG BEIR benchmark — the "$0/query vs market embedding models" table.

Builds a NRAG pure-lexical index (preset="fast", $0/query) and, on the same BEIR
datasets, runs 4 market embedding models served through OpenRouter's /embeddings
endpoint (cosine over L2-normalized vectors, the standard dense setup). Every system
is scored at *document* granularity with the same metrics (nDCG@10 / R@10 / R@100 /
MRR), exactly like benchmarks/lexical_tuning_v014.md, so the tables read side by side.

Embeddings are cached to disk (default: .bench_cache/) keyed by model+dataset, so
re-runs are free and a crash never loses paid work. Actual spend is tracked from the
API's per-response `usage.cost` field and written to the results file; a cost cap
(default $5) aborts the run before it can run away.

Usage:
    OPENROUTER_API_KEY=... python benchmarks/dense_bench.py                          # all 3 datasets, all 4 models
    OPENROUTER_API_KEY=... python benchmarks/dense_bench.py --datasets scifact       # one dataset
    OPENROUTER_API_KEY=... python benchmarks/dense_bench.py --models qwen/qwen3-embedding-4b
    python benchmarks/dense_bench.py --skip-dense                                    # local NRAG baseline only
    python benchmarks/dense_bench.py --limit-docs 200 --limit-queries 20             # smoke test, ~free

The API key is read from OPENROUTER_API_KEY only — never from argv or files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lexical_grid import METRICS, documents, evaluate, load_beir_stdlib
from nrag import Nrag
from nrag._types import FieldWeights
from nrag.eval.ir_metrics import evaluate_run

# (openrouter id, short display name). qwen3-4b is the README's claimed tie — kept in.
MODELS = [
    ("openai/text-embedding-3-small", "text-embedding-3-small"),
    ("openai/text-embedding-3-large", "text-embedding-3-large"),
    ("BAAI/bge-m3", "bge-m3"),
    ("qwen/qwen3-embedding-4b", "qwen3-embedding-4b"),
]

# scifact-only context: NRAG + doc2query x2 from the checked-in scifact_results.md
# (LLM enrichment once at index time, then $0/query). Same defaults era, prior run.
PRIOR_DOC2QUERY_SCIFACT = 0.7291
PUBLISHED_BM25 = {"scifact": 0.665, "nfcorpus": 0.325, "fiqa": 0.236}

API_URL = "https://openrouter.ai/api/v1/embeddings"
MAX_DOC_CHARS = 6000          # truncation guard (≈1500 tokens; all four models have ≥8k ctx)
BATCH = 128
MAX_RETRIES = 6
COST_CAP_USD = 5.0


def slug(model_id: str) -> str:
    return model_id.replace("/", "_")


def cache_dir(base: Path, model_id: str, dataset: str) -> Path:
    return base / slug(model_id) / dataset


# ------------------------------------------------------------------ OpenRouter client
def embed_all(model: str, texts: list[str], *, key: str, cache: Path,
              dataset: str, cost_cap: float, kind: str = "corpus",
              batch: int = BATCH) -> tuple[np.ndarray, dict]:
    """Embed ``texts`` (batched, retried, disk-cached). Returns (matrix, meta).

    Cache is written per batch (``shard_{kind}_NNN.npy`` + ``meta_{kind}.json``), so a
    crash or a cost-cap abort resumes from where it stopped instead of re-paying for
    work. ``kind`` separates the corpus and query matrices in the same model/dataset dir
    (they have different lengths and must never overwrite each other).
    """
    import httpx

    cdir = cache_dir(cache, model, dataset)
    npy = cdir / f"matrix_{kind}.npy"
    meta = cdir / f"meta_{kind}.json"
    if npy.exists() and meta.exists():
        m = json.loads(meta.read_text())
        if m.get("n") == len(texts):
            print(f"  [cache] {model} / {dataset} [{kind}]: {len(texts)} vectors, "
                  f"${m['cost']:.4f} spent previously", file=sys.stderr)
            return np.load(npy), m

    cdir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    n_batches = (len(texts) + batch - 1) // batch
    cost = 0.0
    tokens = 0
    dim = None
    n_done = 0

    # resume: load any completed shards and their recorded cost
    shards: dict[int, np.ndarray] = {}
    if meta.exists():
        m = json.loads(meta.read_text())
        if m.get("n") == len(texts) and m.get("texts_sha") == _sha(texts):
            cost, tokens = m["cost"], m["tokens"]
            for p in sorted(cdir.glob(f"shard_{kind}_*.npy")):
                idx = int(p.stem.split("_")[2])
                arr = np.load(p)
                shards[idx] = arr
                n_done += len(arr)
            dim = m["dim"]
            print(f"  [resume] {model} / {dataset} [{kind}]: {n_done}/{len(texts)} vectors cached, "
                  f"${cost:.4f} already spent", file=sys.stderr)

    client = httpx.Client(timeout=120.0)
    try:
        for i in range(0, len(texts), batch):
            idx = i // batch
            if idx in shards:
                continue
            chunk = []
            for t in texts[i:i + batch]:
                txt = t[:MAX_DOC_CHARS].strip()
                if not txt:  # empty inputs -> 400 "too_small" on OpenRouter; use a neutral vector
                    txt = " "
                chunk.append(txt)
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    r = client.post(API_URL, headers=headers,
                                    json={"model": model, "input": chunk})
                except httpx.HTTPError as e:
                    r = None
                    last = e
                if r is not None and r.status_code == 200:
                    d = r.json()
                    rows = [x["embedding"] for x in d["data"]]
                    if dim is None:
                        dim = len(rows[0])
                    assert all(len(x) == dim for x in rows), "inconsistent dims in batch"
                    shards[idx] = np.asarray(rows, dtype=np.float32)
                    np.save(cdir / f"shard_{kind}_{idx:03d}.npy", shards[idx])
                    u = d.get("usage") or {}
                    cost += float(u.get("cost") or 0.0)
                    tokens += int(u.get("prompt_tokens") or 0)
                    n_done += len(rows)
                    m = {"model": model, "dataset": dataset, "kind": kind, "n": len(texts),
                         "dim": dim, "cost": cost, "tokens": tokens,
                         "texts_sha": _sha(texts), "done": n_done,
                         "date": time.strftime("%Y-%m-%d")}
                    meta.write_text(json.dumps(m, indent=2))
                    break
                last = r.status_code if r is not None else last
                body = ""
                if r is not None:
                    try:
                        body = r.text[:200].replace("\n", " ")
                    except Exception:
                        body = ""
                wait = min(30.0, 1.5 ** attempt * (0.5 + 0.5 * np.random.random()))
                if r is not None and r.status_code == 429:
                    wait = max(wait, float(r.headers.get("retry-after", wait)))
                print(f"  [retry {attempt}/{MAX_RETRIES}] {model} {last}: {body} — wait {wait:.1f}s",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                raise RuntimeError(f"embedding failed for {model} after {MAX_RETRIES} tries: {last}")
            if n_done % (batch * 25) == 0:
                print(f"  {model} / {dataset}: {n_done}/{len(texts)} "
                      f"(~${cost:.3f})", file=sys.stderr)
            if cost > cost_cap:
                raise RuntimeError(f"cost cap ${cost_cap:.2f} exceeded (spent ${cost:.3f}); aborting. "
                                   f"Rerun to resume from cache.")
    finally:
        client.close()

    mat = np.concatenate([shards[i] for i in range(n_batches)], axis=0)
    m = {"model": model, "dataset": dataset, "kind": kind, "n": len(texts), "dim": dim,
         "cost": cost, "tokens": tokens, "texts_sha": _sha(texts), "done": len(texts),
         "date": time.strftime("%Y-%m-%d")}
    np.save(npy, mat)
    for p in cdir.glob(f"shard_{kind}_*.npy"):
        p.unlink()
    meta.write_text(json.dumps(m, indent=2))
    return mat, m


def _sha(texts: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\x1f".join(texts).encode("utf-8")).hexdigest()[:16]


def dense_run(emb_corpus: np.ndarray, emb_queries: np.ndarray,
              doc_ids: list[str], query_ids: list[str], qrels, *, k: int = 100) -> dict:
    """Cosine top-k retrieval + the same metrics as the lexical side."""
    q = emb_queries / np.maximum(np.linalg.norm(emb_queries, axis=1, keepdims=True), 1e-12)
    d = emb_corpus / np.maximum(np.linalg.norm(emb_corpus, axis=1, keepdims=True), 1e-12)
    sims = q @ d.T  # (n_queries, n_docs), in [-1, 1]
    run: dict[str, dict[str, float]] = {}
    for qi, qid in enumerate(query_ids):
        top = np.argsort(-sims[qi])[:k]
        run[qid] = {doc_ids[j]: float(sims[qi, j]) for j in top}
    return evaluate_run(qrels, run, METRICS)


# ------------------------------------------------------------------ report
def fmt(v) -> str:
    return "—" if v is None else f"{v:.4f}"


def write_report(rows: list[dict], datasets: list[str], out: Path) -> None:
    """rows: {dataset, system, emb, dims, query_cost, scores:{...}, spend}"""
    lines = [
        "# Dense embeddings vs NRAG — BEIR (scifact · nfcorpus · fiqa)",
        "",
        "Cost-fair comparison on three BEIR datasets. **NRAG = pure lexical** "
        "(`preset=\"fast\"`, BM25 + trigrams + title, no LLM): **$0 per query**.",
        "Dense = OpenRouter `/embeddings` + cosine over L2-normalized vectors, the "
        "standard dense setup (docs truncated to 6000 chars). Chunk/doc scores are "
        "max-pooled to document ids to match BEIR's document-level judgments.",
        "",
        f"Run date: {time.strftime('%Y-%m-%d')} · methodology: "
        "`python benchmarks/dense_bench.py` (cache in `.bench_cache/`).",
        "",
    ]
    for ds in datasets:
        lines += [f"## {ds}", "", "| System | Emb. | Dims | Query cost | nDCG@10 | R@10 | R@100 | MRR | Spend |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in rows if x["dataset"] == ds]:
            s = r["scores"]
            spend = f"${r['spend']:.3f}" if r["query_cost"] != "$0/query" else "—"
            lines.append(
                f"| {r['system']} | {r['emb']} | {r['dims']} | {r['query_cost']} | "
                f"{fmt(s['ndcg@10'])} | {fmt(s['recall@10'])} | {fmt(s['recall@100'])} | "
                f"{fmt(s['mrr'])} | {spend} |")
        # note which benchmark models are still missing for this dataset (not run)
        done = {r["system"].removesuffix(" (dense)") for r in rows if r["dataset"] == ds
                and r["query_cost"] != "$0/query"}
        missing = [short for _mid, short in MODELS if short not in done]
        if missing:
            lines.append(f"\n*Pending: `{'`, `'.join(missing)}` on {ds} "
                         "(not run — API key budget). Rerun with "
                         "`OPENROUTER_API_KEY=... python benchmarks/dense_bench.py "
                         f"--datasets {ds} --skip-nrag` to fill in.*")
        lines.append("")
    lines += ["## Honest headline", ""]
    for ds in datasets:
        d = [x for x in rows if x["dataset"] == ds and x["query_cost"] == "$0/query"
             and x["system"] == "NRAG (pure lexical)"]
        dense = [x for x in rows if x["dataset"] == ds and x["query_cost"] != "$0/query"]
        if not d or not dense:
            continue
        best = max(dense, key=lambda x: x["scores"]["ndcg@10"])
        nrag = max(d, key=lambda x: x["scores"]["ndcg@10"])
        lines.append(
            f"- **{ds}:** NRAG {nrag['scores']['ndcg@10']:.4f} vs best dense "
            f"({best['system']}) {best['scores']['ndcg@10']:.4f} nDCG@10 — "
            f"{'NRAG leads' if nrag['scores']['ndcg@10'] >= best['scores']['ndcg@10'] else 'dense leads'}.")
    lines.append("")
    lines += ["## Raw records", "", "```json"]
    for r in sorted(rows, key=lambda x: (x["dataset"], -x["scores"]["ndcg@10"])):
        lines.append(json.dumps(r, sort_keys=True))
    lines += ["```", ""]
    out.write_text("\n".join(lines))
    print(f"wrote {out}", file=sys.stderr)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus", "fiqa"])
    ap.add_argument("--models", nargs="+", default=[m[0] for m in MODELS])
    ap.add_argument("--skip-nrag", action="store_true", help="skip the local NRAG baseline")
    ap.add_argument("--skip-dense", action="store_true", help="skip the embedding runs")
    ap.add_argument("--limit-docs", type=int, default=0, help="subsample corpus (smoke test)")
    ap.add_argument("--limit-queries", type=int, default=0, help="subsample queries (smoke test)")
    ap.add_argument("--cost-cap", type=float, default=COST_CAP_USD)
    ap.add_argument("--cache-dir", default=".bench_cache")
    ap.add_argument("--out", default="benchmarks/dense_results.jsonl")
    ap.add_argument("--report", default="benchmarks/dense_results.md")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not args.skip_dense and not key:
        print("OPENROUTER_API_KEY is required for dense runs (or pass --skip-dense)", file=sys.stderr)
        return 2
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # load (and optionally subsample) datasets once
    data = {}
    for ds in args.datasets:
        corpus, queries, qrels = load_beir_stdlib(ds)
        if args.limit_docs:
            corpus = dict(list(corpus.items())[: args.limit_docs])
        if args.limit_queries:
            queries = dict(list(queries.items())[: args.limit_queries])
            qrels = {q: rel for q, rel in qrels.items() if q in queries}
        data[ds] = (corpus, queries, qrels)
        print(f"{ds}: {len(corpus)} docs, {len(queries)} queries", file=sys.stderr)

    rows: list[dict] = []
    total_spend = 0.0

    # merge with any previously persisted results so partial runs accumulate
    out_path = Path(args.out)
    if out_path.exists():
        try:
            rows = json.loads(out_path.read_text())
            print(f"[merge] loaded {len(rows)} existing rows from {out_path}", file=sys.stderr)
        except Exception:
            rows = []

    def upsert(row: dict) -> None:
        key = (row["dataset"], row["system"])
        for i, r in enumerate(rows):
            if (r["dataset"], r["system"]) == key:
                rows[i] = row
                return
        rows.append(row)

    for ds in args.datasets:
        corpus, queries, qrels = data[ds]
        doc_ids = list(corpus.keys())
        query_ids = list(queries.keys())

        # ---- NRAG pure-lexical baseline ($0/query)
        if not args.skip_nrag:
            print(f"[nrag] {ds}: indexing + searching {len(query_ids)} queries...", file=sys.stderr)
            rag = Nrag(preset="fast")
            rag.add(list(documents(corpus)))
            scores = evaluate(rag, queries, qrels)
            # "normal FTS" pillar: the same index, but plain word-only BM25 — no char-trigram
            # field, no title boost. What a stock full-text search gives you before NRAG's
            # multi-signal tuning; the v0.1.4 zero-weight legs are skipped entirely.
            fts_scores = evaluate(rag, queries, qrels,
                                  field_weights=FieldWeights(body=1.0, ngram=0.0, title=0.0))
            rag.close()
            upsert({"dataset": ds, "system": "NRAG (pure lexical)", "emb": "✗",
                         "dims": "—", "query_cost": "$0/query", "scores": scores,
                         "spend": 0.0})
            upsert({"dataset": ds, "system": "normal FTS (BM25 word-only)", "emb": "✗",
                         "dims": "—", "query_cost": "$0/query", "scores": fts_scores,
                         "spend": 0.0})
            print(f"  -> {json.dumps(scores)}", file=sys.stderr)
            print(f"  -> FTS {json.dumps(fts_scores)}", file=sys.stderr)

        # ---- dense models
        if not args.skip_dense:
            texts = [(corpus[i]["title"] + "\n\n" + corpus[i]["text"])
                     if corpus[i].get("title") else corpus[i]["text"]
                     for i in doc_ids]
            qtexts = [queries[i] for i in query_ids]
            aborted = False
            # preserve the user's --models order (cheap models can run first)
            by_id = dict(MODELS)
            chosen = [(m, by_id[m]) for m in args.models if m in by_id]
            for mid, short in chosen:
                emb_c, mc = embed_all(mid, texts, key=key, cache=cache, dataset=ds,
                                      cost_cap=args.cost_cap, kind="corpus")
                emb_q, mq = embed_all(mid, qtexts, key=key, cache=cache, dataset=ds,
                                      cost_cap=args.cost_cap, kind="queries")
                spend = mc["cost"] + mq["cost"]
                total_spend += spend
                scores = dense_run(emb_c, emb_q, doc_ids, query_ids, qrels)
                upsert({"dataset": ds, "system": f"{short} (dense)", "emb": "✔",
                             "dims": mc["dim"], "query_cost": "embed/query",
                             "scores": scores, "spend": spend})
                print(f"  -> {short}: {json.dumps(scores)}  (${spend:.4f})", file=sys.stderr)
                if total_spend > args.cost_cap:
                    print(f"ABORT: total spend ${total_spend:.3f} exceeds cap ${args.cost_cap:.2f}",
                          file=sys.stderr)
                    aborted = True
                    break
            if aborted:
                break  # stop after the dataset that blew the cap

    # published-BM25 reference rows
    for ds in args.datasets:
        if ds in PUBLISHED_BM25:
            upsert({"dataset": ds, "system": "BM25 (Anserini, published)", "emb": "✗",
                         "dims": "—", "query_cost": "$0/query",
                         "scores": {"ndcg@10": PUBLISHED_BM25[ds], "recall@10": None,
                                    "recall@100": None, "mrr": None},
                         "spend": 0.0})
    if "scifact" in args.datasets:
        upsert({"dataset": "scifact", "system": "NRAG + doc2query x2 (prior run)",
                     "emb": "✗", "dims": "—", "query_cost": "$0/query",
                     "scores": {"ndcg@10": PRIOR_DOC2QUERY_SCIFACT, "recall@10": None,
                                "recall@100": None, "mrr": 0.7042},
                     "spend": 0.0, "note": "LLM index-time enrichment, cached; from scifact_results.md"})

    out = Path(args.out)
    complete = [r for r in rows if r["scores"].get("ndcg@10") is not None]
    out.write_text(json.dumps(complete, indent=2) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    # report every dataset we have rows for (canonical order, then any extras)
    order = ["scifact", "nfcorpus", "fiqa", "arguana", "trec-covid"]
    present = [d for d in order if any(r["dataset"] == d for r in rows)]
    present += sorted({r["dataset"] for r in rows} - set(order))
    write_report(complete, present, Path(args.report))
    print(f"total spend this run: ${total_spend:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
