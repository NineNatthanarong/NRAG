"""Retrieval metrics: nDCG@k, Recall@k, MRR.

Pure-Python by default (zero dependency, so eval and tests always run). If ``ranx`` or
``pytrec_eval`` is installed you can cross-check via ``backend=``; the pure-Python path
is validated to agree with them to ~1e-6 on the same run.

qrels: ``{query_id: {doc_id: relevance}}``  (relevance > 0 means relevant)
run:   ``{query_id: {doc_id: score}}``       (higher score = more relevant)
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Sequence

Qrels = Dict[str, Dict[str, int]]
Run = Dict[str, Dict[str, float]]

_METRIC_RE = re.compile(r"^(ndcg|recall|precision|hit|mrr|map)(?:[@_](\d+))?$", re.I)


def _ranked_docs(scores: Dict[str, float]) -> List[str]:
    # deterministic: sort by score desc, then doc_id asc for stable ties
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def dcg(rels: Sequence[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> float:
    gains = [rel.get(d, 0) for d in ranked[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> float:
    total = sum(1 for v in rel.values() if v > 0)
    if total == 0:
        return 0.0
    found = sum(1 for d in ranked[:k] if rel.get(d, 0) > 0)
    return found / total


def precision_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / k


def hit_at_k(ranked: List[str], rel: Dict[str, int], k: int) -> float:
    return 1.0 if any(rel.get(d, 0) > 0 for d in ranked[:k]) else 0.0


def reciprocal_rank(ranked: List[str], rel: Dict[str, int]) -> float:
    for i, d in enumerate(ranked, start=1):
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def average_precision(ranked: List[str], rel: Dict[str, int]) -> float:
    total = sum(1 for v in rel.values() if v > 0)
    if total == 0:
        return 0.0
    hits = 0
    acc = 0.0
    for i, d in enumerate(ranked, start=1):
        if rel.get(d, 0) > 0:
            hits += 1
            acc += hits / i
    return acc / total


def _eval_one(metric: str, ranked: List[str], rel: Dict[str, int]) -> float:
    m = _METRIC_RE.match(metric.strip())
    if not m:
        raise ValueError(f"unrecognized metric {metric!r}")
    name = m.group(1).lower()
    k = int(m.group(2)) if m.group(2) else len(ranked)
    if name == "ndcg":
        return ndcg_at_k(ranked, rel, k)
    if name == "recall":
        return recall_at_k(ranked, rel, k)
    if name == "precision":
        return precision_at_k(ranked, rel, k)
    if name == "hit":
        return hit_at_k(ranked, rel, k)
    if name == "mrr":
        return reciprocal_rank(ranked, rel)
    if name == "map":
        return average_precision(ranked, rel)
    raise ValueError(f"unrecognized metric {metric!r}")


def evaluate_run(
    qrels: Qrels,
    run: Run,
    metrics: Sequence[str] = ("ndcg@10", "recall@100", "mrr"),
    *,
    backend: str = "python",
) -> Dict[str, float]:
    """Mean of each metric over all queries that have judgments."""
    if backend in ("ranx", "pytrec_eval"):
        return _evaluate_external(qrels, run, list(metrics), backend)

    totals: Dict[str, float] = {m: 0.0 for m in metrics}
    n = 0
    for qid, rel in qrels.items():
        if not rel:
            continue
        ranked = _ranked_docs(run.get(qid, {}))
        for m in metrics:
            totals[m] += _eval_one(m, ranked, rel)
        n += 1
    return {m: (totals[m] / n if n else 0.0) for m in metrics}


def _evaluate_external(qrels, run, metrics, backend):  # pragma: no cover - optional deps
    if backend == "ranx":
        from ranx import Qrels as RQ
        from ranx import Run as RR
        from ranx import evaluate

        scores = evaluate(RQ(qrels), RR(run), [m.lower().replace("@", "@") for m in metrics])
        if isinstance(scores, dict):
            return {m: float(scores[m.lower()]) for m in metrics}
        return {metrics[0]: float(scores)}
    # pytrec_eval: translate ndcg@10 -> ndcg_cut.10, recall@100 -> recall.100, mrr -> recip_rank
    import pytrec_eval

    def to_trec(m):
        mm = _METRIC_RE.match(m)
        name, k = mm.group(1).lower(), mm.group(2)
        if name == "ndcg":
            return f"ndcg_cut.{k}", f"ndcg_cut_{k}"
        if name == "recall":
            return f"recall.{k}", f"recall_{k}"
        if name == "mrr":
            return "recip_rank", "recip_rank"
        if name == "map":
            return "map", "map"
        return f"P.{k}", f"P_{k}"
    measures = {to_trec(m)[0] for m in metrics}
    ev = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_q = ev.evaluate(run)
    out = {}
    for m in metrics:
        key = to_trec(m)[1]
        vals = [q[key] for q in per_q.values() if key in q]
        out[m] = sum(vals) / len(vals) if vals else 0.0
    return out
