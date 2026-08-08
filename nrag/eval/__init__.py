"""Evaluation: IR metrics (pure-Python) + optional BEIR / RAGAS runners."""

from __future__ import annotations

from .ir_metrics import (
    evaluate_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "evaluate_answers",
    "evaluate_run",
    "load_beir",
    "load_bright",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_beir",
    "run_bright",
    "run_bright_all",
]


def __getattr__(name):
    if name in ("run_beir", "load_beir", "BeirReport", "PUBLISHED_BM25"):
        from . import beir

        return getattr(beir, name)
    if name in ("run_bright", "run_bright_all", "load_bright", "BrightReport",
                "BRIGHT_SUBSETS", "BRIGHT_REFERENCE"):
        from . import bright

        return getattr(bright, name)
    if name in ("evaluate_answers",):
        from .ragas_runner import evaluate_answers

        return evaluate_answers
    raise AttributeError(name)
