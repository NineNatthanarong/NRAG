"""Phase 3 — the adaptive query router (STRATEGY §5 Pillar 5, §2.4).

The only query-time LLM use in Compiled Retrieval, and it is *gated*. The first lexical pass
is ~1 ms and $0; most queries (exact / precise) end there. Only when a **cheap confidence
signal** says the result set is weak or ambiguous do we spend one LLM call to escalate
(query expansion + a second search). This is the Adaptive-RAG / TARG pattern: recover most of
the always-on quality at a fraction of the cost — and, crucially, dodge the §2.4 *precision
trap* (expansion helps weak retrieval but hurts precise queries) by construction.

The signal is deliberately model-free and explainable:

  * **no_hits**    — the first pass found nothing → escalate (nothing to lose).
  * **low_top_score** — the top score is below an (optional, engine-specific) absolute floor.
  * **low_recall** — a *long* query returned fewer than ``router_min_hits`` → under-recall.
  * **low_margin** — a *long* query's top hit does not lead the 2nd by ``router_min_margin``
                     (relative) → the ranking is ambiguous.

Short queries are treated as precise exact-match and are **never** escalated regardless of
margin — that is the precision-trap guard. Thresholds live on :class:`~kapi.config.Config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .._types import Hit
from ..config import Config


@dataclass(frozen=True)
class RouterDecision:
    escalate: bool
    reason: str
    top_score: float = 0.0
    margin: float = 1.0
    n_hits: int = 0


def assess(hits: List[Hit], query: str, cfg: Config) -> RouterDecision:
    """Decide — from the first lexical pass alone — whether to spend an LLM escalation."""
    n = len(hits)
    if n == 0:
        return RouterDecision(True, "no_hits", 0.0, 0.0, 0)

    s1 = hits[0].score
    if cfg.router_min_top_score > 0 and s1 < cfg.router_min_top_score:
        return RouterDecision(True, "low_top_score", s1, 0.0, n)

    margin = ((s1 - hits[1].score) / s1) if (n > 1 and s1 > 0) else 1.0

    # Short queries are exact-match territory (BM25's home turf): leave them untouched even
    # if the margin is thin — expanding them is the precision trap (§2.4).
    long_query = len(query.split()) > cfg.router_short_query_words
    if long_query:
        if n < cfg.router_min_hits:
            return RouterDecision(True, "low_recall", s1, margin, n)
        if margin < cfg.router_min_margin:
            return RouterDecision(True, "low_margin", s1, margin, n)

    return RouterDecision(False, "confident", s1, margin, n)
