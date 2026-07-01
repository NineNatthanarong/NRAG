"""Phase 4 — the TCO calculator (STRATEGY §2.6, §6): the "$0 query cost" pitch, quantified.

Compiled Retrieval's cost story is not a vibe, it is arithmetic: move the smart compute to a
*one-time* offline compile, and query time becomes pure lexical — **$0 per query, no vector
RAM**. A dense + vector-DB stack pays the opposite bill: a cheaper one-time index, but a
*recurring* one — an embedding call on every query, plus RAM to hold the vectors resident.

This module makes that comparison concrete and honest. It is a transparent model, not a
benchmark: every rate is an overridable input (defaults cite §2.1/§2.6), and the output is a
full breakdown plus the break-even horizon, so a reader can plug in their own numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

_PER_MILLION = 1_000_000.0
_REF_DIMS = 1536          # the dimension the §2.6 RAM figure (6.1 GB / 1M docs) is quoted at


@dataclass(frozen=True)
class TCOInputs:
    # ---- corpus & traffic ----
    n_docs: int = 1_000_000
    tokens_per_doc: int = 500
    queries_per_month: int = 1_000_000
    tokens_per_query: int = 20
    months: int = 12

    # ---- KAPI (Compiled Retrieval) — one-time compile, $0 queries ----
    consensus_k: int = 3
    # KAPI's recommended path is a small/local compiler (§9) — effectively electricity-only.
    # A frontier hosted model with prompt caching is ~$1.02/1M (§2.1); override for that case.
    compile_cost_per_1m_tokens: float = 0.10

    # ---- dense + vector DB — cheap index, recurring queries + RAM ----
    embed_cost_per_1m_tokens: float = 0.02        # low end of §2.6
    embedding_dims: int = 1536
    vector_ram_gb_per_1m_docs: float = 6.1        # §2.6, at 1536-d
    ram_cost_per_gb_month: float = 5.0            # typical managed-RAM $/GB-month
    vectordb_flat_monthly: float = 0.0            # optional fixed hosting fee


@dataclass(frozen=True)
class TCOResult:
    months: int
    kapi_index_one_time: float
    dense_index_one_time: float
    dense_recurring_monthly: float
    kapi_total: float
    dense_total: float
    savings: float
    savings_pct: float
    breakeven_months: float                        # inf if dense never catches up in the horizon
    breakdown: Dict[str, float] = field(default_factory=dict)


def compute_tco(inp: TCOInputs) -> TCOResult:
    corpus_tokens = inp.n_docs * inp.tokens_per_doc

    # KAPI: one cached offline pass per chunk, sampled consensus_k times; queries are lexical.
    kapi_index = corpus_tokens * inp.consensus_k * inp.compile_cost_per_1m_tokens / _PER_MILLION
    kapi_query_monthly = 0.0
    kapi_storage_monthly = 0.0                      # no resident vectors — index lives on disk
    kapi_total = kapi_index + (kapi_query_monthly + kapi_storage_monthly) * inp.months

    # Dense: embed the corpus once, then embed every query forever, and keep vectors in RAM.
    dense_index = corpus_tokens * inp.embed_cost_per_1m_tokens / _PER_MILLION
    query_tokens_monthly = inp.queries_per_month * inp.tokens_per_query
    dense_query_monthly = query_tokens_monthly * inp.embed_cost_per_1m_tokens / _PER_MILLION
    ram_gb = (inp.n_docs / _PER_MILLION) * inp.vector_ram_gb_per_1m_docs * \
        (inp.embedding_dims / _REF_DIMS)
    dense_ram_monthly = ram_gb * inp.ram_cost_per_gb_month
    dense_recurring = dense_query_monthly + dense_ram_monthly + inp.vectordb_flat_monthly
    dense_total = dense_index + dense_recurring * inp.months

    savings = dense_total - kapi_total
    savings_pct = (savings / dense_total * 100.0) if dense_total > 0 else 0.0

    # Break-even: KAPI is flat after its one-time compile; dense grows linearly. Solve for the
    # month where dense catches up to KAPI's (higher) one-time cost.
    if dense_recurring > 0:
        breakeven = max(0.0, (kapi_total - dense_index) / dense_recurring)
    else:
        breakeven = float("inf")            # no recurring cost -> dense never overtakes

    return TCOResult(
        months=inp.months,
        kapi_index_one_time=kapi_index,
        dense_index_one_time=dense_index,
        dense_recurring_monthly=dense_recurring,
        kapi_total=kapi_total,
        dense_total=dense_total,
        savings=savings,
        savings_pct=savings_pct,
        breakeven_months=breakeven,
        breakdown={
            "corpus_tokens": float(corpus_tokens),
            "kapi_index_one_time": kapi_index,
            "kapi_query_monthly": kapi_query_monthly,
            "dense_index_one_time": dense_index,
            "dense_query_monthly": dense_query_monthly,
            "dense_ram_gb": ram_gb,
            "dense_ram_monthly": dense_ram_monthly,
            "dense_recurring_monthly": dense_recurring,
        },
    )


def format_report(inp: TCOInputs, res: TCOResult) -> str:
    be = ("never" if res.breakeven_months == float("inf")
          else f"~{res.breakeven_months:.1f} months")
    return "\n".join([
        f"KAPI vs dense+vectorDB TCO over {res.months} months",
        f"  corpus: {inp.n_docs:,} docs x {inp.tokens_per_doc} tok | "
        f"traffic: {inp.queries_per_month:,} queries/mo x {inp.tokens_per_query} tok",
        "  ---------------------------------------------------------------",
        f"  KAPI   one-time compile : ${res.kapi_index_one_time:,.2f}",
        f"  KAPI   per-query cost    : $0.00   (lexical, no model, no vector RAM)",
        f"  KAPI   {res.months}-month total   : ${res.kapi_total:,.2f}",
        "  ---------------------------------------------------------------",
        f"  dense  one-time embed    : ${res.dense_index_one_time:,.2f}",
        f"  dense  recurring/month   : ${res.dense_recurring_monthly:,.2f}"
        f"  (query ${res.breakdown['dense_query_monthly']:,.2f} + "
        f"RAM ${res.breakdown['dense_ram_monthly']:,.2f} for "
        f"{res.breakdown['dense_ram_gb']:.1f} GB)",
        f"  dense  {res.months}-month total   : ${res.dense_total:,.2f}",
        "  ---------------------------------------------------------------",
        f"  savings with KAPI        : ${res.savings:,.2f}  ({res.savings_pct:.1f}%)",
        f"  dense overtakes KAPI at  : {be}",
    ])
