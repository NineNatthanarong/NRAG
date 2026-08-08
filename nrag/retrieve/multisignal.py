"""The single retrieval entrypoint.

Branches on the engine's capabilities:
  * Path A (``supports_multifield``, e.g. Tantivy): one engine-fused query already
    combines word/ngram/title with field boosts — results pass straight through.
  * Path B (single-field engines, e.g. SQLite/bm25s): one query per signal, fused
    here with RRF.

Then it hydrates hits with full chunk text from the store, applies any metadata filter
the engine couldn't push down, and truncates to ``k``.
"""

from __future__ import annotations

from typing import List, Optional

from .._types import FieldWeights, Hit, MetaFilter
from ..store.metadata import MetadataStore
from . import fuse as _fuse


def search(
    engine,
    store: MetadataStore,
    query: str,
    *,
    k: int = 10,
    candidates: Optional[int] = None,
    field_weights: FieldWeights = FieldWeights(),
    fuzzy: bool = False,
    filter: Optional[MetaFilter] = None,
    fusion: str = "rrf",
    rrf_k: int = 60,
    weights: Optional[list[float]] = None,
) -> List[Hit]:
    if not query or not query.strip():
        return []
    candidates = candidates or max(k, 50)

    # Determine up front whether the engine can push down *every* filter clause.
    # Engines only ever cover a subset of fields (e.g. doc_id/section), so anything
    # else must be post-filtered here after hydration — silently skipping it would
    # make metadata filters a no-op (a correctness/data-isolation bug).
    needs_post_filter = False
    if filter is not None and not filter.is_empty():
        covers = getattr(engine, "prefilter_covers", None)
        fully_covered = bool(covers(filter)) if callable(covers) else False
        needs_post_filter = not fully_covered
        if needs_post_filter:
            # widen the candidate pool so post-filtering can still fill k results
            candidates = max(candidates, k * 10, 100)

    if getattr(engine, "supports_multifield", False):
        hits = engine.search(query, k=candidates, field_weights=field_weights,
                             fuzzy=fuzzy, filter=filter)
    else:
        # Path B: query each signal separately, then fuse. Zero-weight legs are
        # skipped entirely; convex fusion (default) mirrors the configured field
        # weights so single-field engines honor FieldWeights like Tantivy does.
        fw_map = field_weights.as_dict()
        rank_lists, leg_weights = [], []
        for signal in ("body", "ngram", "title"):
            w = fw_map.get(signal, 1.0)
            if w <= 0:
                continue
            sub = engine.search(query, k=candidates, signal=signal, field_weights=field_weights,
                                fuzzy=fuzzy, filter=filter)
            if sub:
                rank_lists.append(sub)
                leg_weights.append(w)
        if fusion == "convex":
            hits = _fuse.fuse(rank_lists, method="convex", weights=list(weights or leg_weights))
        else:
            hits = _fuse.fuse(rank_lists, method="rrf", k=rrf_k, weights=weights)

    hits = store.hydrate(hits)

    # residual post-filter for every clause the engine didn't fully pre-filter
    if needs_post_filter:
        hits = [h for h in hits if h.chunk is not None and filter.matches(h.chunk)]

    for rank, h in enumerate(hits[:k], start=1):
        h.rank = rank
    return hits[:k]
