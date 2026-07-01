"""Phase 4 — drop-in retriever adapters (STRATEGY §6): make trying KAPI one line, not a
migration. The heavyweight frameworks (LangChain, LlamaIndex) are *optional* deps — the
adapters import them lazily and raise a clear message if absent, so core KAPI stays lean.

The framework-agnostic part — turning KAPI :class:`~kapi._types.Hit`\\s into plain records —
lives here as :func:`hits_to_records` and is what the per-framework shims map from.
"""

from __future__ import annotations

from typing import Dict, List

from .._types import Hit


def hits_to_records(hits: List[Hit]) -> List[Dict]:
    """Normalize hydrated hits into plain dicts (text/score/source/chunk_id/metadata)."""
    records: List[Dict] = []
    for h in hits:
        records.append({
            "text": h.text,
            "score": float(h.score),
            "chunk_id": h.chunk_id,
            "source": h.source,
            "metadata": dict(h.chunk.metadata) if h.chunk is not None else {},
        })
    return records


def to_langchain_retriever(kapi, *, k=None):
    """Wrap a :class:`~kapi.Kapi` as a LangChain ``BaseRetriever``. Needs ``langchain-core``."""
    from .langchain import to_langchain_retriever as _impl

    return _impl(kapi, k=k)


def to_llamaindex_retriever(kapi, *, k=None):
    """Wrap a :class:`~kapi.Kapi` as a LlamaIndex ``BaseRetriever``. Needs ``llama-index-core``."""
    from .llamaindex import to_llamaindex_retriever as _impl

    return _impl(kapi, k=k)


__all__ = ["hits_to_records", "to_langchain_retriever", "to_llamaindex_retriever"]
