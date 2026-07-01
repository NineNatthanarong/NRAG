"""LlamaIndex retriever adapter (Phase 4). ``llama-index-core`` is an optional dependency."""

from __future__ import annotations

from . import hits_to_records


def to_llamaindex_retriever(kapi, *, k=None):
    """Return a LlamaIndex ``BaseRetriever`` backed by ``kapi.search``.

    Usage::

        from kapi.integrations import to_llamaindex_retriever
        retriever = to_llamaindex_retriever(rag, k=5)
        query_engine = RetrieverQueryEngine.from_args(retriever)   # standard LlamaIndex
    """
    try:
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import NodeWithScore, TextNode
    except ImportError as exc:                       # pragma: no cover - exercised only w/o dep
        raise ImportError(
            "LlamaIndex is not installed. `pip install llama-index-core` to use "
            "to_llamaindex_retriever()."
        ) from exc

    class KapiRetriever(BaseRetriever):
        """LlamaIndex retriever that delegates to a KAPI index (no embeddings, no vector DB)."""

        def _retrieve(self, query_bundle):
            query = getattr(query_bundle, "query_str", None) or str(query_bundle)
            nodes = []
            for r in hits_to_records(kapi.search(query, k=k)):
                node = TextNode(text=r["text"], id_=r["chunk_id"],
                                metadata={**r["metadata"], "source": r["source"]})
                nodes.append(NodeWithScore(node=node, score=r["score"]))
            return nodes

    return KapiRetriever()
