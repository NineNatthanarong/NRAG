"""LangChain retriever adapter (Phase 4). ``langchain-core`` is an optional dependency."""

from __future__ import annotations

from . import hits_to_records


def to_langchain_retriever(kapi, *, k=None):
    """Return a LangChain ``BaseRetriever`` backed by ``kapi.search``.

    Usage::

        from kapi.integrations import to_langchain_retriever
        retriever = to_langchain_retriever(rag, k=5)
        chain = create_retrieval_chain(retriever, ...)   # standard LangChain from here
    """
    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
    except ImportError as exc:                       # pragma: no cover - exercised only w/o dep
        raise ImportError(
            "LangChain is not installed. `pip install langchain-core` to use "
            "to_langchain_retriever()."
        ) from exc

    class KapiRetriever(BaseRetriever):
        """LangChain retriever that delegates to a KAPI index (no embeddings, no vector DB)."""

        def _get_relevant_documents(self, query, *, run_manager=None):   # noqa: D401
            return [
                Document(
                    page_content=r["text"],
                    metadata={**r["metadata"], "score": r["score"],
                              "source": r["source"], "chunk_id": r["chunk_id"]},
                )
                for r in hits_to_records(kapi.search(query, k=k))
            ]

    return KapiRetriever()
