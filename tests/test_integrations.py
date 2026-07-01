"""Phase 4: framework retriever adapters — the dependency-free mapping, plus a clear
ImportError when the optional framework isn't installed."""

from __future__ import annotations

import importlib.util

import pytest

from kapi import Kapi
from kapi.integrations import hits_to_records, to_langchain_retriever, to_llamaindex_retriever


def _hydrated_hits():
    rag = Kapi(preset="fast")
    rag.add_texts(["Dijkstra computes shortest paths in a graph.", "Tomato soup needs basil."])
    hits = rag.search("shortest paths graph", k=2)
    rag.close()
    return hits


def test_hits_to_records_shape():
    recs = hits_to_records(_hydrated_hits())
    assert recs
    for r in recs:
        assert {"text", "score", "chunk_id", "source", "metadata"} <= set(r)
        assert isinstance(r["score"], float) and isinstance(r["metadata"], dict)
    assert "Dijkstra" in recs[0]["text"]


@pytest.mark.skipif(importlib.util.find_spec("langchain_core") is not None,
                    reason="langchain-core installed; ImportError path not applicable")
def test_langchain_adapter_raises_without_dep():
    rag = Kapi(preset="fast")
    with pytest.raises(ImportError, match="LangChain"):
        to_langchain_retriever(rag)
    rag.close()


@pytest.mark.skipif(importlib.util.find_spec("llama_index") is not None,
                    reason="llama-index-core installed; ImportError path not applicable")
def test_llamaindex_adapter_raises_without_dep():
    rag = Kapi(preset="fast")
    with pytest.raises(ImportError, match="LlamaIndex"):
        to_llamaindex_retriever(rag)
    rag.close()
