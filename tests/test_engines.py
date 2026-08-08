from __future__ import annotations

import pytest

from nrag import MetaFilter, Nrag

ENGINES = ["tantivy", "sqlite", "bm25s"]


def _top_doc_ids(hits):
    return {h.chunk.doc_id for h in hits if h.chunk}


@pytest.fixture(params=ENGINES)
def rag(request, corpus_docs):
    try:
        r = Nrag(engine=request.param)  # no LLM -> pure lexical
    except RuntimeError as exc:  # optional engine missing
        pytest.skip(str(exc))
    r.add(corpus_docs)
    yield r
    r.close()


def test_topk_retrieval_quality(rag, queries):
    engine = rag.config.engine
    hits_at_3 = 0
    for q, expected in queries:
        # bm25s is word-only: skip the pure-typo query that needs the ngram signal
        if engine == "bm25s" and "recurssion" in q:
            continue
        top = rag.search(q, k=3)
        if expected in _top_doc_ids(top):
            hits_at_3 += 1
    # strong lexical overlap -> should nail almost all
    assert hits_at_3 >= len(queries) - 2


def test_delete_doc_removes_chunks(rag):
    before = len(rag.search("photosynthesis sunlight plants", k=5))
    assert before >= 1
    rag.remove("photosynthesis")
    after = _top_doc_ids(rag.search("photosynthesis sunlight plants", k=5))
    assert "photosynthesis" not in after


def test_metadata_filter(rag):
    hits = rag.search("recursion", k=10, filter=MetaFilter(equals={"doc_id": "recursion"}))
    assert hits and all(h.chunk.doc_id == "recursion" for h in hits)


def test_typo_tolerance_default_engine(corpus_docs):
    rag = Nrag(engine="tantivy")
    rag.add(corpus_docs)
    top = _top_doc_ids(rag.search("recurssion", k=3))
    assert "recursion" in top
    rag.close()


def test_ngram_regression_when_disabled(corpus_docs):
    # disabling the ngram signal should drop the typo match
    rag = Nrag(engine="tantivy", enable_ngram=False)
    rag.add(corpus_docs)
    top = _top_doc_ids(rag.search("recurssion", k=3))
    assert "recursion" not in top
    rag.close()
