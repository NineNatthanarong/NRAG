"""Regression tests: metadata filters must be enforced on every engine.

Bug fixed: Tantivy/SQLite advertise ``supports_prefilter=True`` but only push down
``doc_id``/``section`` clauses. Filters on arbitrary metadata keys or ``mtime_after``
were silently ignored (both documents returned when filtering for one team), which is
a correctness / data-isolation defect. The retrieve layer now asks the engine whether
a filter is *fully* covered (``prefilter_covers``) and post-filters otherwise.
"""

from __future__ import annotations

import pytest

from nrag import Document, MetaFilter, Nrag

ENGINES = ["tantivy", "sqlite"]
try:  # optional extra
    import bm25s  # noqa: F401

    ENGINES.append("bm25s")
except ImportError:
    pass


def _rag(engine: str) -> Nrag:
    rag = Nrag(preset="fast", engine=engine)
    rag.add(
        [
            Document(
                doc_id="a",
                text="alpha document about refunds and billing policy",
                source="a.md",
                metadata={"team": "billing", "mtime": 100.0},
            ),
            Document(
                doc_id="b",
                text="beta document about refunds and support policy",
                source="b.md",
                metadata={"team": "support", "mtime": 200.0},
            ),
        ]
    )
    return rag


def _doc_ids(hits):
    return {h.chunk.doc_id for h in hits}


@pytest.mark.parametrize("engine", ENGINES)
def test_custom_metadata_equals_is_enforced(engine):
    rag = _rag(engine)
    try:
        hits = rag.search("refunds", k=10, filter=MetaFilter(equals={"team": "billing"}))
        assert hits, "filtered search should still return the matching doc"
        assert _doc_ids(hits) == {"a"}
    finally:
        rag.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_custom_metadata_any_of_is_enforced(engine):
    rag = _rag(engine)
    try:
        hits = rag.search("refunds", k=10, filter=MetaFilter(any_of={"team": ["support"]}))
        assert hits and _doc_ids(hits) == {"b"}
    finally:
        rag.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_mtime_after_is_enforced(engine):
    rag = _rag(engine)
    try:
        hits = rag.search("refunds", k=10, filter=MetaFilter(mtime_after=150.0))
        assert hits and _doc_ids(hits) == {"b"}
    finally:
        rag.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_doc_id_filter_still_works(engine):
    rag = _rag(engine)
    try:
        hits = rag.search("refunds", k=10, filter=MetaFilter(equals={"doc_id": "a"}))
        assert hits and _doc_ids(hits) == {"a"}
    finally:
        rag.close()


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_filter_matches_everything(engine):
    rag = _rag(engine)
    try:
        hits = rag.search("refunds", k=10, filter=MetaFilter())
        assert _doc_ids(hits) == {"a", "b"}
    finally:
        rag.close()


def test_prefilter_covers_semantics_tantivy():
    from nrag.engine.tantivy_engine import TantivyEngine

    eng = TantivyEngine.open(None)
    try:
        assert eng.prefilter_covers(None)
        assert eng.prefilter_covers(MetaFilter())
        assert eng.prefilter_covers(MetaFilter(equals={"doc_id": "x"}))
        assert eng.prefilter_covers(MetaFilter(equals={"section": "s"}, any_of={"doc_id": ["a"]}))
        assert not eng.prefilter_covers(MetaFilter(equals={"team": "billing"}))
        assert not eng.prefilter_covers(MetaFilter(mtime_after=1.0))
        assert not eng.prefilter_covers(
            MetaFilter(equals={"doc_id": "x", "team": "billing"})
        )
    finally:
        eng.close()


def test_prefilter_covers_semantics_sqlite():
    from nrag.engine.sqlite_engine import SQLiteFTS5Engine

    eng = SQLiteFTS5Engine.open(None)
    try:
        assert eng.prefilter_covers(MetaFilter(equals={"doc_id": "x"}))
        assert not eng.prefilter_covers(MetaFilter(equals={"team": "billing"}))
        assert not eng.prefilter_covers(MetaFilter(mtime_after=1.0))
    finally:
        eng.close()
