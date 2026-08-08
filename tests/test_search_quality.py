"""Tests for the v0.1.4 retrieval-quality work: tuned defaults, weighted convex
fusion on single-field engines, bm25s k1/b passthrough, and document-level search."""

from __future__ import annotations

import pytest

from nrag import Config, Document, Nrag
from nrag._types import FieldWeights


def test_tuned_default_field_weights():
    """Defaults must match the benchmarked winners (BEIR grid, benchmarks/)."""
    fw = FieldWeights()
    assert (fw.body, fw.ngram, fw.title) == (1.0, 0.3, 0.5)
    cfg = Config()
    assert (cfg.weight_body, cfg.weight_ngram, cfg.weight_title) == (1.0, 0.3, 0.5)
    assert cfg.fusion == "convex"


def test_csc_two_leg_fusion_still_rrf_by_default():
    """The compiled preset's two-leg fusion keeps its validated RRF setting even
    though single-field-engine fusion moved to convex."""
    assert Config.compiled().csc_fusion == "rrf"


def _corpus():
    return [
        Document(doc_id="graphs", source="graphs.md",
                 text="Dijkstra's algorithm computes shortest paths in weighted graphs. "
                      "It uses a priority queue for efficiency."),
        Document(doc_id="cooking", source="cooking.md",
                 text="Tomato soup needs basil, salt, and ripe tomatoes. "
                      "Simmer gently for twenty minutes."),
    ]


@pytest.mark.parametrize("engine", ["tantivy", "sqlite"])
def test_convex_fusion_returns_relevant_results(engine):
    rag = Nrag(preset="fast", engine=engine)  # fusion="convex" default
    rag.add(_corpus())
    hits = rag.search("shortest path algorithm", k=2)
    assert hits and hits[0].chunk.doc_id == "graphs"
    rag.close()


def test_zero_weight_leg_is_skipped(monkeypatch):
    """With ngram weight 0, the ngram signal must never be queried on Path B."""
    rag = Nrag(preset="fast", engine="sqlite", weight_ngram=0.0)
    rag.add(_corpus())
    seen = []
    orig = rag.engine.search

    def spy(query, **kw):
        seen.append(kw.get("signal", "body"))
        return orig(query, **kw)

    monkeypatch.setattr(rag.engine, "search", spy)
    rag.search("tomato soup", k=1)
    assert "ngram" not in seen and "body" in seen
    rag.close()


def test_bm25_params_flow_to_engine_config():
    rag = Nrag(preset="fast", bm25_k1=0.9, bm25_b=0.4)
    assert rag._engine_config.bm25_k1 == 0.9
    assert rag._engine_config.bm25_b == 0.4
    rag.close()


def test_bm25s_engine_uses_k1_b():
    pytest.importorskip("bm25s")
    rag = Nrag(preset="fast", engine="bm25s", bm25_k1=0.9, bm25_b=0.4)
    rag.add(_corpus())
    assert rag.engine._retriever.k1 == pytest.approx(0.9)
    assert rag.engine._retriever.b == pytest.approx(0.4)
    hits = rag.search("shortest path", k=1)
    assert hits and hits[0].chunk.doc_id == "graphs"
    rag.close()


# ---------------------------------------------------------------- search_docs
def _long_corpus():
    graph_text = " ".join(
        f"Section {i}: graph algorithms and shortest path routing details part {i}."
        for i in range(40)
    )
    return [
        Document(doc_id="long_graphs", source="long.md", text=graph_text),
        Document(doc_id="cooking", source="cooking.md",
                 text="Tomato soup needs basil and salt."),
    ]


def test_search_docs_one_hit_per_document():
    rag = Nrag(preset="fast")
    rag.add(_long_corpus())
    docs = rag.search_docs("shortest path routing", k=5)
    doc_ids = [h.chunk.doc_id for h in docs]
    assert len(doc_ids) == len(set(doc_ids)), "one hit per document expected"
    assert doc_ids[0] == "long_graphs"
    assert all(h.signal == "doc" for h in docs)
    assert [h.rank for h in docs] == list(range(1, len(docs) + 1))
    rag.close()


def test_search_docs_agg_modes_and_validation():
    rag = Nrag(preset="fast")
    rag.add(_long_corpus())
    mx = rag.search_docs("graph algorithms", k=2, agg="max")
    sm = rag.search_docs("graph algorithms", k=2, agg="sum")
    assert mx and sm and mx[0].chunk.doc_id == sm[0].chunk.doc_id == "long_graphs"
    assert sm[0].score >= mx[0].score  # sum over chunks >= best chunk
    with pytest.raises(ValueError, match="unknown agg"):
        rag.search_docs("x", agg="mean")
    rag.close()


def test_stdlib_beir_loader_parses_format(tmp_path):
    """The dependency-free BEIR loader must parse the on-disk format correctly."""
    import json

    root = tmp_path / "toy"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "T", "text": "body"}) + "\n")
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "a query"}) + "\n"
        + json.dumps({"_id": "q2", "text": "unjudged"}) + "\n")
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n")

    from nrag.eval.beir import _load_beir_stdlib

    corpus, queries, qrels = _load_beir_stdlib("toy", "test", str(tmp_path))
    assert corpus == {"d1": {"title": "T", "text": "body"}}
    assert queries == {"q1": "a query"}  # unjudged queries filtered out
    assert qrels == {"q1": {"d1": 1}}
