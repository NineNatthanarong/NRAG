"""Leg B — the consensus-weighted sparse index: dot-product ranking, IDF query weights,
incremental delete, and JSON persistence."""

from __future__ import annotations

from kapi.retrieve.sparse import SparseConsensusIndex
from kapi.tokenize.text import WordTokenizer

_TOK = WordTokenizer("english")


def _vec(text: str, w: float = 1.0) -> dict:
    # build doc vectors through the SAME tokenizer the query side uses (stem-consistent)
    return {t: w for t in _TOK(text)}


def test_dot_product_ranks_overlap_first():
    idx = SparseConsensusIndex(None)
    idx.add("a::0", _vec("weighted graph shortest cheapest route between nodes"))
    idx.add("b::0", _vec("tomato soup basil salt simple recipe"))
    hits = idx.search("cheapest route nodes", k=10)
    assert hits and hits[0].chunk_id == "a::0"
    assert all(h.signal == "csc" for h in hits)


def test_weights_influence_score():
    idx = SparseConsensusIndex(None)
    idx.add("hi", _vec("graph", w=5.0))
    idx.add("lo", _vec("graph", w=1.0))
    hits = idx.search("graph", k=10)
    assert [h.chunk_id for h in hits] == ["hi", "lo"]   # higher consensus weight ranks first


def test_idf_downweights_common_terms():
    idx = SparseConsensusIndex(None)
    # 'common' in both docs (low IDF), 'rare' in one (high IDF)
    idx.add("a::0", {"common": 1.0, "rare": 1.0})
    idx.add("b::0", {"common": 1.0})
    hits = idx.search("common rare", k=10)
    assert hits[0].chunk_id == "a::0"   # the rare-term match dominates via IDF


def test_delete_and_delete_doc():
    idx = SparseConsensusIndex(None)
    idx.add("d1::0", _vec("graph theory"))
    idx.add("d1::1", _vec("graph search"))
    idx.add("d2::0", _vec("cooking"))
    idx.delete_doc("d1")
    assert not idx.search("graph")
    assert idx.search("cooking")


def test_persistence_roundtrip(tmp_path):
    p = str(tmp_path / "idx")
    idx = SparseConsensusIndex(p)
    idx.add("a::0", {"alpha": 2.0, "beta": 0.5})
    idx.commit()

    reopened = SparseConsensusIndex(p)
    assert reopened.vectors["a::0"]["alpha"] == 2.0
    assert reopened.search("alpha", k=5)[0].chunk_id == "a::0"


def test_empty_and_no_match():
    idx = SparseConsensusIndex(None)
    assert idx.search("anything") == []
    idx.add("a::0", {"x": 1.0})
    assert idx.search("nomatch") == []
    assert idx.search("") == []
