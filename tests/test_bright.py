"""BRIGHT runner — exercised offline by monkeypatching the dataset loader. Verifies
excluded-id dropping, chunk->doc max-pooling, and the report format (no network/datasets)."""

from __future__ import annotations

from kapi import Kapi
from kapi.eval import bright


def _fake_loader(subset="biology", *, data_repo="xlangai/BRIGHT"):
    corpus = {
        "d1": {"text": "Apples and oranges are fruit grown on trees."},
        "d2": {"text": "Cars and trucks are vehicles with engines."},
        # the query-source passage: matches 'fruit' strongly, must be EXCLUDED from scoring
        "src": {"text": "fruit fruit fruit query source passage."},
    }
    queries = {"q1": "which items are fruit"}
    qrels = {"q1": {"d1": 1}}
    excluded = {"q1": {"src"}}
    return corpus, queries, qrels, excluded


def test_run_bright_drops_excluded_and_scores(monkeypatch):
    monkeypatch.setattr(bright, "load_bright", _fake_loader)
    report = bright.run_bright(lambda: Kapi(), "biology")    # pure-lexical, no LLM
    assert report.n_docs == 3 and report.n_queries == 1
    # 'src' out-matches 'd1' on raw term frequency but is excluded -> the relevant doc d1
    # is the top scored survivor -> perfect nDCG@10.
    assert report.scores["ndcg@10"] == 1.0


def test_bright_report_str(monkeypatch):
    monkeypatch.setattr(bright, "load_bright", _fake_loader)
    report = bright.run_bright(lambda: Kapi(), "biology")
    s = str(report)
    assert "BRIGHT[biology]" in s
    assert "off-the-shelf dense" in s and "LATTICE" in s


def test_bright_subsets_and_reference_exposed():
    assert "biology" in bright.BRIGHT_SUBSETS and len(bright.BRIGHT_SUBSETS) == 12
    assert bright.BRIGHT_REFERENCE["dense_offtheshelf_avg"] == 18.3
