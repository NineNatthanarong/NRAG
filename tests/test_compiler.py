"""Phase 1 + 2: the offline compiler — bundle parsing, enrichment into indexed_text,
content-hash cache, cost guard, and the CSC consensus weighting + literal anchoring."""

from __future__ import annotations

import re

import pytest

from nrag import Config, Document, Nrag
from nrag._types import Chunk
from nrag.augment.compiler import Compiler
from nrag.results import CostGuardError
from nrag.tokenize.text import WordTokenizer

_TOK = WordTokenizer("english")


class CompilerFakeLLM:
    """Content-aware fake: emits a parseable bundle and maps a few topic tokens to
    *non-literal* inference phrases (so reasoning-expansion retrieval is testable).
    With ``inject_noise`` it adds a hapax term to one in every three samples — exactly
    the hallucination that consensus must suppress."""

    SYN = {
        "shortest": "cheapest route between nodes",
        "reimburse": "money back guarantee for unhappy buyers",
        "logarithmic": "scales to very large inputs efficiently",
    }

    def __init__(self, inject_noise: bool = False):
        self.calls = 0
        self.model_name = "gpt-4o-mini"   # a priced model, so the cost guard is exercisable
        self.inject_noise = inject_noise

    @staticmethod
    def _chunk_of(prompt: str) -> str:
        m = re.search(r"<chunk>\n(.*?)\n</chunk>", prompt, re.S)
        return m.group(1) if m else ""

    def complete(self, prompt, *, max_tokens=None, temperature=0.0, stop=None, system=None):
        self.calls += 1
        chunk = self._chunk_of(prompt).lower()
        lines = [
            "CONTEXT: This chunk is part of a reference document about the topic.",
            "QUESTIONS:",
            "- What does this passage explain?",
            "PROPOSITIONS:",
            "- The passage states a definition.",
            "INFERENCES:",
        ]
        inferred = [phrase for key, phrase in self.SYN.items() if key in chunk]
        lines += [f"- {p}" for p in inferred] or ["- general background knowledge"]
        if self.inject_noise and self.calls % 3 == 1:
            lines.append("- zzqqx hapaxnoise gibberishterm")
        return "\n".join(lines)


def _docs():
    return [
        Document("dijkstra", "Dijkstra computes shortest paths in a weighted graph.",
                 source="dijkstra.md", metadata={"source": "dijkstra.md"}),
        Document("soup", "A simple tomato soup recipe uses tomatoes, basil and salt.",
                 source="soup.md", metadata={"source": "soup.md"}),
    ]


# ---------------------------------------------------------------- Phase 1: enrichment
def test_compile_enriches_indexed_text_only():
    rag = Nrag(llm=CompilerFakeLLM(), preset="compiled")
    rep = rag.compile(_docs())
    assert rep.num_chunks > 0 and rep.contextualized == rep.num_chunks
    chunk = rag.store.all_chunks()[0]
    # enrichment landed in indexed_text; raw_text stays clean for citations
    assert len(chunk.indexed_text) > len(chunk.raw_text)
    assert chunk.indexed_text.endswith(chunk.raw_text)
    assert "cheapest route" not in chunk.raw_text
    rag.close()


def test_compile_is_cached_by_content_hash():
    fake = CompilerFakeLLM()
    rag = Nrag(llm=fake, preset="compiled")
    rag.compile(_docs())
    before = fake.calls
    assert before > 0
    rag.compile(_docs(), force=True)        # force re-ingest, but cache -> no new LLM calls
    assert fake.calls == before
    rag.close()


def test_compile_alias_matches_add():
    rag = Nrag(llm=CompilerFakeLLM(), preset="compiled")
    rep = rag.compile(_docs())
    assert rep.num_docs == 2
    rag.close()


def test_cost_guard_blocks_expensive_compile():
    rag = Nrag(llm=CompilerFakeLLM(), preset="compiled",
               contextual_cost_guard_usd=1e-9)   # absurdly low -> must trip on a priced model
    with pytest.raises(CostGuardError):
        rag.compile(_docs())
    rag.close()


# ---------------------------------------------------------------- Phase 2: CSC weighting
def _compile_one(raw: str, **cfg_over):
    cfg = Config.compiled(compile_concurrency=1, **cfg_over)
    comp = Compiler(CompilerFakeLLM(inject_noise=True), cfg, None)
    chunk = Chunk("d::0", "d", 0, raw, raw)
    bundles = comp.compile([(raw, chunk)])
    return bundles["d::0"]


def test_consensus_suppresses_hallucinations_and_keeps_agreed_terms():
    bundle = _compile_one("Dijkstra computes shortest paths in a weighted graph.",
                          consensus_k=3, consensus_min_agreement=0.5)
    tw = bundle.term_weights
    # expansion term agreed in all 3 samples -> weight 1.0
    cheap = _TOK("cheapest")[0]
    assert tw.get(cheap) == pytest.approx(1.0)
    # hapax noise present in only 1 of 3 samples -> below the 0.5 floor -> dropped
    assert not any("hapaxnoise" in t or "zzqqx" in t or "gibberish" in t for t in tw)


def test_literal_anchoring_floor():
    bundle = _compile_one("Dijkstra computes shortest paths in a weighted graph.",
                          consensus_k=3, literal_floor=1.0)
    tw = bundle.term_weights
    graph = _TOK("graph")[0]          # a source-literal term the LLM never emitted
    assert tw.get(graph) == pytest.approx(1.0)   # anchored at the floor, never dropped


# ---------------------------------------------------------------- two-leg retrieval
def test_reasoning_expansion_retrieves_via_non_literal_term():
    rag = Nrag(llm=CompilerFakeLLM(), preset="compiled")
    rag.compile(_docs())
    # "cheapest route" appears in NEITHER raw text — only in the compiled inferential closure
    hits = rag.search("cheapest route between nodes", k=2)
    assert hits and hits[0].chunk.doc_id == "dijkstra"
    assert rag.stats()["csc"]["num_chunks"] > 0
    rag.close()


def test_exact_match_still_wins_in_compiled_mode():
    rag = Nrag(llm=CompilerFakeLLM(), preset="compiled")
    rag.compile(_docs())
    hits = rag.search("tomato soup basil", k=2)
    assert hits and hits[0].chunk.doc_id == "soup"
    rag.close()


# ---------------------------------------------------------------- degradation + persistence
def test_no_llm_disables_compiler():
    rag = Nrag(preset="compiled")            # no LLM
    assert not rag.config.compile_enabled and not rag.config.csc_enabled
    assert rag._legb is None
    rag.add(_docs())
    assert rag.search("tomato soup")          # pure lexical still works
    rag.close()


def test_compile_once_serve_without_llm(tmp_path):
    idx = str(tmp_path / "idx")
    rag = Nrag(llm=CompilerFakeLLM(), preset="compiled", path=idx)
    rag.compile(_docs())
    rag.close()

    reopened = Nrag.open(idx)                  # reopen with NO llm
    assert reopened._legb is not None and reopened._legb.vectors
    hits = reopened.search("cheapest route between nodes", k=2)
    assert hits and hits[0].chunk.doc_id == "dijkstra"   # Leg B serves model-free
    reopened.close()


# ---------------------------------------------------------------- compile flags
def test_compile_flags_control_sections():
    """Every ``compile_*`` flag must actually gate its section (regression:
    ``compile_reasoning=False`` was a no-op because the compiler looked up a
    non-existent ``compile_inferences`` attribute)."""
    llm = CompilerFakeLLM()

    assert Compiler(llm, Config.compiled(), None)._sections == [
        "blurb", "questions", "propositions", "inferences"]

    c = Compiler(llm, Config.compiled(compile_reasoning=False), None)
    assert "inferences" not in c._sections

    c = Compiler(llm, Config.compiled(compile_blurb=False), None)
    assert "blurb" not in c._sections and "inferences" in c._sections

    c = Compiler(llm, Config.compiled(compile_questions=False, compile_propositions=False), None)
    assert c._sections == ["blurb", "inferences"]


def test_compile_reasoning_off_removes_inferences_from_prompt_and_cache_key():
    llm = CompilerFakeLLM()
    on = Compiler(llm, Config.compiled(), None)
    off = Compiler(llm, Config.compiled(compile_reasoning=False), None)
    chunk = Chunk(chunk_id="d::0", doc_id="d", ordinal=0,
                  raw_text="Dijkstra computes shortest paths.",
                  indexed_text="Dijkstra computes shortest paths.")
    assert on._key(chunk) != off._key(chunk)  # flag participates in the cache key
