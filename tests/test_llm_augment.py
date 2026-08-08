from __future__ import annotations

from nrag import Config, Nrag
from nrag._types import Chunk, Hit
from nrag.augment.expand import QueryExpander
from nrag.generate.answer import Generator
from nrag.llm import CallableLLM


# ---------------------------------------------------------------- no-LLM degradation
def test_no_llm_pure_lexical(corpus_docs):
    rag = Nrag()  # no llm
    assert not rag.config.contextual_enabled
    assert not rag.config.expand_enabled
    assert not rag.config.generate_enabled
    rag.add(corpus_docs)
    res = rag.query("programming language", k=2)
    assert res.answer is None
    assert res.hits and res.citations  # hits + citations still returned
    rag.close()


# ---------------------------------------------------------------- contextual indexing
def test_contextual_indexing_and_cache(corpus_docs, fake_llm):
    rag = Nrag(llm=fake_llm)  # quality preset
    rep = rag.add(corpus_docs)
    assert rep.contextualized == rep.num_chunks
    assert fake_llm.calls["contextual"] == rep.num_chunks

    # force re-index: blurb cache (keyed by content hash) -> zero new LLM calls
    before = fake_llm.calls["contextual"]
    rag.add(corpus_docs, force=True)
    assert fake_llm.calls["contextual"] == before
    rag.close()


def test_fast_preset_disables_enhancers(fake_llm):
    rag = Nrag(llm=fake_llm, preset="fast")
    assert not rag.config.contextual_enabled
    assert not rag.config.expand_enabled
    assert rag.config.generate_enabled  # generation still on
    rag.close()


# ---------------------------------------------------------------- query expansion
def test_expand_auto_routes_and_repeats(fake_llm):
    exp = QueryExpander(fake_llm, Config.quality())
    short = exp.expand("money back")
    assert short.mode == "cot_keywords"
    assert short.assembled.count("money back") >= 5  # query repeated ~5x

    long = exp.expand("how can a customer get their money back after a purchase")
    assert long.mode == "query2doc"


def test_expand_graceful_on_llm_error():
    def boom(prompt, **kw):
        raise RuntimeError("llm down")

    exp = QueryExpander(CallableLLM(boom, accepts_kwargs=True), Config.quality())
    out = exp.expand("anything")
    assert out.assembled == "anything" and out.mode == "none"


# ---------------------------------------------------------------- generation
def test_generation_citation_mapping(fake_llm):
    gen = Generator(fake_llm, Config.quality())
    hits = [
        Hit(chunk_id="a::0", score=2.0, rank=1,
            chunk=Chunk("a::0", "a", 0, "alpha text", "alpha text",
                        metadata={"source": "a.md"})),
        Hit(chunk_id="b::0", score=1.0, rank=2,
            chunk=Chunk("b::0", "b", 0, "beta text", "beta text",
                        metadata={"source": "b.md"})),
    ]
    ans = gen.answer("question?", hits)
    assert "[1]" in ans.text
    # only [1] used -> one citation, mapped to the first source
    assert [c.marker for c in ans.citations] == ["[1]"]
    assert ans.citations[0].source == "a.md"
    assert len(ans.all_context) == 2


def test_query_uses_expansion_to_fix_vocab_mismatch(fake_llm):
    # 'money back' should reach the refund doc via expansion (+contextual)
    rag = Nrag(llm=fake_llm)
    rag.add([d for d in _refund_corpus()])
    res = rag.query("money back", k=3)
    assert any(h.chunk.doc_id == "refund" for h in res.hits)
    assert res.answer is not None
    rag.close()


def _refund_corpus():
    from nrag._types import Document

    data = [
        ("refund", "The store reimburses purchases to the original card after approval."),
        ("python", "Python is a programming language."),
        ("fox", "The quick brown fox jumps over the lazy dog."),
    ]
    return [Document(doc_id=i, text=t, source=f"{i}.md",
                     metadata={"content_type": "text", "source": f"{i}.md"})
            for i, t in data]
