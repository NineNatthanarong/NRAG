"""Phase 3: the adaptive query router — cheap by default, escalate only on weak/ambiguous
results, and never on short (precise) queries (the §2.4 precision-trap guard)."""

from __future__ import annotations

from kapi import Config, Kapi
from kapi._types import Hit
from kapi.retrieve.router import assess


def _h(score: float) -> Hit:
    return Hit("c", score)


_CFG = Config.compiled()   # router on; defaults: short<=4 words, min_hits=2, min_margin=0.12


# ---------------------------------------------------------------- assess() unit tests
def test_no_hits_escalates():
    assert assess([], "a reasonably long natural language query", _CFG).reason == "no_hits"


def test_short_query_never_escalates_even_on_a_tie():
    d = assess([_h(1.0), _h(0.99)], "refund now", _CFG)     # 2 words -> precise, leave alone
    assert not d.escalate and d.reason == "confident"


def test_long_query_low_margin_escalates():
    d = assess([_h(1.0), _h(0.99)], "how do i get my money back please", _CFG)
    assert d.escalate and d.reason == "low_margin"


def test_long_query_dominant_top_is_confident():
    d = assess([_h(1.0), _h(0.1)], "how do i get my money back please", _CFG)
    assert not d.escalate and d.reason == "confident"


def test_long_query_low_recall_escalates():
    d = assess([_h(1.0)], "how do i get my money back please", _CFG)   # 1 hit < min_hits
    assert d.escalate and d.reason == "low_recall"


# ---------------------------------------------------------------- end-to-end gating
class _CountingLLM:
    """Counts query-time calls; expands to bridging vocabulary so escalation actually helps."""

    model_name = "gpt-4o-mini"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, max_tokens=None, temperature=0.0, stop=None, system=None):
        self.calls += 1
        return "You can get a full refund under the reimbursement and return policy."


_DOCS = [
    "The reimbursement policy lets customers return items within thirty days for a full refund.",
    "Shipping times vary by region and the selected carrier.",
    "Account passwords must be at least twelve characters long.",
]


def _router_kapi(llm):
    # fast preset = no index-time LLM work; force the router on so only query-time calls count.
    # ngram off so char-trigrams can't substring-match coined tokens (keeps "no overlap" real).
    rag = Kapi(llm=llm, preset="fast", router_enabled=True, enable_ngram=False)
    rag.add_texts(_DOCS)
    return rag


def test_easy_query_does_not_escalate_no_llm_call():
    llm = _CountingLLM()
    rag = _router_kapi(llm)
    hits = rag.search("refund", k=3)          # short, precise, strong lexical hit
    assert hits and "refund" in hits[0].text
    assert rag.last_route is not None and not rag.last_route.escalate
    assert llm.calls == 0                      # the cheap path stayed cheap ($0)
    rag.close()


def test_hard_query_escalates_and_recovers():
    llm = _CountingLLM()
    rag = _router_kapi(llm)
    # A long query whose (deliberately coined) tokens overlap NO document -> first pass is
    # empty -> the router escalates; expansion then bridges to the real vocabulary.
    hits = rag.search("zqxrefundless zqxmoneyback zqxregretful zqxpurchasing zqxcashback", k=3)
    assert rag.last_route is not None and rag.last_route.escalate
    assert rag.last_route.reason == "no_hits"
    assert llm.calls >= 1                        # escalated -> exactly the expensive path fired
    assert hits and "refund" in hits[0].text     # expansion bridged the vocab gap to doc 1
    rag.close()


def test_no_llm_disables_router():
    rag = Kapi(preset="compiled")               # compiled enables router, but no LLM
    assert rag.config.router_enabled is False   # for_no_llm() turned it off
    rag.add_texts(_DOCS)
    assert rag.search("refund", k=2)            # pure lexical still works
    rag.close()
