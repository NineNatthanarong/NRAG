"""Optional end-to-end answer evaluation via RAGAS (faithfulness, answer relevancy, ...).

Runs questions through an ``Nrag`` instance and scores the generated answers with RAGAS,
using the LLM you plugged in (wrapped for RAGAS) unless you pass an explicit evaluator
LLM. Requires:  pip install nrag[eval]
"""

from __future__ import annotations

from typing import List, Optional


def _wrap_nrag_llm_for_ragas(nrag_llm):
    """Best-effort adapter: wrap an Nrag LLM as a LangChain LLM for RAGAS.

    RAGAS is version-sensitive; if this fails, pass ``ragas_llm`` explicitly.
    """
    from langchain_core.language_models.llms import LLM as LCBaseLLM  # type: ignore
    from ragas.llms import LangchainLLMWrapper  # type: ignore

    class _LC(LCBaseLLM):
        @property
        def _llm_type(self) -> str:
            return "nrag"

        def _call(self, prompt, stop=None, run_manager=None, **kwargs):
            return nrag_llm.complete(prompt, stop=stop)

    return LangchainLLMWrapper(_LC())


def evaluate_answers(
    rag,
    questions: List[str],
    references: Optional[List[str]] = None,
    *,
    metrics=None,
    ragas_llm=None,
):
    """Evaluate generated answers with RAGAS. Returns the RAGAS result object."""
    try:
        from ragas import EvaluationDataset, evaluate  # type: ignore
        from ragas.metrics import Faithfulness, ResponseRelevancy  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("RAGAS eval requires: pip install nrag[eval]") from exc

    samples = []
    for i, q in enumerate(questions):
        res = rag.query(q)
        sample = {
            "user_input": q,
            "response": res.answer or "",
            "retrieved_contexts": [h.text for h in res.hits],
        }
        if references is not None and i < len(references):
            sample["reference"] = references[i]
        samples.append(sample)

    dataset = EvaluationDataset.from_list(samples)
    llm = ragas_llm or _wrap_nrag_llm_for_ragas(rag.llm)
    metrics = metrics or [Faithfulness(), ResponseRelevancy()]
    return evaluate(dataset=dataset, metrics=metrics, llm=llm)
