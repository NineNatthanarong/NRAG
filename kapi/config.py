"""Configuration and presets for Kapi.

A single frozen ``Config`` is the source of truth for all behavior. Presets are
factory classmethods; any field is overridable via kwargs to ``Kapi(...)``. The
default preset is ``quality`` (LLM enhancers ON) per the product brief; ``fast``
collapses to pure-lexical; ``for_no_llm`` is applied automatically when no LLM is
supplied so "no LLM still works" is guaranteed by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

ExpandMode = Literal["query2doc", "cot_keywords", "auto"]
Preset = Literal["quality", "fast", "compiled"]


@dataclass(frozen=True)
class Config:
    # ---- top-level ----
    preset: Preset = "quality"

    # ---- contextual indexing (offline, default ON in 'quality') ----
    contextual_enabled: bool = True
    contextual_model: Optional[str] = None          # None -> plugged-in LLM default
    contextual_max_tokens: int = 128                 # blurb budget (~50-100 tokens of content)
    contextual_concurrency: int = 8                  # bounded parallel offline calls
    contextual_window: Literal["full_doc", "section"] = "full_doc"
    contextual_window_token_limit: int = 6000        # above this, fall back to section window
    contextual_cost_guard_usd: float = 5.0           # 0 disables the guard
    contextual_cost_guard_mode: Literal["raise", "warn", "off"] = "raise"

    # ---- Compiled Retrieval: the offline index-time compiler (default ON in 'compiled') ----
    # One cached, cost-guarded offline pass per chunk emits an enrichment *bundle*
    # (Anthropic blurb + doc2query questions + Dense-X propositions + reasoning expansion);
    # the readable bundle lands in ``indexed_text`` (Leg A). See augment/compiler.py.
    compile_enabled: bool = False
    compile_blurb: bool = True            # contextual blurb (Anthropic, 2024)
    compile_questions: bool = True        # doc2query — questions/claims the chunk answers
    compile_propositions: bool = True     # Dense X — atomic decontextualized facts
    compile_reasoning: bool = True        # inferential closure — the BRIGHT-winning signal
    compile_max_tokens: int = 384         # bundle budget per sample
    compile_concurrency: int = 8          # bounded parallel offline calls

    # ---- CSC: Consensus Sparse Compilation (Leg B), the novel core (§5) ----
    # When enabled, the compiler is sampled ``consensus_k`` times at temperature>0; a term's
    # weight is its agreement across samples (self-consistency = a free doc2query-- filter +
    # graded learned-sparse weight). Literal terms keep a floor weight (literal anchoring).
    csc_enabled: bool = False             # build + fuse the consensus-weighted sparse leg
    consensus_k: int = 1                  # samples per chunk (k>1 enables consensus filtering)
    consensus_temperature: float = 0.7    # sampling temperature for k>1
    consensus_min_agreement: float = 0.5  # keep an expansion term iff it recurs in >= this fraction
    literal_floor: float = 1.0            # anchoring floor weight for source-literal terms in Leg B
    csc_leg_weights: tuple = (1.0, 1.0)   # (legA, legB) weights for convex two-leg fusion

    # ---- adaptive query router (Phase 3, §5 Pillar 5): the only query-time LLM use ----
    # Cheap by default, accurate on demand. The first lexical pass is ~1 ms / $0; a cheap
    # confidence signal (no hits, low recall, or an ambiguous top-vs-2nd margin) decides
    # whether to spend one LLM call escalating (query expansion + re-search). Short queries
    # are treated as precise exact-match and never escalated — the §2.4 precision-trap guard.
    router_enabled: bool = False
    router_short_query_words: int = 4     # <= this many words -> precise; never escalate
    router_min_hits: int = 2              # a long query returning fewer hits -> under-recall
    router_min_margin: float = 0.12       # top hit must lead the 2nd by this frac, else ambiguous
    router_min_top_score: float = 0.0     # optional absolute floor (0 = off; engine-specific)

    # ---- query expansion (online, single call, default ON in 'quality') ----
    expand_enabled: bool = True
    expand_mode: ExpandMode = "auto"
    expand_query_repeat: int = 5                      # query2doc ~5x repetition trick (sparse retrieval)
    expand_max_tokens: int = 256
    expand_auto_short_query_words: int = 5            # 'auto': <= N words -> cot_keywords

    # ---- retrieval ----
    k: int = 10                                       # final top-k returned
    retrieve_k: int = 50                              # candidates fetched before truncation
    fusion: Literal["rrf", "convex"] = "rrf"
    rrf_k: int = 60
    # field weights (mirrors FieldWeights defaults; kept here so presets can tune them)
    weight_body: float = 1.0
    weight_ngram: float = 0.6
    weight_title: float = 2.5
    enable_ngram: bool = True
    fuzzy: bool = False

    # ---- generation ----
    generate_enabled: bool = True                     # auto-False if no LLM
    answer_max_tokens: int = 512
    context_token_budget: int = 6000                  # cap on chunk tokens packed into the prompt
    citation_style: Literal["bracket", "none"] = "bracket"

    # ---- engine / io ----
    engine: Literal["tantivy", "sqlite", "bm25s"] = "tantivy"
    language: str = "english"
    path: Optional[str] = None                         # None -> in-memory

    # ---------------------------------------------------------------- presets
    @classmethod
    def quality(cls, **overrides) -> "Config":
        return replace(cls(preset="quality"), **overrides)

    @classmethod
    def fast(cls, **overrides) -> "Config":
        base = cls(preset="fast", contextual_enabled=False, expand_enabled=False)
        return replace(base, **overrides)

    @classmethod
    def compiled(cls, **overrides) -> "Config":
        """Compiled Retrieval preset: the offline compiler + CSC consensus leg ON.

        The compiler subsumes the contextual blurb, so plain ``contextual_enabled`` is
        off here. Always-on query expansion stays off — the §2.4 precision trap says it must
        be a selective *router* (Phase 3), which is enabled here: it fires expansion only on
        the weak/ambiguous queries that benefit, and stays out of the way on precise ones.
        """
        base = cls(preset="compiled", contextual_enabled=False, expand_enabled=False,
                   compile_enabled=True, csc_enabled=True, consensus_k=3, router_enabled=True)
        return replace(base, **overrides)

    @classmethod
    def from_preset(cls, preset: Preset = "quality", **overrides) -> "Config":
        if preset == "fast":
            return cls.fast(**overrides)
        if preset == "compiled":
            return cls.compiled(**overrides)
        return cls.quality(**overrides)

    def for_no_llm(self) -> "Config":
        """Disable every feature that needs an LLM (graceful degradation)."""
        return replace(self, contextual_enabled=False, expand_enabled=False,
                       compile_enabled=False, csc_enabled=False, generate_enabled=False,
                       router_enabled=False)

    def with_overrides(self, **overrides) -> "Config":
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})
