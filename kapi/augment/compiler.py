"""The index-time reasoning compiler — the heart of Compiled Retrieval (STRATEGY §4-§5).

Everyone else puts the smart compute on the hot path (an embedding model per query, or an
LLM reasoning per query). Compiled Retrieval moves it to *index time*: an LLM runs **once
per chunk, cached forever**, and compiles each chunk into its *queryable + inferential
closure* — a purely lexical, enriched representation. Query time stays ~1 ms BM25.

One offline pass per chunk emits a *bundle* (the unification, Pillar 1):

  * **blurb**        — a chunk-specific context sentence (Anthropic Contextual Retrieval).
  * **questions**    — questions/claims this chunk answers (doc2query / docTTTTTquery).
  * **propositions** — atomic, decontextualized facts (Dense X) so rare entities match.
  * **inferences**   — second-order facts / multi-hop bridges a knowledgeable reader draws
                       that are *not lexically present* (the BRIGHT-winning signal, precomputed).

The readable bundle is prepended to ``indexed_text`` (Leg A — Contextual BM25), never to
``raw_text`` (so citations stay clean). Pillars 2-3 (CSC) add the consensus-weighted,
literal-anchored term vector for Leg B — see :class:`CompiledBundle.term_weights` and §5.

Reuses the proven offline machinery: content-hash cache (re-indexing is free), bounded
concurrency, and the cost guard that is a no-op for local/unknown models.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .._types import Chunk, attach_context
from ..config import Config
from ..llm.base import batched_complete, estimate_tokens, model_name
from ..results import CostEstimate, CostGuardError
from ..tokenize.text import WordTokenizer

_PROMPT_VERSION = "v1"

_SYSTEM = (
    "You are an index-time retrieval compiler. Given a text chunk and its document, you "
    "emit a compact bundle that makes the chunk findable by lexical search, including the "
    "queries it answers and the facts a knowledgeable reader would infer from it. "
    "Respond ONLY in the exact section format requested, no preamble."
)

# Section directives are assembled per-config so disabled sections cost no tokens.
_SECTION_SPECS: Dict[str, Tuple[str, str]] = {
    "blurb": ("CONTEXT",
              "CONTEXT: one sentence situating the chunk in its document."),
    "questions": ("QUESTIONS",
                  "QUESTIONS: 3-6 distinct questions or claims this chunk directly answers, "
                  "one per line starting with '- '."),
    "propositions": ("PROPOSITIONS",
                     "PROPOSITIONS: 3-6 atomic, self-contained facts from the chunk with all "
                     "pronouns/references resolved to explicit entities, one per '- ' line."),
    "inferences": ("INFERENCES",
                   "INFERENCES: 2-5 second-order facts, implications, or multi-hop bridges a "
                   "knowledgeable reader would infer that are NOT stated verbatim, one per '- ' line."),
}

# USD per 1M tokens (input, output) for common cloud models; unknown/local -> (0, 0) so the
# cost guard never fires for local models. (Mirrors augment/contextual.py.)
_PRICES = {
    "gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.5, 10.0), "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.0, 8.0), "gpt-3.5": (0.5, 1.5), "haiku": (0.80, 4.0),
    "sonnet": (3.0, 15.0), "opus": (15.0, 75.0),
}


def _price_for(model: str) -> Tuple[float, float]:
    m = (model or "").lower()
    for key, price in _PRICES.items():
        if key in m:
            return price
    return (0.0, 0.0)


@dataclass
class CompiledBundle:
    """The compiled, purely-lexical closure of one chunk.

    ``enrichment_text()`` is the readable augmentation for Leg A (``indexed_text``).
    ``term_weights`` is the consensus-weighted, literal-anchored sparse vector for Leg B.
    """

    blurb: str = ""
    questions: List[str] = field(default_factory=list)
    propositions: List[str] = field(default_factory=list)
    inferences: List[str] = field(default_factory=list)
    term_weights: Dict[str, float] = field(default_factory=dict)

    def enrichment_text(self) -> str:
        parts: List[str] = []
        if self.blurb:
            parts.append(self.blurb)
        for items in (self.questions, self.propositions, self.inferences):
            parts.extend(items)
        return "\n".join(p for p in parts if p).strip()

    def is_empty(self) -> bool:
        return not (self.blurb or self.questions or self.propositions or self.inferences)


class _BundleCache:
    """key -> bundle-record JSON. SQLite when persistent, dict when in-memory.

    Stores the parsed samples (not the final weights) so consensus thresholds / the literal
    floor stay config-time tunable while the expensive LLM samples remain cached.
    """

    def __init__(self, path: Optional[str]) -> None:
        self.conn = None
        self._mem: Dict[str, str] = {}
        if path:
            cache_dir = os.path.join(path, ".kapi_cache")
            os.makedirs(cache_dir, exist_ok=True)
            self.conn = sqlite3.connect(os.path.join(cache_dir, "compiler.sqlite"),
                                        check_same_thread=False)
            self.conn.execute("CREATE TABLE IF NOT EXISTS bundles (key TEXT PRIMARY KEY, rec TEXT)")
            self.conn.commit()

    def get(self, key: str) -> Optional[dict]:
        if self.conn is not None:
            row = self.conn.execute("SELECT rec FROM bundles WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None
        rec = self._mem.get(key)
        return json.loads(rec) if rec else None

    def put_many(self, items: List[Tuple[str, dict]]) -> None:
        if not items:
            return
        rows = [(k, json.dumps(rec, ensure_ascii=False)) for k, rec in items]
        if self.conn is not None:
            self.conn.executemany("INSERT OR REPLACE INTO bundles (key, rec) VALUES (?,?)", rows)
            self.conn.commit()
        else:
            self._mem.update(dict(rows))


class Compiler:
    """Compiles chunks offline into enriched lexical representations + CSC term weights."""

    def __init__(self, llm, config: Config, path: Optional[str]) -> None:
        self.llm = llm
        self.config = config
        self.model = config.contextual_model or model_name(llm)
        self.cache = _BundleCache(path)
        self._tok = WordTokenizer(config.language)
        self._sections = [name for name in ("blurb", "questions", "propositions", "inferences")
                          if getattr(config, f"compile_{name}", True)]

    # ------------------------------------------------------------------ sampling params
    @property
    def k(self) -> int:
        return max(1, self.config.consensus_k) if self.config.csc_enabled else 1

    @property
    def temperature(self) -> float:
        return self.config.consensus_temperature if self.k > 1 else 0.0

    # ------------------------------------------------------------------ keys/windows/prompt
    def _key(self, chunk: Chunk) -> str:
        raw = (f"{_PROMPT_VERSION}|{self.model}|{self.config.contextual_window}|k{self.k}|"
               f"t{self.temperature}|{','.join(self._sections)}|{chunk.content_hash}")
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _window(self, doc_text: str, chunk: Chunk) -> str:
        limit = self.config.contextual_window_token_limit
        if self.config.contextual_window == "full_doc" and estimate_tokens(self.llm, doc_text) <= limit:
            return doc_text
        pad = 2000
        start = max(0, chunk.start_offset - pad)
        end = min(len(doc_text), chunk.end_offset + pad)
        return doc_text[start:end]

    def _prompt(self, doc_text: str, chunk: Chunk) -> str:
        directives = "\n".join(_SECTION_SPECS[name][1] for name in self._sections)
        return (
            f"<document>\n{self._window(doc_text, chunk)}\n</document>\n\n"
            f"<chunk>\n{chunk.raw_text}\n</chunk>\n\n"
            f"Compile the chunk for retrieval. Emit exactly these sections:\n{directives}"
        )

    # ------------------------------------------------------------------ estimate / guard
    def estimate(self, items_or_chunks) -> CostEstimate:
        items = self._normalize(items_or_chunks)
        uncached = [(doc, c) for doc, c in items if self.cache.get(self._key(c)) is None]
        in_tok = len(uncached) * estimate_tokens(self.llm, _SYSTEM)
        for doc, c in uncached:
            in_tok += estimate_tokens(self.llm, self._prompt(doc, c))
        in_tok *= self.k                                              # k samples per chunk
        out_tok = len(uncached) * self.config.compile_max_tokens * self.k
        pin, pout = _price_for(self.model)
        usd = pin * in_tok / 1e6 + pout * out_tok / 1e6
        return CostEstimate(len(uncached), in_tok, out_tok, usd, self.model)

    def _check_guard(self, est: CostEstimate) -> None:
        guard = self.config.contextual_cost_guard_usd
        mode = self.config.contextual_cost_guard_mode
        if guard <= 0 or mode == "off" or est.est_usd <= guard:
            return
        msg = (f"Compilation estimated at ${est.est_usd:.2f} for {est.n_chunks} chunks "
               f"(k={self.k}) on '{est.model}' exceeds the ${guard:.2f} guard. Use a local "
               f"model, lower consensus_k, or raise contextual_cost_guard_usd.")
        if mode == "warn":
            import warnings

            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        else:
            raise CostGuardError(msg)

    # ------------------------------------------------------------------ run
    def compile(self, items_or_chunks) -> Dict[str, CompiledBundle]:
        """Compile each chunk; returns {chunk_id: CompiledBundle}. Mutates ``indexed_text``."""
        items = self._normalize(items_or_chunks)
        if not items:
            return {}
        self._check_guard(self.estimate(items))

        # Resolve cached records; queue the rest as (chunk, prompt) repeated k times.
        records: Dict[str, dict] = {}
        to_gen: List[Tuple[str, Chunk]] = []         # (key, chunk) — one per chunk
        prompts: List[str] = []                      # k*len(to_gen) prompts, chunk-major
        for doc, c in items:
            key = self._key(c)
            cached = self.cache.get(key)
            if cached is not None:
                records[c.chunk_id] = cached
            else:
                to_gen.append((key, c))
                prompts.extend([self._prompt(doc, c)] * self.k)

        if prompts:
            raw = batched_complete(
                self.llm, prompts,
                concurrency=self.config.compile_concurrency,
                max_tokens=self.config.compile_max_tokens,
                temperature=self.temperature,
                system=_SYSTEM,
            )
            new_cache: List[Tuple[str, dict]] = []
            for i, (key, chunk) in enumerate(to_gen):
                samples = raw[i * self.k:(i + 1) * self.k]
                rec = self._record_from_samples([self._parse(s or "") for s in samples])
                records[chunk.chunk_id] = rec
                new_cache.append((key, rec))
            self.cache.put_many(new_cache)

        out: Dict[str, CompiledBundle] = {}
        for doc, c in items:
            bundle = self._bundle_from_record(records.get(c.chunk_id, {}), c)
            attach_context(c, bundle.enrichment_text())     # enrich Leg A's indexed_text
            out[c.chunk_id] = bundle
        return out

    # ------------------------------------------------------------------ parsing
    def _parse(self, text: str) -> dict:
        """Parse one LLM sample into {blurb, questions, propositions, inferences}."""
        header_to_field = {spec[0]: name for name, spec in _SECTION_SPECS.items()}
        fields: Dict[str, List[str]] = {n: [] for n in _SECTION_SPECS}
        blurb_lines: List[str] = []
        current: Optional[str] = None
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            head, _, rest = stripped.partition(":")
            field_name = header_to_field.get(head.strip().upper())
            if field_name is not None:
                current = field_name
                rest = rest.strip()
                if current == "blurb" and rest:
                    blurb_lines.append(rest)
                elif rest and rest not in ("-", "•"):
                    fields[current].append(self._clean_item(rest))
                continue
            if current == "blurb":
                blurb_lines.append(stripped)
            elif current in fields:
                item = self._clean_item(stripped)
                if item:
                    fields[current].append(item)
        return {
            "blurb": " ".join(blurb_lines).strip(),
            "questions": fields["questions"],
            "propositions": fields["propositions"],
            "inferences": fields["inferences"],
        }

    @staticmethod
    def _clean_item(s: str) -> str:
        return s.lstrip("-•* \t").strip()

    # ------------------------------------------------------------------ aggregation
    def _record_from_samples(self, samples: List[dict]) -> dict:
        """Aggregate k parsed samples into a cacheable record.

        Readable lists (Leg A) = order-preserving union across samples (recall-oriented
        augmentation; BM25 tolerates breadth). ``sample_terms`` (Leg B) = the token set of
        each sample's enrichment, kept per-sample so consensus weights can be recomputed at
        apply time against the current thresholds.
        """
        blurb = next((s["blurb"] for s in samples if s.get("blurb")), "")
        merged = {k: self._union(s.get(k, []) for s in samples)
                  for k in ("questions", "propositions", "inferences")}
        sample_terms: List[List[str]] = []
        for s in samples:
            text = "\n".join([s.get("blurb", ""), *s.get("questions", []),
                              *s.get("propositions", []), *s.get("inferences", [])])
            sample_terms.append(sorted(set(self._tok(text))))
        return {"blurb": blurb, **merged, "sample_terms": sample_terms}

    @staticmethod
    def _union(lists) -> List[str]:
        """Order-preserving dedup (by lowercased text), keeping the first surface form."""
        seen: Dict[str, str] = {}
        for items in lists:
            for it in items:
                surface = it.strip()
                key = surface.lower()
                if key and key not in seen:
                    seen[key] = surface
        return list(seen.values())

    # ------------------------------------------------------------------ bundle + CSC weights
    def _bundle_from_record(self, rec: dict, chunk: Chunk) -> CompiledBundle:
        bundle = CompiledBundle(
            blurb=rec.get("blurb", ""),
            questions=rec.get("questions", []),
            propositions=rec.get("propositions", []),
            inferences=rec.get("inferences", []),
        )
        if self.config.csc_enabled:
            bundle.term_weights = self._consensus_weights(rec.get("sample_terms", []), chunk)
        return bundle

    def _consensus_weights(self, sample_terms: List[List[str]], chunk: Chunk) -> Dict[str, float]:
        """Pillar 2 (consensus) + Pillar 3 (literal anchoring).

        Expansion term weight = fraction of samples containing it (self-consistency: a free
        doc2query-- filter + graded learned-sparse weight). Kept iff >= ``consensus_min_agreement``.
        Source-literal terms keep a ``literal_floor`` so exact-match retrieval is structurally
        protected (the fix to the BM42/PROSPER under-weighting failure).
        """
        k = max(1, len(sample_terms)) if sample_terms else 1
        counts: Counter[str] = Counter()
        for terms in sample_terms:
            counts.update(set(terms))
        consensus = {t: c / k for t, c in counts.items()}

        literal = set(self._tok(chunk.raw_text))
        floor = self.config.literal_floor
        min_agree = self.config.consensus_min_agreement

        weights: Dict[str, float] = {t: floor for t in literal}     # anchor literals
        for t, w in consensus.items():
            if t in literal:
                weights[t] = max(floor, w)                          # anchored, consensus may lift
            elif w >= min_agree:
                weights[t] = w                                      # expansion above the floor
        return weights

    # ------------------------------------------------------------------ misc
    @staticmethod
    def _normalize(items_or_chunks) -> List[Tuple[str, Chunk]]:
        out: List[Tuple[str, Chunk]] = []
        for it in items_or_chunks:
            out.append(it if isinstance(it, tuple) else (it.raw_text, it))
        return out
