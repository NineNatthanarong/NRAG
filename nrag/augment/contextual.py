"""Contextual indexing — the biggest cheap quality lever (Anthropic, Sept 2024).

For each chunk, the LLM writes a 1-2 sentence blurb situating it in its document; the
blurb is prepended to ``indexed_text`` only (Contextual BM25), so retrieval improves
while the displayed/cited text stays clean. Reported effect: ~-49% retrieval failures.

This work is OFFLINE and amortized: results are cached by chunk content-hash (so
re-indexing is free), batched with bounded concurrency, and gated by a cost guard that
is a no-op for local/unknown models (which is why "quality" can default this ON while
staying low-cost — point it at a local Ollama model).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from typing import List, Optional

from .._types import Chunk, attach_context
from ..config import Config
from ..llm.base import batched_complete, estimate_tokens, model_name
from ..results import CostEstimate, CostGuardError

_PROMPT_VERSION = "v1"

_SYSTEM = (
    "You write a very short context blurb that situates a text chunk within its source "
    "document, to improve search retrieval. Respond with ONLY the blurb: 1-2 sentences, "
    "50-100 tokens, no preamble, no quotes."
)
_USER = (
    "<document>\n{document}\n</document>\n\n"
    "Here is the chunk we want to situate within the document:\n<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short, succinct context to situate this chunk within the document for the "
    "purposes of improving search retrieval of the chunk. Answer only with the context."
)

# USD per 1M tokens (input, output) for common cloud models. Unknown/local -> (0, 0),
# so the cost guard never fires for local models.
_PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-3.5": (0.5, 1.5),
    "haiku": (0.80, 4.0),
    "sonnet": (3.0, 15.0),
    "opus": (15.0, 75.0),
}


def _price_for(model: str) -> tuple[float, float]:
    m = (model or "").lower()
    for key, price in _PRICES.items():
        if key in m:
            return price
    return (0.0, 0.0)


class _BlurbCache:
    """Tiny key->blurb cache; SQLite when persistent, dict when in-memory."""

    def __init__(self, path: Optional[str]) -> None:
        self.conn = None
        self._mem: dict[str, str] = {}
        if path:
            cache_dir = os.path.join(path, ".nrag_cache")
            os.makedirs(cache_dir, exist_ok=True)
            self.conn = sqlite3.connect(os.path.join(cache_dir, "contextual.sqlite"),
                                        check_same_thread=False)
            self.conn.execute("CREATE TABLE IF NOT EXISTS ctx (key TEXT PRIMARY KEY, blurb TEXT)")
            self.conn.commit()

    def get(self, key: str) -> Optional[str]:
        if self.conn is not None:
            row = self.conn.execute("SELECT blurb FROM ctx WHERE key=?", (key,)).fetchone()
            return row[0] if row else None
        return self._mem.get(key)

    def put_many(self, items: list[tuple[str, str]]) -> None:
        if not items:
            return
        if self.conn is not None:
            self.conn.executemany("INSERT OR REPLACE INTO ctx (key, blurb) VALUES (?,?)", items)
            self.conn.commit()
        else:
            self._mem.update(dict(items))


class ContextualIndexer:
    def __init__(self, llm, config: Config, path: Optional[str]) -> None:
        self.llm = llm
        self.config = config
        self.model = config.contextual_model or model_name(llm)
        self.cache = _BlurbCache(path)

    # ------------------------------------------------------------------ keys/windows
    def _key(self, chunk: Chunk) -> str:
        raw = f"{_PROMPT_VERSION}|{self.model}|{self.config.contextual_window}|{chunk.content_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _window(self, doc_text: str, chunk: Chunk) -> str:
        limit = self.config.contextual_window_token_limit
        if self.config.contextual_window == "full_doc" and estimate_tokens(self.llm, doc_text) <= limit:
            return doc_text
        # local window around the chunk (cheap context for long docs)
        pad = 2000
        start = max(0, chunk.start_offset - pad)
        end = min(len(doc_text), chunk.end_offset + pad)
        return doc_text[start:end]

    # ------------------------------------------------------------------ estimate / guard
    def estimate(self, items_or_chunks) -> CostEstimate:
        items = self._normalize(items_or_chunks)
        uncached = [(doc, c) for doc, c in items if self.cache.get(self._key(c)) is None]
        in_tok = 0
        for doc, c in uncached:
            in_tok += estimate_tokens(self.llm, self._window(doc, c)) + estimate_tokens(self.llm, c.raw_text)
        in_tok += len(uncached) * estimate_tokens(self.llm, _SYSTEM)
        out_tok = len(uncached) * self.config.contextual_max_tokens
        pin, pout = _price_for(self.model)
        usd = pin * in_tok / 1e6 + pout * out_tok / 1e6
        return CostEstimate(len(uncached), in_tok, out_tok, usd, self.model)

    def _check_guard(self, est: CostEstimate) -> None:
        guard = self.config.contextual_cost_guard_usd
        mode = self.config.contextual_cost_guard_mode
        if guard <= 0 or mode == "off" or est.est_usd <= guard:
            return
        msg = (f"Contextual indexing estimated at ${est.est_usd:.2f} for {est.n_chunks} "
               f"chunks on '{est.model}' exceeds the ${guard:.2f} guard. "
               f"Use preset='fast', a local model, or raise contextual_cost_guard_usd.")
        if mode == "warn":
            import warnings

            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        else:
            raise CostGuardError(msg)

    # ------------------------------------------------------------------ run
    def contextualize(self, items_or_chunks) -> int:
        items = self._normalize(items_or_chunks)
        if not items:
            return 0
        est = self.estimate(items)
        self._check_guard(est)

        # resolve cached vs to-generate
        to_gen: list[tuple[str, Chunk, str]] = []  # (key, chunk, prompt)
        applied = 0
        for doc, c in items:
            key = self._key(c)
            cached = self.cache.get(key)
            if cached is not None:
                attach_context(c, cached)
                applied += 1
            else:
                prompt = _USER.format(document=self._window(doc, c), chunk=c.raw_text)
                to_gen.append((key, c, prompt))

        if to_gen:
            blurbs = batched_complete(
                self.llm,
                [p for _k, _c, p in to_gen],
                concurrency=self.config.contextual_concurrency,
                max_tokens=self.config.contextual_max_tokens,
                temperature=0.0,
                system=_SYSTEM,
            )
            new_cache: list[tuple[str, str]] = []
            for (key, chunk, _p), blurb in zip(to_gen, blurbs, strict=False):
                blurb = (blurb or "").strip()
                attach_context(chunk, blurb)
                new_cache.append((key, blurb))
                applied += 1
            self.cache.put_many(new_cache)
        return applied

    # ------------------------------------------------------------------ misc
    @staticmethod
    def _normalize(items_or_chunks) -> List[tuple[str, Chunk]]:
        """Accept either [(doc_text, chunk)] or [chunk] (window falls back to raw_text)."""
        out: List[tuple[str, Chunk]] = []
        for it in items_or_chunks:
            if isinstance(it, tuple):
                out.append(it)
            else:
                out.append((it.raw_text, it))
        return out
