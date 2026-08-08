"""Shared data contracts for Nrag.

These small, dependency-free dataclasses are the backbone every layer agrees on:
ingest produces ``Document`` -> ``Chunk``; engines index ``Chunk.indexed_text`` and
return ``Hit``s; the retrieve layer fuses and hydrates them. Keeping them here (with no
imports beyond the stdlib) means every module can depend on them without cycles.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional


def content_hash(text: str) -> str:
    """Stable content fingerprint used for change-detection and caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    """A source document before chunking.

    ``doc_id`` is a stable identifier for the *source* (usually its path, or a
    user-supplied id). ``mtime`` is the filesystem modification time when known.
    """

    doc_id: str
    text: str
    source: Optional[str] = None
    mtime: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """An indexable unit of a document.

    The crucial invariant: ``raw_text`` is what we show the caller/LLM and use for
    citations, while ``indexed_text`` is what the tokenizers see. They are equal by
    default; the contextual-indexing step prepends an LLM blurb to ``indexed_text``
    only (see :func:`attach_context`), so retrieval improves without polluting output.
    """

    chunk_id: str
    doc_id: str
    ordinal: int
    raw_text: str
    indexed_text: str
    title: str = ""
    section: str = ""
    start_offset: int = 0
    end_offset: int = 0
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.raw_text)

    @staticmethod
    def make_id(doc_id: str, ordinal: int) -> str:
        """Deterministic chunk id; also the engine delete/update key."""
        return f"{doc_id}::{ordinal}"


@dataclass
class Hit:
    """A single ranked retrieval result."""

    chunk_id: str
    score: float
    rank: int = 0  # 1-based position within a result list (used by RRF)
    chunk: Optional[Chunk] = None
    signal: str = "fused"  # which signal produced it: body|ngram|title|fused

    # convenience accessors that fall back gracefully when not hydrated
    @property
    def text(self) -> str:
        return self.chunk.raw_text if self.chunk else ""

    @property
    def source(self) -> Optional[str]:
        if self.chunk is None:
            return None
        return self.chunk.metadata.get("source") or self.chunk.doc_id


@dataclass(frozen=True)
class FieldWeights:
    """Per-field boosts for multi-signal lexical retrieval.

    Defaults reflect the benchmark grid (BEIR scifact/nfcorpus/fiqa/arguana, see
    benchmarks/lexical_grid.py): words carry the signal, the char-ngram field is a
    soft typo/morphology booster (0.3), and a gentle title boost (0.5) — large title
    boosts consistently hurt ranking across all four datasets.
    """

    body: float = 1.0
    ngram: float = 0.3
    title: float = 0.5

    def as_dict(self) -> dict[str, float]:
        return {"body": self.body, "ngram": self.ngram, "title": self.title}


@dataclass(frozen=True)
class MetaFilter:
    """A simple metadata pre-filter applied (where possible) before scoring.

    ``equals`` requires exact field==value; ``any_of`` requires field in values;
    ``mtime_after`` keeps chunks whose source mtime is newer than the bound.
    Empty filter matches everything.
    """

    equals: dict[str, Any] = field(default_factory=dict)
    any_of: dict[str, list[Any]] = field(default_factory=dict)
    mtime_after: Optional[float] = None

    def is_empty(self) -> bool:
        return not self.equals and not self.any_of and self.mtime_after is None

    def matches(self, chunk: Chunk) -> bool:
        """Pure-Python evaluation, used for post-filtering fallback engines."""
        meta = {**chunk.metadata, "doc_id": chunk.doc_id, "section": chunk.section,
                "title": chunk.title}
        for k, v in self.equals.items():
            if meta.get(k) != v:
                return False
        for k, vals in self.any_of.items():
            if meta.get(k) not in vals:
                return False
        if self.mtime_after is not None:
            mt = chunk.metadata.get("mtime")
            if mt is None or mt <= self.mtime_after:
                return False
        return True


@dataclass(frozen=True)
class EngineConfig:
    """Analyzer / index configuration shared by engine implementations."""

    language: str = "english"
    ngram_min: int = 3
    ngram_max: int = 3
    enable_ngram: bool = True
    enable_stemming: bool = True
    stopwords: bool = True
    ascii_fold: bool = True
    writer_heap_bytes: int = 128 * 1024 * 1024
    writer_threads: int = 1
    # BM25 parameters. Honored by the bm25s engine; tantivy and SQLite FTS5 ship
    # fixed k1=1.2/b=0.75 and ignore these (documented engine capability difference).
    bm25_k1: Optional[float] = None
    bm25_b: Optional[float] = None


def attach_context(chunk: Chunk, blurb: str) -> Chunk:
    """Prepend a contextual blurb to ``indexed_text`` only (Contextual BM25).

    Mutates and returns the chunk. ``raw_text`` is left untouched so generation and
    citations still show the clean source text.
    """
    blurb = (blurb or "").strip()
    if blurb:
        chunk.indexed_text = f"{blurb}\n\n{chunk.raw_text}"
    return chunk
