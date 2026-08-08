"""Optional in-memory engine: bm25s (the fastest pure-Python BM25, Lù 2024).

bm25s precomputes a sparse score matrix at index time, so it is batch/in-memory with no
incremental append — we keep all chunk text in memory and rebuild the index on
``commit()`` (fast). Best for fixed corpora where raw query speed matters most; persists
by saving the text map and re-indexing on open. Word signal only (the simple fast path).

Requires the optional extra:  pip install nrag[bm25s]
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional

from .._types import Chunk, EngineConfig, FieldWeights, Hit, MetaFilter


def _require_bm25s():
    try:
        import bm25s  # type: ignore

        return bm25s
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "The bm25s engine requires the 'bm25s' extra: pip install nrag[bm25s]"
        ) from exc


class InMemoryBM25Engine:
    def __init__(self, config: EngineConfig, path: Optional[str]):
        self._bm25s = _require_bm25s()
        self.config = config
        self.path = path
        self._docs: dict[str, str] = {}          # chunk_id -> indexed_text
        self._meta: dict[str, tuple[str, str]] = {}  # chunk_id -> (doc_id, section)
        self._retriever = None
        self._ids: List[str] = []
        self._dirty = False
        self._stemmer = self._make_stemmer(config)

    @staticmethod
    def _make_stemmer(config: EngineConfig):
        if not config.enable_stemming:
            return None
        try:
            import snowballstemmer  # type: ignore

            lang = config.language if config.language != "porter" else "english"
            return snowballstemmer.stemmer(lang).stemWords
        except Exception:
            return None

    @classmethod
    def open(cls, path: Optional[str] = None, *, create: bool = True,
             config: Optional[EngineConfig] = None) -> "InMemoryBM25Engine":
        eng = cls(config or EngineConfig(), path)
        if path:
            os.makedirs(path, exist_ok=True)
            data = os.path.join(path, "bm25s_docs.json")
            if os.path.exists(data):
                with open(data, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                eng._docs = payload.get("docs", {})
                eng._meta = {k: tuple(v) for k, v in payload.get("meta", {}).items()}
                eng._rebuild()
        return eng

    # ------------------------------------------------------------------ mutation
    def add(self, chunks: Iterable[Chunk]) -> int:
        n = 0
        for c in chunks:
            self._docs[c.chunk_id] = c.indexed_text
            self._meta[c.chunk_id] = (c.doc_id, c.section or "")
            n += 1
        self._dirty = True
        return n

    def update(self, chunk_id: str, chunk: Chunk) -> None:
        self.add([chunk])

    def delete(self, chunk_ids: Iterable[str]) -> int:
        n = 0
        for cid in chunk_ids:
            if self._docs.pop(cid, None) is not None:
                self._meta.pop(cid, None)
                n += 1
        self._dirty = True
        return n

    def delete_doc(self, doc_id: str) -> int:
        victims = [cid for cid, (d, _s) in self._meta.items() if d == doc_id]
        return self.delete(victims)

    def commit(self) -> None:
        if self._dirty:
            self._rebuild()
            self._dirty = False
        if self.path:
            with open(os.path.join(self.path, "bm25s_docs.json"), "w", encoding="utf-8") as fh:
                json.dump({"docs": self._docs, "meta": self._meta}, fh)

    def _rebuild(self) -> None:
        bm25s = self._bm25s
        self._ids = list(self._docs.keys())
        if not self._ids:
            self._retriever = None
            return
        corpus = [self._docs[cid] for cid in self._ids]
        tokens = bm25s.tokenize(corpus, stopwords="en" if self.config.stopwords else None,
                                stemmer=self._stemmer, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokens, show_progress=False)
        self._retriever = retriever

    # ------------------------------------------------------------------ search
    def search(self, query: str, *, k: int = 10, signal: str = "body",
               field_weights: FieldWeights = FieldWeights(), fuzzy: bool = False,
               filter: Optional[MetaFilter] = None) -> List[Hit]:
        if self._retriever is None or not self._ids:
            return []
        bm25s = self._bm25s
        q_tokens = bm25s.tokenize([query], stopwords="en" if self.config.stopwords else None,
                                  stemmer=self._stemmer, show_progress=False)
        k = max(1, min(k, len(self._ids)))
        results, scores = self._retriever.retrieve(
            q_tokens, corpus=self._ids, k=k, return_as="tuple", show_progress=False)
        hits: List[Hit] = []
        for rank, (cid, score) in enumerate(zip(results[0], scores[0], strict=False), start=1):
            hits.append(Hit(chunk_id=str(cid), score=float(score), rank=rank, signal="body"))
        return hits

    # ------------------------------------------------------------------ misc
    @property
    def supports_incremental(self) -> bool:
        return False

    @property
    def supports_multifield(self) -> bool:
        return True  # single word signal returned directly (Path A, no external fusion)

    @property
    def supports_prefilter(self) -> bool:
        return False  # retrieve layer post-filters using hydrated chunks

    def prefilter_covers(self, filter: Optional[MetaFilter]) -> bool:
        """No pushdown at all: every non-empty filter needs the residual post-filter."""
        return filter is None or filter.is_empty()

    def stats(self) -> dict:
        return {"engine": "bm25s", "path": self.path, "num_chunks": len(self._ids),
                "ngram": False}

    def close(self) -> None:
        self.commit()
