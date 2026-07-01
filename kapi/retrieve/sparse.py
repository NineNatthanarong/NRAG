"""Leg B — the consensus-weighted sparse index (CSC, STRATEGY §5 Pillar 4).

A learned-sparse retriever with **no trained model**: documents carry the per-term weights
the compiler produced by consensus (Pillar 2) + literal anchoring (Pillar 3); the query side
stays model-free, weighting each query term by IDF (Geng et al. / Nardini et al. show the
query encoder is droppable at tiny cost — §2.1). Scoring is a sparse dot product, served at
BM25 speed with no embedding model and no vector DB.

This is "Leg B" of the asymmetric all-sparse hybrid: it fuses with the plain-lexical Leg A
(the inverted-index engine) to reproduce hybrid's two-error-profile win without anything dense.

Pure-Python, dependency-free, persisted as one JSON next to the index so ``Kapi.open``
reloads it. The inverted index + document frequencies are derived from the per-chunk vectors
(the source of truth) and rebuilt lazily after any mutation.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

from .._types import Hit
from ..tokenize.text import WordTokenizer

_STORE = "legb.json"


class SparseConsensusIndex:
    """In-memory weighted sparse index over per-chunk consensus term vectors."""

    def __init__(self, path: Optional[str] = None, *, language: str = "english") -> None:
        self.path = path
        self._tok = WordTokenizer(language)
        self.vectors: Dict[str, Dict[str, float]] = {}     # chunk_id -> {term: weight}
        self._inverted: Dict[str, List[tuple[str, float]]] = {}
        self._df: Dict[str, int] = {}
        self._dirty = True
        if path:
            self._load()

    # ------------------------------------------------------------------ mutation
    def add(self, chunk_id: str, term_weights: Dict[str, float]) -> None:
        if term_weights:
            self.vectors[chunk_id] = dict(term_weights)
            self._dirty = True

    def add_many(self, items: Dict[str, Dict[str, float]]) -> None:
        for cid, w in items.items():
            self.add(cid, w)

    def delete(self, chunk_id: str) -> None:
        if self.vectors.pop(chunk_id, None) is not None:
            self._dirty = True

    def delete_doc(self, doc_id: str) -> None:
        prefix = f"{doc_id}::"
        gone = [cid for cid in self.vectors if cid == doc_id or cid.startswith(prefix)]
        for cid in gone:
            del self.vectors[cid]
        if gone:
            self._dirty = True

    # ------------------------------------------------------------------ index build
    def _rebuild(self) -> None:
        inverted: Dict[str, List[tuple[str, float]]] = {}
        df: Dict[str, int] = {}
        for cid, vec in self.vectors.items():
            for term, w in vec.items():
                inverted.setdefault(term, []).append((cid, w))
                df[term] = df.get(term, 0) + 1
        self._inverted, self._df, self._dirty = inverted, df, False

    def _idf(self, term: str) -> float:
        """BM25-style IDF, always positive: ln(1 + (N - df + 0.5)/(df + 0.5))."""
        n = len(self.vectors)
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    # ------------------------------------------------------------------ search
    def search(self, query: str, *, k: int = 50) -> List[Hit]:
        if self._dirty:
            self._rebuild()
        if not self.vectors:
            return []
        q_terms = self._tok(query)
        if not q_terms:
            return []
        # query-side weight = IDF (model-free), one weight per distinct query term
        q_weights: Dict[str, float] = {}
        for t in q_terms:
            if t not in q_weights:
                q_weights[t] = self._idf(t)

        scores: Dict[str, float] = {}
        for term, qw in q_weights.items():
            if qw <= 0.0:
                continue
            for cid, dw in self._inverted.get(term, ()):  # sparse dot product
                scores[cid] = scores.get(cid, 0.0) + qw * dw
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(chunk_id=cid, score=sc, rank=i, signal="csc")
                for i, (cid, sc) in enumerate(ranked, start=1)]

    # ------------------------------------------------------------------ persistence
    def commit(self) -> None:
        if not self.path:
            return
        cache_dir = os.path.join(self.path, ".kapi_csc")
        os.makedirs(cache_dir, exist_ok=True)
        tmp = os.path.join(cache_dir, _STORE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"vectors": self.vectors}, fh, ensure_ascii=False)
        os.replace(tmp, os.path.join(cache_dir, _STORE))

    def _load(self) -> None:
        store = os.path.join(self.path, ".kapi_csc", _STORE)  # type: ignore[arg-type]
        if not os.path.exists(store):
            return
        try:
            with open(store, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.vectors = {cid: {t: float(w) for t, w in vec.items()}
                            for cid, vec in data.get("vectors", {}).items()}
            self._dirty = True
        except Exception:
            self.vectors = {}

    # ------------------------------------------------------------------ misc
    def stats(self) -> dict:
        if self._dirty:
            self._rebuild()
        return {"leg": "csc", "num_chunks": len(self.vectors), "vocab": len(self._df),
                "path": self.path}
