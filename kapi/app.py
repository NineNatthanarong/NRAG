"""The public ``Kapi`` facade — orchestrates lexical retrieval + optional LLM augmentation.

Design rules:
  * Pure-lexical retrieval ALWAYS works; LLM features are pure add-ons. When ``llm`` is
    None the config is reduced via ``for_no_llm()`` so contextual indexing, query
    expansion and generation are skipped by construction.
  * All LLM cost is offline (contextual indexing at ``add`` time, cached) or a single
    online call (query expansion). ``preset="fast"`` disables both for sub-10ms retrieval.
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterable, Iterator, List, Optional, Union

from ._types import Chunk, Document, EngineConfig, FieldWeights, Hit, MetaFilter, content_hash
from .config import Config
from .engine.base import open_engine
from .ingest.chunker import Chunker, ChunkConfig
from .ingest.loaders import documents_from_texts, load_paths
from .results import AddReport, Citation, CostEstimate, QueryResult
from .retrieve import fuse, multisignal, router
from .store.metadata import DocFingerprint, MetadataStore

Source = Union[str, Document, Iterable[Union[str, Document]]]
_MANIFEST = "kapi.json"


class Kapi:
    def __init__(
        self,
        llm=None,
        *,
        preset: str = "quality",
        path: Optional[str] = None,
        config: Optional[Config] = None,
        engine: Optional[str] = None,
        chunk_config: Optional[ChunkConfig] = None,
        **overrides,
    ) -> None:
        cfg = config or Config.from_preset(preset)  # type: ignore[arg-type]
        cfg = cfg.with_overrides(path=path, engine=engine, **overrides)
        self.llm = llm
        self.config = cfg if llm is not None else cfg.for_no_llm()

        if self.config.path:
            self._merge_manifest(self.config.path)

        self.chunker = Chunker(chunk_config)
        self._engine_config = EngineConfig(
            language=self.config.language, enable_ngram=self.config.enable_ngram
        )
        self.engine = open_engine(self.config.engine, path=self.config.path,
                                  config=self._engine_config)
        self.store = MetadataStore(self.config.path)
        if self.config.path:
            self._write_manifest(self.config.path)

        # lazily-constructed LLM helpers
        self._exp = None
        self._ctx = None
        self._cmp = None
        self._gen = None
        self.last_route = None      # the last router decision (introspection); set by search()

        # Leg B — the CSC consensus-weighted sparse index. Built offline (needs an LLM) but
        # serves model-free, so an already-compiled one is loaded even with no LLM at all
        # ("compile once, serve cheap/local forever"). See retrieve/sparse.py.
        self._legb = None
        if self.config.csc_enabled or self._legb_store_exists():
            from .retrieve.sparse import SparseConsensusIndex

            self._legb = SparseConsensusIndex(self.config.path, language=self.config.language)

    # ------------------------------------------------------------------ open
    @classmethod
    def open(cls, path: str, llm=None, *, preset: str = "quality", **overrides) -> "Kapi":
        return cls(llm=llm, preset=preset, path=path, **overrides)

    # ------------------------------------------------------------------ ingest
    def add(self, source: Source, *, estimate_only: bool = False,
            force: bool = False) -> AddReport:
        docs = list(self._documents(source))
        return self._ingest(docs, sync=False, estimate_only=estimate_only, force=force)

    def add_texts(self, texts: Iterable[str], *, prefix: str = "doc") -> AddReport:
        return self._ingest(list(documents_from_texts(texts, prefix=prefix)), sync=False)

    # ``compile`` is the Compiled-Retrieval verb (STRATEGY §4): same offline ingest as
    # ``add``, named for the mental model — the LLM compiles each doc into a lexical index
    # once. The depth of compilation is set by the preset/config (preset="compiled").
    def compile(self, source: Source, *, estimate_only: bool = False,
                force: bool = False) -> AddReport:
        return self.add(source, estimate_only=estimate_only, force=force)

    def compile_texts(self, texts: Iterable[str], *, prefix: str = "doc") -> AddReport:
        return self.add_texts(texts, prefix=prefix)

    def sync(self, source: Source, *, force: bool = False) -> AddReport:
        """Like ``add`` but also deletes indexed docs no longer present in ``source``."""
        docs = list(self._documents(source))
        return self._ingest(docs, sync=True, force=force)

    def remove(self, doc_id: str) -> None:
        self.engine.delete_doc(doc_id)
        self.store.delete_doc(doc_id)
        if self._legb is not None:
            self._legb.delete_doc(doc_id)
            self._legb.commit()
        self.engine.commit()
        self.store.commit()

    def _ingest(self, docs: List[Document], *, sync: bool, estimate_only: bool = False,
                force: bool = False) -> AddReport:
        fingerprints = [
            DocFingerprint(d.doc_id, content_hash(d.text), d.mtime, d.source) for d in docs
        ]
        diff = self.store.diff(fingerprints)
        changed_set = set(diff.changed)
        added_set = set(diff.added)
        todo = [d for d in docs if force or d.doc_id in changed_set or d.doc_id in added_set]

        # chunk everything we will (re)index
        chunked: list[tuple[Document, list[Chunk]]] = [(d, self.chunker.chunk(d)) for d in todo]
        all_chunks = [c for _d, cs in chunked for c in cs]

        # pair each chunk with its full document text so contextual indexing can
        # situate it (the document window is what makes the blurb useful)
        ctx_items = [(d.text, c) for d, cs in chunked for c in cs]

        if estimate_only:
            est = (self._estimate_compile_cost(ctx_items) if self.config.compile_enabled
                   else self._estimate_contextual_cost(ctx_items))
            return AddReport(num_docs=len(todo), num_chunks=len(all_chunks),
                             added=len(diff.added), changed=len(diff.changed),
                             unchanged=len(diff.unchanged),
                             contextualized=est.n_chunks if est else 0)

        # offline enrichment, prepended to indexed_text only: the Compiled-Retrieval compiler
        # (bundle + CSC consensus weights) when enabled, else the original contextual blurb.
        legb_weights: dict[str, dict] = {}
        if self.config.compile_enabled and self.llm is not None:
            n_ctx, legb_weights = self._compile(ctx_items)
        else:
            n_ctx = self._contextualize(ctx_items)

        # apply mutations
        for d, cs in chunked:
            if d.doc_id not in added_set:  # existing doc: replace old chunks (changed or forced)
                self.engine.delete_doc(d.doc_id)
                self.store.delete_doc(d.doc_id)
                if self._legb is not None:
                    self._legb.delete_doc(d.doc_id)
            self.engine.add(cs)
            self.store.upsert_chunks(cs)
            self.store.record_doc(d.doc_id, content_hash(d.text), d.mtime, d.source,
                                  len(cs), time.time())
        if self._legb is not None and legb_weights:
            self._legb.add_many(legb_weights)

        if sync:
            for doc_id in diff.deleted:
                self.engine.delete_doc(doc_id)
                self.store.delete_doc(doc_id)
                if self._legb is not None:
                    self._legb.delete_doc(doc_id)

        self.engine.commit()
        self.store.commit()
        if self._legb is not None:
            self._legb.commit()
        return AddReport(
            num_docs=len(todo), num_chunks=len(all_chunks),
            added=len(diff.added), changed=len(diff.changed),
            unchanged=len(diff.unchanged), deleted=len(diff.deleted) if sync else 0,
            contextualized=n_ctx,
        )

    # ------------------------------------------------------------------ retrieve
    def search(self, query: str, k: Optional[int] = None, *,
               filter: Optional[MetaFilter] = None) -> List[Hit]:
        k = k or self.config.k
        candidates = self.config.retrieve_k
        # Phase 3: when the adaptive router is on (and an LLM is available), a cheap lexical
        # pass runs first and escalates only if it looks weak. Otherwise use the always-on
        # expansion policy (expand if enabled, else raw) and dispatch once.
        if self.config.router_enabled and self.llm is not None:
            return self._routed_search(query, k, candidates, filter)
        expanded = self._expand(query)
        return self._dispatch(query, expanded, k, candidates, filter)

    def _dispatch(self, raw_query: str, effective_query: str, k: int, candidates: int,
                  filter: Optional[MetaFilter]) -> List[Hit]:
        """Run retrieval for a query string (``effective_query`` may be expansion-augmented).

        Leg B always scores the *raw* query (its query side is model-free by design); Leg A
        scores ``effective_query``. With no Leg B, this is the plain multi-signal search.
        """
        if self._legb is not None and self._legb.vectors:
            return self._search_fused(raw_query, effective_query, k, candidates, filter)
        return multisignal.search(
            self.engine, self.store, effective_query,
            k=k, candidates=candidates,
            field_weights=self._field_weights(),
            fuzzy=self.config.fuzzy,
            filter=filter,
            fusion=self.config.fusion,
            rrf_k=self.config.rrf_k,
        )

    def _routed_search(self, query: str, k: int, candidates: int,
                       filter: Optional[MetaFilter]) -> List[Hit]:
        """Adaptive router (Phase 3): $0 lexical pass, escalate to expansion only if weak."""
        first = self._dispatch(query, query, k, candidates, filter)   # no expansion
        decision = router.assess(first, query, self.config)
        self.last_route = decision
        if not decision.escalate:
            return first
        expanded = self._router_expand(query)                          # one LLM call
        if not expanded or expanded == query:
            return first
        second = self._dispatch(query, expanded, k, candidates, filter)
        return second if second else first

    def _router_expand(self, query: str) -> str:
        """Force query expansion for a router escalation (bypasses ``expand_enabled``)."""
        if self.llm is None:
            return query
        if self._exp is None:
            from .augment.expand import QueryExpander

            self._exp = QueryExpander(self.llm, self.config)
        try:
            return self._exp.expand(query).assembled
        except Exception:
            return query

    def _search_fused(self, raw_query: str, expanded: str, k: int, candidates: int,
                      filter: Optional[MetaFilter]) -> List[Hit]:
        """Asymmetric all-sparse fusion (Pillar 4): Leg A (lexical) + Leg B (CSC), no dense.

        Leg A scores the enriched ``indexed_text``; Leg B scores the consensus-weighted
        sparse vectors using the *raw* query (model-free IDF weights — the query side stays
        dumb by design, §2.1/§2.4). The two are fused by rank, reproducing hybrid's
        two-error-profile win with no embedding model.
        """
        leg_a = multisignal.search(
            self.engine, self.store, expanded,
            k=candidates, candidates=candidates,
            field_weights=self._field_weights(), fuzzy=self.config.fuzzy,
            filter=filter, fusion=self.config.fusion, rrf_k=self.config.rrf_k,
        )
        leg_b = self._legb.search(raw_query, k=candidates)
        if not leg_b:
            hits = leg_a
        else:
            method = "convex" if self.config.fusion == "convex" else "rrf"
            weights = list(self.config.csc_leg_weights) if method == "convex" else None
            fused = fuse.fuse([leg_a, leg_b], method=method, k=self.config.rrf_k, weights=weights)
            hits = self.store.hydrate(fused)  # fill chunks for Leg-B-only hits

        # post-filter any Leg-B-only hits the engine couldn't pre-filter
        if filter is not None and not filter.is_empty():
            hits = [h for h in hits if h.chunk is not None and filter.matches(h.chunk)]
        for rank, h in enumerate(hits[:k], start=1):
            h.rank = rank
        return hits[:k]

    def query(self, question: str, k: Optional[int] = None, *,
              filter: Optional[MetaFilter] = None) -> QueryResult:
        hits = self.search(question, k=k, filter=filter)
        if self.config.generate_enabled and self.llm is not None:
            ans = self._generator().answer(question, hits)
            return QueryResult(question, hits, answer=ans.text, citations=ans.citations)
        return QueryResult(question, hits, answer=None, citations=self._cites_from_hits(hits))

    def query_stream(self, question: str, k: Optional[int] = None) -> Iterator[str]:
        hits = self.search(question, k=k)
        if self.config.generate_enabled and self.llm is not None:
            yield from self._generator().stream(question, hits)
        else:
            for i, h in enumerate(hits, 1):
                yield f"[{i}] {h.text}\n\n"

    def estimate_index_cost(self, source: Source) -> CostEstimate:
        docs = list(self._documents(source))
        items = [(d.text, c) for d in docs for c in self.chunker.chunk(d)]
        est = (self._estimate_compile_cost(items) if self.config.compile_enabled
               else self._estimate_contextual_cost(items))
        return est or CostEstimate(0, 0, 0, 0.0)

    # ------------------------------------------------------------------ helpers
    def _documents(self, source: Source) -> Iterator[Document]:
        if isinstance(source, Document):
            yield source
            return
        if isinstance(source, str):
            yield from load_paths(source)
            return
        items = list(source)
        if items and all(isinstance(it, Document) for it in items):
            yield from items  # type: ignore[misc]
        else:
            yield from load_paths([it for it in items if isinstance(it, str)])

    def _field_weights(self) -> FieldWeights:
        return FieldWeights(
            body=self.config.weight_body,
            ngram=self.config.weight_ngram if self.config.enable_ngram else 0.0,
            title=self.config.weight_title,
        )

    def _expand(self, query: str) -> str:
        exp = self._expander()
        return exp.expand(query).assembled if exp is not None else query

    def _expander(self):
        if self._exp is None and self.llm is not None and self.config.expand_enabled:
            from .augment.expand import QueryExpander

            self._exp = QueryExpander(self.llm, self.config)
        return self._exp if (self.llm is not None and self.config.expand_enabled) else None

    def _contextualize(self, items) -> int:
        if self.llm is None or not self.config.contextual_enabled or not items:
            return 0
        return self._contextual_indexer().contextualize(items)

    def _estimate_contextual_cost(self, items) -> Optional[CostEstimate]:
        if self.llm is None or not self.config.contextual_enabled or not items:
            return None
        return self._contextual_indexer().estimate(items)

    def _contextual_indexer(self):
        if self._ctx is None:
            from .augment.contextual import ContextualIndexer

            self._ctx = ContextualIndexer(self.llm, self.config, self.config.path)
        return self._ctx

    def _compile(self, items) -> tuple[int, dict]:
        """Run the offline compiler; returns (n_enriched, {chunk_id: csc_term_weights})."""
        if self.llm is None or not self.config.compile_enabled or not items:
            return 0, {}
        bundles = self._compiler().compile(items)
        n = sum(1 for b in bundles.values() if not b.is_empty())
        weights = {cid: b.term_weights for cid, b in bundles.items() if b.term_weights}
        return n, weights

    def _estimate_compile_cost(self, items) -> Optional[CostEstimate]:
        if self.llm is None or not self.config.compile_enabled or not items:
            return None
        return self._compiler().estimate(items)

    def _compiler(self):
        if self._cmp is None:
            from .augment.compiler import Compiler

            self._cmp = Compiler(self.llm, self.config, self.config.path)
        return self._cmp

    def _generator(self):
        if self._gen is None:
            from .generate.answer import Generator

            self._gen = Generator(self.llm, self.config)
        return self._gen

    @staticmethod
    def _cites_from_hits(hits: List[Hit]) -> List[Citation]:
        return [
            Citation(marker=f"[{i}]", source=h.source, chunk_id=h.chunk_id, score=h.score)
            for i, h in enumerate(hits, 1)
        ]

    # ------------------------------------------------------------------ manifest
    def _merge_manifest(self, path: str) -> None:
        mpath = os.path.join(path, _MANIFEST)
        if not os.path.exists(mpath):
            return
        try:
            with open(mpath, "r", encoding="utf-8") as fh:
                m = json.load(fh)
        except Exception:
            return
        # persisted index settings must win so analyzers match what's on disk
        self.config = self.config.with_overrides(
            engine=m.get("engine"), language=m.get("language"),
        )
        if "enable_ngram" in m:
            from dataclasses import replace

            self.config = replace(self.config, enable_ngram=bool(m["enable_ngram"]))

    def _legb_store_exists(self) -> bool:
        p = self.config.path
        return bool(p) and os.path.exists(os.path.join(p, ".kapi_csc", "legb.json"))

    def _write_manifest(self, path: str) -> None:
        mpath = os.path.join(path, _MANIFEST)
        if os.path.exists(mpath):
            return
        data = {
            "kapi_version": "0.1.0",
            "engine": self.config.engine,
            "language": self.config.language,
            "enable_ngram": self.config.enable_ngram,
        }
        try:
            with open(mpath, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError:
            pass

    # ------------------------------------------------------------------ portable bundles (Phase 5)
    def export_bundle(self, dest: str, *, include_cache: bool = False):
        """Export this on-disk index as a portable, air-gapped serving bundle (STRATEGY §8).

        The index must be persisted (``path`` set). Pending mutations are flushed, then the
        directory is archived into ``dest`` (a ``.kapi.tgz``). The bundle opens anywhere with
        :meth:`import_bundle` / :meth:`open` and serves model-free — no LLM, no network.
        """
        if not self.config.path:
            raise ValueError("in-memory index has nothing to export; construct Kapi(path=...)")
        self.engine.commit()
        self.store.commit()
        if self._legb is not None:
            self._legb.commit()
        from .portable import export_index

        return export_index(self.config.path, dest, include_cache=include_cache)

    @classmethod
    def import_bundle(cls, bundle: str, dest: str, *, llm=None, overwrite: bool = False,
                      **open_overrides) -> "Kapi":
        """Unpack a serving bundle into ``dest`` and open it (model-free by default)."""
        from .portable import import_index

        import_index(bundle, dest, overwrite=overwrite)
        return cls.open(dest, llm=llm, **open_overrides)

    # ------------------------------------------------------------------ lifecycle
    def stats(self) -> dict:
        s = {"engine": self.engine.stats(), "store": self.store.stats(),
             "preset": self.config.preset, "llm": self.llm is not None}
        if self._legb is not None:
            s["csc"] = self._legb.stats()
        return s

    def close(self) -> None:
        try:
            if self._legb is not None:
                self._legb.commit()
        finally:
            try:
                self.engine.close()
            finally:
                self.store.close()

    def __enter__(self) -> "Kapi":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
