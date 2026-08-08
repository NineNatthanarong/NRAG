"""Default engine: Tantivy (Rust, Lucene-class, persistent + incremental).

One schema, multiple analyzed fields → multi-signal BM25 in a single engine-fused query:
  - ``body``  : word analyzer (lowercase + ascii-fold + stopword + Snowball stemmer)
  - ``ngram`` : char-trigram analyzer (typo / morphology / multilingual robustness)
  - ``title`` : word analyzer, boosted
  - ``chunk_id`` / ``doc_id`` / ``section`` : ``raw`` tokenizer (exact match for delete/filter)

The query is built from explicit per-field ``term_query``s (no query-parser, so arbitrary
user/expanded text can never trigger a syntax error) OR-combined and per-field boosted.
The ngram signal in particular MUST be an OR-of-trigrams: ``parse_query`` builds a
position-sensitive phrase query over the trigrams, which defeats typo tolerance — verified
during design.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

from .._types import Chunk, EngineConfig, FieldWeights, Hit, MetaFilter


def _require_tantivy():
    try:
        import tantivy  # type: ignore

        return tantivy
    except Exception as exc:  # pragma: no cover - core dep, but be explicit
        raise RuntimeError(
            "The Tantivy engine requires the 'tantivy' package (a core dependency). "
            "Install it with: pip install tantivy"
        ) from exc


class TantivyEngine:
    def __init__(self, index, schema, analyzers, config: EngineConfig, path: Optional[str]):
        self._tantivy = _require_tantivy()
        self.index = index
        self.schema = schema
        self._word_analyzer, self._ngram_analyzer = analyzers
        self.config = config
        self.path = path
        self._writer = None

    # ------------------------------------------------------------------ open
    @classmethod
    def open(
        cls,
        path: Optional[str] = None,
        *,
        create: bool = True,
        config: Optional[EngineConfig] = None,
    ) -> "TantivyEngine":
        tantivy = _require_tantivy()
        config = config or EngineConfig()
        schema = cls._build_schema(tantivy, config)

        if path is None:
            index = tantivy.Index(schema)
        else:
            os.makedirs(path, exist_ok=True)
            if os.path.exists(os.path.join(path, "meta.json")):
                index = tantivy.Index.open(path)
            elif create:
                index = tantivy.Index(schema, path=path)
            else:
                raise FileNotFoundError(f"no Tantivy index at {path!r}")

        analyzers = cls._register_analyzers(tantivy, index, config)
        return cls(index, schema, analyzers, config, path)

    # ------------------------------------------------------------------ schema/analyzers
    @staticmethod
    def _build_schema(tantivy, config: EngineConfig):
        sb = tantivy.SchemaBuilder()
        sb.add_text_field("chunk_id", stored=True, tokenizer_name="raw")
        sb.add_text_field("doc_id", stored=True, tokenizer_name="raw")
        sb.add_text_field("section", stored=True, tokenizer_name="raw")
        sb.add_text_field("body", stored=False, tokenizer_name="nrag_words",
                          index_option="position")
        if config.enable_ngram:
            sb.add_text_field("ngram", stored=False, tokenizer_name="nrag_ngram",
                              index_option="position")
        sb.add_text_field("title", stored=True, tokenizer_name="nrag_words",
                          index_option="position")
        return sb.build()

    @staticmethod
    def _register_analyzers(tantivy, index, config: EngineConfig):
        Tokenizer, Filter, TAB = tantivy.Tokenizer, tantivy.Filter, tantivy.TextAnalyzerBuilder
        wb = TAB(Tokenizer.simple()).filter(Filter.remove_long(40)).filter(Filter.lowercase())
        if config.ascii_fold:
            wb = wb.filter(Filter.ascii_fold())
        if config.stopwords:
            wb = wb.filter(Filter.stopword(config.language))
        if config.enable_stemming:
            wb = wb.filter(Filter.stemmer(config.language))
        word_analyzer = wb.build()
        index.register_tokenizer("nrag_words", word_analyzer)

        ngram_analyzer = None
        if config.enable_ngram:
            nb = TAB(Tokenizer.ngram(config.ngram_min, config.ngram_max, False)).filter(
                Filter.lowercase())
            if config.ascii_fold:
                nb = nb.filter(Filter.ascii_fold())
            ngram_analyzer = nb.build()
            index.register_tokenizer("nrag_ngram", ngram_analyzer)
        return word_analyzer, ngram_analyzer

    # ------------------------------------------------------------------ writer
    @property
    def writer(self):
        if self._writer is None:
            self._writer = self.index.writer(self.config.writer_heap_bytes,
                                             self.config.writer_threads)
        return self._writer

    # ------------------------------------------------------------------ mutation
    def add(self, chunks: Iterable[Chunk]) -> int:
        tantivy = self._tantivy
        w = self.writer
        n = 0
        for c in chunks:
            doc = tantivy.Document()
            doc.add_text("chunk_id", c.chunk_id)
            doc.add_text("doc_id", c.doc_id)
            doc.add_text("section", c.section or "")
            doc.add_text("body", c.indexed_text)
            if self.config.enable_ngram:
                doc.add_text("ngram", c.indexed_text)
            doc.add_text("title", c.title or "")
            w.add_document(doc)
            n += 1
        return n

    def update(self, chunk_id: str, chunk: Chunk) -> None:
        self.writer.delete_documents("chunk_id", chunk_id)
        self.add([chunk])

    def delete(self, chunk_ids: Iterable[str]) -> int:
        w = self.writer
        n = 0
        for cid in chunk_ids:
            w.delete_documents("chunk_id", cid)
            n += 1
        return n

    def delete_doc(self, doc_id: str) -> int:
        self.writer.delete_documents("doc_id", doc_id)
        return 1

    def commit(self) -> None:
        if self._writer is not None:
            self._writer.commit()
            self._writer.wait_merging_threads()
            self._writer = None  # writer is consumed by wait_merging_threads
        self.index.reload()

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        *,
        k: int = 10,
        field_weights: FieldWeights = FieldWeights(),
        fuzzy: bool = False,
        filter: Optional[MetaFilter] = None,
    ) -> List[Hit]:
        tantivy = self._tantivy
        Query, Occur = tantivy.Query, tantivy.Occur

        score_clauses = self._field_clauses(query, field_weights, fuzzy)
        if not score_clauses:
            return []
        score_query = Query.boolean_query(score_clauses)

        filter_clauses = self._filter_clauses(filter)
        if filter_clauses:
            final = Query.boolean_query([(Occur.Must, score_query), *filter_clauses])
        else:
            final = score_query

        searcher = self.index.searcher()
        result = searcher.search(final, k)
        hits: List[Hit] = []
        for rank, (score, addr) in enumerate(result.hits, start=1):
            cid = searcher.doc(addr).get_first("chunk_id")
            hits.append(Hit(chunk_id=cid, score=float(score), rank=rank, signal="fused"))
        return hits

    def _field_clauses(self, query: str, fw: FieldWeights, fuzzy: bool):
        Occur = self._tantivy.Occur
        clauses = []

        word_terms = self._word_analyzer.analyze(query)
        if word_terms:
            if fw.body > 0:
                clauses.append((Occur.Should,
                                self._field_or("body", word_terms, fw.body, fuzzy)))
            if fw.title > 0:
                clauses.append((Occur.Should,
                                self._field_or("title", word_terms, fw.title, fuzzy)))

        if self.config.enable_ngram and self._ngram_analyzer is not None and fw.ngram > 0:
            ngram_terms = self._ngram_analyzer.analyze(query)
            if ngram_terms:
                clauses.append((Occur.Should,
                                self._field_or("ngram", ngram_terms, fw.ngram, False)))
        return clauses

    def _field_or(self, field: str, terms, weight: float, fuzzy: bool):
        tantivy = self._tantivy
        Query, Occur = tantivy.Query, tantivy.Occur
        subs = []
        seen = set()
        for t in terms:
            if t in seen:
                continue
            seen.add(t)
            if fuzzy and field == "body":
                q = Query.fuzzy_term_query(self.schema, field, t, 1, True, False)
            else:
                q = Query.term_query(self.schema, field, t)
            subs.append((Occur.Should, q))
        or_query = Query.boolean_query(subs)
        return Query.boost_query(or_query, weight) if weight != 1.0 else or_query

    def _filter_clauses(self, filter: Optional[MetaFilter]):
        if filter is None or filter.is_empty():
            return []
        tantivy = self._tantivy
        Query, Occur = tantivy.Query, tantivy.Occur
        clauses = []
        for key in ("doc_id", "section"):
            if key in filter.equals:
                clauses.append((Occur.Must,
                                Query.term_query(self.schema, key, str(filter.equals[key]))))
            if key in filter.any_of and filter.any_of[key]:
                subs = [(Occur.Should, Query.term_query(self.schema, key, str(v)))
                        for v in filter.any_of[key]]
                clauses.append((Occur.Must, Query.boolean_query(subs)))
        return clauses

    # ------------------------------------------------------------------ misc
    @property
    def supports_incremental(self) -> bool:
        return True

    @property
    def supports_multifield(self) -> bool:
        return True

    @property
    def supports_prefilter(self) -> bool:
        return True

    #: filter fields this engine can push down into the index query
    PREFILTER_FIELDS = frozenset({"doc_id", "section"})

    def prefilter_covers(self, filter: Optional[MetaFilter]) -> bool:
        """Only ``doc_id``/``section`` clauses are translated by ``_filter_clauses``;
        arbitrary metadata keys and ``mtime_after`` need the residual post-filter."""
        if filter is None or filter.is_empty():
            return True
        if filter.mtime_after is not None:
            return False
        keys = set(filter.equals) | set(filter.any_of)
        return keys <= self.PREFILTER_FIELDS

    def stats(self) -> dict:
        try:
            num = self.index.searcher().num_docs
        except Exception:
            num = None
        return {"engine": "tantivy", "path": self.path, "num_chunks": num,
                "ngram": self.config.enable_ngram}

    def close(self) -> None:
        if self._writer is not None:
            self._writer.commit()
            self._writer.wait_merging_threads()
            self._writer = None
