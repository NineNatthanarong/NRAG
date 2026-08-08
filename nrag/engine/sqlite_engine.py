"""Optional zero-dependency engine: SQLite FTS5.

Uses only the stdlib ``sqlite3`` (FTS5 ships in standard CPython builds). Because FTS5
binds one tokenizer per table, the word signal and the char-trigram signal live in two
tables; this engine is therefore single-field per query (``supports_multifield=False``),
so the retrieve layer fuses the per-signal result lists with RRF (Path B).

Note: SQLite ``bm25()`` returns *negative* values (more relevant = more negative); we
negate so higher == better, consistent with the other engines.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterable, List, Optional

from .._types import Chunk, EngineConfig, FieldWeights, Hit, MetaFilter
from ..tokenize.ngram import CharNgramTokenizer
from ..tokenize.text import WordTokenizer


def _probe_fts5() -> None:
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE VIRTUAL TABLE _p USING fts5(x)")
    except sqlite3.OperationalError as exc:  # pragma: no cover - rare minimal builds
        raise RuntimeError(
            "SQLite FTS5 is not available in this Python's sqlite3 build; "
            "use the default Tantivy engine instead."
        ) from exc
    finally:
        con.close()


class SQLiteFTS5Engine:
    def __init__(self, conn, config: EngineConfig, path: Optional[str]):
        self.conn = conn
        self.config = config
        self.path = path
        self._words = WordTokenizer(config.language, stopwords=config.stopwords,
                                    stemming=config.enable_stemming)
        self._ngram = CharNgramTokenizer(config.ngram_min)

    # ------------------------------------------------------------------ open
    @classmethod
    def open(cls, path: Optional[str] = None, *, create: bool = True,
             config: Optional[EngineConfig] = None) -> "SQLiteFTS5Engine":
        _probe_fts5()
        config = config or EngineConfig()
        db = ":memory:" if path is None else os.path.join(path, "fts.sqlite")
        if path:
            os.makedirs(path, exist_ok=True)
        conn = sqlite3.connect(db, check_same_thread=False)
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
            "chunk_id UNINDEXED, doc_id UNINDEXED, section UNINDEXED, body, title, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        if config.enable_ngram:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_tri USING fts5("
                "chunk_id UNINDEXED, doc_id UNINDEXED, section UNINDEXED, body, "
                "tokenize='trigram')"
            )
        conn.commit()
        return cls(conn, config, path)

    # ------------------------------------------------------------------ mutation
    def add(self, chunks: Iterable[Chunk]) -> int:
        n = 0
        for c in chunks:
            self.conn.execute(
                "INSERT INTO chunks_fts (chunk_id, doc_id, section, body, title) "
                "VALUES (?,?,?,?,?)",
                (c.chunk_id, c.doc_id, c.section or "", c.indexed_text, c.title or ""),
            )
            if self.config.enable_ngram:
                self.conn.execute(
                    "INSERT INTO chunks_tri (chunk_id, doc_id, section, body) VALUES (?,?,?,?)",
                    (c.chunk_id, c.doc_id, c.section or "", c.indexed_text))
            n += 1
        return n

    def update(self, chunk_id: str, chunk: Chunk) -> None:
        self.delete([chunk_id])
        self.add([chunk])

    def delete(self, chunk_ids: Iterable[str]) -> int:
        n = 0
        for cid in chunk_ids:
            self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (cid,))
            if self.config.enable_ngram:
                self.conn.execute("DELETE FROM chunks_tri WHERE chunk_id=?", (cid,))
            n += 1
        return n

    def delete_doc(self, doc_id: str) -> int:
        cids = [r[0] for r in self.conn.execute(
            "SELECT chunk_id FROM chunks_fts WHERE doc_id=?", (doc_id,)).fetchall()]
        self.delete(cids)
        return len(cids)

    def commit(self) -> None:
        self.conn.commit()

    # ------------------------------------------------------------------ search
    def search(self, query: str, *, k: int = 10, signal: str = "body",
               field_weights: FieldWeights = FieldWeights(), fuzzy: bool = False,
               filter: Optional[MetaFilter] = None) -> List[Hit]:
        if signal == "ngram":
            if not self.config.enable_ngram:
                return []
            terms = self._ngram(query)
            return self._run("chunks_tri", terms, None, k, "ngram", filter)
        terms = self._words(query)
        column = "title" if signal == "title" else "body"
        return self._run("chunks_fts", terms, column, k, signal, filter)

    def _run(self, table: str, terms: List[str], column: Optional[str], k: int,
             signal: str, filter: Optional[MetaFilter]) -> List[Hit]:
        terms = [t for t in dict.fromkeys(terms) if t]
        if not terms:
            return []
        or_expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        match = f'{column} : ({or_expr})' if column else or_expr
        where = [f"{table} MATCH ?"]
        params: list = [match]
        where += self._filter_sql(table, filter, params)
        sql = (f"SELECT chunk_id, bm25({table}) AS r FROM {table} "
               f"WHERE {' AND '.join(where)} ORDER BY r LIMIT ?")
        params.append(k)
        rows = self.conn.execute(sql, params).fetchall()
        return [Hit(chunk_id=cid, score=-float(r), rank=i, signal=signal)
                for i, (cid, r) in enumerate(rows, start=1)]

    @staticmethod
    def _filter_sql(table: str, filter: Optional[MetaFilter], params: list) -> List[str]:
        clauses: List[str] = []
        if filter is None or filter.is_empty():
            return clauses
        for key in ("doc_id", "section"):
            if key in filter.equals:
                clauses.append(f"{key} = ?")
                params.append(str(filter.equals[key]))
            if filter.any_of.get(key):
                marks = ",".join("?" * len(filter.any_of[key]))
                clauses.append(f"{key} IN ({marks})")
                params.extend(str(v) for v in filter.any_of[key])
        return clauses

    # ------------------------------------------------------------------ misc
    @property
    def supports_incremental(self) -> bool:
        return True

    @property
    def supports_multifield(self) -> bool:
        return False

    @property
    def supports_prefilter(self) -> bool:
        return True

    #: filter fields this engine can push down into the FTS query
    PREFILTER_FIELDS = frozenset({"doc_id", "section"})

    def prefilter_covers(self, filter: Optional[MetaFilter]) -> bool:
        """Only ``doc_id``/``section`` clauses are translated by ``_filter_sql``;
        arbitrary metadata keys and ``mtime_after`` need the residual post-filter."""
        if filter is None or filter.is_empty():
            return True
        if filter.mtime_after is not None:
            return False
        keys = set(filter.equals) | set(filter.any_of)
        return keys <= self.PREFILTER_FIELDS

    def stats(self) -> dict:
        n = self.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        return {"engine": "sqlite", "path": self.path, "num_chunks": n,
                "ngram": self.config.enable_ngram}

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
