"""Word tokenizer with a graceful stemmer fallback.

Used by the Python-side engines (SQLite, bm25s) and by eval/debugging. The default
Tantivy engine tokenizes inside Rust via registered analyzers, so this is *not* on
its hot path — but the pure-Python fallback chain matters precisely because the other
engines depend on it: PyStemmer (C, fastest) -> snowballstemmer (pure-Python wheel)
-> identity + one-time warning. Nothing here ever hard-requires a C extension.
"""

from __future__ import annotations

import re
import warnings
from typing import Callable, Optional

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# A short, conservative English stopword list (user-overridable). Kept intentionally
# small: aggressive stopwording hurts more than it helps for BM25 in practice.
DEFAULT_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with", "from", "we", "you", "your", "i", "he", "she", "his", "her"]
)

# snowballstemmer language names differ slightly from common usage; map the ones we use.
_SNOWBALL_LANGS = {
    "english", "porter", "french", "spanish", "german", "italian", "portuguese",
    "dutch", "swedish", "norwegian", "danish", "finnish", "russian", "hungarian",
    "romanian", "turkish", "arabic",
}


def _resolve_stemmer(language: str) -> tuple[Optional[Callable[[str], str]], str]:
    """Return (stem_fn, backend_name). stem_fn is None for the identity fallback."""
    lang = language.lower()
    # 1) PyStemmer (C extension, fastest) — may be absent or fail to build.
    try:
        import Stemmer  # type: ignore

        st = Stemmer.Stemmer(lang)
        return (lambda w: st.stemWord(w)), "pystemmer"
    except Exception:
        pass
    # 2) snowballstemmer (pure-Python, always ships wheels) — the safe default.
    try:
        import snowballstemmer  # type: ignore

        name = lang if lang in _SNOWBALL_LANGS else "english"
        st = snowballstemmer.stemmer(name)
        return (lambda w: st.stemWord(w)), "snowball"
    except Exception:
        warnings.warn(
            "No stemmer available (PyStemmer/snowballstemmer not importable); "
            "falling back to identity stemming.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, "none"


class WordTokenizer:
    """Lowercase -> split on \\w+ -> drop stopwords/short tokens -> stem."""

    def __init__(
        self,
        language: str = "english",
        *,
        stopwords: bool | frozenset[str] = True,
        stemming: bool = True,
        min_len: int = 1,
        max_len: int = 40,
    ) -> None:
        self.language = language
        if stopwords is True:
            self.stopwords = DEFAULT_STOPWORDS
        elif stopwords is False:
            self.stopwords = frozenset()
        else:
            self.stopwords = frozenset(stopwords)
        self.min_len = min_len
        self.max_len = max_len
        if stemming:
            self._stem, self.stemmer_backend = _resolve_stemmer(language)
        else:
            self._stem, self.stemmer_backend = None, "disabled"

    def __call__(self, text: str) -> list[str]:
        out: list[str] = []
        stop = self.stopwords
        stem = self._stem
        for raw in _WORD_RE.findall(text.lower()):
            if len(raw) > self.max_len or len(raw) < self.min_len:
                continue
            if raw in stop:
                continue
            out.append(stem(raw) if stem else raw)
        return out
