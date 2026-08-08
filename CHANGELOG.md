# Changelog

All notable changes to NRAG are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/)
(pre-1.0: minor versions may break APIs).

## [0.1.4] — unreleased

Retrieval-quality release: the lexical core was benchmarked on four BEIR datasets and
re-tuned. **nDCG@10 improves on every dataset×engine tested, up to +0.066, at $0 query
cost.** Full grids and reproduction commands: `benchmarks/lexical_tuning_v014.md`.

### Changed (affects ranking — re-run your own evals if you tuned weights)

- **Default field weights**: `ngram 0.6 → 0.3`, `title 2.5 → 0.5`. The old title boost
  was the single largest source of lost precision on all four benchmarked datasets
  (SciFact +0.021, NFCorpus +0.011, FiQA +0.008, ArguAna +0.004 nDCG@10 vs old defaults).
- **Single-field engines (SQLite/bm25s path) now honor `FieldWeights`**: legs are fused
  by weighted convex combination (min-max normalized) instead of unweighted RRF, matching
  Tantivy's semantics. SciFact/sqlite: **0.6304 → 0.6965 (+0.066)**. `fusion="rrf"` remains
  available; `rrf_k` default 60 → 10 (low k won every RRF sweep). Zero-weight legs are
  skipped entirely.
- The two-leg CSC fusion is decoupled from the engine fusion default via `csc_fusion`
  (stays `"rrf"`, the setting the compiled-preset results were validated with).

### Added

- **`Nrag.search_docs(query, k, agg="max"|"sum")`** — document-level retrieval (one hit
  per source document, aggregated over its chunks).
- **Tunable BM25 parameters**: `Nrag(bm25_k1=..., bm25_b=...)`, honored by the bm25s
  engine (tantivy/SQLite ship fixed upstream parameters and document that).
- **Dependency-free BEIR evaluation**: `nrag.eval.load_beir` falls back to a
  standard-library loader when the `beir` package is absent — quality evals now run on
  the core install.
- **Checked-in, reproducible benchmark harness** (`benchmarks/lexical_grid.py`) with
  per-run provenance logging (`benchmarks/grid_results.jsonl`) — replaces the
  previously-referenced `scratch/` scripts.

## [0.1.3] — unreleased

### Fixed

- **Metadata filters are now enforced on every engine.** Tantivy and SQLite only push down
  `doc_id`/`section` clauses; filters on arbitrary metadata keys or `mtime_after` were
  silently ignored and returned unfiltered results. Engines now declare exactly what they
  cover (`prefilter_covers`) and the retrieve layer post-filters the rest. If you relied on
  metadata filters for tenancy or permissions, upgrade immediately.
- **`compile_reasoning=False` now actually disables the reasoning/inferences section** of the
  compiler (it was a no-op due to a wrong config attribute lookup) and participates in the
  compile cache key.
- **Version is single-sourced.** The index manifest and `nrag serve` reported `0.1.0`
  regardless of the installed version; both now derive from `nrag.__version__`, and the CLI
  gained `nrag --version`.
- `Config.from_preset()` raises `ValueError` for unknown presets instead of silently
  falling back to `quality`.

### Changed

- Removed the unused `html` extra (`selectolax` was declared but never imported; the HTML
  loader uses the standard library). `pip install nrag[html]` should be dropped from install
  commands.
- The RAGAS runner's install hint now points at the real extra: `pip install "nrag[eval]"`.

### Added

- CI: test matrix (Python 3.10–3.13, Linux/macOS/Windows), ruff lint, mypy baseline,
  wheel/sdist build + clean-venv install smoke test. Publishing now runs the test suite first.
- `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff`, this changelog.

## [0.1.2] — 2026-07-01

Initial public release line: compiler + CSC consensus weighting, two-leg sparse fusion,
adaptive router, portable bundles, hosted compile service, TCO model, LangChain/LlamaIndex
adapters, BEIR/BRIGHT runners.
