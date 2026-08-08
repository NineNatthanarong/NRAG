# Changelog

All notable changes to NRAG are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/)
(pre-1.0: minor versions may break APIs).

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
