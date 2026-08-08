# Contributing to NRAG

Thanks for your interest. NRAG is early and moving fast — small, focused PRs are the easiest to review and merge.

## Development setup

```bash
git clone https://github.com/NineNatthanarong/NRAG
cd NRAG
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # extras: openai, bm25s, eval + pytest/ruff/mypy
```

## Running tests

```bash
python -m pytest -q -m "not eval and not live_llm"   # the default deterministic suite
```

Test markers:

- **(no marker)** — deterministic unit tests. Must pass offline with the core install (plus extras if present). CI runs these.
- **`eval`** — dataset-backed evaluation (BEIR/BRIGHT downloads, `nrag[eval]`). Opt-in: `pytest -m eval`.
- **`live_llm`** — tests that call a real LLM endpoint. Opt-in and requires `NRAG_LLM_BASE_URL`, `NRAG_LLM_MODEL`, `NRAG_LLM_API_KEY`. These can cost money; keep new ones tiny.

## Style & checks

```bash
ruff check nrag tests
ruff format nrag tests
mypy nrag --ignore-missing-imports
```

CI runs lint on every PR. Type checking is currently a non-blocking baseline that we tighten over time — new code should be typed.

## What makes a good PR

- A failing test first for bug fixes (all recent correctness fixes ship with regression tests — please keep that bar).
- Engines must keep their capability contracts honest: if your engine cannot push down a filter clause or a config option, say so via `prefilter_covers` / capability properties rather than silently ignoring it.
- Benchmark claims need a checked-in, runnable script and stated model/dataset/seed. No screenshots without a reproduction path.
- Keep the core dependency-light. New required dependencies need a strong reason; prefer optional extras.

## Reporting bugs

Open a GitHub issue with a minimal reproduction (a few lines using `add_texts` + `search` is perfect). For security issues, see [SECURITY.md](SECURITY.md) — please do not open public issues for vulnerabilities.
