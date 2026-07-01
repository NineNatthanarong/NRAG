"""``kapi`` command line — the Compiled-Retrieval mental model: ``compile`` then ``query``.

    kapi compile ./docs --index ./idx --preset compiled --base-url ... --model ...
    kapi query  "how do refunds work?" --index ./idx
    kapi stats  --index ./idx

An LLM is optional. With ``--base-url`` (or ``KAPI_LLM_BASE_URL``) a chunk is compiled
offline into its lexical closure; with no LLM, ``compile`` builds a pure-lexical index and
``query`` returns ranked chunks (no generated answer). All retrieval stays embedding-free.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


def _build_llm(args) -> Optional[object]:
    base_url = args.base_url or os.environ.get("KAPI_LLM_BASE_URL")
    if not base_url:
        return None
    from .llm import OpenAICompatLLM

    model = args.model or os.environ.get("KAPI_LLM_MODEL") or "gpt-4o-mini"
    api_key = args.api_key or os.environ.get("KAPI_LLM_API_KEY") or "not-needed"
    return OpenAICompatLLM(base_url=base_url, model=model, api_key=api_key)


def _open_kapi(args, *, llm):
    from . import Kapi

    return Kapi(llm=llm, preset=args.preset, path=args.index, engine=args.engine)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index", default=None, help="on-disk index dir (default: in-memory)")
    p.add_argument("--preset", default="compiled", choices=("compiled", "quality", "fast"))
    p.add_argument("--engine", default=None, choices=("tantivy", "sqlite", "bm25s"))
    p.add_argument("--base-url", default=None, help="OpenAI-compatible LLM endpoint")
    p.add_argument("--model", default=None, help="LLM model id")
    p.add_argument("--api-key", default=None, help="LLM api key (default: not-needed)")


def cmd_compile(args) -> int:
    llm = _build_llm(args)
    if llm is None and args.preset == "compiled":
        print("note: no --base-url/LLM -> compiling a pure-lexical index (no enrichment).",
              file=sys.stderr)
    rag = _open_kapi(args, llm=llm)
    try:
        rep = rag.compile(args.source, force=args.force)
        print(rep)
        print(rag.stats())
    finally:
        rag.close()
    return 0


def cmd_query(args) -> int:
    llm = _build_llm(args)
    rag = _open_kapi(args, llm=llm)
    try:
        res = rag.query(args.text, k=args.k)
        if res.answer is not None:
            print(res.answer)
            print("\nsources:")
        for i, h in enumerate(res.hits, 1):
            print(f"  [{i}] {h.score:.4f}  {h.source}")
            if not res.answer:
                print(f"       {h.text[:160].strip()}")
    finally:
        rag.close()
    return 0


def cmd_stats(args) -> int:
    rag = _open_kapi(args, llm=None)
    try:
        print(rag.stats())
    finally:
        rag.close()
    return 0


def cmd_export(args) -> int:
    from .portable import export_index

    info = export_index(args.index, args.out, include_cache=args.include_cache)
    print(f"exported {info.num_files} files -> {info.path} "
          f"(engine={info.engine}, cache={'yes' if info.includes_cache else 'no'})")
    return 0


def cmd_import(args) -> int:
    from .portable import import_index, read_bundle_manifest

    man = read_bundle_manifest(args.bundle)
    import_index(args.bundle, args.index, overwrite=args.overwrite)
    print(f"imported bundle -> {args.index} (engine={man.get('engine')}); "
          f'serve with:  kapi query "..." --index {args.index}')
    return 0


def cmd_serve(args) -> int:
    from .service import serve

    llm = _build_llm(args)
    if llm is None:
        print("note: no --base-url/LLM -> service compiles pure-lexical bundles (no enrichment).",
              file=sys.stderr)
    serve(host=args.host, port=args.port, llm=llm, preset=args.preset, engine=args.engine)
    return 0


def cmd_tco(args) -> int:
    from .tco import TCOInputs, compute_tco, format_report

    inp = TCOInputs(n_docs=args.docs, tokens_per_doc=args.tokens_per_doc,
                    queries_per_month=args.queries_per_month,
                    tokens_per_query=args.tokens_per_query, months=args.months,
                    consensus_k=args.consensus_k, embedding_dims=args.dims)
    print(format_report(inp, compute_tco(inp)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kapi",
                                     description="Compiled Retrieval — embedding-free RAG.")
    sub = parser.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("compile", help="compile/index a dir, file, or glob")
    pc.add_argument("source", help="path, file, or glob to ingest")
    pc.add_argument("--force", action="store_true", help="recompile even if unchanged")
    _add_common(pc)
    pc.set_defaults(func=cmd_compile)

    pq = sub.add_parser("query", help="query the index")
    pq.add_argument("text", help="the query string")
    pq.add_argument("-k", type=int, default=10, help="top-k results")
    _add_common(pq)
    pq.set_defaults(func=cmd_query)

    ps = sub.add_parser("stats", help="show index stats")
    _add_common(ps)
    ps.set_defaults(func=cmd_stats)

    # ---- Phase 5: portable bundles + hosted compilation service ----
    pe = sub.add_parser("export", help="export an index as a portable serving bundle")
    pe.add_argument("--index", required=True, help="index dir to export")
    pe.add_argument("--out", required=True, help="destination .kapi.tgz bundle")
    pe.add_argument("--include-cache", action="store_true",
                    help="also bundle the offline LLM compile cache (bigger; free re-compiles)")
    pe.set_defaults(func=cmd_export)

    pi = sub.add_parser("import", help="unpack a serving bundle into an index dir")
    pi.add_argument("bundle", help="the .kapi.tgz bundle to unpack")
    pi.add_argument("--index", required=True, help="destination index dir")
    pi.add_argument("--overwrite", action="store_true", help="replace a non-empty dest")
    pi.set_defaults(func=cmd_import)

    pv = sub.add_parser("serve", help="run the hosted compilation service")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)
    pv.add_argument("--preset", default="compiled", choices=("compiled", "quality", "fast"))
    pv.add_argument("--engine", default=None, choices=("tantivy", "sqlite", "bm25s"))
    pv.add_argument("--base-url", default=None, help="OpenAI-compatible LLM endpoint")
    pv.add_argument("--model", default=None, help="LLM model id")
    pv.add_argument("--api-key", default=None, help="LLM api key (default: not-needed)")
    pv.set_defaults(func=cmd_serve)

    pt = sub.add_parser("tco", help="model KAPI vs dense+vectorDB total cost of ownership")
    pt.add_argument("--docs", type=int, default=1_000_000)
    pt.add_argument("--tokens-per-doc", type=int, default=500)
    pt.add_argument("--queries-per-month", type=int, default=1_000_000)
    pt.add_argument("--tokens-per-query", type=int, default=20)
    pt.add_argument("--months", type=int, default=12)
    pt.add_argument("--consensus-k", type=int, default=3)
    pt.add_argument("--dims", type=int, default=1536, help="dense embedding dimensions")
    pt.set_defaults(func=cmd_tco)
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
