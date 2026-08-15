"""Render benchmarks/dense_results.jsonl as assets/dense-bench.svg (no deps).

A small-multiples chart: one horizontal-bar panel per BEIR dataset (scifact, nfcorpus,
fiqa), bars = nDCG@10 per system. Missing data is skipped entirely (e.g.
text-embedding-3-large on fiqa, doc2query x2 outside scifact) — no placeholder bars.
Colors follow assets/perf-scifact.svg: purple gradient = Nrag (no embeddings, $0/query
or offline-LLM tier), lavender = lexical reference ($0/query), gray = dense embedder
(model pass per query).

Usage:  python benchmarks/plot_dense.py [--json benchmarks/dense_results.jsonl] [--out assets/dense-bench.svg]
"""

from __future__ import annotations

import argparse
import json
import math

W, H = 1200, 1240
CARD = "#ffffff"
STROKE = "#e2e8f0"
INK = "#0f172a"
MUTED = "#475569"
AXIS = "#64748b"
GRID = "#e2e8f0"
GRAY = "#cbd5e1"          # dense embedder bars
LAV = "#ddd6fe"           # lexical reference bars ($0)
PURPLE = "#6d28d9"        # Nrag text
GRAD = "url(#nrag)"

ORDER = [
    "text-embedding-3-large", "qwen3-embedding-4b", "text-embedding-3-small",
    "NRAG + doc2query x2 (prior run)", "NRAG (pure lexical)",
    "normal FTS (BM25 word-only)", "BM25 (Anserini, published)", "bge-m3",
]

LABELS = {
    "text-embedding-3-large": "text-embedding-3-large",
    "text-embedding-3-small": "text-embedding-3-small",
    "qwen3-embedding-4b": "qwen3-embedding-4b",
    "bge-m3": "bge-m3",
    "NRAG (pure lexical)": "NRAG — pure lexical ($0)",
    "NRAG + doc2query x2 (prior run)": "★ NRAG + doc2query ×2 (offline LLM)",
    "normal FTS (BM25 word-only)": "normal FTS — BM25 word-only ($0)",
    "BM25 (Anserini, published)": "BM25 (Anserini, published)",
}


def tier(system: str) -> str:
    if system.startswith("NRAG"):
        return "nrag"
    if system.startswith("normal FTS") or system.startswith("BM25"):
        return "ref"
    return "dense"


def norm(system: str) -> str:
    """Strip the ' (dense)' suffix so dense rows match ORDER/LABELS keys."""
    return system.removesuffix(" (dense)")


def nice_ticks(vmin: float, vmax: float, n: int = 5):
    """Ticks that fully cover [vmin, vmax] (first tick <= vmin, last >= vmax)."""
    lo, hi = vmin, vmax
    span = hi - lo
    step = 10 ** math.floor(math.log10(max(span / (n - 1), 1e-9)))
    for mult in (1, 2, 5, 10):
        if span / (step * mult) <= n - 1:
            step *= mult
            break
    t0 = math.floor(lo / step - 1e-9) * step
    t1 = math.ceil(hi / step + 1e-9) * step
    n_ticks = int(round((t1 - t0) / step)) + 1
    return [round(t0 + i * step, 4) for i in range(n_ticks)]


def panel(ds_rows: list[tuple[str, float]], ds_label: str, sub: str,
          y0: int, bar_h: int = 26, pitch: int = 38) -> tuple[str, int]:
    """Draw one dataset panel; returns (svg_chunk, next_y)."""
    x0, x1 = 260, 1160          # label column -> plot right edge
    n = len(ds_rows)
    vals = [v for _, v in ds_rows]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.08, 0.008)
    ticks = nice_ticks(lo - pad, hi + pad)
    tmin, tmax = ticks[0], ticks[-1]

    def x(v: float) -> float:
        return x0 + (v - tmin) / (tmax - tmin) * (x1 - x0)

    h = n * pitch + 70
    s = [
        f'<text x="32" y="{y0 + 22}" font-size="20" font-weight="800" fill="{INK}">{ds_label}</text>',
        f'<text x="32" y="{y0 + 42}" font-size="13" fill="{MUTED}">{sub}</text>',
        # gridlines + tick labels
        '<g stroke="%s" stroke-width="1">' % GRID,
    ]
    for t in ticks:
        s.append(f'<line x1="{x(t):.1f}" y1="{y0 + 56}" x2="{x(t):.1f}" y2="{y0 + 56 + n * pitch}"/>')
    s.append("</g>")
    s.append('<g class="ax" text-anchor="middle">')
    for t in ticks:
        s.append(f'<text x="{x(t):.1f}" y="{y0 + 56 + n * pitch + 16}">{t:.2f}</text>')
    s.append("</g>")

    for i, (name, v) in enumerate(ds_rows):
        ry = y0 + 56 + i * pitch
        t = tier(name)
        fill = GRAD if t == "nrag" else (LAV if t == "ref" else GRAY)
        opacity = "0.85" if t == "nrag" and name == "NRAG (pure lexical)" else "1"
        bold = ' font-weight="800" fill="%s"' % PURPLE if t == "nrag" else ""
        w = x(v) - x0
        s.append(f'<text class="name" x="{x0 - 12}" y="{ry + 18}" text-anchor="end"{bold}>{LABELS[norm(name)]}</text>')
        s.append(f'<rect x="{x0}" y="{ry}" width="{w:.1f}" height="{bar_h}" rx="5" '
                 f'fill="{fill}" opacity="{opacity}"/>')
        vx = x(v) + 10
        anchor = "start"
        if vx + 58 > x1 + 6:
            vx, anchor = x(v) - 6, "end"
        fillv = PURPLE if t == "nrag" else INK
        s.append(f'<text class="val" x="{vx}" y="{ry + 18}" fill="{fillv}" text-anchor="{anchor}">{v:.4f}</text>')
    return "\n".join(s), y0 + 56 + n * pitch + 34


def build(rows: list[dict], out: str) -> None:
    by_ds: dict[str, dict[str, float]] = {}
    for r in rows:
        v = r["scores"].get("ndcg@10")
        if v is None:
            continue
        by_ds.setdefault(r["dataset"], {})[r["system"]] = float(v)

    meta = {
        "scifact": "BEIR scifact · 5,183 abstracts · 300 claims",
        "nfcorpus": "BEIR nfcorpus · 3,633 docs · 323 queries",
        "fiqa": "BEIR fiqa · 57,638 docs · 648 queries",
    }

    panels = []
    y = 88
    for ds in ("scifact", "nfcorpus", "fiqa"):
        if ds not in by_ds:
            continue
        ds_rows = []
        for key in ORDER:
            match = [s for s in by_ds[ds] if norm(s) == key]
            if match:
                ds_rows.append((match[0], by_ds[ds][match[0]]))
        ds_rows.sort(key=lambda kv: -kv[1])
        chunk, y = panel(ds_rows, ds, meta.get(ds, ""), y)
        panels.append(chunk)

    legend_y = y + 26
    svg = f"""<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Dense vs NRAG on BEIR — nDCG@10">
  <defs>
    <linearGradient id="nrag" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#7c3aed"/>
      <stop offset="1" stop-color="#22d3ee"/>
    </linearGradient>
    <style>
      text{{font-family:'Segoe UI',system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif}}
      .name{{font-size:15px;fill:#0f172a}}
      .val{{font-size:14px;font-weight:700;fill:#0f172a}}
      .ax{{font-size:12px;fill:#64748b}}
    </style>
  </defs>

  <rect x="0" y="0" width="{W}" height="{H}" rx="16" fill="{CARD}" stroke="{STROKE}"/>

  <text x="32" y="42" font-size="23" font-weight="800" fill="{INK}">Dense vs NRAG · BEIR nDCG@10 (document-level)</text>
  <text x="32" y="68" font-size="15" fill="{MUTED}">No embedding model, no GPU, no vector DB — Nrag (purple) at $0/query vs four market embedders (gray). Missing bars = not run.</text>

{chr(10).join(panels)}

  <g>
    <rect x="32" y="{legend_y}" width="26" height="15" rx="4" fill="{GRAD}"/>
    <text x="68" y="{legend_y + 13}" font-size="14" fill="#334155">Nrag — no embeddings ($0/query or offline-LLM tier)</text>
    <rect x="420" y="{legend_y}" width="26" height="15" rx="4" fill="{LAV}"/>
    <text x="456" y="{legend_y + 13}" font-size="14" fill="#334155">lexical reference — $0/query</text>
    <rect x="700" y="{legend_y}" width="26" height="15" rx="4" fill="{GRAY}"/>
    <text x="736" y="{legend_y + 13}" font-size="14" fill="#334155">dense embedder — one model pass / query</text>
  </g>
  <text class="ax" x="32" y="{legend_y + 40}">Per-dataset axis (ticks labelled) · every score costs $0 for the purple/lavender rows · full tables → benchmarks/dense_results.md · regenerate: python benchmarks/plot_dense.py</text>
</svg>
"""
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(f"wrote {out} ({len(svg)} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="benchmarks/dense_results.jsonl")
    ap.add_argument("--out", default="assets/dense-bench.svg")
    args = ap.parse_args()
    rows = json.load(open(args.json, encoding="utf-8"))
    build(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
