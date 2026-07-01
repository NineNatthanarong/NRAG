"""Phase 5 — the hosted compilation service (STRATEGY §6, §8: the Wedge).

The business model Compiled Retrieval implies: OSS the fast, embedding-free *retriever*
(serving is free and local); monetize the expensive, value-adding step — *compilation*. This
is that step behind an HTTP API. A client POSTs raw documents; the service runs the offline
LLM compiler once and returns a portable, model-free serving bundle (:mod:`kapi.portable`)
the client opens locally with ``Kapi.open`` (or ``kapi import``). The smart compute stays
server-side; query time stays ~1 ms BM25 on the client, and no embedding model crosses the
wire.

Dependency-free: stdlib ``http.server`` only. The compiler LLM is *injected* (built from env
by ``kapi serve``), so tests drive the whole pipeline with a stub LLM and no network.

Endpoints::

    GET  /health            -> {"status":"ok", ...}
    POST /compile           body {"texts":[...], "preset":"compiled", "prefix":"doc"}
                            -> {"job", "bundle_url", "report", "engine", "num_files"}
    GET  /bundle/<job>      -> the gzip-tar serving bundle for that job (application/gzip)
    POST /query             body {"job":"<id>", "query":"...", "k":5}
                            -> {"hits":[...]}   (convenience; the client normally serves local)
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from .portable import export_index

VERSION = "0.1.0"


class CompilationService:
    """Runs the offline compiler on submitted texts and keeps a registry of on-disk jobs.

    Each job gets its own index directory under ``work_dir`` plus an exported serving bundle.
    The compiler ``llm`` is injected; with ``llm=None`` the service compiles a pure-lexical
    bundle (no enrichment) — still a valid, servable index.
    """

    def __init__(self, llm=None, *, preset: str = "compiled", work_dir: Optional[str] = None,
                 engine: Optional[str] = None) -> None:
        self.llm = llm
        self.preset = preset
        self.engine = engine
        self._owns_work_dir = work_dir is None
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="kapi-compile-")
        os.makedirs(self.work_dir, exist_ok=True)
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ pipeline
    def compile_texts(self, texts: List[str], *, prefix: str = "doc",
                      preset: Optional[str] = None) -> dict:
        """Compile ``texts`` into a fresh index + serving bundle; return the job record."""
        if not texts:
            raise ValueError("no texts to compile")
        job = uuid.uuid4().hex[:12]
        job_dir = os.path.join(self.work_dir, job)
        idx_dir = os.path.join(job_dir, "idx")
        os.makedirs(idx_dir, exist_ok=True)

        from . import Kapi

        rag = Kapi(llm=self.llm, preset=preset or self.preset, path=idx_dir, engine=self.engine)
        try:
            rep = rag.compile_texts(texts, prefix=prefix)
        finally:
            rag.close()

        bundle_path = os.path.join(job_dir, "bundle.kapi.tgz")
        info = export_index(idx_dir, bundle_path)
        record = {
            "job": job,
            "index_dir": idx_dir,
            "bundle_path": bundle_path,
            "engine": info.engine,
            "num_files": info.num_files,
            "report": {"num_docs": rep.num_docs, "num_chunks": rep.num_chunks,
                       "added": rep.added, "contextualized": rep.contextualized},
        }
        with self._lock:
            self._jobs[job] = record
        return record

    def bundle_path(self, job: str) -> Optional[str]:
        with self._lock:
            rec = self._jobs.get(job)
        return rec["bundle_path"] if rec else None

    def query(self, job: str, query: str, k: int = 5) -> Optional[list]:
        """Serve a compiled job model-free (what the client would do locally)."""
        with self._lock:
            rec = self._jobs.get(job)
        if not rec:
            return None
        from . import Kapi

        rag = Kapi.open(rec["index_dir"])          # no LLM: pure lexical serving
        try:
            hits = rag.search(query, k=k)
            return [{"chunk_id": h.chunk_id, "score": h.score, "source": h.source,
                     "text": (h.text or "")[:400]} for h in hits]
        finally:
            rag.close()

    def close(self) -> None:
        if self._owns_work_dir and os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)


# ---------------------------------------------------------------------- HTTP glue
def _make_handler(service: CompilationService):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"kapi/{VERSION}"

        # ---- io helpers ----
        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def log_message(self, *args) -> None:      # keep the server quiet under tests
            pass

        # ---- routes ----
        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok", "service": "kapi-compile",
                                      "version": VERSION, "llm": service.llm is not None})
                return
            if self.path.startswith("/bundle/"):
                job = self.path[len("/bundle/"):]
                path = service.bundle_path(job)
                if not path or not os.path.exists(path):
                    self._send_json(404, {"error": "unknown job"})
                    return
                with open(path, "rb") as fh:
                    data = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Disposition",
                                 f'attachment; filename="{job}.kapi.tgz"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
            except ValueError:
                self._send_json(400, {"error": "invalid JSON body"})
                return

            if self.path == "/compile":
                texts = payload.get("texts")
                if not isinstance(texts, list) or not texts:
                    self._send_json(400, {"error": "body must include a non-empty 'texts' list"})
                    return
                try:
                    rec = service.compile_texts(texts, prefix=payload.get("prefix", "doc"),
                                                preset=payload.get("preset"))
                except Exception as exc:            # compile / LLM / cost-guard errors
                    self._send_json(500, {"error": str(exc)})
                    return
                self._send_json(200, {"job": rec["job"], "bundle_url": f"/bundle/{rec['job']}",
                                      "report": rec["report"], "engine": rec["engine"],
                                      "num_files": rec["num_files"]})
                return

            if self.path == "/query":
                job, q = payload.get("job"), payload.get("query")
                if not job or not q:
                    self._send_json(400, {"error": "body must include 'job' and 'query'"})
                    return
                hits = service.query(job, q, k=int(payload.get("k", 5)))
                if hits is None:
                    self._send_json(404, {"error": "unknown job"})
                    return
                self._send_json(200, {"job": job, "query": q, "hits": hits})
                return

            self._send_json(404, {"error": "not found"})

    return Handler


def build_server(host: str, port: int, service: CompilationService) -> ThreadingHTTPServer:
    """Build (but do not start) a threaded HTTP server. ``port=0`` picks an ephemeral port."""
    return ThreadingHTTPServer((host, port), _make_handler(service))


def serve(host: str = "127.0.0.1", port: int = 8000, *, llm=None, preset: str = "compiled",
          work_dir: Optional[str] = None, engine: Optional[str] = None) -> None:
    """Run the compilation service until interrupted (blocking)."""
    service = CompilationService(llm=llm, preset=preset, work_dir=work_dir, engine=engine)
    httpd = build_server(host, port, service)
    kind = "with LLM compiler" if llm is not None else "NO LLM (pure-lexical compile)"
    print(f"kapi compile service on http://{host}:{port}  [{kind}]  work_dir={service.work_dir}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        service.close()
