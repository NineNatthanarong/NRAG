"""Phase 5: the hosted compilation service — POST raw docs, get back a portable serving
bundle, open it locally with no LLM. Driven over a real socket (ephemeral port) with a stub
LLM, so the whole wedge pipeline is exercised without network."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from kapi import Kapi
from kapi.portable import import_index
from kapi.service import CompilationService, build_server


class _FakeLLM:
    model_name = "gpt-4o-mini"

    def complete(self, prompt, *, max_tokens=None, temperature=0.0, stop=None, system=None):
        chunk = prompt.lower()
        lines = ["CONTEXT: reference doc.", "QUESTIONS:", "- Q?", "PROPOSITIONS:", "- P.",
                 "INFERENCES:"]
        lines.append("- cheapest route between nodes" if "shortest" in chunk
                     else "- general background")
        return "\n".join(lines)


_DOCS = ["Dijkstra computes shortest paths in a weighted graph.",
         "A tomato soup recipe uses tomatoes, basil and salt."]


@pytest.fixture
def live_service(tmp_path):
    service = CompilationService(llm=_FakeLLM(), work_dir=str(tmp_path / "work"))
    httpd = build_server("127.0.0.1", 0, service)          # port 0 -> ephemeral
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", service
    finally:
        httpd.shutdown()
        httpd.server_close()
        service.close()


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, json.load(resp)


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.status, resp.read()


def test_health(live_service):
    base, _ = live_service
    status, body = _get(base + "/health")
    assert status == 200 and json.loads(body)["status"] == "ok"


def test_compile_download_and_serve_offline(live_service, tmp_path):
    base, _ = live_service
    status, res = _post(base + "/compile", {"texts": _DOCS})
    assert status == 200
    assert res["report"]["num_chunks"] > 0
    assert res["bundle_url"].startswith("/bundle/")

    status, blob = _get(base + res["bundle_url"])           # download the serving bundle
    assert status == 200 and blob[:2] == b"\x1f\x8b"        # gzip magic
    bundle = tmp_path / "dl.kapi.tgz"
    bundle.write_bytes(blob)

    dest = str(tmp_path / "local")                          # serve it with NO LLM — the wedge
    import_index(str(bundle), dest)
    served = Kapi.open(dest)
    try:
        hits = served.search("cheapest route between nodes", k=2)
        assert hits and "Dijkstra" in hits[0].chunk.raw_text
    finally:
        served.close()


def test_query_endpoint(live_service):
    base, _ = live_service
    _, res = _post(base + "/compile", {"texts": _DOCS})
    status, out = _post(base + "/query",
                        {"job": res["job"], "query": "cheapest route between nodes", "k": 3})
    assert status == 200 and out["hits"]
    assert "Dijkstra" in out["hits"][0]["text"]


def test_compile_rejects_empty_body(live_service):
    base, _ = live_service
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(base + "/compile", {"texts": []})
    assert exc.value.code == 400


def test_bundle_unknown_job_is_404(live_service):
    base, _ = live_service
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(base + "/bundle/does-not-exist")
    assert exc.value.code == 404
