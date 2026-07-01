"""Phase 5: portable, air-gapped bundles — export a compiled index, ship it, and serve it
with no LLM and no network. Plus the import-time path-traversal guard."""

from __future__ import annotations

import io
import os
import tarfile

import pytest

from kapi import Kapi
from kapi.portable import export_index, import_index, read_bundle_manifest


class _FakeLLM:
    """Compact compiler stub: emits a parseable bundle and maps 'shortest' -> a non-literal
    inference phrase, so the Leg B consensus leg is exercisable end to end."""

    model_name = "gpt-4o-mini"

    def complete(self, prompt, *, max_tokens=None, temperature=0.0, stop=None, system=None):
        chunk = prompt.lower()
        lines = ["CONTEXT: reference doc.", "QUESTIONS:", "- What does this explain?",
                 "PROPOSITIONS:", "- A definition.", "INFERENCES:"]
        lines.append("- cheapest route between nodes" if "shortest" in chunk
                     else "- general background")
        return "\n".join(lines)


_DOCS = ["Dijkstra computes shortest paths in a weighted graph.",
         "A simple tomato soup recipe uses tomatoes, basil and salt."]


def _compile_index(path: str) -> None:
    rag = Kapi(llm=_FakeLLM(), preset="compiled", path=path)
    rag.compile_texts(_DOCS)
    rag.close()


def test_export_import_roundtrip_serves_offline(tmp_path):
    src = str(tmp_path / "idx")
    _compile_index(src)

    bundle = str(tmp_path / "ship.kapi.tgz")
    info = export_index(src, bundle)
    assert info.num_files > 0 and os.path.exists(bundle)

    dest = str(tmp_path / "restored")           # a *different* directory (simulates another box)
    import_index(bundle, dest)
    served = Kapi.open(dest)                     # NO llm
    try:
        assert served._legb is not None and served._legb.vectors    # Leg B travelled
        hits = served.search("cheapest route between nodes", k=2)
        assert hits and "Dijkstra" in hits[0].chunk.raw_text
        # the winning term lives only in the compiled closure, never the cited raw text
        assert all("cheapest" not in h.chunk.raw_text for h in hits)
    finally:
        served.close()


def test_bundle_excludes_llm_cache_by_default(tmp_path):
    src = str(tmp_path / "idx")
    _compile_index(src)
    assert os.path.isdir(os.path.join(src, ".kapi_cache"))          # compiler wrote a cache

    default = str(tmp_path / "d.kapi.tgz")
    export_index(src, default)
    with tarfile.open(default) as t:
        names = t.getnames()
    assert not any(n.split("/")[0] == ".kapi_cache" for n in names)
    assert read_bundle_manifest(default)["includes_cache"] is False
    assert read_bundle_manifest(default)["serve_only"] is True

    withcache = str(tmp_path / "c.kapi.tgz")
    export_index(src, withcache, include_cache=True)
    with tarfile.open(withcache) as t:
        names = t.getnames()
    assert any(".kapi_cache" in n for n in names)


def test_facade_export_bundle_roundtrip(tmp_path):
    """The Kapi.export_bundle / import_bundle convenience wrappers on an open index."""
    src = str(tmp_path / "idx")
    rag = Kapi(llm=_FakeLLM(), preset="compiled", path=src)
    rag.compile_texts(_DOCS)
    bundle = str(tmp_path / "b.kapi.tgz")
    rag.export_bundle(bundle)                   # flushes + archives while open
    rag.close()

    served = Kapi.import_bundle(bundle, str(tmp_path / "dest"))
    try:
        hits = served.search("cheapest route between nodes", k=2)
        assert hits and "Dijkstra" in hits[0].chunk.raw_text
    finally:
        served.close()


def test_export_requires_persisted_index(tmp_path):
    rag = Kapi(llm=_FakeLLM(), preset="compiled")   # in-memory: no path
    rag.compile_texts(_DOCS)
    with pytest.raises(ValueError):
        rag.export_bundle(str(tmp_path / "x.kapi.tgz"))
    rag.close()


def test_import_rejects_path_traversal(tmp_path):
    evil = str(tmp_path / "evil.kapi.tgz")
    with tarfile.open(evil, "w:gz") as t:
        data = b"pwned"
        member = tarfile.TarInfo("../escape.txt")   # tries to write outside dest
        member.size = len(data)
        t.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError):
        import_index(evil, str(tmp_path / "dest"))
    assert not os.path.exists(tmp_path / "escape.txt")


def test_import_refuses_nonempty_dest_without_overwrite(tmp_path):
    src = str(tmp_path / "idx")
    _compile_index(src)
    bundle = str(tmp_path / "b.kapi.tgz")
    export_index(src, bundle)

    dest = str(tmp_path / "dest")
    os.makedirs(dest)
    open(os.path.join(dest, "sentinel"), "w").close()
    with pytest.raises(FileExistsError):
        import_index(bundle, dest)
    import_index(bundle, dest, overwrite=True)       # explicit overwrite is allowed
