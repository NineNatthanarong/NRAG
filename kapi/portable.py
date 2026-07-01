"""Phase 5 — portable, air-gapped index bundles (STRATEGY §6, §8: the Wedge).

Compiled Retrieval's business wedge is a *compute-placement* split: the expensive step
(LLM compilation) runs once — offline, or on a hosted service (:mod:`kapi.service`) — while
the *serving* index is a plain lexical artifact that needs no LLM, no network, and no vector
DB. This module makes that artifact portable: a single gzip-tar file you ship to an on-prem /
air-gapped box and open with ``Kapi.open(dir)`` (no ``llm``).

A bundle archives the on-disk index directory — the engine files, the ``meta.sqlite``
metadata store, the Leg B consensus vectors (``.kapi_csc/legb.json``) and the ``kapi.json``
manifest — minus transient locks and, by default, the offline LLM compile cache
(``.kapi_cache/``; not needed to serve, and usually the largest part). ``include_cache=True``
keeps it so a later re-compile on the target is free. A ``kapi_bundle.json`` manifest records
provenance (kapi/engine/language) so import can inspect an archive before unpacking it.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from dataclasses import dataclass
from typing import List, Optional

BUNDLE_MANIFEST = "kapi_bundle.json"
BUNDLE_VERSION = 1

_CACHE_DIR = ".kapi_cache"          # offline LLM compile cache — dropped from serve bundles
_EXCLUDE_SUFFIXES = (".lock", ".tmp")


@dataclass
class BundleInfo:
    path: str
    num_files: int
    engine: Optional[str]
    language: Optional[str]
    includes_cache: bool


def _index_manifest(index_dir: str) -> dict:
    """Read the index's ``kapi.json`` (engine/language/version) for bundle provenance."""
    try:
        with open(os.path.join(index_dir, "kapi.json"), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _should_include(rel: str, *, include_cache: bool) -> bool:
    parts = rel.split(os.sep)
    if not include_cache and parts and parts[0] == _CACHE_DIR:
        return False                       # LLM cache: not needed to serve
    if rel.endswith(_EXCLUDE_SUFFIXES):
        return False                       # transient lock / tmp files
    return True


def _iter_index_files(index_dir: str, *, include_cache: bool) -> List[str]:
    out: List[str] = []
    for root, _dirs, files in os.walk(index_dir):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), index_dir)
            if _should_include(rel, include_cache=include_cache):
                out.append(rel)
    return sorted(out)


def export_index(index_dir: str, dest: str, *, include_cache: bool = False) -> BundleInfo:
    """Archive a persisted index directory into a portable gzip-tar bundle at ``dest``.

    The index must already be committed/closed — files are read off disk verbatim. Everything
    under ``index_dir`` is captured except transient locks and (unless ``include_cache``) the
    offline LLM compile cache. Returns a :class:`BundleInfo`.
    """
    if not os.path.isdir(index_dir):
        raise NotADirectoryError(f"not an index directory: {index_dir}")
    rels = _iter_index_files(index_dir, include_cache=include_cache)
    if not rels:
        raise FileNotFoundError(f"no index files under {index_dir} (nothing to export)")

    man = _index_manifest(index_dir)
    meta = {
        "kapi_bundle_version": BUNDLE_VERSION,
        "kapi_version": man.get("kapi_version"),
        "engine": man.get("engine"),
        "language": man.get("language"),
        "enable_ngram": man.get("enable_ngram"),
        "includes_cache": include_cache,
        "serve_only": not include_cache,
        "files": rels,
    }

    parent = os.path.dirname(os.path.abspath(dest))
    os.makedirs(parent, exist_ok=True)
    with tarfile.open(dest, "w:gz") as tar:
        blob = json.dumps(meta, indent=2).encode("utf-8")   # provenance manifest, from memory
        info = tarfile.TarInfo(BUNDLE_MANIFEST)
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
        for rel in rels:
            tar.add(os.path.join(index_dir, rel), arcname=rel)

    return BundleInfo(path=dest, num_files=len(rels), engine=meta["engine"],
                      language=meta["language"], includes_cache=include_cache)


def read_bundle_manifest(bundle: str) -> dict:
    """Return the provenance manifest inside a bundle (``{}`` if absent)."""
    with tarfile.open(bundle, "r:gz") as tar:
        try:
            member = tar.getmember(BUNDLE_MANIFEST)
        except KeyError:
            return {}
        fh = tar.extractfile(member)
        return json.load(fh) if fh is not None else {}


def _is_within(base: str, target: str) -> bool:
    base = os.path.abspath(base)
    target = os.path.abspath(target)
    return target == base or target.startswith(base + os.sep)


def import_index(bundle: str, dest: str, *, overwrite: bool = False) -> str:
    """Unpack a bundle into ``dest`` (created if absent) and return ``dest``.

    Security: rejects any archive member with an absolute path, a ``..`` traversal that would
    escape ``dest``, or a non-file/dir type (symlink/device) — a hostile bundle cannot write
    outside the destination.
    """
    if os.path.isdir(dest) and os.listdir(dest) and not overwrite:
        raise FileExistsError(f"{dest} is non-empty; pass overwrite=True to replace")
    os.makedirs(dest, exist_ok=True)

    with tarfile.open(bundle, "r:gz") as tar:
        for m in tar.getmembers():
            if os.path.isabs(m.name) or m.name.startswith("/"):
                raise ValueError(f"unsafe absolute path in bundle: {m.name}")
            if not _is_within(dest, os.path.join(dest, m.name)):
                raise ValueError(f"unsafe path escapes destination: {m.name}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"unsupported archive member type: {m.name}")
        # 'data' filter (py3.12+, backported) is a second guard against traversal/links.
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(dest, **kw)
    return dest
