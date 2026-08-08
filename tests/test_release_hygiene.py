"""Release hygiene: one version everywhere, presets fail loudly, CLI --version."""

from __future__ import annotations

import json
import os

import pytest

import nrag
from nrag import Config, Nrag


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset"):
        Config.from_preset("qualty")  # typo must not silently become "quality"


def test_known_presets_still_work():
    assert Config.from_preset("fast").preset == "fast"
    assert Config.from_preset("quality").preset == "quality"
    assert Config.from_preset("compiled").preset == "compiled"


def test_service_version_matches_package():
    from nrag import service

    assert service.VERSION == nrag.__version__


def test_manifest_version_matches_package(tmp_path):
    rag = Nrag(preset="fast", path=str(tmp_path / "idx"))
    rag.add_texts(["hello world"])
    rag.close()
    with open(tmp_path / "idx" / "nrag.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["nrag_version"] == nrag.__version__


def test_cli_version(capsys):
    from nrag.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert nrag.__version__ in capsys.readouterr().out
