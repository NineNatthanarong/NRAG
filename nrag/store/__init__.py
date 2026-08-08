"""Persistent metadata store (source of truth for chunk text + change detection)."""

from __future__ import annotations

from .metadata import DocFingerprint, IngestDiff, MetadataStore

__all__ = ["DocFingerprint", "IngestDiff", "MetadataStore"]
