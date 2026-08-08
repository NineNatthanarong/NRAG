"""Ingestion: load documents and split them into chunks."""

from __future__ import annotations

from .chunker import Chunk, ChunkConfig, Chunker
from .loaders import (
    DEFAULT_LOADERS,
    HTMLLoader,
    Loader,
    MarkdownLoader,
    PDFTextLoader,
    TextLoader,
    documents_from_texts,
    load_paths,
)

__all__ = [
    "DEFAULT_LOADERS",
    "Chunk",
    "ChunkConfig",
    "Chunker",
    "HTMLLoader",
    "Loader",
    "MarkdownLoader",
    "PDFTextLoader",
    "TextLoader",
    "documents_from_texts",
    "load_paths",
]
