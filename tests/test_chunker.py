from __future__ import annotations

from nrag._types import Document, attach_context
from nrag.ingest.chunker import ChunkConfig, Chunker


def _doc(text, ctype="markdown"):
    return Document(doc_id="d1", text=text, source="d1.md",
                    metadata={"content_type": ctype})


def test_offsets_reconstruct_exactly():
    text = "# Title\n\n" + " ".join(f"word{i}" for i in range(400))
    doc = _doc(text)
    chunks = Chunker(ChunkConfig(target_tokens=50, overlap_tokens=8)).chunk(doc)
    assert len(chunks) > 3
    for c in chunks:
        assert doc.text[c.start_offset:c.end_offset] == c.raw_text


def test_structure_section_paths():
    text = "# A\nintro text here\n\n## B\nbody of b section\n\n### C\ndeep content here"
    chunks = Chunker(ChunkConfig(target_tokens=100)).chunk(_doc(text))
    sections = {c.section for c in chunks}
    assert any("A > B" in s for s in sections)
    assert any("A > B > C" in s for s in sections)


def test_overlap_present():
    text = "# T\n\n" + ". ".join(f"sentence number {i} here" for i in range(60))
    chunks = Chunker(ChunkConfig(target_tokens=40, overlap_tokens=12)).chunk(_doc(text))
    # consecutive chunks within a section should overlap in character span
    overlaps = [chunks[i].start_offset < chunks[i - 1].end_offset
                for i in range(1, len(chunks))]
    assert any(overlaps)


def test_indexed_equals_raw_by_default_and_context_hook():
    chunks = Chunker().chunk(_doc("# T\n\nsome content about things", "markdown"))
    c = chunks[0]
    assert c.indexed_text == c.raw_text
    attach_context(c, "Situating blurb.")
    assert c.indexed_text.startswith("Situating blurb.")
    assert c.raw_text == "some content about things" or "some content" in c.raw_text


def test_plain_text_no_structure():
    chunks = Chunker(ChunkConfig(target_tokens=5, overlap_tokens=1, min_tokens=1)).chunk(
        _doc("One sentence. Two sentence. Three sentence. Four.", "text"))
    assert len(chunks) >= 2
    for c in chunks:
        assert c.section == ""
