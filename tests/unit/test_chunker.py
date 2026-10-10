"""Unit tests for src/ingest/chunker.py.

All tests are offline (no live services, no file I/O beyond the corpus fixture).
Coverage targets:
- UUID5 determinism and cross-chunk uniqueness
- Character window slicing, step computation, and final-chunk boundary
- chunk_overlap enforcement
- Whitespace-only content handling
- Invalid constructor arguments
- Metadata propagation
- Integration with RawDocument through DocumentChunker.chunk()
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from src.ingest.chunker import DocumentChunker, TextChunk, _make_chunk_id
from src.ingest.loader import RawDocument

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(
    content: str,
    document_id: str = "doc-test-001",
    metadata: dict[str, Any] | None = None,
) -> RawDocument:
    """Build a minimal RawDocument for testing."""
    return RawDocument(
        document_id=document_id,
        source_uri="tests/fixtures/test.md",
        title="Test Document",
        doc_type="technical_doc",
        content=content,
        metadata=metadata or {"env": "test"},
    )


# ---------------------------------------------------------------------------
# _make_chunk_id — determinism and UUID validity
# ---------------------------------------------------------------------------


class TestMakeChunkId:
    def test_returns_valid_uuid_string(self) -> None:
        chunk_id = _make_chunk_id("doc-abc", 0)
        parsed = uuid.UUID(chunk_id)
        assert parsed.version == 5

    def test_is_deterministic(self) -> None:
        assert _make_chunk_id("doc-abc", 0) == _make_chunk_id("doc-abc", 0)
        assert _make_chunk_id("doc-xyz", 3) == _make_chunk_id("doc-xyz", 3)

    def test_different_index_gives_different_id(self) -> None:
        assert _make_chunk_id("doc-abc", 0) != _make_chunk_id("doc-abc", 1)

    def test_different_document_gives_different_id(self) -> None:
        assert _make_chunk_id("doc-aaa", 0) != _make_chunk_id("doc-bbb", 0)

    def test_lowercase_hyphenated_output(self) -> None:
        chunk_id = _make_chunk_id("doc-abc", 0)
        assert chunk_id == chunk_id.lower()
        assert chunk_id.count("-") == 4  # standard UUID has 4 hyphens


# ---------------------------------------------------------------------------
# DocumentChunker — constructor validation
# ---------------------------------------------------------------------------


class TestDocumentChunkerInit:
    def test_valid_parameters_accepted(self) -> None:
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 20

    def test_default_parameters(self) -> None:
        chunker = DocumentChunker()
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_zero_overlap_accepted(self) -> None:
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=0)
        assert chunker.chunk_overlap == 0

    def test_chunk_size_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            DocumentChunker(chunk_size=0)

    def test_negative_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            DocumentChunker(chunk_size=-10)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
            DocumentChunker(chunk_size=100, chunk_overlap=-1)

    def test_overlap_equal_to_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap.*must be strictly less than"):
            DocumentChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap.*must be strictly less than"):
            DocumentChunker(chunk_size=50, chunk_overlap=75)


# ---------------------------------------------------------------------------
# DocumentChunker.chunk() — content splitting
# ---------------------------------------------------------------------------


class TestDocumentChunkerChunk:
    def test_empty_content_returns_empty_list(self) -> None:
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        doc = _make_doc(content="")
        assert chunker.chunk(doc) == []

    def test_whitespace_only_content_returns_empty_list(self) -> None:
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        doc = _make_doc(content="   \n\t  ")
        assert chunker.chunk(doc) == []

    def test_short_content_produces_single_chunk(self) -> None:
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        doc = _make_doc(content="Hello, world!")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello, world!"
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len("Hello, world!")

    def test_chunk_index_is_zero_based_and_contiguous(self) -> None:
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
        doc = _make_doc(content="A" * 35)
        chunks = chunker.chunk(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_ids_are_unique_within_document(self) -> None:
        chunker = DocumentChunker(chunk_size=10, chunk_overlap=2)
        doc = _make_doc(content="X" * 50)
        chunks = chunker.chunk(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected"

    def test_chunk_ids_are_deterministic(self) -> None:
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=5)
        doc = _make_doc(content="The quick brown fox jumps over the lazy dog. " * 5)
        first_run = [c.chunk_id for c in chunker.chunk(doc)]
        second_run = [c.chunk_id for c in chunker.chunk(doc)]
        assert first_run == second_run

    def test_char_start_end_are_correct_slices(self) -> None:
        content = "abcdefghij"  # 10 chars
        chunker = DocumentChunker(chunk_size=4, chunk_overlap=1)
        doc = _make_doc(content=content)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert content[chunk.char_start : chunk.char_end] == chunk.content

    def test_final_chunk_does_not_exceed_content_length(self) -> None:
        content = "A" * 13
        chunker = DocumentChunker(chunk_size=5, chunk_overlap=1)
        doc = _make_doc(content=content)
        chunks = chunker.chunk(doc)
        assert chunks[-1].char_end <= len(content)

    def test_overlap_causes_shared_characters(self) -> None:
        content = "0123456789"  # 10 chars
        chunker = DocumentChunker(chunk_size=6, chunk_overlap=2)
        # step = 6 - 2 = 4; positions: 0, 4, 8 → 3 chunks
        doc = _make_doc(content=content)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        # Verify the overlap property holds for every adjacent pair of chunks
        for i in range(len(chunks) - 1):
            prev_tail = chunks[i].content[-chunker.chunk_overlap :]
            next_head = chunks[i + 1].content[: len(prev_tail)]
            assert prev_tail == next_head, (
                f"Overlap mismatch between chunk {i} and {i + 1}: {prev_tail!r} != {next_head!r}"
            )

    def test_zero_overlap_chunks_are_non_overlapping(self) -> None:
        content = "abcdefghijklmno"  # 15 chars
        chunker = DocumentChunker(chunk_size=5, chunk_overlap=0)
        doc = _make_doc(content=content)
        chunks = chunker.chunk(doc)
        for i in range(len(chunks) - 1):
            assert chunks[i].char_end == chunks[i + 1].char_start

    def test_document_id_propagated_to_all_chunks(self) -> None:
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        doc = _make_doc(content="word " * 20, document_id="doc-special-42")
        chunks = chunker.chunk(doc)
        assert all(c.document_id == "doc-special-42" for c in chunks)

    def test_metadata_propagated_to_all_chunks(self) -> None:
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        meta = {"team": "platform", "env": "production", "priority": 1}
        doc = _make_doc(content="word " * 20, metadata=meta)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata == meta

    def test_metadata_is_shallow_copy(self) -> None:
        """Mutating returned metadata must not affect other chunks."""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        doc = _make_doc(content="word " * 20, metadata={"key": "value"})
        chunks = chunker.chunk(doc)
        chunks[0].metadata["injected"] = True
        # Other chunks should not be affected
        for chunk in chunks[1:]:
            assert "injected" not in chunk.metadata

    def test_chunk_ids_differ_across_documents(self) -> None:
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=5)
        content = "The same content repeated many times. " * 5
        doc_a = _make_doc(content=content, document_id="doc-aaa")
        doc_b = _make_doc(content=content, document_id="doc-bbb")
        ids_a = {c.chunk_id for c in chunker.chunk(doc_a)}
        ids_b = {c.chunk_id for c in chunker.chunk(doc_b)}
        assert ids_a.isdisjoint(ids_b), "IDs from different docs must not collide"

    def test_chunk_id_matches_make_chunk_id_helper(self) -> None:
        chunker = DocumentChunker(chunk_size=30, chunk_overlap=5)
        doc = _make_doc(content="word " * 30, document_id="doc-verify-id")
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            expected = _make_chunk_id(doc.document_id, chunk.chunk_index)
            assert chunk.chunk_id == expected


# ---------------------------------------------------------------------------
# TextChunk dataclass
# ---------------------------------------------------------------------------


class TestTextChunk:
    def test_char_length_property(self) -> None:
        chunk = TextChunk(
            chunk_id="test-id",
            document_id="doc-001",
            chunk_index=0,
            content="hello world",
            char_start=0,
            char_end=11,
        )
        assert chunk.char_length == 11

    def test_frozen_prevents_mutation(self) -> None:
        chunk = TextChunk(
            chunk_id="test-id",
            document_id="doc-001",
            chunk_index=0,
            content="hello",
            char_start=0,
            char_end=5,
        )
        with pytest.raises((AttributeError, TypeError)):
            chunk.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration: chunker + settings fixture
# ---------------------------------------------------------------------------


class TestChunkerWithSettings:
    def test_chunker_honours_settings_defaults(self, mock_settings: Any) -> None:
        """DocumentChunker created from mock_settings produces valid output."""
        chunker = DocumentChunker(
            chunk_size=mock_settings.chunk_size,
            chunk_overlap=mock_settings.chunk_overlap,
        )
        doc = _make_doc(content="enterprise data " * 100)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)
        # Verify chunk_size constraint: each content slice <= chunk_size chars
        for chunk in chunks:
            assert len(chunk.content) <= mock_settings.chunk_size
