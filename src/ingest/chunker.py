"""Deterministic text chunker for the ingestion pipeline.

Splits ``RawDocument`` content into overlapping character windows and assigns
each chunk a **canonical UUID5** identifier derived from the document ID and
chunk index.  The same document always produces the same set of ``chunk_id``
values, satisfying the dual-store invariant shared between PostgreSQL and Neo4j.

Chunk size and overlap are sourced from ``Settings`` (character counts, not
token counts) so behaviour is deterministic without requiring a tokeniser at
this stage.

Design notes
------------
* ``chunk_id`` = ``UUID5(NAMESPACE_OID, f"{document_id}:{chunk_index}")``
* Chunk boundaries are character-based (``chunk_size``, ``chunk_overlap``).
* The final chunk may be shorter than ``chunk_size``.
* Chunks that are entirely whitespace are skipped.
* ``char_start`` / ``char_end`` record the byte slice into the *original*
  document content for later offset-based retrieval.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from src.ingest.loader import RawDocument


@dataclass(frozen=True)
class TextChunk:
    """A single text chunk produced by the chunker.

    Attributes
    ----------
    chunk_id:
        Canonical UUID5 identifier shared between PostgreSQL and Neo4j.
    document_id:
        Parent document identifier inherited from the source ``RawDocument``.
    chunk_index:
        Zero-based position of this chunk within the document.
    content:
        The raw text slice for this chunk.
    char_start:
        Start character offset (inclusive) in the original document content.
    char_end:
        End character offset (exclusive) in the original document content.
    metadata:
        Shallow copy of the parent document metadata for downstream enrichment.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Derived helpers
    # -----------------------------------------------------------------------

    @property
    def char_length(self) -> int:
        """Number of characters in this chunk."""
        return self.char_end - self.char_start


def _make_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a deterministic UUID5 for a chunk.

    The namespace is ``uuid.NAMESPACE_OID`` (a stable standard namespace).
    The name string is ``f"{document_id}:{chunk_index}"`` so every
    (document_id, chunk_index) pair maps to exactly one UUID.

    Parameters
    ----------
    document_id:
        Stable document identifier from ``RawDocument.document_id``.
    chunk_index:
        Zero-based chunk position within the document.

    Returns
    -------
    str
        Lowercase hyphenated UUID string, e.g.
        ``"5d6e7f80-1a2b-5c3d-8e4f-9a0b1c2d3e4f"``.
    """
    name = f"{document_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, name))


class DocumentChunker:
    """Splits a ``RawDocument`` into overlapping character-window ``TextChunk``s.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk (must be >= 1).
    chunk_overlap:
        Number of characters shared between consecutive chunks.
        Must satisfy ``0 <= chunk_overlap < chunk_size``.

    Raises
    ------
    ValueError
        If ``chunk_size < 1`` or ``chunk_overlap >= chunk_size``.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk(self, document: RawDocument) -> list[TextChunk]:
        """Split *document* into a list of ``TextChunk`` objects.

        An empty or whitespace-only document content produces an empty list.

        Parameters
        ----------
        document:
            A validated ``RawDocument`` instance to chunk.

        Returns
        -------
        list[TextChunk]
            Ordered list of chunks; chunk_index is 0-based and contiguous.
        """
        content = document.content
        if not content.strip():
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[TextChunk] = []
        chunk_index = 0
        pos = 0
        total = len(content)

        while pos < total:
            end = min(pos + self.chunk_size, total)
            slice_text = content[pos:end]

            # Skip chunks that are entirely whitespace
            if slice_text.strip():
                chunk_id = _make_chunk_id(document.document_id, chunk_index)
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        content=slice_text,
                        char_start=pos,
                        char_end=end,
                        metadata=dict(document.metadata),
                    )
                )
                chunk_index += 1

            pos += step

        return chunks
