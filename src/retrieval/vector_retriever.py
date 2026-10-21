"""Vector Retriever executing pgvector cosine distance similarity search with metadata filtering.

Design & Invariants:
- Uses parameterized SQL exclusively against PostgreSQL/pgvector.
- Computes cosine similarity: ``similarity = 1 - (c.embedding <=> $query_embedding)``.
- Applies ``similarity >= $similarity_threshold`` filter.
- Validates query vector dimensionality and ensures finite numeric values.
- Supports optional metadata key-value filtering, doc_type filtering, and document_id filtering.
- Supports fetching specific chunks by their canonical UUIDs (for graph-to-vector hydration).
- Zero paid API calls / zero dynamic Cypher.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.config import Settings, get_settings
from src.db.postgres import PostgresClient, PostgresClientError

logger = logging.getLogger(__name__)


class VectorRetrieverError(PostgresClientError):
    """Base exception for vector retrieval operations."""


class InvalidQueryVectorError(VectorRetrieverError):
    """Raised when the query vector has incorrect dimensionality or invalid values."""


@dataclass(frozen=True)
class RetrievedChunk:
    """Standardized result record for a vector-retrieved chunk.

    Attributes
    ----------
    chunk_id:
        Canonical UUID string of the chunk.
    document_id:
        Parent document identifier.
    content:
        Raw text content of the chunk.
    similarity_score:
        Cosine similarity score in range [-1.0, 1.0] (typically [0.0, 1.0]).
    token_count:
        Approximate token count / char length.
    char_start:
        Start offset in document content.
    char_end:
        End offset in document content.
    metadata:
        Attached metadata dictionary.
    """

    chunk_id: str
    document_id: str
    content: str
    similarity_score: float
    token_count: int = 0
    char_start: int = 0
    char_end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorRetriever:
    """Performs semantic dense vector search and chunk hydration using PostgreSQL/pgvector."""

    def __init__(
        self,
        client: PostgresClient,
        settings: Settings | None = None,
    ) -> None:
        self.client = client
        self.settings = settings or get_settings()

    def _validate_vector(self, query_vector: Sequence[float]) -> str:
        """Validate query vector dimensionality, finite values, and format as pgvector literal string."""
        expected_dim = self.settings.embedding_dimensions
        if len(query_vector) != expected_dim:
            raise InvalidQueryVectorError(
                f"Query vector dimension {len(query_vector)} does not match expected {expected_dim}"
            )

        for i, val in enumerate(query_vector):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise InvalidQueryVectorError(
                    f"Query vector contains non-finite or non-numeric value '{val}' at dimension {i}"
                )

        return str(list(query_vector))

    async def search(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        document_id: str | None = None,
        doc_type: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform cosine similarity vector search over text chunks.

        Cosine similarity is calculated as ``1 - (c.embedding <=> $1::vector)``.

        Parameters
        ----------
        query_vector:
            Query embedding vector of dimension ``settings.embedding_dimensions``.
        top_k:
            Max number of results to return (defaults to ``settings.vector_top_k``).
        similarity_threshold:
            Minimum similarity threshold (defaults to ``settings.vector_similarity_threshold``).
        document_id:
            Optional filter to restrict search to a single document.
        doc_type:
            Optional filter to restrict search to a document type (joined from ``documents`` table).
        metadata_filter:
            Optional JSONB key-value subset filter on ``chunks.metadata``.

        Returns
        -------
        list[RetrievedChunk]
            Ordered list of retrieved chunks sorted descending by similarity score.
        """
        vector_str = self._validate_vector(query_vector)
        limit = top_k if top_k is not None else self.settings.vector_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.settings.vector_similarity_threshold
        )

        if limit < 1:
            raise ValueError(f"top_k must be >= 1, got {limit}")

        # Select target vector column strictly based on expected dimension
        # Whitelisted literal column names only: never user-controlled input
        dim = len(query_vector)
        if dim == 384:
            col = "c.embedding_384"
        elif dim == 1536:
            col = "c.embedding"
        else:
            col = "c.embedding"

        # Build parameterized query dynamically with safe parameter placeholders
        where_clauses: list[str] = [
            f"{col} IS NOT NULL",
            f"(1 - ({col} <=> $1::vector)) >= $2",
        ]
        params: list[Any] = [vector_str, threshold]
        param_idx = 3

        join_documents = False

        if document_id is not None:
            where_clauses.append(f"c.document_id = ${param_idx}")
            params.append(document_id)
            param_idx += 1

        if doc_type is not None:
            join_documents = True
            where_clauses.append(f"d.doc_type = ${param_idx}")
            params.append(doc_type)
            param_idx += 1

        if metadata_filter:
            where_clauses.append(f"c.metadata @> ${param_idx}::jsonb")
            params.append(json.dumps(metadata_filter))
            param_idx += 1

        where_sql = " AND ".join(where_clauses)
        join_sql = "JOIN documents d ON c.document_id = d.document_id" if join_documents else ""

        query_sql = f"""
        SELECT
            c.id AS chunk_id,
            c.document_id,
            c.content,
            c.token_count,
            c.char_start,
            c.char_end,
            c.metadata,
            (1 - ({col} <=> $1::vector)) AS similarity_score
        FROM chunks c
        {join_sql}
        WHERE {where_sql}
        ORDER BY {col} <=> $1::vector ASC
        LIMIT ${param_idx};
        """
        params.append(limit)

        try:
            async with self.client.connection() as conn:
                rows = await conn.fetch(query_sql, *params)
                results: list[RetrievedChunk] = []
                for row in rows:
                    raw_meta = row["metadata"]
                    meta_dict = (
                        json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
                    )
                    results.append(
                        RetrievedChunk(
                            chunk_id=str(row["chunk_id"]),
                            document_id=str(row["document_id"]),
                            content=str(row["content"]),
                            similarity_score=float(row["similarity_score"]),
                            token_count=int(row["token_count"]),
                            char_start=int(row["char_start"]),
                            char_end=int(row["char_end"]),
                            metadata=meta_dict,
                        )
                    )
                logger.debug(
                    "Vector search returned %d chunks (threshold=%.2f, limit=%d)",
                    len(results),
                    threshold,
                    limit,
                )
                return results

        except Exception as exc:
            logger.error("Vector search query failed: %s", exc, exc_info=True)
            raise VectorRetrieverError(f"Vector search failed: {exc}") from exc

    async def get_chunks_by_ids(self, chunk_ids: Sequence[str | uuid.UUID]) -> list[RetrievedChunk]:
        """Hydrate full chunk records given a list of canonical chunk IDs.

        Used to fetch supporting text chunks identified during graph traversal.

        Parameters
        ----------
        chunk_ids:
            Sequence of chunk UUID strings or UUID objects.

        Returns
        -------
        list[RetrievedChunk]
            Retrieved chunk records.
        """
        if not chunk_ids:
            return []

        clean_uuids: list[uuid.UUID] = []
        for cid in chunk_ids:
            try:
                clean_uuids.append(uuid.UUID(str(cid)))
            except ValueError:
                logger.warning("Skipping invalid UUID string in get_chunks_by_ids: %s", cid)
                continue

        if not clean_uuids:
            return []

        sql = """
        SELECT
            id AS chunk_id,
            document_id,
            content,
            token_count,
            char_start,
            char_end,
            metadata
        FROM chunks
        WHERE id = ANY($1::uuid[]);
        """

        try:
            async with self.client.connection() as conn:
                rows = await conn.fetch(sql, clean_uuids)
                results: list[RetrievedChunk] = []
                for row in rows:
                    raw_meta = row["metadata"]
                    meta_dict = (
                        json.loads(raw_meta) if isinstance(raw_meta, str) else dict(raw_meta or {})
                    )
                    results.append(
                        RetrievedChunk(
                            chunk_id=str(row["chunk_id"]),
                            document_id=str(row["document_id"]),
                            content=str(row["content"]),
                            similarity_score=1.0,  # Exact ID match
                            token_count=int(row["token_count"]),
                            char_start=int(row["char_start"]),
                            char_end=int(row["char_end"]),
                            metadata=meta_dict,
                        )
                    )
                return results
        except Exception as exc:
            logger.error("Failed to fetch chunks by IDs: %s", exc, exc_info=True)
            raise VectorRetrieverError(f"Failed to fetch chunks by IDs: {exc}") from exc
