"""Unit tests for VectorRetriever (TASK-204).

Validates:
1. Vector similarity search query construction and cosine similarity conversion.
2. Similarity threshold filtering (1 - distance >= threshold).
3. Parameterized query filters: document_id, doc_type, metadata_filter.
4. Top-k limiting and input validation (top_k >= 1, dimension mismatch, NaN/Inf).
5. get_chunks_by_ids canonical UUID hydration.
6. Empty results and database error handling.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.postgres import PostgresClient
from src.retrieval.vector_retriever import (
    InvalidQueryVectorError,
    RetrievedChunk,
    VectorRetriever,
    VectorRetrieverError,
)


@pytest.fixture
def mock_retriever_settings() -> Settings:
    return Settings(
        environment="test",
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="kgr_test_db",
        postgres_user="kgr_test_user",
        postgres_password=SecretStr("mock_secret_pw"),
        embedding_dimensions=1536,
        vector_top_k=5,
        vector_similarity_threshold=0.60,
    )


class TestVectorRetrieverUnit:
    """Unit test suite for VectorRetriever with mocked PostgresClient."""

    @pytest.mark.asyncio
    async def test_search_happy_path(
        self,
        mock_retriever_settings: Settings,
    ) -> None:
        chunk_uuid = uuid.uuid4()
        mock_row = {
            "chunk_id": chunk_uuid,
            "document_id": "doc-001",
            "content": "Payment Gateway Spec content",
            "similarity_score": 0.88,
            "token_count": 10,
            "char_start": 0,
            "char_end": 28,
            "metadata": {"doc_type": "technical_doc"},
        }

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [mock_row]

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_retriever_settings, pool=mock_pool)
        retriever = VectorRetriever(client=client, settings=mock_retriever_settings)

        query_vec = [0.05] * 1536
        results = await retriever.search(query_vec, top_k=3, similarity_threshold=0.70)

        assert len(results) == 1
        res = results[0]
        assert isinstance(res, RetrievedChunk)
        assert res.chunk_id == str(chunk_uuid)
        assert res.document_id == "doc-001"
        assert res.similarity_score == 0.88
        assert res.metadata == {"doc_type": "technical_doc"}

        # Verify SQL parameter structure
        mock_conn.fetch.assert_called_once()
        sql_query, *args = mock_conn.fetch.call_args[0]
        assert "(1 - (c.embedding <=> $1::vector)) >= $2" in sql_query
        assert "LIMIT $3" in sql_query
        assert args[1] == 0.70  # similarity threshold passed as parameter
        assert args[2] == 3  # top_k passed as parameter

    @pytest.mark.asyncio
    async def test_search_with_document_id_and_metadata_filter(
        self,
        mock_retriever_settings: Settings,
    ) -> None:
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = []

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_retriever_settings, pool=mock_pool)
        retriever = VectorRetriever(client=client, settings=mock_retriever_settings)

        query_vec = [0.01] * 1536
        await retriever.search(
            query_vec,
            document_id="doc-001",
            doc_type="architecture_spec",
            metadata_filter={"author": "Team A"},
        )

        mock_conn.fetch.assert_called_once()
        sql_query, *args = mock_conn.fetch.call_args[0]
        assert "JOIN documents d ON c.document_id = d.document_id" in sql_query
        assert "c.document_id = $3" in sql_query
        assert "d.doc_type = $4" in sql_query
        assert "c.metadata @> $5::jsonb" in sql_query
        assert "LIMIT $6" in sql_query

    @pytest.mark.asyncio
    async def test_search_dimension_mismatch_raises_invalid_query_vector_error(
        self,
        mock_retriever_settings: Settings,
    ) -> None:
        client = PostgresClient(settings=mock_retriever_settings)
        retriever = VectorRetriever(client=client, settings=mock_retriever_settings)

        # Pass 768 dimensions instead of 1536
        wrong_vec = [0.1] * 768
        with pytest.raises(
            InvalidQueryVectorError, match="dimension 768 does not match expected 1536"
        ):
            await retriever.search(wrong_vec)

    @pytest.mark.asyncio
    async def test_search_nan_in_vector_raises_invalid_query_vector_error(
        self,
        mock_retriever_settings: Settings,
    ) -> None:
        client = PostgresClient(settings=mock_retriever_settings)
        retriever = VectorRetriever(client=client, settings=mock_retriever_settings)

        corrupt_vec = [0.1] * 1536
        corrupt_vec[5] = float("nan")
        with pytest.raises(InvalidQueryVectorError, match="non-finite or non-numeric value"):
            await retriever.search(corrupt_vec)

    @pytest.mark.asyncio
    async def test_search_invalid_top_k_raises_value_error(
        self,
        mock_retriever_settings: Settings,
    ) -> None:
        client = PostgresClient(settings=mock_retriever_settings)
        retriever = VectorRetriever(client=client, settings=mock_retriever_settings)

        query_vec = [0.01] * 1536
        with pytest.raises(ValueError, match="top_k must be >= 1"):
            await retriever.search(query_vec, top_k=0)

    @pytest.mark.asyncio
    async def test_search_database_failure_raises_retriever_error(
        self,
        mock_retriever_settings: Settings,
    ) -> None:
        mock_conn = AsyncMock()
        mock_conn.fetch.side_effect = RuntimeError("Connection closed unexpectedly")

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_retriever_settings, pool=mock_pool)
        retriever = VectorRetriever(client=client, settings=mock_retriever_settings)

        query_vec = [0.01] * 1536
        with pytest.raises(VectorRetrieverError, match="Vector search failed"):
            await retriever.search(query_vec)

    @pytest.mark.asyncio
    async def test_get_chunks_by_ids_success(
        self,
        mock_retriever_settings: Settings,
    ) -> None:
        chunk_uuid = uuid.uuid4()
        mock_row = {
            "chunk_id": chunk_uuid,
            "document_id": "doc-002",
            "content": "Checkout service specs",
            "token_count": 8,
            "char_start": 0,
            "char_end": 22,
            "metadata": {},
        }

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [mock_row]

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_retriever_settings, pool=mock_pool)
        retriever = VectorRetriever(client=client, settings=mock_retriever_settings)

        results = await retriever.get_chunks_by_ids([str(chunk_uuid), "invalid-uuid-ignored"])
        assert len(results) == 1
        assert results[0].chunk_id == str(chunk_uuid)
        assert results[0].similarity_score == 1.0

    @pytest.mark.asyncio
    async def test_get_chunks_by_ids_empty(
        self,
        mock_retriever_settings: Settings,
    ) -> None:
        client = PostgresClient(settings=mock_retriever_settings)
        retriever = VectorRetriever(client=client, settings=mock_retriever_settings)
        results = await retriever.get_chunks_by_ids([])
        assert results == []
