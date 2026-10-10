"""Unit tests for the PostgresWriter component (TASK-203).

Validates:
1. Document upsert with idempotent parameter passing.
2. Batch document upsert with transaction chunking.
3. Batch chunk upsert with vector string serialization.
4. Entity links writing and previous link purging.
5. Atomic document bundle writing (document + chunks + entity links in single transaction).
6. Error handling on database/transaction failures.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.postgres import PostgresClient
from src.db.postgres_writer import (
    ChunkWriteInput,
    PostgresWriter,
    PostgresWriterError,
)
from src.extraction.schemas import ExtractedEntity, ExtractionProvenance
from src.ingest.chunker import TextChunk
from src.ingest.loader import RawDocument


@pytest.fixture
def mock_postgres_settings() -> Settings:
    return Settings(
        environment="test",
        postgres_host="localhost",
        postgres_port=5432,
        postgres_db="kgr_test_db",
        postgres_user="kgr_test_user",
        postgres_password=SecretStr("mock_secret_pw"),
        embedding_dimensions=1536,
    )


@pytest.fixture
def sample_raw_doc() -> RawDocument:
    return RawDocument(
        document_id="doc-payments-001",
        source_uri="docs/payments.md",
        title="Payment Service Architecture",
        doc_type="architecture_spec",
        content="Payment Gateway handles debit/credit card charges.",
        metadata={"author": "Payments Team"},
    )


@pytest.fixture
def sample_chunk_input() -> ChunkWriteInput:
    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_OID, "doc-payments-001:0"))
    chunk = TextChunk(
        chunk_id=chunk_id,
        document_id="doc-payments-001",
        chunk_index=0,
        content="Payment Gateway handles debit/credit card charges.",
        char_start=0,
        char_end=50,
        metadata={"author": "Payments Team"},
    )
    prov = ExtractionProvenance(
        chunk_id=chunk_id,
        document_id="doc-payments-001",
    )
    entity = ExtractedEntity(
        canonical_id="service:payment-gateway",
        name="Payment Gateway",
        entity_type="Service",
        description="Processes payments",
        aliases=["PaySvc"],
        confidence=0.95,
        provenance=prov,
    )
    fake_embedding = [0.05] * 1536
    return ChunkWriteInput(
        chunk=chunk,
        embedding=fake_embedding,
        entities=[entity],
    )


class TestPostgresWriterUnit:
    """Unit test suite for PostgresWriter with mocked PostgresClient connection pool."""

    @pytest.mark.asyncio
    async def test_upsert_document_success(
        self,
        mock_postgres_settings: Settings,
        sample_raw_doc: RawDocument,
    ) -> None:
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = "doc-payments-001"

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)
        writer = PostgresWriter(client=client)

        doc_id = await writer.upsert_document(sample_raw_doc)
        assert doc_id == "doc-payments-001"
        mock_conn.fetchval.assert_called_once()
        args = mock_conn.fetchval.call_args[0]
        assert args[1] == "doc-payments-001"
        assert args[2] == "docs/payments.md"
        assert json.loads(args[6]) == {"author": "Payments Team"}

    @pytest.mark.asyncio
    async def test_upsert_document_failure_raises_writer_error(
        self,
        mock_postgres_settings: Settings,
        sample_raw_doc: RawDocument,
    ) -> None:
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = RuntimeError("DB connection dropped")

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)
        writer = PostgresWriter(client=client)

        with pytest.raises(PostgresWriterError, match="Failed to upsert document"):
            await writer.upsert_document(sample_raw_doc)

    @pytest.mark.asyncio
    async def test_upsert_documents_batch_success(
        self,
        mock_postgres_settings: Settings,
        sample_raw_doc: RawDocument,
    ) -> None:
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)

        mock_conn = AsyncMock()
        mock_conn.transaction = MagicMock(return_value=mock_tx)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)
        writer = PostgresWriter(client=client)

        count = await writer.upsert_documents_batch([sample_raw_doc], batch_size=10)
        assert count == 1
        mock_conn.executemany.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_chunks_batch_success(
        self,
        mock_postgres_settings: Settings,
        sample_chunk_input: ChunkWriteInput,
    ) -> None:
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)

        mock_conn = AsyncMock()
        mock_conn.transaction = MagicMock(return_value=mock_tx)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)
        writer = PostgresWriter(client=client)

        count = await writer.upsert_chunks_batch([sample_chunk_input], batch_size=10)
        assert count == 1
        mock_conn.executemany.assert_called_once()
        records = mock_conn.executemany.call_args[0][1]
        assert len(records) == 1
        # Check canonical UUID object and embedding vector serialization
        assert isinstance(records[0][0], uuid.UUID)
        assert records[0][8] is not None
        assert records[0][8].startswith("[")

    @pytest.mark.asyncio
    async def test_write_entity_links_success(
        self,
        mock_postgres_settings: Settings,
        sample_chunk_input: ChunkWriteInput,
    ) -> None:
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)

        mock_conn = AsyncMock()
        mock_conn.transaction = MagicMock(return_value=mock_tx)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)
        writer = PostgresWriter(client=client)

        count = await writer.write_entity_links(
            chunk_id=sample_chunk_input.chunk.chunk_id,
            entities=sample_chunk_input.entities,
        )
        assert count == 1
        mock_conn.execute.assert_called_once()  # delete step
        mock_conn.executemany.assert_called_once()  # insert step

    @pytest.mark.asyncio
    async def test_write_document_bundle_atomic_transaction(
        self,
        mock_postgres_settings: Settings,
        sample_raw_doc: RawDocument,
        sample_chunk_input: ChunkWriteInput,
    ) -> None:
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)

        mock_conn = AsyncMock()
        mock_conn.transaction = MagicMock(return_value=mock_tx)

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)
        writer = PostgresWriter(client=client)

        chunks_written, links_written = await writer.write_document_bundle(
            document=sample_raw_doc,
            chunk_items=[sample_chunk_input],
        )

        assert chunks_written == 1
        assert links_written == 1
        mock_conn.transaction.assert_called_once()
        assert mock_conn.execute.call_count >= 1
        assert mock_conn.executemany.call_count >= 1
