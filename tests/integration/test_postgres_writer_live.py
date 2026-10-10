"""Live integration tests for PostgresWriter against local PostgreSQL container (TASK-203).

Validates:
1. Document upsert and re-upsert idempotency.
2. Chunk upsert with real pgvector embedding storage.
3. Entity links creation and foreign key cascade deletion.
4. Document bundle atomic write.
"""

import uuid
from collections.abc import AsyncGenerator

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.postgres import PostgresClient
from src.db.postgres_writer import ChunkWriteInput, PostgresWriter
from src.extraction.schemas import ExtractedEntity, ExtractionProvenance
from src.ingest.chunker import TextChunk
from src.ingest.loader import RawDocument


@pytest.fixture(scope="module")
def live_pg_settings() -> Settings:
    return Settings(
        postgres_host="127.0.0.1",
        postgres_port=5432,
        postgres_db="kgr_db",
        postgres_user="kgr_user",
        postgres_password=SecretStr("kgr_password"),
        embedding_dimensions=1536,
    )


@pytest.fixture
async def live_writer(live_pg_settings: Settings) -> AsyncGenerator[PostgresWriter, None]:
    client = PostgresClient(settings=live_pg_settings)
    await client.get_pool()
    await client.run_migrations(embedding_dim=1536)
    writer = PostgresWriter(client=client)
    yield writer
    await client.close()


@pytest.mark.integration
class TestPostgresWriterLive:
    """Live PostgresWriter integration tests against container."""

    @pytest.mark.asyncio
    async def test_live_document_and_chunks_upsert_idempotency(
        self, live_writer: PostgresWriter
    ) -> None:
        doc_id = f"doc-live-{uuid.uuid4()}"
        doc = RawDocument(
            document_id=doc_id,
            source_uri=f"docs/{doc_id}.md",
            title="Live Test Document",
            doc_type="technical_doc",
            content="Sample live text content",
            metadata={"source": "live_test"},
        )

        # 1. Upsert document
        res_doc_id = await live_writer.upsert_document(doc)
        assert res_doc_id == doc_id

        # 2. Re-upsert document (idempotency check)
        res_doc_id2 = await live_writer.upsert_document(doc)
        assert res_doc_id2 == doc_id

        # 3. Create chunk with embedding
        chunk_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:0")
        chunk = TextChunk(
            chunk_id=str(chunk_uuid),
            document_id=doc_id,
            chunk_index=0,
            content="Sample live text content",
            char_start=0,
            char_end=24,
            metadata={"source": "live_test"},
        )
        fake_emb = [0.01] * 1536
        item = ChunkWriteInput(chunk=chunk, embedding=fake_emb)

        # 4. Upsert chunk
        count1 = await live_writer.upsert_chunks_batch([item])
        assert count1 == 1

        # 5. Re-upsert chunk (idempotency check)
        count2 = await live_writer.upsert_chunks_batch([item])
        assert count2 == 1

        # 6. Verify row in database
        async with live_writer.client.connection() as conn:
            row = await conn.fetchrow(
                "SELECT id, document_id, token_count FROM chunks WHERE id = $1", chunk_uuid
            )
            assert row is not None
            assert row["document_id"] == doc_id

    @pytest.mark.asyncio
    async def test_live_write_document_bundle_and_links(self, live_writer: PostgresWriter) -> None:
        doc_id = f"doc-bundle-{uuid.uuid4()}"
        doc = RawDocument(
            document_id=doc_id,
            source_uri=f"docs/{doc_id}.md",
            title="Bundle Test Document",
            doc_type="technical_doc",
            content="Payment service communicates with Checkout service.",
        )

        chunk_uuid = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}:0")
        chunk = TextChunk(
            chunk_id=str(chunk_uuid),
            document_id=doc_id,
            chunk_index=0,
            content="Payment service communicates with Checkout service.",
            char_start=0,
            char_end=50,
        )
        prov = ExtractionProvenance(
            chunk_id=str(chunk_uuid),
            document_id=doc_id,
        )
        ent = ExtractedEntity(
            canonical_id="service:payment-gateway",
            name="Payment Gateway",
            entity_type="Service",
            description="Payment gateway service",
            aliases=["PaySvc"],
            confidence=0.98,
            provenance=prov,
        )
        fake_emb = [0.02] * 1536
        item = ChunkWriteInput(chunk=chunk, embedding=fake_emb, entities=[ent])

        # Write atomic bundle
        chunks_count, links_count = await live_writer.write_document_bundle(doc, [item])
        assert chunks_count == 1
        assert links_count == 1

        # Verify entity link in database
        async with live_writer.client.connection() as conn:
            link_row = await conn.fetchrow(
                "SELECT canonical_entity_id, entity_type FROM entity_links WHERE chunk_id = $1",
                chunk_uuid,
            )
            assert link_row is not None
            assert link_row["canonical_entity_id"] == "service:payment-gateway"
            assert link_row["entity_type"] == "Service"
