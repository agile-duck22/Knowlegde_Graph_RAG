"""Live integration tests for PostgreSQL/pgvector client and DDL migrations against running Docker container (TASK-201).

Tagged with @pytest.mark.integration so standard offline unit test runs do not execute them.
"""

from collections.abc import AsyncGenerator

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.postgres import PostgresClient


@pytest.fixture(scope="module")
def live_pg_settings() -> Settings:
    """Settings pointing to local PostgreSQL container (kgr-postgres)."""
    return Settings(
        postgres_host="127.0.0.1",
        postgres_port=5432,
        postgres_db="kgr_db",
        postgres_user="kgr_user",
        postgres_password=SecretStr("kgr_password"),
        embedding_dimensions=1536,
    )


@pytest.fixture
async def live_pg_client(live_pg_settings: Settings) -> AsyncGenerator[PostgresClient, None]:
    """Provide connected PostgresClient to local container."""
    client = PostgresClient(settings=live_pg_settings)
    await client.get_pool()
    yield client
    await client.close()


@pytest.mark.integration
class TestPostgresLiveIntegration:
    """Live PostgreSQL / pgvector container verification."""

    @pytest.mark.asyncio
    async def test_01_verify_connectivity(self, live_pg_client: PostgresClient) -> None:
        """Driver can ping PostgreSQL over TCP."""
        await live_pg_client.verify_connectivity()

    @pytest.mark.asyncio
    async def test_02_apply_migrations_idempotently(self, live_pg_client: PostgresClient) -> None:
        """Migrations can be executed repeatedly without errors."""
        await live_pg_client.run_migrations(embedding_dim=1536)
        # Re-apply to verify idempotency
        await live_pg_client.run_migrations(embedding_dim=1536)

    @pytest.mark.asyncio
    async def test_03_schema_health_verification(self, live_pg_client: PostgresClient) -> None:
        """Verify vector extension, tables, and HNSW index exist."""
        health = await live_pg_client.check_schema_health()
        assert health["vector_extension"] is True
        assert health["documents"] is True
        assert health["chunks"] is True
        assert health["entity_links"] is True
        assert health["hnsw_index"] is True

    @pytest.mark.asyncio
    async def test_04_relational_constraints_and_cascades(self, live_pg_client: PostgresClient) -> None:
        """Test document -> chunk -> entity_link relational constraints and ON DELETE CASCADE."""
        import uuid

        doc_id = f"test-doc-{uuid.uuid4()}"
        chunk_uuid = uuid.uuid4()

        async with live_pg_client.connection() as conn:
            # 1. Insert document
            await conn.execute(
                """
                INSERT INTO documents (document_id, source_uri, title, doc_type, content_hash)
                VALUES ($1, $2, $3, $4, $5)
                """,
                doc_id,
                f"docs/{doc_id}.md",
                "Test Doc Title",
                "technical_doc",
                "a" * 64,
            )

            # 2. Insert chunk
            await conn.execute(
                """
                INSERT INTO chunks (id, document_id, chunk_index, content, char_start, char_end)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                chunk_uuid,
                doc_id,
                0,
                "Sample chunk text content",
                0,
                24,
            )

            # 3. Insert entity link
            await conn.execute(
                """
                INSERT INTO entity_links (chunk_id, canonical_entity_id, entity_type, mention_text, confidence)
                VALUES ($1, $2, $3, $4, $5)
                """,
                chunk_uuid,
                "service:test-svc",
                "Service",
                "Test Service",
                0.95,
            )

            # 4. Verify presence
            chunk_row = await conn.fetchrow("SELECT document_id FROM chunks WHERE id = $1", chunk_uuid)
            assert chunk_row is not None
            assert chunk_row["document_id"] == doc_id

            link_row = await conn.fetchrow("SELECT canonical_entity_id FROM entity_links WHERE chunk_id = $1", chunk_uuid)
            assert link_row is not None
            assert link_row["canonical_entity_id"] == "service:test-svc"

            # 5. Delete document and verify cascade
            await conn.execute("DELETE FROM documents WHERE document_id = $1", doc_id)

            remaining_chunks = await conn.fetchval("SELECT count(*) FROM chunks WHERE id = $1", chunk_uuid)
            remaining_links = await conn.fetchval("SELECT count(*) FROM entity_links WHERE chunk_id = $1", chunk_uuid)
            assert remaining_chunks == 0
            assert remaining_links == 0
