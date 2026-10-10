"""Unit tests for PostgreSQL client, schema migration SQL generation, and error handling (TASK-201)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.postgres import (
    MigrationError,
    PostgresClient,
    PostgresConnectionError,
    build_ddl_migration_sql,
)


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


class TestDDLMigrationSQL:
    """Validate DDL structure, constraints, indexes, and vector dimensions."""

    def test_ddl_contains_required_extensions_and_tables(self) -> None:
        sql = build_ddl_migration_sql(embedding_dim=1536)
        assert "CREATE EXTENSION IF NOT EXISTS vector;" in sql
        assert "CREATE TABLE IF NOT EXISTS documents" in sql
        assert "CREATE TABLE IF NOT EXISTS chunks" in sql
        assert "CREATE TABLE IF NOT EXISTS entity_links" in sql

    def test_ddl_embedding_dimension_configurable(self) -> None:
        sql_384 = build_ddl_migration_sql(embedding_dim=384)
        assert "embedding VECTOR(384)" in sql_384

        sql_1536 = build_ddl_migration_sql(embedding_dim=1536)
        assert "embedding VECTOR(1536)" in sql_1536

    def test_ddl_contains_hnsw_cosine_index(self) -> None:
        sql = build_ddl_migration_sql(embedding_dim=1536)
        assert "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw" in sql
        assert "USING hnsw (embedding vector_cosine_ops)" in sql
        assert "WITH (m = 16, ef_construction = 64)" in sql

    def test_ddl_foreign_keys_and_cascades(self) -> None:
        sql = build_ddl_migration_sql(embedding_dim=1536)
        assert "REFERENCES documents(document_id) ON DELETE CASCADE" in sql
        assert "REFERENCES chunks(id) ON DELETE CASCADE" in sql
        assert "CONSTRAINT uq_document_chunk UNIQUE (document_id, chunk_index)" in sql


class TestPostgresClientUnit:
    """Unit tests for PostgresClient using mocked asyncpg pool and connections."""

    @pytest.mark.asyncio
    async def test_client_lifecycle_context_manager(self) -> None:
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()

        client = PostgresClient(pool=mock_pool)
        async with client:
            assert client.pool == mock_pool

        # Since pool was injected externally, client shouldn't close it by default
        mock_pool.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_connectivity_success(
        self,
        mock_postgres_settings: Settings,
    ) -> None:
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)
        await client.verify_connectivity()
        mock_conn.fetchval.assert_called_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_verify_connectivity_failure_raises_postgres_connection_error(
        self,
        mock_postgres_settings: Settings,
    ) -> None:
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = RuntimeError("Database unreachable")

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)
        with pytest.raises(PostgresConnectionError, match="connectivity verification failed"):
            await client.verify_connectivity()

    @pytest.mark.asyncio
    async def test_run_migrations_success(
        self,
        mock_postgres_settings: Settings,
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
        await client.run_migrations(embedding_dim=1536)

        mock_conn.execute.assert_called_once()
        executed_sql = mock_conn.execute.call_args[0][0]
        assert "CREATE EXTENSION IF NOT EXISTS vector;" in executed_sql
        assert "VECTOR(1536)" in executed_sql

    @pytest.mark.asyncio
    async def test_run_migrations_failure_raises_migration_error(
        self,
        mock_postgres_settings: Settings,
    ) -> None:
        mock_tx = MagicMock()
        mock_tx.__aenter__ = AsyncMock(return_value=None)
        mock_tx.__aexit__ = AsyncMock(return_value=None)

        mock_conn = AsyncMock()
        mock_conn.transaction = MagicMock(return_value=mock_tx)
        mock_conn.execute.side_effect = RuntimeError("Syntax error in DDL")

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)
        with pytest.raises(MigrationError, match="PostgreSQL migration failed"):
            await client.run_migrations()

    @pytest.mark.asyncio
    async def test_check_schema_health_all_present(
        self,
        mock_postgres_settings: Settings,
    ) -> None:
        mock_conn = AsyncMock()
        # Mock responses for vector extension, 3 tables, and 1 index
        mock_conn.fetchval.side_effect = [True, True, True, True, True]

        mock_pool = MagicMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.acquire.return_value.__aexit__.return_value = None

        client = PostgresClient(settings=mock_postgres_settings, pool=mock_pool)
        health = await client.check_schema_health()

        assert health["vector_extension"] is True
        assert health["documents"] is True
        assert health["chunks"] is True
        assert health["entity_links"] is True
        assert health["hnsw_index"] is True
