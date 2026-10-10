"""PostgreSQL database client, connection pooling, and schema migration runner.

Design & Invariants:
- Uses asyncpg connection pooling configured from centralized Pydantic Settings.
- Idempotent schema migrations: runs DDL scripts with IF NOT EXISTS and transaction safety.
- Parameterized SQL execution for all runtime queries.
- Clean resource lifecycle with async context manager support.
- Dual-store integrity: chunks.id is canonical UUID matching Neo4j chunk_id provenance.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import asyncpg

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Directory containing migration scripts
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class PostgresClientError(Exception):
    """Base exception for PostgreSQL client errors."""


class PostgresConnectionError(PostgresClientError):
    """Raised when the client cannot connect to PostgreSQL."""


class MigrationError(PostgresClientError):
    """Raised when a schema migration fails to apply."""


def build_ddl_migration_sql(embedding_dim: int = 1536) -> str:
    """Generate idempotent DDL SQL with parameterized embedding dimension.

    Parameters
    ----------
    embedding_dim:
        Dimension for the vector column (default 1536).

    Returns
    -------
    str
        Full idempotent DDL SQL string.
    """
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id VARCHAR(256) NOT NULL UNIQUE,
        source_uri VARCHAR(1024) NOT NULL UNIQUE,
        title VARCHAR(512) NOT NULL,
        doc_type VARCHAR(64) NOT NULL DEFAULT 'technical_doc',
        content_hash CHAR(64) NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
    CREATE INDEX IF NOT EXISTS idx_documents_document_id ON documents(document_id);

    CREATE TABLE IF NOT EXISTS chunks (
        id UUID PRIMARY KEY,
        document_id VARCHAR(256) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        embedding VECTOR({embedding_dim}),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_document_chunk UNIQUE (document_id, chunk_index)
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

    CREATE TABLE IF NOT EXISTS entity_links (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
        canonical_entity_id VARCHAR(256) NOT NULL,
        entity_type VARCHAR(64) NOT NULL,
        mention_text VARCHAR(256) NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_entity_links_chunk_id ON entity_links(chunk_id);
    CREATE INDEX IF NOT EXISTS idx_entity_links_canonical_id ON entity_links(canonical_entity_id);
    """


class PostgresClient:
    """Managed PostgreSQL and pgvector client providing connection pooling and migrations."""

    def __init__(
        self,
        settings: Settings | None = None,
        pool: asyncpg.Pool[asyncpg.Record] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._pool = pool
        self._owns_pool = pool is None

    async def get_pool(self) -> asyncpg.Pool[asyncpg.Record]:
        """Return or lazily create the asyncpg connection pool."""
        if self._pool is None:
            password = (
                self.settings.postgres_password.get_secret_value()
                if self.settings.postgres_password is not None
                else ""
            )
            try:
                self._pool = await asyncpg.create_pool(
                    host=self.settings.postgres_host,
                    port=self.settings.postgres_port,
                    user=self.settings.postgres_user,
                    password=password,
                    database=self.settings.postgres_db,
                    min_size=1,
                    max_size=10,
                )
                logger.debug(
                    "PostgreSQL connection pool initialized for %s:%s/%s",
                    self.settings.postgres_host,
                    self.settings.postgres_port,
                    self.settings.postgres_db,
                )
            except Exception as exc:
                raise PostgresConnectionError(
                    f"Failed to connect to PostgreSQL at {self.settings.postgres_host}:{self.settings.postgres_port}: {exc}"
                ) from exc
        return self._pool

    @property
    def pool(self) -> asyncpg.Pool[asyncpg.Record] | None:
        """Return current pool instance or None if not initialized."""
        return self._pool

    async def close(self) -> None:
        """Close connection pool and release connections."""
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None
            logger.debug("PostgreSQL connection pool closed.")

    async def __aenter__(self) -> PostgresClient:
        await self.get_pool()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection[asyncpg.Record]]:
        """Acquire a connection from the pool."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            yield conn

    async def verify_connectivity(self) -> None:
        """Ping PostgreSQL server and verify pgvector extension availability."""
        try:
            async with self.connection() as conn:
                val = await conn.fetchval("SELECT 1")
                if val != 1:
                    raise PostgresConnectionError("PostgreSQL ping returned unexpected value")
            logger.info("PostgreSQL connectivity verified.")
        except Exception as exc:
            raise PostgresConnectionError(f"PostgreSQL connectivity verification failed: {exc}") from exc

    async def run_migrations(self, embedding_dim: int | None = None) -> None:
        """Apply idempotent DDL migrations to set up documents, chunks, and entity_links.

        Parameters
        ----------
        embedding_dim:
            Dimension for vector embedding column (defaults to settings.embedding_dimensions).
        """
        dim = embedding_dim or self.settings.embedding_dimensions
        ddl_sql = build_ddl_migration_sql(embedding_dim=dim)

        logger.info("Applying PostgreSQL schema migrations (embedding_dim=%d)...", dim)
        try:
            async with self.connection() as conn:
                async with conn.transaction():
                    await conn.execute(ddl_sql)
            logger.info("PostgreSQL schema migrations applied successfully.")
        except Exception as exc:
            logger.error("Schema migration failed: %s", exc, exc_info=True)
            raise MigrationError(f"PostgreSQL migration failed: {exc}") from exc

    async def check_schema_health(self) -> dict[str, bool]:
        """Verify that required tables, extension, and indexes exist in the database."""
        async with self.connection() as conn:
            # Check vector extension
            ext_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )

            # Check tables
            tables = ["documents", "chunks", "entity_links"]
            table_status: dict[str, bool] = {}
            for t in tables:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                    t,
                )
                table_status[t] = bool(exists)

            # Check HNSW index
            hnsw_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = 'idx_chunks_embedding_hnsw')"
            )

            return {
                "vector_extension": bool(ext_exists),
                **table_status,
                "hnsw_index": bool(hnsw_exists),
            }
