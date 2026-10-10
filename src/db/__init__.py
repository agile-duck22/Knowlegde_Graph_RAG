"""Database layer — Neo4j and PostgreSQL clients and writers."""

from src.db.neo4j import (
    BatchWriteError,
    Neo4jClient,
    Neo4jClientError,
    Neo4jConnectionError,
    UnsupportedEntityTypeError,
    UnsupportedRelationshipTypeError,
)
from src.db.postgres import (
    MigrationError,
    PostgresClient,
    PostgresClientError,
    PostgresConnectionError,
    build_ddl_migration_sql,
)
from src.db.postgres_writer import (
    BatchUpsertError,
    ChunkWriteInput,
    DocumentNotFoundError,
    PostgresWriter,
    PostgresWriterError,
)

__all__ = [
    "BatchUpsertError",
    "BatchWriteError",
    "ChunkWriteInput",
    "DocumentNotFoundError",
    "MigrationError",
    "Neo4jClient",
    "Neo4jClientError",
    "Neo4jConnectionError",
    "PostgresClient",
    "PostgresClientError",
    "PostgresConnectionError",
    "PostgresWriter",
    "PostgresWriterError",
    "UnsupportedEntityTypeError",
    "UnsupportedRelationshipTypeError",
    "build_ddl_migration_sql",
]
