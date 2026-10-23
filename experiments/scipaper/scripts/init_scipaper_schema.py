"""Initialize database schemas in isolated scientific-paper stores only.

Enforces:
- Direct loading of experiments/scipaper/.env.scipaper (no fallback to root .env).
- Strict endpoint verification: PostgreSQL must be on 127.0.0.1:5433 with scipaper_db.
  Neo4j must be on bolt://127.0.0.1:7688.
- Idempotent schema initialization for PostgreSQL (pgvector extension, documents, chunks, entity_links, HNSW index).
- Idempotent schema initialization for Neo4j (Entity uniqueness constraint, fulltext search index).
- Read-only verification of created schema objects.
- Zero ingestion, zero embedding generation, zero external API calls.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import asyncpg
from neo4j import GraphDatabase

from experiments.scipaper.scripts.preflight_check import (
    REQUIRED_NEO4J_BOLT_PORT,
    REQUIRED_POSTGRES_PORT,
    parse_env_file_isolated,
    validate_environment_dict,
)

logger = logging.getLogger("scipaper_schema_init")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = BASE_DIR / ".env.scipaper"


def build_scipaper_postgres_ddl(embedding_dim: int = 384) -> str:
    """Return idempotent SQL DDL for scientific-paper PostgreSQL/pgvector store.

    Preserves full schema compatibility with the canonical migration:
    - documents table with content_hash and metadata
    - chunks table with both embedding (dimension-configured) and embedding_384 columns
    - entity_links table with UUID chunk references
    - HNSW cosine indexes and B-tree indexes
    """
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS documents (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id VARCHAR(256) NOT NULL UNIQUE,
        source_uri VARCHAR(1024) NOT NULL UNIQUE,
        title VARCHAR(512) NOT NULL,
        doc_type VARCHAR(64) NOT NULL DEFAULT 'scientific_paper',
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
        embedding_384 VECTOR(384),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT uq_document_chunk UNIQUE (document_id, chunk_index)
    );

    ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_384 VECTOR(384);

    CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_384_hnsw
    ON chunks USING hnsw (embedding_384 vector_cosine_ops)
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


def get_scipaper_neo4j_ddl() -> list[str]:
    """Return idempotent Cypher DDL statements for scientific-paper Neo4j store."""
    return [
        # Generic entity uniqueness constraint
        "CREATE CONSTRAINT unique_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        # Fulltext index over entity name and description
        "CREATE FULLTEXT INDEX entity_text_index IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.description]",
    ]


async def init_postgres_schema(env_vars: dict[str, str]) -> dict[str, Any]:
    """Connect to isolated PostgreSQL and initialize schemas idempotently."""
    host = env_vars.get("POSTGRES_HOST", "localhost")
    port = int(env_vars["POSTGRES_PORT"])
    user = env_vars["POSTGRES_USER"]
    password = env_vars["POSTGRES_PASSWORD"]
    database = env_vars["POSTGRES_DB"]
    embedding_dim = int(env_vars.get("EMBEDDING_DIMENSIONS", "384"))

    # Strict isolation assertion
    if port != REQUIRED_POSTGRES_PORT:
        raise ValueError(
            f"Aborting: PostgreSQL port {port} is not the isolated port {REQUIRED_POSTGRES_PORT}"
        )
    if database == "kgr_db":
        raise ValueError("Aborting: Target database name cannot be 'kgr_db'")

    logger.info("Connecting to isolated PostgreSQL at %s:%d/%s ...", host, port, database)
    conn = await asyncpg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
    )
    try:
        # Check pgvector extension availability in server
        ext_avail = await conn.fetchval(
            "SELECT COUNT(*) FROM pg_available_extensions WHERE name = 'vector'"
        )
        if not ext_avail:
            raise RuntimeError(
                "pgvector extension is not available in the PostgreSQL container image."
            )

        ddl = build_scipaper_postgres_ddl(embedding_dim=embedding_dim)
        logger.info("Executing PostgreSQL DDL in transaction (embedding_dim=%d)...", embedding_dim)
        async with conn.transaction():
            await conn.execute(ddl)
        logger.info("PostgreSQL DDL executed successfully.")

        # Read-only verification
        tables = ["documents", "chunks", "entity_links"]
        table_statuses: dict[str, bool] = {}
        for t in tables:
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
                t,
            )
            table_statuses[t] = bool(exists)

        vector_ext_active = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )

        indexes = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE tablename IN ('documents', 'chunks', 'entity_links')"
        )
        index_names = [r["indexname"] for r in indexes]

        columns = await conn.fetch(
            "SELECT column_name, data_type, udt_name FROM information_schema.columns WHERE table_name = 'chunks'"
        )
        chunk_cols = {r["column_name"]: r["udt_name"] for r in columns}

        return {
            "tables": table_statuses,
            "vector_extension": bool(vector_ext_active),
            "indexes": index_names,
            "chunk_columns": chunk_cols,
            "embedding_dim": embedding_dim,
        }
    finally:
        await conn.close()


def init_neo4j_schema(env_vars: dict[str, str]) -> dict[str, Any]:
    """Connect to isolated Neo4j and initialize schemas idempotently."""
    uri = env_vars["NEO4J_URI"]
    user = env_vars["NEO4J_USER"]
    password = env_vars["NEO4J_PASSWORD"]

    parsed = urlparse(uri)
    port = parsed.port
    if port != REQUIRED_NEO4J_BOLT_PORT:
        raise ValueError(
            f"Aborting: Neo4j port {port} is not the isolated port {REQUIRED_NEO4J_BOLT_PORT}"
        )

    logger.info("Connecting to isolated Neo4j at %s ...", uri)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        # Connectivity verification
        with driver.session() as session:
            ping = session.run("RETURN 1 AS ping").single()
            if not ping or ping["ping"] != 1:
                raise RuntimeError("Neo4j ping failed.")

            # Apply DDL statements
            statements = get_scipaper_neo4j_ddl()
            for stmt in statements:
                logger.info("Executing Cypher DDL: %s", stmt)
                session.run(stmt)

            # Read-only verification
            constraints_result = session.run("SHOW CONSTRAINTS").data()
            constraint_names = [c.get("name") for c in constraints_result]

            indexes_result = session.run("SHOW INDEXES").data()
            index_names = [i.get("name") for i in indexes_result]

            return {
                "constraints": constraint_names,
                "indexes": index_names,
            }
    finally:
        driver.close()


async def run_initialization(env_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run full schema initialization on isolated stores with strict target checks."""
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    env_vars = parse_env_file_isolated(env_path)
    errors = validate_environment_dict(env_vars, allow_placeholders=False)
    if errors:
        raise ValueError(f"Preflight validation failed for {env_path}: {errors}")

    logger.info("Preflight validation passed. Initializing isolated schemas...")
    pg_results = await init_postgres_schema(env_vars)
    neo_results = init_neo4j_schema(env_vars)

    return pg_results, neo_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize schemas for the scientific-paper RAG experiment in isolated containers."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Path to .env.scipaper environment file",
    )
    args = parser.parse_args()

    print("=================================================================")
    print("Scientific-Paper RAG Schema Initialization (Isolated Only)")
    print("=================================================================")
    print(f"Environment file: {args.env_file}")
    print("-----------------------------------------------------------------")

    try:
        pg_results, neo_results = asyncio.run(run_initialization(args.env_file))
        print("\n[SUCCESS] PostgreSQL Schema Initialized:")
        print(f"  - Vector extension active: {pg_results['vector_extension']}")
        print(f"  - Tables created: {pg_results['tables']}")
        print(f"  - Configured embedding dimension: {pg_results['embedding_dim']}")
        print(
            f"  - Chunks vector columns: {pg_results['chunk_columns'].get('embedding')} / {pg_results['chunk_columns'].get('embedding_384')}"
        )
        print(
            f"  - Indexes: {len(pg_results['indexes'])} verified ({', '.join(pg_results['indexes'])})"
        )

        print("\n[SUCCESS] Neo4j Schema Initialized:")
        print(f"  - Constraints: {neo_results['constraints']}")
        print(f"  - Indexes: {neo_results['indexes']}")

        print("\n[CONFIRMATION] All operations completed against isolated containers.")
        print("  - Zero ingestion performed.")
        print("  - Zero embeddings generated.")
        print("  - Zero external APIs contacted.")
    except Exception as exc:
        logger.error("Initialization failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
