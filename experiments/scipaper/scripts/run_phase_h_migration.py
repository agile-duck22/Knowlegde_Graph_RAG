"""Phase H Full Corpus Resumable Batch Migration Coordinator (Gates D & E).

Architecture & Constraints:
- Processes 120 papers in batches of 10 papers (12 batches: 11 batches of 10, 1 batch of 10).
- Deterministic ordering by document_id.
- Persists progress to phase_h_migration_checkpoint.json.
- Skips already-verified completed batches on resume.
- Pre-batch snapshot and post-batch validation for every batch.
- Strictly isolated: PostgreSQL port 5433, Neo4j port 7688.
- Zero paid APIs, zero model downloads, zero modification of chunks or embeddings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
from neo4j import GraphDatabase

from experiments.scipaper.schemas.extraction_schemas import (
    ScientificChunkExtractionResult,
)
from experiments.scipaper.schemas.extractor import OfflineScientificExtractor
from experiments.scipaper.schemas.neo4j_writer import (
    IsolatedScientificNeo4jWriter,
    TargetIsolationError,
)
from experiments.scipaper.scripts.preflight_check import (
    ENTERPRISE_PORTS,
    REQUIRED_NEO4J_BOLT_PORT,
    REQUIRED_POSTGRES_PORT,
    parse_env_file_isolated,
)
from src.ingest.chunker import DocumentChunker
from src.ingest.loader import RawDocument

logger = logging.getLogger("phase_h_migration")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV = BASE_DIR / ".env.scipaper"
DEFAULT_CORPUS = BASE_DIR / "data" / "corpus_scipaper.jsonl"
DEFAULT_CHECKPOINT = BASE_DIR / "eval" / "results" / "phase_h_migration_checkpoint.json"
DEFAULT_RESULTS_DIR = BASE_DIR / "eval" / "results"
DEFAULT_BATCH_SIZE = 10
PHASE_G_DOC_ID = "paper-arxiv-2411.18583"


def enforce_isolation_guards(env_vars: dict[str, str]) -> None:
    """Enforce strict target guards before attempting any live connection."""
    pg_host = (env_vars.get("POSTGRES_HOST") or "localhost").lower()
    if pg_host not in ("localhost", "127.0.0.1", "::1"):
        raise TargetIsolationError(f"PostgreSQL host '{pg_host}' is not loopback.")

    pg_port = int(env_vars.get("POSTGRES_PORT", 0))
    if pg_port in ENTERPRISE_PORTS or pg_port != REQUIRED_POSTGRES_PORT:
        raise TargetIsolationError(
            f"PostgreSQL port {pg_port} is invalid. Refusing connection to non-isolated port."
        )

    pg_db = env_vars.get("POSTGRES_DB", "")
    if pg_db != "scipaper_db":
        raise TargetIsolationError(
            f"PostgreSQL DB '{pg_db}' is invalid. Refusing connection to non-scipaper DB."
        )

    from urllib.parse import urlparse

    neo4j_uri = env_vars.get("NEO4J_URI", "")
    parsed = urlparse(neo4j_uri)
    neo_host = (parsed.hostname or "").lower()
    if neo_host not in ("localhost", "127.0.0.1", "::1"):
        raise TargetIsolationError(f"Neo4j host '{neo_host}' is not loopback.")

    if parsed.port in ENTERPRISE_PORTS or parsed.port != REQUIRED_NEO4J_BOLT_PORT:
        raise TargetIsolationError(
            f"Neo4j port {parsed.port} is invalid. Refusing connection to non-isolated port."
        )

    allow_paid = env_vars.get("ALLOW_PAID_EMBEDDINGS", "").strip().lower()
    if allow_paid != "false":
        raise TargetIsolationError(f"ALLOW_PAID_EMBEDDINGS={allow_paid}; must be false.")


@dataclass
class BatchRecord:
    batch_index: int
    paper_ids: list[str]
    status: str  # 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED'
    started_at: str | None = None
    completed_at: str | None = None
    pre_nodes_count: int = 0
    pre_rels_count: int = 0
    post_nodes_count: int = 0
    post_rels_count: int = 0
    pg_elinks_added: int = 0
    error_message: str | None = None


@dataclass
class MigrationCheckpoint:
    migration_started_at: str
    last_updated_at: str
    total_papers: int
    batch_size: int
    total_batches: int
    completed_batches_count: int
    completed_paper_ids: list[str]
    batches: list[BatchRecord] = field(default_factory=list)


def load_corpus_documents(corpus_path: Path) -> list[RawDocument]:
    docs: list[RawDocument] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            docs.append(
                RawDocument(
                    document_id=rec["document_id"],
                    source_uri=rec["source_uri"],
                    title=rec["title"],
                    doc_type=rec["doc_type"],
                    content=rec["content"],
                    metadata=rec.get("metadata", {}),
                )
            )
    # Sort deterministically by document_id
    docs.sort(key=lambda d: d.document_id)
    return docs


def load_or_create_checkpoint(
    checkpoint_path: Path, docs: list[RawDocument], batch_size: int
) -> MigrationCheckpoint:
    if checkpoint_path.exists():
        data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        batches = [BatchRecord(**b) for b in data["batches"]]
        data["batches"] = batches
        checkpoint = MigrationCheckpoint(**data)
        logger.info(
            "Loaded existing checkpoint: %d/%d batches completed.",
            checkpoint.completed_batches_count,
            checkpoint.total_batches,
        )
        return checkpoint

    # Create fresh checkpoint structure
    total_papers = len(docs)
    total_batches = (total_papers + batch_size - 1) // batch_size
    fresh_batches: list[BatchRecord] = []
    for b_idx in range(total_batches):
        b_docs = docs[b_idx * batch_size : (b_idx + 1) * batch_size]
        fresh_batches.append(
            BatchRecord(
                batch_index=b_idx,
                paper_ids=[d.document_id for d in b_docs],
                status="PENDING",
            )
        )

    now = datetime.now(UTC).isoformat()
    checkpoint = MigrationCheckpoint(
        migration_started_at=now,
        last_updated_at=now,
        total_papers=total_papers,
        batch_size=batch_size,
        total_batches=total_batches,
        completed_batches_count=0,
        completed_paper_ids=[],
        batches=fresh_batches,
    )
    save_checkpoint(checkpoint_path, checkpoint)
    return checkpoint


def save_checkpoint(checkpoint_path: Path, checkpoint: MigrationCheckpoint) -> None:
    checkpoint.last_updated_at = datetime.now(UTC).isoformat()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(asdict(checkpoint), indent=2), encoding="utf-8")


async def execute_batch(
    batch: BatchRecord,
    docs_by_id: dict[str, RawDocument],
    env_vars: dict[str, str],
    checkpoint_path: Path,
    checkpoint: MigrationCheckpoint,
) -> None:
    logger.info(
        "Starting Batch %d/%d with %d papers: %s",
        batch.batch_index + 1,
        checkpoint.total_batches,
        len(batch.paper_ids),
        batch.paper_ids,
    )
    batch.started_at = datetime.now(UTC).isoformat()
    batch.status = "RUNNING"
    save_checkpoint(checkpoint_path, checkpoint)

    # 1. Pre-batch database count snapshot
    driver = GraphDatabase.driver(
        env_vars["NEO4J_URI"], auth=(env_vars["NEO4J_USER"], env_vars["NEO4J_PASSWORD"])
    )
    with driver.session() as s:
        rec_n = s.run("MATCH (n) RETURN count(n) AS cnt").single()
        rec_r = s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
        pre_nodes = rec_n["cnt"] if rec_n else 0
        pre_rels = rec_r["cnt"] if rec_r else 0
    batch.pre_nodes_count = pre_nodes
    batch.pre_rels_count = pre_rels

    # 2. Extract and write
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    extractor = OfflineScientificExtractor()
    neo_writer = IsolatedScientificNeo4jWriter(
        uri=env_vars["NEO4J_URI"],
        user=env_vars["NEO4J_USER"],
        password=env_vars["NEO4J_PASSWORD"],
    )

    pg_conn = await asyncpg.connect(
        host=env_vars["POSTGRES_HOST"],
        port=int(env_vars["POSTGRES_PORT"]),
        database=env_vars["POSTGRES_DB"],
        user=env_vars["POSTGRES_USER"],
        password=env_vars["POSTGRES_PASSWORD"],
    )

    batch_elinks_added = 0
    try:
        for doc_id in batch.paper_ids:
            doc = docs_by_id[doc_id]
            chunks = chunker.chunk(doc)

            # Assert chunks exist in PG
            db_chunks = await pg_conn.fetch(
                "SELECT id, chunk_index FROM chunks WHERE document_id = $1 ORDER BY chunk_index;",
                doc_id,
            )
            if len(db_chunks) != len(chunks):
                raise ValueError(
                    f"PG chunk mismatch on {doc_id}: expected {len(chunks)}, got {len(db_chunks)}"
                )

            extracted_results: list[ScientificChunkExtractionResult] = [
                extractor.extract(c) for c in chunks
            ]

            # Write to Neo4j
            for res in extracted_results:
                neo_writer.write_chunk_result(res)

            # Write entity links to PostgreSQL
            for res in extracted_results:
                for ent in res.entities:
                    existing = await pg_conn.fetchrow(
                        "SELECT id FROM entity_links WHERE chunk_id = $1::uuid AND canonical_entity_id = $2;",
                        res.chunk_id,
                        ent.canonical_id,
                    )
                    if not existing:
                        await pg_conn.execute(
                            """
                            INSERT INTO entity_links (chunk_id, canonical_entity_id, entity_type, mention_text, confidence, created_at)
                            VALUES ($1::uuid, $2, $3, $4, $5, NOW())
                            """,
                            res.chunk_id,
                            ent.canonical_id,
                            ent.entity_type,
                            ent.name,
                            ent.confidence,
                        )
                        batch_elinks_added += 1

        neo_writer.close()

        # 3. Post-batch database count verification
        with driver.session() as s:
            rec_n = s.run("MATCH (n) RETURN count(n) AS cnt").single()
            rec_r = s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
            post_nodes = rec_n["cnt"] if rec_n else 0
            post_rels = rec_r["cnt"] if rec_r else 0

        # Assert PG documents and chunks have not been altered
        pg_docs_cnt = await pg_conn.fetchval("SELECT count(*) FROM documents;")
        pg_chunks_cnt = await pg_conn.fetchval("SELECT count(*) FROM chunks;")
        pg_emb384_cnt = await pg_conn.fetchval(
            "SELECT count(*) FROM chunks WHERE embedding_384 IS NOT NULL;"
        )
        if pg_docs_cnt != 120 or pg_chunks_cnt != 474 or pg_emb384_cnt != 474:
            raise ValueError(
                f"PostgreSQL corpus corruption detected! docs={pg_docs_cnt}, chunks={pg_chunks_cnt}, vectors={pg_emb384_cnt}"
            )

        batch.post_nodes_count = post_nodes
        batch.post_rels_count = post_rels
        batch.pg_elinks_added = batch_elinks_added
        batch.completed_at = datetime.now(UTC).isoformat()
        batch.status = "COMPLETED"

        checkpoint.completed_batches_count += 1
        checkpoint.completed_paper_ids.extend(batch.paper_ids)
        save_checkpoint(checkpoint_path, checkpoint)

        logger.info(
            "Batch %d COMPLETED: nodes %d->%d (+%d), rels %d->%d (+%d), elinks added: %d",
            batch.batch_index + 1,
            pre_nodes,
            post_nodes,
            post_nodes - pre_nodes,
            pre_rels,
            post_rels,
            post_rels - pre_rels,
            batch_elinks_added,
        )

    except Exception as exc:
        batch.status = "FAILED"
        batch.error_message = str(exc)
        save_checkpoint(checkpoint_path, checkpoint)
        logger.error("Batch %d FAILED: %s", batch.batch_index + 1, exc)
        raise
    finally:
        await pg_conn.close()
        driver.close()


async def run_full_migration(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    corpus_path: Path = DEFAULT_CORPUS,
    env_path: Path = DEFAULT_ENV,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> MigrationCheckpoint:
    env_vars = parse_env_file_isolated(env_path)
    enforce_isolation_guards(env_vars)

    docs = load_corpus_documents(corpus_path)
    docs_by_id = {d.document_id: d for d in docs}

    checkpoint = load_or_create_checkpoint(checkpoint_path, docs, batch_size)

    logger.info(
        "Starting/Resuming Phase H Migration: %d papers total, %d batches, %d completed.",
        checkpoint.total_papers,
        checkpoint.total_batches,
        checkpoint.completed_batches_count,
    )

    for batch in checkpoint.batches:
        if batch.status == "COMPLETED":
            logger.info("Skipping already completed Batch %d.", batch.batch_index + 1)
            continue

        await execute_batch(batch, docs_by_id, env_vars, checkpoint_path, checkpoint)

    logger.info("All %d batches successfully completed!", checkpoint.total_batches)
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase H Resumable Migration")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    asyncio.run(run_full_migration(batch_size=args.batch_size))
