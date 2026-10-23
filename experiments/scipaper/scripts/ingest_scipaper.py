"""CLI and Execution Adapter for Scientific-Paper Ingestion.

Modes:
- --validate-only (default): Validate corpus, manifest, schemas, IDs, and config offline.
  Zero database connections, zero embeddings, zero writes.
- --dry-run: Plan ingestion, verify isolated configuration, count chunks, but perform
  zero writes and zero embedding generation.
- --execute-ingestion: Live ingestion execution for the 120-paper scientific corpus.
  Executes in bounded batches of 20 documents (6 batches total) with post-batch verification.
  Fail-closed if targets do not match isolated ports 5433 / 7688 or if ALLOW_PAID_EMBEDDINGS is true.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from sentence_transformers import SentenceTransformer

from experiments.scipaper.schemas.batch_coordinator import (
    DocumentState,
    ScientificBatchCoordinator,
    enforce_strict_isolation_guards,
)
from experiments.scipaper.schemas.neo4j_writer import IsolatedScientificNeo4jWriter
from experiments.scipaper.scripts.preflight_check import (
    ENTERPRISE_PORTS,
    REQUIRED_NEO4J_BOLT_PORT,
    REQUIRED_POSTGRES_PORT,
    parse_env_file_isolated,
    validate_environment_dict,
)
from src.config import Settings
from src.db.postgres import PostgresClient
from src.ingest.chunker import DocumentChunker
from src.ingest.loader import RawDocument

logger = logging.getLogger("scipaper_ingest")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV = BASE_DIR / ".env.scipaper"
DEFAULT_CORPUS = BASE_DIR / "data" / "corpus_scipaper.jsonl"
DEFAULT_MANIFEST = BASE_DIR / "data" / "manifest.jsonl"


def validate_corpus_offline(corpus_path: Path, manifest_path: Path) -> tuple[int, int]:
    """Validate all corpus documents and count deterministic chunks offline."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    docs_count = 0
    total_chunks = 0
    seen_doc_ids = set()
    seen_uris = set()

    with open(corpus_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            doc = RawDocument.model_validate(data)
            if doc.document_id in seen_doc_ids:
                raise ValueError(f"Duplicate document_id '{doc.document_id}' at line {line_num}")
            if doc.source_uri in seen_uris:
                raise ValueError(f"Duplicate source_uri '{doc.source_uri}' at line {line_num}")
            if doc.doc_type != "scientific_paper":
                raise ValueError(f"Invalid doc_type '{doc.doc_type}' at line {line_num}")

            seen_doc_ids.add(doc.document_id)
            seen_uris.add(doc.source_uri)
            docs_count += 1

            chunks = chunker.chunk(doc)
            total_chunks += len(chunks)

    return docs_count, total_chunks


def load_all_corpus_documents(corpus_path: Path) -> list[RawDocument]:
    """Load and sort all corpus documents deterministically by document_id."""
    documents: list[RawDocument] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            documents.append(RawDocument.model_validate(data))
    documents.sort(key=lambda d: d.document_id)
    return documents


def verify_target_safety_invariants(env_vars: dict[str, str]) -> None:
    """Strictly assert that environment configuration points ONLY to isolated targets."""
    # 1. PostgreSQL checks
    pg_host = (env_vars.get("POSTGRES_HOST") or "localhost").lower()
    if pg_host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(
            f"Target PostgreSQL host '{pg_host}' is invalid. Must be loopback (localhost or 127.0.0.1)."
        )

    pg_port = int(env_vars.get("POSTGRES_PORT", 0))
    if pg_port in ENTERPRISE_PORTS or pg_port != REQUIRED_POSTGRES_PORT:
        raise ValueError(
            f"Target PostgreSQL port {pg_port} is invalid. Refusing connection to non-isolated port."
        )

    pg_db = env_vars.get("POSTGRES_DB", "")
    if pg_db == "kgr_db" or not pg_db:
        raise ValueError(f"Target PostgreSQL database '{pg_db}' is forbidden. Must be isolated db.")

    # 2. Neo4j checks
    neo4j_uri = env_vars.get("NEO4J_URI", "")
    parsed_neo = urlparse(neo4j_uri)
    neo_host = (parsed_neo.hostname or "").lower()
    if neo_host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(
            f"Target Neo4j host '{neo_host}' is invalid. Must be loopback (localhost or 127.0.0.1)."
        )

    if parsed_neo.port in ENTERPRISE_PORTS or parsed_neo.port != REQUIRED_NEO4J_BOLT_PORT:
        raise ValueError(
            f"Target Neo4j URI '{neo4j_uri}' is invalid. Refusing connection to non-isolated target."
        )

    # 3. Paid embeddings check
    allow_paid = env_vars.get("ALLOW_PAID_EMBEDDINGS", "").strip().lower()
    if allow_paid != "false":
        raise ValueError(f"ALLOW_PAID_EMBEDDINGS is '{allow_paid}'. Paid embeddings must be false.")


async def execute_batch_ingestion(
    env_vars: dict[str, str],
    corpus_path: Path,
    batch_size: int = 20,
) -> int:
    """Execute live batch ingestion in bounded batches with post-batch verification."""
    enforce_strict_isolation_guards(env_vars)

    # Verify model cache locally
    model_name = env_vars.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    logger.info("Loading cached local embedding model '%s' (local_files_only=True)...", model_name)
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        logger.error("Failed to load local model '%s': %s", model_name, exc)
        return 1

    # Connect to isolated stores
    pg_settings = Settings(
        postgres_host=env_vars["POSTGRES_HOST"],
        postgres_port=int(env_vars["POSTGRES_PORT"]),
        postgres_db=env_vars["POSTGRES_DB"],
        postgres_user=env_vars["POSTGRES_USER"],
        postgres_password=env_vars["POSTGRES_PASSWORD"],
        embedding_dimensions=384,
        vector_top_k=5,
        vector_similarity_threshold=0.2,
    )
    pg_client = PostgresClient(settings=pg_settings)
    await pg_client.get_pool()

    neo_writer = IsolatedScientificNeo4jWriter(
        uri=env_vars["NEO4J_URI"],
        user=env_vars["NEO4J_USER"],
        password=env_vars["NEO4J_PASSWORD"],
    )
    neo_writer.verify_connectivity()

    try:
        documents = load_all_corpus_documents(corpus_path)
        total_docs = len(documents)
        logger.info("Loaded %d documents from corpus for bounded batch ingestion.", total_docs)

        coordinator = ScientificBatchCoordinator(
            pg_client=pg_client,
            neo_writer=neo_writer,
            embedding_model=cast(Any, model),
            batch_doc_limit=batch_size,
        )

        num_batches = (total_docs + batch_size - 1) // batch_size
        print(
            f"\n[EXECUTION PLAN] Total Documents: {total_docs} across {num_batches} batches of up to {batch_size} documents."
        )

        overall_completed = 0
        overall_failed = 0
        overall_needs_reconciliation = 0
        per_batch_summaries: list[dict[str, Any]] = []

        for b_idx in range(num_batches):
            batch_start = b_idx * batch_size
            batch_end = min(batch_start + batch_size, total_docs)
            batch_docs = documents[batch_start:batch_end]

            print(
                f"\n>>> Running Batch {b_idx + 1}/{num_batches} (Docs {batch_start + 1} to {batch_end}) <<<"
            )
            batch_report = await coordinator.ingest_batch(batch_docs)

            overall_completed += batch_report.completed
            overall_failed += batch_report.failed
            overall_needs_reconciliation += batch_report.needs_reconciliation

            # Verify stores after each batch
            async with pg_client.connection() as conn:
                pg_d_cnt = await conn.fetchval("SELECT count(*) FROM documents;")
                pg_c_cnt = await conn.fetchval("SELECT count(*) FROM chunks;")
                pg_e_cnt = await conn.fetchval("SELECT count(*) FROM entity_links;")

            with neo_writer._driver.session() as s:
                rec_n = s.run("MATCH (n) RETURN count(n) AS cnt").single()
                rec_r = s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
                neo_n_cnt = rec_n["cnt"] if rec_n else 0
                neo_r_cnt = rec_r["cnt"] if rec_r else 0

            batch_summary = {
                "batch_index": b_idx + 1,
                "batch_docs_count": len(batch_docs),
                "completed": batch_report.completed,
                "failed": batch_report.failed,
                "needs_reconciliation": batch_report.needs_reconciliation,
                "pg_documents_cumulative": pg_d_cnt,
                "pg_chunks_cumulative": pg_c_cnt,
                "pg_entity_links_cumulative": pg_e_cnt,
                "neo4j_nodes_cumulative": neo_n_cnt,
                "neo4j_relationships_cumulative": neo_r_cnt,
            }
            per_batch_summaries.append(batch_summary)

            print(
                f"Batch {b_idx + 1} Result: {batch_report.completed} complete, {batch_report.failed} failed, {batch_report.needs_reconciliation} needs reconciliation."
            )
            print(
                f"Cumulative Counts -> PG Docs: {pg_d_cnt}, Chunks: {pg_c_cnt}, Links: {pg_e_cnt} | Neo4j Nodes: {neo_n_cnt}, Rels: {neo_r_cnt}"
            )

            # Stop immediately on failure or partial write
            if batch_report.failed > 0 or batch_report.needs_reconciliation > 0:
                print(
                    f"\n[FATAL] Stopping ingestion due to unresolved failures in Batch {b_idx + 1}!"
                )
                for doc_id, d_rep in batch_report.document_reports.items():
                    if d_rep.state != DocumentState.COMPLETE:
                        err_str = (
                            (d_rep.error_message or "")
                            .encode("ascii", "backslashreplace")
                            .decode("ascii")
                        )
                        print(
                            f"  - Document '{doc_id}': state={d_rep.state}, stage={d_rep.failed_stage}, store={d_rep.affected_store}, error={err_str}"
                        )
                return 1

        print("\n=================================================================")
        print("Batch Ingestion Successfully Completed for All Documents")
        print("=================================================================")
        print(f"Total Documents Processed: {overall_completed}/{total_docs}")
        print(f"Total Failed:              {overall_failed}")
        print(f"Total Needs Reconciliation:{overall_needs_reconciliation}")
        return 0

    finally:
        await pg_client.close()
        neo_writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scientific-Paper Ingestion CLI and Offline Planning Tool."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV,
        help="Path to .env.scipaper configuration file",
    )
    parser.add_argument(
        "--corpus-file",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to corpus_scipaper.jsonl",
    )
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to manifest.jsonl",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run offline corpus, schema, and config validation (default mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan ingestion without writing to databases or generating embeddings",
    )
    parser.add_argument(
        "--execute-ingestion",
        action="store_true",
        help="Explicit flag required to perform actual live ingestion",
    )

    args = parser.parse_args()

    # Default to validate-only if neither dry-run nor execute-ingestion is passed
    is_dry_run = args.dry_run
    is_execute = args.execute_ingestion
    is_validate_only = args.validate_only or (not is_dry_run and not is_execute)

    print("=================================================================")
    print("Scientific-Paper Ingestion Pipeline Adapter")
    print("=================================================================")
    print(f"Env File:      {args.env_file}")
    print(f"Corpus:        {args.corpus_file}")
    if is_validate_only:
        print("Mode:          VALIDATE-ONLY (Zero connections, zero writes)")
    elif is_dry_run:
        print("Mode:          DRY-RUN (Planning only, zero database writes)")
    else:
        print("Mode:          LIVE INGESTION (Explicit execution requested)")
    print("-----------------------------------------------------------------")

    # 1. Parse and validate environment isolated from root .env
    if not args.env_file.exists():
        print(f"[ERROR] Environment file does not exist: {args.env_file}")
        sys.exit(1)

    env_vars = parse_env_file_isolated(args.env_file)
    errors = validate_environment_dict(env_vars, allow_placeholders=False)
    if errors:
        print(f"[ERROR] Environment validation failed ({len(errors)} errors):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # 2. Strict target safety invariants check
    try:
        verify_target_safety_invariants(env_vars)
    except ValueError as exc:
        print(f"[SECURITY BLOCKED] {exc}")
        sys.exit(1)

    # 3. Offline Corpus & Manifest Validation
    try:
        docs_count, chunks_count = validate_corpus_offline(args.corpus_file, args.manifest_file)
        print(f"[PASS] Offline corpus verified: {docs_count} documents, {chunks_count} chunks.")
    except Exception as exc:
        print(f"[ERROR] Offline corpus validation failed: {exc}")
        sys.exit(1)

    if is_validate_only:
        print("\n[SUCCESS] All offline validation checks passed.")
        print("Target Infrastructure Configuration:")
        print(
            f"  - PostgreSQL Target: {env_vars.get('POSTGRES_HOST')}:{env_vars.get('POSTGRES_PORT')}/{env_vars.get('POSTGRES_DB')}"
        )
        print(f"  - Neo4j Bolt Target: {env_vars.get('NEO4J_URI')}")
        print(
            f"  - Embedding Provider: {env_vars.get('EMBEDDING_PROVIDER')} ({env_vars.get('EMBEDDING_MODEL')}, dim={env_vars.get('EMBEDDING_DIMENSIONS')})"
        )
        print(f"  - Paid Embeddings: {env_vars.get('ALLOW_PAID_EMBEDDINGS')}")
        print("No database connections opened. No documents ingested.")
        sys.exit(0)

    if is_dry_run:
        print("\n[DRY RUN PLAN]")
        print(f"  - Documents to ingest: {docs_count}")
        print(f"  - Total chunks to generate: {chunks_count}")
        print(
            f"  - Destination PostgreSQL DB: {env_vars.get('POSTGRES_DB')} on port {env_vars.get('POSTGRES_PORT')}"
        )
        print(f"  - Destination Neo4j: {env_vars.get('NEO4J_URI')}")
        print("Dry run completed. Zero writes performed. Zero embeddings generated.")
        sys.exit(0)

    if is_execute:
        exit_code = asyncio.run(
            execute_batch_ingestion(
                env_vars=env_vars,
                corpus_path=args.corpus_file,
                batch_size=20,
            )
        )
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
