"""Controlled Single-Document End-to-End Live Integration Test for Phase C3A.

Strict Constraints:
- Authorizes exactly ONE paper: paper-arxiv-2411.18583 (the first deterministically sorted record).
- Fails closed if more than one document is processed.
- Connects ONLY to isolated targets:
    PostgreSQL: 127.0.0.1:5433 / scipaper_db
    Neo4j: bolt://127.0.0.1:7688
- Uses local cached embedding model 'all-MiniLM-L6-v2' (dim=384, local_files_only=True).
- Strictly zero paid/remote API calls (ALLOW_PAID_EMBEDDINGS=false).
- Idempotency check: ingests the same document twice and asserts zero duplicates across both stores.
- Executes vector retrieval against isolated PostgreSQL and verifies chunk retrieval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from sentence_transformers import SentenceTransformer

from experiments.scipaper.schemas.extraction_schemas import (
    ScientificChunkExtractionResult,
)
from experiments.scipaper.schemas.extractor import MockScientificExtractor
from experiments.scipaper.schemas.neo4j_writer import (
    IsolatedScientificNeo4jWriter,
    TargetIsolationError,
)
from experiments.scipaper.schemas.resolver import ScientificEntityResolver
from experiments.scipaper.scripts.preflight_check import (
    ENTERPRISE_PORTS,
    REQUIRED_NEO4J_BOLT_PORT,
    REQUIRED_POSTGRES_PORT,
    parse_env_file_isolated,
    validate_environment_dict,
)
from src.config import Settings
from src.db.postgres import PostgresClient
from src.db.postgres_writer import ChunkWriteInput, PostgresWriter
from src.ingest.chunker import DocumentChunker, TextChunk
from src.ingest.loader import RawDocument
from src.retrieval.vector_retriever import VectorRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("single_doc_live_test")

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env.scipaper"
CORPUS_PATH = BASE_DIR / "data" / "corpus_scipaper.jsonl"
MANIFEST_PATH = BASE_DIR / "data" / "manifest.jsonl"
TARGET_DOC_ID = "paper-arxiv-2411.18583"


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


def load_selected_document(corpus_path: Path, target_id: str) -> RawDocument:
    """Load exactly one document by ID from corpus."""
    found: RawDocument | None = None
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line.strip())
            if data.get("document_id") == target_id:
                found = RawDocument.model_validate(data)
                break

    if not found:
        raise ValueError(f"Target document '{target_id}' not found in corpus.")
    return found


def embed_chunks_locally(
    chunks: list[TextChunk],
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[SentenceTransformer, list[list[float]]]:
    """Generate 384d embeddings using local cached model only."""
    logger.info("Loading cached embedding model '%s' (local_files_only=True)...", model_name)
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Model '{model_name}' is not cached locally: {exc}. Aborting per safety rules."
        ) from exc

    texts = [c.content for c in chunks]
    embeddings_np = model.encode(texts, convert_to_numpy=True)

    vectors: list[list[float]] = []
    for i, emb in enumerate(embeddings_np):
        vec = [float(x) for x in emb]
        if len(vec) != 384:
            raise ValueError(f"Chunk {i} embedding dimension is {len(vec)}, expected 384")
        for val in vec:
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Chunk {i} embedding contains non-finite values")
        vectors.append(vec)

    logger.info("Generated %d embeddings with dimension 384.", len(vectors))
    return model, vectors


async def get_pg_counts(client: PostgresClient) -> dict[str, int]:
    """Query current table row counts in PostgreSQL."""
    async with client.connection() as conn:
        doc_count = await conn.fetchval("SELECT count(*) FROM documents;")
        chunk_count = await conn.fetchval("SELECT count(*) FROM chunks;")
        link_count = await conn.fetchval("SELECT count(*) FROM entity_links;")
        target_doc_exists = await conn.fetchval(
            "SELECT count(*) FROM documents WHERE document_id = $1;", TARGET_DOC_ID
        )
    return {
        "documents": int(doc_count),
        "chunks": int(chunk_count),
        "entity_links": int(link_count),
        "target_doc_exists": int(target_doc_exists),
    }


def get_neo4j_counts(neo_writer: IsolatedScientificNeo4jWriter) -> dict[str, int]:
    """Query current node and relationship counts in Neo4j."""
    with neo_writer._driver.session() as session:
        rec_nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()
        rec_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
        rec_target = session.run(
            "MATCH (e:Entity {id: $id}) RETURN count(e) AS cnt",
            {"id": f"paper:{TARGET_DOC_ID}"},
        ).single()
        node_count = rec_nodes["cnt"] if rec_nodes else 0
        rel_count = rec_rels["cnt"] if rec_rels else 0
        target_doc_nodes = rec_target["cnt"] if rec_target else 0
    return {
        "nodes": int(node_count),
        "relationships": int(rel_count),
        "target_doc_nodes": int(target_doc_nodes),
    }


async def run_single_document_ingestion(
    doc: RawDocument,
    pg_client: PostgresClient,
    neo_writer: IsolatedScientificNeo4jWriter,
    model: SentenceTransformer,
    vectors: list[list[float]],
    chunks: list[TextChunk],
    extraction_results: list[ScientificChunkExtractionResult],
) -> dict[str, Any]:
    """Execute live ingestion of the single document into PostgreSQL and Neo4j."""
    pg_writer = PostgresWriter(pg_client)

    # 1. Write Document to PostgreSQL
    logger.info("Writing document '%s' to PostgreSQL...", doc.document_id)
    doc_res = await pg_writer.upsert_document(doc)

    # 2. Prepare and write Chunks to PostgreSQL (with embedding_384)
    chunk_inputs = [
        ChunkWriteInput(chunk=c, embedding=v) for c, v in zip(chunks, vectors, strict=True)
    ]
    logger.info("Writing %d chunks to PostgreSQL...", len(chunk_inputs))
    chunks_upserted = await pg_writer.upsert_chunks_batch(chunk_inputs)

    # 3. Write Entity Links to PostgreSQL
    total_links_written = 0
    for res in extraction_results:
        if res.entities:
            # PostgresWriter expects ExtractedEntity protocol (requires id, canonical_id, entity_type, name, confidence)
            links_written = await pg_writer.write_entity_links(
                res.chunk_id, cast(Any, res.entities)
            )
            total_links_written += links_written
    logger.info("Wrote %d entity links to PostgreSQL.", total_links_written)

    # 4. Write Entities and Relationships to Neo4j
    total_entities_neo = 0
    total_rels_neo = 0
    logger.info("Writing entities and relationships to Neo4j...")
    for res in extraction_results:
        e_cnt, r_cnt = neo_writer.write_chunk_result(res)
        total_entities_neo += e_cnt
        total_rels_neo += r_cnt

    logger.info(
        "Neo4j write complete: %d entity items processed, %d relationship items processed.",
        total_entities_neo,
        total_rels_neo,
    )

    return {
        "doc_id": doc_res,
        "chunks_upserted": chunks_upserted,
        "entity_links_written": total_links_written,
        "neo_entities_processed": total_entities_neo,
        "neo_rels_processed": total_rels_neo,
    }


async def read_back_verification(
    doc: RawDocument,
    chunks: list[TextChunk],
    pg_client: PostgresClient,
    neo_writer: IsolatedScientificNeo4jWriter,
) -> dict[str, Any]:
    """Read back and verify document, chunks, vectors, entity links, nodes, and edges."""
    logger.info("Reading back data from isolated stores for verification...")

    # Read back from PostgreSQL
    async with pg_client.connection() as conn:
        doc_row = await conn.fetchrow(
            "SELECT document_id, title, doc_type, content_hash FROM documents WHERE document_id = $1;",
            doc.document_id,
        )
        chunk_rows = await conn.fetch(
            "SELECT id, chunk_index, token_count, embedding_384 IS NOT NULL AS has_emb_384, (embedding_384 IS NOT NULL) AND (embedding IS NULL) AS isolated_384 "
            "FROM chunks WHERE document_id = $1 ORDER BY chunk_index ASC;",
            doc.document_id,
        )
        link_rows = await conn.fetch(
            "SELECT el.chunk_id, el.canonical_entity_id, el.entity_type, el.confidence "
            "FROM entity_links el JOIN chunks c ON el.chunk_id = c.id "
            "WHERE c.document_id = $1;",
            doc.document_id,
        )

    # Read back from Neo4j
    with neo_writer._driver.session() as session:
        node_records = session.run(
            "MATCH (e:Entity) RETURN e.id AS id, labels(e) AS labels, e.name AS name"
        ).data()
        rel_records = session.run(
            "MATCH (s:Entity)-[r]->(t:Entity) "
            "RETURN s.id AS source, type(r) AS rel_type, t.id AS target, r.provenance_chunk_id AS prov_chunk"
        ).data()

    return {
        "pg_document": dict(doc_row) if doc_row else None,
        "pg_chunk_count": len(chunk_rows),
        "pg_chunks": [dict(r) for r in chunk_rows],
        "pg_link_count": len(link_rows),
        "neo_node_count": len(node_records),
        "neo_nodes": node_records,
        "neo_rel_count": len(rel_records),
        "neo_rels": rel_records,
    }


async def test_vector_retrieval(
    model: SentenceTransformer,
    pg_client: PostgresClient,
    env_vars: dict[str, str],
) -> dict[str, Any]:
    """Execute a semantic vector search query using VectorRetriever against isolated PostgreSQL."""
    logger.info("Testing vector retrieval against isolated PostgreSQL...")
    settings = Settings(
        postgres_host=env_vars["POSTGRES_HOST"],
        postgres_port=int(env_vars["POSTGRES_PORT"]),
        postgres_db=env_vars["POSTGRES_DB"],
        postgres_user=env_vars["POSTGRES_USER"],
        postgres_password=env_vars["POSTGRES_PASSWORD"],
        embedding_dimensions=384,
        vector_top_k=5,
        vector_similarity_threshold=0.2,
    )

    retriever = VectorRetriever(client=pg_client, settings=settings)

    query_text = "automated literature review and paper analysis"
    query_emb_np = model.encode([query_text], convert_to_numpy=True)[0]
    query_vector = [float(x) for x in query_emb_np]

    results = await retriever.search(query_vector=query_vector, top_k=3)

    return {
        "query": query_text,
        "results_count": len(results),
        "top_result": {
            "chunk_id": results[0].chunk_id,
            "document_id": results[0].document_id,
            "similarity_score": results[0].similarity_score,
            "snippet": results[0].content[:150] + "...",
        }
        if results
        else None,
    }


async def main_async() -> int:
    print("=================================================================")
    print("Phase C3A — One-Document Live Integration Test")
    print("=================================================================")

    # 1. Load and validate environment
    if not ENV_PATH.exists():
        print(f"[FATAL] Environment file missing: {ENV_PATH}")
        return 1

    env_vars = parse_env_file_isolated(ENV_PATH)
    errs = validate_environment_dict(env_vars, allow_placeholders=False)
    if errs:
        print(f"[FATAL] Environment validation failed: {errs}")
        return 1

    enforce_isolation_guards(env_vars)
    print("[PASS] Target isolation guards verified.")
    print(
        f"  PostgreSQL: {env_vars['POSTGRES_HOST']}:{env_vars['POSTGRES_PORT']}/{env_vars['POSTGRES_DB']}"
    )
    print(f"  Neo4j Bolt: {env_vars['NEO4J_URI']}")
    print(
        f"  Embeddings: {env_vars['EMBEDDING_PROVIDER']} (dim={env_vars['EMBEDDING_DIMENSIONS']})"
    )

    # 2. Select exactly ONE document deterministically
    doc = load_selected_document(CORPUS_PATH, TARGET_DOC_ID)
    print(f"[PASS] Selected document: {doc.document_id} ({doc.source_uri})")

    # 3. Setup clients
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
    print("[PASS] Verified connectivity to isolated PostgreSQL and Neo4j.")

    try:
        # 4. Check initial baseline state
        pg_base = await get_pg_counts(pg_client)
        neo_base = get_neo4j_counts(neo_writer)
        print(f"[BASELINE] PostgreSQL: {pg_base}")
        print(f"[BASELINE] Neo4j:      {neo_base}")

        if pg_base["target_doc_exists"] > 0 or neo_base["target_doc_nodes"] > 0:
            print(
                f"[COLLISION WARNING] Target document {TARGET_DOC_ID} already exists in database!"
            )
            print("Aborting to avoid ambiguous state.")
            return 1

        # 5. Document Chunking
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(doc)
        print(f"[PASS] Chunked document into {len(chunks)} chunks.")

        # 6. Extraction & Resolution
        resolver = ScientificEntityResolver()
        extractor = MockScientificExtractor(resolver=resolver)
        extraction_results: list[ScientificChunkExtractionResult] = []
        for c in chunks:
            res = extractor.extract(c)
            extraction_results.append(res)
        print(f"[PASS] Extracted entities/relationships for {len(extraction_results)} chunks.")

        # 7. Local Embedding
        model, vectors = embed_chunks_locally(chunks, env_vars["EMBEDDING_MODEL"])
        print(f"[PASS] Generated {len(vectors)} local embeddings (all 384d, finite).")

        # 8. First Ingestion Write
        print("\n--- Executing Ingestion Pass 1 ---")
        write_summary = await run_single_document_ingestion(
            doc=doc,
            pg_client=pg_client,
            neo_writer=neo_writer,
            model=model,
            vectors=vectors,
            chunks=chunks,
            extraction_results=extraction_results,
        )
        print(f"[PASS] Ingestion pass 1 completed: {write_summary}")

        # 9. Read-back verification
        read_back = await read_back_verification(doc, chunks, pg_client, neo_writer)
        print("\n--- Read-Back Verification ---")
        print(
            f"PG Document: {read_back['pg_document']['document_id'] if read_back['pg_document'] else 'MISSING'}"
        )
        print(f"PG Chunks: {read_back['pg_chunk_count']} chunks found")
        for ck in read_back["pg_chunks"]:
            print(
                f"  Chunk idx={ck['chunk_index']} uuid={ck['id']} has_384={ck['has_emb_384']} isolated_384={ck['isolated_384']}"
            )
        print(f"PG Entity Links: {read_back['pg_link_count']}")
        print(f"Neo4j Nodes: {read_back['neo_node_count']}")
        for n in read_back["neo_nodes"]:
            print(f"  Node: {n['id']} labels={n['labels']} name={n['name']}")
        print(f"Neo4j Relationships: {read_back['neo_rel_count']}")
        for r in read_back["neo_rels"]:
            print(
                f"  Rel: ({r['source']})-[:{r['rel_type']}]->({r['target']}) chunk={r['prov_chunk']}"
            )

        # 10. Vector Retrieval Test
        print("\n--- Vector Retrieval Test ---")
        retrieval_res = await test_vector_retrieval(model, pg_client, env_vars)
        print(f"Query: '{retrieval_res['query']}'")
        print(f"Total results: {retrieval_res['results_count']}")
        print(f"Top result: {retrieval_res['top_result']}")
        assert retrieval_res["results_count"] > 0, "Vector retrieval returned 0 results"
        assert retrieval_res["top_result"]["document_id"] == TARGET_DOC_ID, (
            "Top result did not match test document"
        )

        # 11. Idempotency Test (Ingestion Pass 2)
        print("\n--- Executing Idempotency Pass 2 (Exact Duplicate) ---")
        pass2_summary = await run_single_document_ingestion(
            doc=doc,
            pg_client=pg_client,
            neo_writer=neo_writer,
            model=model,
            vectors=vectors,
            chunks=chunks,
            extraction_results=extraction_results,
        )
        print(f"[PASS] Ingestion pass 2 completed: {pass2_summary}")

        # Post-pass-2 counts
        pg_post = await get_pg_counts(pg_client)
        neo_post = get_neo4j_counts(neo_writer)
        print("\n[IDEMPOTENCY CHECK]")
        print(f"PG Documents:      Pass 1={1}, Post Pass 2={pg_post['documents']}")
        print(f"PG Chunks:         Pass 1={len(chunks)}, Post Pass 2={pg_post['chunks']}")
        print(
            f"PG Entity Links:   Pass 1={read_back['pg_link_count']}, Post Pass 2={pg_post['entity_links']}"
        )
        print(
            f"Neo4j Nodes:       Pass 1={read_back['neo_node_count']}, Post Pass 2={neo_post['nodes']}"
        )
        print(
            f"Neo4j Rels:        Pass 1={read_back['neo_rel_count']}, Post Pass 2={neo_post['relationships']}"
        )

        assert pg_post["documents"] == 1, f"Expected 1 document, found {pg_post['documents']}"
        assert pg_post["chunks"] == len(chunks), (
            f"Expected {len(chunks)} chunks, found {pg_post['chunks']}"
        )
        assert pg_post["entity_links"] == read_back["pg_link_count"], (
            "Duplicate entity links detected!"
        )
        assert neo_post["nodes"] == read_back["neo_node_count"], "Duplicate Neo4j nodes detected!"
        assert neo_post["relationships"] == read_back["neo_rel_count"], (
            "Duplicate Neo4j relationships detected!"
        )

        print("\n[SUCCESS] Idempotency test passed with 0 duplicates and stable canonical IDs.")
        return 0

    finally:
        await pg_client.close()
        neo_writer.close()


def main() -> None:
    code = asyncio.run(main_async())
    sys.exit(code)


if __name__ == "__main__":
    main()
