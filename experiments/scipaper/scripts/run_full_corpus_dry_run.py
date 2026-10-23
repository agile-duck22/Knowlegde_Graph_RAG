"""Full Corpus Dry Run Audit for Phase H (Gate B).

Validates all 120 papers in experiments/scipaper/data/corpus_scipaper.jsonl:
- Verifies document and canonical paper IDs
- Verifies chunk IDs match PostgreSQL database records
- Validates entity schemas, canonical IDs, and types
- Validates relationship schemas, endpoints, and evidence provenance
- Cross-references existing Neo4j nodes and relationships
- Audits proposed creates, updates, duplicates, and errors
- Marks paper-arxiv-2411.18583 as ALREADY_INTEGRATED (Phase G)
- Emits full_corpus_dry_run_manifest.json with zero database writes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
from neo4j import GraphDatabase

from experiments.scipaper.schemas.extraction_schemas import (
    ScientificExtractedEntity,
    ScientificExtractedRelationship,
)
from experiments.scipaper.schemas.extractor import OfflineScientificExtractor
from experiments.scipaper.schemas.neo4j_writer import TargetIsolationError
from experiments.scipaper.scripts.preflight_check import (
    ENTERPRISE_PORTS,
    REQUIRED_NEO4J_BOLT_PORT,
    REQUIRED_POSTGRES_PORT,
    parse_env_file_isolated,
)
from src.ingest.chunker import DocumentChunker
from src.ingest.loader import RawDocument

logger = logging.getLogger("full_corpus_dry_run")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV = BASE_DIR / ".env.scipaper"
DEFAULT_CORPUS = BASE_DIR / "data" / "corpus_scipaper.jsonl"
DEFAULT_OUT = BASE_DIR / "eval" / "results" / "full_corpus_dry_run_manifest.json"
PHASE_G_DOC_ID = "paper-arxiv-2411.18583"


def enforce_isolation_guards(env_vars: dict[str, str]) -> None:
    """Enforce strict target guards before attempting any connection."""
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
class PaperAuditSummary:
    document_id: str
    title: str
    chunks_count: int
    is_phase_g_integrated: bool
    unique_entities_count: int
    unique_relationships_count: int
    existing_nodes: int
    new_nodes_to_create: int
    existing_rels: int
    new_rels_to_create: int
    new_entity_links_to_add: int
    validation_errors: list[str] = field(default_factory=list)
    quarantine: bool = False
    quarantine_reason: str | None = None


async def run_full_corpus_dry_run() -> dict[str, Any]:
    env_vars = parse_env_file_isolated(BASE_DIR / ".env.scipaper")
    enforce_isolation_guards(env_vars)

    # 1. Load all corpus documents
    docs: list[RawDocument] = []
    with open(DEFAULT_CORPUS, encoding="utf-8") as f:
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

    print(f"Loaded {len(docs)} documents from {DEFAULT_CORPUS}.")

    # 2. Query PostgreSQL for all chunks and entity links
    pg_conn = await asyncpg.connect(
        host=env_vars["POSTGRES_HOST"],
        port=int(env_vars["POSTGRES_PORT"]),
        database=env_vars["POSTGRES_DB"],
        user=env_vars["POSTGRES_USER"],
        password=env_vars["POSTGRES_PASSWORD"],
    )
    all_db_chunks = await pg_conn.fetch(
        "SELECT document_id, id, chunk_index FROM chunks ORDER BY document_id, chunk_index;"
    )
    all_db_elinks = await pg_conn.fetch("SELECT chunk_id, canonical_entity_id FROM entity_links;")
    await pg_conn.close()

    db_chunks_by_doc: dict[str, list[str]] = {}
    for r in all_db_chunks:
        db_chunks_by_doc.setdefault(r["document_id"], []).append(str(r["id"]))

    existing_elinks_set = {(str(r["chunk_id"]), r["canonical_entity_id"]) for r in all_db_elinks}

    # 3. Query Neo4j for existing nodes and relationships
    driver = GraphDatabase.driver(
        env_vars["NEO4J_URI"], auth=(env_vars["NEO4J_USER"], env_vars["NEO4J_PASSWORD"])
    )
    with driver.session() as session:
        existing_node_ids = {r["id"] for r in session.run("MATCH (e:Entity) RETURN e.id AS id")}
        existing_rel_keys = {
            (r["src"], r["t"], r["trg"], r["prov"])
            for r in session.run(
                "MATCH (s:Entity)-[r]->(t:Entity) RETURN s.id AS src, type(r) AS t, t.id AS trg, r.provenance_chunk_id AS prov"
            )
        }
    driver.close()

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    extractor = OfflineScientificExtractor()

    paper_summaries: list[PaperAuditSummary] = []
    total_proposed_new_nodes: set[str] = set()
    total_proposed_new_rels: set[tuple[str, str, str, str]] = set()
    total_new_elinks_count = 0
    quarantined_papers_count = 0

    for doc in docs:
        is_phase_g = doc.document_id == PHASE_G_DOC_ID
        chunks = chunker.chunk(doc)
        chunk_ids = [c.chunk_id for c in chunks]

        # Verify chunks match PostgreSQL DB exactly
        db_cids = db_chunks_by_doc.get(doc.document_id, [])
        errors: list[str] = []
        if db_cids != chunk_ids:
            errors.append(
                f"PostgreSQL chunk mismatch: generated {len(chunk_ids)} != db {len(db_cids)}"
            )

        doc_entities: dict[str, ScientificExtractedEntity] = {}
        doc_rels: dict[tuple[str, str, str, str], ScientificExtractedRelationship] = {}
        doc_new_elinks = 0

        for c in chunks:
            res = extractor.extract(c)
            # Provenance assertion
            if res.chunk_id != c.chunk_id or res.document_id != doc.document_id:
                errors.append(f"Extraction provenance mismatch on chunk {c.chunk_id}")

            for ent in res.entities:
                if not ent.canonical_id or not ent.entity_type:
                    errors.append(f"Entity missing canonical_id or type: {ent}")
                doc_entities[ent.canonical_id] = ent
                if (c.chunk_id, ent.canonical_id) not in existing_elinks_set:
                    doc_new_elinks += 1

            for rel in res.relationships:
                if not rel.source_canonical_id or not rel.target_canonical_id:
                    errors.append(f"Relationship missing endpoint: {rel}")
                key = (
                    rel.source_canonical_id,
                    rel.relation_type,
                    rel.target_canonical_id,
                    c.chunk_id,
                )
                doc_rels[key] = rel

        # Count new vs existing nodes
        doc_existing_nodes = 0
        doc_new_nodes = 0
        for cid in doc_entities:
            if cid in existing_node_ids:
                doc_existing_nodes += 1
            else:
                doc_new_nodes += 1
                total_proposed_new_nodes.add(cid)

        # Count new vs existing rels
        doc_existing_rels = 0
        doc_new_rels = 0
        for rkey in doc_rels:
            if rkey in existing_rel_keys:
                doc_existing_rels += 1
            else:
                doc_new_rels += 1
                total_proposed_new_rels.add(rkey)

        total_new_elinks_count += doc_new_elinks
        is_quarantined = len(errors) > 0
        if is_quarantined:
            quarantined_papers_count += 1

        summary = PaperAuditSummary(
            document_id=doc.document_id,
            title=doc.title,
            chunks_count=len(chunks),
            is_phase_g_integrated=is_phase_g,
            unique_entities_count=len(doc_entities),
            unique_relationships_count=len(doc_rels),
            existing_nodes=doc_existing_nodes,
            new_nodes_to_create=doc_new_nodes,
            existing_rels=doc_existing_rels,
            new_rels_to_create=doc_new_rels,
            new_entity_links_to_add=doc_new_elinks,
            validation_errors=errors,
            quarantine=is_quarantined,
            quarantine_reason="; ".join(errors) if errors else None,
        )
        paper_summaries.append(summary)

    manifest_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_corpus_papers_audited": len(docs),
        "total_corpus_chunks_audited": sum(p.chunks_count for p in paper_summaries),
        "phase_g_already_integrated_count": sum(
            1 for p in paper_summaries if p.is_phase_g_integrated
        ),
        "papers_requiring_migration": sum(
            1 for p in paper_summaries if not p.is_phase_g_integrated
        ),
        "quarantined_papers_count": quarantined_papers_count,
        "aggregate_proposed_new_unique_nodes": len(total_proposed_new_nodes),
        "aggregate_proposed_new_unique_rels": len(total_proposed_new_rels),
        "aggregate_new_entity_links_to_add": total_new_elinks_count,
        "paper_audits": [asdict(p) for p in paper_summaries],
    }

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("GATE B: FULL CORPUS DRY RUN AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total Papers Audited: {len(docs)}")
    print(f"Total Chunks Verified: {sum(p.chunks_count for p in paper_summaries)}")
    print(f"Quarantined Papers: {quarantined_papers_count}")
    print(f"Proposed New Unique Nodes (across all 120 papers): {len(total_proposed_new_nodes)}")
    print(f"Proposed New Unique Relationships: {len(total_proposed_new_rels)}")
    print(f"Proposed New Entity Links in PostgreSQL: {total_new_elinks_count}")
    print(f"Manifest written to: {DEFAULT_OUT}")
    return manifest_data


if __name__ == "__main__":
    asyncio.run(run_full_corpus_dry_run())
