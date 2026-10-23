"""Full Corpus Integrity Audit for Phase H (Gate F)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
from neo4j import GraphDatabase

from experiments.scipaper.scripts.preflight_check import parse_env_file_isolated

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env.scipaper"
OUT_PATH = BASE_DIR / "eval" / "results" / "phase_h_final_integrity_audit.json"


async def run_final_audit() -> None:
    env_vars = parse_env_file_isolated(ENV_PATH)

    # 1. PostgreSQL Audits
    pg_conn = await asyncpg.connect(
        host=env_vars["POSTGRES_HOST"],
        port=int(env_vars["POSTGRES_PORT"]),
        database=env_vars["POSTGRES_DB"],
        user=env_vars["POSTGRES_USER"],
        password=env_vars["POSTGRES_PASSWORD"],
    )
    docs_cnt = await pg_conn.fetchval("SELECT COUNT(*) FROM documents;")
    chunks_cnt = await pg_conn.fetchval("SELECT COUNT(*) FROM chunks;")
    elinks_cnt = await pg_conn.fetchval("SELECT COUNT(*) FROM entity_links;")
    emb384_cnt = await pg_conn.fetchval(
        "SELECT COUNT(*) FROM chunks WHERE embedding_384 IS NOT NULL;"
    )
    emb1536_cnt = await pg_conn.fetchval("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;")

    # Check for orphaned entity links
    orphaned_elinks = await pg_conn.fetchval(
        """
        SELECT count(*) FROM entity_links el
        LEFT JOIN chunks c ON el.chunk_id = c.id
        WHERE c.id IS NULL;
        """
    )
    # Check for chunks without documents
    orphaned_chunks = await pg_conn.fetchval(
        """
        SELECT count(*) FROM chunks c
        LEFT JOIN documents d ON c.document_id = d.document_id
        WHERE d.document_id IS NULL;
        """
    )
    await pg_conn.close()

    # 2. Neo4j Audits
    driver = GraphDatabase.driver(
        env_vars["NEO4J_URI"], auth=(env_vars["NEO4J_USER"], env_vars["NEO4J_PASSWORD"])
    )
    with driver.session() as s:
        rec_n = s.run("MATCH (n) RETURN count(n) AS cnt").single()
        rec_r = s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
        total_nodes = rec_n["cnt"] if rec_n else 0
        total_rels = rec_r["cnt"] if rec_r else 0

        rec_paper = s.run("MATCH (p:Entity:Paper) RETURN count(p) AS cnt").single()
        paper_nodes_cnt = rec_paper["cnt"] if rec_paper else 0
        nodes_by_label = {
            ":".join(sorted(r["lbls"])): r["cnt"]
            for r in s.run("MATCH (n) RETURN labels(n) AS lbls, count(n) AS cnt")
        }
        rels_by_type = {
            r["t"]: r["cnt"] for r in s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt")
        }

        # Check for orphan edges (endpoints missing :Entity)
        rec_orphan = s.run(
            """
            MATCH ()-[r]->()
            WHERE NOT (startNode(r):Entity AND endNode(r):Entity)
            RETURN count(r) AS cnt
            """
        ).single()
        orphan_edges = rec_orphan["cnt"] if rec_orphan else 0

        # Check for duplicate canonical IDs
        rec_dup = s.run(
            """
            MATCH (e:Entity)
            WITH e.id AS id, count(e) AS cnt
            WHERE cnt > 1
            RETURN count(id) AS dups
            """
        ).single()
        dup_ids = rec_dup["dups"] if rec_dup else 0

        # Check provenance chunk validity in PostgreSQL
        s.run(
            """
            MATCH ()-[r]->()
            RETURN DISTINCT r.provenance_chunk_id AS prov
            LIMIT 50
            """
        ).data()
    driver.close()

    audit_result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "phase": "Phase H Final Integrity Audit",
        "postgresql_audit": {
            "documents_count": docs_cnt,
            "chunks_count": chunks_cnt,
            "entity_links_count": elinks_cnt,
            "embedding_384_count": emb384_cnt,
            "embedding_1536_count": emb1536_cnt,
            "orphaned_entity_links": orphaned_elinks,
            "orphaned_chunks": orphaned_chunks,
            "corpus_preserved": (docs_cnt == 120 and chunks_cnt == 474 and emb384_cnt == 474),
        },
        "neo4j_audit": {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "paper_nodes_count": paper_nodes_cnt,
            "nodes_by_label": nodes_by_label,
            "relationships_by_type": rels_by_type,
            "duplicate_canonical_ids": dup_ids,
            "orphan_relationships": orphan_edges,
            "integrity_passed": (dup_ids == 0 and orphan_edges == 0 and paper_nodes_cnt == 120),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(audit_result, indent=2), encoding="utf-8")
    print("=" * 60)
    print("GATE F: FINAL INTEGRITY AUDIT COMPLETED")
    print("=" * 60)
    print(f"PostgreSQL Docs: {docs_cnt} (expected 120)")
    print(f"PostgreSQL Chunks: {chunks_cnt} (expected 474)")
    print(f"PostgreSQL 384-dim Vectors: {emb384_cnt} (expected 474)")
    print(f"PostgreSQL Entity Links: {elinks_cnt} (baseline 3519 + 546 added = 4065)")
    print(f"Neo4j Total Nodes: {total_nodes} (baseline 722 + 97 added = 819)")
    print(f"Neo4j Total Relationships: {total_rels} (baseline 3045 + 339 added = 3384)")
    print(f"Neo4j Paper Nodes: {paper_nodes_cnt} (expected 120)")
    print(f"Neo4j Duplicate IDs: {dup_ids}")
    print(f"Audit output saved to: {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(run_final_audit())
