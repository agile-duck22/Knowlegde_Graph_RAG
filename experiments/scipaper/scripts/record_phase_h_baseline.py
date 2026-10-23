"""Record Phase H initial baseline counts."""

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
OUT_PATH = BASE_DIR / "eval" / "results" / "phase_h_initial_baseline.json"


async def record_baseline() -> None:
    env_vars = parse_env_file_isolated(ENV_PATH)
    pg_conn = await asyncpg.connect(
        host=env_vars["POSTGRES_HOST"],
        port=int(env_vars["POSTGRES_PORT"]),
        database=env_vars["POSTGRES_DB"],
        user=env_vars["POSTGRES_USER"],
        password=env_vars["POSTGRES_PASSWORD"],
    )
    pg_docs = await pg_conn.fetchval("SELECT COUNT(*) FROM documents;")
    pg_chunks = await pg_conn.fetchval("SELECT COUNT(*) FROM chunks;")
    pg_elinks = await pg_conn.fetchval("SELECT COUNT(*) FROM entity_links;")
    pg_emb384 = await pg_conn.fetchval(
        "SELECT COUNT(*) FROM chunks WHERE embedding_384 IS NOT NULL;"
    )
    pg_emb1536 = await pg_conn.fetchval("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL;")
    await pg_conn.close()

    driver = GraphDatabase.driver(
        env_vars["NEO4J_URI"], auth=(env_vars["NEO4J_USER"], env_vars["NEO4J_PASSWORD"])
    )
    with driver.session() as session:
        node_counts = {
            ":".join(sorted(r["lbls"])): r["cnt"]
            for r in session.run("MATCH (n) RETURN labels(n) AS lbls, count(n) AS cnt")
        }
        rel_counts = {
            r["t"]: r["cnt"]
            for r in session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS cnt")
        }
        rec_n = session.run("MATCH (n) RETURN count(n) AS cnt").single()
        rec_r = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
        total_nodes = rec_n["cnt"] if rec_n else 0
        total_rels = rec_r["cnt"] if rec_r else 0
    driver.close()

    baseline = {
        "timestamp": datetime.now(UTC).isoformat(),
        "phase": "Phase H Initial Baseline",
        "postgresql": {
            "documents": pg_docs,
            "chunks": pg_chunks,
            "entity_links": pg_elinks,
            "embedding_384_count": pg_emb384,
            "embedding_1536_count": pg_emb1536,
        },
        "neo4j": {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "nodes_by_label": node_counts,
            "relationships_by_type": rel_counts,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Recorded baseline to: {OUT_PATH}")
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    asyncio.run(record_baseline())
