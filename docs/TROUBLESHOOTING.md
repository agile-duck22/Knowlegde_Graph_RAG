# Troubleshooting Runbook & Diagnostics

This guide provides diagnostic steps, likely causes, and safe remediation runbooks for common operational failure modes.

---

## 1. Diagnostic Matrix

| Symptom | Diagnostic Step | Likely Root Cause | Remediation Runbook |
| :--- | :--- | :--- | :--- |
| **Neo4j Connection Refused** (`ServiceUnavailable`) | `docker compose ps` & `nc -zv localhost 7687` | Container not started or memory limit exceeded | Run `docker compose up -d neo4j`. Check container logs with `docker logs kgr-neo4j`. |
| **PostgreSQL Connection Failed** (`ConnectionRefusedError`) | Check port 5432 and `pg_isready` | PostgreSQL container down or password mismatch in `.env` | Verify `.env` matches `docker-compose.yml` credentials. |
| **Embedding Dimension Mismatch** (`dimensions do not match`) | Inspect PostgreSQL table schema vs model output dimension | Model produces 384 dimensions, but column defined as `VECTOR(1536)` | Update `EMBEDDING_DIMENSIONS` in `.env` or run migration to alter column dimension. |
| **Citation Validation Warnings** (`HALLUCINATED_CITATION`) | Inspect query response `warnings` and raw answer | Claude cited a `chunk_id` not present in retrieved context set | Review context prompt framing in `src/synthesis/synthesizer.py`. Citation guard automatically scrubs false citations. |
| **Slow Graph Queries ($> 2.0\text{s}$)** | Run `EXPLAIN` or `PROFILE` on Cypher query in Neo4j Browser | Missing index on `(e:Entity {id})` or unbounded path traversal | Verify uniqueness constraint exists. Ensure traversal specifies `*1..2` hop limits. |
| **LLM Rate Limits (`429 Too Many Requests`)** | Check Anthropic API response headers and logs | Ingestion batch size too high for current API tier | Reduce `EXTRACTION_BATCH_SIZE` in `.env` and enable exponential backoff. |
| **Inconsistent Provenance** (`Orphaned Chunk ID`) | Run `src/validation/provenance_check.py` | Ingestion aborted midway between Postgres and Neo4j writes | Re-run ingestion pipeline for failed document to trigger idempotent upsert reconciliation. |

---

## 2. Emergency Recovery Runbooks

### Runbook A: Full Database Reset (Development Only)
```bash
# WARNING: Deletes all local database volumes
docker compose down -v
docker compose up -d
python -m src.db.migrate
python -m src.ingest --corpus data/raw/corpus.jsonl
```

### Runbook B: Verify Database Health & Constraints
```bash
# Verify Neo4j constraints
python -c "from src.db.neo4j import verify_constraints; verify_constraints()"

# Verify PostgreSQL vector indexes
python -c "from src.db.postgres import verify_hnsw_indexes; verify_hnsw_indexes()"
```
