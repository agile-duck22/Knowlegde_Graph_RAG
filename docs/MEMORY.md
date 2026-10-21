# Persistent Agent Memory & Session State

> **Last Updated**: 2026-10-09  
> **Current Project Status**: Phase 1 — Ingestion Pipeline (TASK-103 complete)

---

## 1. Confirmed Architecture Decisions

- **Storage Architecture**: Neo4j 5.x for Knowledge Graph relationship traversal; PostgreSQL 16 + `pgvector` for raw text chunks and dense vector search.
- **Dual-Store Invariant**: Canonical UUID `chunk_id` is shared between PostgreSQL `chunks.id` and Neo4j provenance properties (`source_chunk_ids`).
- **Query Safety**: Zero execution of dynamic LLM-generated raw Cypher. All graph traversals use pre-audited, parameterized Cypher templates with traversal depth limits ($D \le 3$).
- **Groundedness**: All synthesis responses use Anthropic Claude with strict bracketed citations validated against retrieved candidate `chunk_id`s before returning to the user.
- **Test Strategy**: Offline unit tests with mocks by default; opt-in integration tests for live database containers.

---

## 2. Active Invariants & Rules

1. **No Code Without Tests**: Every module must have corresponding unit tests.
2. **No Hallucinated Test Status**: Never report a task as completed without executing the tests and inspecting real output.
3. **No Dynamic Cypher**: Never construct Cypher queries using Python string formatting (`f"..."` or `+`).
4. **Idempotency**: All database write operations must use `MERGE` or `ON CONFLICT DO UPDATE`.

---

## 3. Implementation Status Summary

- **Phase 0 (Scaffolding)**: `DONE` (`TASK-001`–`TASK-004` all verified & passing)
- **Phase 1 (Ingestion & Graph)**: `DONE` (`TASK-101`–`TASK-107` all verified & passing)
- **Phase 2 (pgvector & Vector Search)**: `DONE` (`TASK-201`–`TASK-205` all verified & passing)
- **Phase 3 (Routing & Cypher Templates)**: `DONE` (`TASK-301` ✅, `TASK-302` ✅, `TASK-303` ✅, `TASK-304` ✅, `TASK-305` ✅)
- **Phase 4 (Synthesis & Citations)**: `DONE` (`TASK-401` ✅, `TASK-402` ✅, `TASK-403` ✅, `TASK-404` ✅, `TASK-405` ✅, Multi-Provider LLM Adapter ✅)
- **Phase 5 (Evaluation & Benchmarks)**: `IN PROGRESS` (`TASK-501` ✅, `TASK-502` ✅, `TASK-504` ✅, `TASK-503` & `TASK-505` `TODO`)

---

## 4. Current Blockers & Open Questions

- **Corpus Selection**: Resolved. Synthetic enterprise technology corpus generated (145 interconnected documents at `data/raw/corpus.jsonl`).
- **Embedding Model & Zero-Cost Guard**: Added zero-cost local embedding provider (`sentence_transformers`, `huggingface`, `local`) supporting models like `all-MiniLM-L6-v2` (384 dimensions) alongside OpenAI `text-embedding-3-small` (1536 dim). Schema updated with non-destructive dual-dimension column strategy: `chunks.embedding` (1536-dim) is preserved, and migration `002_add_embedding_384.sql` defines `chunks.embedding_384 VECTOR(384)` with HNSW index `idx_chunks_embedding_384_hnsw`. Writers use `COALESCE` on conflict to prevent nulling out either dimension, and vector retrieval dynamically queries the column matching the query vector dimension.
- **LLM Provider Configuration**: Complete. Per-component provider and model settings active (Gemini `gemini-3.1-flash-lite` for extraction/synthesis, Groq `openai/gpt-oss-20b` for routing).
- **TASK-503 Prerequisites**: Requires applying migration `002` to PostgreSQL, populating Neo4j and PostgreSQL databases, and configuring active API keys in `.env` before running live benchmarks.

---

## 5. Last Completed Task & Next Recommended Action

- **Last Completed Task**: Dual-Dimension (384 & 1536) Local Vector Support. Features:
  - Created migration script [`src/db/migrations/002_add_embedding_384.sql`](file:///c:/Users/workp/OneDrive/Desktop/kgr/src/db/migrations/002_add_embedding_384.sql) for `chunks.embedding_384 VECTOR(384)` and HNSW index.
  - Updated schema generator and health checks in [`src/db/postgres.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/src/db/postgres.py).
  - Updated [`src/db/postgres_writer.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/src/db/postgres_writer.py) to write to the appropriate vector column by dimension and preserve existing vectors via `COALESCE`.
  - Updated [`src/retrieval/vector_retriever.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/src/retrieval/vector_retriever.py) to dynamically select `c.embedding_384` or `c.embedding`.
  - Configured [`.env`](file:///c:/Users/workp/OneDrive/Desktop/kgr/.env) for local `sentence_transformers` (`all-MiniLM-L6-v2`, 384 dim, zero cost).
  - Updated tests and documentation (`docs/DATA_MODEL.md`).
- **Next Recommended Action**: Apply migration 002 to live PostgreSQL when ready to execute ingestion and benchmarks.

