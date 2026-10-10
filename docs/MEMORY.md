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
- **Phase 4 (Synthesis & Citations)**: `TODO`
- **Phase 5 (Evaluation & Benchmarks)**: `TODO`

---

## 4. Current Blockers & Open Questions

- **Corpus Selection**: Resolved. Synthetic enterprise technology corpus generated (145 interconnected documents at `data/raw/corpus.jsonl`).
- **Embedding Model**: Default selected as `text-embedding-3-small` (1536 dim) / `all-MiniLM-L6-v2` (384 dim local fallback), pending confirmation. See [`docs/DECISIONS.md#ADR-003`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DECISIONS.md).

---

## 5. Last Completed Task & Next Recommended Action

- **Last Completed Task**: [`TASK-305`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md#phase-3-query-routing--safe-cypher-templates) — Implemented Router Engine (`src/routing/router.py`, `src/routing/__init__.py`). Features: `RouterEngine` coordinating `QueryIntentClassifier`, `EntityLinker`, `SafeGraphRetriever`, and `VectorRetriever`. Supports `GRAPH_ONLY`, `VECTOR_ONLY`, `HYBRID`, and `OUT_OF_SCOPE` query intents with robust fallback logic when entities cannot be resolved or linked. Typed data contracts (`RoutedQueryRequest`, `RoutedQueryResponse`, `GraphExecutionDetails`, `VectorExecutionDetails`, `RetrievalRoutingError`). Verified via 17 unit tests with mocks, `ruff` clean, `mypy` clean. Full test suite: 342 unit tests passed (89% line coverage), 25 integration tests passed.
- **Next Recommended Action**: Conduct Phase 3 Gate Review and proceed with Phase 4 ([`TASK-401`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md#phase-4-hybrid-retrieval-synthesis--citation-validation) — Context Normalizer & Token Budget Manager).
