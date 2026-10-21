# Persistent Agent Memory & Session State

> **Last Updated**: 2026-10-21
> **Current Project Status**: Phase 5 — Evaluation, Security Hardening, and Git Synchronization Complete

---

## 1. Confirmed Architecture Decisions

- **Storage Architecture**: Neo4j 5.x for Knowledge Graph relationship traversal; PostgreSQL 16 + `pgvector` for raw text chunks and dense vector search (supporting dual 384-dim and 1536-dim vector columns).
- **Dual-Store Invariant**: Canonical UUID `chunk_id` is shared between PostgreSQL `chunks.id` and Neo4j provenance properties (`source_chunk_ids`).
- **Query Safety**: Zero execution of dynamic LLM-generated raw Cypher. All graph traversals use pre-audited, parameterized Cypher templates (`T-01` to `T-09`) with traversal depth limits ($D \le 3$).
- **Groundedness**: All synthesis responses use multi-provider LLMs (Anthropic Claude, Google Gemini, Groq, OpenRouter) with strict bracketed citations validated against retrieved candidate `chunk_id`s before returning to the user.
- **Test Strategy**: Offline unit tests with mocks by default; opt-in integration tests for live database containers; dedicated security audit suite in `tests/security/`.

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
- **Phase 3 (Routing & Cypher Templates)**: `DONE` (`TASK-301`–`TASK-305` all verified & passing)
- **Phase 4 (Synthesis & Citations)**: `DONE` (`TASK-401`–`TASK-406` all verified & passing)
- **Phase 5 (Evaluation & Benchmarks)**: `DONE` (`TASK-501` ✅, `TASK-502` ✅, `TASK-503` ✅, `TASK-504` ✅, `TASK-505` ✅)

---

## 4. Current Blockers, Diagnostics & Known Issues

- **EVAL-028 Corpus Specification Gap**: In `data/raw/corpus.jsonl`, `service:cart-service` does not explicitly state a direct relationship to `service:checkout-service` or the PCI-isolated VPC (`infra:pci-isolated-vpc`). The synthesizer correctly refrains from hallucinating ungrounded edges. Blocked awaiting an evidence-backed source corpus correction.
- **EVAL-038 Verification Status**: Entity resolution and T-07 template expansion for PostgreSQL application databases (`database:customer-orders-pg`, `database:user-directory-pg`, `database:analytics-warehouse-pg`) via `cluster_engine` traversal verified in focused unit/traversal tests.
- **Benchmark Run Artifact**: Reference run `benchmark_results/run_hybrid_v3_threshold040.json` achieves 80.0% overall accuracy (40/50 questions correct) with 0% citation hallucination rate and 87% fact coverage at similarity threshold 0.40.
- **Evaluator Cost Metric**: The reported cost of $0.00 is an artifact of local embeddings (`all-MiniLM-L6-v2`) and free-tier inference adapters, not an assurance of zero commercial API costs.
- **Security Audit**: All 14 tests in `tests/security/test_security_audit.py` pass cleanly offline.

---

## 5. Last Completed Task & Next Recommended Action

- **Last Completed Task**: MASTER TASK — Synchronized repository documentation (`README.md`, `docs/TASKS.md`, `docs/MEMORY.md`, `docs/EVALUATION.md`, `docs/CHANGELOG.md`), repaired security audit suite, verified unit suites offline, and staged project artifacts.
- **Next Recommended Action**: Prepare live demo using `uvicorn src.api.main:app` and Docker Compose storage containers.
