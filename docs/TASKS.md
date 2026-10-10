# Implementation Tasks & Backlog

This document is the authoritative task backlog for the Knowledge Graph RAG platform. Implementation proceeds sequentially through Phase 0 to Phase 5.

---

## 1. Task Status Workflow

- `TODO`: Not started.
- `IN_PROGRESS`: Currently being implemented.
- `BLOCKED`: Blocked on dependency or unresolved decision.
- `DONE`: Fully implemented, tests executed and passing, definition of done satisfied.

> [!IMPORTANT]
> Never mark a task as `DONE` without executing tests and verifying implementation correctness.

---

## 2. Phase Breakdown

### Phase 0: Project Scaffolding & Test Infrastructure
*Objective: Establish repository structure, package management, Docker Compose databases, configuration models, and unit test harnesses.*

| Task ID | Description | Priority | Dependencies | Expected Modules | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TASK-001` | Create Python packaging setup (`pyproject.toml`, `requirements.txt`) with FastAPI, Neo4j, asyncpg, pgvector, pydantic, anthropic, pytest | P0 | None | `pyproject.toml`, `requirements.txt` | `DONE` |
| `TASK-002` | Create Docker Compose file for Neo4j 5.x and PostgreSQL 16 with pgvector extension | P0 | None | `docker-compose.yml`, `docker/` | `DONE` |
| `TASK-003` | Implement Pydantic Settings configuration module and `.env.example` validation | P0 | `TASK-001` | `src/config.py`, `.env.example` | `DONE` |
| `TASK-004` | Setup pytest test harness, fixtures, mock factories, and CI test runner | P0 | `TASK-001` | `tests/conftest.py`, `tests/` | `DONE` |

### Phase 1: Corpus, Ontology, Extraction & Graph Persistence
*Objective: Build document ingestion, chunking, LLM structured entity/relationship extraction, and Neo4j property graph persistence.*

| Task ID | Description | Priority | Dependencies | Expected Modules | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TASK-101` | Create reference enterprise corpus dataset (100+ documents in JSONL format) | P0 | `TASK-001` | `data/raw/corpus.jsonl` | `DONE` |
| `TASK-102` | Implement Document Loader and Semantic Text Chunker with canonical `chunk_id` UUID generation | P0 | `TASK-101` | `src/ingest/loader.py`, `src/ingest/chunker.py` | `DONE` |
| `TASK-103` | Define Pydantic extraction schemas and Claude prompt templates for domain ontology | P0 | `TASK-102` | `src/extraction/schemas.py`, `src/extraction/prompts.py` | `DONE` |
| `TASK-104` | Implement Structured LLM Extractor with retry logic and Pydantic validation | P0 | `TASK-103` | `src/extraction/extractor.py` | `DONE` |
| `TASK-105` | Implement Entity Resolver and Alias Normalizer | P1 | `TASK-104` | `src/extraction/resolver.py` | `DONE` |
| `TASK-106` | Implement Neo4j database driver, constraints, and idempotent batch Cypher write pipelines | P0 | `TASK-105`, `TASK-002` | `src/db/neo4j.py` | `DONE` |
| `TASK-107` | Create end-to-end ingestion pipeline script and CLI for graph store | P1 | `TASK-106` | `src/ingest/pipeline.py` | `DONE` |

### Phase 2: PostgreSQL / pgvector Indexing & Vector Search
*Objective: Store raw chunks and vector embeddings in PostgreSQL with HNSW indexes; validate dual-store provenance.*

| Task ID | Description | Priority | Dependencies | Expected Modules | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TASK-201` | Create PostgreSQL DDL migration scripts (tables: `documents`, `chunks`, `entity_links`) and HNSW index creation | P0 | `TASK-002` | `src/db/migrations/`, `src/db/postgres.py` | `DONE` |
| `TASK-202` | Implement Embedding Generator client supporting configurable models with batching | P0 | `TASK-003` | `src/embeddings/client.py` | `DONE` |
| `TASK-203` | Implement PostgreSQL chunk and vector embedding writer with idempotent upserts | P0 | `TASK-201`, `TASK-202` | `src/db/postgres_writer.py` | `DONE` |
| `TASK-204` | Implement Vector Retriever with cosine distance search, top-$k$, and metadata filtering | P0 | `TASK-203` | `src/retrieval/vector_retriever.py` | `DONE` |
| `TASK-205` | Build Dual-Store Provenance & Consistency Validator test suite | P1 | `TASK-106`, `TASK-203` | `src/validation/provenance_check.py` | `DONE` |

### Phase 3: Query Routing & Safe Cypher Templates
*Objective: Implement query intent classifier, entity linker, and parameterized Cypher query template library.*

| Task ID | Description | Priority | Dependencies | Expected Modules | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TASK-301` | Create audited Parameterized Cypher Query Template catalog with depth and node limits | P0 | `TASK-106` | `src/retrieval/cypher_templates.py` | `TODO` |
| `TASK-302` | Implement Safe Graph Retriever executing approved Cypher templates | P0 | `TASK-301` | `src/retrieval/graph_retriever.py` | `TODO` |
| `TASK-303` | Implement Query Intent Classifier (Graph vs Vector vs Hybrid vs Out-of-Scope) | P0 | `TASK-003` | `src/routing/intent_classifier.py` | `TODO` |
| `TASK-304` | Implement Entity Linker matching question terms to canonical Neo4j entity IDs | P0 | `TASK-105` | `src/routing/entity_linker.py` | `TODO` |
| `TASK-305` | Integrate Router Engine coordinating classification, linking, and retrieval delegation | P0 | `TASK-302`, `TASK-303`, `TASK-304` | `src/routing/router.py` | `TODO` |

### Phase 4: Hybrid Retrieval, Synthesis & Citation Validation
*Objective: Combine retrieved subgraphs and vector chunks, synthesize answers with Claude, and enforce citation integrity.*

| Task ID | Description | Priority | Dependencies | Expected Modules | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TASK-401` | Implement Context Normalizer & Token Budget Manager merging graph and vector contexts | P0 | `TASK-204`, `TASK-302` | `src/synthesis/normalizer.py` | `TODO` |
| `TASK-402` | Implement Grounded Answer Synthesizer using Anthropic Claude with bracketed citations | P0 | `TASK-401` | `src/synthesis/synthesizer.py` | `TODO` |
| `TASK-403` | Implement Citation & Grounding Guardrail verifying citations against retrieved `chunk_id`s | P0 | `TASK-402` | `src/synthesis/citation_validator.py` | `TODO` |
| `TASK-404` | Implement FastAPI application endpoints (`GET /health`, `GET /ready`, `POST /query`) | P0 | `TASK-403` | `src/api/main.py`, `src/api/routes.py` | `TODO` |
| `TASK-405` | Implement Structured Logging and Request Telemetry Middleware | P1 | `TASK-404` | `src/observability/logging.py`, `src/observability/metrics.py` | `TODO` |

### Phase 5: Evaluation Benchmark, Hardening & Final Documentation
*Objective: Execute 50–100 question evaluation benchmark comparing Vector vs Hybrid RAG, harden API, and finalize results.*

| Task ID | Description | Priority | Dependencies | Expected Modules | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TASK-501` | Create 50–100 question benchmark dataset categorized by single-hop, 2-hop, 3-hop, and aggregation | P0 | `TASK-101` | `data/eval/benchmark_questions.json` | `TODO` |
| `TASK-502` | Implement automated Evaluation Runner executing Vector-Only vs. Hybrid RAG runs | P0 | `TASK-404`, `TASK-501` | `src/eval/runner.py`, `src/eval/metrics.py` | `TODO` |
| `TASK-503` | Execute benchmark, compute accuracy, citation validity, latency, and cost matrices | P0 | `TASK-502` | `docs/EVALUATION.md`, `benchmark_results/` | `TODO` |
| `TASK-504` | Perform end-to-end security audit and vulnerability scan | P1 | `TASK-404` | `tests/security/` | `TODO` |
| `TASK-505` | Finalize README.md, documentation updates, and release changelog | P1 | `TASK-503`, `TASK-504` | `README.md`, `docs/CHANGELOG.md` | `TODO` |

---

## 3. Weekly Implementation Schedule & Milestones

- **Week 1 (Milestone 1)**: Complete Phase 0 and Phase 1. Ingestion pipeline, ontology extraction, and Neo4j graph storage verified.
- **Week 2 (Milestone 2)**: Complete Phase 2 and Phase 3. PostgreSQL/pgvector indexing, intent classification, and safe Cypher templates verified.
- **Week 3 (Milestone 3)**: Complete Phase 4. Hybrid context ranking, Claude grounded synthesis, citation guardrails, and FastAPI endpoints verified.
- **Week 4 (Milestone 4)**: Complete Phase 5. Evaluation benchmarks executed, performance metrics documented, security hardening verified.

---

## 4. Final Acceptance Checklist

- [ ] All 27 tasks marked `DONE` with verified passing tests.
- [ ] 100% of graph relations trace back to valid `chunk_id`s in PostgreSQL.
- [ ] Zero unparameterized Cypher queries in codebase.
- [ ] Unit test suite passes offline without network/API dependencies.
- [ ] Empirical evaluation results recorded in `docs/EVALUATION.md` and `README.md`.
