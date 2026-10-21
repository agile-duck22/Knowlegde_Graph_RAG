# Changelog & Milestone History

All notable architectural, documentation, and implementation changes will be documented chronologically in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.5.0-benchmark] - 2026-10-21
### Added
- **`TASK-501` to `TASK-505` Implementation**:
  - Full 50-question benchmark dataset in `data/eval/benchmark_questions.json` covering single-hop, 2-hop, 3-hop, aggregation, and out-of-scope categories.
  - Automated evaluation runner `src/eval/runner.py` and evaluation metric calculation `src/eval/metrics.py`.
  - Full benchmark execution artifact `benchmark_results/run_hybrid_v3_threshold040.json` achieving **80.0% overall accuracy** (40/50 cases correct), 87% fact coverage, and 0% citation hallucination rate at similarity threshold 0.40.
  - End-to-end security audit suite `tests/security/test_security_audit.py` verifying Cypher injection defense, SQL injection defense, indirect prompt injection framing, DoS limits, and key masking.
  - Synchronized repository documentation (`README.md`, `docs/TASKS.md`, `docs/MEMORY.md`, `docs/EVALUATION.md`).

## [0.4.0-synthesis] - 2026-10-12
### Added
- **`TASK-401` to `TASK-406` Implementation**:
  - Context normalizer and token budget manager `src/synthesis/normalizer.py` merging graph and vector contexts.
  - Grounded answer synthesizer `src/synthesis/synthesizer.py` with bracketed chunk citations.
  - Citation guardrail `src/synthesis/citation_validator.py` scrubbing unretrieved citations.
  - FastAPI application endpoints `src/api/main.py` and `src/api/routes.py` (`/health`, `/ready`, `/api/v1/query`).
  - Multi-provider LLM adapter `src/llm/client.py` supporting Gemini, Groq, OpenRouter, and Anthropic with fail-closed allowlists.
  - Structured JSON telemetry logging in `src/observability/logging.py` with API key redaction.

## [0.3.0-routing] - 2026-10-11
### Added
- **`TASK-301` to `TASK-305` Implementation**:
  - Parameterized Cypher query catalog `src/retrieval/cypher_templates.py` (`T-01` to `T-09`).
  - Safe graph retriever `src/retrieval/graph_retriever.py` with parameter enforcement and depth bounds.
  - Query intent classifier `src/routing/intent_classifier.py` and entity linker `src/routing/entity_linker.py`.
  - Centralized router coordinator `src/routing/router.py`.

## [0.2.1-pgvector] - 2026-10-10
### Added
- **`TASK-201` to `TASK-205` Implementation**:
  - PostgreSQL schema migrations for `documents`, `chunks`, `entity_links`, and dual-dimension vector columns (384-dim and 1536-dim).
  - SentenceTransformers local embedding generator supporting `all-MiniLM-L6-v2`.
  - Vector retriever `src/retrieval/vector_retriever.py` with cosine distance search and dynamic column selection.
  - Dual-store provenance check test suite `src/validation/provenance_check.py`.

## [0.1.1-scaffold] - 2026-10-09
### Added
- **`TASK-001` Implementation**:
  - `pyproject.toml`: Modern packaging specification with dependencies (`fastapi`, `neo4j`, `pgvector`, `asyncpg`, `sqlalchemy`, `anthropic`, `pydantic`, `pytest`), pytest configuration, ruff linter settings, and strict mypy configuration.
  - `requirements.txt`: Frozen core runtime dependencies.
  - `requirements-dev.txt`: Development and testing dependencies (`pytest`, `ruff`, `mypy`).
  - `src/__init__.py`: Base package initialization with semantic version `0.1.0`.
  - `tests/__init__.py`, `tests/conftest.py`, `tests/unit/test_scaffolding.py`: Base offline test harness and fixture validation.
- **`TASK-002` Implementation**:
  - `docker-compose.yml`: Multi-container specification for Neo4j 5.18 Community and PostgreSQL 16 (pgvector). Configured with localhost-only port bindings (`127.0.0.1`), health checks (`pg_isready` and `cypher-shell`), and persistent named volumes (`kgr_pgdata`, `kgr_neo4jdata`).
  - `tests/unit/test_docker_compose.py`: Automated specification test verifying pinned image tags, security policies, and volume definitions.
- **`TASK-003` Implementation**:
  - `src/config.py`: Centralized Pydantic Settings configuration module with full validation for PostgreSQL, Neo4j, Claude, embedding parameters, chunk size/overlap constraints, traversal limits, secret masking in `repr`, and cached singleton accessor `get_settings()`.
  - `tests/unit/test_config.py`: Comprehensive test suite verifying defaults, environment overrides, constraint validations, and secret redaction.
- **`TASK-004` Implementation**:
  - Refactored `src/config.py` to use a single canonical Pydantic Settings implementation, removing duplicate stdlib fallback code.
  - `tests/conftest.py`: Added lightweight mock factories for Anthropic Claude, Neo4j driver/sessions, and asyncpg PostgreSQL connection pools without live services or paid API keys.
  - `tests/unit/test_mock_factories.py`: Verified mock factory execution offline.
  - `.github/workflows/ci.yml`: Created multi-version matrix GitHub Actions CI workflow (Python 3.11, 3.12, 3.13) running Ruff, Mypy, and Pytest with code coverage tracking.
  - Validated 100% test passing (24/24 tests), 100% code coverage on `src/`, 0 Ruff lint/format errors, and 0 Mypy strict type errors.

## [0.2.0-corpus] - 2026-10-09
### Added
- **`TASK-101` Implementation**:
  - `src/data/generate_corpus.py`: Deterministic generator script creating 145 richly connected synthetic enterprise technology documents.
  - `data/raw/corpus.jsonl`: Reference enterprise dataset across 11 document types (`team_charter`, `service_spec`, `api_contract`, `database_spec`, `infra_spec`, `repository_manifest`, `security_advisory`, `architecture_decision_record`, `post_mortem`, `operational_runbook`, `integration_spec`).
  - `tests/unit/test_corpus.py`: Automated integrity test suite validating count, uniqueness, schema typing, entity prefix standards, domain distribution, and multi-hop continuity.

---

## [0.1.0-doc] - 2026-10-09
### Added
- **Initial Documentation System**:
  - `README.md`: Project landing page, architecture overview, planned quickstart, and roadmap.
  - `GEMINI.md` & `AGENTS.md`: Strict agent guidelines, architectural invariants, and interaction protocol.
  - `.gitignore`: Comprehensive ignore rules for Python, virtual environments, secrets, and database files.
  - `docs/INDEX.md`: Documentation navigation hub and authority hierarchy.
  - `docs/PROJECT_OVERVIEW.md`: Vision, scope, personas, and completion criteria.
  - `docs/REQUIREMENTS.md`: Functional (`FR-001` to `FR-016`) and Non-Functional (`NFR-001` to `NFR-010`) requirements.
  - `docs/ARCHITECTURE.md`: Component diagrams, sequence diagrams, and failure isolation model.
  - `docs/SECURITY.md`: Threat model, Cypher injection defenses, prompt isolation, and secret management.
  - `docs/TASKS.md`: Implementation backlog across Phases 0 to 5 with 27 discrete tasks.
  - `docs/MEMORY.md`: Persistent session memory and confirmed invariants.
  - `docs/DATA_MODEL.md`: Dual-store schema contracts, DDL, Cypher constraints, and shared `chunk_id` design.
  - `docs/ONTOLOGY.md`: Domain ontology (7 entity types, 10 relationships, resolution rules).
  - `docs/RETRIEVAL_STRATEGY.md`: Routing taxonomy, parameterized Cypher templates, and vector search strategy.
  - `docs/API_SPEC.md`: REST API specification (OpenAPI contract) for FastAPI endpoints.
  - `docs/TESTING.md`: Test strategy, mocking rules, and unit/integration test case catalog.
  - `docs/EVALUATION.md`: 50-question evaluation benchmark protocol and reporting matrix.
  - `docs/OBSERVABILITY.md`: Structured JSON logging schema, metrics, and token cost tracking.
  - `docs/DEPLOYMENT.md`: Docker Compose configuration and operational runbooks.
  - `docs/CONFIGURATION.md`: Environment variables and empirical tuning parameters reference.
  - `docs/DECISIONS.md`: Architectural Decision Records (`ADR-001` to `ADR-004`) and open questions.
  - `docs/TROUBLESHOOTING.md`: Diagnostic matrix and recovery runbooks.
  - `docs/CHANGELOG.md`: Initial change log.
