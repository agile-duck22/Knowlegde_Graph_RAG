# Testing Strategy & Test Plan

This document defines the testing methodology, suite structure, mocking guidelines, and test cases across unit, integration, API, security, and end-to-end levels.

---

## 1. Testing Philosophy & Invariants

1. **Fast Offline Unit Tests by Default**: The entire standard unit test suite (`pytest tests/unit`) must execute in $< 30$ seconds completely offline without live database containers or external API keys.
2. **Strict Mocking of Paid LLM APIs**: External Anthropic API calls are mocked in unit tests using deterministic fixtures.
3. **Opt-in Live Integration Tests**: Tests against live Neo4j and PostgreSQL instances require Docker Compose and are tagged with `@pytest.mark.integration`.
4. **No Fabricated Assertions**: All tests must verify real assertions against Pydantic models, query results, and error exceptions.

---

## 2. Test Suite Structure

```text
tests/
├── conftest.py                   # Shared pytest fixtures, mock clients, test data
├── unit/
│   ├── test_chunker.py           # Chunking boundaries, UUIDs, character offsets
│   ├── test_extraction.py        # Pydantic extraction parsing and prompt formatting
│   ├── test_entity_resolver.py   # Normalization, alias resolution, and deduplication
│   ├── test_cypher_templates.py  # Parameterized Cypher query validation and safety
│   ├── test_intent_router.py     # Intent classification logic and mode switching
│   ├── test_normalizer.py        # Context merging and token budgeting
│   └── test_citation_guard.py    # Citation integrity checks and hallucination scrubbing
├── integration/
│   ├── test_neo4j_adapter.py     # Live Neo4j constraints, MERGE logic, and traversal
│   ├── test_postgres_vector.py   # Live pgvector HNSW indexing and cosine similarity
│   └── test_dual_store_sync.py   # Shared chunk_id provenance validation
├── api/
│   ├── test_health_routes.py     # /health and /ready endpoint responses
│   └── test_query_route.py       # /api/v1/query payload validation, responses, errors
└── security/
    ├── test_cypher_injection.py  # Hostile input validation against Cypher templates
    ├── test_prompt_injection.py  # Document prompt injection isolation
    └── test_secret_redaction.py  # Telemetry log scrubber validation
```

---

## 3. Key Test Cases

### 1. Ingestion & Chunking
- **`TEST-CHUNK-01`**: Verify `chunk_size=500` and `chunk_overlap=50` produces overlapping chunks with valid `char_start` and `char_end`.
- **`TEST-CHUNK-02`**: Verify every chunk receives a valid UUID v4 `chunk_id`.

### 2. Entity Normalization & Resolution
- **`TEST-NORM-01`**: Verify `"Payment Svc"` and `"payment-service"` resolve to identical canonical ID `service:payment-service`.
- **`TEST-NORM-02`**: Verify duplicate relationship extraction from the same chunk results in a single edge with aggregated provenance.

### 3. Safe Cypher Execution
- **`TEST-CYPHER-01`**: Verify template parameters cannot escape query boundaries (e.g., passing `' OR 1=1 --` into `$entity_id`).
- **`TEST-CYPHER-02`**: Verify traversal paths strictly terminate at `max_hops` limit ($D \le 3$).

### 4. Citation Validation & Guardrail
- **`TEST-CITE-01`**: Answer contains citation `[chunk-1]`, which exists in retrieved set $\rightarrow$ Citation validated, passed through.
- **`TEST-CITE-02`**: Answer contains hallucinated citation `[chunk-999]` not in retrieved set $\rightarrow$ Citation scrubbed, warning recorded in response metadata.

### 5. Dual-Store Invariant & Failure Isolation
- **`TEST-DUAL-01`**: Verify that every `source_chunk_ids` entry on a Neo4j relationship exists in the PostgreSQL `chunks` table.
- **`TEST-DUAL-02`**: Verify query pipeline falls back gracefully to `VECTOR_ONLY` if Neo4j returns a connection timeout.

---

## 4. Test Execution Commands

```bash
# Run unit tests only (Fast, offline, zero API keys required)
pytest tests/unit

# Run unit tests with code coverage report
pytest tests/unit --cov=src --cov-report=term-missing

# Run live integration tests (Requires Docker Compose Neo4j & Postgres)
pytest -m integration

# Run full API and security tests
pytest tests/api tests/security
```
