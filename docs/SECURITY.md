# Security Policy & Threat Mitigation

This document defines the security architecture, threat model, input validation policies, query safety rules, and credential handling requirements for the Knowledge Graph RAG platform.

---

## 1. Security Invariants & Golden Rules

1. **Untrusted Inputs Principle**: All external inputs—including user queries, raw ingestion documents, and LLM generated strings—are strictly untrusted.
2. **Zero Dynamic Cypher Execution**: Direct LLM-to-Cypher generation without template constraints is strictly prohibited. All Neo4j interactions must execute parameterized Cypher templates.
3. **Zero Dynamic SQL**: All PostgreSQL interactions must use parameterized queries or SQLAlchemy ORM filters. String formatting/concatenation in SQL statements is strictly prohibited.
4. **Credential Isolation**: No secrets, tokens, or connection strings shall ever be committed to source control or logged in runtime telemetry.

---

## 2. Threat Model & Mitigations

| Threat | Impact | Mitigation Strategy | Residual Risk & Limitations |
| :--- | :--- | :--- | :--- |
| **Cypher Injection** | Unauthorized graph data exfiltration, database corruption, or Denial of Service. | Strict template catalog (`src/retrieval/cypher_templates.py`). User parameters are passed strictly via Cypher query parameter maps (`$entity_id`, `$limit`). | Complex multi-hop queries not covered by templates require expanding the template library. |
| **SQL Injection** | Vector index tampering, unauthorized data access. | Parameterized queries via SQLAlchemy / asyncpg. Vector search passes embeddings as parameter arrays. | None when strict parameterization is enforced. |
| **Indirect Prompt Injection** | Malicious document text overrides system prompt instructions during extraction or synthesis. | Strict XML delimiter framing (`<retrieved_context>...</retrieved_context>`), Pydantic schema validation, and system instruction priority enforcement. | Extremely subtle adversarial jailbreaks in document corpora might degrade extraction accuracy. |
| **Hallucinated Citations** | Synthesized answers claim false authoritative grounding. | Automated post-synthesis citation verification against retrieved candidate `chunk_id` set before returning API payload. | Real chunk text may still be misattributed if LLM misinterprets the retrieved context. |
| **Resource Exhaustion & ReDoS** | Complex graph traversals causing CPU/memory exhaustion in Neo4j. | Mandatory traversal depth limits ($D \le 3$), query execution timeouts (default 5.0s), and `LIMIT` clauses on all traversals. | Very high degree "hub nodes" may still require tuning neighborhood fetch limits. |
| **Secret Leakage in Logs** | API keys or credentials exposed in application logs or stack traces. | Custom logging formatter redacting Authorization headers, API keys, passwords, and connection URIs. | Developer custom print statements during debugging (mitigated via pre-commit linter checks). |

---

## 3. Safe Cypher Query Execution Guardrail

### Forbidden Pattern (Never Permitted)
```python
# FORBIDDEN: NEVER DO THIS
query = f"MATCH (e:Entity {{name: '{user_input}'}})-[r]->(target) RETURN target"
session.run(query)
```

### Approved Pattern (Mandatory)
```python
# MANDATORY: USE PARAMETERIZED TEMPLATES
QUERY_TEMPLATE = """
MATCH (e:Entity {id: $entity_id})-[r:DEPENDS_ON|USES*1..2]->(target:Entity)
RETURN e.name AS source, type(r) AS rel_type, target.name AS target_name, r.source_chunk_ids AS provenance
LIMIT $limit
"""
session.run(QUERY_TEMPLATE, {"entity_id": sanitized_entity_id, "limit": 20})
```

---

## 4. Environment Variables & Secret Handling

- All secrets must be supplied via environment variables or a local `.env` file (which is ignored by Git).
- A reference template [`.env.example`](file:///c:/Users/workp/OneDrive/Desktop/kgr/.env.example) is provided with placeholder values.
- In production, credentials should be injected via container runtime secrets or secure secret managers.

### Required Secret Variables
- `ANTHROPIC_API_KEY`: API key for Claude 3.5 Sonnet / Claude 3 Haiku.
- `NEO4J_PASSWORD`: Authentication credential for Neo4j database.
- `POSTGRES_PASSWORD`: Authentication credential for PostgreSQL database.

---

## 5. API Security & Input Validation

- **Request Payload Limits**: `POST /query` enforces maximum payload size of 10KB and question length limit of 1,000 characters.
- **Input Sanitization**: User inputs are stripped of control characters and normalized to UTF-8 before entering routing and retrieval pipelines.
- **Authentication**: In production deployment, FastAPI endpoints require Bearer token validation via `HTTPBearer` security schemes.

---

## 6. Least-Privilege Database Permissions

1. **Neo4j Database User**:
   - Ingestion role: `neo4j` (read/write access to nodes, relationships, and constraints).
   - Read-only query role (optional production hardening): read access restricted to `MATCH` operations.
2. **PostgreSQL Database User**:
   - Application role: CRUD permissions restricted strictly to `documents`, `chunks`, and `entity_resolution` tables. No superuser or schema alteration privileges during runtime query execution.
