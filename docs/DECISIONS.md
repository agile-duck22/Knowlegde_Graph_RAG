# Architecture Decision Records (ADRs) & Open Questions

This document logs all foundational architectural decisions, evaluation trade-offs, and open questions requiring explicit resolution.

---

## 1. Decision Log

### `ADR-001`: Selection of Reference Domain Corpus
- **Date**: 2026-10-09
- **Status**: `PROPOSED` (Pending User Confirmation)
- **Context**: The project requires a realistic enterprise corpus with rich multi-hop relationships (teams, services, APIs, databases, vulnerabilities).
- **Options Considered**:
  1. *Enterprise Technical Architecture & Service Catalog* (Services, Teams, Databases, APIs, Vulnerabilities).
  2. *Biomedical Research Papers & Clinical Trials*.
  3. *Financial 10-K Filings & Corporate Governance*.
- **Decision & Rationale**: Select Option 1 (*Enterprise Technical Architecture & Service Catalog*). It provides deterministic multi-hop relationship chains (service $\rightarrow$ database $\rightarrow$ team $\rightarrow$ vulnerability) with easily verifiable ground truth.
- **Consequences**: Ontology and evaluation benchmarks are tailored to software architecture and infrastructure entities.
- **Related Documents**: [`docs/ONTOLOGY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ONTOLOGY.md), [`docs/DATA_MODEL.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DATA_MODEL.md).

---

### `ADR-002`: Dual-Store Shared Identifier Architecture
- **Date**: 2026-10-09
- **Status**: `ACCEPTED`
- **Context**: Relationships in Neo4j must provide auditable provenance back to exact document chunks stored in PostgreSQL.
- **Options Considered**:
  1. *Shared Canonical UUID `chunk_id`* across both PostgreSQL and Neo4j provenance properties.
  2. *Document URI only* (loses chunk-level precision).
  3. *Separate Graph Nodes for Chunks* (increases graph size by 10x with dense text property overhead).
- **Decision & Rationale**: Adopt Option 1. Neo4j relationship edges store `source_chunk_ids: [UUID]` pointing directly to PostgreSQL `chunks.id`. Keeps graph traversal lightweight while ensuring $100\%$ provenance auditability.
- **Consequences**: Graph queries return `source_chunk_ids`, which can be hydrated from PostgreSQL as needed.
- **Related Documents**: [`docs/DATA_MODEL.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DATA_MODEL.md), [`docs/ARCHITECTURE.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ARCHITECTURE.md).

---

### `ADR-003`: Embedding Provider & Vector Dimension Selection
- **Date**: 2026-10-09
- **Status**: `PROPOSED` (Pending User Confirmation)
- **Context**: pgvector requires fixed vector dimensions at schema definition time.
- **Options Considered**:
  1. `text-embedding-3-small` (OpenAI, 1536 dimensions).
  2. `all-MiniLM-L6-v2` via HuggingFace (Local, 384 dimensions).
  3. Cohere `embed-english-v3.0` (1024 dimensions).
- **Decision & Rationale**: Default to Option 1 (`text-embedding-3-small` / 1536 dim) with an abstract `EmbeddingClient` interface allowing zero-downtime swap to local models.
- **Consequences**: Schema default set to `VECTOR(1536)`.
- **Related Documents**: [`docs/CONFIGURATION.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/CONFIGURATION.md).

---

### `ADR-004`: Parameterized Cypher Templates vs. Dynamic Text-to-Cypher
- **Date**: 2026-10-09
- **Status**: `ACCEPTED`
- **Context**: LLMs generating arbitrary raw Cypher queries frequently produce syntax hallucinations, Cartesian explosive traversals, and potential Cypher injection vectors.
- **Options Considered**:
  1. *Audited Parameterized Cypher Template Catalog* with fixed entry points and bound parameters.
  2. *Unrestricted LLM Text-to-Cypher* with post-generation regex filters.
- **Decision & Rationale**: Adopt Option 1. Guarantees $100\%$ injection safety, bounded traversal latency, and predictable graph query execution.
- **Consequences**: The system can only execute graph patterns covered by the template catalog. New graph traversal patterns require adding templates.
- **Related Documents**: [`docs/SECURITY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/SECURITY.md), [`docs/RETRIEVAL_STRATEGY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/RETRIEVAL_STRATEGY.md).

---

## 2. Unresolved Open Questions

| Question ID | Question Summary | Current Proposed Default | Impacted Modules |
| :--- | :--- | :--- | :--- |
| `OQ-01` | Should we include an offline local embedding fallback (`sentence-transformers`) for local development without OpenAI/Anthropic network calls? | Yes, add `LocalEmbeddingClient` fallback in Phase 2. | `src/embeddings/client.py` |
| `OQ-02` | What is the maximum acceptable corpus size for the Phase 5 benchmark run to stay within API token budgets ($< \$10$ total run cost)? | Target 100 documents ($\approx 250\text{k}$ total tokens). | `src/eval/runner.py` |
