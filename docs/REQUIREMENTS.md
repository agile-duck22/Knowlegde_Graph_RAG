# Requirements Specification

This document defines the functional and non-functional requirements for the Knowledge Graph RAG system.

---

## 1. Functional Requirements

### Document Ingestion & Chunking
- **`FR-001`**: The system shall parse unstructured documents (Markdown, text, structured JSONL) and divide them into overlapping semantic text chunks.
  - *Acceptance Criteria*: Chunking respects configurable `chunk_size` (tokens/characters) and `chunk_overlap`. Each chunk is assigned a unique UUID `chunk_id` and tracks `document_id`, `chunk_index`, and character offsets.
- **`FR-002`**: The system shall support idempotent re-ingestion of documents.
  - *Acceptance Criteria*: Re-ingesting a document with an identical hash updates existing records or performs a clean replacement without creating orphaned nodes, edges, or duplicate vector embeddings.

### Entity & Relationship Extraction
- **`FR-003`**: The system shall extract entities and relationships from chunks using structured LLM outputs conforming to the target domain ontology.
  - *Acceptance Criteria*: Extraction outputs parse strictly into Pydantic models. Extracted entities include `name`, `type`, `description`, and `aliases`. Extracted relationships include `source_entity`, `target_entity`, `relation_type`, `description`, and `confidence`.
- **`FR-004`**: Every extracted entity and relationship must retain explicit source provenance.
  - *Acceptance Criteria*: Extracted items record the source `chunk_id` and exact text excerpt supporting the extraction.

### Entity Resolution & Normalization
- **`FR-005`**: The system shall normalize entity names and resolve aliases to canonical entity identifiers.
  - *Acceptance Criteria*: Exact and alias matches resolve to a single canonical entity node in Neo4j. Ambiguous or low-confidence matches are logged and stored with distinct candidate IDs.

### Neo4j Property Graph Persistence
- **`FR-006`**: The system shall persist canonical entities and relationships into Neo4j using parameterized Cypher statements.
  - *Acceptance Criteria*: Uniqueness constraints on `(e:Entity {id})` prevent duplicate entity creation. Relationships store `source_chunk_ids`, `confidence`, and `last_updated_at`.
- **`FR-007`**: The system shall maintain bidirectional graph traversal capabilities up to a configurable maximum depth ($D \le 3$).
  - *Acceptance Criteria*: Traversal queries reject or truncate paths exceeding maximum configured hop depth to prevent combinatorial explosion.

### PostgreSQL & pgvector Indexing
- **`FR-008`**: The system shall persist chunk metadata and dense vector embeddings into PostgreSQL with pgvector.
  - *Acceptance Criteria*: Chunks are stored in the `chunks` table with `id`, `document_id`, `content`, `metadata`, and `embedding` vector column indexed using HNSW (`vector_cosine_ops`).

### Query Intent Routing
- **`FR-009`**: The system shall classify incoming user queries into one of four retrieval modes: `GRAPH_ONLY`, `VECTOR_ONLY`, `HYBRID`, or `OUT_OF_SCOPE`.
  - *Acceptance Criteria*: Routing model returns the mode, confidence score ($\in [0, 1]$), extracted entity candidates, and routing rationale. Low confidence ($< 0.65$) automatically triggers `HYBRID` fallback.

### Retrieval & Safe Cypher Execution
- **`FR-010`**: The system shall execute graph retrieval strictly through approved, parameterized Cypher templates.
  - *Acceptance Criteria*: Zero execution of model-generated raw Cypher strings. Cypher templates support 1-hop, 2-hop, 3-hop traversal, entity neighborhood, and dependency path aggregation.
- **`FR-011`**: The system shall execute vector retrieval using cosine similarity search in pgvector.
  - *Acceptance Criteria*: Vector search queries return top-$k$ nearest chunks matching similarity threshold, filtered by optional metadata tags.

### Context Normalization & Answer Synthesis
- **`FR-012`**: The system shall deduplicate and normalize retrieved context from graph and vector pipelines within a defined token budget.
  - *Acceptance Criteria*: Combined context merges overlapping chunk texts, presents structured graph paths clearly, and truncates safely within the LLM context window.
- **`FR-013`**: The system shall generate grounded natural language answers using Anthropic Claude with inline citations.
  - *Acceptance Criteria*: Answers strictly use bracketed citations (e.g., `[chunk-uuid]`) referencing retrieved context.

### Citation Validation Guardrail
- **`FR-014`**: The system shall validate all inline citations before returning the final response to the client.
  - *Acceptance Criteria*: Every bracketed citation in the answer is checked against the set of retrieved `chunk_id`s. Any hallucinated citation triggers automated response correction or citation scrubbing with warning metadata.

### API Interface & Telemetry
- **`FR-015`**: The system shall expose standard REST endpoints: `GET /health`, `GET /ready`, and `POST /query`.
  - *Acceptance Criteria*: `POST /query` returns JSON containing `answer`, `citations`, `retrieval_mode`, `retrieved_chunks`, `graph_paths`, and execution `request_id`.
- **`FR-016`**: The system shall log structured telemetry and track estimated token costs per request.
  - *Acceptance Criteria*: JSON structured logs record request latency, retrieval duration, token counts, and estimated USD cost.

---

## 2. Non-Functional Requirements

### Performance & Scalability
- **`NFR-001`**: End-to-end query response latency for standard queries must meet p50 $< 2.5\text{s}$ and p95 $< 5.0\text{s}$ (excluding cold-start network latency).
- **`NFR-002`**: Vector search in PostgreSQL over 50,000 chunks must execute in $< 50\text{ms}$.
- **`NFR-003`**: Neo4j parameterized graph traversal up to 2 hops must execute in $< 100\text{ms}$.

### Security & Safety
- **`NFR-004`**: Zero possibility of Cypher injection or SQL injection. All database interactions must use parameterized statements or typed ORMs.
- **`NFR-005`**: API credentials, database connection strings, and tokens must never appear in source code, logs, or error responses.
- **`NFR-006`**: Untrusted document content must be isolated from system prompt instructions to prevent indirect prompt injection.

### Reliability & Resilience
- **`NFR-007`**: Storage failures must be isolated; a transient failure in Neo4j must gracefully degrade to vector-only search with a degradation notice rather than hard crashing.
- **`NFR-008`**: Rate-limit handling and exponential backoff retry for Anthropic API calls must be implemented across all LLM pipeline stages.

### Maintainability & Testability
- **`NFR-009`**: Minimum 85% test coverage on core business logic, routers, extractors, and validators.
- **`NFR-010`**: Complete unit test execution must run entirely offline in $< 30\text{s}$ without requiring live database connections or paid API credentials.

---

## 3. Out-of-Scope Features

To prevent scope creep, the following capabilities are explicitly out of scope for the MVP:
1. Multi-tenant access control and row-level document security filters.
2. Real-time streaming synchronization with enterprise SaaS connectors (e.g., Salesforce, Jira, Google Drive).
3. Live voice or conversational multi-turn chat memory across distinct sessions.
4. Autonomous agentic tool generation or code execution environments.
