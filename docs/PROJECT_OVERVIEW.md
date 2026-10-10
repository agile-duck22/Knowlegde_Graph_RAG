# Project Overview: Knowledge Graph RAG for Enterprise Data

---

## 1. Executive Summary & Problem Statement

Standard enterprise RAG (Retrieval-Augmented Generation) architectures rely primarily on dense vector embeddings for semantic document retrieval. While effective for localized factual retrieval and fuzzy semantic match, vector-only RAG exhibits critical systemic failure modes:
1. **Multi-Hop Blindness**: Failing to traverse explicit chains of relationships (e.g., *“Which microservices authored by Team A depend on packages with CVE vulnerabilities?”*).
2. **Global Aggregation Gaps**: Incapable of structured summarization across multiple entities (e.g., *“List all critical dependencies across all payment processing modules”*).
3. **Hallucinated Connections**: Attempting to bridge distant text chunks by cosine similarity without structural evidence.

This project delivers a **Hybrid Knowledge Graph (KG) + Vector RAG** system combining the precision and traversal power of a property graph (**Neo4j**) with the dense semantic retrieval of **pgvector (PostgreSQL)**, coordinated by a FastAPI orchestration service and powered by Anthropic Claude.

---

## 2. Target Users & Enterprise Use Cases

### Target Personas
- **Enterprise Software Architects & Engineers**: Exploring service dependencies, architecture decision records, and API interactions.
- **Compliance & Security Analysts**: Auditing regulatory constraints, security advisories, and cross-team dependencies.
- **Data & AI Engineers**: Developing production-grade RAG pipelines with verifiable grounding and measurable evaluation.

### Representative Use Cases
- **Multi-Hop Dependency Analysis**: Tracing software library dependencies and vulnerabilities through 2-to-3 degrees of separation.
- **Cross-Document Entity Synthesis**: Correlating policies, systems, and ownership models across disparate documentation silos.
- **High-Assurance QA**: Providing grounded answers where every statement requires verifiable, auditable citations back to raw source chunks.

---

## 3. Project Goals and Non-Goals

### Goals
- **Measurable Hybrid Advantage**: Demonstrate statistically significant retrieval precision improvements on multi-hop questions compared to vector-only baselines.
- **Zero Raw Cypher Execution**: Enforce absolute query safety using parameterized Cypher query templates and schema validation.
- **Dual-Store Invariant**: Ensure strict provenance linking between Neo4j graph entities/relationships and PostgreSQL text chunks via canonical UUID `chunk_id`.
- **Reproducible Evaluation**: Provide an automated benchmark suite running on a standardized question set (50–100 questions) evaluating accuracy, latency, and cost.
- **Production-Ready Clean Architecture**: Implement modular Python code with strict typing (`mypy`), linting (`ruff`), and comprehensive test coverage (`pytest`).

### Non-Goals
- **No Unrestricted Autonomous Agents**: The system is a deterministic retrieval and synthesis pipeline, not an open-ended autonomous agent with arbitrary tool execution.
- **No Infinite Real-Time Web Crawling**: Ingestion operates on discrete enterprise document corpora (JSONL/Markdown/PDF/Text).
- **No Distributed Multi-Cluster DB Setup**: Designed for single-instance or primary-replica Neo4j and PostgreSQL instances appropriate for enterprise departmental scale.
- **No Proprietary Vendor Lock-In**: LLM and embedding layers must use modular interfaces allowing swap between Anthropic, OpenAI, Cohere, or local models.

---

## 4. Scope Definition

### MVP Scope (3–4 Weeks Part-Time Timeline)
- **Domain Corpus**: Enterprise Technical Architecture & Service Catalog documentation corpus (100–500 documents).
- **Ontology**: 6–8 core entity types (`Service`, `Team`, `API`, `Database`, `Vulnerability`, `Repository`) and 10–12 relationship types.
- **Storage**: Dockerized Neo4j 5.x and PostgreSQL 16 + pgvector.
- **Ingestion**: Chunking, LLM-based structured extraction via Pydantic, entity resolution, and dual-store indexing.
- **Retrieval**: Intent classifier router, safe Cypher templates, pgvector HNSW search, and hybrid context ranker.
- **API**: FastAPI endpoints (`/health`, `/ready`, `/query`).
- **Evaluation**: 50-question benchmark comparing Vector-Only vs. Hybrid RAG.

### Stretch Goals
- Dynamic graph schema expansion with human-in-the-loop review.
- Hierarchical document summary nodes (Knowledge Graph hierarchical clustering).
- Graph visualization UI integration (e.g., Neo4j Bloom or interactive D3.js canvas).

---

## 5. Expected Deliverables

1. **Production Codebase**: Clean, typed Python package (`src/`) containing ingestion, graph store, vector store, routing, synthesis, and API modules.
2. **Infrastructure Configuration**: `docker-compose.yml` defining Neo4j, PostgreSQL with pgvector, and migration scripts.
3. **Curated Reference Dataset & Ontology**: Reference enterprise dataset with standardized ingestion scripts.
4. **Automated Evaluation Harness**: Test suite measuring retrieval recall, answer correctness, citation validity, latency, and cost.
5. **Documentation & Benchmarks**: Complete operational documentation and empirical benchmark results.

---

## 6. Acceptance Criteria for Project Completion

1. **Schema & Provenance Integrity**: 100% of graph entities and relationships trace back to a valid, existing `chunk_id` in PostgreSQL.
2. **Query Safety**: Zero dynamic Cypher string interpolations; all graph queries run through audited, parameterized templates.
3. **Groundedness**: All synthesis responses contain valid citations that match retrieved source chunks with zero dangling references.
4. **Evaluation Benchmark**: Complete benchmark executed over 50+ standardized questions demonstrating empirical performance metrics across single-hop, multi-hop, and aggregation queries.
5. **Test Suite Health**: Minimum 85% test coverage across unit and integration suites without requiring live API keys during standard unit test execution.
