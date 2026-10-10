# Knowledge Graph RAG for Enterprise Data

> **Status**: Planning & Documentation Stage (Implementation not started)

A production-grade, hybrid Retrieval-Augmented Generation (RAG) system combining Knowledge Graph (Neo4j) structured relationship traversal with Vector Search (PostgreSQL + pgvector) dense semantic retrieval, orchestrated via FastAPI and powered by Anthropic Claude.

---

## 1. Project Purpose & Problem Statement

Standard vector-only RAG architectures struggle with multi-hop queries, relationship aggregation, and global dependency reasoning across complex enterprise documentation. Conversely, pure graph traversal misses semantic nuance, unstructured details, and dense factual lookups.

This project implements a **Hybrid Knowledge Graph + Vector RAG** system designed to:
1. Traverse multi-hop entity relationships deterministically via Neo4j.
2. Retrieve semantic context and unstructured facts via pgvector.
3. Dynamically route queries to the optimal retrieval engine using confidence scoring.
4. Generate strictly grounded answers with verifiable, chunk-level citations.
5. Provide a reproducible benchmark comparing Hybrid RAG against a Vector-Only baseline.

---

## 2. Key Capabilities

- **Hybrid Query Routing**: Automated intent classification routing questions to Graph, Vector, or Hybrid execution pipelines.
- **Safe Template-Driven Cypher Execution**: Zero raw LLM Cypher execution; all graph queries run through parameterized, audited query templates with strict traversal depth limits.
- **Dual-Store Provenance**: Every graph entity/relationship and vector record shares a canonical `chunk_id` referencing the exact raw document source chunk.
- **Grounded Answer Synthesis**: Strict citation validation ensuring every factual claim points to verified supporting source chunks.
- **Reproducible Evaluation Framework**: Standardized evaluation suite comparing single-hop, multi-hop, aggregation, and out-of-scope queries across latency, token cost, and retrieval precision.

---

## 3. Technology Stack

- **Language & Runtime**: Python 3.11+
- **API Framework**: FastAPI (Uvicorn / Pydantic v2)
- **Knowledge Graph Store**: Neo4j 5.x
- **Vector & Relational Store**: PostgreSQL 16 + `pgvector`
- **LLM & Embeddings**: Anthropic Claude API (`claude-3-5-sonnet` / `claude-3-haiku`), OpenAI/Cohere/HuggingFace embeddings (configurable)
- **Orchestration**: Custom lightweight pipelines with LangChain used selectively where beneficial
- **Testing & Quality**: pytest, pytest-asyncio, ruff, mypy
- **Containerization**: Docker Compose

---

## 4. Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Client / Enterprise    │
                          └─────────────┬────────────┘
                                        │ HTTP POST /query
                                        ▼
                          ┌──────────────────────────┐
                          │     FastAPI Gateway      │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │   Query Router Engine    │
                          │ (Intent & Entity Linking)│
                          └──────┬────────────┬──────┘
                   Graph Intent  │            │  Vector Intent
                   (Multi-hop)   │            │  (Semantic/Fact)
                                 ▼            ▼
                   ┌───────────────┐        ┌───────────────┐
                   │ Neo4j Engine  │        │ pgvector Store│
                   │(Cypher Templ.)│        │ (HNSW Search) │
                   └───────┬───────┘        └───────┬───────┘
                           │                        │
                           └───────────┬────────────┘
                                       │ Retrieved Context & Chunk IDs
                                       ▼
                          ┌──────────────────────────┐
                          │   Context Normalizer &   │
                          │   Deduplication Engine   │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Answer Synthesis (Claude)│
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Citation Validator &     │
                          │ Grounding Guardrail      │
                          └─────────────┬────────────┘
                                        │ Validated Response
                                        ▼
                          ┌──────────────────────────┐
                          │ JSON Response + Provenance│
                          └──────────────────────────┘
```

---

## 5. Documentation Navigation

Comprehensive project specifications are maintained in the [`docs/`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/) directory:

| Document | Purpose |
| :--- | :--- |
| [**INDEX.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/INDEX.md) | Central documentation navigation hub and reading guide |
| [**PROJECT_OVERVIEW.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/PROJECT_OVERVIEW.md) | Vision, scope, personas, deliverables, and acceptance criteria |
| [**REQUIREMENTS.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/REQUIREMENTS.md) | Functional (`FR-xxx`) and Non-Functional (`NFR-xxx`) specifications |
| [**ARCHITECTURE.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ARCHITECTURE.md) | Component architecture, sequence diagrams, and failure isolation |
| [**DATA_MODEL.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DATA_MODEL.md) | Neo4j property graph schema, PostgreSQL DDL, and dual-store invariants |
| [**ONTOLOGY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ONTOLOGY.md) | Entity types, relationship schemas, and normalization rules |
| [**RETRIEVAL_STRATEGY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/RETRIEVAL_STRATEGY.md) | Routing logic, safe Cypher templates, vector search, and ranking |
| [**API_SPEC.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/API_SPEC.md) | OpenAPI REST contract, request/response models, and error codes |
| [**SECURITY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/SECURITY.md) | Threat model, Cypher injection defenses, prompt isolation, and secrets |
| [**TASKS.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md) | Authoritative Phase 0–5 implementation backlog and task definitions |
| [**MEMORY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/MEMORY.md) | Persistent session memory, active status, confirmed invariants |
| [**TESTING.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TESTING.md) | Unit, integration, security, and mocked testing strategy |
| [**EVALUATION.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/EVALUATION.md) | Benchmark protocol, metrics, question sets, and empty result matrices |
| [**CONFIGURATION.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/CONFIGURATION.md) | Environment variables, defaults, tuning knobs, and config schema |
| [**DEPLOYMENT.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DEPLOYMENT.md) | Docker Compose infrastructure, migrations, and local runbooks |
| [**DECISIONS.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DECISIONS.md) | Architecture Decision Records (ADRs) and unresolved decisions |
| [**OBSERVABILITY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/OBSERVABILITY.md) | Structured logging, metrics, tracing, and token cost tracking |
| [**TROUBLESHOOTING.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TROUBLESHOOTING.md) | Common failure modes, diagnostics, and recovery runbooks |
| [**CHANGELOG.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/CHANGELOG.md) | Chronological release and milestone history |

---

## 6. Prerequisites & Planned Setup (Not Yet Implemented)

> [!WARNING]
> The setup instructions below are design specifications and have **not yet been executed or verified**. Implementation begins in Phase 0.

### Prerequisites
- Docker Engine 24+ & Docker Compose v2+
- Python 3.11+
- Anthropic API Key (`ANTHROPIC_API_KEY`)

### Planned Quickstart
```bash
# 1. Clone repository and set up virtual environment
git clone <repo-url>
cd kgr
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your credentials

# 3. Start storage infrastructure
docker compose up -d neo4j postgres

# 4. Run database migrations and schema constraints
python -m src.db.migrate

# 5. Ingest reference dataset
python -m src.ingest --corpus data/raw/corpus.jsonl

# 6. Start API service
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 7. Project Status & Implementation Roadmap

- [x] **Documentation & Architecture Phase**: Complete specification, data modeling, security rules, and task roadmap.
- [ ] **Phase 0**: Scaffolding, container infrastructure, configuration validation, and test harness.
- [ ] **Phase 1**: Ingestion pipeline, LLM entity/relationship extraction, and Neo4j property graph persistence.
- [ ] **Phase 2**: PostgreSQL/pgvector indexing, HNSW vector search, and dual-store provenance validation.
- [ ] **Phase 3**: Intent classification, query routing, entity linking, and safe parameterized Cypher templates.
- [ ] **Phase 4**: Hybrid context merging, Claude grounded answer generation, and strict citation validation.
- [ ] **Phase 5**: 50–100 question evaluation benchmark, API hardening, latency/cost analysis, and portfolio readiness.

---

## 8. Planned Benchmark Results

*Experimental evaluation results will be published here upon completion of Phase 5.*

| Query Category | Vector-Only Accuracy | Hybrid KG+Vector Accuracy | Citation Validity | Latency p95 (Vector) | Latency p95 (Hybrid) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Single-Hop Direct Fact | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` |
| 2-Hop Relationship | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` |
| 3-Hop Dependency Path | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` |
| Relationship Aggregation | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` |
| Out-of-Scope / Refusal | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` | `[Pending Phase 5]` |

---
*Maintained under strict Architecture Decision Records. See [`docs/DECISIONS.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DECISIONS.md).*
