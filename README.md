# Knowledge Graph RAG for Enterprise Data

> **Status**: Verified Implementation & Benchmarked Prototype (80% Benchmark Accuracy on HYBRID v3)

A production-grade, hybrid Retrieval-Augmented Generation (RAG) system combining Knowledge Graph (Neo4j) structured relationship traversal with Vector Search (PostgreSQL + pgvector) dense semantic retrieval, orchestrated via FastAPI and powered by multi-provider LLMs (Anthropic Claude, Google Gemini, Groq, OpenRouter) and local SentenceTransformers embeddings (`all-MiniLM-L6-v2`).

---

## 1. Project Purpose & Problem Statement

Standard vector-only RAG architectures struggle with multi-hop queries, relationship aggregation, and global dependency reasoning across complex enterprise documentation. Conversely, pure graph traversal misses semantic nuance, unstructured details, and dense factual lookups.

This project implements a **Hybrid Knowledge Graph + Vector RAG** system designed to:
1. Traverse multi-hop entity relationships deterministically via Neo4j.
2. Retrieve semantic context and unstructured facts via pgvector (supporting both 384-dim local and 1536-dim embeddings).
3. Dynamically route queries to the optimal retrieval engine (Graph, Vector, Hybrid, Out-of-Scope) using confidence scoring.
4. Generate strictly grounded answers with verifiable, chunk-level citations.
5. Provide a reproducible benchmark comparing Hybrid RAG against a Vector-Only baseline.

---

## 2. Key Capabilities

- **Hybrid Query Routing**: Automated intent classification routing questions to Graph, Vector, Hybrid, or Out-of-Scope pipelines with deterministic heuristic fast-paths and LLM fallback.
- **Safe Template-Driven Cypher Execution**: Zero raw LLM Cypher execution; all graph queries run through parameterized, audited query templates (`T-01` through `T-09`) with strict traversal depth limits.
- **Dual-Store Provenance**: Every graph entity/relationship and vector record shares a canonical `chunk_id` (UUID format) referencing the exact raw document source chunk.
- **Grounded Answer Synthesis**: Strict citation validation ensuring every factual claim points to verified supporting source chunks.
- **Multi-Provider LLM & Zero-Cost Guardrails**: Native adapter supporting Gemini (`gemini-3.1-flash-lite`), Groq (`openai/gpt-oss-20b`), OpenRouter (`:free` models), and Anthropic Claude, with local SentenceTransformers embeddings for zero-cost operation.
- **Reproducible Evaluation Framework**: Standardized evaluation suite comparing single-hop, multi-hop, aggregation, and out-of-scope queries across accuracy, latency, cost, and retrieval precision.

---

## 3. Technology Stack

- **Language & Runtime**: Python 3.11+ / Python 3.12
- **API Framework**: FastAPI (Uvicorn / Pydantic v2)
- **Knowledge Graph Store**: Neo4j 5.x Community
- **Vector & Relational Store**: PostgreSQL 16 + `pgvector`
- **Embeddings**: Local SentenceTransformers (`all-MiniLM-L6-v2`, 384 dimensions) or OpenAI (`text-embedding-3-small`, 1536 dimensions)
- **LLM Backends**: Multi-provider adapter (Google Gemini, Groq, OpenRouter, Anthropic)
- **Testing & Quality**: pytest, pytest-asyncio, ruff, mypy
- **Containerization**: Docker Compose

---

## 4. Architecture Overview

```
                          ┌──────────────────────────┐
                          │   Client / Enterprise    │
                          └─────────────┬────────────┘
                                        │ HTTP POST /api/v1/query
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
                          │ Answer Synthesis (LLM)   │
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
| [**RETRIEVAL_STRATEGY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/RETRIEVAL_STRATEGY.md) | Routing logic, safe Cypher templates (`T-01`–`T-09`), vector search, and ranking |
| [**API_SPEC.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/API_SPEC.md) | OpenAPI REST contract, request/response models, and error codes |
| [**SECURITY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/SECURITY.md) | Threat model, Cypher injection defenses, prompt isolation, and secrets |
| [**TASKS.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md) | Authoritative Phase 0–5 implementation backlog and task definitions |
| [**MEMORY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/MEMORY.md) | Persistent session memory, active status, confirmed invariants |
| [**TESTING.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TESTING.md) | Unit, integration, security, and mocked testing strategy |
| [**EVALUATION.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/EVALUATION.md) | Benchmark protocol, metrics, question sets, and evaluation results |
| [**CONFIGURATION.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/CONFIGURATION.md) | Environment variables, defaults, tuning knobs, and config schema |
| [**DEPLOYMENT.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DEPLOYMENT.md) | Docker Compose infrastructure, migrations, and local runbooks |
| [**DECISIONS.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DECISIONS.md) | Architecture Decision Records (ADRs) and unresolved decisions |
| [**OBSERVABILITY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/OBSERVABILITY.md) | Structured logging, metrics, tracing, and token cost tracking |
| [**TROUBLESHOOTING.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TROUBLESHOOTING.md) | Common failure modes, diagnostics, and recovery runbooks |
| [**CHANGELOG.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/CHANGELOG.md) | Chronological release and milestone history |

---

## 6. Quickstart & Local Demo Guide

### Prerequisites
- Docker Engine 24+ & Docker Compose v2+
- Python 3.11+ or 3.12
- LLM API key (Google Gemini, Groq, OpenRouter, or Anthropic)

### 1. Setup Environment
```bash
# Clone and enter workspace
git clone https://github.com/agile-duck22/Knowlegde_Graph_RAG.git
cd Knowlegde_Graph_RAG

# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure environment variables
copy .env.example .env
# Edit .env to set your LLM_PROVIDER and matching API key (e.g. GEMINI_API_KEY)
```

### 2. Launch Storage Infrastructure
```bash
docker compose up -d neo4j postgres
```

### 3. Initialize Schema & Ingest Reference Corpus
```bash
# Run database migrations (PostgreSQL DDL & Neo4j constraints)
python -m src.db.migrate

# Ingest the 145-document enterprise reference corpus
python -m src.ingest --corpus data/raw/corpus.jsonl
```

### 4. Run the API Server
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI is available at `http://localhost:8000/docs`.

### 5. Query the System
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which teams maintain services that write to the Customer Orders Database?"}'
```

---

## 7. Implementation Status & Phase Summary

- [x] **Phase 0 (Scaffolding)**: Pydantic settings, Docker Compose, mock test fixtures, CI workflow (`TASK-001`–`TASK-004`).
- [x] **Phase 1 (Ingestion & Graph)**: 145-doc enterprise corpus, semantic chunker, structured extractor, entity resolver, Neo4j idempotent batch writer (`TASK-101`–`TASK-107`).
- [x] **Phase 2 (Vector Search & Provenance)**: PostgreSQL + pgvector migrations (dual 384/1536 dim), local SentenceTransformers embedding generator, vector retriever, dual-store provenance check (`TASK-201`–`TASK-205`).
- [x] **Phase 3 (Routing & Safe Templates)**: Parameterized Cypher catalog (`T-01`–`T-09`), SafeGraphRetriever, intent classifier router, entity linker (`TASK-301`–`TASK-305`).
- [x] **Phase 4 (Synthesis & Hardening)**: Context normalizer, grounded synthesizer, citation guardrail, FastAPI endpoints, structured logging, multi-provider LLM adapter (`TASK-401`–`TASK-406`).
- [x] **Phase 5 (Evaluation & Verification)**: 50-question benchmark dataset, automated evaluation runner, security audit suite (`TASK-501`, `TASK-502`, `TASK-504`). Full benchmark executed (`run_hybrid_v3_threshold040.json`).

---

## 8. Benchmark Results (HYBRID v3)

*Empirical results from full 50-question benchmark artifact: `benchmark_results/run_hybrid_v3_threshold040.json` (similarity threshold = 0.40).*

| Query Category | Questions | Accuracy (%) | Mean Fact Coverage | Citation Precision | Citation Hallucination | Mean Latency (ms) | Latency p95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Single-Hop Direct Fact** | 10 | **100.0%** (10/10) | 1.00 | 0.70 | 0.0% | 29,508 ms | 99,511 ms |
| **2-Hop Relationship** | 15 | **80.0%** (12/15) | 0.90 | 0.40 | 0.0% | 14,801 ms | 31,233 ms |
| **3-Hop Dependency** | 10 | **70.0%** (7/10) | 0.77 | 0.45 | 0.0% | 7,651 ms | 17,211 ms |
| **Relationship Aggregation** | 10 | **60.0%** (6/10) | 0.83 | 0.40 | 0.0% | 10,617 ms | 18,349 ms |
| **Out-of-Scope / Refusal** | 5 | **100.0%** (5/5) | 1.00 | 1.00 | 0.0% | 1,489 ms | 2,781 ms |
| **Overall Summary** | **50** | **80.0%** (40/50) | **0.87** | **0.495** | **0.0%** | **13,456 ms** | **33,044 ms** |

> [!NOTE]
> - **Cost Metric**: Evaluator-reported cost of $0.00 reflects the zero-cost local embedding pipeline and free-tier LLM inference settings; it does not imply commercial third-party API calls are inherently zero cost.
> - **EVAL-028 Ingestion Gap**: In `data/raw/corpus.jsonl`, `service:cart-service` does not explicitly state a direct dependency on `service:checkout-service` or the PCI-isolated VPC. The synthesizer correctly refrains from hallucinating this connection, causing EVAL-028 to record lower coverage until an evidence-backed corpus update is applied.
> - **EVAL-038 Fix**: RDS PostgreSQL cluster traversal and normalizer expansion were validated via offline unit tests and template execution, connecting RDS instances to their parent cluster (`database:postgres-16`).

---
*Maintained under strict Architecture Decision Records. See [`docs/DECISIONS.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DECISIONS.md).*
