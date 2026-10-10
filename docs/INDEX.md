# Documentation Navigation Hub & Authority Index

This index defines the structure, reading order, and authority hierarchy of the project documentation.

---

## 1. Authoritative Hierarchy

When resolving conflicting specifications or making architectural choices, follow this hierarchy:

```
┌─────────────────────────────────────────────────────────┐
│ 1. SECURITY.md (Non-negotiable security & safety rules) │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│ 2. ARCHITECTURE.md & DECISIONS.md (System boundaries)   │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│ 3. DATA_MODEL.md & ONTOLOGY.md (Schemas & contracts)    │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│ 4. REQUIREMENTS.md & RETRIEVAL_STRATEGY.md (Functional) │
└────────────────────────────┬────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────┐
│ 5. TASKS.md & MEMORY.md (Execution state & backlog)     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Document Map

| Document | Primary Purpose | When an Agent Must Read | Depends On |
| :--- | :--- | :--- | :--- |
| [**PROJECT_OVERVIEW.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/PROJECT_OVERVIEW.md) | High-level vision, scope, personas, and completion criteria | Onboarding, scope validation | — |
| [**REQUIREMENTS.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/REQUIREMENTS.md) | Functional (`FR-xxx`) and non-functional (`NFR-xxx`) specs | Writing acceptance criteria, validating scope | `PROJECT_OVERVIEW.md` |
| [**ARCHITECTURE.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ARCHITECTURE.md) | Component diagrams, sequence flows, data pipelines | Designing modules, interfaces, and cross-store data flows | `REQUIREMENTS.md` |
| [**TASKS.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md) | Authoritative implementation backlog & task status | **Before and after every single implementation task** | `ARCHITECTURE.md`, `REQUIREMENTS.md` |
| [**SECURITY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/SECURITY.md) | Threat model, query safety, credential handling | Writing database queries, handling API inputs, logging | `ARCHITECTURE.md` |
| [**MEMORY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/MEMORY.md) | Persistent session memory, active blockers, verified progress | **At the start and end of every session** | `TASKS.md` |
| [**DATA_MODEL.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DATA_MODEL.md) | Neo4j property graph schema, PostgreSQL DDL, shared IDs | Implementing database schemas, migrations, or models | `ARCHITECTURE.md`, `ONTOLOGY.md` |
| [**ONTOLOGY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ONTOLOGY.md) | Target domain entity types, relationships, resolution rules | Implementing LLM extraction, normalization, resolution | `PROJECT_OVERVIEW.md` |
| [**RETRIEVAL_STRATEGY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/RETRIEVAL_STRATEGY.md) | Intent routing, Cypher templates, vector search, ranking | Implementing query routing, Cypher queries, vector retrieval | `DATA_MODEL.md`, `ARCHITECTURE.md` |
| [**API_SPEC.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/API_SPEC.md) | OpenAPI REST specs, request/response models, errors | Implementing FastAPI routes, schemas, and error handlers | `RETRIEVAL_STRATEGY.md` |
| [**TESTING.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TESTING.md) | Test hierarchy, mocking rules, test cases per component | Writing tests, running CI, validating tasks | `REQUIREMENTS.md`, `API_SPEC.md` |
| [**EVALUATION.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/EVALUATION.md) | Benchmark protocol, question dataset format, metrics | Implementing benchmark scripts, running evaluation | `RETRIEVAL_STRATEGY.md` |
| [**OBSERVABILITY.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/OBSERVABILITY.md) | Structured logging, metrics, latency/cost tracking | Adding instrumentation, logging, telemetry | `ARCHITECTURE.md`, `SECURITY.md` |
| [**DEPLOYMENT.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DEPLOYMENT.md) | Docker Compose services, startup runbooks, migrations | Setting up local infrastructure or deployment scripts | `ARCHITECTURE.md`, `CONFIGURATION.md` |
| [**CONFIGURATION.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/CONFIGURATION.md) | Environment variables, defaults, tuning knobs | Accessing settings, managing environment configs | `ARCHITECTURE.md` |
| [**DECISIONS.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DECISIONS.md) | Architectural Decision Records (ADRs) & open questions | Proposing or altering architectural patterns | `ARCHITECTURE.md` |
| [**TROUBLESHOOTING.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TROUBLESHOOTING.md) | Diagnostic runbooks for database, LLM, retrieval issues | Debugging errors, resolving test failures | `ARCHITECTURE.md`, `DEPLOYMENT.md` |
| [**CHANGELOG.md**](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/CHANGELOG.md) | Chronological release and milestone history | Completing milestone phases | `TASKS.md` |

---

## 3. Recommended Reading Sequences

### For New Coding Agents Onboarding
1. [`docs/INDEX.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/INDEX.md)
2. [`docs/MEMORY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/MEMORY.md)
3. [`docs/ARCHITECTURE.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ARCHITECTURE.md)
4. [`docs/SECURITY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/SECURITY.md)
5. [`docs/TASKS.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TASKS.md)

### For Ingestion & Extraction Pipeline Work (Phase 1)
1. [`docs/ONTOLOGY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/ONTOLOGY.md)
2. [`docs/DATA_MODEL.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/DATA_MODEL.md)
3. [`docs/SECURITY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/SECURITY.md)

### For Retrieval & Routing Work (Phases 2, 3, 4)
1. [`docs/RETRIEVAL_STRATEGY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/RETRIEVAL_STRATEGY.md)
2. [`docs/API_SPEC.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/API_SPEC.md)
3. [`docs/TESTING.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/TESTING.md)

### For Evaluation & Benchmarking (Phase 5)
1. [`docs/EVALUATION.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/EVALUATION.md)
2. [`docs/OBSERVABILITY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/OBSERVABILITY.md)
