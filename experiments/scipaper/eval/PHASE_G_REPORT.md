# Phase G Report: Single-Paper Graph Integration, Safety Verification, and Migration Readiness

**Date**: 2026-10-23  
**Domain**: Scientific Papers (Isolated Experiment: `experiments/scipaper/`)  
**Status**: COMPLETE (All Gates A through E Passed)  
**Safety & Compliance**: 100% Offline, Zero Remote API Calls, Zero Model Downloads, Zero Destruction of Baseline Data  
**Integration Scope**: Exactly ONE paper (`paper-arxiv-2411.18583`). The other 119 papers remain unmodified and pending migration approval.

---

## 1. Executive Summary

Phase G successfully completed a safe, verified single-paper graph integration into the isolated scientific experiment environment (PostgreSQL port `5433` and Neo4j Bolt port `7688`).

All required sequential gates passed:
- **Gate A (Schema Compatibility Audit)**: Verified that `IsolatedScientificNeo4jWriter` supports all 8 scientific entity types and 7 relationship types with parameterized Cypher templates, and verified that existing Cypher retrieval templates (`SCI-T01` to `SCI-T07`) and Neo4j constraints (`unique_entity_id`) are compatible.
- **Gate B (Structural Dry-Run)**: Generated a machine-readable dry-run migration manifest for `paper-arxiv-2411.18583` showing 7 proposed new nodes and 5 proposed new relationships. Verified zero writes to PostgreSQL or Neo4j.
- **Gate C (Single-Paper Integration & Idempotency)**: Integrated `paper-arxiv-2411.18583` into the graph. Observed delta matched the manifest exactly (+7 nodes, +5 relationships, +10 entity links). A second identical run was executed, verifying 100% idempotency (+0 nodes, +0 relationships, +0 entity links).
- **Gate D (Retrieval Regression Benchmark)**: Ran the 50-question benchmark across Vector, Graph, and Hybrid RAG. Achieved identical MRR and Hit Rates (Hybrid MRR 0.5619, HR@10 0.9000) with zero query regressions.
- **Gate E (Migration Readiness)**: Authored a comprehensive full-corpus migration plan (`PROPOSED_MIGRATION_PLAN.md`). Stopped execution without migrating the remaining 119 papers.

---

## 2. Gate-by-Gate Verification Audit

### 2.1 Initial Baseline Recording
Baseline database counts were captured at `2026-10-23T10:03:15Z` and saved to `experiments/scipaper/eval/results/baseline_db_counts.json`:
- **PostgreSQL**: 120 documents, 474 chunks, 3,519 entity links, 474 `embedding_384` vectors, 0 1536-dim vectors.
- **Neo4j**: 722 nodes (120 `Paper`, 597 `Author`, 5 `Method`), 3,045 relationships (2,631 `AUTHORED`, 414 `USES_METHOD`).

### 2.2 Gate A — Schema Compatibility Audit
- **Entities**: `IsolatedScientificNeo4jWriter` supports `Paper`, `Author`, `Model`, `Dataset`, `Metric`, `Task`, `Method`, `ResearchOrganization`. Each node is merged with primary label `:Entity` and specific secondary label (e.g. `:Entity:Model`).
- **Relationships**: Parameterized templates support `AUTHORED`, `PROPOSES`, `EVALUATED_ON`, `REPORTS_METRIC`, `USES_METHOD`, `AFFILIATED_WITH`, `CITES`.
- **Constraint Compatibility**: Neo4j constraint `CONSTRAINT unique_entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE` covers all 8 types under the primary `:Entity` label.
- **Retrieval Compatibility**: `SCI-T07` matches `(p:Entity:Paper)-[r]-(neighbor:Entity)` and returns all neighbor types dynamically. `SCI-T01` to `SCI-T06` target `:Paper`, `:Author`, and `:Method`. No retrieval queries fail on the new types.
- **Result**: **PASS**.

### 2.3 Gate B — Genuine Dry-Run Implementation & Verification
- Implemented `run_dry_run_analysis()` in `experiments/scipaper/scripts/run_phase_g_integration.py`.
- Evaluated `paper-arxiv-2411.18583`:
  - 4 text chunks verified in PostgreSQL.
  - Extracted 13 unique entities (6 already in graph, 7 new proposed nodes).
  - Extracted 23 relationship instances (18 already in graph, 5 new proposed edges).
- Produced `single_paper_dry_run_manifest.json`.
- Tested in `experiments/scipaper/tests/test_phase_g_dry_run.py` to assert zero calls to write methods and zero mutation Cypher queries.
- Verified database counts remained unchanged.
- **Result**: **PASS**.

### 2.4 Gate C — Single-Paper Integration & Idempotency
- Executed `run_phase_g.py --execute` targeting `paper-arxiv-2411.18583`.
- **Post-Integration Verification**:
  - PostgreSQL entity links: 3,519 $\rightarrow$ 3,529 (+10 links added for the newly extracted concepts).
  - PostgreSQL documents and chunks: exactly 120 documents, 474 chunks, 474 `embedding_384` vectors (100% preserved).
  - Neo4j nodes: 722 $\rightarrow$ 729 (+7 nodes: 1 `Method` [nlp], 1 `Dataset` [scitldr], 1 `Metric` [rouge], 2 `Model` [gpt-3.5, t5], 1 `ResearchOrganization` [arxiv], 1 `Task` [literature-review-generation]).
  - Neo4j relationships: 3,045 $\rightarrow$ 3,050 (+5 edges: 2 `USES_METHOD` [nlp], 1 `EVALUATED_ON` [scitldr], 2 `REPORTS_METRIC` [rouge]).
  - **Delta Match**: Observed delta (+7 nodes, +5 edges) matched manifest predictions exactly.
- **Idempotency Pass**: A second identical integration run was executed on the same paper:
  - 0 new nodes added.
  - 0 new relationships added.
  - 0 new entity links added.
  - Total Neo4j nodes remained 729; total relationships remained 3,050.
- Result saved to `experiments/scipaper/eval/results/single_paper_integration_verification.json`.
- **Result**: **PASS**.

### 2.5 Gate D — Retrieval Regression Test
The Phase E comparative retrieval benchmark was executed across all 50 questions post-integration:

| Retriever | Reference Phase E MRR | Post-Gate C MRR | Reference HR@10 | Post-Gate C HR@10 | Reference Rec@10 | Post-Gate C Rec@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vector-only** | 0.4475 | **0.4475** | 0.7000 | **0.7000** | 0.3583 | **0.3583** |
| **Graph-only** | 0.5875 | **0.5875** | 0.6250 | **0.6250** | 0.2729 | **0.2896** |
| **Hybrid RRF** | 0.5619 | **0.5619** | 0.9000 | **0.9000** | 0.4854 | **0.4896** |

- **Findings**:
  - Zero regression across all 50 benchmark questions.
  - Graph-only Recall@10 improved slightly (+0.0167) and Hybrid Recall@10 improved (+0.0042) because graph traversal on `paper-arxiv-2411.18583` now retrieves connected methods and datasets.
- Result saved to `experiments/scipaper/eval/results/retrieval_regression_comparison.json`.
- **Result**: **PASS**.

### 2.6 Gate E — Full-Corpus Migration Plan
- Authored full proposal in `experiments/scipaper/eval/PROPOSED_MIGRATION_PLAN.md`.
- Outlined batching (6 batches of 20 docs), transaction boundaries, failure recovery, idempotency guarantees, and estimated runtime ($\approx 42$ seconds).
- **Stopped execution without touching the remaining 119 papers.**
- **Result**: **PASS**.

---

## 3. Database Count Summary

| Target Store | Table / Label | Pre-Phase G Baseline | Post-Phase G Final | Delta | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **PostgreSQL** | `documents` | 120 | 120 | 0 | Preserved |
| | `chunks` | 474 | 474 | 0 | Preserved |
| | `chunks (embedding_384)` | 474 | 474 | 0 | Preserved |
| | `chunks (embedding 1536)` | 0 | 0 | 0 | Preserved |
| | `entity_links` | 3,519 | 3,529 | +10 | Expected new mentions |
| **Neo4j** | Total Nodes | 722 | 729 | +7 | Expected delta |
| | - `Paper` | 120 | 120 | 0 | Preserved |
| | - `Author` | 597 | 597 | 0 | Preserved |
| | - `Method` | 5 | 6 | +1 | `method:nlp` |
| | - `Model` | 0 | 2 | +2 | `model:gpt-3.5`, `model:t5` |
| | - `Dataset` | 0 | 1 | +1 | `dataset:scitldr` |
| | - `Metric` | 0 | 1 | +1 | `metric:rouge` |
| | - `Task` | 0 | 1 | +1 | `task:literature-review...` |
| | - `ResearchOrganization` | 0 | 1 | +1 | `org:arxiv` |
| | Total Relationships | 3,045 | 3,050 | +5 | Expected delta |
| | - `AUTHORED` | 2,631 | 2,631 | 0 | Preserved |
| | - `USES_METHOD` | 414 | 416 | +2 | +2 chunk mentions |
| | - `EVALUATED_ON` | 0 | 1 | +1 | `dataset:scitldr` |
| | - `REPORTS_METRIC` | 0 | 2 | +2 | `metric:rouge` (2 chunks) |

---

## 4. Quality & Verification Gates

1. **Unit & Integration Tests**:
   - `pytest experiments/scipaper/tests/ -v`: **65 passed in 18.36s** (0 failed, 0 errors).
2. **Linting**:
   - `ruff check experiments/scipaper`: **All checks passed**.
3. **Formatting**:
   - `ruff format --check experiments/scipaper`: **46 files verified formatted**.
4. **Type Checking**:
   - `mypy experiments/scipaper`: **Success: no issues found in 39 source files**.

---

## 5. Artifacts Created & Updated

1. `experiments/scipaper/eval/results/baseline_db_counts.json` (Pre-run database baseline)
2. `experiments/scipaper/eval/results/single_paper_dry_run_manifest.json` (Gate B dry-run manifest)
3. `experiments/scipaper/eval/results/single_paper_integration_verification.json` (Gate C integration verification)
4. `experiments/scipaper/eval/results/retrieval_regression_comparison.json` (Gate D benchmark regression results)
5. `experiments/scipaper/eval/PROPOSED_MIGRATION_PLAN.md` (Full migration plan)
6. `experiments/scipaper/scripts/run_phase_g_integration.py` (Integration & dry-run implementation)
7. `experiments/scipaper/scripts/run_phase_g.py` (CLI entry point)
8. `experiments/scipaper/tests/test_phase_g_dry_run.py` (Structural dry-run test)
9. `experiments/scipaper/eval/PHASE_G_REPORT.md` (This report)

---

## 6. Stop Condition Confirmation

Phase G is fully concluded. Exactly 1 paper was integrated and verified. Zero writes were made to the other 119 papers, and no further migration will occur without your explicit authorization.
