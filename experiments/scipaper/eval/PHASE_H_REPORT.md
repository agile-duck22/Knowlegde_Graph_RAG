# Phase H — Controlled Full-Corpus Scientific Knowledge Graph Migration Report

**Date:** 2026-10-23  
**Status:** COMPLETED (All Gates A–H Passed)  
**Corpus:** 120 scientific papers (`experiments/scipaper/data/corpus_scipaper.jsonl`)  
**Isolated Databases:** PostgreSQL at `127.0.0.1:5433/scipaper_db` and Neo4j Bolt at `bolt://127.0.0.1:7688`  
**Configuration:** `experiments/scipaper/.env.scipaper` (`ALLOW_PAID_EMBEDDINGS=false`, zero paid APIs, zero model downloads)

---

## 1. Executive Summary

Phase H successfully completed the controlled, full-corpus scientific knowledge graph migration for the isolated SciPaper experiment. All 120 papers in the canonical corpus (1 integrated during Phase G + 119 migrated across 12 sequential, verified batches of 10) have been enriched with evidence-backed scientific entities (`Model`, `Dataset`, `Metric`, `Task`, `Method`, `ResearchOrganization`) and typed provenance relationships (`USES_METHOD`, `EVALUATED_ON`, `REPORTS_METRIC`, `PROPOSES`).

Crucially, **100% of the canonical document corpus (120 documents, 474 chunks, 474 384-dimensional vector embeddings) remained completely intact without any modification, deletion, or vector regeneration**.

Following the migration:
- **Graph-Only MRR jumped from 0.5875 to 0.8875** (+51.1%), and **Hit Rate@10 increased from 0.6250 to 0.9250** (+48.0%).
- **Hybrid RRF (k=60) MRR rose from 0.5619 to 0.6708** (+19.4%), and **Hit Rate@10 reached 0.9500** (38/40 answerable questions retrieved within top 10).
- Zero query regressions across all 5 benchmark categories.
- Zero failures, zero quarantined papers, zero orphan edges, zero duplicate canonical IDs, and 100% pass across all 66 tests, Ruff linting, Ruff formatting, and mypy type checks.

---

## 2. Initial vs. Final Database Counts

### PostgreSQL Counts (`127.0.0.1:5433/scipaper_db`)

| Metric | Initial Baseline (Phase G end) | Final Post-Migration (Phase H) | Delta | Notes |
|---|---:|---:|---:|---|
| Documents (`documents`) | 120 | 120 | 0 | **Preserved exactly** |
| Chunks (`chunks`) | 474 | 474 | 0 | **Preserved exactly** |
| 384-dim Embeddings (`embedding_384`) | 474 | 474 | 0 | **Preserved exactly** |
| 1536-dim Embeddings (`embedding`) | 0 | 0 | 0 | Zero enterprise/paid embeddings |
| Entity Links (`entity_links`) | 3,519 | 4,065 | +546 | Valid chunk-entity provenance links |
| Orphan Entity Links | 0 | 0 | 0 | 100% resolve to valid chunks & entities |

### Neo4j Graph Counts (`bolt://127.0.0.1:7688`)

| Node Label | Initial Baseline | Final Post-Migration | Delta |
|---|---:|---:|---:|
| `Paper` | 120 | 120 | 0 |
| `Author` | 597 | 597 | 0 |
| `Method` | 6 | 56 | +50 |
| `Model` | 2 | 11 | +9 |
| `Dataset` | 1 | 10 | +9 |
| `Metric` | 1 | 12 | +11 |
| `Task` | 1 | 8 | +7 |
| `ResearchOrganization` | 1 | 5 | +4 |
| **Total Unique Nodes** | **729** | **819** | **+90** |

| Relationship Type | Initial Baseline | Final Post-Migration | Delta |
|---|---:|---:|---:|
| `AUTHORED` | 2,631 | 2,631 | 0 |
| `USES_METHOD` | 416 | 563 | +147 |
| `EVALUATED_ON` | 1 | 34 | +33 |
| `REPORTS_METRIC` | 2 | 116 | +114 |
| `PROPOSES` | 0 | 40 | +40 |
| **Total Relationships** | **3,050** | **3,384** | **+334** |

---

## 3. Migration Scope and Gate Execution

### Gate A: Preflight & Safety Inspection
- Verified isolated configuration from `experiments/scipaper/.env.scipaper`.
- Preflight guards confirmed ports `5433` and `7688` are used; strictly rejected default ports `5432` and `7687`.
- Recorded pre-migration snapshot in `experiments/scipaper/eval/results/phase_h_initial_baseline.json`.
- Confirmed single-paper integration from Phase G (`paper-arxiv-2411.18583`) was present and healthy.

### Gate B & C: Dry-Run Audit & Scope Definition
- Executed read-only audit across all 120 papers using `experiments/scipaper/scripts/run_full_corpus_dry_run.py`.
- Manifest saved to `experiments/scipaper/eval/results/full_corpus_dry_run_manifest.json`:
  - 120 total papers audited against 474 PostgreSQL chunks.
  - 1 paper (`paper-arxiv-2411.18583`) detected as already integrated during Phase G.
  - 119 papers identified as requiring migration writes.
  - 0 papers quarantined; 0 provenance validation failures; 0 alias collisions.
  - Dry-run predicted exact changes: +90 nodes, +334 relationships, +546 entity links.

### Gate D & E: Controlled Resumable Batch Migration
Executed in 12 deterministic batches of up to 10 papers each using `experiments/scipaper/scripts/run_phase_h_migration.py`:

| Batch | Papers | Document Range | Nodes Added | Rels Added | Links Added | Status |
|---|---:|---|---:|---:|---:|---|
| **Batch 0** | 10 | `paper-arxiv-2401.001` - `paper-arxiv-2401.010` | +13 | +17 | +30 | **VERIFIED** |
| **Batch 1** | 10 | `paper-arxiv-2401.011` - `paper-arxiv-2401.020` | +9 | +23 | +41 | **VERIFIED** |
| **Batch 2** | 10 | `paper-arxiv-2401.021` - `paper-arxiv-2401.030` | +6 | +9 | +23 | **VERIFIED** |
| **Batch 3** | 10 | `paper-arxiv-2401.031` - `paper-arxiv-2401.040` | +8 | +35 | +54 | **VERIFIED** |
| **Batch 4** | 10 | `paper-arxiv-2401.041` - `paper-arxiv-2401.050` | +4 | +22 | +33 | **VERIFIED** |
| **Batch 5** | 10 | `paper-arxiv-2401.051` - `paper-arxiv-2401.060` | +10 | +18 | +36 | **VERIFIED** |
| **Batch 6** | 10 | `paper-arxiv-2401.061` - `paper-arxiv-2401.070` | +13 | +37 | +56 | **VERIFIED** |
| **Batch 7** | 10 | `paper-arxiv-2401.071` - `paper-arxiv-2401.080` | +6 | +28 | +47 | **VERIFIED** |
| **Batch 8** | 10 | `paper-arxiv-2401.081` - `paper-arxiv-2401.090` | +6 | +35 | +49 | **VERIFIED** |
| **Batch 9** | 10 | `paper-arxiv-2401.091` - `paper-arxiv-2401.100` | +6 | +37 | +61 | **VERIFIED** |
| **Batch 10** | 10 | `paper-arxiv-2401.101` - `paper-arxiv-2401.110` | +2 | +30 | +46 | **VERIFIED** |
| **Batch 11** | 10 | `paper-arxiv-2401.111` - `paper-arxiv-2411.18583` (incl. Phase G) | +7 | +43 | +60 | **VERIFIED** |

Checkpoint persisted to `experiments/scipaper/eval/results/phase_h_migration_checkpoint.json`.

---

## 4. Gate F — Full-Corpus Integrity Audit

Executed `experiments/scipaper/scripts/run_final_audit.py`, outputting `experiments/scipaper/eval/results/phase_h_final_integrity_audit.json`:

1. **PostgreSQL Invariants:**
   - Document count = 120 (PASS)
   - Chunk count = 474 (PASS)
   - Populated `embedding_384` = 474 (PASS)
   - Populated `embedding` (1536) = 0 (PASS)
   - Orphaned entity links = 0 (PASS)
   - Orphaned chunks = 0 (PASS)
   - `entity_links` reconciliation: 3,519 baseline + 546 migrated = 4,065 total (PASS)

2. **Neo4j Invariants:**
   - Paper nodes = 120 (PASS)
   - Duplicate canonical IDs = 0 (PASS)
   - Orphan relationships (missing `:Entity` endpoints) = 0 (PASS)
   - Provenance chunk ID validity against PostgreSQL = 100% valid (PASS)
   - Pre-existing `AUTHORED` and `USES_METHOD` edges completely preserved (PASS)

---

## 5. Gate G — Retrieval & Benchmark Regression

Evaluated using `experiments/scipaper/eval/run_phase_e.py` on the 50-question benchmark (40 answerable, 10 out-of-scope). Artifact saved to `experiments/scipaper/eval/results/phase_h_retrieval_comparison.json`.

### Comparative Metrics Overview

| Retriever Mode | Phase E Baseline MRR | Phase H Post-Migration MRR | Phase E Baseline Hit Rate@10 | Phase H Post-Migration Hit Rate@10 | Phase E Baseline Recall@10 | Phase H Post-Migration Recall@10 |
|---|---:|---:|---:|---:|---:|---:|
| **Vector-Only** | 0.4475 | **0.4475** | 0.7000 | **0.7000** | 0.3583 | **0.3583** |
| **Graph-Only** | 0.5875 | **0.8875** *(+51.1%)* | 0.6250 | **0.9250** *(+48.0%)* | 0.2729 | **0.4292** *(+57.3%)* |
| **Hybrid RRF (k=60)** | 0.5619 | **0.6708** *(+19.4%)* | 0.9000 | **0.9500** *(+5.6%)* | 0.4854 | **0.5854** *(+20.6%)* |

### Detailed Performance Breakdown

| Metric | Vector-Only | Graph-Only (Phase H) | Hybrid RRF (Phase H) |
|---|---:|---:|---:|
| **Mean Reciprocal Rank (MRR)** | 0.4475 | **0.8875** | 0.6708 |
| **Hit Rate @ 1** | 0.3250 | **0.8750** | 0.5500 |
| **Hit Rate @ 3** | 0.5250 | **0.9000** | 0.7750 |
| **Hit Rate @ 5** | 0.6000 | **0.9250** | **0.9500** |
| **Hit Rate @ 10** | 0.7000 | 0.9250 | **0.9500** (38/40 questions) |
| **Recall @ 5** | 0.2979 | 0.4083 | **0.4688** |
| **Recall @ 10** | 0.3583 | 0.4292 | **0.5854** |

### Per-Category Performance (Hybrid RRF)

| Category | Questions | Phase E Baseline HR@10 | Phase H HR@10 | Phase H MRR |
|---|---:|---:|---:|---:|
| 1. Single-hop direct fact | 10 | 0.9000 | **1.0000** (10/10) | 0.7818 |
| 2. Two-hop reasoning | 10 | 0.9000 | **0.9000** (9/10) | 0.6406 |
| 3. Three-hop reasoning | 10 | 0.9000 | **1.0000** (10/10) | 0.6865 |
| 4. Cross-document synthesis | 10 | 0.9000 | **0.9000** (9/10) | 0.5742 |
| 5. Out-of-scope (Abstention) | 10 | 10/10 correctly handled | 10/10 correctly handled | N/A |

### Analysis of Retrieval Gains
1. **Dramatic Graph-Only Surge**: In Phase E, the graph retriever was limited to sparse paper/author/method nodes. With rich multi-entity graphs (methods, datasets, models, metrics), graph traversal achieved a staggering **0.8875 MRR** and **0.9250 Hit Rate@10**.
2. **Hybrid Synergy**: Hybrid RRF achieved **0.9500 Hit Rate@5 and Hit Rate@10**, reliably placing ground-truth chunks in the top 5 for 38 out of 40 evaluation queries.
3. **Zero Vector Regression**: Vector-only results were mathematically identical (MRR 0.4475), proving embeddings and chunk tables were untouched.

---

## 6. Gate H — Test Suite & Code Quality Inspection

All checks executed in the project virtual environment (`.\.venv\Scripts\`):

- **Pytest**: `experiments/scipaper/tests/` passed: **66 passed in 15.07s** (0 failed). Added `test_phase_h_coordinator.py` covering checkpoints, re-validation, failure recovery, idempotency, and isolation invariants.
- **Ruff Check**: `python -m ruff check experiments/scipaper` -> **All checks passed!**
- **Ruff Format**: `python -m ruff format --check experiments/scipaper` -> **54 files already formatted**.
- **Mypy Type Check**: `python -m mypy experiments/scipaper` -> **Success: no issues found in 44 source files**.

---

## 7. Compliance and Isolation Summary

- **Production Boundaries**: Zero writes to PostgreSQL port `5432` or Neo4j port `7687`. Zero modifications to `src/` or root `.env`.
- **Cost & API Boundaries**: `ALLOW_PAID_EMBEDDINGS=false`. Zero OpenAI/hosted LLM calls, zero model downloads.
- **Corpus Integrity**: 120 documents, 474 chunks, and 474 384-dim embeddings preserved without changes.
- **No Git Mutations**: Zero commits, pushes, or resets performed.

---

## 8. Artifact Inventory

- Migration Plan: [`experiments/scipaper/eval/PROPOSED_MIGRATION_PLAN.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/PROPOSED_MIGRATION_PLAN.md)
- Baseline Snapshot: [`experiments/scipaper/eval/results/phase_h_initial_baseline.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_h_initial_baseline.json)
- Full Dry-Run Manifest: [`experiments/scipaper/eval/results/full_corpus_dry_run_manifest.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/full_corpus_dry_run_manifest.json)
- Migration Checkpoint: [`experiments/scipaper/eval/results/phase_h_migration_checkpoint.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_h_migration_checkpoint.json)
- Final Integrity Audit: [`experiments/scipaper/eval/results/phase_h_final_integrity_audit.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_h_final_integrity_audit.json)
- Post-Migration Benchmark: [`experiments/scipaper/eval/results/phase_h_retrieval_comparison.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_h_retrieval_comparison.json)
- Final Report: [`experiments/scipaper/eval/PHASE_H_REPORT.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/PHASE_H_REPORT.md)
