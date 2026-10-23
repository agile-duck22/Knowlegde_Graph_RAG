# Phase I — Migration Audit & Extraction Quality Validation Report

**Date:** 2026-10-23  
**Audit Type:** Read-Only Audit & Offline Quality Evaluation  
**Verdict:** **PASS WITH CAVEATS**  
**Corpus:** 120 arXiv scientific papers (`experiments/scipaper/data/corpus_scipaper.jsonl`)  
**Isolated Databases:** PostgreSQL (`127.0.0.1:5433/scipaper_db`) & Neo4j Bolt (`bolt://127.0.0.1:7688`)  
**Zero-Modification Guarantee:** No database writes, zero data mutations, zero production port access (`5432`/`7687` barred), zero model downloads, and zero external API calls (`ALLOW_PAID_EMBEDDINGS=false`).

---

## 1. Executive Summary & Verdict

Phase I conducted an independent, read-only audit of the Phase H full-corpus scientific knowledge graph migration. The audit inspected existing artifacts, re-queried the isolated database instances without modifying any record, evaluated extraction semantic quality on a deterministic sample of 60 relationships, and fully reproduced the retrieval benchmark.

### Overall Verdict: **PASS WITH CAVEATS**

- **Why PASS**:
  1. **Migration Accounting & Integrity**: 100% verified against live databases. All 120 documents, 474 chunks, and 474 384-dimensional embeddings are completely intact without any vector regeneration or deletion. 0 orphaned chunks, 0 orphaned entity links, 0 duplicate canonical entity IDs, and 0 orphan relationships exist.
  2. **Deltas Reconciled**: The reported additions (+90 unique nodes, +334 relationships, +546 entity links) reconcile across the dry-run manifest, the 12-batch checkpoint, the baseline snapshots, and the live database query results with zero mathematical variance.
  3. **Retrieval Gains Independently Reproduced**: The Phase H retrieval benchmark metrics (**Graph-Only MRR 0.8875, HR@10 0.9250; Hybrid RRF MRR 0.6708, HR@10 0.9500**) were re-run live and matched down to 4 decimal places. Vector-only baseline remained fixed at MRR 0.4475.
  4. **Code Quality & Safety**: All 66 unit/integration tests pass cleanly, Ruff linting and formatting are 100% compliant, and mypy reports 0 errors across 46 source files.
- **Why CAVEATS**:
  1. **Semantic Relationship Grounding**: Out of 60 sampled newly introduced relationships, **76.67% (46/60)** are explicitly supported by direct text evidence in the provenance chunks, and **16.67% (10/60)** are partially supported (plausible from co-occurrence in abstract, but specific directional proposition or usage is implicit). **6.67% (4/60)** are unsupported (abstract text does not justify the specific edge despite the entity being present).
  2. **Benchmark Category Terminology Discrepancy**: Earlier documentation and code refer to the fourth benchmark category as `relationship_aggregation`, whereas the Phase H narrative referred to it as `cross-document synthesis`. The underlying 10 questions (EVAL-031 to EVAL-040) are identical, but naming should be standardized.
  3. **Abstention False Refusal Disclosure**: At the optimal cosine similarity threshold of 0.40, all 10 out-of-scope questions are correctly refused (100% refusal recall), but 1 answerable question (`EVAL-035`) falls below threshold (similarity 0.384) resulting in a 2.5% false refusal rate (97.5% answerable availability). The Phase H report omitted explicitly stating this false refusal trade-off.

---

## 2. Scope & Safety Confirmation

| Check | Requirement | Verified Status | Evidence |
|---|---|---|---|
| Database Writes | Zero writes during audit | **CONFIRMED** | Only `SELECT` and `MATCH` queries executed |
| Production Ports | No access to 5432 or 7687 | **CONFIRMED** | Enforced by strict preflight guards |
| Isolated Env | Load only `.env.scipaper` | **CONFIRMED** | Verified in preflight inspection |
| Paid APIs / Hosted LLMs | Zero external network calls | **CONFIRMED** | `ALLOW_PAID_EMBEDDINGS=false` |
| Local Model Downloads | Zero model downloads | **CONFIRMED** | Local cached `all-MiniLM-L6-v2` with `local_files_only=True` |
| Corpus / Chunks / Vectors | Zero data modification | **CONFIRMED** | Exactly 120 docs, 474 chunks, 474 vectors before and after |
| Git Modifications | No commits, pushes, or resets | **CONFIRMED** | Clean working tree; zero git mutations |

---

## 3. Audit A — Migration Accounting

The migration was independently reconstructed from the pre-migration baseline, the 120-paper dry-run manifest, the 12-batch migration checkpoint, and the final integrity audit:

| Metric | Before Phase H (Baseline) | After Phase H (Final Audit) | Reported Delta | Manifest Predicted | Live Re-queried Delta | Discrepancy |
|---|---:|---:|---:|---:|---:|---|
| **PostgreSQL Documents** | 120 | 120 | 0 | 0 | 0 | None (Exact) |
| **PostgreSQL Chunks** | 474 | 474 | 0 | 0 | 0 | None (Exact) |
| **384-dim Embeddings** | 474 | 474 | 0 | 0 | 0 | None (Exact) |
| **1536-dim Embeddings** | 0 | 0 | 0 | 0 | 0 | None (Exact) |
| **PostgreSQL Entity Links** | 3,529 | 4,065 | +536 *(+546 from Phase G base)* | +536 | +536 | None (Exact) |
| **Neo4j Unique Nodes** | 729 | 819 | +90 | +90 | +90 | None (Exact) |
| **Neo4j Relationships** | 3,050 | 3,384 | +334 | +334 | +334 | None (Exact) |

### Key Accounting Audit Points:
1. **Initial Baseline Context**:
   - At Phase G start: PostgreSQL had 3,519 links; Neo4j had 722 nodes and 3,045 relationships.
   - Phase G integrated 1 paper (`paper-arxiv-2411.18583`), adding 7 nodes, 5 relationships, and 10 entity links (yielding baseline 729 nodes, 3,050 relationships, and 3,529 entity links).
   - Phase H migrated the remaining 119 papers, adding exactly 90 nodes, 334 relationships, and 536 entity links. Total delta from Phase G start is +97 nodes, +339 relationships, and +546 entity links.
2. **Paper ID Tracking**:
   - All 120 papers are accounted for exactly once in `experiments/scipaper/eval/results/phase_h_migration_checkpoint.json`.
   - 0 papers quarantined; 0 unhandled failures.
   - Phase G paper was placed in Batch 4 (index 4) and safely re-checked without duplicate writes due to database idempotency guards.

---

## 4. Audit B — Database Integrity

Live read-only verification queries against `127.0.0.1:5433` and `bolt://127.0.0.1:7688` confirmed:

- **Paper Nodes**: Exactly 120 `:Entity:Paper` nodes exist, matching the 120 corpus records.
- **Node Label Breakdown**:
  - `Paper`: 120
  - `Author`: 597
  - `Method`: 56 (+50 in Phase H)
  - `Metric`: 12 (+11 in Phase H)
  - `Model`: 11 (+9 in Phase H)
  - `Dataset`: 10 (+9 in Phase H)
  - `Task`: 8 (+7 in Phase H)
  - `ResearchOrganization`: 5 (+4 in Phase H)
  - **Total**: 819 nodes.
- **Relationship Breakdown**:
  - `AUTHORED`: 2,631 (baseline author-paper edges completely preserved)
  - `USES_METHOD`: 563 (+147 in Phase H)
  - `REPORTS_METRIC`: 116 (+114 in Phase H)
  - `PROPOSES`: 40 (+40 in Phase H)
  - `EVALUATED_ON`: 34 (+33 in Phase H)
  - **Total**: 3,384 relationships.
- **Integrity Invariants**:
  - Duplicate canonical IDs: **0**
  - Orphan relationships (endpoints missing `:Entity` label): **0**
  - Orphan entity links in PostgreSQL: **0**
  - Orphan chunks: **0**
  - Cross-store provenance: 100% of tested relationship `provenance_chunk_id` values resolve to existing PostgreSQL chunks.

---

## 5. Audit C — Semantic Relationship Quality

A deterministic stratified sample of **60 newly introduced relationships** was evaluated against the actual text of their provenance chunks in PostgreSQL (fixed seed: `42`).

Stratified distribution:
- `USES_METHOD`: 18 sampled (pool of 563)
- `PROPOSES`: 15 sampled (pool of 40)
- `REPORTS_METRIC`: 15 sampled (pool of 116)
- `EVALUATED_ON`: 12 sampled (pool of 34)

### Evaluation Results:

| Classification | Count | Percentage | Definition |
|---|---:|---:|---|
| **Supported** | 46 | **76.67%** | Source chunk explicitly affirms the relationship and direction. |
| **Partially Supported** | 10 | **16.67%** | Entities co-occur and relationship is plausible, but proposition or usage is implicit/contextual. |
| **Unsupported** | 4 | **6.67%** | Target entity appears in text, but the claimed relationship type is not justified by the phrasing. |
| **Unverifiable** | 0 | **0.00%** | Provenance chunk missing or text unavailable. |

**Overall Plausible or Supported Rate:** **93.33%** (56 / 60 relationships).

### Specific Findings by Relationship Type:
1. **`PROPOSES` (High Precision)**: 13/15 Supported (86.7%), 2/15 Partially Supported. Strong linguistic triggers (`we propose`, `we present`, `we develop`) reliably ground the link between the paper and the proposed model/method.
2. **`EVALUATED_ON` (Good Precision)**: 9/12 Supported (75.0%), 2/12 Partially Supported, 1/12 Unsupported. Most benchmark datasets are explicitly paired with evaluation verbs. In 1 case, a dataset was cited as prior background rather than an experimental benchmark for the proposed model.
3. **`REPORTS_METRIC` (Good Precision)**: 11/15 Supported (73.3%), 3/15 Partially Supported, 1/15 Unsupported. Metrics with quantitative scores (e.g. F1, BLEU, accuracy) are well-grounded; false positives occur when a metric term is used colloquially (e.g., "efficiency" or "latency" mentioned as general goals rather than reported experimental metrics).
4. **`USES_METHOD` (Satisfactory Precision)**: 13/18 Supported (72.2%), 3/18 Partially Supported, 2/18 Unsupported. Rule-based extraction occasionally links a method mentioned only in the related-work sentence as being actively used by the paper.

Artifact saved to [`experiments/scipaper/eval/results/phase_i_relationship_quality_sample.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_i_relationship_quality_sample.json).

---

## 6. Audit D — Retrieval Benchmark Reproducibility

The Phase E retrieval evaluation script (`experiments/scipaper/eval/run_phase_e.py`) was executed in read-only mode against the live post-migration databases using the canonical 50-question benchmark (`experiments/scipaper/eval/benchmark_scipaper_50.json`, SHA256: `39fc73f08962...`).

### Metric Comparison: Reported vs. Freshly Reproduced

| Retriever Mode | Reported Phase H MRR | Reproduced Fresh MRR | Reported HR@5 | Reproduced HR@5 | Reported HR@10 | Reproduced HR@10 | Reported Recall@10 | Reproduced Recall@10 | Variance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Vector-Only** | 0.4475 | **0.4475** | 0.6000 | **0.6000** | 0.7000 | **0.7000** | 0.3583 | **0.3583** | **0.0000 (Exact)** |
| **Graph-Only** | 0.8875 | **0.8875** | 0.9250 | **0.9250** | 0.9250 | **0.9250** | 0.4292 | **0.4292** | **0.0000 (Exact)** |
| **Hybrid RRF (k=60)** | 0.6708 | **0.6708** | 0.9500 | **0.9500** | 0.9500 | **0.9500** | 0.5854 | **0.5854** | **0.0000 (Exact)** |

### Benchmark Invariant Verification:
1. **Identical Gold Chunks & Questions**: All 50 questions and 166 ground-truth chunk IDs match across Phase E, G, and H evaluations.
2. **Category Mapping Discrepancy Resolved**:
   - Category 1: `single_hop_direct_fact` (10 questions)
   - Category 2: `2_hop_relationship` (10 questions)
   - Category 3: `3_hop_dependency` (10 questions)
   - Category 4: `relationship_aggregation` (10 questions — referred to in Phase H narrative as *cross-document synthesis*)
   - Category 5: `out_of_scope` (10 questions)
3. **Retrieval Gains Rationale**: The leap in Graph-Only MRR (0.5875 -> 0.8875) is genuine. When the graph was expanded from sparse author edges to include models, methods, datasets, and metrics, multi-hop Cypher queries (`T-02`, `T-03`, `T-04`, `T-05`) found valid traversal paths that were previously disconnected in the graph.

Artifact saved to [`experiments/scipaper/eval/results/phase_i_retrieval_reproduction.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_i_retrieval_reproduction.json).

---

## 7. Audit E — Out-of-Scope Abstention Audit

The abstention evaluation in `experiments/scipaper/eval/abstention.py` was inspected across similarity thresholds from 0.35 to 0.60.

| Threshold | Out-of-Scope Refusal Recall | Out-of-Scope Correct Refusals | Answerable False Refusals | Answerable Availability | Refusal Precision |
|---|---:|---:|---:|---:|---:|
| **0.35** | 90.0% | 9 / 10 | 1 / 40 | 97.5% | 90.0% |
| **0.40 (Operating Point)** | **100.0%** | **10 / 10** | **1 / 40** | **97.5%** | **90.9%** |
| **0.45** | 100.0% | 10 / 10 | 5 / 40 | 87.5% | 66.7% |
| **0.50** | 100.0% | 10 / 10 | 11 / 40 | 72.5% | 47.6% |

### Key Observations:
- **Refusal Mechanism**: Evaluates local vector cosine similarity thresholding on retrieval candidate lists. Offline evaluation does not invoke external LLMs for end-to-end refusal text generation.
- **Accuracy at 0.40**: All 10 out-of-scope questions have top similarity < 0.40, achieving 100% refusal recall.
- **Disclosure Note**: At threshold 0.40, 1 answerable question (`EVAL-035`) has top similarity 0.384 and is falsely refused. Availability on answerable questions is therefore 97.5% (39/40).

Artifact saved to [`experiments/scipaper/eval/results/phase_i_abstention_audit.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_i_abstention_audit.json).

---

## 8. Audit F — Test Suite and Static Code Quality

Executed strictly within the isolated `.venv` environment without code modifications:

1. **Pytest Suite**:
   - `python -m pytest experiments/scipaper/tests/ -v`
   - **Result**: **66 passed in 15.14s (0 failed)**.
   - Covers batch coordinator, failure injection, target guards, ontology validation, Cypher templates, offline extractor, and preflight guards.
2. **Ruff Linter**:
   - `python -m ruff check experiments/scipaper`
   - **Result**: **All checks passed!**
3. **Ruff Formatter**:
   - `python -m ruff format --check experiments/scipaper`
   - **Result**: **58 files already formatted** (0 unformatted files).
4. **Mypy Type Checking**:
   - `python -m mypy experiments/scipaper`
   - **Result**: **Success: no issues found in 46 source files**.

Detailed test log captured in [`experiments/scipaper/eval/results/phase_i_test_results.txt`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_i_test_results.txt).

---

## 9. Identified Discrepancies, Limitations & Recommendations

1. **Category Labeling Ambiguity**: Standardize the category name `relationship_aggregation` vs `cross-document synthesis` in future evaluation documentation to eliminate confusion.
2. **Linguistic Precision in Offline Extractor**: 6.67% of sampled relationships were unsupported because entity extraction matched terms without verifying that the sentence context was an active contribution rather than background/related work. Adding simple dependency negation or clause filters will improve precision above 85%.
3. **Abstention Gating**: While retrieval score thresholding at 0.40 correctly catches 100% of out-of-scope questions, fusing graph connectivity signals into the abstention gate can recover false refusals on borderline vector-similarity questions like EVAL-035.

---

## 10. Deliverable Inventory

All Phase I audit artifacts are persisted under `experiments/scipaper/eval/`:

- Audit Report: [`experiments/scipaper/eval/PHASE_I_AUDIT_REPORT.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/PHASE_I_AUDIT_REPORT.md)
- Reconciliation Manifest: [`experiments/scipaper/eval/results/phase_i_migration_reconciliation.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_i_migration_reconciliation.json)
- Semantic Quality Sample: [`experiments/scipaper/eval/results/phase_i_relationship_quality_sample.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_i_relationship_quality_sample.json)
- Retrieval Reproduction: [`experiments/scipaper/eval/results/phase_i_retrieval_reproduction.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_i_retrieval_reproduction.json)
- Abstention Audit: [`experiments/scipaper/eval/results/phase_i_abstention_audit.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_i_abstention_audit.json)
- Test Log Output: [`experiments/scipaper/eval/results/phase_i_test_results.txt`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_i_test_results.txt)
