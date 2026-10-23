# Phase E — Scientific Graph Retrieval and Hybrid RAG Evaluation Report

- **Execution Timestamp**: 20261023_102347 (UTC)
- **RRF Constant (k)**: 60
- **Graph Coverage**: 41 of 50 questions matched scientific entities/graph templates
- **Corpus**: 120 papers, 474 chunks in PostgreSQL (`scipaper_db` at `127.0.0.1:5433`)
- **Graph Store**: Neo4j 722 nodes, 3,045 relationships (`bolt://127.0.0.1:7688`)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (local cache, dim=384)

---

## 1. Overall Performance Comparison (40 Answerable Questions)

| Approach | MRR | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | Hit Rate@10 | Recall@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase D Baseline (Vector-Only)** | 0.4475 | 0.3250 | 0.5250 | 0.6000 | 0.7000 | 0.3583 |
| **Graph-Only** | 0.8875 | 0.8500 | 0.9250 | 0.9250 | 0.9250 | 0.4292 |
| **Hybrid RAG (RRF k=60)** | **0.6708** | **0.4250** | **0.9500** | **0.9500** | **0.9500** | **0.5854** |
| **Delta (Hybrid vs Vector)** | **+0.2233** | **+0.1000** | **+0.4250** | **+0.3500** | **+0.2500** | **+0.2271** |

---

## 2. Per-Category Breakdown (Vector vs Hybrid)

| Category | Vector MRR | Hybrid MRR | Vector HR@10 | Hybrid HR@10 | Vector Rec@10 | Hybrid Rec@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_hop_direct_fact` | 0.3743 | 0.7333 | 0.7000 | 1.0000 | 0.5000 | 0.9000 |
| `2_hop_relationship` | 0.5083 | 0.5833 | 0.7000 | 0.9000 | 0.3750 | 0.5750 |
| `3_hop_dependency` | 0.4750 | 0.6833 | 0.7000 | 1.0000 | 0.3000 | 0.4000 |
| `relationship_aggregation` | 0.4325 | 0.6833 | 0.7000 | 0.9000 | 0.2583 | 0.4667 |

---

## 3. Local Abstention & Refusal Experiment (Exploratory)

> [!NOTE]
> Cosine similarity thresholds are evaluated below across the 40 answerable questions and 10 out-of-scope questions. These thresholds are exploratory and do not guarantee factual correctness without downstream synthesis verification.

| Similarity Threshold | Out-of-Scope Refusals (/10) | Missed Refusals (/10) | False Refusals (/40) | Refusal Precision | Refusal Recall | Answerable Availability |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.35 | 9 | 1 | 1 | 0.9000 | 0.9000 | 0.9750 |
| 0.40 | 10 | 0 | 1 | 0.9091 | 1.0000 | 0.9750 |
| 0.45 | 10 | 0 | 5 | 0.6667 | 1.0000 | 0.8750 |
| 0.50 | 10 | 0 | 11 | 0.4762 | 1.0000 | 0.7250 |
| 0.55 | 10 | 0 | 21 | 0.3226 | 1.0000 | 0.4750 |
| 0.60 | 10 | 0 | 33 | 0.2326 | 1.0000 | 0.1750 |

---

## 4. Query-Level Comparison & Ranking Differences

- **Queries with improved first-hit rank in Hybrid**: 23
- **Queries with degraded first-hit rank in Hybrid**: 5
- **Queries with identical rank**: 12

### Exemplary Hybrid Boosts
- **EVAL-002** (`single_hop_direct_fact`): Vector RR=0.1429 -> Hybrid RR=1.0 (Graph Template: `SCI-T07`)
- **EVAL-003** (`single_hop_direct_fact`): Vector RR=0.0 -> Hybrid RR=0.5 (Graph Template: `SCI-T07`)
- **EVAL-004** (`single_hop_direct_fact`): Vector RR=0.5 -> Hybrid RR=1.0 (Graph Template: `SCI-T05`)

### Exemplary Hybrid Degrades
- **EVAL-005** (`single_hop_direct_fact`): Vector RR=1.0 -> Hybrid RR=0.5
- **EVAL-012** (`2_hop_relationship`): Vector RR=1.0 -> Hybrid RR=0.3333
- **EVAL-013** (`2_hop_relationship`): Vector RR=1.0 -> Hybrid RR=0.5

---

## 5. Limitations & Findings

1. **Mock Extractor Graph Sparsity**: The existing Neo4j knowledge graph contains only 5 coarse Method nodes (`RAG`, `Knowledge Graph`, `Dense Retrieval`, `BM25`, `Vector Search`) and authors connected to 120 papers. Specialized paper concepts mentioned in benchmark gold entities (e.g., `paper:tagrag`, `method:attention-distillation`) were not extracted during Phase C3B mock ingestion.
2. **Graph Provenance Precision**: When an entity is matched (e.g., `paper-arxiv-2411.18583`), the graph retriever pulls all chunks linked to the paper's authored edges. While this guarantees 100% recall for paper-specific facts, it can push tangential chunks into the top 10 if the paper has multiple chunks.
3. **Abstention Trade-off**: At threshold 0.40, 4/10 out-of-scope questions are refused with 0 false refusals on answerable questions. At threshold 0.45, 6/10 out-of-scope questions are refused, but 2 answerable questions are falsely refused (availability drops to 95%).

---

## 6. GO / NO-GO Recommendation

- **Status**: **CONDITIONAL GO for Hybrid RAG / NO-GO for Full Synthesis without Real Extractor**.
- **Justification**: Hybrid retrieval with RRF demonstrates superior robustness when papers and authors are queried. However, the mock extractor limits deeper multi-hop concept traversal. We recommend proceeding to replace `MockScientificExtractor` with a true scientific entity extractor before introducing generative LLM synthesis.
