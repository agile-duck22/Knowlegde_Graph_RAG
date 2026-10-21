# Evaluation & Benchmark Protocol

This document defines the scientific evaluation protocol, question dataset schema, quantitative metrics, and reporting template for comparing **Hybrid KG+Vector RAG** against a baseline **Vector-Only RAG** system.

---

## 1. Evaluation Objectives & Controlled Variables

The evaluation assesses whether combining knowledge graph traversal with vector search provides measurable improvements in answering complex enterprise questions.

### Controlled Parameters
To isolate the effect of retrieval strategy:
- **Same Document Corpus**: Identical enterprise architecture documents ingested into both systems.
- **Same LLM Model**: `claude-3-5-sonnet` used for answer generation across all benchmark runs.
- **Same Prompt Structure**: Identical system prompt and citation formatting rules.
- **Same Temperature & Seed**: `temperature=0.0` for deterministic generation.

---

## 2. Benchmark Question Dataset Taxonomy (50 Questions)

The benchmark comprises 50 curated questions across five distinct categories:

1. **Single-Hop Direct Fact (10 Questions)**: Direct lookups of service properties, owners, or configuration values (e.g., *"Which database is used by the Notification Service?"*).
2. **2-Hop Relationship Queries (15 Questions)**: Relationship traversal requiring bridging two entities (e.g., *"Which services are maintained by the team that owns the Payment Gateway?"*).
3. **3-Hop Dependency Chains (10 Questions)**: Deep dependency analysis (e.g., *"Which databases are written to by services that depend on the Auth Service?"*).
4. **Relationship Aggregations (10 Questions)**: Global grouping and counting (e.g., *"List all services affected by CVE-2024-21626 and their corresponding repository links"*).
5. **Out-of-Scope & Adversarial (5 Questions)**: Non-domain or unsupported queries testing refusal and guardrail behavior (e.g., *"How do I bake sourdough bread?"*).

### Benchmark Item Schema (`data/eval/benchmark_questions.json`)
```json
{
  "id": "EVAL-014",
  "category": "2_hop_relationship",
  "question": "Which teams maintain services that write to the Customer Orders Database?",
  "ground_truth_entities": ["service:order-processor", "team:order-engineering"],
  "ground_truth_chunk_ids": ["chunk-a1b2c3d4", "chunk-e5f6g7h8"],
  "expected_facts": [
    "Order Processor service writes to Customer Orders Database",
    "Order Processor is maintained by Order Engineering team"
  ]
}
```

---

## 3. Evaluation Metrics

1. **Answer Correctness (Accuracy %)**: Percentage of questions where all expected ground-truth facts are correctly stated in the generated answer (evaluated via LLM-as-a-judge with strict rubrics and manual verification).
2. **Citation Precision & Recall**:
   $$\text{Precision} = \frac{|\text{Retrieved Cited Chunks} \cap \text{Ground Truth Chunks}|}{|\text{Retrieved Cited Chunks}|}$$
3. **Citation Hallucination Rate**: Percentage of citations in the generated answer that do not correspond to retrieved chunks.
4. **End-to-End Latency**: Median (p50) and 95th percentile (p95) execution duration in milliseconds.
5. **Cost per Query**: Average token consumption and estimated USD cost per query.

---

## 4. Benchmark Results Matrix (Verified HYBRID v3)

> [!NOTE]
> Values below reflect the empirical results from the full 50-question benchmark run recorded in `benchmark_results/run_hybrid_v3_threshold040.json` (similarity threshold = 0.40).
> Evaluator-reported cost of $0.00 is an artifact of the local SentenceTransformers embedding pipeline and free-tier inference settings; it does not guarantee that third-party commercial model APIs are free.

| Category | System | Questions | Accuracy (%) | Mean Fact Coverage | Citation Precision | Citation Hallucination | Latency p50 (ms) | Latency p95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Single-Hop Fact** | **Hybrid KG+Vector** | 10 | **100.0%** (10/10) | 1.00 | 0.70 | 0.0% | 22,174 ms | 99,511 ms |
| **2-Hop Relationship** | **Hybrid KG+Vector** | 15 | **80.0%** (12/15) | 0.90 | 0.40 | 0.0% | 16,314 ms | 31,233 ms |
| **3-Hop Dependency** | **Hybrid KG+Vector** | 10 | **70.0%** (7/10) | 0.77 | 0.45 | 0.0% | 6,368 ms | 17,211 ms |
| **Aggregation** | **Hybrid KG+Vector** | 10 | **60.0%** (6/10) | 0.83 | 0.40 | 0.0% | 8,970 ms | 18,349 ms |
| **Out-of-Scope** | **Hybrid KG+Vector** | 5 | **100.0%** (5/5) | 1.00 | 1.00 | 0.0% | 1,466 ms | 2,781 ms |
| **Overall Summary** | **Hybrid KG+Vector** | **50** | **80.0%** (40/50) | **0.87** | **0.495** | **0.0%** | **9,267 ms** | **33,044 ms** |

### Known Evaluation Diagnoses

- **EVAL-028 (Corpus Specification Gap)**: In `data/raw/corpus.jsonl`, `service:cart-service` does not explicitly state a direct dependency on `service:checkout-service` or traversing the PCI-isolated VPC (`infra:pci-isolated-vpc`). The synthesizer correctly refrains from hallucinating this connection, causing EVAL-028 to record lower coverage until an evidence-backed corpus update is applied.
- **EVAL-038 (PostgreSQL Relational DB Aggregation)**: Resolved entity resolution and template expansion for PostgreSQL application databases (`database:customer-orders-pg`, `database:user-directory-pg`, `database:analytics-warehouse-pg`) via `cluster_engine` traversal. Unit and traversal tests verify all application database records and their owning teams are properly fetched.

---

## 5. Automated Evaluation Runner

```bash
# Execute evaluation benchmark across all 50 questions
python -m src.eval.runner --dataset data/eval/benchmark_questions.json --output benchmark_results/run_v1.json

# Generate markdown comparison report
python -m src.eval.report --input benchmark_results/run_v1.json
```
