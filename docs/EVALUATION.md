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

## 4. Benchmark Results Matrix (Template)

> [!NOTE]
> Values below are unpopulated placeholders. Actual empirical results will be recorded upon completion of Phase 5 benchmark execution. Never fabricate experimental results.

| Category | System | Accuracy (%) | Citation Precision | Citation Hallucination | Latency p50 (ms) | Latency p95 (ms) | Cost / 1k Queries ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Single-Hop Fact** | Vector-Only | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| | **Hybrid KG+Vector** | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| **2-Hop Relationship** | Vector-Only | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| | **Hybrid KG+Vector** | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| **3-Hop Dependency** | Vector-Only | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| | **Hybrid KG+Vector** | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| **Aggregation** | Vector-Only | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| | **Hybrid KG+Vector** | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| **Out-of-Scope** | Vector-Only | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| | **Hybrid KG+Vector** | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| **Overall Summary** | **Vector-Only** | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |
| | **Hybrid KG+Vector** | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` | `[Pending]` |

---

## 5. Automated Evaluation Runner

```bash
# Execute evaluation benchmark across all 50 questions
python -m src.eval.runner --dataset data/eval/benchmark_questions.json --output benchmark_results/run_v1.json

# Generate markdown comparison report
python -m src.eval.report --input benchmark_results/run_v1.json
```
