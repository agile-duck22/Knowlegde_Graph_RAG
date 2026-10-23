# Scientific-Paper RAG Benchmark Evaluation Report

- **Execution Timestamp**: 20261022_174436 (UTC)
- **Corpus**: 120 papers, 474 chunks in PostgreSQL (`scipaper_db` at `127.0.0.1:5433`)
- **Graph Store**: Neo4j 722 nodes, 3,045 relationships (`bolt://127.0.0.1:7688`)
- **Benchmark**: `benchmark_scipaper_50.json` (50 questions)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (local cache, dim=384)

---

## 1. Benchmark Validation Outcome

- **Total Questions**: 50 (40 answerable, 10 out-of-scope)
- **Categories**: 5 categories with 10 questions each
- **Gold Chunk References**: 166 (reconciled across 28 unique chunks)
- **Missing Gold Chunks in DB**: 0 (all verified present)
- **Schema Validation Status**: **PASSED**

---

## 2. Overall Vector Retrieval Performance (40 Answerable Questions)

| Metric | k=1 | k=3 | k=5 | k=10 |
| :--- | :---: | :---: | :---: | :---: |
| **Hit Rate@k** | 0.3250 | 0.5250 | 0.6000 | 0.7000 |
| **Recall@k** | 0.0813 | 0.2125 | 0.2625 | 0.3583 |

- **Mean Reciprocal Rank (MRR)**: **0.4475**
- **Zero-Result Queries**: 0

---

## 3. Per-Category Retrieval Performance

| Category | Questions | MRR | Hit Rate@5 | Recall@5 | Hit Rate@10 | Recall@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_hop_direct_fact` | 10 | 0.3743 | 0.5000 | 0.4000 | 0.7000 | 0.5000 |
| `2_hop_relationship` | 10 | 0.5083 | 0.7000 | 0.2500 | 0.7000 | 0.3750 |
| `3_hop_dependency` | 10 | 0.4750 | 0.6000 | 0.2167 | 0.7000 | 0.3000 |
| `relationship_aggregation` | 10 | 0.4325 | 0.6000 | 0.1833 | 0.7000 | 0.2583 |

---

## 4. Graph Retrieval Availability & Analysis

- **Graph Store State**: Neo4j contains 722 nodes and 3,045 relationships.
- **Graph Query Engine Status**: **UNAVAILABLE for scientific ontology**.
- **Technical Reason**: The existing repository query catalog (`src/retrieval/cypher_templates.py`) and router are hardcoded to the enterprise IT domain ontology (`Service`, `Team`, `API`, `Database`, `OWNED_BY`, `DEPENDS_ON`). Parameterized Cypher query templates tailored for the scientific ontology (`Paper`, `Author`, `Method`, `AUTHORED`, `USES_METHOD`) have not yet been implemented in a dedicated scientific graph retriever.
- **Impact on Multi-Hop & Aggregation Questions**: Multi-hop reasoning categories (`2_hop_relationship`, `3_hop_dependency`, `relationship_aggregation`) currently rely solely on dense vector retrieval over raw chunk text without graph traversal guidance.

---

## 5. Answer Generation, Grounding & Refusal Availability

- **Status**: **UNAVAILABLE (Evaluation constrained to offline retrieval)**.
- **Technical Reason**: The production synthesizer (`src/synthesis/synthesizer.py`) requires external paid inference APIs (Anthropic Claude / OpenAI). Per non-negotiable safety rules (`ALLOW_PAID_EMBEDDINGS=false`, zero paid API calls, zero downloaded model weights), answer generation and LLM citation grounding evaluation were not executed.
- **Out-of-Scope Retrieval Behavior**: For the 10 out-of-scope questions, the vector retriever returned dense matches with average similarity score of ~0.45-0.55 due to nearest-neighbor search without an LLM refusal gate.

---

## 6. Success & Failure Analysis

### Representative Retrieval Successes (MRR = 1.0)
- **EVAL-001** (`single_hop_direct_fact`): First hit at Rank 1. Gold: `['be7ff9bf-3730-57d7-a2d6-fc30758eb313', 'd55eb84a-1e7d-5d8b-9f4f-192c731940ee']`. Top retrieved: `['d55eb84a-1e7d-5d8b-9f4f-192c731940ee']` (Similarity: 0.6870)
- **EVAL-005** (`single_hop_direct_fact`): First hit at Rank 1. Gold: `['b95ec804-b981-5fb7-9ec6-ee8490f7e6e9', '002ac2b7-4388-5eac-bc45-16cfdc7ff1fa']`. Top retrieved: `['b95ec804-b981-5fb7-9ec6-ee8490f7e6e9']` (Similarity: 0.4271)
- **EVAL-012** (`2_hop_relationship`): First hit at Rank 1. Gold: `['19878e0f-5b1e-5ed0-afd5-a8be9908dd69', '2afccf33-fbd9-5443-b186-e8e107ce4979', '28c01433-7b1a-5089-8e31-e22ba5dc9020', '29b50b6f-c444-5164-aa50-15877ee2530b']`. Top retrieved: `['19878e0f-5b1e-5ed0-afd5-a8be9908dd69']` (Similarity: 0.6671)

### Representative Retrieval Failures / Misses (MRR = 0.0)
- **EVAL-003** (`single_hop_direct_fact`): Zero gold chunks in top 10. Gold: `['0fc596ad-0ec4-5510-9af9-51c255c36579', '34ca61d6-88b7-58cc-ab72-081f1fac28fd']`. Top retrieved: `['420be994-31c2-5508-a2a7-13518ee3a521', '0c172672-c9c6-59ae-908f-8ae1b4933be8']`
- **EVAL-006** (`single_hop_direct_fact`): Zero gold chunks in top 10. Gold: `['cdc9688f-7ca9-5182-bd48-45a9820c517b', 'cd249ddd-8edb-52a1-bc14-9421c44faded']`. Top retrieved: `['05712658-fc45-5b64-af5c-eba7e90d8e1a', '9e0881b6-ab18-5978-98e0-3d718a37d612']`
- **EVAL-007** (`single_hop_direct_fact`): Zero gold chunks in top 10. Gold: `['a0166796-f679-5a90-8ed6-9e4c47e4afb6', 'd75cefb2-fe07-56f9-9a13-18b43316e7e6']`. Top retrieved: `['bf932bac-630a-5e39-94d0-c2d71cd1df85', '8860f908-da4c-50e8-a1b1-57f232d0ba9c']`

---

## 7. Artifacts Summary
- `benchmark_validation.json`: Detailed question-level and database chunk validation.
- `retrieval_results.jsonl`: Complete ranked retrieval outputs for all 50 questions.
- `retrieval_metrics.json`: Aggregated macro metrics.
- `category_metrics.json`: Stratified metrics by evaluation category.
