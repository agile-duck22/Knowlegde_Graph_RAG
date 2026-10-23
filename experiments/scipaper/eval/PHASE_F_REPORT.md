# Phase F Report: Offline Scientific Paper Extraction for Knowledge Graph RAG

**Date**: 2026-10-23  
**Domain**: Scientific Papers (Isolated Experiment: `experiments/scipaper/`)  
**Status**: COMPLETE (Verified & Evaluated)  
**Safety Compliance**: 100% Offline, Zero Remote API Calls, Zero Model Downloads, Zero Database Writes, Target Invariants Preserved  

---

## 1. Executive Summary

Phase F replaced the preliminary mock extraction implementation (`MockScientificExtractor`) with a deterministic, evidence-backed, fully offline scientific paper extraction pipeline (`OfflineScientificExtractor`).

The extractor operates on real arXiv titles, metadata, and abstracts, mining 8 ontology entity types and 7 scientific relationship types with strict provenance tracking back to parent documents and chunk UUIDs.

Evaluation across a 20-paper sample against an independently curated 10-paper gold annotation subset yielded:
- **Entity Extraction**: **0.8116 F1** (Precision: 0.7368, Recall: 0.9032)
- **Relationship Extraction**: **0.9032 F1** (Precision: 0.8537, Recall: 0.9589)
- **Provenance Validity Rate**: **100.00%** (zero orphaned claims, zero invalid chunk/doc IDs)
- **Invalid Record Count**: **0**
- **Database Preservation**: 100% preserved (0 writes to PostgreSQL or Neo4j)

---

## 2. Extraction Architecture & Ontology

### 2.1 Pipeline Components
1. **Metadata Parsing**:
   - Parses structured paper identity and author rosters from arXiv metadata.
   - Generates high-confidence `Author` entities and `AUTHORED` relationships bound to paper identifiers.
2. **Deterministic Gazetteers & Alias Mapping (`patterns.py`)**:
   - Curated scientific concept dictionaries covering Methods (e.g. *RAG, Dense Retrieval, BM25, Multi-hop Reasoning*), Models (*GPT-4, Llama 3, Gemma, Mistral, T5, BERT*), Datasets (*HotpotQA, Natural Questions, SciTLDR, MusiQue, 2WikiMultiHopQA*), Metrics (*ROUGE, BLEU, F1-Score, NDCG, Hit Rate, Exact Match*), Tasks, and Organizations.
   - Word-boundary regex matching prevents false partial-token triggers.
3. **Linguistic Syntactic Pattern Mining**:
   - Compiles deterministic patterns detecting system proposals (`PROPOSES`), experimental evaluations (`EVALUATED_ON`), and performance reporting (`REPORTS_METRIC`).
4. **Ontology Alignment & Entity Resolution (`resolver.py`)**:
   - Enforces strict entity type segregation so overlapping names are not conflated across domains.
   - Deterministic slugification and namespace mapping (`paper:<slug>`, `author:<slug>`, `method:<slug>`).
5. **Dual-Store Evidence & Provenance Binding**:
   - Every entity and relationship is bound to the exact chunk UUID and document ID from which it was extracted (`ExtractionProvenance`), satisfying the dual-store invariant.

### 2.2 Supported Schema

| Entity Type | Source Evidence | Default Relationship |
| :--- | :--- | :--- |
| `Paper` | Structured metadata / Title | N/A (Root target) |
| `Author` | Structured author metadata roster | `(Author)-[:AUTHORED]->(Paper)` |
| `Method` | Text mention / Gazetteer / Linguistic match | `(Paper)-[:USES_METHOD]->(Method)` |
| `Model` | Text mention / Gazetteer | `(Paper)-[:USES_MODEL]->(Model)` |
| `Dataset` | Text mention / Benchmark gazetteer | `(Paper)-[:EVALUATED_ON]->(Dataset)` |
| `Metric` | Text mention / Evaluation gazetteer | `(Paper)-[:REPORTS_METRIC]->(Metric)` |
| `Task` | Text mention / Task gazetteer | `(Paper)-[:ADDRESSES_TASK]->(Task)` |
| `ResearchOrganization` | Text mention / Institutional roster | `(Author)-[:AFFILIATED_WITH]->(ResearchOrganization)` |

---

## 3. Evaluation Dataset & Methodology

### 3.1 Corpus & Evaluation Split
- **Total Corpus**: 120 papers (`experiments/scipaper/data/corpus_scipaper.jsonl`)
- **Evaluation Corpus**: 20 papers processed into 78 chunks.
- **Gold Evaluation Subset**: 10 papers manually annotated strictly from actual paper titles, authors, and abstract content without using extractor outputs (`experiments/scipaper/eval/data/extraction_gold_subset.json`).

### 3.2 Evaluation Rules
- Gold annotations were produced independently prior to evaluation.
- Metrics are calculated on exact canonical ID matches and exact `(source, relation_type, target)` triples.
- Extraction was executed entirely in memory: **zero database writes were performed**.

---

## 4. Evaluation Results

The evaluation results are recorded in machine-readable JSON format at `experiments/scipaper/eval/results/extraction_evaluation.json`.

### 4.1 Overall Performance Summary

| Metric | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **All Entities** | **0.7368** | **0.9032** | **0.8116** |
| **All Relationships** | **0.8537** | **0.9589** | **0.9032** |
| **Provenance Validity** | **100.00%** | **100.00%** | **1.0000** |

### 4.2 Entity Metrics by Type (10 Gold Papers)

| Entity Type | Gold | Extracted (TP + FP) | TP | FP | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Paper` | 10 | 10 | 10 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `Author` | 52 | 52 | 52 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `Dataset` | 4 | 5 | 4 | 1 | 0 | 0.8000 | 1.0000 | 0.8889 |
| `Metric` | 4 | 6 | 4 | 2 | 0 | 0.6667 | 1.0000 | 0.8000 |
| `Method` | 13 | 19 | 10 | 9 | 3 | 0.5263 | 0.7692 | 0.6250 |
| `Model` | 7 | 6 | 3 | 3 | 4 | 0.5000 | 0.4286 | 0.4615 |
| `Task` | 3 | 7 | 1 | 6 | 2 | 0.1429 | 0.3333 | 0.2000 |
| `ResearchOrganization`| 0 | 9 | 0 | 9 | 0 | 0.0000 | 0.0000 | 0.0000 |

### 4.3 Relationship Metrics by Type

| Relation Type | Gold | Extracted | TP | FP | FN | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AUTHORED` | 52 | 52 | 52 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| `EVALUATED_ON` | 4 | 5 | 4 | 1 | 0 | 0.8000 | 1.0000 | 0.8889 |
| `REPORTS_METRIC`| 4 | 6 | 4 | 2 | 0 | 0.6667 | 1.0000 | 0.8000 |
| `USES_METHOD` | 13 | 15 | 10 | 5 | 3 | 0.6667 | 0.7692 | 0.7143 |
| `PROPOSES` | 0 | 4 | 0 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 |

---

## 5. Verification & Quality Gates

### 5.1 Test Suite Results
- `pytest experiments/scipaper/tests/ -v`: **64 passed, 0 failed** (including 7 new dedicated tests for offline extraction, unicode handling, and determinism).
- Backward compatibility: `MockScientificExtractor` is exported as an alias/subclass of `OfflineScientificExtractor`, ensuring existing tests in `test_ingest_adapter.py` and `test_batch_coordinator.py` continue to pass without regression.

### 5.2 Lint, Format, and Type Checking
- `ruff check experiments/scipaper`: **Passed** (All checks passed).
- `ruff format --check experiments/scipaper`: **Passed** (40 files verified).
- `mypy experiments/scipaper`: **Passed** (Success: no issues found in 35 source files).

### 5.3 Database Preservation Verification
Direct queries against PostgreSQL (port `5433`) and Neo4j (port `7688`) before and after Phase F execution confirm exact retention of the baseline state:

| Target Store | Metric / Table | Expected Baseline | Post-Phase F Count | Status |
| :--- | :--- | :---: | :---: | :---: |
| **PostgreSQL** | Documents | 120 | 120 | Unchanged |
| | Chunks | 474 | 474 | Unchanged |
| | Entity Links | 3,519 | 3,519 | Unchanged |
| | `embedding_384` | 474 | 474 | Unchanged |
| | `embedding` (1536-dim) | 0 | 0 | Unchanged |
| **Neo4j** | Total Nodes | 722 | 722 | Unchanged |
| | - `Paper` Nodes | 120 | 120 | Unchanged |
| | - `Author` Nodes | 597 | 597 | Unchanged |
| | - `Method` Nodes | 5 | 5 | Unchanged |
| | Total Relationships | 3,045 | 3,045 | Unchanged |
| | - `AUTHORED` | 2,631 | 2,631 | Unchanged |
| | - `USES_METHOD` | 414 | 414 | Unchanged |

---

## 6. Offline Compliance & Limitations

### 6.1 Compliance Audit
- Zero remote API calls made (no OpenAI, Gemini, Claude, or Groq calls).
- Zero model downloads initiated (`ALLOW_PAID_EMBEDDINGS=false` preserved).
- All NLP processing uses Python standard library, deterministic regex matching, and curated domain gazetteers.

### 6.2 Known Limitations & Failure Cases
1. **Abstract-Only Context**:
   - The corpus contains abstracts rather than full PDFs. Detailed experimental configurations, architecture hyperparameters, full author affiliations, and comprehensive metric tables are absent from abstracts.
2. **Ambiguity in Generic Terminology**:
   - Terms like "multi-hop reasoning" or "text generation" can function as either a general research task or a methodological description depending on context.
3. **Research Organization Detection**:
   - arXiv metadata in this corpus does not include structured institution tags. Detecting organizations from unstructured author blocks is prone to false positives without full-text affiliation sections.

---

## 7. Recommendations for Next Phase

Before initiating any graph migration or full corpus re-ingestion:
1. **Graph Schema Migration Plan**: Update Neo4j constraints to index the new entity labels (`Model`, `Dataset`, `Metric`, `Task`).
2. **Phase G Ingestion Authorization**: Run a dry-run ingestion on 1 paper to verify writer idempotency with the expanded relationship types before touching the 120 papers.
3. **Hybrid Retrieval Optimization**: Update `ScientificCypherCatalog` in Phase E to leverage `EVALUATED_ON` and `USES_MODEL` paths for multi-hop graph retrieval.
