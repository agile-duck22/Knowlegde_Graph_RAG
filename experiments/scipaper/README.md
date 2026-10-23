# Scientific-Paper Generalization Experiment (Phase A Protocol)

## 1. Overview & Research Hypothesis

### Hypothesis
A Hybrid Knowledge Graph + Vector RAG architecture outperforms a baseline Vector-Only RAG system on complex queries over academic research literature by:
1. Faithfully resolving cross-paper methodological dependencies (`2_hop_relationship` and `3_hop_dependency`).
2. Synthesizing comparative architectural trade-offs across literature (`relationship_aggregation`).
3. Preventing ungrounded hallucinations on out-of-domain concepts (`out_of_scope`).

### Domain Scope
- **Subdomain**: *Retrieval-Augmented Generation (RAG) & Large Language Model Grounding* (e.g. dense passage retrieval, graph-augmented retrieval, hallucination mitigation, token/passage chunking, vector index structures).
- **Target Corpus Scale**: 100–150 open-access peer-reviewed conference/journal papers or preprints (2020–2024).

---

## 2. Dataset Inclusion, Licensing & Acquisition Protocol

### Inclusion Criteria
- Open-access papers available via arXiv or PubMed Central under permissive redistribution licenses (e.g., CC-BY-4.0, CC-BY-NC-SA, or arXiv perpetual non-exclusive distribution license).
- Topics directly relevant to LLM retrieval, indexing, synthesis, or graph structures.
- Both abstract-only and full-text extractions are recorded with explicit provenance tagging.

### Exclusion Criteria
- Non-English publications.
- Non-peer-reviewed blog posts or proprietary whitepapers lacking open licenses.
- Papers behind paywalls or restrictive copyright (no commercial scraping or terms-of-service violations).

### Reproducible Acquisition Procedure
1. Record stable source identifier (`arxiv_id` and/or `doi`), canonical URL, title, authors, year, and venue into `data/manifest.jsonl`.
2. Extract text (abstract or permissible open-access full text) without fabricated content.
3. Compute and record SHA-256 content checksum in both `manifest.jsonl` and `corpus_scipaper.jsonl`.
4. Validate the dataset using the offline validator:
   ```bash
   python -m experiments.scipaper.scripts.validate_experiment --strict-count
   ```

---

### Corpus Scope & Content Representation
- **Corpus Scale & Nature**: The corpus consists of **120 unique arXiv records** covering title, author list, publication venue, and **abstract text**—it does not contain full-text PDF document extractions.
- **Evaluation Purpose**: Results obtained from this corpus evaluate abstract-level knowledge retrieval, cross-abstract synthesis, and grounded answering over scientific summaries.
- **Full-Text vs. Abstract Transparency**: Every record explicitly sets `content_type: "abstract"`. Any future expansion to full-text PDF extractions will be separately versioned and verified.

### Manifest Contract & Licensing Provenance
Every line in `data/manifest.jsonl` tracks legal provenance:
- `paper_id`, `title`, `authors`, `year`, `venue`, `source_url`, `license`, `content_type`, `sha256`, `doi`, `arxiv_id`.
- **Licensing Provenance**: All 120 records are public e-prints hosted by arXiv and recorded under `"arXiv.org perpetual non-exclusive license / CC-BY"`. The offline harvest metadata preserves the public arXiv preprint terms of use. Specific Creative Commons variants (e.g. CC-BY-4.0, CC-BY-NC-SA) applied by individual authors require extracting per-paper Atom `<link title="license" href="...">` nodes via a dedicated network audit pass, and public availability is not presumed to imply unconstrained commercial re-use.

---

## 4. Benchmark Taxonomy & Gold-Label Policy

The evaluation dataset in `eval/benchmark_scipaper_50.json` adheres to `src.eval.schema.BenchmarkDataset` with exactly 50 questions across 5 taxonomy categories:

| Category | Target Count | Evaluation Objective |
|---|:---:|---|
| `single_hop_direct_fact` | 10 | Exact parameter, benchmark metric, or architectural component lookup. |
| `2_hop_relationship` | 10 | Cross-paper links (Paper A uses Method B evaluated on Benchmark C). |
| `3_hop_dependency` | 10 | Multi-hop dependency chains (Method A → Base Model B → Dataset C). |
| `relationship_aggregation` | 10 | Comparative analysis across multiple papers on trade-offs and latency. |
| `out_of_scope` | 10 | Unanswerable inquiries evaluating hallucination refusal behavior. |

### Ground Truth Integrity
- **Expected Facts**: Verified atomic propositions that must be cited in the generated answer.
- **Reference Answer**: Authoritative ground-truth answer withheld completely from the retrieval context and prompt.
- **Provenance Linkage**: Metadata contains `source_paper_ids` cross-referenced with `manifest.jsonl`.
- *Status Note*: In Phase A, template items are explicitly flagged `status: "template_unverified"` until the real 100–150 paper corpus is acquired and chunked.

---

## 5. Scientific Ontology Architecture

### Domain Concept Specialization
Rather than forcing academic literature into enterprise IT types (`Service`, `Team`, `API`, `Database`), the scientific ontology defines:
- **Entity Types**: `Paper`, `Author`, `Model`, `Dataset`, `Metric`, `Task`, `Method`, `ResearchOrganization`.
- **Relationship Types**: `AUTHORED`, `PROPOSES`, `EVALUATED_ON`, `REPORTS_METRIC`, `USES_METHOD`, `AFFILIATED_WITH`, `CITES`.
- **Canonical Prefixes**: `paper:`, `author:`, `model:`, `dataset:`, `metric:`, `task:`, `method:`, `org:`.

### Enterprise Compatibility Strategy
- All Phase A models and validation reside in `experiments/scipaper/schemas/models.py`.
- The enterprise production schemas in `src/extraction/schemas.py`, `src/extraction/resolver.py`, and `src/db/neo4j.py` remain **100% untouched**.
- In future phases, domain-specific extractors and resolvers will be instantiated via dependency injection or a domain registry, ensuring zero regression on the enterprise codebase.

---

## 6. Zero-Cost & Offline Validation

- **Embedding Model**: Local `sentence-transformers/all-MiniLM-L6-v2` (dimension: 384).
- **Cost Guard**: `ALLOW_PAID_EMBEDDINGS=false` is strictly preserved.
- **Offline Validation Tool**:
  ```bash
  python -m experiments.scipaper.scripts.validate_experiment
  ```
- **Test Suite**:
  ```bash
  pytest experiments/scipaper/tests/ -v
  ```

---

## 7. Phase Gates & Governance

```
[Phase A: Data & Protocol Design] (CURRENT — COMPLETE)
       ↓
[Gate A: Review & Approval of Protocol, Taxonomy, and Schemas]
       ↓
[Phase B: Isolated Infrastructure & Store Provisioning]
       ↓
[Gate B: Verification of Isolated DB Ports (5433 / 7688) & Preflight Check]
       ↓
[Phase C: Corpus Acquisition & Dual-Store Ingestion]
       ↓
[Gate C: Integrity Audit of Neo4j & pgvector Chunks]
       ↓
[Phase D: Benchmark Execution & Comparative Evaluation]
```

**Status**: Phase A is complete. All experimental artifacts are isolated in `experiments/scipaper/`. No databases were contacted, no external APIs were called, and existing enterprise data remains pristine.
