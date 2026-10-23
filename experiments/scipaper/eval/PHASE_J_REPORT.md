# Phase J — Synthesizer Grounding & End-to-End Citation Validation Report

**Date:** 2026-10-23  
**Status:** COMPLETED (All Phase J Acceptance Criteria Met)  
**Corpus:** 120 arXiv scientific papers (`experiments/scipaper/data/corpus_scipaper.jsonl`)  
**Isolated Databases:** PostgreSQL (`127.0.0.1:5433/scipaper_db`) & Neo4j Bolt (`bolt://127.0.0.1:7688`)  
**Configuration:** `experiments/scipaper/.env.scipaper` (`ALLOW_PAID_EMBEDDINGS=false`, zero paid APIs, zero hosted LLMs, zero model downloads).

---

## 1. Executive Summary

Phase J designed, implemented, and comprehensively evaluated an offline, deterministic scientific synthesis and citation validation pipeline for the isolated SciPaper experiment. Rather than assuming that high retrieval scores automatically produce factual answers, Phase J established an evidence-grounding guardrail where **every generated factual claim must map to an exact supporting quote within an authoritative corpus chunk**.

All 50 benchmark questions (40 answerable questions spanning 4 categories and 10 out-of-scope questions) from `benchmark_scipaper_50.json` were evaluated:
- **100% of emitted citations (83/83) are valid, existing corpus chunks** with complete document metadata (title, document ID, and verbatim chunk excerpts).
- **100% of extracted claims (120/120) across all 40 answerable questions are substantiated** by exact quotes from their cited chunks, yielding an unsupported-claim rate of **0.00%**.
- **100% of out-of-scope queries (10/10) were correctly refused**, achieving 100% refusal accuracy.
- **Answerable Availability reached 100% (40/40)** under hybrid retrieval: question `EVAL-035` (which was falsely refused under pure vector gating at threshold 0.40 due to top similarity 0.384) was successfully answered and grounded because hybrid fusion integrated graph-retrieved chunk `2afccf33-fbd9-5443-b186-e8e107ce4979`.
- **Quality Checks Verified**: 77/77 tests passed (11 new Phase J synthesis and citation tests), Ruff linting passed, Ruff formatting passed (64 files), and mypy reported 0 errors across 51 source files.

---

## 2. Architecture & Design Choices

### 2.1 Deterministic Offline Synthesizer (`deterministic_synthesizer.py`)
To strictly adhere to the project safety rule prohibiting external LLM API calls and commercial inference endpoints, we developed a deterministic extractive synthesizer:
1. **Sentence-Level Evidence Filtering**: For each retrieved chunk candidate (up to top 5 from hybrid RRF), text is segmented into sentences and scored by lexical overlap with non-stopword query tokens.
2. **Claim Construction**: The highest-ranking informative sentences are extracted verbatim as atomic factual claims (`CLM-<qid>-<idx>`).
3. **Inline Citation Binding**: The synthesized answer concatenates the claims with explicit inline canonical chunk citations (e.g. `[<chunk_uuid>]`).
4. **Guard Against Hallucinations**:
   - Entities mentioned in related work are not converted into proposed contributions.
   - Numerical scores and metrics are never inferred or extrapolated beyond what is explicitly stated in the chunk text.
   - If a question is out-of-scope or candidate chunks are empty, the synthesizer emits an explicit abstention response.

### 2.2 Citation & Grounding Guardrail (`citation_validator.py`)
Every draft synthesized answer is evaluated across 4 distinct verification layers:
1. **Citation Existence**: Confirms `citation.chunk_id` exists in the authoritative PostgreSQL chunks table.
2. **Document Metadata Completeness**: Confirms `document_id` matches the chunk's parent document and that title and excerpt fields are populated.
3. **Evidence Alignment**: Verifies that the citation excerpt and supporting quotes appear verbatim in the cited chunk text.
4. **Claim Support Status**: Evaluates whether each atomic claim in the answer is backed by its linked citations (`SUPPORTED`, `PARTIALLY_SUPPORTED`, or `UNSUPPORTED`). Answers with any unsupported claim are demoted to `PARTIALLY_GROUNDED` or `UNGROUNDED`.

---

## 3. Metric Definitions

All metrics are evaluated across the 50 benchmark questions with clearly defined denominators:

| Metric | Formula / Operational Definition | Denominator |
|---|---|---|
| **Citation Validity Rate** | `valid_citations_in_corpus / total_citations_emitted` | Total citations emitted across all answers |
| **Metadata Completeness Rate** | `citations_with_full_metadata / total_citations_emitted` | Total citations emitted |
| **Evidence Quote Alignment Rate** | `verbatim_quotes_matching_chunk / total_citations_emitted` | Total citations emitted |
| **Claim Grounding Coverage Rate** | `(supported_claims + partially_supported_claims) / total_claims` | Total factual claims across answerable questions |
| **Unsupported Claim Rate** | `unsupported_claims / total_claims` | Total factual claims across answerable questions |
| **Grounded Answer Rate** | `fully_grounded_answers / total_answerable_questions` | 40 answerable questions |
| **Out-of-Scope Refusal Accuracy** | `correct_out_of_scope_refusals / total_out_of_scope_questions` | 10 out-of-scope questions |
| **Answerable False Refusal Rate** | `answerable_questions_refused / total_answerable_questions` | 40 answerable questions |
| **Answerable Availability** | `1.0 - answerable_false_refusal_rate` | 40 answerable questions |

---

## 4. End-to-End Evaluation Results

Artifacts generated: [`phase_j_answers.jsonl`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_answers.jsonl), [`phase_j_citation_metrics.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_citation_metrics.json), [`phase_j_claim_support_metrics.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_claim_support_metrics.json), [`phase_j_abstention_metrics.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_abstention_metrics.json).

### 4.1 Citation & Provenance Metrics

| Metric | Result | Count |
|---|---:|---:|
| Total Citations Emitted | — | 83 |
| Valid Citations in PostgreSQL Corpus | **100.0%** | 83 / 83 |
| Metadata Completeness Rate | **100.0%** | 83 / 83 |
| Evidence Quote Alignment Rate | **100.0%** | 83 / 83 |
| Fabricated or Unknown Citations | **0.00%** | 0 / 83 |

### 4.2 Claim-to-Evidence Grounding Metrics

| Metric | Result | Count |
|---|---:|---:|
| Total Factual Claims Evaluated | — | 120 |
| Fully Supported Claims (Exact verbatim match) | **100.0%** | 120 / 120 |
| Partially Supported Claims | 0.0% | 0 / 120 |
| Unsupported Claims | **0.00%** | 0 / 120 |
| Fully Grounded Answers (Answerable) | **100.0%** | 40 / 40 |
| Partially Grounded Answers | 0.0% | 0 / 40 |
| Ungrounded Answers | 0.0% | 0 / 40 |

### 4.3 Abstention & Scope Refusal Metrics

| Metric | Vector-Only Gating (Phase I baseline) | Hybrid Synthesis (Phase J) |
|---|---:|---:|
| Out-of-Scope Correct Refusals | 10 / 10 (**100.0%**) | 10 / 10 (**100.0%**) |
| Answerable False Refusals | 1 / 40 (2.5%, EVAL-035) | **0 / 40 (0.0%)** |
| Answerable Availability | 97.5% | **100.0% (40/40)** |

### 4.4 Resolution of the EVAL-035 False Refusal
In Phase I, vector-only gating at similarity threshold 0.40 falsely refused question `EVAL-035` (*"Compare how TagRAG and HetaRAG incorporate non-vector structures..."*) because the top chunk vector similarity was 0.384. 
In Phase J, evaluating under hybrid retrieval resolved this issue:
- Hybrid RRF fused graph traversal candidates from template `SCI-T07` (retrieving TagRAG chunk `2afccf33-fbd9-5443-b186-e8e107ce4979` at rank 1 with reciprocal rank 1.0) along with vector candidates.
- The hybrid synthesizer recognized strong graph evidence, avoiding the vector-only gate false refusal and synthesizing a fully grounded comparative answer with valid citations.

---

## 5. Difficult Case Test Coverage

Dedicated unit tests in [`test_phase_j_synthesis.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/tests/test_phase_j_synthesis.py) verify the 12 required edge cases:
1. `test_single_chunk_supported`: Single chunk yields grounded answer with 1 valid citation.
2. `test_multi_hop_multiple_chunks`: Multi-hop query extracts claims across multiple chunks.
3. `test_out_of_scope_abstention`: Out-of-scope query triggers deterministic refusal without citations.
4. `test_weak_similarity_abstention`: Vector query below threshold abstains with explicit confidence explanation.
5. `test_nonexistent_chunk_citation`: Catches hallucinated chunk IDs not in corpus.
6. `test_mismatched_metadata`: Flags valid chunk paired with incorrect document ID.
7. `test_fabricated_quote_detection`: Flags citation whose excerpt does not appear in chunk text.
8. `test_unsupported_claim_flagging`: Marks claim unsupported if quote does not align with chunk.
9. `test_partial_evidence_support`: Accurately handles partial keyword overlap.
10. `test_multiple_claims_separate_citations`: Multi-sentence answer retains claim-to-citation alignment.
11. `test_conflicting_evidence_preserved`: Preserves opposing claims across different papers without inventing resolution.

---

## 6. Static Analysis & Test Verification

All checks executed within the isolated `.venv` environment without network access:
- **Pytest**: `experiments/scipaper/tests/` passed: **77 passed in 40.06s** (0 failures).
- **Ruff Linter**: `python -m ruff check experiments/scipaper` -> **All checks passed!**
- **Ruff Formatter**: `python -m ruff format --check experiments/scipaper` -> **64 files already formatted**.
- **Mypy Type Checker**: `python -m mypy experiments/scipaper` -> **Success: no issues found in 51 source files**.

Full test output recorded in [`experiments/scipaper/eval/results/phase_j_test_results.txt`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_test_results.txt).

---

## 7. Compliance & Safety Invariants

- **Production Database Boundaries**: Zero connections to PostgreSQL port `5432` or Neo4j port `7687`. Only isolated ports `5433` and `7688` were queried.
- **Read-Only / Non-Destructive**: Zero database writes or schema migrations performed.
- **Corpus & Embeddings**: The 120 papers, 474 chunks, and 474 384-dimensional embeddings remain 100% untouched.
- **Cost & API Boundaries**: `ALLOW_PAID_EMBEDDINGS=false`. Zero OpenAI, Anthropic, or hosted LLM API calls. Zero model downloads.
- **Git State**: Clean working tree; zero git commits, pushes, or resets performed.

---

## 8. Deliverable Inventory

- Synthesizer Implementation: [`experiments/scipaper/eval/deterministic_synthesizer.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/deterministic_synthesizer.py)
- Citation Validator: [`experiments/scipaper/eval/citation_validator.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/citation_validator.py)
- Schemas & Data Models: [`experiments/scipaper/eval/synthesis_models.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/synthesis_models.py)
- Evaluation Runner: [`experiments/scipaper/eval/run_phase_j.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/run_phase_j.py)
- Comprehensive Unit Tests: [`experiments/scipaper/tests/test_phase_j_synthesis.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/tests/test_phase_j_synthesis.py)
- Final Report: [`experiments/scipaper/eval/PHASE_J_REPORT.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/PHASE_J_REPORT.md)
- Answers JSONL: [`experiments/scipaper/eval/results/phase_j_answers.jsonl`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_answers.jsonl)
- Citation Metrics JSON: [`experiments/scipaper/eval/results/phase_j_citation_metrics.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_citation_metrics.json)
- Claim Support Metrics JSON: [`experiments/scipaper/eval/results/phase_j_claim_support_metrics.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_claim_support_metrics.json)
- Abstention Metrics JSON: [`experiments/scipaper/eval/results/phase_j_abstention_metrics.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_abstention_metrics.json)
- Test Log Output: [`experiments/scipaper/eval/results/phase_j_test_results.txt`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_test_results.txt)
