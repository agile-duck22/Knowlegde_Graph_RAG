# Phase J.1 — Adversarial Grounding Audit Report

**Date:** 2026-10-23  
**Audit Objective:** Independently stress-test and challenge the Phase J synthesis, citation validation, and claim-support results.  
**Audited Artifacts:** `experiments/scipaper/eval/results/phase_j_answers.jsonl`, `citation_validator.py`, `deterministic_synthesizer.py`, and `benchmark_scipaper_50.json`.  
**Safety & Compliance:** Read-only against PostgreSQL (`127.0.0.1:5433`) and Neo4j (`bolt://127.0.0.1:7688`). No modifications to corpus, chunks, embeddings, databases, or existing Phase J artifacts. Zero paid/hosted APIs.

---

## 1. Executive Summary & Audit Verdict

Phase J reported 100% citation validity (83/83) and 100% claim-support (120/120 supported claims) under deterministic extractive synthesis. 

This **Phase J.1 Adversarial Audit** independently investigated whether these metrics represent genuine semantic correctness and answer completeness, or whether they mask underlying limitations of purely extractive RAG.

### Key Audit Verdict: **CRITICAL DISTINCTIONS UNCOVERED**

1. **Separation of Grounding from Relevance and Completeness**:
   - **Verbatim Citation Validity**: 100% (83/83 citations link to real corpus chunks with matching excerpts).
   - **Question Relevance Rate**: **97.5% (39/40 answerable questions)**. In 1 case (`EVAL-036`), retrieved candidates were completely off-topic; while the extracted sentences were 100% verbatim grounded in the retrieved chunks, the answer failed to answer what the question asked.
   - **Answer Completeness Rate**: **82.5% (33/40 answerable questions)**. Across comparative and multi-hop questions, **7 out of 40 answerable questions (17.5%)** suffered from partial coverage or sentence truncation, including `EVAL-035` which compared only TagRAG while completely omitting HetaRAG.
2. **Adversarial Claim-Support Vulnerabilities**:
   - The Phase J `citation_validator.py` evaluates claim support primarily via string and bag-of-words keyword overlap with cited chunks.
   - Across **12 adversarial claim cases**, **8 cases exposed significant semantic blind spots** where false, reversed, out-of-context, or non-contributive assertions passed validator checks simply because the text substring occurred in the source chunk.
3. **The EVAL-035 Deep-Dive**:
   - In Phase I, `EVAL-035` was falsely refused under vector-only gating (similarity 0.384 < 0.40).
   - In Phase J, hybrid retrieval avoided false refusal and synthesized an answer citing TagRAG chunks.
   - **Adversarial Audit Finding**: While the answer is 100% grounded to TagRAG, **it is one-sided and incomplete**. HetaRAG chunks were not present in the top-5 candidate window, so the comparative question was answered for only one side. Under a strict semantic standard, the system should either retrieve candidates for both entities or explicitly abstain/qualify its answer.

---

## 2. Independent Claim-Support Assessment (12 Adversarial Cases)

We designed 12 adversarial test cases to inspect whether `citation_validator.py` validates true semantic meaning in context or merely detects substring presence.

Artifact saved to [`experiments/scipaper/eval/results/phase_j1_adversarial_cases.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j1_adversarial_cases.json):

| Case ID | Adversarial Test Scenario | Expected Semantic Behavior | Actual Validator Status | Vulnerability Exposed |
|---|---|---|---|---|
| **Case 1** | Method mentioned only in related work cited as paper contribution | **UNSUPPORTED** as contribution | `SUPPORTED` | Extractive sentence matches source verbatim, failing to distinguish prior work from author contribution. |
| **Case 2** | Method explicitly proposed by current paper | `SUPPORTED` | `SUPPORTED` | Verified positive baseline. |
| **Case 3** | Method explicitly rejected by paper | **UNSUPPORTED** as recommendation | `SUPPORTED` | Sentence is factual in isolation, but can present rejected baselines as positive findings. |
| **Case 4** | Dataset mentioned without experimental usage | **UNSUPPORTED** as benchmark | `SUPPORTED` | Validator accepts the sentence, ignoring that the paper explicitly stated the dataset was NOT used. |
| **Case 5** | Metric mentioned without numerical results | **UNSUPPORTED** as reported result | `SUPPORTED` | Metric mentioned as unmeasured future work is treated as experimental evidence. |
| **Case 6** | Prior work numerical result misattributed | **MISATTRIBUTION** | `SUPPORTED` | Verbatim match passes 100%, but historical attribution context is stripped. |
| **Case 7** | Reversed relationship direction | **UNSUPPORTED** | `SUPPORTED` (via bag-of-words fallback) | Contains all identical keywords (`dense`, `vector`, `re-rank`), passing keyword fallback despite inverted causal direction. |
| **Case 8** | Same entities with different relationship | `UNSUPPORTED` | `UNSUPPORTED` | **Successfully caught**: new relational words (`fine-tuned`, `synthetic`) do not appear in chunk. |
| **Case 9** | One-sided comparison (asks A & B, only A present) | **INCOMPLETE / QUALIFY** | `GROUNDED` | Answers half the question; validator marks whole answer as 100% grounded. |
| **Case 10** | Contradictory evidence across two chunks | **FLAG CONFLICT** | `GROUNDED` | Mutually contradictory statements are concatenated and pass as 100% grounded. |
| **Case 11** | Verbatim chunk sentence irrelevant to question | **IRRELEVANT** | `SUPPORTED` | Topical irrelevance passes because claim grounding is decoupled from question relevance. |
| **Case 12** | Factually true claim unsupported by cited chunk | `UNSUPPORTED` | `UNSUPPORTED` | **Successfully caught**: chunk text source-of-truth enforced. |

**Audit Conclusion on Validator**: The current validator effectively prevents hallucinated quotes, ghost chunk IDs, and forged facts (Cases 8 and 12). However, it cannot detect semantic role inversion, prior-work misattribution, or cross-chunk contradictions without deeper natural language inference (NLI).

---

## 3. Answer Relevance and Completeness Audit (All 50 Benchmark Questions)

We audited all 50 answers in [`phase_j_answers.jsonl`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j_answers.jsonl) across the 6 separate dimensions:

Detailed results saved to [`experiments/scipaper/eval/results/phase_j1_answer_review.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j1_answer_review.json):

```
+---------------------------------------------------------------------------------------+
| PHASE J.1 ANSWER QUALITY DECOMPOSITION (40 Answerable Questions)                     |
+---------------------------------------------------------------------------------------+
| 1. Citation Validity Rate       : 100.0% (83/83 citations exist in PostgreSQL)        |
| 2. Quote Alignment Rate         : 100.0% (120/120 claims verbatim in chunks)          |
| 3. Question Relevance Rate      :  97.5% (39/40 questions relevantly addressed)      |
| 4. Answer Completeness Rate     :  82.5% (33/40 questions completely answered)        |
| 5. Fully Adequate Answers       :  82.5% (33/40 questions both relevant & complete)   |
| 6. Partially Adequate Answers   :  15.0% ( 6/40 questions relevant but incomplete)   |
| 7. Deficient Answers            :   2.5% ( 1/40 questions completely off-topic)       |
+---------------------------------------------------------------------------------------+
```

### Breakdown of Deficient & Incomplete Cases:
1. **Deficient (1 Question — `EVAL-036`)**:
   - **Question**: *"Contrast the training-time optimization in paper-arxiv-2402.11794 with inference-time optimizations in paper-arxiv-2402.11626 and paper-arxiv-2510.22344."*
   - **Observed Answer**: Cites `paper-arxiv-2403.18684` (*Scaling Laws*) and `paper-arxiv-2608.08143` (*Touché Debate*).
   - **Diagnosis**: Retrieval returned zero candidates for the requested papers. The synthesizer extracted sentences from the retrieved chunks that were completely irrelevant to the prompt. **The system should have abstained due to missing entities**.
2. **Partially Adequate / Incomplete (6 Questions)**:
   - `EVAL-035`: One-sided comparison (TagRAG present, HetaRAG missing).
   - `EVAL-031`, `EVAL-032`, `EVAL-034`: In multi-paper aggregations, 1 out of 3 requested papers was omitted due to top-5 retrieval window cutoffs.
   - `EVAL-035` and `EVAL-036`: Chunk token window boundaries truncated sentences mid-clause (e.g. ending with `which extra` or `data augment`).

---

## 4. Deep-Dive on EVAL-035 & Abstention Policy

### Context
In Phase I, vector-only similarity score gating at 0.40 refused `EVAL-035` (*"Compare how TagRAG (paper-arxiv-2601.05254) and HetaRAG (paper-arxiv-2509.21336) incorporate non-vector structures to improve retrieval coverage"*). In Phase J, hybrid retrieval avoided false refusal and generated an answer.

### Audit Findings for EVAL-035:
- **Retrieved Chunks**:
  1. `e7bf42cf...` (TagRAG, winning rate 78.36%)
  2. `2afccf33...` (TagRAG, tag knowledge graph construction)
- **HetaRAG Presence**: **0 chunks** from `paper-arxiv-2509.21336` were included in the top-5 candidate set.
- **Synthesized Answer**:
  > *"Extensive experiments on UltraDomain datasets... demonstrate that TagRAG achieves an average winning rate of 78.36%... We propose TagRAG, a tag-guided hierarchical knowledge graph RAG framework... [2afccf33-fbd9-5443-b186-e8e107ce4979]"*
- **Audit Assessment**:
  - The answer is **100% citation-valid** and **100% quote-aligned** to TagRAG.
  - However, **it completely fails to compare with HetaRAG**. A user asking for a comparison receives information on only one model.
  - **Verdict**: In hybrid retrieval, avoiding score-based false refusal is an improvement, but without a **multi-entity coverage validator**, the system emits one-sided answers. The synthesizer should explicitly qualify: *"Information was retrieved for TagRAG, but no evidence was found for HetaRAG."*

---

## 5. Adversarial Test Suite Results

We implemented 11 adversarial tests in [`experiments/scipaper/tests/test_phase_j1_adversarial.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/tests/test_phase_j1_adversarial.py):
- `test_real_chunk_with_false_title`: PASSED (flags fabricated document title).
- `test_real_chunk_wrong_parent_doc`: PASSED (flags mismatched document ID).
- `test_valid_quote_unsupported_inference`: PASSED (flags inference not in quote).
- `test_fabricated_quote`: PASSED (flags non-existent text quote).
- `test_quote_omits_negation_vulnerability`: PASSED (verifies bag-of-words fallback vulnerability).
- `test_unrelated_verbatim_sentence`: PASSED (demonstrates that topical relevance is decoupled from grounding).
- `test_multiple_claims_sharing_one_citation`: PASSED (validates individual claims independently).
- `test_missing_citation_for_claim`: PASSED (flags claims without chunk IDs).
- `test_citation_to_nonexistent_chunk`: PASSED (flags non-existent chunk UUIDs).
- `test_comparison_one_side_only`: PASSED (exposes one-sided comparison vulnerability).
- `test_conflicting_claims_across_chunks`: PASSED (exposes lack of contradiction handling).

Total test suite across `experiments/scipaper/tests/`: **88 passed in 26.49s (0 failed)**.  
Ruff linting: passed. Ruff formatting: 67 files formatted. Mypy: 0 errors in 53 source files.

Full test output recorded in [`phase_j1_test_results.txt`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j1_test_results.txt).

---

## 6. Audit Metrics Summary

Quantified results saved in [`experiments/scipaper/eval/results/phase_j1_metrics.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j1_metrics.json):

| Metric Category | Metric Name | Value | Meaning |
|---|---|---:|---|
| **Out-of-Scope (10 Qs)** | Refusal Accuracy | **100.0%** (10/10) | All 10 out-of-scope questions correctly refused. |
| **Answerable (40 Qs)** | Verbatim Citation Validity | **100.0%** (83/83) | Every citation points to a real chunk with matching excerpt. |
| **Answerable (40 Qs)** | Question Relevance Rate | **97.5%** (39/40) | Answers address question topics (1 failure: EVAL-036). |
| **Answerable (40 Qs)** | Answer Completeness Rate | **82.5%** (33/40) | Answers fully cover all requested entities/parts (7 incomplete). |
| **Answerable (40 Qs)** | Fully Adequate Answers | **82.5%** (33/40) | Both relevant and complete. |
| **Answerable (40 Qs)** | Partially Adequate Answers | **15.0%** (6/40) | Relevant but missing one comparison side or truncated. |
| **Answerable (40 Qs)** | Deficient Answers | **2.5%** (1/40) | Off-topic retrieval output synthesized as answer. |
| **Adversarial Claims (12 Cases)** | Semantic Vulnerabilities Exposed | **8 / 12** | Extractive substring matching fails on attribution, negation, and roles. |

---

## 7. Deliverable Inventory

- Audit Report: [`experiments/scipaper/eval/PHASE_J1_ADVERSARIAL_AUDIT.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/PHASE_J1_ADVERSARIAL_AUDIT.md)
- Adversarial Engine: [`experiments/scipaper/eval/run_adversarial_audit.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/run_adversarial_audit.py)
- Adversarial Unit Tests: [`experiments/scipaper/tests/test_phase_j1_adversarial.py`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/tests/test_phase_j1_adversarial.py)
- Per-Question Review JSON: [`experiments/scipaper/eval/results/phase_j1_answer_review.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j1_answer_review.json)
- Adversarial Cases JSON: [`experiments/scipaper/eval/results/phase_j1_adversarial_cases.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j1_adversarial_cases.json)
- Metrics Summary JSON: [`experiments/scipaper/eval/results/phase_j1_metrics.json`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j1_metrics.json)
- Test Log Output: [`experiments/scipaper/eval/results/phase_j1_test_results.txt`](file:///c:/Users/workp/OneDrive/Desktop/kgr/experiments/scipaper/eval/results/phase_j1_test_results.txt)
- Session Memory: [`docs/MEMORY.md`](file:///c:/Users/workp/OneDrive/Desktop/kgr/docs/MEMORY.md)

---

## 8. Recommendations for Production Synthesizer

1. **Entity Coverage Gate**: In comparative questions mentioning multiple entities (e.g. `TagRAG` and `HetaRAG`), verify that retrieved candidates contain evidence for **all** requested entities before synthesizing. If only partial evidence is found, explicitly qualify the answer rather than presenting a one-sided extract.
2. **Disentangle Substring Matching from Claim Support**: Do not rely on bag-of-words or substring occurrence to declare a claim semantically supported. Check for negation modifiers (`not`, `fails to`, `prior to`) and sentence contribution context (`we propose` vs `prior work relies on`).
3. **Query-Candidate Relevance Gating**: When hybrid retrieval returns low-confidence candidates (e.g. `EVAL-036`), abstain from generation rather than extracting irrelevant sentences from whichever chunk ranked highest.
