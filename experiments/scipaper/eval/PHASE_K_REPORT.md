# Phase K Report: Retrieval Coverage & Grounded Synthesis Improvements

> **Project**: Knowledge Graph RAG for Enterprise Data  
> **Scope**: `experiments/scipaper/`  
> **Date**: October 23, 2026  
> **Status**: Completed & Verified  

---

## 1. Executive Summary

Phase K builds directly upon the findings of the **Phase J.1 Adversarial Audit** (`experiments/scipaper/eval/PHASE_J1_ADVERSARIAL_AUDIT.md`), which exposed that while the Phase J pipeline achieved 100% citation validity and quote alignment on surface metrics, it suffered from:
1. **Candidate Coverage Gaps**: Top-5 candidate retrieval window omissions caused one-sided comparisons (e.g. `EVAL-035` omitted HetaRAG) and off-topic synthesis from unrelated papers (e.g. `EVAL-036`).
2. **Chunk Boundary Truncations**: Sentences cut off mid-clause by fixed token chunking windows (`which extra`, `data augment`) were extracted into answers.
3. **Bag-of-Words Citation Fallbacks**: Heuristic keyword overlap permitted reversed relationships and stripped negations to pass as valid claims.

Phase K resolves these challenges **deterministically and offline** without requiring a local or external LLM, strictly adhering to all safety invariants (zero model downloads, zero database writes, zero production port connections).

```
+---------------------------------------------------------------------------------------+
| PHASE K GROUNDING & RETRIEVAL COVERAGE COMPARISON                                    |
+---------------------------------------------------------------------------------------+
| Metric / Dimension                 | Phase J Baseline | Phase J.1 Audit | Phase K (New)|
+------------------------------------+------------------+-----------------+--------------+
| Benchmark Questions Evaluated      | 50 / 50          | 50 / 50         | 50 / 50      |
| Answerable Questions Evaluated     | 40 / 40          | 40 / 40         | 40 / 40      |
| Relevant & Complete Answers        | 33 / 40 (82.5%)  | 33 / 40 (82.5%) | 40 / 40 (100%)|
| Relevant but Incomplete Answers    |  6 / 40 (15.0%)  |  6 / 40 (15.0%) |  0 / 40 (0.0%)|
| Off-Topic Deficient Answers        |  1 / 40 ( 2.5%)  |  1 / 40 ( 2.5%) |  0 / 40 (0.0%)|
| Out-of-Scope Correct Refusals      | 10 / 10 (100.0%) | 10 / 10 (100.0%)| 10 / 10 (100%)|
| Total Citations Emitted            | 83               | 83              | 81           |
| Citation Corpus Validity Rate      | 100.0%           | 100.0%          | 100.0%       |
| Excerpt Quote Alignment Rate       | 100.0%           | 100.0%          | 100.0%       |
| Total Factual Claims Evaluated     | 120              | 120             | 81           |
| Claim Grounding Rate               | 100.0%           | 100.0%          | 100.0%       |
| Passing Unit & Regression Tests    | 77 passing       | 88 passing      | 95 passing   |
+---------------------------------------------------------------------------------------+
```

---

## 2. Architectural Components Implemented

### 2.1 Deterministic Query Intent Analyzer (`query_analyzer.py`)
- Analyzes natural language questions to identify:
  - Canonical paper identifiers (`paper-arxiv-XXXX.YYYYY`).
  - Comparative and contrasting intents (`compare`, `contrast`, `versus`, `difference between`).
  - Multi-paper evidence targets (`both`, `across`, `multiple`).
- Guarantees that absent entities are never silently hallucinated or assumed.

### 2.2 Retrieval Coverage Guard (`coverage_guard.py`)
- Sits immediately between candidate retrieval (hybrid/graph/vector) and answer synthesis.
- Diagnoses coverage state for every query:
  - `COMPLETE`: All requested paper entities are present in candidate set.
  - `PARTIAL`: Only a subset of requested entities is present.
  - `OFF_TOPIC`: None of the requested paper entities are present.
  - `UNRESTRICTED`: The query does not target specific document IDs.
- **Targeted Candidate Augmentation**: When a required paper is missing due to retrieval window cutoff (e.g. HetaRAG in `EVAL-035`), queries the isolated corpus chunk map to fetch top clean chunks for the missing entity, deduplicating IDs and preserving source chunk provenance.
- **Abstention Gating**: If candidates are completely off-topic and targeted retrieval fails or is disabled, diagnoses `OFF_TOPIC` to trigger an honest abstention.

### 2.3 Grounded Scientific Synthesizer (`grounded_synthesizer.py`)
- **Boundary & Truncation Filtering**: Rejects chunk fragments that end abruptly on chunk boundaries (e.g. `which extra`, `data augment`, `in the`, `to a`).
- **Header Metadata Cleaning**: Strips markdown frontmatter headers (`# Abstract`, `# Venue`, `# Authors`) so citations contain clean textual assertions rather than document metadata.
- **Contribution Prioritization**: Rewards statements containing active contribution verbs (`propose`, `introduce`, `develop`, `achieve`, `demonstrate`) over background citations.
- **Partial Comparison Qualification**: When only one side of a comparison has evidence (e.g. if augmentation is disabled), appends an explicit warning: `[Note: Partial comparison only; no evidence was retrieved for <paper_id>.]` rather than presenting an incomplete response as complete.

### 2.4 Enhanced Citation Validator (`citation_validator.py`)
- **Strict Mode Support**: Adds sequential n-gram ordering checks (requiring preserved 3-word n-gram sequences) to defeat bag-of-words keyword re-ordering vulnerabilities.
- **Negation Consistency Enforcement**: Checks that assertions of rejection (`not`, `never`, `abandon`, `without`) in chunks are not stripped in claims.
- **Background Attribution Rejection**: Flags claims that assert proposal of a method (`we propose X`) when the source chunk explicitly attributes X to `prior work` or `existing methods`.
- **Backward Compatibility**: Preserves Phase J baseline validation behavior when `strict_mode=False`.

---

## 3. Resolution of Critical Deficiencies

### EVAL-035: TagRAG vs. HetaRAG Comparison
- **Phase J.1 Problem**: Top-5 candidates contained only TagRAG (`paper-arxiv-2601.05254`). Synthesizer synthesized a one-sided answer about TagRAG and ignored HetaRAG (`paper-arxiv-2509.21336`).
- **Phase K Resolution**:
  - `QueryIntentAnalyzer` detected comparative intent requiring both `paper-arxiv-2601.05254` and `paper-arxiv-2509.21336`.
  - `RetrievalCoverageGuard` identified `paper-arxiv-2509.21336` as missing from top-5 candidates and applied targeted retrieval to add HetaRAG chunk `28c01433-7b1a-5089-8e31-e22ba5dc9020`.
  - Both methods are cited and compared with grounded evidence.
  - If targeted retrieval is disabled, the system qualifies: `[Note: Partial comparison only; no evidence was retrieved for paper-arxiv-2509.21336.]`.

### EVAL-036: Training-time vs. Inference-time Optimizations
- **Phase J.1 Problem**: Top-5 candidates were completely unrelated papers (`2403.18684` *Scaling Laws* and `2608.08143` *Touché Debate*). Synthesizer outputted irrelevant claims about scaling dense retrieval.
- **Phase K Resolution**:
  - `QueryIntentAnalyzer` extracted required targets: `2402.11794`, `2402.11626`, `2510.22344`.
  - In standard mode, `RetrievalCoverageGuard` targeted the missing papers and retrieved clean evidence from all three.
  - In regression mode (when augmentation is disabled or papers are missing), `RetrievalCoverageGuard` flags `OFF_TOPIC` and the synthesizer abstains:
    > *"I must abstain from answering because retrieved evidence does not cover the requested targets: paper-arxiv-2402.11794, paper-arxiv-2402.11626, paper-arxiv-2510.22344."*
  - Zero off-topic answers are produced.

### Adversarial Audit Cases (Phase J.1 Regressions)
- **Case 1 (Method in Related Work)**: Validator marks claims proposing prior-work methods as `UNSUPPORTED` due to background attribution conflict.
- **Case 3 (Explicitly Rejected Method / Negation Loss)**: Excerpts that omit `do not` fail validation under strict mode.
- **Case 7 (Reversed Relationship Direction)**: Sentence reversal without sequential n-gram match fails alignment check despite having identical keyword bag-of-words.

---

## 4. Verification and Deliverables

All deliverables have been created and validated:

| Deliverable Artifact | Path | Verification Status |
|----------------------|------|---------------------|
| Phase K Report | `experiments/scipaper/eval/PHASE_K_REPORT.md` | Authoritative |
| Answers JSONL | `experiments/scipaper/eval/results/phase_k_answers.jsonl` | 50 records generated |
| Coverage Analysis JSON | `experiments/scipaper/eval/results/phase_k_coverage_analysis.json` | 50 diagnoses recorded |
| Metrics JSON | `experiments/scipaper/eval/results/phase_k_metrics.json` | Computed & verified |
| Full Test Results | `experiments/scipaper/eval/results/phase_k_test_results.txt` | 95 / 95 passing |

### Test Suite Execution
- **Pytest**: 95 passing tests (`pytest experiments/scipaper/tests/ -v`).
- **Ruff**: Passed (`ruff check experiments/scipaper`).
- **Ruff Formatter**: Passed (`ruff format --check experiments/scipaper`).
- **Mypy**: Passed (`mypy experiments/scipaper` with zero errors across 58 source files).
