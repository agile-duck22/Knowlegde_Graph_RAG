"""Evaluation Metrics Calculation for KG+Vector RAG vs Vector-Only Baseline (TASK-502).

Implements mathematical and semantic evaluation metrics strictly matching docs/EVALUATION.md:
1. Citation Precision:
   Precision = |Retrieved Cited Chunks ∩ Ground Truth Chunks| / |Retrieved Cited Chunks|
2. Citation Recall:
   Recall = |Retrieved Cited Chunks ∩ Ground Truth Chunks| / |Ground Truth Chunks|
3. Citation Hallucination Rate:
   Percentage of citations in the generated answer that do not correspond to retrieved chunks.
4. Retrieval Precision & Recall:
   Evaluating candidate retrieved chunks against ground truth chunk IDs.
5. Answer Fact Coverage (Correctness Proxy):
   Checks presence / coverage of expected facts in the generated answer.
6. Execution Latency & Cost Aggregations:
   Median (p50), 95th percentile (p95), mean latency, and USD cost per 1k queries.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field


class CaseMetrics(BaseModel):
    """Quantitative evaluation metrics computed for a single benchmark question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Retrieval Metrics (Candidate context vs Ground Truth)
    retrieval_precision: float = Field(
        ge=0.0, le=1.0, description="|Retrieved ∩ GroundTruth| / |Retrieved|"
    )
    retrieval_recall: float = Field(
        ge=0.0, le=1.0, description="|Retrieved ∩ GroundTruth| / |GroundTruth|"
    )

    # Citation Metrics (Generated answer citations vs Candidate Context and Ground Truth)
    citation_precision: float = Field(ge=0.0, le=1.0, description="|Cited ∩ GroundTruth| / |Cited|")
    citation_recall: float = Field(
        ge=0.0, le=1.0, description="|Cited ∩ GroundTruth| / |GroundTruth|"
    )
    citation_hallucination_rate: float = Field(
        ge=0.0, le=1.0, description="|Cited - RetrievedCandidates| / |Cited|"
    )

    # Answer Correctness / Fact Coverage
    fact_coverage: float = Field(
        ge=0.0, le=1.0, description="Proportion of expected ground-truth facts verified in answer"
    )
    all_facts_present: bool = Field(
        description="Whether 100% of expected facts are covered (Binary Accuracy)"
    )

    # Performance
    execution_time_ms: float = Field(ge=0.0, description="Latency in milliseconds")
    estimated_cost_usd: float = Field(ge=0.0, description="LLM synthesis cost in USD")
    total_tokens: int = Field(ge=0, description="Total prompt and completion tokens")


class AggregatedMetrics(BaseModel):
    """Aggregated benchmark statistics across multiple cases or within a category."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_cases: int = Field(ge=0, description="Total evaluated cases")
    successful_cases: int = Field(ge=0, description="Cases without system/runtime error")
    failed_cases: int = Field(ge=0, description="Cases that produced execution errors")

    # Accuracy / Fact Coverage
    accuracy: float = Field(
        ge=0.0, le=1.0, description="Proportion of cases with all_facts_present == True"
    )
    mean_fact_coverage: float = Field(ge=0.0, le=1.0, description="Mean fact coverage ratio")

    # Citation Metrics
    mean_citation_precision: float = Field(ge=0.0, le=1.0, description="Mean citation precision")
    mean_citation_recall: float = Field(ge=0.0, le=1.0, description="Mean citation recall")
    mean_citation_hallucination_rate: float = Field(
        ge=0.0, le=1.0, description="Mean citation hallucination rate"
    )

    # Retrieval Metrics
    mean_retrieval_precision: float = Field(ge=0.0, le=1.0, description="Mean retrieval precision")
    mean_retrieval_recall: float = Field(ge=0.0, le=1.0, description="Mean retrieval recall")

    # Latency & Cost
    latency_p50_ms: float = Field(ge=0.0, description="Median latency in milliseconds")
    latency_p95_ms: float = Field(ge=0.0, description="95th percentile latency in milliseconds")
    mean_latency_ms: float = Field(ge=0.0, description="Mean latency in milliseconds")
    total_cost_usd: float = Field(ge=0.0, description="Total USD cost across all queries")
    cost_per_1k_queries_usd: float = Field(
        ge=0.0, description="Extrapolated USD cost for 1,000 queries"
    )


def compute_citation_precision(
    cited_chunk_ids: Sequence[str],
    ground_truth_chunk_ids: Sequence[str],
    is_out_of_scope: bool = False,
) -> float:
    """Compute citation precision = |Cited ∩ GroundTruth| / |Cited|.

    Rules:
    - Out of scope: If cited is empty, precision is 1.0 (correctly cited nothing).
      If model cited something for OOS, precision is 0.0.
    - In scope: If cited is empty, returns 0.0.
    """
    cited_set = set(cited_chunk_ids)
    gt_set = set(ground_truth_chunk_ids)

    if is_out_of_scope:
        return 1.0 if not cited_set else 0.0

    if not cited_set:
        return 0.0

    intersection = cited_set.intersection(gt_set)
    return len(intersection) / len(cited_set)


def compute_citation_recall(
    cited_chunk_ids: Sequence[str],
    ground_truth_chunk_ids: Sequence[str],
    is_out_of_scope: bool = False,
) -> float:
    """Compute citation recall = |Cited ∩ GroundTruth| / |GroundTruth|.

    Rules:
    - Out of scope: GT is empty. If cited is empty, recall is 1.0; else 0.0.
    - In scope: If GT is empty, returns 1.0 if cited is empty else 0.0.
    """
    cited_set = set(cited_chunk_ids)
    gt_set = set(ground_truth_chunk_ids)

    if is_out_of_scope or not gt_set:
        return 1.0 if not cited_set else 0.0

    intersection = cited_set.intersection(gt_set)
    return len(intersection) / len(gt_set)


def compute_citation_hallucination_rate(
    cited_chunk_ids: Sequence[str],
    candidate_chunk_ids: Sequence[str],
) -> float:
    """Compute hallucination rate = |Cited - CandidateChunkIDs| / |Cited|.

    If no citations were emitted, hallucination rate is 0.0.
    """
    cited_set = set(cited_chunk_ids)
    if not cited_set:
        return 0.0

    candidates_set = set(candidate_chunk_ids)
    hallucinated = cited_set - candidates_set
    return len(hallucinated) / len(cited_set)


def compute_retrieval_precision_recall(
    retrieved_chunk_ids: Sequence[str],
    ground_truth_chunk_ids: Sequence[str],
    is_out_of_scope: bool = False,
) -> tuple[float, float]:
    """Compute (precision, recall) for retrieved candidate chunks against ground truth."""
    retrieved_set = set(retrieved_chunk_ids)
    gt_set = set(ground_truth_chunk_ids)

    if is_out_of_scope:
        prec = 1.0 if not retrieved_set else 0.0
        rec = 1.0 if not retrieved_set else 0.0
        return prec, rec

    if not retrieved_set:
        prec = 0.0
    else:
        prec = len(retrieved_set.intersection(gt_set)) / len(retrieved_set)

    if not gt_set:
        rec = 1.0 if not retrieved_set else 0.0
    else:
        rec = len(retrieved_set.intersection(gt_set)) / len(gt_set)

    return prec, rec


def compute_fact_coverage(
    answer_text: str,
    expected_facts: Sequence[str],
    is_out_of_scope: bool = False,
) -> tuple[float, bool]:
    """Evaluate factual proposition presence in generated answer.

    For out-of-scope queries:
    - Verifies refusal / out-of-scope phrasing (e.g. 'outside the scope', 'not find relevant', 'refuse').

    For in-scope queries:
    - Performs keyword / token overlap checks for expected fact sentences.

    Returns:
        tuple[fact_coverage_ratio, all_facts_present_bool]
    """
    if is_out_of_scope:
        refusal_phrases = [
            "outside the scope",
            "could not find",
            "cannot find",
            "not find relevant",
            "no information",
            "unsupported",
            "refuse",
            "not mentioned",
            "outside of the enterprise",
        ]
        lower_ans = answer_text.lower()
        is_refusal = any(phrase in lower_ans for phrase in refusal_phrases)
        return (1.0, True) if is_refusal else (0.0, False)

    if not expected_facts:
        return 1.0, True

    lower_ans = answer_text.lower()
    covered_count = 0

    for fact in expected_facts:
        # Extract keywords of length > 2 from the fact string, splitting by punctuation
        raw_words = re.findall(r"[a-zA-Z0-9_\-]+", fact)
        words = [w.lower() for w in raw_words if len(w) > 2]
        if not words:
            covered_count += 1
            continue

        # If >= 50% of significant fact terms occur in the answer, consider the fact covered
        matched_words = sum(1 for w in words if w in lower_ans)
        match_ratio = matched_words / len(words)
        if match_ratio >= 0.5:
            covered_count += 1

    ratio = covered_count / len(expected_facts)
    return ratio, (covered_count == len(expected_facts))


def calculate_case_metrics(
    retrieved_chunk_ids: Sequence[str],
    cited_chunk_ids: Sequence[str],
    ground_truth_chunk_ids: Sequence[str],
    expected_facts: Sequence[str],
    answer_text: str,
    execution_time_ms: float,
    estimated_cost_usd: float = 0.0,
    total_tokens: int = 0,
    is_out_of_scope: bool = False,
) -> CaseMetrics:
    """Compute all case-level metrics from execution components."""
    ret_prec, ret_rec = compute_retrieval_precision_recall(
        retrieved_chunk_ids=retrieved_chunk_ids,
        ground_truth_chunk_ids=ground_truth_chunk_ids,
        is_out_of_scope=is_out_of_scope,
    )
    cit_prec = compute_citation_precision(
        cited_chunk_ids=cited_chunk_ids,
        ground_truth_chunk_ids=ground_truth_chunk_ids,
        is_out_of_scope=is_out_of_scope,
    )
    cit_rec = compute_citation_recall(
        cited_chunk_ids=cited_chunk_ids,
        ground_truth_chunk_ids=ground_truth_chunk_ids,
        is_out_of_scope=is_out_of_scope,
    )
    cit_halluc = compute_citation_hallucination_rate(
        cited_chunk_ids=cited_chunk_ids,
        candidate_chunk_ids=retrieved_chunk_ids,
    )
    fact_cov, all_facts = compute_fact_coverage(
        answer_text=answer_text,
        expected_facts=expected_facts,
        is_out_of_scope=is_out_of_scope,
    )

    return CaseMetrics(
        retrieval_precision=round(ret_prec, 4),
        retrieval_recall=round(ret_rec, 4),
        citation_precision=round(cit_prec, 4),
        citation_recall=round(cit_rec, 4),
        citation_hallucination_rate=round(cit_halluc, 4),
        fact_coverage=round(fact_cov, 4),
        all_facts_present=all_facts,
        execution_time_ms=round(execution_time_ms, 2),
        estimated_cost_usd=round(estimated_cost_usd, 6),
        total_tokens=total_tokens,
    )


def aggregate_case_metrics(
    metrics_list: Sequence[CaseMetrics], failed_count: int = 0
) -> AggregatedMetrics:
    """Aggregate a sequence of CaseMetrics into overall statistics."""
    total = len(metrics_list) + failed_count
    if not metrics_list:
        return AggregatedMetrics(
            total_cases=total,
            successful_cases=0,
            failed_cases=failed_count,
            accuracy=0.0,
            mean_fact_coverage=0.0,
            mean_citation_precision=0.0,
            mean_citation_recall=0.0,
            mean_citation_hallucination_rate=0.0,
            mean_retrieval_precision=0.0,
            mean_retrieval_recall=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            mean_latency_ms=0.0,
            total_cost_usd=0.0,
            cost_per_1k_queries_usd=0.0,
        )

    n_succ = len(metrics_list)
    accuracy = (
        sum(1 for m in metrics_list if m.all_facts_present) / total
    )  # Penalize failures against total
    mean_fact_cov = sum(m.fact_coverage for m in metrics_list) / n_succ
    mean_cit_prec = sum(m.citation_precision for m in metrics_list) / n_succ
    mean_cit_rec = sum(m.citation_recall for m in metrics_list) / n_succ
    mean_cit_halluc = sum(m.citation_hallucination_rate for m in metrics_list) / n_succ
    mean_ret_prec = sum(m.retrieval_precision for m in metrics_list) / n_succ
    mean_ret_rec = sum(m.retrieval_recall for m in metrics_list) / n_succ

    latencies = sorted(m.execution_time_ms for m in metrics_list)
    p50_idx = int(len(latencies) * 0.50)
    p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p50 = latencies[p50_idx]
    p95 = latencies[p95_idx]
    mean_lat = sum(latencies) / n_succ

    total_cost = sum(m.estimated_cost_usd for m in metrics_list)
    cost_per_1k = (total_cost / n_succ) * 1000.0 if n_succ > 0 else 0.0

    return AggregatedMetrics(
        total_cases=total,
        successful_cases=n_succ,
        failed_cases=failed_count,
        accuracy=round(accuracy, 4),
        mean_fact_coverage=round(mean_fact_cov, 4),
        mean_citation_precision=round(mean_cit_prec, 4),
        mean_citation_recall=round(mean_cit_rec, 4),
        mean_citation_hallucination_rate=round(mean_cit_halluc, 4),
        mean_retrieval_precision=round(mean_ret_prec, 4),
        mean_retrieval_recall=round(mean_ret_rec, 4),
        latency_p50_ms=round(p50, 2),
        latency_p95_ms=round(p95, 2),
        mean_latency_ms=round(mean_lat, 2),
        total_cost_usd=round(total_cost, 6),
        cost_per_1k_queries_usd=round(cost_per_1k, 4),
    )
