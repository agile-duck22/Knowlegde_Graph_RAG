"""Evaluation and Benchmarking package exports (TASK-501 & TASK-502)."""

from __future__ import annotations

from src.eval.metrics import (
    AggregatedMetrics,
    CaseMetrics,
    aggregate_case_metrics,
    calculate_case_metrics,
    compute_citation_hallucination_rate,
    compute_citation_precision,
    compute_citation_recall,
    compute_fact_coverage,
    compute_retrieval_precision_recall,
)
from src.eval.runner import (
    BenchmarkRunReport,
    EvaluationCaseResult,
    EvaluationRunner,
    EvaluationSystemMode,
)
from src.eval.schema import (
    BenchmarkCategory,
    BenchmarkDataset,
    BenchmarkDifficulty,
    BenchmarkQuestion,
    load_benchmark_dataset,
)

__all__ = [
    "AggregatedMetrics",
    "BenchmarkCategory",
    "BenchmarkDataset",
    "BenchmarkDifficulty",
    "BenchmarkQuestion",
    "BenchmarkRunReport",
    "CaseMetrics",
    "EvaluationCaseResult",
    "EvaluationRunner",
    "EvaluationSystemMode",
    "aggregate_case_metrics",
    "calculate_case_metrics",
    "compute_citation_hallucination_rate",
    "compute_citation_precision",
    "compute_citation_recall",
    "compute_fact_coverage",
    "compute_retrieval_precision_recall",
    "load_benchmark_dataset",
]
