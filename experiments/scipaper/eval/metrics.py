"""Offline Metric Calculation Utilities for Scientific RAG Evaluation.

Defines explicit, deterministic metrics:
- Hit Rate@k: fraction of queries with at least one gold chunk in top k.
- Recall@k: fraction of gold chunks retrieved within top k, averaged across answerable queries.
- MRR (Mean Reciprocal Rank): 1 / rank of first retrieved gold chunk (0 if not found), averaged across queries.
- Out-of-scope handling: evaluated separately, never mixed into answerable retrieval averages.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionRetrievalEvaluation:
    question_id: str
    category: str
    is_out_of_scope: bool
    gold_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    similarity_scores: list[float] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    ranks: list[int] = field(default_factory=list)

    # Computed metrics
    reciprocal_rank: float = 0.0
    hits_at_k: dict[int, bool] = field(default_factory=dict)
    recall_at_k: dict[int, float] = field(default_factory=dict)

    def evaluate(self, k_values: Sequence[int] = (1, 3, 5, 10)) -> None:
        if self.is_out_of_scope or not self.gold_chunk_ids:
            self.reciprocal_rank = 0.0
            for k in k_values:
                self.hits_at_k[k] = False
                self.recall_at_k[k] = 0.0
            return

        gold_set = set(self.gold_chunk_ids)

        # Reciprocal Rank (1-indexed)
        first_rank: int | None = None
        for idx, cid in enumerate(self.retrieved_chunk_ids, start=1):
            if cid in gold_set:
                first_rank = idx
                break
        self.reciprocal_rank = 1.0 / first_rank if first_rank is not None else 0.0

        # Hits@k and Recall@k
        for k in k_values:
            top_k_retrieved = set(self.retrieved_chunk_ids[:k])
            intersect = top_k_retrieved.intersection(gold_set)
            self.hits_at_k[k] = len(intersect) > 0
            self.recall_at_k[k] = len(intersect) / len(gold_set)


@dataclass
class AggregatedMetrics:
    total_questions: int
    evaluated_questions: int
    mean_reciprocal_rank: float
    hit_rate: dict[int, float]
    mean_recall: dict[int, float]
    zero_result_queries: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "evaluated_questions": self.evaluated_questions,
            "mrr": round(self.mean_reciprocal_rank, 4),
            "hit_rate": {f"hit_rate@{k}": round(v, 4) for k, v in self.hit_rate.items()},
            "recall": {f"recall@{k}": round(v, 4) for k, v in self.mean_recall.items()},
            "zero_result_queries": self.zero_result_queries,
        }


def compute_metrics(
    evaluations: Sequence[QuestionRetrievalEvaluation],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> AggregatedMetrics:
    """Compute aggregate retrieval metrics over answerable questions."""
    answerable = [e for e in evaluations if not e.is_out_of_scope and e.gold_chunk_ids]

    if not answerable:
        return AggregatedMetrics(
            total_questions=len(evaluations),
            evaluated_questions=0,
            mean_reciprocal_rank=0.0,
            hit_rate=dict.fromkeys(k_values, 0.0),
            mean_recall=dict.fromkeys(k_values, 0.0),
            zero_result_queries=sum(1 for e in evaluations if len(e.retrieved_chunk_ids) == 0),
        )

    for e in answerable:
        e.evaluate(k_values)

    n = len(answerable)
    mrr = sum(e.reciprocal_rank for e in answerable) / n
    hit_rate: dict[int, float] = {}
    mean_recall: dict[int, float] = {}

    for k in k_values:
        hit_rate[k] = sum(1 for e in answerable if e.hits_at_k[k]) / n
        mean_recall[k] = sum(e.recall_at_k[k] for e in answerable) / n

    zero_results = sum(1 for e in answerable if len(e.retrieved_chunk_ids) == 0)

    return AggregatedMetrics(
        total_questions=len(evaluations),
        evaluated_questions=n,
        mean_reciprocal_rank=mrr,
        hit_rate=hit_rate,
        mean_recall=mean_recall,
        zero_result_queries=zero_results,
    )


def compute_per_category_metrics(
    evaluations: Sequence[QuestionRetrievalEvaluation],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, dict[str, Any]]:
    """Compute retrieval metrics grouped by benchmark category."""
    by_category: dict[str, list[QuestionRetrievalEvaluation]] = {}
    for e in evaluations:
        by_category.setdefault(e.category, []).append(e)

    results: dict[str, dict[str, Any]] = {}
    for cat, evals in by_category.items():
        if cat == "out_of_scope":
            results[cat] = {
                "total_questions": len(evals),
                "is_out_of_scope": True,
                "notes": "Out-of-scope queries evaluated separately; retrieval recall/hit-rate is non-applicable.",
                "retrieved_results_count": [len(e.retrieved_chunk_ids) for e in evals],
            }
        else:
            cat_metrics = compute_metrics(evals, k_values=k_values)
            results[cat] = cat_metrics.to_dict()

    return results
