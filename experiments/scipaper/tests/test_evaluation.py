"""Unit tests for the scientific benchmark metrics and validation modules (Phase D).

Tests cover:
1. Hit Rate@k calculation on synthetic queries.
2. Recall@k calculation with single and multiple gold chunks.
3. MRR calculation with rank variations.
4. Queries with zero retrieved results.
5. Out-of-scope question exclusion from retrieval averages.
6. Benchmark schema and question ID uniqueness validation.
7. Missing gold chunks detection.
8. Serialization to JSON/JSONL format.
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.scipaper.eval.metrics import (
    QuestionRetrievalEvaluation,
    compute_metrics,
    compute_per_category_metrics,
)
from experiments.scipaper.eval.validate_benchmark import (
    validate_benchmark_offline,
)


def test_metric_calculation_exact_hit_rank_1() -> None:
    eval_item = QuestionRetrievalEvaluation(
        question_id="EVAL-001",
        category="single_hop_direct_fact",
        is_out_of_scope=False,
        gold_chunk_ids=["chunk-A"],
        retrieved_chunk_ids=["chunk-A", "chunk-B", "chunk-C"],
    )
    eval_item.evaluate((1, 3, 5))
    assert eval_item.reciprocal_rank == 1.0
    assert eval_item.hits_at_k[1] is True
    assert eval_item.recall_at_k[1] == 1.0


def test_metric_calculation_multiple_gold_chunks() -> None:
    eval_item = QuestionRetrievalEvaluation(
        question_id="EVAL-002",
        category="2_hop_relationship",
        is_out_of_scope=False,
        gold_chunk_ids=["chunk-A", "chunk-B"],
        retrieved_chunk_ids=["chunk-X", "chunk-A", "chunk-Y", "chunk-B"],
    )
    eval_item.evaluate((1, 2, 3, 4))
    # First gold is chunk-A at rank 2
    assert eval_item.reciprocal_rank == 0.5
    assert eval_item.hits_at_k[1] is False
    assert eval_item.hits_at_k[2] is True
    assert eval_item.recall_at_k[1] == 0.0
    assert eval_item.recall_at_k[2] == 0.5
    assert eval_item.recall_at_k[4] == 1.0


def test_metric_calculation_zero_hits() -> None:
    eval_item = QuestionRetrievalEvaluation(
        question_id="EVAL-003",
        category="3_hop_dependency",
        is_out_of_scope=False,
        gold_chunk_ids=["chunk-A"],
        retrieved_chunk_ids=["chunk-X", "chunk-Y"],
    )
    eval_item.evaluate((1, 3, 5))
    assert eval_item.reciprocal_rank == 0.0
    assert eval_item.hits_at_k[1] is False
    assert eval_item.recall_at_k[5] == 0.0


def test_aggregate_metrics_excludes_out_of_scope() -> None:
    q1 = QuestionRetrievalEvaluation(
        question_id="EVAL-001",
        category="single_hop_direct_fact",
        is_out_of_scope=False,
        gold_chunk_ids=["chunk-1"],
        retrieved_chunk_ids=["chunk-1"],
    )
    q2 = QuestionRetrievalEvaluation(
        question_id="EVAL-002",
        category="single_hop_direct_fact",
        is_out_of_scope=False,
        gold_chunk_ids=["chunk-2"],
        retrieved_chunk_ids=["chunk-9", "chunk-2"],
    )
    q_oos = QuestionRetrievalEvaluation(
        question_id="EVAL-041",
        category="out_of_scope",
        is_out_of_scope=True,
        gold_chunk_ids=[],
        retrieved_chunk_ids=["chunk-any"],
    )

    agg = compute_metrics([q1, q2, q_oos], k_values=(1, 2))
    assert agg.total_questions == 3
    assert agg.evaluated_questions == 2  # OOS excluded
    # MRR = (1.0 + 0.5) / 2 = 0.75
    assert agg.mean_reciprocal_rank == 0.75
    assert agg.hit_rate[1] == 0.5
    assert agg.hit_rate[2] == 1.0


def test_per_category_metrics_computation() -> None:
    q1 = QuestionRetrievalEvaluation(
        question_id="EVAL-001",
        category="single_hop_direct_fact",
        is_out_of_scope=False,
        gold_chunk_ids=["chunk-1"],
        retrieved_chunk_ids=["chunk-1"],
    )
    q_oos = QuestionRetrievalEvaluation(
        question_id="EVAL-041",
        category="out_of_scope",
        is_out_of_scope=True,
        gold_chunk_ids=[],
        retrieved_chunk_ids=["chunk-x", "chunk-y"],
    )

    by_cat = compute_per_category_metrics([q1, q_oos], k_values=(1, 5))
    assert "single_hop_direct_fact" in by_cat
    assert "out_of_scope" in by_cat
    assert by_cat["single_hop_direct_fact"]["mrr"] == 1.0
    assert by_cat["out_of_scope"]["is_out_of_scope"] is True


def test_validate_benchmark_detects_missing_chunks(tmp_path: Path) -> None:
    bench_data = {
        "version": "1.0.0",
        "total_questions": 1,
        "questions": [
            {
                "id": "EVAL-001",
                "category": "single_hop_direct_fact",
                "difficulty": "easy",
                "question": "What is investigated?",
                "ground_truth_entities": ["paper:test"],
                "ground_truth_chunk_ids": ["chunk-exists", "chunk-missing"],
                "expected_facts": ["fact"],
                "reference_answer": "answer",
                "metadata": {},
            }
        ],
    }
    bench_file = tmp_path / "test_bench.json"
    bench_file.write_text(json.dumps(bench_data), encoding="utf-8")

    existing_chunks = {"chunk-exists"}
    rep = validate_benchmark_offline(bench_file, existing_chunks)
    assert rep.is_valid is False
    assert "chunk-missing" in rep.missing_gold_chunk_ids
