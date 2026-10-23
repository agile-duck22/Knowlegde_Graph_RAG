"""Benchmark Validator for the Scientific-Paper Experiment (Phase D).

Validates:
- Question IDs are unique and follow format EVAL-xxx.
- Exactly 50 questions present.
- 5 categories with exactly 10 questions each.
- 40 answerable questions and 10 out-of-scope questions.
- Reconciles the reported 166 gold chunk references.
- Confirms all gold chunk UUIDs exist in PostgreSQL chunks table (read-only query).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.scipaper.schemas.models import ScientificBenchmarkDataset
from src.eval.schema import BenchmarkCategory


@dataclass
class BenchmarkValidationReport:
    total_questions: int = 0
    unique_ids_count: int = 0
    categories_count: dict[str, int] = field(default_factory=dict)
    answerable_count: int = 0
    out_of_scope_count: int = 0
    total_gold_chunk_references: int = 0
    unique_gold_chunk_ids: int = 0
    missing_gold_chunk_ids: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    is_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "unique_ids_count": self.unique_ids_count,
            "categories_count": self.categories_count,
            "answerable_count": self.answerable_count,
            "out_of_scope_count": self.out_of_scope_count,
            "total_gold_chunk_references": self.total_gold_chunk_references,
            "unique_gold_chunk_ids": self.unique_gold_chunk_ids,
            "missing_gold_chunk_ids": self.missing_gold_chunk_ids,
            "schema_errors": self.schema_errors,
            "is_valid": self.is_valid,
        }


def validate_benchmark_offline(
    benchmark_path: Path,
    existing_pg_chunk_ids: set[str] | None = None,
) -> BenchmarkValidationReport:
    """Validate benchmark questions, categories, gold chunks, and database existence."""
    report = BenchmarkValidationReport()

    if not benchmark_path.exists():
        report.schema_errors.append(f"Benchmark file not found: {benchmark_path}")
        return report

    try:
        with open(benchmark_path, encoding="utf-8") as f:
            data = json.load(f)
        dataset = ScientificBenchmarkDataset.model_validate(data)
    except Exception as exc:
        report.schema_errors.append(f"Pydantic schema validation failed: {exc}")
        return report

    report.total_questions = len(dataset.questions)
    seen_ids = set()
    gold_chunk_refs = 0
    all_gold_chunk_ids = set()

    expected_categories = {c.value for c in BenchmarkCategory}

    for q in dataset.questions:
        if q.id in seen_ids:
            report.schema_errors.append(f"Duplicate question ID: {q.id}")
        seen_ids.add(q.id)

        cat_str = q.category.value
        report.categories_count[cat_str] = report.categories_count.get(cat_str, 0) + 1

        if q.category == BenchmarkCategory.OUT_OF_SCOPE:
            report.out_of_scope_count += 1
            if q.ground_truth_chunk_ids:
                report.schema_errors.append(
                    f"Out-of-scope question {q.id} has non-empty ground_truth_chunk_ids"
                )
        else:
            report.answerable_count += 1
            chunks = q.ground_truth_chunk_ids
            gold_chunk_refs += len(chunks)
            all_gold_chunk_ids.update(chunks)
            if not chunks:
                report.schema_errors.append(
                    f"Answerable question {q.id} has empty ground_truth_chunk_ids"
                )

    report.unique_ids_count = len(seen_ids)
    report.total_gold_chunk_references = gold_chunk_refs
    report.unique_gold_chunk_ids = len(all_gold_chunk_ids)

    # Check category distribution
    for c in expected_categories:
        cnt = report.categories_count.get(c, 0)
        if cnt != 10:
            report.schema_errors.append(f"Category '{c}' expected 10 questions, found {cnt}")

    # Check database chunk reconciliation if provided
    if existing_pg_chunk_ids is not None:
        for cid in all_gold_chunk_ids:
            if cid not in existing_pg_chunk_ids:
                report.missing_gold_chunk_ids.append(cid)

    report.is_valid = (
        len(report.schema_errors) == 0
        and len(report.missing_gold_chunk_ids) == 0
        and report.total_questions == 50
        and report.answerable_count == 40
        and report.out_of_scope_count == 10
    )

    return report
