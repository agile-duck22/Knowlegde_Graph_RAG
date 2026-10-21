"""Benchmark Dataset Models and Loader (TASK-501).

Defines standard benchmark item schema, category taxonomy, and loader utility
matching docs/EVALUATION.md.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BenchmarkCategory(StrEnum):
    """Evaluation taxonomy category for benchmark questions."""

    SINGLE_HOP_DIRECT_FACT = "single_hop_direct_fact"
    TWO_HOP_RELATIONSHIP = "2_hop_relationship"
    THREE_HOP_DEPENDENCY = "3_hop_dependency"
    RELATIONSHIP_AGGREGATION = "relationship_aggregation"
    OUT_OF_SCOPE = "out_of_scope"


class BenchmarkDifficulty(StrEnum):
    """Difficulty rating of benchmark question."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BenchmarkQuestion(BaseModel):
    """Single curated benchmark question with ground truth annotations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., pattern=r"^EVAL-\d{3}$", description="Unique question ID (e.g. EVAL-001)")
    category: BenchmarkCategory = Field(..., description="Taxonomy category")
    difficulty: BenchmarkDifficulty = Field(
        default=BenchmarkDifficulty.MEDIUM, description="Question difficulty"
    )
    question: str = Field(..., min_length=5, description="Benchmark natural language inquiry")
    ground_truth_entities: list[str] = Field(
        default_factory=list, description="Canonical entity IDs expected in retrieval context"
    )
    ground_truth_chunk_ids: list[str] = Field(
        default_factory=list, description="Canonical chunk UUIDs expected in context"
    )
    expected_facts: list[str] = Field(
        default_factory=list,
        description="Key factual propositions that must be present in the answer",
    )
    reference_answer: str = Field(
        ..., min_length=1, description="Authoritative reference answer grounded in corpus"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional query metadata or routing notes"
    )

    @field_validator("question", "reference_answer")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("String field cannot be empty or blank")
        return trimmed


class BenchmarkDataset(BaseModel):
    """Collection of benchmark evaluation questions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(default="1.0.0", description="Dataset schema version")
    total_questions: int = Field(..., ge=1, description="Total questions count")
    questions: list[BenchmarkQuestion] = Field(..., min_length=1, description="List of questions")

    @field_validator("questions")
    @classmethod
    def validate_unique_ids(cls, v: list[BenchmarkQuestion]) -> list[BenchmarkQuestion]:
        ids = [q.id for q in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Benchmark question IDs must be globally unique")
        return v


def load_benchmark_dataset(
    file_path: str | Path = "data/eval/benchmark_questions.json",
) -> BenchmarkDataset:
    """Load and validate benchmark question dataset from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found at {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        data = {"version": "1.0.0", "total_questions": len(data), "questions": data}

    return BenchmarkDataset.model_validate(data)
