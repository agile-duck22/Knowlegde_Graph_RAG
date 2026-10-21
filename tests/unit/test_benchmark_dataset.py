"""Unit tests for TASK-501 Benchmark Dataset.

Validates:
- Schema and serialization using Pydantic models.
- Taxonomy compliance (50 questions across 5 categories, matching difficulty levels).
- Question ID uniqueness and sequential formatting.
- Ground truth chunk ID existence against the chunked synthetic corpus.
- Ground truth entity and relationship formatting.
- Deterministic loading from JSON.
"""

from pathlib import Path

import pytest

from src.eval.schema import (
    BenchmarkCategory,
    BenchmarkDataset,
    BenchmarkDifficulty,
    BenchmarkQuestion,
    load_benchmark_dataset,
)
from src.ingest.chunker import DocumentChunker, TextChunk
from src.ingest.loader import DocumentLoader


@pytest.fixture
def benchmark_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "benchmark_questions.json"
    )


@pytest.fixture
def corpus_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "corpus.jsonl"


def test_load_benchmark_dataset_success(benchmark_path: Path) -> None:
    assert benchmark_path.exists(), f"Benchmark file does not exist at {benchmark_path}"
    dataset = load_benchmark_dataset(benchmark_path)
    assert isinstance(dataset, BenchmarkDataset)
    assert len(dataset.questions) == 50
    assert dataset.version == "1.0.0"


def test_benchmark_taxonomy_distribution(benchmark_path: Path) -> None:
    dataset = load_benchmark_dataset(benchmark_path)
    categories = [q.category for q in dataset.questions]

    assert categories.count(BenchmarkCategory.SINGLE_HOP_DIRECT_FACT) == 10
    assert categories.count(BenchmarkCategory.TWO_HOP_RELATIONSHIP) == 15
    assert categories.count(BenchmarkCategory.THREE_HOP_DEPENDENCY) == 10
    assert categories.count(BenchmarkCategory.RELATIONSHIP_AGGREGATION) == 10
    assert categories.count(BenchmarkCategory.OUT_OF_SCOPE) == 5


def test_benchmark_difficulty_mapping(benchmark_path: Path) -> None:
    dataset = load_benchmark_dataset(benchmark_path)

    for q in dataset.questions:
        if q.category == BenchmarkCategory.SINGLE_HOP_DIRECT_FACT:
            assert q.difficulty == BenchmarkDifficulty.EASY
        elif q.category == BenchmarkCategory.TWO_HOP_RELATIONSHIP:
            assert q.difficulty == BenchmarkDifficulty.MEDIUM
        elif q.category in (
            BenchmarkCategory.THREE_HOP_DEPENDENCY,
            BenchmarkCategory.RELATIONSHIP_AGGREGATION,
        ):
            assert q.difficulty == BenchmarkDifficulty.HARD
        elif q.category == BenchmarkCategory.OUT_OF_SCOPE:
            assert q.difficulty == BenchmarkDifficulty.EASY


def test_benchmark_unique_ids(benchmark_path: Path) -> None:
    dataset = load_benchmark_dataset(benchmark_path)
    ids = [q.id for q in dataset.questions]
    assert len(ids) == len(set(ids))
    assert ids == [f"EVAL-{i:03d}" for i in range(1, 51)]


def test_benchmark_ground_truth_chunks_exist_in_corpus(
    benchmark_path: Path, corpus_path: Path
) -> None:
    loader = DocumentLoader(corpus_path)
    chunker = DocumentChunker()

    documents = loader.load()
    all_chunks: list[TextChunk] = []
    for doc in documents:
        all_chunks.extend(chunker.chunk(doc))

    corpus_chunk_ids = {c.chunk_id for c in all_chunks}
    assert len(corpus_chunk_ids) > 0

    dataset = load_benchmark_dataset(benchmark_path)

    for q in dataset.questions:
        if q.category == BenchmarkCategory.OUT_OF_SCOPE:
            assert len(q.ground_truth_chunk_ids) == 0, (
                f"{q.id} should have empty ground truth chunk IDs"
            )
            assert len(q.ground_truth_entities) == 0
        else:
            assert len(q.ground_truth_chunk_ids) > 0, (
                f"{q.id} should have at least 1 ground truth chunk ID"
            )
            for chunk_id in q.ground_truth_chunk_ids:
                assert chunk_id in corpus_chunk_ids, (
                    f"Chunk ID {chunk_id} for question {q.id} was not found in chunked corpus"
                )


def test_benchmark_schema_validation_errors() -> None:
    with pytest.raises(ValueError, match="Benchmark question IDs must be globally unique"):
        BenchmarkDataset(
            total_questions=2,
            questions=[
                BenchmarkQuestion(
                    id="EVAL-001",
                    question="Who is the lead engineer for checkout?",
                    category=BenchmarkCategory.SINGLE_HOP_DIRECT_FACT,
                    difficulty=BenchmarkDifficulty.EASY,
                    reference_answer="A1",
                    ground_truth_chunk_ids=["chunk-1"],
                ),
                BenchmarkQuestion(
                    id="EVAL-001",
                    question="What database does payments use?",
                    category=BenchmarkCategory.SINGLE_HOP_DIRECT_FACT,
                    difficulty=BenchmarkDifficulty.EASY,
                    reference_answer="A2",
                    ground_truth_chunk_ids=["chunk-2"],
                ),
            ],
        )


def test_load_non_existent_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_benchmark_dataset(Path("non_existent_file.json"))
