"""Unit tests for Evaluation Metrics and Runner (TASK-502).

Tests:
1. Mathematical metric calculations on small hand-verifiable examples.
2. Zero-denominator and empty result sets handling.
3. Out-of-scope / refusal cases evaluation.
4. Hallucinated and ungrounded citation detection.
5. EvaluationRunner with mocked dependencies (offline, zero API costs).
6. Partial failures and error isolation without silent masking.
7. Serialization of BenchmarkRunReport and Category Breakdowns.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.eval.metrics import (
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
)
from src.routing.entity_linker import EntityLinkingResult
from src.routing.intent_classifier import IntentClassificationResult, QueryIntent
from src.routing.router import RoutingResult
from src.synthesis.citation_validator import (
    CitationProvenance,
    GroundingStatus,
    GuardrailResult,
)
from src.synthesis.normalizer import NormalizedChunkItem, NormalizedContext
from src.synthesis.synthesizer import DraftAnswer, TokenUsageMetrics

# ---------------------------------------------------------------------------
# 1. Metric Calculation Unit Tests
# ---------------------------------------------------------------------------


def test_compute_citation_precision_hand_verifiable() -> None:
    # 2 out of 3 cited are in ground truth
    cited = ["c1", "c2", "c3"]
    gt = ["c1", "c2", "c4"]
    assert compute_citation_precision(cited, gt) == pytest.approx(2 / 3)

    # All cited in ground truth
    assert compute_citation_precision(["c1", "c2"], ["c1", "c2", "c3"]) == 1.0

    # None in ground truth
    assert compute_citation_precision(["c5", "c6"], ["c1", "c2"]) == 0.0

    # Empty cited list
    assert compute_citation_precision([], ["c1", "c2"]) == 0.0


def test_compute_citation_recall_hand_verifiable() -> None:
    # 2 out of 3 ground truth chunks were cited
    cited = ["c1", "c2"]
    gt = ["c1", "c2", "c3"]
    assert compute_citation_recall(cited, gt) == pytest.approx(2 / 3)

    # 100% recall
    assert compute_citation_recall(["c1", "c2", "c3", "c4"], ["c1", "c2"]) == 1.0

    # Zero recall
    assert compute_citation_recall(["c5"], ["c1", "c2"]) == 0.0


def test_compute_citation_hallucination_rate() -> None:
    # 1 citation out of 2 is NOT in candidates
    cited = ["c1", "c_hallucinated"]
    candidates = ["c1", "c2", "c3"]
    assert compute_citation_hallucination_rate(cited, candidates) == 0.5

    # 0 hallucinated
    assert compute_citation_hallucination_rate(["c1", "c2"], candidates) == 0.0

    # Empty cited
    assert compute_citation_hallucination_rate([], candidates) == 0.0


def test_compute_retrieval_precision_recall() -> None:
    retrieved = ["c1", "c2", "c3", "c4"]
    gt = ["c1", "c2"]
    prec, rec = compute_retrieval_precision_recall(retrieved, gt)
    assert prec == pytest.approx(2 / 4)
    assert rec == pytest.approx(2 / 2)

    # Zero retrieved
    prec_zero, rec_zero = compute_retrieval_precision_recall([], gt)
    assert prec_zero == 0.0
    assert rec_zero == 0.0


def test_metrics_for_out_of_scope_cases() -> None:
    # Out of scope with valid refusal
    ans = "I could not find relevant information in the enterprise documentation. The requested topic is outside the scope."
    cov, all_facts = compute_fact_coverage(ans, ["Refusal required"], is_out_of_scope=True)
    assert cov == 1.0
    assert all_facts is True

    # Citations for OOS: empty is perfect (1.0)
    assert compute_citation_precision([], [], is_out_of_scope=True) == 1.0
    assert compute_citation_recall([], [], is_out_of_scope=True) == 1.0

    # If OOS emitted a chunk citation, precision and recall drop to 0.0
    assert compute_citation_precision(["c1"], [], is_out_of_scope=True) == 0.0
    assert compute_citation_recall(["c1"], [], is_out_of_scope=True) == 0.0

    # Retrieval for OOS: empty is perfect (1.0)
    ret_prec, ret_rec = compute_retrieval_precision_recall([], [], is_out_of_scope=True)
    assert ret_prec == 1.0
    assert ret_rec == 1.0


def test_compute_fact_coverage_keyword_matching() -> None:
    ans = "The Payment Gateway uses database:payment-vault-pg located in PCI VPC and owned by Checkout Engineering."
    expected = [
        "Payment Gateway uses database:payment-vault-pg",
        "PCI isolated VPC deployment",
        "Owned by Checkout Engineering",
    ]
    cov, all_facts = compute_fact_coverage(ans, expected)
    assert cov == 1.0
    assert all_facts is True

    # Missing one fact
    expected_with_missing = expected + ["Operates under GDPR EU data residency"]
    cov_partial, all_facts_partial = compute_fact_coverage(ans, expected_with_missing)
    assert cov_partial == 0.75
    assert all_facts_partial is False


def test_aggregate_case_metrics_calculation() -> None:
    case1 = calculate_case_metrics(
        retrieved_chunk_ids=["c1", "c2"],
        cited_chunk_ids=["c1"],
        ground_truth_chunk_ids=["c1", "c2"],
        expected_facts=["Fact 1"],
        answer_text="Fact 1 is present.",
        execution_time_ms=100.0,
        estimated_cost_usd=0.002,
        total_tokens=200,
    )
    case2 = calculate_case_metrics(
        retrieved_chunk_ids=["c1"],
        cited_chunk_ids=["c1"],
        ground_truth_chunk_ids=["c1"],
        expected_facts=["Fact 2"],
        answer_text="Fact 2 is present.",
        execution_time_ms=200.0,
        estimated_cost_usd=0.004,
        total_tokens=400,
    )

    agg = aggregate_case_metrics([case1, case2], failed_count=0)
    assert agg.total_cases == 2
    assert agg.successful_cases == 2
    assert agg.failed_cases == 0
    assert agg.accuracy == 1.0
    assert agg.mean_latency_ms == 150.0
    assert agg.total_cost_usd == pytest.approx(0.006)
    assert agg.cost_per_1k_queries_usd == pytest.approx(3.0)


def test_aggregate_case_metrics_empty() -> None:
    agg = aggregate_case_metrics([], failed_count=2)
    assert agg.total_cases == 2
    assert agg.successful_cases == 0
    assert agg.failed_cases == 2
    assert agg.accuracy == 0.0
    assert agg.latency_p50_ms == 0.0


# ---------------------------------------------------------------------------
# 2. EvaluationRunner Integration with Mocked Components
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_benchmark_question() -> BenchmarkQuestion:
    return BenchmarkQuestion(
        id="EVAL-001",
        category=BenchmarkCategory.SINGLE_HOP_DIRECT_FACT,
        difficulty=BenchmarkDifficulty.EASY,
        question="Which database is used by Notification Service?",
        ground_truth_entities=["service:notification-service", "database:notification-pg"],
        ground_truth_chunk_ids=["chunk-1"],
        expected_facts=["Notification Service connects to database:notification-pg"],
        reference_answer="Notification Service uses PostgreSQL database:notification-pg.",
    )


@pytest.mark.asyncio
async def test_evaluation_runner_evaluate_question_success(
    sample_benchmark_question: BenchmarkQuestion,
) -> None:
    mock_router = MagicMock()
    mock_router.route_and_retrieve = AsyncMock(
        return_value=RoutingResult(
            query="Which database is used by Notification Service?",
            mode=QueryIntent.HYBRID,
            classification=IntentClassificationResult(
                query="Which database is used by Notification Service?",
                mode=QueryIntent.HYBRID,
                confidence=0.95,
                reasoning="Hybrid reasoning",
            ),
            entity_linking=EntityLinkingResult(
                query="Which database is used by Notification Service?",
                links=[],
            ),
            graph_results=[],
            vector_chunks=[],
        )
    )

    mock_normalizer = MagicMock()
    mock_normalizer.normalize.return_value = NormalizedContext(
        formatted_context="Notification Service connects to database:notification-pg.",
        chunks=[
            NormalizedChunkItem(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="Notification Service connects to database:notification-pg.",
            )
        ],
        graph_triples=[],
        graph_entities=[],
        total_tokens=50,
        max_budget_tokens=4000,
        candidate_chunk_ids=["chunk-1"],
        all_entity_ids=[],
        is_truncated=False,
        truncated_chunks_count=0,
        truncated_triples_count=0,
    )

    mock_synthesizer = MagicMock()
    mock_synthesizer.synthesize.return_value = DraftAnswer(
        question="Which database is used by Notification Service?",
        raw_answer="Notification Service uses database:notification-pg [chunk-1].",
        cited_chunk_ids=["chunk-1"],
        candidate_chunk_ids=["chunk-1"],
        token_usage=TokenUsageMetrics(
            prompt_tokens=100, completion_tokens=20, total_tokens=120, estimated_cost_usd=0.001
        ),
        execution_time_ms=50.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    mock_validator = MagicMock()
    mock_validator.validate.return_value = GuardrailResult(
        question="Which database is used by Notification Service?",
        original_answer="Notification Service uses database:notification-pg [chunk-1].",
        sanitized_answer="Notification Service uses database:notification-pg [chunk-1].",
        status=GroundingStatus.VALID,
        is_grounded=True,
        valid_citations=[
            CitationProvenance(
                chunk_id="chunk-1",
                document_id="doc-1",
                excerpt="Notification Service connects to database:notification-pg.",
            )
        ],
        valid_chunk_ids=["chunk-1"],
        invalid_chunk_ids=[],
        candidate_chunk_ids=["chunk-1"],
        warnings=[],
    )

    runner = EvaluationRunner(
        router=mock_router,
        normalizer=mock_normalizer,
        synthesizer=mock_synthesizer,
        validator=mock_validator,
    )

    result = await runner.evaluate_question(
        sample_benchmark_question, system_mode=EvaluationSystemMode.HYBRID
    )

    assert isinstance(result, EvaluationCaseResult)
    assert result.id == "EVAL-001"
    assert result.status == "SUCCESS"
    assert result.metrics is not None
    assert result.metrics.citation_precision == 1.0
    assert result.metrics.citation_recall == 1.0
    assert result.metrics.all_facts_present is True


@pytest.mark.asyncio
async def test_evaluation_runner_handles_partial_error(
    sample_benchmark_question: BenchmarkQuestion,
) -> None:
    mock_router = MagicMock()
    mock_router.route_and_retrieve = AsyncMock(side_effect=RuntimeError("Database disconnected"))

    mock_normalizer = MagicMock()
    mock_synthesizer = MagicMock()
    mock_validator = MagicMock()

    runner = EvaluationRunner(
        router=mock_router,
        normalizer=mock_normalizer,
        synthesizer=mock_synthesizer,
        validator=mock_validator,
    )

    result = await runner.evaluate_question(sample_benchmark_question)
    assert result.status == "ERROR"
    assert "Database disconnected" in (result.error_message or "")
    assert result.metrics is None


@pytest.mark.asyncio
async def test_evaluation_runner_run_benchmark(
    sample_benchmark_question: BenchmarkQuestion,
) -> None:
    mock_router = MagicMock()
    mock_router.route_and_retrieve = AsyncMock(
        return_value=RoutingResult(
            query="Which database is used by Notification Service?",
            mode=QueryIntent.VECTOR_ONLY,
            classification=IntentClassificationResult(
                query="Which database is used by Notification Service?",
                mode=QueryIntent.VECTOR_ONLY,
                confidence=0.99,
                reasoning="Vector reasoning",
            ),
            entity_linking=EntityLinkingResult(
                query="Which database is used by Notification Service?",
                links=[],
            ),
            graph_results=[],
            vector_chunks=[],
        )
    )

    mock_normalizer = MagicMock()
    mock_normalizer.normalize.return_value = NormalizedContext(
        formatted_context="Notification Service connects to database:notification-pg.",
        chunks=[
            NormalizedChunkItem(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="Notification Service connects to database:notification-pg.",
            )
        ],
        graph_triples=[],
        graph_entities=[],
        total_tokens=50,
        max_budget_tokens=4000,
        candidate_chunk_ids=["chunk-1"],
        all_entity_ids=[],
        is_truncated=False,
        truncated_chunks_count=0,
        truncated_triples_count=0,
    )

    mock_synthesizer = MagicMock()
    mock_synthesizer.synthesize.return_value = DraftAnswer(
        question="Which database is used by Notification Service?",
        raw_answer="Notification Service uses database:notification-pg [chunk-1].",
        cited_chunk_ids=["chunk-1"],
        candidate_chunk_ids=["chunk-1"],
        token_usage=TokenUsageMetrics(
            prompt_tokens=100, completion_tokens=20, total_tokens=120, estimated_cost_usd=0.001
        ),
        execution_time_ms=50.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    mock_validator = MagicMock()
    mock_validator.validate.return_value = GuardrailResult(
        question="Which database is used by Notification Service?",
        original_answer="Notification Service uses database:notification-pg [chunk-1].",
        sanitized_answer="Notification Service uses database:notification-pg [chunk-1].",
        status=GroundingStatus.VALID,
        is_grounded=True,
        valid_citations=[
            CitationProvenance(
                chunk_id="chunk-1",
                document_id="doc-1",
                excerpt="Notification Service connects to database:notification-pg.",
            )
        ],
        valid_chunk_ids=["chunk-1"],
        invalid_chunk_ids=[],
        candidate_chunk_ids=["chunk-1"],
        warnings=[],
    )

    runner = EvaluationRunner(
        router=mock_router,
        normalizer=mock_normalizer,
        synthesizer=mock_synthesizer,
        validator=mock_validator,
    )

    dataset = BenchmarkDataset(
        version="1.0.0",
        total_questions=1,
        questions=[sample_benchmark_question],
    )

    report = await runner.run_benchmark(
        dataset=dataset, system_mode=EvaluationSystemMode.VECTOR_ONLY
    )

    assert isinstance(report, BenchmarkRunReport)
    assert report.total_questions == 1
    assert report.system_mode == EvaluationSystemMode.VECTOR_ONLY
    assert report.summary.accuracy == 1.0
    assert len(report.case_results) == 1
    assert BenchmarkCategory.SINGLE_HOP_DIRECT_FACT.value in report.category_breakdown
