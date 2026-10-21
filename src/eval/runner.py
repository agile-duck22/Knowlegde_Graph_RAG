"""Automated Evaluation Runner for KG+Vector RAG vs Vector-Only Baseline (TASK-502).

Executes benchmark evaluations comparing Hybrid KG+Vector RAG against Baseline Vector-Only RAG:
1. Loads validated benchmark question dataset from disk (TASK-501).
2. Executes each query against the target pipeline mode (HYBRID or VECTOR_ONLY).
3. Computes quantitative retrieval, citation, fact coverage, latency, and cost metrics.
4. Preserves comprehensive case-level results and generates category-level aggregations.
5. Saves machine-readable JSON evaluation run reports adhering to docs/EVALUATION.md.

Design & Architectural Invariants:
- Deterministic execution and offline testability via dependency injection.
- Zero paid API calls in default mock/unit test mode.
- Explicit failure tracking without silent success masking.
- Strict isolation of reference answers and expected facts from retrieval inputs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies import (
    ServiceContainer,
    create_service_container,
)
from src.config import Settings, get_settings
from src.eval.metrics import (
    AggregatedMetrics,
    CaseMetrics,
    aggregate_case_metrics,
    calculate_case_metrics,
)
from src.eval.schema import (
    BenchmarkCategory,
    BenchmarkDataset,
    BenchmarkQuestion,
    load_benchmark_dataset,
)
from src.routing.intent_classifier import QueryIntent
from src.routing.router import RouterEngine, RoutingRequest
from src.synthesis.citation_validator import CitationValidator, GuardrailResult
from src.synthesis.normalizer import ContextNormalizer, NormalizationInput
from src.synthesis.synthesizer import GroundedAnswerSynthesizer, SynthesisRequest

logger = logging.getLogger(__name__)


class EvaluationSystemMode(StrEnum):
    """System retrieval mode under benchmark evaluation."""

    HYBRID = "HYBRID"
    VECTOR_ONLY = "VECTOR_ONLY"


class EvaluationCaseResult(BaseModel):
    """Execution result and metrics for a single benchmark question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., description="Benchmark question ID (e.g. EVAL-001)")
    category: BenchmarkCategory = Field(..., description="Taxonomy category")
    question: str = Field(..., description="Question inquiry")
    system_mode: EvaluationSystemMode = Field(..., description="Evaluated system mode")

    # Inputs & Ground Truth (Preserved from dataset)
    ground_truth_entities: list[str] = Field(default_factory=list)
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    reference_answer: str = Field(...)

    # Execution Outputs
    generated_answer: str = Field(default="", description="Generated answer text")
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list, description="Retrieved candidate chunk UUIDs"
    )
    cited_chunk_ids: list[str] = Field(
        default_factory=list, description="Valid cited chunk UUIDs from answer"
    )
    routing_mode: str = Field(default="", description="Observed query routing classification")

    # Metrics & Performance
    metrics: CaseMetrics | None = Field(
        default=None, description="Computed quantitative case metrics"
    )
    status: str = Field(default="SUCCESS", description="'SUCCESS' or 'ERROR'")
    error_message: str | None = Field(default=None, description="Error detail if failed")


class BenchmarkRunReport(BaseModel):
    """Complete machine-readable evaluation report for a benchmark run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(default="1.0.0", description="Report format version")
    timestamp: str = Field(..., description="ISO 8601 execution timestamp")
    system_mode: EvaluationSystemMode = Field(..., description="Evaluated system mode")
    dataset_version: str = Field(default="1.0.0", description="Benchmark dataset version")
    total_questions: int = Field(ge=0, description="Total questions attempted")

    # Overall Summary Metrics
    summary: AggregatedMetrics = Field(..., description="Overall aggregated benchmark metrics")

    # Category Breakdown Metrics
    category_breakdown: dict[str, AggregatedMetrics] = Field(
        ..., description="Aggregated metrics grouped by taxonomy category"
    )

    # Detailed Individual Case Results
    case_results: list[EvaluationCaseResult] = Field(
        ..., description="Individual execution results and scores"
    )


class EvaluationRunner:
    """Automated benchmark executor evaluating Vector vs Hybrid RAG pipelines."""

    def __init__(
        self,
        router: RouterEngine,
        normalizer: ContextNormalizer,
        synthesizer: GroundedAnswerSynthesizer,
        validator: CitationValidator,
        settings: Settings | None = None,
    ) -> None:
        self.router = router
        self.normalizer = normalizer
        self.synthesizer = synthesizer
        self.validator = validator
        self.settings = settings or get_settings()

    @classmethod
    def from_container(cls, container: ServiceContainer) -> EvaluationRunner:
        """Instantiate evaluation runner from standard FastAPI ServiceContainer."""
        return cls(
            router=container.get_router(),
            normalizer=container.get_normalizer(),
            synthesizer=container.get_synthesizer(),
            validator=container.get_validator(),
            settings=container.settings,
        )

    async def evaluate_question(
        self,
        question: BenchmarkQuestion,
        system_mode: EvaluationSystemMode = EvaluationSystemMode.HYBRID,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> EvaluationCaseResult:
        """Execute and score a single benchmark question."""
        start_time = time.perf_counter()
        is_oos = question.category == BenchmarkCategory.OUT_OF_SCOPE

        try:
            # 1. Determine retrieval override based on evaluation mode
            retrieval_override = (
                QueryIntent.VECTOR_ONLY if system_mode == EvaluationSystemMode.VECTOR_ONLY else None
            )

            # 2. Execute Routing and Retrieval (NEVER leak ground-truth answers/facts into query)
            routing_req = RoutingRequest(
                query=question.question,
                retrieval_override=retrieval_override,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            routing_result = await self.router.route_and_retrieve(routing_req)

            # 3. Context Normalization & Provenance Hydration
            hydrated_chunks = []
            if routing_result.all_provenance_chunk_ids and self.router.vector_retriever is not None:
                try:
                    hydrated_chunks = await self.router.vector_retriever.get_chunks_by_ids(
                        routing_result.all_provenance_chunk_ids
                    )
                except Exception as hyd_err:
                    logger.warning("Failed to hydrate graph provenance chunks: %s", hyd_err)

            norm_input = NormalizationInput(
                graph_results=routing_result.graph_results,
                vector_chunks=routing_result.vector_chunks,
                hydrated_graph_chunks=hydrated_chunks,
                query=question.question,
                max_context_tokens=self.settings.max_context_tokens,
            )
            normalized_context = self.normalizer.normalize(norm_input)

            # 4. Grounded Answer Synthesis
            synthesis_req = SynthesisRequest(
                question=question.question,
                context=normalized_context,
            )
            draft_answer = self.synthesizer.synthesize(synthesis_req)

            # 5. Citation Guardrail Validation
            guardrail_result: GuardrailResult = self.validator.validate(
                draft=draft_answer,
                context=normalized_context,
            )

            execution_time_ms = (time.perf_counter() - start_time) * 1000.0

            # Candidate chunk IDs from normalized context (includes text chunks & graph provenance)
            retrieved_chunk_ids = list(normalized_context.candidate_chunk_ids)
            # Valid citations emitted
            cited_chunk_ids = [c.chunk_id for c in guardrail_result.valid_citations]

            # 6. Calculate Quantitative Metrics
            metrics = calculate_case_metrics(
                retrieved_chunk_ids=retrieved_chunk_ids,
                cited_chunk_ids=cited_chunk_ids,
                ground_truth_chunk_ids=question.ground_truth_chunk_ids,
                expected_facts=question.expected_facts,
                answer_text=guardrail_result.sanitized_answer,
                execution_time_ms=execution_time_ms,
                estimated_cost_usd=draft_answer.token_usage.estimated_cost_usd,
                total_tokens=draft_answer.token_usage.total_tokens,
                is_out_of_scope=is_oos,
            )

            return EvaluationCaseResult(
                id=question.id,
                category=question.category,
                question=question.question,
                system_mode=system_mode,
                ground_truth_entities=question.ground_truth_entities,
                ground_truth_chunk_ids=question.ground_truth_chunk_ids,
                expected_facts=question.expected_facts,
                reference_answer=question.reference_answer,
                generated_answer=guardrail_result.sanitized_answer,
                retrieved_chunk_ids=retrieved_chunk_ids,
                cited_chunk_ids=cited_chunk_ids,
                routing_mode=routing_result.mode.value,
                metrics=metrics,
                status="SUCCESS",
                error_message=None,
            )

        except Exception as exc:
            logger.error("Error evaluating question %s: %s", question.id, exc, exc_info=True)
            return EvaluationCaseResult(
                id=question.id,
                category=question.category,
                question=question.question,
                system_mode=system_mode,
                ground_truth_entities=question.ground_truth_entities,
                ground_truth_chunk_ids=question.ground_truth_chunk_ids,
                expected_facts=question.expected_facts,
                reference_answer=question.reference_answer,
                generated_answer="",
                retrieved_chunk_ids=[],
                cited_chunk_ids=[],
                routing_mode="",
                metrics=None,
                status="ERROR",
                error_message=str(exc),
            )

    async def run_benchmark(
        self,
        dataset: BenchmarkDataset,
        system_mode: EvaluationSystemMode = EvaluationSystemMode.HYBRID,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ) -> BenchmarkRunReport:
        """Run full evaluation suite over all questions in the dataset."""
        import datetime

        case_results: list[EvaluationCaseResult] = []
        for question in dataset.questions:
            result = await self.evaluate_question(
                question=question,
                system_mode=system_mode,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            case_results.append(result)

        # Aggregate overall summary
        successful_metrics = [r.metrics for r in case_results if r.metrics is not None]
        failed_count = sum(1 for r in case_results if r.status != "SUCCESS")
        summary = aggregate_case_metrics(successful_metrics, failed_count=failed_count)

        # Aggregate category breakdowns
        category_breakdown: dict[str, AggregatedMetrics] = {}
        for cat in BenchmarkCategory:
            cat_results = [r for r in case_results if r.category == cat]
            cat_metrics = [r.metrics for r in cat_results if r.metrics is not None]
            cat_failed = sum(1 for r in cat_results if r.status != "SUCCESS")
            category_breakdown[cat.value] = aggregate_case_metrics(
                cat_metrics, failed_count=cat_failed
            )

        return BenchmarkRunReport(
            version="1.0.0",
            timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
            system_mode=system_mode,
            dataset_version=dataset.version,
            total_questions=len(dataset.questions),
            summary=summary,
            category_breakdown=category_breakdown,
            case_results=case_results,
        )


async def main() -> None:
    """CLI entrypoint for executing benchmark evaluation runs."""
    parser = argparse.ArgumentParser(
        description="Automated Evaluation Runner for Knowledge Graph RAG"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval/benchmark_questions.json",
        help="Path to benchmark questions JSON file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["HYBRID", "VECTOR_ONLY"],
        default="HYBRID",
        help="Retrieval system mode to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/run.json",
        help="Path to save output benchmark report JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieval limit",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Optional cosine similarity threshold override for vector retrieval",
    )

    args = parser.parse_args()
    dataset = load_benchmark_dataset(args.dataset)
    container = create_service_container(init_clients=True)

    try:
        runner = EvaluationRunner.from_container(container)
        mode = EvaluationSystemMode(args.mode)

        print(f"Starting evaluation run: {mode.value} on {len(dataset.questions)} questions...")
        report = await runner.run_benchmark(
            dataset=dataset,
            system_mode=mode,
            top_k=args.top_k,
            similarity_threshold=args.similarity_threshold,
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2)

        print(f"Evaluation complete! Results saved to {output_path.resolve()}")
        print(
            f"Accuracy: {report.summary.accuracy * 100:.1f}% | Latency p50: {report.summary.latency_p50_ms:.1f}ms | Cost/1k: ${report.summary.cost_per_1k_queries_usd:.4f}"
        )
    finally:
        await container.close()


if __name__ == "__main__":
    asyncio.run(main())
