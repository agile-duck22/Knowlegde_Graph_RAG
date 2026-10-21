"""FastAPI Route Handlers (TASK-404).

Implements:
- GET /health: Liveness probe
- GET /ready: Readiness probe checking Neo4j and PostgreSQL
- POST /api/v1/query: End-to-end RAG question answering pipeline
"""

from __future__ import annotations

import logging
import time
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.dependencies import ServiceContainer, get_service_container, utc_now_iso
from src.api.schemas import (
    CitationResponseItem,
    DatabaseStatus,
    ErrorResponse,
    GraphPathResponseItem,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    TokenUsageResponse,
)
from src.routing.intent_classifier import QueryIntent
from src.routing.router import RoutingRequest
from src.synthesis.normalizer import NormalizationInput
from src.synthesis.synthesizer import SynthesisRequest

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Health & Readiness Probes
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness Probe",
    description="Basic liveness check verifying HTTP service is running.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Return process health status."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=utc_now_iso(),
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    responses={
        200: {"model": ReadyResponse, "description": "All databases connected"},
        503: {"model": ReadyResponse, "description": "One or more databases disconnected"},
    },
    summary="Readiness Probe",
    description="Readiness check verifying Neo4j and PostgreSQL database connectivity.",
    tags=["System"],
)
async def readiness_check(
    container: Annotated[ServiceContainer, Depends(get_service_container)],
) -> ReadyResponse:
    """Verify Neo4j and PostgreSQL connections."""
    is_ready, db_status, error_msg = await container.check_databases_readiness()
    databases = DatabaseStatus(
        neo4j="connected" if db_status.get("neo4j") == "connected" else "disconnected",
        postgres="connected" if db_status.get("postgres") == "connected" else "disconnected",
    )

    if not is_ready:
        resp = ReadyResponse(
            status="unavailable",
            databases=databases,
            timestamp=utc_now_iso(),
            error=error_msg or "One or more databases unavailable",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=resp.model_dump(),
        )

    return ReadyResponse(
        status="ready",
        databases=databases,
        timestamp=utc_now_iso(),
        error=None,
    )


# ---------------------------------------------------------------------------
# Query Pipeline Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    responses={
        200: {"model": QueryResponse, "description": "Successful grounded synthesis"},
        400: {"model": ErrorResponse, "description": "Validation error or invalid input"},
        422: {"model": ErrorResponse, "description": "Unprocessable entity schema error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Upstream service or database failure"},
    },
    summary="Knowledge Graph RAG Query",
    description="Executes query routing, retrieval, context normalization, Claude synthesis, and citation verification.",
    tags=["RAG Pipeline"],
)
async def query_endpoint(
    payload: QueryRequest,
    request: Request,
    container: Annotated[ServiceContainer, Depends(get_service_container)],
) -> QueryResponse:
    """Execute end-to-end question answering pipeline."""
    start_time = time.perf_counter()
    request_id = getattr(request.state, "request_id", None) or f"req-{uuid4()}"

    # 1. Validate Input Constraints
    clean_question = payload.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "VALIDATION_ERROR",
                "message": "Question cannot be empty or blank.",
                "request_id": request_id,
            },
        )

    if len(clean_question) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "VALIDATION_ERROR",
                "message": "Question cannot exceed 1000 characters.",
                "request_id": request_id,
            },
        )

    try:
        # 2. Query Routing & Retrieval
        router_engine = container.get_router()
        routing_req = RoutingRequest(
            query=clean_question,
            retrieval_override=payload.retrieval_override,
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
        )

        routing_result = await router_engine.route_and_retrieve(routing_req)

        # 3. Context Normalization & Token Budgeting
        hydrated_chunks = []
        if routing_result.all_provenance_chunk_ids and router_engine.vector_retriever is not None:
            try:
                hydrated_chunks = await router_engine.vector_retriever.get_chunks_by_ids(
                    routing_result.all_provenance_chunk_ids
                )
            except Exception as hyd_err:
                logger.warning("Failed to hydrate graph provenance chunks: %s", hyd_err)

        normalizer = container.get_normalizer()
        norm_input = NormalizationInput(
            graph_results=routing_result.graph_results,
            vector_chunks=routing_result.vector_chunks,
            hydrated_graph_chunks=hydrated_chunks,
            query=clean_question,
            max_context_tokens=container.settings.max_context_tokens,
        )
        normalized_context = normalizer.normalize(norm_input)

        # 4. Grounded Answer Synthesis
        synthesizer = container.get_synthesizer()
        synthesis_req = SynthesisRequest(
            question=clean_question,
            context=normalized_context,
        )
        draft_answer = synthesizer.synthesize(synthesis_req)

        # 5. Citation & Grounding Guardrail
        validator = container.get_validator()
        guardrail_result = validator.validate(
            draft=draft_answer,
            context=normalized_context,
        )

        # 6. Format Response Payload
        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Citations
        citations_response = [
            CitationResponseItem(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                excerpt=c.excerpt,
            )
            for c in guardrail_result.valid_citations
        ]

        # Graph Paths / Relations (if include_provenance is True)
        graph_paths_response: list[GraphPathResponseItem] = []
        if payload.include_provenance:
            for triple in normalized_context.graph_triples:
                graph_paths_response.append(
                    GraphPathResponseItem(
                        source=triple.source_name or triple.source_id,
                        relation=triple.relationship_type,
                        target=triple.target_name or triple.target_id,
                    )
                )

        # Warnings
        warnings_list: list[str] = [w.message for w in guardrail_result.warnings]
        if routing_result.mode == QueryIntent.OUT_OF_SCOPE:
            warnings_list.append("Query was classified as OUT_OF_SCOPE.")

        return QueryResponse(
            request_id=request_id,
            question=clean_question,
            answer=guardrail_result.sanitized_answer,
            retrieval_mode=routing_result.mode.value,
            confidence=round(routing_result.classification.confidence, 2),
            citations=citations_response,
            graph_paths=graph_paths_response,
            execution_time_ms=execution_time_ms,
            token_usage=TokenUsageResponse(
                prompt_tokens=draft_answer.token_usage.prompt_tokens,
                completion_tokens=draft_answer.token_usage.completion_tokens,
                total_tokens=draft_answer.token_usage.total_tokens,
                estimated_cost_usd=draft_answer.token_usage.estimated_cost_usd,
            ),
            warnings=warnings_list,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled pipeline error for request %s: %s", request_id, exc, exc_info=True)
        # Redact raw exception details and secrets
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An error occurred while processing the request. Please try again later.",
                "request_id": request_id,
            },
        ) from exc
