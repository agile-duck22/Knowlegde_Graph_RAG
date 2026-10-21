"""Unit tests for FastAPI Application Endpoints (TASK-404).

Verifies:
- GET /health returns 200 OK with liveness metadata.
- GET /ready returns 200 OK when databases are connected, 503 Service Unavailable when disconnected.
- POST /api/v1/query returns 200 OK with sanitized grounded answer and validated citations.
- POST /api/v1/query handles OUT_OF_SCOPE and empty-context refusals gracefully.
- Input validation rejects questions exceeding length constraints or invalid payloads (400 / 422).
- Internal exceptions return 500 without leaking secrets or stack traces.
- Zero live databases or paid Anthropic API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import ServiceContainer, set_service_container
from src.api.main import create_app
from src.config import Settings
from src.retrieval.graph_retriever import GraphQueryResult
from src.retrieval.vector_retriever import RetrievedChunk
from src.routing.entity_linker import EntityLinkingResult
from src.routing.intent_classifier import IntentClassificationResult, QueryIntent
from src.routing.router import RouterEngine, RoutingResult
from src.synthesis.synthesizer import DraftAnswer, GroundedAnswerSynthesizer, TokenUsageMetrics


@pytest.fixture
def mock_container() -> ServiceContainer:
    settings = Settings(
        app_name="kgr-test",
        environment="test",
        max_context_tokens=1000,
    )
    container = ServiceContainer(settings=settings)
    return container


@pytest.fixture
def client(mock_container: ServiceContainer) -> TestClient:
    app = create_app(settings=mock_container.settings)
    set_service_container(mock_container)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health & Readiness Tests
# ---------------------------------------------------------------------------


def test_get_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data


def test_get_ready_success(client: TestClient, mock_container: ServiceContainer) -> None:
    # Mock successful database checks
    mock_container.check_databases_readiness = AsyncMock(  # type: ignore[method-assign]
        return_value=(True, {"neo4j": "connected", "postgres": "connected"}, None)
    )

    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["databases"]["neo4j"] == "connected"
    assert data["databases"]["postgres"] == "connected"
    assert data["error"] is None


def test_get_ready_failure_service_unavailable(
    client: TestClient, mock_container: ServiceContainer
) -> None:
    # Mock Neo4j failure
    mock_container.check_databases_readiness = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            False,
            {"neo4j": "disconnected", "postgres": "connected"},
            "Failed to connect to Neo4j database",
        )
    )

    response = client.get("/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unavailable"
    assert data["databases"]["neo4j"] == "disconnected"
    assert data["databases"]["postgres"] == "connected"
    assert "Failed to connect to Neo4j" in data["error"]


# ---------------------------------------------------------------------------
# Query Pipeline Tests
# ---------------------------------------------------------------------------


def test_post_query_success_hybrid_flow(
    client: TestClient, mock_container: ServiceContainer
) -> None:
    # 1. Mock RouterEngine
    mock_router = MagicMock(spec=RouterEngine)
    routing_result = RoutingResult(
        query="Which services depend on Checkout?",
        mode=QueryIntent.HYBRID,
        classification=IntentClassificationResult(
            query="Which services depend on Checkout?",
            mode=QueryIntent.HYBRID,
            confidence=0.92,
            reasoning="Broad question",
        ),
        entity_linking=EntityLinkingResult(
            query="Which services depend on Checkout?",
            links=[],
        ),
        graph_results=[
            GraphQueryResult(
                template_id="T-01",
                template_name="Direct Neighborhood",
                parameters={"entity_id": "service:checkout-service"},
                records=[
                    {
                        "source": "service:payment-gateway",
                        "relation": "DEPENDS_ON",
                        "target": "service:checkout-service",
                        "source_chunk_ids": ["c2222222-2222-2222-2222-222222222222"],
                    }
                ],
                record_count=1,
                source_chunk_ids=["c2222222-2222-2222-2222-222222222222"],
                entity_ids=["service:checkout-service"],
            )
        ],
        vector_chunks=[
            RetrievedChunk(
                chunk_id="c1111111-1111-1111-1111-111111111111",
                document_id="doc-checkout-arch",
                content="Checkout Service is maintained by Checkout Team.",
                similarity_score=0.95,
            )
        ],
    )

    mock_router.route_and_retrieve = AsyncMock(return_value=routing_result)
    mock_container.router = mock_router

    # 2. Mock Synthesizer
    mock_synthesizer = MagicMock(spec=GroundedAnswerSynthesizer)
    draft = DraftAnswer(
        question="Which services depend on Checkout?",
        raw_answer=(
            "Checkout Service is maintained by Checkout Team [c1111111-1111-1111-1111-111111111111]. "
            "Payment Gateway depends on Checkout [c2222222-2222-2222-2222-222222222222]."
        ),
        cited_chunk_ids=[
            "c1111111-1111-1111-1111-111111111111",
            "c2222222-2222-2222-2222-222222222222",
        ],
        candidate_chunk_ids=[
            "c1111111-1111-1111-1111-111111111111",
            "c2222222-2222-2222-2222-222222222222",
        ],
        token_usage=TokenUsageMetrics(
            prompt_tokens=450,
            completion_tokens=65,
            total_tokens=515,
            estimated_cost_usd=0.0023,
        ),
        execution_time_ms=250.0,
        model_name="claude-3-5-sonnet-20241022",
    )
    mock_synthesizer.synthesize.return_value = draft
    mock_container.synthesizer = mock_synthesizer

    # 3. Execute Request
    payload = {
        "question": "Which services depend on Checkout?",
        "top_k": 5,
        "include_provenance": True,
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["question"] == "Which services depend on Checkout?"
    assert "Checkout Service is maintained by Checkout Team" in data["answer"]
    assert data["retrieval_mode"] == "HYBRID"
    assert data["confidence"] == 0.92
    assert len(data["citations"]) == 2
    assert data["citations"][0]["chunk_id"] == "c1111111-1111-1111-1111-111111111111"
    assert data["citations"][0]["document_id"] == "doc-checkout-arch"
    assert len(data["graph_paths"]) == 1
    assert data["graph_paths"][0]["source"] == "service:payment-gateway"
    assert data["graph_paths"][0]["relation"] == "DEPENDS_ON"
    assert data["token_usage"]["total_tokens"] == 515
    assert data["token_usage"]["estimated_cost_usd"] == 0.0023
    assert data["warnings"] == []


def test_post_query_out_of_scope_refusal(
    client: TestClient, mock_container: ServiceContainer
) -> None:
    mock_router = MagicMock(spec=RouterEngine)
    routing_result = RoutingResult(
        query="What is the weather in Paris?",
        mode=QueryIntent.OUT_OF_SCOPE,
        classification=IntentClassificationResult(
            query="What is the weather in Paris?",
            mode=QueryIntent.OUT_OF_SCOPE,
            confidence=0.99,
            reasoning="Non-enterprise topic",
        ),
        entity_linking=EntityLinkingResult(
            query="What is the weather in Paris?",
            links=[],
        ),
        graph_results=[],
        vector_chunks=[],
    )
    mock_router.route_and_retrieve = AsyncMock(return_value=routing_result)
    mock_container.router = mock_router

    payload = {"question": "What is the weather in Paris?"}
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["retrieval_mode"] == "OUT_OF_SCOPE"
    assert "could not find" in data["answer"].lower()
    assert data["citations"] == []
    assert any("OUT_OF_SCOPE" in w for w in data["warnings"])


def test_post_query_hallucinated_citation_scrubbing(
    client: TestClient, mock_container: ServiceContainer
) -> None:
    mock_router = MagicMock(spec=RouterEngine)
    routing_result = RoutingResult(
        query="Who owns Checkout?",
        mode=QueryIntent.VECTOR_ONLY,
        classification=IntentClassificationResult(
            query="Who owns Checkout?",
            mode=QueryIntent.VECTOR_ONLY,
            confidence=0.88,
            reasoning="Vector search",
        ),
        entity_linking=EntityLinkingResult(
            query="Who owns Checkout?",
            links=[],
        ),
        graph_results=[],
        vector_chunks=[
            RetrievedChunk(
                chunk_id="c1111111-1111-1111-1111-111111111111",
                document_id="doc-checkout-arch",
                content="Checkout Service is maintained by Checkout Team.",
                similarity_score=0.95,
            )
        ],
    )
    mock_router.route_and_retrieve = AsyncMock(return_value=routing_result)
    mock_container.router = mock_router

    mock_synthesizer = MagicMock(spec=GroundedAnswerSynthesizer)
    draft = DraftAnswer(
        question="Who owns Checkout?",
        raw_answer="Checkout is owned by Checkout Team [c1111111-1111-1111-1111-111111111111] and runs on port 80 [fake-chunk-999].",
        cited_chunk_ids=["c1111111-1111-1111-1111-111111111111", "fake-chunk-999"],
        candidate_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        token_usage=TokenUsageMetrics(
            prompt_tokens=100, completion_tokens=25, total_tokens=125, estimated_cost_usd=0.001
        ),
        execution_time_ms=80.0,
        model_name="claude-3-5-sonnet-20241022",
    )
    mock_synthesizer.synthesize.return_value = draft
    mock_container.synthesizer = mock_synthesizer

    payload = {"question": "Who owns Checkout?"}
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Hallucinated citation [fake-chunk-999] was scrubbed
    assert "[c1111111-1111-1111-1111-111111111111]" in data["answer"]
    assert "[fake-chunk-999]" not in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "c1111111-1111-1111-1111-111111111111"
    assert any("fake-chunk-999" in w for w in data["warnings"])


# ---------------------------------------------------------------------------
# Validation & Error Handling Tests
# ---------------------------------------------------------------------------


def test_post_query_empty_question_validation_error(client: TestClient) -> None:
    response = client.post("/api/v1/query", json={"question": "   "})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "UNPROCESSABLE_ENTITY"
    assert "validation failed" in data["message"].lower()


def test_post_query_oversized_question_validation_error(client: TestClient) -> None:
    long_question = "Which service? " * 100  # > 1000 characters
    response = client.post("/api/v1/query", json={"question": long_question})
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "UNPROCESSABLE_ENTITY"


def test_post_query_invalid_top_k_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/v1/query", json={"question": "Valid question", "top_k": 50}
    )  # max is 20
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "UNPROCESSABLE_ENTITY"


def test_post_query_internal_error_sanitization(
    client: TestClient, mock_container: ServiceContainer
) -> None:
    mock_router = MagicMock(spec=RouterEngine)
    mock_router.route_and_retrieve = AsyncMock(
        side_effect=RuntimeError("Database crashed: secret_key=super_secret_password")
    )
    mock_container.router = mock_router

    response = client.post("/api/v1/query", json={"question": "Valid question"})
    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "INTERNAL_SERVER_ERROR"
    # Secrets/internal details must not leak in response message
    assert "super_secret_password" not in data["message"]
    assert "secret_key" not in data["message"]


# ---------------------------------------------------------------------------
# Lifecycle & Security Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_container_lifecycle_cleanup() -> None:
    """Verify ServiceContainer closes postgres and neo4j clients on shutdown."""
    mock_pg = MagicMock()
    mock_pg.close = AsyncMock()
    mock_neo4j = MagicMock()
    mock_neo4j.close = MagicMock()

    container = ServiceContainer(
        settings=Settings(app_name="kgr-test", environment="test"),
        postgres_client=mock_pg,
        neo4j_client=mock_neo4j,
    )

    await container.close()

    mock_pg.close.assert_awaited_once()
    mock_neo4j.close.assert_called_once()


def test_cors_preflight_headers(client: TestClient) -> None:
    """Verify CORS middleware responds with correct headers."""
    response = client.options(
        "/api/v1/query",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
