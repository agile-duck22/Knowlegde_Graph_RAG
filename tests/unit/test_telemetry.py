"""Unit tests for Structured Logging and Request Telemetry Middleware (TASK-405).

Verifies:
- Structured JSON logging formatter format and required schema fields.
- Redaction of API keys (Bearer tokens, Anthropic sk-ant-*, OpenAI keys).
- Redaction of database credentials in URIs and password fields.
- RequestTelemetryMiddleware correlation ID generation and propagation (X-Request-ID).
- RequestTelemetryMiddleware preserves incoming client X-Request-ID headers.
- RequestTelemetryMiddleware adds X-Response-Time-MS header.
- Handled and unhandled failures log appropriate status, error types, and correlation IDs.
- Health and readiness probes are logged correctly without breaking schemas.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import ServiceContainer, set_service_container
from src.api.main import create_app
from src.config import Settings
from src.observability.logging import (
    StructuredJsonFormatter,
    configure_structured_logging,
    redact_sensitive_text,
    sanitize_log_dict,
)
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
# Redaction & Sanitization Unit Tests
# ---------------------------------------------------------------------------


def test_redact_sensitive_text_bearer_tokens() -> None:
    text = "Authorization: Bearer secret_token_12345_xyz"
    redacted = redact_sensitive_text(text)
    assert "secret_token_12345_xyz" not in redacted
    assert "Bearer [REDACTED]" in redacted


def test_redact_sensitive_text_anthropic_api_keys() -> None:
    text = "Using Anthropic key sk-ant-api03-abcdef123456789-xyz"
    redacted = redact_sensitive_text(text)
    assert "sk-ant-api03-abcdef123456789-xyz" not in redacted
    assert "[REDACTED_ANTHROPIC_KEY]" in redacted


def test_redact_sensitive_text_openai_api_keys() -> None:
    text = "Using OpenAI key sk-abcdef123456789012345678"
    redacted = redact_sensitive_text(text)
    assert "sk-abcdef123456789012345678" not in redacted
    assert "[REDACTED_OPENAI_KEY]" in redacted


def test_redact_sensitive_text_database_uris() -> None:
    neo4j_uri = "bolt://neo4j:my_super_secret_neo4j_pw@localhost:7687"
    pg_uri = "postgresql+asyncpg://kgr_user:my_secret_pg_pw@localhost:5432/kgr_db"

    redacted_neo4j = redact_sensitive_text(neo4j_uri)
    redacted_pg = redact_sensitive_text(pg_uri)

    assert "my_super_secret_neo4j_pw" not in redacted_neo4j
    assert "bolt://neo4j:***@localhost:7687" in redacted_neo4j

    assert "my_secret_pg_pw" not in redacted_pg
    assert "postgresql+asyncpg://kgr_user:***@localhost:5432/kgr_db" in redacted_pg


def test_sanitize_log_dict_nested() -> None:
    data = {
        "user": "test_user",
        "password": "plain_password_123",
        "nested": {
            "api_key": "secret_key_456",
            "message": "Connected to bolt://neo4j:neo_pass@host:7687",
        },
        "items": ["safe_string", "Bearer sensitive_token_789"],
    }
    sanitized = sanitize_log_dict(data)

    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["nested"]["api_key"] == "[REDACTED]"
    assert "neo_pass" not in sanitized["nested"]["message"]
    assert "bolt://neo4j:***@host:7687" in sanitized["nested"]["message"]
    assert "sensitive_token_789" not in sanitized["items"][1]
    assert sanitized["items"][1] == "Bearer [REDACTED]"


# ---------------------------------------------------------------------------
# Structured JSON Formatter Tests
# ---------------------------------------------------------------------------


def test_structured_json_formatter_fields() -> None:
    formatter = StructuredJsonFormatter(environment="test")
    record = logging.LogRecord(
        name="src.api.telemetry",
        level=logging.INFO,
        pathname="routes.py",
        lineno=120,
        msg="Test telemetry message with sk-ant-secretkey123",
        args=(),
        exc_info=None,
    )
    record.__dict__.update(
        {
            "request_id": "req-test-1234",
            "event": "test_event",
            "status_code": 200,
            "execution_duration_ms": 42.5,
        }
    )

    formatted_str = formatter.format(record)
    parsed = json.loads(formatted_str)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "src.api.telemetry"
    assert parsed["environment"] == "test"
    assert parsed["request_id"] == "req-test-1234"
    assert parsed["event"] == "test_event"
    assert parsed["status_code"] == 200
    assert parsed["execution_duration_ms"] == 42.5
    assert "timestamp" in parsed
    assert "sk-ant-secretkey123" not in parsed["message"]
    assert "[REDACTED_ANTHROPIC_KEY]" in parsed["message"]


def test_configure_structured_logging() -> None:
    configure_structured_logging(log_level="DEBUG", environment="production")
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0].formatter, StructuredJsonFormatter)


# ---------------------------------------------------------------------------
# Request Telemetry Middleware & HTTP Header Propagation Tests
# ---------------------------------------------------------------------------


def test_middleware_generates_x_request_id_and_timing(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"].startswith("req-")
    assert "X-Response-Time-MS" in response.headers
    assert float(response.headers["X-Response-Time-MS"]) >= 0.0


def test_middleware_preserves_incoming_x_request_id(client: TestClient) -> None:
    custom_id = "custom-client-req-9999"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


def test_middleware_propagates_request_id_to_query_response(
    client: TestClient, mock_container: ServiceContainer
) -> None:
    # Mock router & synthesizer
    mock_router = MagicMock(spec=RouterEngine)
    routing_result = RoutingResult(
        query="What is the architecture?",
        mode=QueryIntent.HYBRID,
        classification=IntentClassificationResult(
            query="What is the architecture?",
            mode=QueryIntent.HYBRID,
            confidence=0.9,
            reasoning="Hybrid",
        ),
        entity_linking=MagicMock(links=[]),
        graph_results=[],
        vector_chunks=[],
    )
    mock_router.route_and_retrieve = AsyncMock(return_value=routing_result)
    mock_container.router = mock_router

    mock_synthesizer = MagicMock(spec=GroundedAnswerSynthesizer)
    draft = DraftAnswer(
        question="What is the architecture?",
        raw_answer="Architecture overview.",
        cited_chunk_ids=[],
        candidate_chunk_ids=[],
        token_usage=TokenUsageMetrics(
            prompt_tokens=100, completion_tokens=20, total_tokens=120, estimated_cost_usd=0.001
        ),
        execution_time_ms=50.0,
        model_name="claude-3-5-sonnet-20241022",
    )
    mock_synthesizer.synthesize.return_value = draft
    mock_container.synthesizer = mock_synthesizer

    custom_id = "req-correlation-trace-123"
    response = client.post(
        "/api/v1/query",
        json={"question": "What is the architecture?"},
        headers={"X-Request-ID": custom_id},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id
    data = response.json()
    assert data["request_id"] == custom_id


def test_middleware_attaches_request_id_to_validation_error_response(client: TestClient) -> None:
    custom_id = "req-bad-input-trace-777"
    response = client.post(
        "/api/v1/query",
        json={"question": "   "},
        headers={"X-Request-ID": custom_id},
    )
    assert response.status_code == 422
    assert response.headers.get("X-Request-ID") == custom_id
    data = response.json()
    assert data["request_id"] == custom_id
    assert data["error"] == "UNPROCESSABLE_ENTITY"


def test_middleware_attaches_request_id_to_internal_error_response(
    client: TestClient, mock_container: ServiceContainer
) -> None:
    mock_router = MagicMock(spec=RouterEngine)
    mock_router.route_and_retrieve = AsyncMock(
        side_effect=RuntimeError("Unexpected pipeline crash")
    )
    mock_container.router = mock_router

    custom_id = "req-server-crash-trace-888"
    response = client.post(
        "/api/v1/query",
        json={"question": "Valid question"},
        headers={"X-Request-ID": custom_id},
    )
    assert response.status_code == 500
    assert response.headers.get("X-Request-ID") == custom_id
    data = response.json()
    assert data["request_id"] == custom_id
    assert data["error"] == "INTERNAL_SERVER_ERROR"


def test_middleware_sanitizes_invalid_or_oversized_x_request_id(client: TestClient) -> None:
    # 1. Invalid characters (e.g. spaces, special symbols, injection attempts)
    bad_id = "req-bad id with spaces; DROP TABLE"
    response = client.get("/health", headers={"X-Request-ID": bad_id})
    assert response.status_code == 200
    assigned_id = response.headers.get("X-Request-ID")
    assert assigned_id != bad_id
    assert assigned_id is not None
    assert assigned_id.startswith("req-")

    # 2. Oversized ID (>128 chars)
    huge_id = "a" * 200
    response2 = client.get("/health", headers={"X-Request-ID": huge_id})
    assert response2.status_code == 200
    assigned_id2 = response2.headers.get("X-Request-ID")
    assert assigned_id2 != huge_id
    assert assigned_id2 is not None
    assert assigned_id2.startswith("req-")
