"""Unit tests for ChunkExtractor (TASK-104).

Verifies structured extraction, retry behavior on transient errors,
immediate failure on non-retryable errors, malformed response handling,
and schema validation — strictly without live Anthropic API calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import anthropic
import pytest
from httpx import Request, Response

from src.config import Settings
from src.extraction.extractor import (
    ChunkExtractor,
    ExtractionError,
    NonRetryableExtractionError,
    RetryExhaustedError,
)
from src.extraction.schemas import ChunkExtractionResult
from src.ingest.chunker import TextChunk

SAMPLE_CHUNK = TextChunk(
    chunk_id="00000000-0000-0000-0000-000000000001",
    document_id="doc-arch-001",
    chunk_index=0,
    content="The Payment Gateway service (service:payment-gateway) is owned by Team Core (team:core).",
    char_start=0,
    char_end=85,
    metadata={"title": "Payment Architecture", "doc_type": "architecture_spec"},
)

VALID_PAYLOAD = {
    "chunk_id": "00000000-0000-0000-0000-000000000001",
    "document_id": "doc-arch-001",
    "extraction_model": "claude-3-5-sonnet-20241022",
    "entities": [
        {
            "canonical_id": "service:payment-gateway",
            "name": "Payment Gateway",
            "entity_type": "Service",
            "description": "Payment Gateway service",
            "aliases": [],
            "confidence": 0.95,
            "provenance": {
                "chunk_id": "00000000-0000-0000-0000-000000000001",
                "document_id": "doc-arch-001",
            },
        },
        {
            "canonical_id": "team:core",
            "name": "Team Core",
            "entity_type": "Team",
            "description": "Core engineering team",
            "aliases": [],
            "confidence": 0.90,
            "provenance": {
                "chunk_id": "00000000-0000-0000-0000-000000000001",
                "document_id": "doc-arch-001",
            },
        },
    ],
    "relationships": [
        {
            "source_canonical_id": "service:payment-gateway",
            "source_type": "Service",
            "target_canonical_id": "team:core",
            "target_type": "Team",
            "relation_type": "OWNED_BY",
            "description": "Payment Gateway is owned by Team Core",
            "confidence": 0.95,
            "source_chunk_ids": ["00000000-0000-0000-0000-000000000001"],
            "provenance": {
                "chunk_id": "00000000-0000-0000-0000-000000000001",
                "document_id": "doc-arch-001",
            },
        }
    ],
}


def _make_mock_response(text: str) -> MagicMock:
    """Helper to build a mock Anthropic message response."""
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.text = text
    mock_resp.content = [mock_content]
    mock_resp.usage.input_tokens = 100
    mock_resp.usage.output_tokens = 50
    return mock_resp


def _make_api_status_error(
    error_cls: type[anthropic.APIStatusError],
    status_code: int = 400,
    headers: dict[str, str] | None = None,
) -> anthropic.APIStatusError:
    """Helper to build Anthropic APIStatusError instances with mock response."""
    req = Request("POST", "https://api.anthropic.com/v1/messages")
    resp = Response(status_code, request=req, headers=headers or {})
    return error_cls(
        message=f"API error with status {status_code}",
        response=resp,  # type: ignore[arg-type]
        body={"error": {"type": "error", "message": "test error"}},
    )


def test_successful_extraction_json() -> None:
    """Test successful extraction when model returns plain JSON string."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response(json.dumps(VALID_PAYLOAD))

    settings = Settings(extraction_max_retries=2)
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    result = extractor.extract(SAMPLE_CHUNK)
    assert isinstance(result, ChunkExtractionResult)
    assert result.chunk_id == SAMPLE_CHUNK.chunk_id
    assert result.document_id == SAMPLE_CHUNK.document_id
    assert len(result.entities) == 2
    assert len(result.relationships) == 1
    assert result.entities[0].canonical_id == "service:payment-gateway"
    assert result.relationships[0].relation_type == "OWNED_BY"


def test_successful_extraction_markdown_fenced_json() -> None:
    """Test extractor correctly parses JSON enclosed in markdown code fences."""
    fenced_text = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response(fenced_text)

    settings = Settings(extraction_max_retries=1)
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    result = extractor.extract(SAMPLE_CHUNK)
    assert isinstance(result, ChunkExtractionResult)
    assert len(result.entities) == 2


def test_non_retryable_auth_error_fails_immediately() -> None:
    """Test AuthenticationError (401) is not retried and fails immediately."""
    mock_client = MagicMock()
    auth_err = _make_api_status_error(anthropic.AuthenticationError, 401)
    mock_client.messages.create.side_effect = auth_err

    settings = Settings(extraction_max_retries=3)
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    with pytest.raises(NonRetryableExtractionError) as exc_info:
        extractor.extract(SAMPLE_CHUNK)

    assert "AuthenticationError" in str(exc_info.value)
    assert mock_client.messages.create.call_count == 1


def test_non_retryable_bad_request_fails_immediately() -> None:
    """Test BadRequestError (400) fails immediately."""
    mock_client = MagicMock()
    bad_req_err = _make_api_status_error(anthropic.BadRequestError, 400)
    mock_client.messages.create.side_effect = bad_req_err

    settings = Settings(extraction_max_retries=3)
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    with pytest.raises(NonRetryableExtractionError):
        extractor.extract(SAMPLE_CHUNK)

    assert mock_client.messages.create.call_count == 1


def test_transient_rate_limit_retry_then_success() -> None:
    """Test RateLimitError (429) is retried with backoff and succeeds on subsequent attempt."""
    mock_client = MagicMock()
    rate_limit_err = _make_api_status_error(anthropic.RateLimitError, 429)

    # 1 fail, then success
    mock_client.messages.create.side_effect = [
        rate_limit_err,
        _make_mock_response(json.dumps(VALID_PAYLOAD)),
    ]

    settings = Settings(extraction_max_retries=2)
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    with patch("tenacity.nap.time.sleep", return_value=None):
        result = extractor.extract(SAMPLE_CHUNK)

    assert isinstance(result, ChunkExtractionResult)
    assert mock_client.messages.create.call_count == 2


def test_transient_rate_limit_with_retry_after_header() -> None:
    """Test RateLimitError (429) respects the retry-after response header during backoff."""
    mock_client = MagicMock()
    rate_limit_err = _make_api_status_error(
        anthropic.RateLimitError, 429, headers={"retry-after": "5"}
    )

    mock_client.messages.create.side_effect = [
        rate_limit_err,
        _make_mock_response(json.dumps(VALID_PAYLOAD)),
    ]

    settings = Settings(extraction_max_retries=2)
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    sleep_calls: list[float] = []

    def mock_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    with patch("tenacity.nap.time.sleep", side_effect=mock_sleep):
        result = extractor.extract(SAMPLE_CHUNK)

    assert isinstance(result, ChunkExtractionResult)
    assert mock_client.messages.create.call_count == 2
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 5.0


def test_transient_internal_server_error_exhausted() -> None:
    """Test InternalServerError (500) retried until max_retries, then raises RetryExhaustedError."""
    mock_client = MagicMock()
    server_err = _make_api_status_error(anthropic.InternalServerError, 500)
    mock_client.messages.create.side_effect = server_err

    settings = Settings(extraction_max_retries=2)  # max_attempts = 3
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    with patch("tenacity.nap.time.sleep", return_value=None):
        with pytest.raises(RetryExhaustedError) as exc_info:
            extractor.extract(SAMPLE_CHUNK)

    assert "attempts" in str(exc_info.value)
    assert mock_client.messages.create.call_count == 3


def test_malformed_json_retried_then_fails() -> None:
    """Test malformed JSON is retried and raises RetryExhaustedError when attempts run out."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response("This is not valid JSON at all")

    settings = Settings(extraction_max_retries=2)
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    with patch("tenacity.nap.time.sleep", return_value=None):
        with pytest.raises(RetryExhaustedError) as exc_info:
            extractor.extract(SAMPLE_CHUNK)

    assert "schema validation" in str(exc_info.value) or "JSON" in str(exc_info.value)
    assert mock_client.messages.create.call_count == 3


def test_schema_validation_failure_retried_then_fails() -> None:
    """Test response with invalid ontology schema (e.g. invalid entity_type) is retried and fails."""
    invalid_schema_payload = {
        **VALID_PAYLOAD,
        "entities": [
            {
                "canonical_id": "invalid:foo",
                "name": "Foo",
                "entity_type": "InvalidTypeNotInOntology",
                "confidence": 0.9,
                "provenance": {
                    "chunk_id": "00000000-0000-0000-0000-000000000001",
                    "document_id": "doc-arch-001",
                },
            }
        ],
    }
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response(json.dumps(invalid_schema_payload))

    settings = Settings(extraction_max_retries=1)
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    with patch("tenacity.nap.time.sleep", return_value=None):
        with pytest.raises(RetryExhaustedError):
            extractor.extract(SAMPLE_CHUNK)

    assert mock_client.messages.create.call_count == 2


def test_empty_entities_and_relationships_valid() -> None:
    """Test that a chunk containing no extractable entities returns a valid empty result."""
    empty_payload = {
        "chunk_id": "00000000-0000-0000-0000-000000000001",
        "document_id": "doc-arch-001",
        "extraction_model": "claude-3-5-sonnet-20241022",
        "entities": [],
        "relationships": [],
    }
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _make_mock_response(json.dumps(empty_payload))

    settings = Settings(extraction_max_retries=1)
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    result = extractor.extract(SAMPLE_CHUNK)
    assert isinstance(result, ChunkExtractionResult)
    assert len(result.entities) == 0
    assert len(result.relationships) == 0


def test_blank_chunk_raises_extraction_error() -> None:
    """Test extracting from a chunk with empty content raises ExtractionError."""
    blank_chunk = TextChunk(
        chunk_id="00000000-0000-0000-0000-000000000001",
        document_id="doc-arch-001",
        chunk_index=0,
        content="   ",
        char_start=0,
        char_end=3,
    )
    mock_client = MagicMock()
    settings = Settings()
    extractor = ChunkExtractor(client=mock_client, settings=settings)

    with pytest.raises(ExtractionError) as exc_info:
        extractor.extract(blank_chunk)

    assert "chunk_text must not be empty" in str(exc_info.value)
    assert mock_client.messages.create.call_count == 0
