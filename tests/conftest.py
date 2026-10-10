"""Global pytest fixtures, configuration, and lightweight mock factories."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from src.config import Settings

# ---------------------------------------------------------------------------
# Base Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings() -> Settings:
    """Isolated settings fixture configured for test environment with mock values."""
    return Settings(
        environment="test",
        log_level="DEBUG",
        anthropic_api_key=SecretStr("sk-ant-mock-key-12345"),
        openai_api_key=SecretStr("sk-mock-openai-key-12345"),
        neo4j_password=SecretStr("mock_neo4j_password"),
        postgres_password=SecretStr("mock_postgres_password"),
        chunk_size=500,
        chunk_overlap=50,
        vector_top_k=3,
        graph_max_hops=2,
    )


# ---------------------------------------------------------------------------
# Data Model Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunk_record() -> dict[str, Any]:
    """Sample chunk metadata record conforming to PostgreSQL data model."""
    return {
        "id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
        "document_id": "doc-payment-gateway-spec",
        "chunk_index": 0,
        "content": "Payment Gateway service makes synchronous gRPC calls to Checkout Service.",
        "token_count": 12,
        "char_start": 0,
        "char_end": 74,
        "metadata": {"doc_type": "architecture_spec"},
    }


@pytest.fixture
def sample_extracted_triples() -> list[dict[str, Any]]:
    """Sample extracted entity triples conforming to domain ontology."""
    return [
        {
            "source_id": "service:payment-gateway",
            "source_name": "Payment Gateway",
            "source_type": "Service",
            "relation_type": "DEPENDS_ON",
            "target_id": "service:checkout-service",
            "target_name": "Checkout Service",
            "target_type": "Service",
            "source_chunk_ids": ["a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"],
            "confidence": 0.95,
        }
    ]


# ---------------------------------------------------------------------------
# Lightweight Client Mock Factories (Zero Live Services Needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic_client() -> MagicMock:
    """Mock factory for Anthropic Claude messages client."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "The Payment Gateway depends directly on Checkout Service [a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d]."
    mock_response.content = [mock_content]
    mock_response.usage.input_tokens = 150
    mock_response.usage.output_tokens = 35
    client.messages.create.return_value = mock_response
    return client


@pytest.fixture
def mock_neo4j_driver() -> MagicMock:
    """Mock factory for Neo4j Python driver and session execution."""
    driver = MagicMock()
    session = MagicMock()

    # Mock Cypher execution results
    mock_result = MagicMock()
    mock_record = {
        "source": "service:payment-gateway",
        "rel_type": "DEPENDS_ON",
        "target_name": "Checkout Service",
        "source_chunk_ids": ["a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"],
    }
    mock_result.__iter__.return_value = [mock_record]
    mock_result.data.return_value = [mock_record]

    session.run.return_value = mock_result
    session.__enter__.return_value = session
    session.__exit__.return_value = False
    driver.session.return_value = session
    return driver


class _AsyncContextManagerMock:
    """Helper for async context managers like pool.acquire()."""

    def __init__(self, target: Any) -> None:
        self.target = target

    async def __aenter__(self) -> Any:
        return self.target

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


@pytest.fixture
def mock_postgres_pool() -> MagicMock:
    """Mock factory for asyncpg PostgreSQL connection pool."""
    pool = MagicMock()
    conn = AsyncMock()

    mock_row = {
        "chunk_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
        "content": "Payment Gateway service makes synchronous gRPC calls to Checkout Service.",
        "similarity_score": 0.88,
    }
    conn.fetch.return_value = [mock_row]
    conn.fetchrow.return_value = mock_row
    conn.execute.return_value = "INSERT 0 1"

    pool.acquire.return_value = _AsyncContextManagerMock(conn)
    return pool
