"""Unit tests verifying test harness mock factories operate offline without network."""

from unittest.mock import MagicMock

import pytest

from src.config import Settings


def test_mock_settings_fixture(mock_settings: Settings) -> None:
    """Verify mock_settings provides valid test configuration."""
    assert mock_settings.environment == "test"
    assert mock_settings.log_level == "DEBUG"
    assert mock_settings.anthropic_api_key is not None
    assert mock_settings.anthropic_api_key.get_secret_value() == "sk-ant-mock-key-12345"


def test_mock_anthropic_client(mock_anthropic_client: MagicMock) -> None:
    """Verify mock Anthropic client returns structured mock completion."""
    response = mock_anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Test query"}],
    )
    assert len(response.content) == 1
    assert "Payment Gateway" in response.content[0].text
    assert response.usage.input_tokens > 0


def test_mock_neo4j_driver(mock_neo4j_driver: MagicMock) -> None:
    """Verify mock Neo4j driver returns structured graph query records."""
    with mock_neo4j_driver.session() as session:
        result = session.run("MATCH (n) RETURN n")
        records = list(result)
        assert len(records) == 1
        assert records[0]["source"] == "service:payment-gateway"
        assert records[0]["rel_type"] == "DEPENDS_ON"


@pytest.mark.asyncio
async def test_mock_postgres_pool(mock_postgres_pool: MagicMock) -> None:
    """Verify mock asyncpg pool executes and fetches rows."""
    async with mock_postgres_pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM chunks")
        assert len(rows) == 1
        assert rows[0]["chunk_id"] == "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"
        assert rows[0]["similarity_score"] == 0.88
