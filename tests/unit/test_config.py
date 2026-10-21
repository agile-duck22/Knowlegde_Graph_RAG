"""Unit tests for configuration loading, validation, and secret masking."""

import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from src.config import Settings, get_settings


def test_default_settings_load() -> None:
    """Verify default settings load cleanly without external connections."""
    settings = Settings(_env_file=None)
    assert settings.app_name == "kgr"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.llm_model == "claude-3-5-sonnet-20241022"
    assert settings.routing_model == "claude-3-haiku-20240307"
    assert settings.embedding_provider == "openai"
    assert settings.embedding_dimensions == 1536
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.router_confidence_threshold == 0.65
    assert settings.graph_max_hops == 2
    assert settings.postgres_port == 5432
    assert settings.neo4j_uri == "bolt://localhost:7687"


def test_environment_variable_overrides() -> None:
    """Verify environment variables override default values."""
    env_overrides = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "CHUNK_SIZE": "800",
        "CHUNK_OVERLAP": "100",
        "ROUTER_CONFIDENCE_THRESHOLD": "0.75",
        "GRAPH_MAX_HOPS": "3",
        "POSTGRES_PORT": "5433",
    }
    with patch.dict(os.environ, env_overrides, clear=False):
        settings = Settings(_env_file=None)
        assert settings.environment == "test"
        assert settings.log_level == "DEBUG"
        assert settings.chunk_size == 800
        assert settings.chunk_overlap == 100
        assert settings.router_confidence_threshold == 0.75
        assert settings.graph_max_hops == 3
        assert settings.postgres_port == 5433


def test_explicit_constructor_kwargs() -> None:
    """Verify constructor keyword arguments work for test fixtures."""
    settings = Settings(
        chunk_size=1000,
        chunk_overlap=200,
        embedding_dimensions=384,
        embedding_provider="huggingface",
    )
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.embedding_dimensions == 384
    assert settings.embedding_provider == "huggingface"


def test_invalid_chunk_overlap_raises_error() -> None:
    """Validation: chunk_overlap >= chunk_size must raise ValidationError."""
    with pytest.raises(ValidationError):
        Settings(chunk_size=500, chunk_overlap=500)
    with pytest.raises(ValidationError):
        Settings(chunk_size=500, chunk_overlap=600)


def test_invalid_port_raises_error() -> None:
    """Validation: postgres_port outside 1-65535 must raise ValidationError."""
    with pytest.raises(ValidationError):
        Settings(postgres_port=0)
    with pytest.raises(ValidationError):
        Settings(postgres_port=70000)


def test_invalid_thresholds_raise_error() -> None:
    """Validation: router_confidence_threshold outside [0.0, 1.0] must raise ValidationError."""
    with pytest.raises(ValidationError):
        Settings(router_confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        Settings(router_confidence_threshold=-0.1)


def test_invalid_graph_hops_raise_error() -> None:
    """Validation: graph_max_hops > 3 must raise ValidationError to prevent combinatorial explosion."""
    with pytest.raises(ValidationError):
        Settings(graph_max_hops=4)
    with pytest.raises(ValidationError):
        Settings(graph_max_hops=0)


def test_invalid_neo4j_uri_raises_error() -> None:
    """Validation: neo4j_uri with invalid protocol must raise ValidationError."""
    with pytest.raises(ValidationError):
        Settings(neo4j_uri="http://localhost:7474")


def test_secret_masking_in_repr() -> None:
    """Security: Passwords and API keys must be masked in string representation."""
    settings = Settings(
        anthropic_api_key=SecretStr("sk-ant-test-secret-key-12345"),
        neo4j_password=SecretStr("super_secret_neo4j_password"),
        postgres_password=SecretStr("super_secret_pg_password"),
    )
    repr_str = repr(settings)
    assert "sk-ant-test-secret-key-12345" not in repr_str
    assert "super_secret_neo4j_password" not in repr_str
    assert "super_secret_pg_password" not in repr_str
    assert "**********" in repr_str


def test_missing_anthropic_key_in_development_is_allowed() -> None:
    """Offline safety: In development/test mode, missing API keys must not block imports."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
        settings = Settings(environment="development", anthropic_api_key=None)
        assert settings.anthropic_api_key is None


def test_missing_anthropic_key_in_production_raises_error() -> None:
    """Security: In production mode, missing active provider key must raise ValidationError."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="production", llm_provider="anthropic", anthropic_api_key=None)


def test_cached_settings_factory() -> None:
    """Verify get_settings() returns a cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
