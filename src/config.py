"""Centralized application settings and environment configuration.

Uses Pydantic Settings (BaseSettings) for typed configuration management,
environment variable loading (.env), secret masking, and constraint validation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_ENV = object()


class Settings(BaseSettings):
    """Centralized, typed application settings for Knowledge Graph RAG."""

    def __init__(
        self,
        _env_file: Any = _DEFAULT_ENV,
        **values: Any,
    ) -> None:
        if _env_file is not _DEFAULT_ENV:
            super().__init__(_env_file=_env_file, **values)
        else:
            super().__init__(**values)

    # Application & Environment
    app_name: str = Field(default="kgr", description="Application service name")
    environment: Literal["development", "test", "production"] = Field(
        default="development", description="Runtime environment"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging verbosity level"
    )

    # LLM Provider Configuration
    llm_provider: Literal["groq", "gemini", "openrouter", "anthropic"] = Field(
        default="anthropic",
        description="Primary LLM provider (groq, gemini, openrouter, anthropic)",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, description="Anthropic API key for synthesis and extraction"
    )
    gemini_api_key: SecretStr | None = Field(
        default=None, description="Google Gemini API key for free-tier inference"
    )
    groq_api_key: SecretStr | None = Field(
        default=None, description="Groq API key for free-tier inference"
    )
    openrouter_api_key: SecretStr | None = Field(
        default=None, description="OpenRouter API key for free-tier inference"
    )
    llm_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Primary LLM model for synthesis and extraction",
    )
    routing_model: str = Field(
        default="claude-3-haiku-20240307",
        description="Lightweight LLM model for query intent routing",
    )
    extraction_model: str | None = Field(
        default=None,
        description="Optional model override for entity extraction (defaults to llm_model)",
    )
    synthesis_model: str | None = Field(
        default=None,
        description="Optional model override for answer synthesis (defaults to llm_model)",
    )
    extraction_provider: Literal["groq", "gemini", "openrouter", "anthropic"] | None = Field(
        default=None,
        description="Optional provider override for entity extraction (defaults to llm_provider)",
    )
    routing_provider: Literal["groq", "gemini", "openrouter", "anthropic"] | None = Field(
        default=None,
        description="Optional provider override for query routing (defaults to llm_provider)",
    )
    synthesis_provider: Literal["groq", "gemini", "openrouter", "anthropic"] | None = Field(
        default=None,
        description="Optional provider override for answer synthesis (defaults to llm_provider)",
    )
    max_completion_tokens: int = Field(
        default=2048, ge=1, le=8192, description="Max output tokens for LLM generation"
    )
    request_timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=300.0, description="Upstream API call timeout in seconds"
    )

    # Embedding Configuration
    embedding_provider: Literal[
        "openai", "sentence_transformers", "huggingface", "local", "cohere"
    ] = Field(
        default="openai",
        description="Vector embedding provider (openai, sentence_transformers, huggingface, local, cohere)",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Vector embedding model name"
    )
    embedding_dimensions: int = Field(
        default=1536,
        ge=1,
        le=4096,
        description="Vector dimension (must match PostgreSQL schema)",
    )
    openai_api_key: SecretStr | None = Field(
        default=None, description="OpenAI API key for embeddings (when provider is openai)"
    )
    allow_paid_embeddings: bool = Field(
        default=False,
        description="Explicit authorization flag required for paid OpenAI embedding generation",
    )

    # Neo4j Graph Database
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j database username")
    neo4j_password: SecretStr | None = Field(
        default=SecretStr("neo4j_password"), description="Neo4j database password"
    )

    # PostgreSQL Relational & Vector Database
    postgres_host: str = Field(default="localhost", description="PostgreSQL host address")
    postgres_port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port number")
    postgres_db: str = Field(default="kgr_db", description="PostgreSQL database name")
    postgres_user: str = Field(default="kgr_user", description="PostgreSQL username")
    postgres_password: SecretStr | None = Field(
        default=SecretStr("kgr_password"), description="PostgreSQL password"
    )

    # Ingestion & Chunking Parameters
    chunk_size: int = Field(default=500, ge=50, le=4000, description="Target chunk size in tokens")
    chunk_overlap: int = Field(
        default=50, ge=0, le=1000, description="Overlap between consecutive chunks in tokens"
    )
    extraction_batch_size: int = Field(
        default=10, ge=1, le=100, description="Concurrent chunk extraction batch size"
    )
    extraction_max_retries: int = Field(
        default=3, ge=0, le=10, description="Max retries for LLM extraction failures"
    )

    # Retrieval & Routing Knobs
    router_confidence_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Confidence threshold below which query triggers HYBRID fallback",
    )
    vector_top_k: int = Field(
        default=5, ge=1, le=50, description="Top-k chunks for VECTOR_ONLY retrieval"
    )
    hybrid_vector_top_k: int = Field(
        default=3, ge=1, le=20, description="Top-k chunks for HYBRID retrieval"
    )
    vector_similarity_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score threshold",
    )
    graph_max_hops: int = Field(
        default=2, ge=1, le=3, description="Maximum graph traversal depth limit (D <= 3)"
    )
    graph_node_limit: int = Field(
        default=25, ge=1, le=200, description="Maximum graph nodes returned per query"
    )
    max_context_tokens: int = Field(
        default=6000,
        ge=500,
        le=32000,
        description="Token budget for merged synthesis context",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("neo4j_uri")
    @classmethod
    def validate_neo4j_uri(cls, v: str) -> str:
        if not (
            v.startswith("bolt://")
            or v.startswith("neo4j://")
            or v.startswith("bolt+s://")
            or v.startswith("neo4j+s://")
        ):
            raise ValueError(
                f"Invalid Neo4j URI scheme: '{v}'. Must start with bolt://, neo4j://, bolt+s://, or neo4j+s://"
            )
        return v

    def get_extraction_model(self) -> str:
        """Return the effective model ID for entity extraction."""
        if self.extraction_model and self.extraction_model.strip():
            return self.extraction_model.strip()
        return self.llm_model.strip()

    def get_routing_model(self) -> str:
        """Return the effective model ID for query routing."""
        return self.routing_model.strip()

    def get_synthesis_model(self) -> str:
        """Return the effective model ID for answer synthesis."""
        if self.synthesis_model and self.synthesis_model.strip():
            return self.synthesis_model.strip()
        return self.llm_model.strip()

    def get_extraction_provider(self) -> str:
        """Return the effective LLM provider for entity extraction."""
        if self.extraction_provider and self.extraction_provider.strip():
            return self.extraction_provider.strip().lower()
        return self.llm_provider.strip().lower()

    def get_routing_provider(self) -> str:
        """Return the effective LLM provider for query routing."""
        if self.routing_provider and self.routing_provider.strip():
            return self.routing_provider.strip().lower()
        return self.llm_provider.strip().lower()

    def get_synthesis_provider(self) -> str:
        """Return the effective LLM provider for answer synthesis."""
        if self.synthesis_provider and self.synthesis_provider.strip():
            return self.synthesis_provider.strip().lower()
        return self.llm_provider.strip().lower()

    def get_api_key_for_provider(self, provider: str | None = None) -> SecretStr | None:
        """Return the configured API key SecretStr for the specified or active LLM provider."""
        p = (provider or self.llm_provider).strip().lower()
        if p == "groq":
            return self.groq_api_key
        if p == "gemini":
            return self.gemini_api_key
        if p == "openrouter":
            return self.openrouter_api_key
        if p == "anthropic":
            return self.anthropic_api_key
        return None

    @model_validator(mode="after")
    def validate_chunk_overlap_and_production_keys(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )
        if self.environment == "production":
            key = self.get_api_key_for_provider(self.llm_provider)
            if not key or not key.get_secret_value() or key.get_secret_value().startswith("your_"):
                raise ValueError(
                    f"{self.llm_provider.upper()}_API_KEY is required in production environment"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached factory function returning application settings singleton."""
    return Settings()
