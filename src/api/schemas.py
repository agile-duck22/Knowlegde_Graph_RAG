"""API Data Models & Schemas (TASK-404).

Defines request and response payloads matching OpenAPI specifications in docs/API_SPEC.md.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.routing.intent_classifier import QueryIntent

# ---------------------------------------------------------------------------
# Health & Readiness Schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness probe response model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["healthy"] = "healthy"
    version: str = Field(default="0.1.0", description="Application version.")
    timestamp: str = Field(..., description="Current UTC timestamp.")


class DatabaseStatus(BaseModel):
    """Database connectivity status map."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    neo4j: Literal["connected", "disconnected"] = Field(..., description="Neo4j connection status.")
    postgres: Literal["connected", "disconnected"] = Field(
        ..., description="PostgreSQL connection status."
    )


class ReadyResponse(BaseModel):
    """Readiness probe response model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ready", "unavailable"] = Field(
        ..., description="Readiness status of the service."
    )
    databases: DatabaseStatus = Field(..., description="Database connection states.")
    timestamp: str = Field(..., description="Current UTC timestamp.")
    error: str | None = Field(default=None, description="Error explanation if unavailable.")


# ---------------------------------------------------------------------------
# Query Endpoint Schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Request payload for POST /api/v1/query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The user question or inquiry (max 1000 chars).",
    )
    retrieval_override: (
        QueryIntent | Literal["GRAPH_ONLY", "VECTOR_ONLY", "HYBRID", "OUT_OF_SCOPE"] | None
    ) = Field(
        default=None,
        description="Optional manual override for retrieval execution mode.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Top-k chunks to retrieve (default: 5, range: 1 to 20).",
    )
    similarity_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional cosine similarity threshold override for vector retrieval.",
    )
    include_provenance: bool = Field(
        default=True,
        description="Whether to include retrieved chunks and graph paths in response.",
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty_question(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Question cannot be empty or whitespace.")
        return trimmed


class CitationResponseItem(BaseModel):
    """Validated citation item in query response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., description="Canonical UUID of the cited chunk.")
    document_id: str = Field(..., description="Source document identifier.")
    excerpt: str = Field(..., description="Text excerpt from the cited chunk.")


class GraphPathResponseItem(BaseModel):
    """Graph path or relationship in query response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(..., description="Source entity ID or label.")
    relation: str = Field(..., description="Relationship type.")
    target: str = Field(..., description="Target entity ID or label.")


class TokenUsageResponse(BaseModel):
    """Token consumption and estimated USD cost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(default=0, description="Tokens in prompt.")
    completion_tokens: int = Field(default=0, description="Tokens in completion.")
    total_tokens: int = Field(default=0, description="Total tokens.")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated USD cost.")


class QueryResponse(BaseModel):
    """Response payload for POST /api/v1/query matching docs/API_SPEC.md."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(
        default_factory=lambda: f"req-{uuid4()}",
        description="Unique request tracing ID.",
    )
    question: str = Field(..., description="Original user question.")
    answer: str = Field(..., description="Grounded, sanitized natural language answer.")
    retrieval_mode: str = Field(..., description="Execution retrieval mode.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Routing/retrieval confidence.")
    citations: list[CitationResponseItem] = Field(
        default_factory=list, description="Validated citations referencing source chunks."
    )
    graph_paths: list[GraphPathResponseItem] = Field(
        default_factory=list, description="Retrieved knowledge graph paths and relations."
    )
    execution_time_ms: float = Field(..., description="Total query execution time in milliseconds.")
    token_usage: TokenUsageResponse = Field(
        ..., description="Token usage and estimated cost breakdown."
    )
    warnings: list[str] = Field(
        default_factory=list, description="Validation or pipeline execution warnings."
    )


# ---------------------------------------------------------------------------
# Error Response Schemas
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error response payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error: str = Field(..., description="Machine-readable error code.")
    message: str = Field(..., description="Human-readable error description.")
    request_id: str | None = Field(default=None, description="Request ID if available.")
    details: dict[str, Any] | None = Field(default=None, description="Optional extra details.")
