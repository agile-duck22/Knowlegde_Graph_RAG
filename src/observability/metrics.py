"""Telemetry and Metrics Data Models (TASK-405).

Provides in-memory telemetry structures and metrics tracking definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TelemetryEvent:
    """Standardized telemetry event payload."""

    event_name: str
    request_id: str
    method: str
    path: str
    status_code: int
    duration_ms: float
    retrieval_mode: str | None = None
    routing_confidence: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    citations_count: int = 0
    warnings_count: int = 0
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
