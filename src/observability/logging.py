"""Structured Logging and Redaction Formatter (TASK-405).

Provides JSON formatted logs to stdout with automatic redaction of:
- API keys (Bearer tokens, Anthropic sk-ant-* keys, OpenAI keys)
- Database credentials in connection URIs (Neo4j bolt://, PostgreSQL postgresql://)
- Raw document and prompt payloads
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

_GROQ_KEY_PATTERN = re.compile(r"gsk_[A-Za-z0-9_\-]{20,}", re.IGNORECASE)
_GEMINI_KEY_PATTERN = re.compile(r"AIzaSy[A-Za-z0-9_\-]{30,}", re.IGNORECASE)
_OPENROUTER_KEY_PATTERN = re.compile(r"sk-or-v1-[A-Za-z0-9_\-]{20,}", re.IGNORECASE)
_AUTH_BEARER_PATTERN = re.compile(r"Bearer\s+([A-Za-z0-9_\-\.]+)", re.IGNORECASE)
_ANTHROPIC_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]+", re.IGNORECASE)
_OPENAI_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{20,}", re.IGNORECASE)
_NEO4J_URI_PASSWORD_PATTERN = re.compile(r"(bolt(?:\+s)?://[^:]+:)([^@]+)(@)", re.IGNORECASE)
_PG_URI_PASSWORD_PATTERN = re.compile(
    r"(postgres(?:ql)?(?:\+asyncpg)?://[^:]+:)([^@]+)(@)", re.IGNORECASE
)
_PASSWORD_FIELD_PATTERN = re.compile(
    r'(["\']?(?:password|secret|api_key|token|auth)["\']?\s*[:=]\s*["\'])([^"\']+)(["\'])',
    re.IGNORECASE,
)


def redact_sensitive_text(text: str) -> str:
    """Redact API keys, tokens, and passwords from log text or messages."""
    if not isinstance(text, str) or not text:
        return text

    sanitized = _AUTH_BEARER_PATTERN.sub("Bearer [REDACTED]", text)
    sanitized = _GROQ_KEY_PATTERN.sub("[REDACTED_GROQ_KEY]", sanitized)
    sanitized = _GEMINI_KEY_PATTERN.sub("[REDACTED_GEMINI_KEY]", sanitized)
    sanitized = _OPENROUTER_KEY_PATTERN.sub("[REDACTED_OPENROUTER_KEY]", sanitized)
    sanitized = _ANTHROPIC_KEY_PATTERN.sub("[REDACTED_ANTHROPIC_KEY]", sanitized)
    sanitized = _OPENAI_KEY_PATTERN.sub("[REDACTED_OPENAI_KEY]", sanitized)
    sanitized = _NEO4J_URI_PASSWORD_PATTERN.sub(r"\1***\3", sanitized)
    sanitized = _PG_URI_PASSWORD_PATTERN.sub(r"\1***\3", sanitized)
    sanitized = _PASSWORD_FIELD_PATTERN.sub(r"\1[REDACTED]\3", sanitized)
    return sanitized


def sanitize_log_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively sanitize string values in log dictionary."""
    sanitized: dict[str, Any] = {}
    for k, v in data.items():
        # Check sensitive keys
        k_lower = k.lower()
        if any(s in k_lower for s in ("password", "secret", "api_key", "authorization", "token")):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, str):
            sanitized[k] = redact_sensitive_text(v)
        elif isinstance(v, dict):
            sanitized[k] = sanitize_log_dict(v)
        elif isinstance(v, list):
            sanitized[k] = [
                sanitize_log_dict(item)
                if isinstance(item, dict)
                else redact_sensitive_text(item)
                if isinstance(item, str)
                else item
                for item in v
            ]
        else:
            sanitized[k] = v
    return sanitized


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as structured single-line JSON objects."""

    def __init__(self, environment: str = "development") -> None:
        super().__init__()
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        """Format a LogRecord as a single-line JSON string conforming to docs/OBSERVABILITY.md."""
        # 1. Base log structure
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        raw_message = record.getMessage()
        sanitized_message = redact_sensitive_text(raw_message)

        log_payload: dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": sanitized_message,
            "environment": getattr(record, "environment", self.environment),
        }

        # 2. Extract correlation / request ID if present
        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            log_payload["request_id"] = request_id

        # 3. Extract standard telemetry event fields if present
        for field_name in (
            "event",
            "method",
            "path",
            "status_code",
            "execution_duration_ms",
            "retrieval_mode",
            "routing_confidence",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "estimated_cost_usd",
            "citations_count",
            "warnings_count",
            "client_ip",
            "error_type",
        ):
            if hasattr(record, field_name):
                log_payload[field_name] = getattr(record, field_name)

        # 4. Include custom extra dict if passed
        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict):
            sanitized_extra = sanitize_log_dict(extra_data)
            for k, v in sanitized_extra.items():
                if k not in log_payload:
                    log_payload[k] = v

        # 5. Include sanitized exception information if present
        if record.exc_info and record.exc_text:
            log_payload["exception"] = redact_sensitive_text(record.exc_text)
        elif record.exc_info:
            log_payload["exception"] = redact_sensitive_text(self.formatException(record.exc_info))

        return json.dumps(log_payload, default=str)


def configure_structured_logging(log_level: str = "INFO", environment: str = "development") -> None:
    """Configure standard root logger with StructuredJsonFormatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Avoid duplicate handlers
    formatter = StructuredJsonFormatter(environment=environment)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Clear existing handlers to enforce uniform JSON logging
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
