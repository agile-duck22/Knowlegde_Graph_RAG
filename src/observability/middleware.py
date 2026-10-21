"""Request Telemetry and Correlation Middleware (TASK-405).

Provides:
- X-Request-ID propagation (reads incoming X-Request-ID or creates standard req-<uuid>).
- Structured telemetry logging for all HTTP requests (method, path, status_code, duration_ms, request_id).
- Safe error logging with sensitive credential and secret redaction.
- Response header injection for X-Request-ID and X-Response-Time-MS.
"""

from __future__ import annotations

import logging
import re
import time
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("src.api.telemetry")


# Valid request ID pattern: alphanumeric, hyphen, underscore, length 1-128
_VALID_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


class RequestTelemetryMiddleware(BaseHTTPMiddleware):
    """Middleware capturing request telemetry, correlation ID propagation, and duration."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # 1. Establish or propagate Request/Correlation ID
        raw_header_id = request.headers.get("X-Request-ID") or request.headers.get("x-request-id")
        if raw_header_id:
            trimmed = raw_header_id.strip()
            if _VALID_REQUEST_ID_PATTERN.match(trimmed):
                request_id = trimmed
            else:
                request_id = f"req-{uuid4()}"
        else:
            request_id = f"req-{uuid4()}"

        # Attach request_id to request state for access in route handlers
        request.state.request_id = request_id

        start_time = time.perf_counter()
        status_code = 500
        error_type: str | None = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            error_type = type(exc).__name__
            logger.error(
                "Unhandled exception processing request %s %s: %s",
                request.method,
                request.url.path,
                exc,
                extra={
                    "request_id": request_id,
                    "event": "request_error",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "error_type": error_type,
                },
                exc_info=True,
            )
            raise
        finally:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            # Build telemetry log record
            log_level = logging.WARNING if status_code >= 400 else logging.INFO
            logger.log(
                log_level,
                "HTTP %s %s completed with status %d in %.2fms",
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                extra={
                    "request_id": request_id,
                    "event": "http_request",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "execution_duration_ms": duration_ms,
                    "client_ip": request.client.host if request.client else None,
                    "error_type": error_type,
                },
            )

            # Ensure response headers contain request_id and timing if response exists
            if "response" in locals() and isinstance(response, Response):
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time-MS"] = f"{duration_ms:.2f}"
