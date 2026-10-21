"""FastAPI Application Entrypoint (TASK-404, TASK-405).

Configures application lifecycle, structured logging, telemetry middleware,
CORS, routes, and custom exception handlers.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import get_service_container
from src.api.routes import router as api_router
from src.config import Settings, get_settings
from src.observability.logging import configure_structured_logging, redact_sensitive_text
from src.observability.middleware import RequestTelemetryMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown events."""
    logger.info("Starting Knowledge Graph RAG API...")
    container = get_service_container()
    yield
    logger.info("Shutting down Knowledge Graph RAG API and cleaning up resources...")
    await container.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory creating configured FastAPI instance."""
    cfg = settings or get_settings()

    # Configure structured JSON logging
    configure_structured_logging(log_level=cfg.log_level, environment=cfg.environment)

    app = FastAPI(
        title="Knowledge Graph RAG API",
        description="Enterprise Knowledge Graph RAG with Dual-Store Hybrid Retrieval and Citation Guardrails.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Telemetry Middleware (outermost custom middleware)
    app.add_middleware(RequestTelemetryMiddleware)

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include Routes
    app.include_router(api_router)

    # Custom Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "Request validation error on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            extra={"request_id": request_id, "event": "validation_error"},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "UNPROCESSABLE_ENTITY",
                "message": "Input validation failed. Please inspect request payload schema.",
                "request_id": request_id,
                "details": {"errors": jsonable_encoder(exc.errors())},
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        if isinstance(exc.detail, dict):
            content = dict(exc.detail)
            if "request_id" not in content and request_id:
                content["request_id"] = request_id
            return JSONResponse(
                status_code=exc.status_code,
                content=content,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP_ERROR",
                "message": redact_sensitive_text(str(exc.detail)),
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.error(
            "Unhandled top-level exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            extra={"request_id": request_id, "event": "unhandled_exception"},
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred.",
                "request_id": request_id,
            },
        )

    return app


app = create_app()
