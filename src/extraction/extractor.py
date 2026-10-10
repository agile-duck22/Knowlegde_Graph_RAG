"""Structured LLM Extractor with retry handling and Pydantic validation.

Integrates Anthropic Claude with the domain prompt templates (``src.extraction.prompts``)
and validates model outputs against the domain schemas (``src.extraction.schemas``).

Security & Invariants
---------------------
- Untrusted inputs: Document text is never logged or concatenated outside delimiters.
- Secrets: API keys, auth tokens, and connection strings are NEVER logged.
- Bounded retry: Transient network/API errors and schema validation/JSON decode errors
  are retried with exponential backoff using tenacity. Non-retryable errors
  (e.g., authentication, permissions, bad requests) fail immediately.
- Dependency injection: Accepts an ``anthropic.Anthropic`` client interface for clean
  mocking and testing without live network or external API requirements.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import anthropic
from pydantic import ValidationError
from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from src.config import Settings
from src.extraction.prompts import build_extraction_messages, get_system_prompt
from src.extraction.schemas import ChunkExtractionResult
from src.ingest.chunker import TextChunk

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Base exception for extraction failures."""


class RetryExhaustedError(ExtractionError):
    """Raised when extraction fails after exceeding maximum retry attempts."""


class NonRetryableExtractionError(ExtractionError):
    """Raised when an unrecoverable error occurs (e.g., invalid auth, bad request)."""


# Errors that should be retried automatically with backoff
RETRYABLE_API_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    anthropic.ServiceUnavailableError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
)

# Errors that must fail immediately without retry
NON_RETRYABLE_API_EXCEPTIONS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.BadRequestError,
    anthropic.NotFoundError,
    anthropic.RequestTooLargeError,
    anthropic.UnprocessableEntityError,
)


class SchemaParseError(ExtractionError):
    """Internal error for JSON decode or Pydantic validation failures during an attempt."""


def _extract_json_string(text: str) -> str:
    """Extract raw JSON string from LLM response text, stripping markdown code fences if present."""
    trimmed = text.strip()
    # Check if text is wrapped in ```json ... ``` or ``` ... ```
    fence_match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", trimmed, flags=re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return trimmed


class ChunkExtractor:
    """Extracts entities and relationships from text chunks using Claude and validates with Pydantic."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        settings: Settings,
    ) -> None:
        self.client = client
        self.settings = settings

    def extract(self, chunk: TextChunk) -> ChunkExtractionResult:
        """Extract ontology-conforming entities and relationships from a text chunk.

        Parameters
        ----------
        chunk:
            The ``TextChunk`` to extract information from.

        Returns
        -------
        ChunkExtractionResult:
            Validated extraction result with entities, relationships, and provenance.

        Raises
        ------
        NonRetryableExtractionError:
            If a non-retryable API error occurs (auth, permissions, bad request).
        RetryExhaustedError:
            If transient API errors or schema validation errors exhaust all configured attempts.
        ExtractionError:
            For general extraction failures.
        """
        max_attempts = max(1, self.settings.extraction_max_retries + 1)

        # Build prompt messages
        doc_metadata = chunk.metadata or {}
        title = str(doc_metadata.get("title") or "")
        doc_type = str(doc_metadata.get("doc_type") or "document")
        source_uri = str(doc_metadata.get("source_uri") or "")

        try:
            messages_payload = build_extraction_messages(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_text=chunk.content,
                title=title,
                doc_type=doc_type,
                source_uri=source_uri,
            )
        except ValueError as exc:
            raise ExtractionError(f"Invalid chunk input for extraction: {exc}") from exc

        system_prompt = get_system_prompt()
        user_message = messages_payload[1]  # {"role": "user", "content": ...}

        def _is_retryable(exception: BaseException) -> bool:
            if isinstance(exception, RETRYABLE_API_EXCEPTIONS):
                return True
            if isinstance(exception, SchemaParseError):
                return True
            return False

        def _before_sleep_log(retry_state: RetryCallState) -> None:
            attempt_num = retry_state.attempt_number
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            exc_name = type(exc).__name__ if exc else "Unknown"
            logger.warning(
                "Extraction attempt %d failed for chunk_id=%s with %s. Retrying...",
                attempt_num,
                chunk.chunk_id,
                exc_name,
            )

        def _calculate_wait(retry_state: RetryCallState) -> float:
            """Calculate wait time, respecting Retry-After header on RateLimitError if present."""
            if retry_state.outcome and retry_state.outcome.failed:
                exc = retry_state.outcome.exception()
                if (
                    isinstance(exc, anthropic.RateLimitError)
                    and hasattr(exc, "response")
                    and exc.response is not None
                ):
                    retry_after = exc.response.headers.get("retry-after")
                    if retry_after:
                        try:
                            # If retry-after is float/int seconds
                            val = float(retry_after)
                            if val > 0:
                                return min(val, 60.0)  # capped at 60s
                        except (ValueError, TypeError):
                            pass
            # Fall back to exponential backoff with jitter
            exp_wait = wait_random_exponential(min=1, max=10)
            return float(exp_wait(retry_state))

        retryer = Retrying(
            stop=stop_after_attempt(max_attempts),
            wait=_calculate_wait,
            retry=retry_if_exception(_is_retryable),
            before_sleep=_before_sleep_log,
            reraise=True,
        )

        try:
            return retryer(
                self._execute_single_extraction,
                chunk=chunk,
                system_prompt=system_prompt,
                user_message=user_message,
            )
        except RetryError as exc:
            raise RetryExhaustedError(
                f"Extraction retries exhausted after {max_attempts} attempts for chunk_id={chunk.chunk_id}"
            ) from exc
        except NON_RETRYABLE_API_EXCEPTIONS as exc:
            raise NonRetryableExtractionError(
                f"Non-retryable Anthropic API error ({type(exc).__name__}) during extraction for chunk_id={chunk.chunk_id}"
            ) from exc
        except SchemaParseError as exc:
            # If retryer ran out of attempts and reraised SchemaParseError
            raise RetryExhaustedError(
                f"Extraction failed schema validation after {max_attempts} attempts for chunk_id={chunk.chunk_id}: {exc}"
            ) from exc
        except RETRYABLE_API_EXCEPTIONS as exc:
            # If retryer ran out of attempts and reraised retryable API exception
            raise RetryExhaustedError(
                f"Extraction failed due to API errors after {max_attempts} attempts for chunk_id={chunk.chunk_id}: {type(exc).__name__}"
            ) from exc

    def _execute_single_extraction(
        self,
        chunk: TextChunk,
        system_prompt: str,
        user_message: dict[str, Any],
    ) -> ChunkExtractionResult:
        """Execute a single call to Claude API and parse/validate response."""
        response = self.client.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.max_completion_tokens,
            system=system_prompt,
            messages=[user_message],  # type: ignore[list-item]
            timeout=self.settings.request_timeout_seconds,
        )

        # Extract text content from response
        if not response.content or not hasattr(response.content[0], "text"):
            raise SchemaParseError("Anthropic response contained no text content blocks")

        response_text = response.content[0].text
        json_str = _extract_json_string(response_text)

        try:
            raw_data = json.loads(json_str)
        except Exception as json_err:
            raise SchemaParseError(
                f"Failed to parse model response as JSON: {json_err}"
            ) from json_err

        if not isinstance(raw_data, dict):
            raise SchemaParseError(
                f"Model response JSON root is not an object (got {type(raw_data).__name__})"
            )

        # Validate against ChunkExtractionResult schema
        try:
            validated = ChunkExtractionResult.model_validate(raw_data)
        except ValidationError as val_err:
            raise SchemaParseError(
                f"Model response failed Pydantic schema validation: {val_err}"
            ) from val_err

        return validated
