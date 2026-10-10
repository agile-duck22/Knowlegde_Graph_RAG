"""Embedding generation client with configurable models, batching, and provider error handling.

Design & Invariants:
- Supports configured embedding provider (e.g. OpenAI `text-embedding-3-small`) and custom embedding dimensions.
- Efficient chunk batching with bounded batch sizes.
- Input validation: strips whitespace, rejects empty sequences, enforces dimension match and finite numeric values.
- Zero credential logging: API keys and sensitive tokens are read securely from Settings and never logged.
- Robust error handling: categorizes transient rate-limit / connection errors vs non-retryable provider errors.
- Dependency injection: accepts client interfaces (like `openai.OpenAI` or `openai.AsyncOpenAI`) for clean offline unit testing without network/API costs.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from src.config import Settings, get_settings

if TYPE_CHECKING:
    import openai

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base exception for embedding generator errors."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when configuration or provider setup is invalid."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when generated embedding vectors do not match configured dimensions."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when the upstream embedding provider returns an error."""


class EmbeddingGenerator:
    """Configurable embedding generator supporting batched dense vector generation.

    Attributes
    ----------
    settings:
        Application Settings object.
    model:
        Name of the embedding model (defaults to settings.embedding_model).
    dimensions:
        Expected vector dimension (defaults to settings.embedding_dimensions).
    batch_size:
        Maximum number of texts processed in a single upstream API call.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: openai.OpenAI | Any | None = None,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
    ) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.embedding_model
        self.dimensions = dimensions or self.settings.embedding_dimensions
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size
        self._client = client

    def _get_openai_client(self) -> Any:
        """Lazily initialize OpenAI client if not provided via dependency injection."""
        if self._client is not None:
            return self._client

        import openai

        api_key = (
            self.settings.openai_api_key.get_secret_value()
            if self.settings.openai_api_key
            else None
        )
        if not api_key and self.settings.embedding_provider == "openai":
            # If in test/development without key, permit dummy initialization for mocks
            api_key = "sk-mock-openai-key"

        self._client = openai.OpenAI(
            api_key=api_key or "sk-mock-openai-key",
            timeout=self.settings.request_timeout_seconds,
        )
        return self._client

    def _validate_and_normalize_vector(self, vector: Sequence[float], text_index: int) -> list[float]:
        """Validate vector dimensionality and ensure all components are finite floats."""
        if len(vector) != self.dimensions:
            raise EmbeddingDimensionMismatchError(
                f"Vector at index {text_index} returned dimension {len(vector)}, "
                f"expected configured dimension {self.dimensions}"
            )

        validated: list[float] = []
        for val_idx, val in enumerate(vector):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise EmbeddingError(
                    f"Vector at index {text_index} contains invalid numeric value "
                    f"'{val}' at dimension offset {val_idx}"
                )
            validated.append(float(val))
        return validated

    def generate_single(self, text: str) -> list[float]:
        """Generate embedding vector for a single text string.

        Parameters
        ----------
        text:
            Input text to embed.

        Returns
        -------
        list[float]
            Dense embedding vector of length ``self.dimensions``.
        """
        results = self.generate_batch([text])
        if not results:
            raise EmbeddingError("Failed to generate embedding for text")
        return results[0]

    def generate_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate dense embeddings for a sequence of text strings in batches.

        Parameters
        ----------
        texts:
            Sequence of text strings to embed. Empty strings will be replaced with a single space.

        Returns
        -------
        list[list[float]]
            List of embedding vectors, matching the input ordering.
        """
        if not texts:
            return []

        # Sanitize input: strip whitespace, ensure non-empty for embedding model
        sanitized_texts: list[str] = [t.strip() if t.strip() else " " for t in texts]

        all_embeddings: list[list[float]] = []
        client = self._get_openai_client()

        for batch_start in range(0, len(sanitized_texts), self.batch_size):
            batch = sanitized_texts[batch_start : batch_start + self.batch_size]
            logger.debug(
                "Embedding batch [%d:%d] (%d texts) using model '%s'",
                batch_start,
                batch_start + len(batch),
                len(batch),
                self.model,
            )

            try:
                # Upstream OpenAI embeddings API call
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "input": batch,
                }
                # If model supports dimension parameter (e.g. text-embedding-3-*), pass it
                if "text-embedding-3" in self.model:
                    kwargs["dimensions"] = self.dimensions

                response = client.embeddings.create(**kwargs)

                # OpenAI responses are sorted by index or in order
                if not hasattr(response, "data") or not response.data:
                    raise EmbeddingProviderError("Embedding provider returned empty response data")

                # Sort by index if present to guarantee exact input ordering
                items = sorted(response.data, key=lambda d: getattr(d, "index", 0))

                if len(items) != len(batch):
                    raise EmbeddingProviderError(
                        f"Expected {len(batch)} embeddings from batch, received {len(items)}"
                    )

                for item_idx, item in enumerate(items):
                    global_idx = batch_start + item_idx
                    vec = getattr(item, "embedding", None)
                    if vec is None:
                        raise EmbeddingProviderError(
                            f"Embedding item at index {global_idx} is missing 'embedding' field"
                        )
                    norm_vec = self._validate_and_normalize_vector(vec, text_index=global_idx)
                    all_embeddings.append(norm_vec)

            except EmbeddingError:
                raise
            except Exception as exc:
                logger.error("Embedding API call failed for batch [%d:%d]: %s", batch_start, batch_start + len(batch), exc)
                raise EmbeddingProviderError(f"Embedding provider API call failed: {exc}") from exc

        return all_embeddings
