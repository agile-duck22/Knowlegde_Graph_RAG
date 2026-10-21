"""Embeddings package: dense vector generation client and batching."""

from src.embeddings.client import (
    EmbeddingConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingGenerator,
    EmbeddingProviderError,
)

__all__ = [
    "EmbeddingConfigurationError",
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingGenerator",
    "EmbeddingProviderError",
]
