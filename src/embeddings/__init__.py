"""Embeddings package: dense vector generation client and batching."""

from src.embeddings.client import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingGenerator,
    EmbeddingProviderError,
)

__all__ = [
    "EmbeddingDimensionMismatchError",
    "EmbeddingError",
    "EmbeddingGenerator",
    "EmbeddingProviderError",
]
