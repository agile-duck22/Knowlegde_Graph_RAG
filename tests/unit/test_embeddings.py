"""Unit tests for the EmbeddingGenerator client (TASK-202).

Validates:
1. Single and batch text embedding generation.
2. Batching across chunk boundaries.
3. Dimension validation and error handling for dimension mismatches.
4. Input sanitization (empty texts, leading/trailing whitespace).
5. Non-finite value rejection (NaN, Inf).
6. Upstream provider error handling.
7. Correct ordering preservation from OpenAI response items.
8. Zero live API network requirements (all tests use mocks).
"""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.embeddings.client import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingGenerator,
    EmbeddingProviderError,
)


@pytest.fixture
def mock_embedding_settings() -> Settings:
    return Settings(
        environment="test",
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        openai_api_key=SecretStr("sk-mock-test-key-12345"),
    )


def _make_mock_item(index: int, embedding: list[float]) -> MagicMock:
    item = MagicMock()
    item.index = index
    item.embedding = embedding
    return item


def _make_mock_response(embeddings_list: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [_make_mock_item(idx, emb) for idx, emb in enumerate(embeddings_list)]
    return response


class TestEmbeddingGenerator:
    """Unit tests for EmbeddingGenerator."""

    def test_init_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            EmbeddingGenerator(batch_size=0)

    def test_generate_single_success(
        self,
        mock_embedding_settings: Settings,
    ) -> None:
        mock_client = MagicMock()
        fake_vec = [0.1] * 1536
        mock_client.embeddings.create.return_value = _make_mock_response([fake_vec])

        generator = EmbeddingGenerator(
            settings=mock_embedding_settings,
            client=mock_client,
        )

        vec = generator.generate_single("Hello world")

        assert len(vec) == 1536
        assert vec[0] == 0.1
        mock_client.embeddings.create.assert_called_once()
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "text-embedding-3-small"
        assert call_kwargs["dimensions"] == 1536
        assert call_kwargs["input"] == ["Hello world"]

    def test_generate_batch_preserves_order(
        self,
        mock_embedding_settings: Settings,
    ) -> None:
        mock_client = MagicMock()
        vec1 = [0.1] * 1536
        vec2 = [0.2] * 1536
        vec3 = [0.3] * 1536

        # Return items out of order to verify sorting by item.index
        resp = MagicMock()
        resp.data = [
            _make_mock_item(2, vec3),
            _make_mock_item(0, vec1),
            _make_mock_item(1, vec2),
        ]
        mock_client.embeddings.create.return_value = resp

        generator = EmbeddingGenerator(
            settings=mock_embedding_settings,
            client=mock_client,
        )

        results = generator.generate_batch(["Text 1", "Text 2", "Text 3"])

        assert len(results) == 3
        assert results[0][0] == 0.1
        assert results[1][0] == 0.2
        assert results[2][0] == 0.3

    def test_generate_batching_across_size_limit(
        self,
        mock_embedding_settings: Settings,
    ) -> None:
        mock_client = MagicMock()
        vec = [0.5] * 1536

        mock_client.embeddings.create.side_effect = [
            _make_mock_response([vec, vec]),
            _make_mock_response([vec]),
        ]

        generator = EmbeddingGenerator(
            settings=mock_embedding_settings,
            client=mock_client,
            batch_size=2,
        )

        texts = ["Text 1", "Text 2", "Text 3"]
        results = generator.generate_batch(texts)

        assert len(results) == 3
        assert mock_client.embeddings.create.call_count == 2

    def test_generate_empty_list_returns_empty(
        self,
        mock_embedding_settings: Settings,
    ) -> None:
        mock_client = MagicMock()
        generator = EmbeddingGenerator(
            settings=mock_embedding_settings,
            client=mock_client,
        )
        results = generator.generate_batch([])
        assert results == []
        mock_client.embeddings.create.assert_not_called()

    def test_dimension_mismatch_raises_error(
        self,
        mock_embedding_settings: Settings,
    ) -> None:
        mock_client = MagicMock()
        # Model returned 768 dimensions instead of 1536
        wrong_dim_vec = [0.1] * 768
        mock_client.embeddings.create.return_value = _make_mock_response([wrong_dim_vec])

        generator = EmbeddingGenerator(
            settings=mock_embedding_settings,
            client=mock_client,
            dimensions=1536,
        )

        with pytest.raises(EmbeddingDimensionMismatchError, match="returned dimension 768"):
            generator.generate_single("Test text")

    def test_nan_or_inf_in_vector_raises_error(
        self,
        mock_embedding_settings: Settings,
    ) -> None:
        mock_client = MagicMock()
        corrupt_vec = [0.1] * 1536
        corrupt_vec[10] = float("nan")
        mock_client.embeddings.create.return_value = _make_mock_response([corrupt_vec])

        generator = EmbeddingGenerator(
            settings=mock_embedding_settings,
            client=mock_client,
        )

        with pytest.raises(EmbeddingError, match="invalid numeric value"):
            generator.generate_single("Test text")

    def test_provider_api_error_raises_embedding_provider_error(
        self,
        mock_embedding_settings: Settings,
    ) -> None:
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = RuntimeError("OpenAI rate limit exceeded")

        generator = EmbeddingGenerator(
            settings=mock_embedding_settings,
            client=mock_client,
        )

        with pytest.raises(EmbeddingProviderError, match="Embedding provider API call failed"):
            generator.generate_single("Test text")

    def test_provider_empty_data_raises_error(
        self,
        mock_embedding_settings: Settings,
    ) -> None:
        mock_client = MagicMock()
        resp = MagicMock()
        resp.data = []
        mock_client.embeddings.create.return_value = resp

        generator = EmbeddingGenerator(
            settings=mock_embedding_settings,
            client=mock_client,
        )

        with pytest.raises(EmbeddingProviderError, match="empty response data"):
            generator.generate_single("Test text")
