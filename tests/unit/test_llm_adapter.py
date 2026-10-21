"""Unit tests for multi-provider LLM adapter, model allowlists, and provider routing.

Verifies:
- Base URLs and model selection for Groq, Gemini, and OpenRouter.
- Missing credentials and placeholder API key detection.
- Fail-closed model allowlist and OpenRouter :free suffix enforcement.
- Entity extraction, intent classification, and answer synthesis routing through adapter.
- Malformed structured responses and provider error encapsulation.
- Bounded retries and zero fallback to paid providers.
- Secret masking in settings for all free-tier API keys.
- Backward compatibility with injected test clients.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.extraction.extractor import ChunkExtractor, ExtractionError
from src.ingest.chunker import TextChunk
from src.llm import (
    GEMINI_BASE_URL,
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
    DisallowedModelError,
    MissingAPIKeyError,
    MultiProviderClient,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    UnsupportedProviderError,
    create_llm_client,
    is_model_allowed,
)
from src.routing.intent_classifier import QueryIntent, QueryIntentClassifier
from src.synthesis.normalizer import NormalizedChunkItem, NormalizedContext
from src.synthesis.synthesizer import (
    DraftAnswer,
    GroundedAnswerSynthesizer,
    ModelExecutionError,
    SynthesisRequest,
    compute_cost_usd,
)

# ---------------------------------------------------------------------------
# Model Allowlist and :free Enforcement Tests
# ---------------------------------------------------------------------------


def test_groq_allowed_models() -> None:
    assert is_model_allowed("groq", "openai/gpt-oss-120b") is True
    assert is_model_allowed("groq", "openai/gpt-oss-20b") is True
    assert is_model_allowed("groq", "llama-3.3-70b-versatile") is True
    assert is_model_allowed("groq", "llama-3.1-8b-instant") is True
    assert is_model_allowed("groq", "llama3-70b-8192") is False
    assert is_model_allowed("groq", "llama-3.1-70b-versatile") is False
    assert is_model_allowed("groq", "mixtral-8x7b-32768") is False
    assert is_model_allowed("groq", "gemma2-9b-it") is False
    assert is_model_allowed("groq", "claude-3-5-sonnet") is False
    assert is_model_allowed("groq", "gpt-4o") is False


def test_gemini_allowed_models() -> None:
    assert is_model_allowed("gemini", "gemini-3.1-flash-lite") is True
    assert is_model_allowed("gemini", "gemini-3.5-flash-lite") is True
    assert is_model_allowed("gemini", "gemini-2.5-flash") is True
    assert is_model_allowed("gemini", "gemini-2.5-flash-lite") is True
    assert is_model_allowed("gemini", "gemini-1.5-flash") is True
    assert is_model_allowed("gemini", "gemini-1.5-flash-8b") is True
    assert is_model_allowed("gemini", "gemini-1.5-pro") is True
    assert is_model_allowed("gemini", "gemini-2.0-flash-exp") is False
    assert is_model_allowed("gemini", "gemini-2.0-flash") is False
    assert is_model_allowed("gemini", "claude-3-opus") is False
    assert is_model_allowed("gemini", "gpt-4") is False


def test_openrouter_free_suffix_enforcement() -> None:
    # Explicitly free models with :free suffix must pass
    assert is_model_allowed("openrouter", "nvidia/nemotron-3.5-lightning:free") is True
    assert is_model_allowed("openrouter", "meta-llama/llama-3.3-70b-instruct:free") is True
    assert is_model_allowed("openrouter", "google/gemini-2.0-flash-exp:free") is True
    assert is_model_allowed("openrouter", "mistralai/mistral-7b-instruct:free") is True

    # Models without :free suffix must FAIL CLOSED
    assert is_model_allowed("openrouter", "meta-llama/llama-3.3-70b-instruct") is False
    assert is_model_allowed("openrouter", "google/gemini-pro") is False
    assert is_model_allowed("openrouter", "anthropic/claude-3.5-sonnet") is False
    assert is_model_allowed("openrouter", "openai/gpt-4o") is False


def test_disallowed_model_fails_closed_in_client() -> None:
    mock_openai = MagicMock()
    client = MultiProviderClient(
        provider="groq",
        client=mock_openai,
        timeout_seconds=30.0,
    )

    with pytest.raises(DisallowedModelError) as exc_info:
        client.messages.create(
            model="unauthorized-paid-model",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert "unauthorized-paid-model" in str(exc_info.value)
    assert "allowed" in str(exc_info.value).lower()
    # Ensure no network call was made
    mock_openai.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Base URL and Provider Factory Tests
# ---------------------------------------------------------------------------


def test_create_llm_client_groq_base_url() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=SecretStr("gsk_test_groq_key_12345"),
    )
    with patch("openai.OpenAI") as mock_openai_cls:
        client = create_llm_client(settings)
        assert isinstance(client, MultiProviderClient)
        assert client.provider == "groq"
        _ = client.get_underlying_client()
        mock_openai_cls.assert_called_once_with(
            base_url=GROQ_BASE_URL,
            api_key="gsk_test_groq_key_12345",
            timeout=30.0,
            default_headers=None,
        )


def test_create_llm_client_gemini_base_url() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        gemini_api_key=SecretStr("AIzaSy_test_gemini_key_12345"),
    )
    with patch("openai.OpenAI") as mock_openai_cls:
        client = create_llm_client(settings)
        assert isinstance(client, MultiProviderClient)
        assert client.provider == "gemini"
        _ = client.get_underlying_client()
        mock_openai_cls.assert_called_once_with(
            base_url=GEMINI_BASE_URL,
            api_key="AIzaSy_test_gemini_key_12345",
            timeout=30.0,
            default_headers=None,
        )


def test_create_llm_client_openrouter_base_url() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openrouter",
        openrouter_api_key=SecretStr("sk-or-v1-test_openrouter_key"),
    )
    with patch("openai.OpenAI") as mock_openai_cls:
        client = create_llm_client(settings)
        assert isinstance(client, MultiProviderClient)
        assert client.provider == "openrouter"
        _ = client.get_underlying_client()
        mock_openai_cls.assert_called_once_with(
            base_url=OPENROUTER_BASE_URL,
            api_key="sk-or-v1-test_openrouter_key",
            timeout=30.0,
            default_headers={
                "HTTP-Referer": "https://github.com/kgr",
                "X-Title": "Knowledge Graph RAG",
            },
        )


def test_missing_or_placeholder_api_key_raises_error() -> None:
    # 1. Missing key
    settings_missing = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=None,
    )
    client_missing = create_llm_client(settings_missing)
    with pytest.raises(MissingAPIKeyError) as exc_info:
        client_missing.get_underlying_client()
    assert "not configured" in str(exc_info.value).lower()

    # 2. Placeholder key
    settings_placeholder = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=SecretStr("your_groq_api_key_here"),
    )
    client_placeholder = create_llm_client(settings_placeholder)
    with pytest.raises(MissingAPIKeyError) as exc_info:
        client_placeholder.get_underlying_client()
    assert "placeholder" in str(exc_info.value).lower()


def test_unsupported_provider_raises_error() -> None:
    with pytest.raises(UnsupportedProviderError) as exc_info:
        MultiProviderClient(provider="unsupported_cloud")
    assert "Unsupported LLM provider" in str(exc_info.value)


# ---------------------------------------------------------------------------
# MultiProviderClient Message Translation & Error Encapsulation Tests
# ---------------------------------------------------------------------------


def test_multiprovider_client_successful_translation() -> None:
    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"greeting": "hello world"}'
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 120
    mock_completion.usage.completion_tokens = 40
    mock_openai.chat.completions.create.return_value = mock_completion

    client = MultiProviderClient(
        provider="groq",
        client=mock_openai,
        timeout_seconds=15.0,
    )

    resp = client.messages.create(
        model="llama-3.3-70b-versatile",
        system="You are a JSON assistant.",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=256,
    )

    # Verify response interface mimics Anthropic
    assert len(resp.content) == 1
    assert resp.content[0].text == '{"greeting": "hello world"}'
    assert resp.usage.input_tokens == 120
    assert resp.usage.output_tokens == 40

    # Verify underlying OpenAI call was correctly constructed
    mock_openai.chat.completions.create.assert_called_once()
    kwargs = mock_openai.chat.completions.create.call_args[1]
    assert kwargs["model"] == "llama-3.3-70b-versatile"
    assert kwargs["messages"][0] == {"role": "system", "content": "You are a JSON assistant."}
    assert kwargs["messages"][1] == {"role": "user", "content": "Say hello"}
    assert kwargs["max_tokens"] == 256


def test_multiprovider_client_empty_choices() -> None:
    mock_openai = MagicMock()
    mock_completion = MagicMock()
    empty_choices: list[Any] = []
    mock_completion.choices = empty_choices
    mock_openai.chat.completions.create.return_value = mock_completion

    client = MultiProviderClient(
        provider="gemini",
        client=mock_openai,
        timeout_seconds=10.0,
    )

    resp = client.messages.create(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": "test"}],
    )
    assert len(resp.content) == 0
    assert resp.text == ""


def test_multiprovider_client_maps_rate_limit_and_auth_errors() -> None:
    import openai

    mock_openai = MagicMock()
    client = MultiProviderClient(provider="groq", client=mock_openai)

    # Test 429 Rate Limit mapping
    mock_openai.chat.completions.create.side_effect = openai.RateLimitError(
        message="Rate limit exceeded",
        response=MagicMock(status_code=429),
        body=None,
    )
    with pytest.raises(ProviderRateLimitError):
        client.messages.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
        )

    # Test 401 Authentication Error mapping
    mock_openai.chat.completions.create.side_effect = openai.AuthenticationError(
        message="Invalid API Key",
        response=MagicMock(status_code=401),
        body=None,
    )
    with pytest.raises(ProviderAuthenticationError):
        client.messages.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
        )

    # Test Timeout mapping
    mock_openai.chat.completions.create.side_effect = openai.APITimeoutError(request=MagicMock())
    with pytest.raises(ProviderTimeoutError):
        client.messages.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "hi"}],
        )


# ---------------------------------------------------------------------------
# Extraction, Intent Classification, and Synthesis Adapter Integration Tests
# ---------------------------------------------------------------------------


def test_extraction_through_groq_adapter() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=SecretStr("gsk_valid_test_key_12345"),
        llm_model="llama-3.3-70b-versatile",
        extraction_max_retries=1,
    )

    mock_openai = MagicMock()
    mock_choice = MagicMock()
    extraction_json = json.dumps(
        {
            "chunk_id": "00000000-0000-0000-0000-000000000001",
            "document_id": "doc-arch-001",
            "extraction_model": "llama-3.3-70b-versatile",
            "entities": [
                {
                    "canonical_id": "service:order-service",
                    "name": "Order Service",
                    "entity_type": "Service",
                    "description": "Handles orders",
                    "aliases": [],
                    "confidence": 0.95,
                    "provenance": {
                        "chunk_id": "00000000-0000-0000-0000-000000000001",
                        "document_id": "doc-arch-001",
                    },
                }
            ],
            "relationships": [],
        }
    )
    mock_choice.message.content = f"```json\n{extraction_json}\n```"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 80
    mock_completion.usage.completion_tokens = 30
    mock_openai.chat.completions.create.return_value = mock_completion

    adapter_client = MultiProviderClient(
        provider="groq",
        client=mock_openai,
        timeout_seconds=10.0,
    )

    extractor = ChunkExtractor(client=adapter_client, settings=settings)
    chunk = TextChunk(
        chunk_id="00000000-0000-0000-0000-000000000001",
        document_id="doc-arch-001",
        chunk_index=0,
        content="Order Service manages checkout.",
        char_start=0,
        char_end=31,
    )

    result = extractor.extract(chunk)
    assert len(result.entities) == 1
    assert result.entities[0].name == "Order Service"
    assert result.entities[0].entity_type == "Service"


def test_extraction_malformed_json_raises_extraction_error() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=SecretStr("gsk_valid_test_key_12345"),
        llm_model="llama-3.3-70b-versatile",
        extraction_max_retries=1,
    )

    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Not a JSON document, sorry!"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_completion

    adapter_client = MultiProviderClient(
        provider="groq",
        client=mock_openai,
        timeout_seconds=10.0,
    )

    extractor = ChunkExtractor(client=adapter_client, settings=settings)
    chunk = TextChunk(
        chunk_id="chunk-test-1",
        document_id="doc-test-1",
        chunk_index=0,
        content="Order Service manages checkout.",
        char_start=0,
        char_end=31,
    )

    with pytest.raises(ExtractionError):
        extractor.extract(chunk)


def test_intent_classification_through_groq_adapter() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=SecretStr("gsk_valid_test_key_12345"),
        routing_model="llama-3.1-8b-instant",
    )

    mock_openai = MagicMock()
    mock_choice = MagicMock()
    intent_payload = {
        "mode": "GRAPH_ONLY",
        "confidence": 0.95,
        "reasoning": "Dependency traversal query",
        "extracted_entities": ["Order Service"],
    }
    mock_choice.message.content = json.dumps(intent_payload)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_completion

    adapter_client = MultiProviderClient(
        provider="groq",
        client=mock_openai,
        timeout_seconds=10.0,
    )

    classifier = QueryIntentClassifier(client=adapter_client, settings=settings)
    classification = classifier.classify("Which services depend on Order Service?")

    assert classification.mode == QueryIntent.GRAPH_ONLY
    assert classification.confidence == 0.95
    assert classification.extracted_entities == ["Order Service"]


def test_intent_classifier_falls_back_to_offline_heuristic_when_key_missing() -> None:
    # When free-tier key is not set, classifier should gracefully fall back to regex heuristic
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=None,
    )

    classifier = QueryIntentClassifier(settings=settings)
    # Query with dependency keywords
    result = classifier.classify("Which services depend on Checkout?")
    assert result.mode == QueryIntent.GRAPH_ONLY
    assert result.method.value == "heuristic"


def test_synthesis_through_groq_adapter_and_zero_cost() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=SecretStr("gsk_valid_test_key_12345"),
        llm_model="llama-3.3-70b-versatile",
        synthesis_model="llama-3.3-70b-versatile",
    )

    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "The Checkout Service is maintained by Checkout Team [c1111111-1111-1111-1111-111111111111]."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 200
    mock_completion.usage.completion_tokens = 50
    mock_openai.chat.completions.create.return_value = mock_completion

    adapter_client = MultiProviderClient(
        provider="groq",
        client=mock_openai,
        timeout_seconds=15.0,
    )

    synthesizer = GroundedAnswerSynthesizer(client=adapter_client, settings=settings)
    chunk = NormalizedChunkItem(
        chunk_id="c1111111-1111-1111-1111-111111111111",
        document_id="doc-1",
        content="Checkout Service is maintained by Checkout Team.",
        similarity_score=0.9,
        provenance_source="vector",
        token_count=10,
    )
    context = NormalizedContext(
        formatted_context=chunk.to_formatted_block(),
        chunks=[chunk],
        graph_triples=[],
        graph_entities=[],
        total_tokens=10,
        max_budget_tokens=1000,
        candidate_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        all_entity_ids=[],
        is_truncated=False,
        truncated_chunks_count=0,
        truncated_triples_count=0,
    )

    answer: DraftAnswer = synthesizer.synthesize(
        SynthesisRequest(question="Who maintains Checkout?", context=context)
    )

    assert answer.cited_chunk_ids == ["c1111111-1111-1111-1111-111111111111"]
    assert answer.model_name == "llama-3.3-70b-versatile"
    # Cost for free-tier model must be exactly 0.0
    assert answer.token_usage.estimated_cost_usd == 0.0
    assert compute_cost_usd("llama-3.3-70b-versatile", 200, 50) == 0.0
    assert compute_cost_usd("openai/gpt-oss-120b", 200, 50) == 0.0
    assert compute_cost_usd("openai/gpt-oss-20b", 200, 50) == 0.0
    assert compute_cost_usd("gemini-2.5-flash", 200, 50) == 0.0
    assert compute_cost_usd("meta-llama/llama-3.3-70b-instruct:free", 200, 50) == 0.0


def test_no_silent_fallback_to_paid_provider_on_failure() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=SecretStr("gsk_valid_test_key_12345"),
        anthropic_api_key=SecretStr("sk-ant-test-anthropic-key"),
        llm_model="llama-3.3-70b-versatile",
        extraction_max_retries=1,
    )

    mock_openai = MagicMock()
    import openai

    # Simulate persistent rate limit failure on Groq
    mock_openai.chat.completions.create.side_effect = openai.RateLimitError(
        message="Groq TPM limit reached",
        response=MagicMock(status_code=429),
        body=None,
    )

    adapter_client = MultiProviderClient(provider="groq", client=mock_openai)
    synthesizer = GroundedAnswerSynthesizer(client=adapter_client, settings=settings)

    chunk = NormalizedChunkItem(
        chunk_id="c1",
        document_id="doc-1",
        content="Test content",
        similarity_score=0.9,
        provenance_source="vector",
        token_count=5,
    )
    context = NormalizedContext(
        formatted_context=chunk.to_formatted_block(),
        chunks=[chunk],
        graph_triples=[],
        graph_entities=[],
        total_tokens=5,
        max_budget_tokens=1000,
        candidate_chunk_ids=["c1"],
        all_entity_ids=[],
        is_truncated=False,
        truncated_chunks_count=0,
        truncated_triples_count=0,
    )

    # Must raise ModelExecutionError and NOT call Anthropic
    with patch("anthropic.Anthropic") as mock_anthropic:
        with pytest.raises(ModelExecutionError):
            synthesizer.synthesize(SynthesisRequest(question="Test question?", context=context))
        mock_anthropic.assert_not_called()


# ---------------------------------------------------------------------------
# Secret Masking Tests for Free-Tier Provider Keys
# ---------------------------------------------------------------------------


def test_free_tier_keys_masked_in_settings_repr() -> None:
    settings = Settings(
        _env_file=None,
        gemini_api_key=SecretStr("super_secret_gemini_key_12345"),
        groq_api_key=SecretStr("super_secret_groq_key_67890"),
        openrouter_api_key=SecretStr("super_secret_openrouter_key_abcde"),
    )
    repr_str = repr(settings)

    assert "super_secret_gemini_key_12345" not in repr_str
    assert "super_secret_groq_key_67890" not in repr_str
    assert "super_secret_openrouter_key_abcde" not in repr_str
    assert "**********" in repr_str


def test_extraction_through_groq_gpt_oss_120b() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=SecretStr("gsk_valid_test_key_12345"),
        extraction_model="openai/gpt-oss-120b",
        extraction_max_retries=1,
    )

    mock_openai = MagicMock()
    mock_choice = MagicMock()
    extraction_json = json.dumps(
        {
            "chunk_id": "00000000-0000-0000-0000-000000000002",
            "document_id": "doc-arch-002",
            "extraction_model": "openai/gpt-oss-120b",
            "entities": [
                {
                    "canonical_id": "service:payment-gateway",
                    "name": "Payment Gateway",
                    "entity_type": "Service",
                    "description": "Processes credit card payments",
                    "aliases": [],
                    "confidence": 0.98,
                    "provenance": {
                        "chunk_id": "00000000-0000-0000-0000-000000000002",
                        "document_id": "doc-arch-002",
                    },
                }
            ],
            "relationships": [],
        }
    )
    mock_choice.message.content = f"```json\n{extraction_json}\n```"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_completion

    adapter_client = MultiProviderClient(provider="groq", client=mock_openai)
    extractor = ChunkExtractor(client=adapter_client, settings=settings)
    chunk = TextChunk(
        chunk_id="00000000-0000-0000-0000-000000000002",
        document_id="doc-arch-002",
        chunk_index=0,
        content="Payment Gateway handles billing.",
        char_start=0,
        char_end=33,
    )

    result = extractor.extract(chunk)
    assert len(result.entities) == 1
    assert result.entities[0].name == "Payment Gateway"
    assert result.extraction_model == "openai/gpt-oss-120b"


def test_intent_classification_through_groq_gpt_oss_20b() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="groq",
        groq_api_key=SecretStr("gsk_valid_test_key_12345"),
        routing_model="openai/gpt-oss-20b",
    )

    mock_openai = MagicMock()
    mock_choice = MagicMock()
    intent_payload: dict[str, Any] = {
        "mode": "VECTOR_ONLY",
        "confidence": 0.92,
        "reasoning": "Semantic search for policy documentation",
        "extracted_entities": [],
    }
    mock_choice.message.content = json.dumps(intent_payload)
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_completion

    adapter_client = MultiProviderClient(provider="groq", client=mock_openai)
    classifier = QueryIntentClassifier(client=adapter_client, settings=settings)
    classification = classifier.classify("What is the data retention policy?")

    assert classification.mode == QueryIntent.VECTOR_ONLY
    assert classification.confidence == 0.92


def test_deprecated_models_rejected_fail_closed() -> None:
    mock_openai = MagicMock()
    client_groq = MultiProviderClient(provider="groq", client=mock_openai)
    client_gemini = MultiProviderClient(provider="gemini", client=mock_openai)

    # Groq deprecated models must raise DisallowedModelError
    with pytest.raises(DisallowedModelError):
        client_groq.messages.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": "hi"}],
        )

    with pytest.raises(DisallowedModelError):
        client_groq.messages.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": "hi"}],
        )

    # Gemini shut-down 2.0 models must raise DisallowedModelError
    with pytest.raises(DisallowedModelError):
        client_gemini.messages.create(
            model="gemini-2.0-flash-exp",
            messages=[{"role": "user", "content": "hi"}],
        )

    with pytest.raises(DisallowedModelError):
        client_gemini.messages.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": "hi"}],
        )

    mock_openai.chat.completions.create.assert_not_called()


def test_target_component_provider_and_model_resolution() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="anthropic",
        extraction_provider="gemini",
        extraction_model="gemini-3.1-flash-lite",
        routing_provider="groq",
        routing_model="openai/gpt-oss-20b",
        synthesis_provider="gemini",
        synthesis_model="gemini-3.1-flash-lite",
        gemini_api_key=SecretStr("AIzaSy_test_key_12345"),
        groq_api_key=SecretStr("gsk_test_key_12345"),
    )

    assert settings.get_extraction_provider() == "gemini"
    assert settings.get_extraction_model() == "gemini-3.1-flash-lite"
    assert settings.get_routing_provider() == "groq"
    assert settings.get_routing_model() == "openai/gpt-oss-20b"
    assert settings.get_synthesis_provider() == "gemini"
    assert settings.get_synthesis_model() == "gemini-3.1-flash-lite"


def test_target_configuration_end_to_end_mocked() -> None:
    settings = Settings(
        _env_file=None,
        extraction_provider="gemini",
        extraction_model="gemini-3.1-flash-lite",
        routing_provider="groq",
        routing_model="openai/gpt-oss-20b",
        synthesis_provider="gemini",
        synthesis_model="gemini-3.1-flash-lite",
        gemini_api_key=SecretStr("AIzaSy_test_key_12345"),
        groq_api_key=SecretStr("gsk_test_key_12345"),
        extraction_max_retries=1,
    )

    # 1. Extraction with Gemini
    mock_gemini_openai = MagicMock()
    mock_choice_ext = MagicMock()
    extraction_json = json.dumps(
        {
            "chunk_id": "00000000-0000-0000-0000-000000000003",
            "document_id": "doc-test-003",
            "extraction_model": "gemini-3.1-flash-lite",
            "entities": [
                {
                    "canonical_id": "service:auth-service",
                    "name": "Auth Service",
                    "entity_type": "Service",
                    "description": "Handles authentication",
                    "aliases": [],
                    "confidence": 0.99,
                    "provenance": {
                        "chunk_id": "00000000-0000-0000-0000-000000000003",
                        "document_id": "doc-test-003",
                    },
                }
            ],
            "relationships": [],
        }
    )
    mock_choice_ext.message.content = f"```json\n{extraction_json}\n```"
    mock_comp_ext = MagicMock()
    mock_comp_ext.choices = [mock_choice_ext]
    mock_gemini_openai.chat.completions.create.return_value = mock_comp_ext

    gemini_client = MultiProviderClient(provider="gemini", client=mock_gemini_openai)
    extractor = ChunkExtractor(client=gemini_client, settings=settings)
    chunk = TextChunk(
        chunk_id="00000000-0000-0000-0000-000000000003",
        document_id="doc-test-003",
        chunk_index=0,
        content="Auth Service manages SSO logins.",
        char_start=0,
        char_end=32,
    )
    res_ext = extractor.extract(chunk)
    assert len(res_ext.entities) == 1
    assert res_ext.entities[0].name == "Auth Service"
    assert res_ext.extraction_model == "gemini-3.1-flash-lite"

    # 2. Routing with Groq openai/gpt-oss-20b
    mock_groq_openai = MagicMock()
    mock_choice_route = MagicMock()
    mock_choice_route.message.content = json.dumps(
        {
            "mode": "GRAPH_ONLY",
            "confidence": 0.96,
            "reasoning": "Auth dependencies lookup",
            "extracted_entities": ["Auth Service"],
        }
    )
    mock_comp_route = MagicMock()
    mock_comp_route.choices = [mock_choice_route]
    mock_groq_openai.chat.completions.create.return_value = mock_comp_route

    groq_client = MultiProviderClient(provider="groq", client=mock_groq_openai)
    classifier = QueryIntentClassifier(client=groq_client, settings=settings)
    res_route = classifier.classify("What depends on Auth Service?")
    assert res_route.mode == QueryIntent.GRAPH_ONLY
    assert res_route.confidence == 0.96

    # 3. Synthesis with Gemini gemini-3.1-flash-lite
    mock_gemini_synth = MagicMock()
    mock_choice_synth = MagicMock()
    mock_choice_synth.message.content = (
        "Auth Service is active [00000000-0000-0000-0000-000000000003]."
    )
    mock_comp_synth = MagicMock()
    mock_comp_synth.choices = [mock_choice_synth]
    mock_gemini_synth.chat.completions.create.return_value = mock_comp_synth

    gemini_synth_client = MultiProviderClient(provider="gemini", client=mock_gemini_synth)
    synthesizer = GroundedAnswerSynthesizer(client=gemini_synth_client, settings=settings)
    norm_chunk = NormalizedChunkItem(
        chunk_id="00000000-0000-0000-0000-000000000003",
        document_id="doc-test-003",
        content="Auth Service manages SSO logins.",
        similarity_score=0.95,
        provenance_source="vector",
        token_count=10,
    )
    context = NormalizedContext(
        formatted_context=norm_chunk.to_formatted_block(),
        chunks=[norm_chunk],
        graph_triples=[],
        graph_entities=[],
        total_tokens=10,
        max_budget_tokens=1000,
        candidate_chunk_ids=["00000000-0000-0000-0000-000000000003"],
        all_entity_ids=[],
        is_truncated=False,
        truncated_chunks_count=0,
        truncated_triples_count=0,
    )
    answer = synthesizer.synthesize(
        SynthesisRequest(question="What is Auth Service?", context=context)
    )
    assert answer.cited_chunk_ids == ["00000000-0000-0000-0000-000000000003"]
    assert answer.model_name == "gemini-3.1-flash-lite"
    assert answer.token_usage.estimated_cost_usd == 0.0


def test_openrouter_nemotron_free_model_synthesis() -> None:
    settings = Settings(
        _env_file=None,
        synthesis_provider="openrouter",
        synthesis_model="nvidia/nemotron-3.5-lightning:free",
        openrouter_api_key=SecretStr("sk-or-v1-test_key_12345"),
    )

    mock_openai = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Nemotron answer [c-123]."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_completion

    openrouter_client = MultiProviderClient(provider="openrouter", client=mock_openai)
    synthesizer = GroundedAnswerSynthesizer(client=openrouter_client, settings=settings)

    norm_chunk = NormalizedChunkItem(
        chunk_id="c-123",
        document_id="doc-1",
        content="Nemotron content",
        similarity_score=0.9,
        provenance_source="vector",
        token_count=5,
    )
    context = NormalizedContext(
        formatted_context=norm_chunk.to_formatted_block(),
        chunks=[norm_chunk],
        graph_triples=[],
        graph_entities=[],
        total_tokens=5,
        max_budget_tokens=1000,
        candidate_chunk_ids=["c-123"],
        all_entity_ids=[],
        is_truncated=False,
        truncated_chunks_count=0,
        truncated_triples_count=0,
    )
    answer = synthesizer.synthesize(SynthesisRequest(question="Test?", context=context))
    assert answer.model_name == "nvidia/nemotron-3.5-lightning:free"
    assert answer.token_usage.estimated_cost_usd == 0.0


def test_embedding_generation_fails_closed_by_default() -> None:
    from src.embeddings.client import EmbeddingConfigurationError, EmbeddingGenerator

    # Settings with allow_paid_embeddings=False (default)
    settings = Settings(
        _env_file=None,
        embedding_provider="openai",
        openai_api_key=SecretStr("sk-real-looking-key-12345"),
        allow_paid_embeddings=False,
    )

    generator = EmbeddingGenerator(settings=settings)
    with pytest.raises(EmbeddingConfigurationError) as exc_info:
        generator.generate_single("test text")

    assert "allow_paid_embeddings=False" in str(exc_info.value)
    assert "zero-cost" in str(exc_info.value).lower()
