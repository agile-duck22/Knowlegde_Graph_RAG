"""Unit tests for Query Intent Classifier (TASK-303).

Verifies deterministic heuristic routing, LLM-assisted classification with mocks,
confidence threshold enforcement, malformed response handling, out-of-scope filtering,
and explicit manual overrides — strictly without live Anthropic API calls.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import anthropic
import pytest
from pydantic import ValidationError

from src.config import Settings
from src.retrieval.cypher_templates import (
    TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
    TEMPLATE_MULTI_HOP_DEPENDENCY,
    TEMPLATE_TEAM_OWNERSHIP_SERVICES,
    TEMPLATE_VULNERABILITY_BLAST_RADIUS,
)
from src.routing.intent_classifier import (
    ClassificationMethod,
    IntentClassificationResult,
    QueryIntent,
    QueryIntentClassifier,
    classify_query_heuristic,
)


def _make_mock_llm_response(payload: dict[str, Any]) -> MagicMock:
    """Helper to create a mocked Anthropic message response returning JSON."""
    mock_resp = MagicMock()
    mock_content = MagicMock()
    mock_content.text = json.dumps(payload)
    mock_resp.content = [mock_content]
    mock_resp.usage.input_tokens = 50
    mock_resp.usage.output_tokens = 30
    return mock_resp


# ===========================================================================
# 1. Deterministic Heuristic Tests (Offline Fast-Path)
# ===========================================================================


class TestHeuristicClassifier:
    """Tests for the offline rule-based heuristic classifier."""

    def test_empty_and_whitespace_query(self) -> None:
        res = classify_query_heuristic("")
        assert res.mode == QueryIntent.OUT_OF_SCOPE
        assert res.confidence == 1.0
        assert res.method == ClassificationMethod.HEURISTIC

        res_space = classify_query_heuristic("   \n\t  ")
        assert res_space.mode == QueryIntent.OUT_OF_SCOPE

    def test_short_garbage_query(self) -> None:
        res = classify_query_heuristic("ab")
        assert res.mode == QueryIntent.OUT_OF_SCOPE

    def test_out_of_scope_queries(self) -> None:
        out_of_scope_samples = [
            "What is the weather forecast for tomorrow in Tokyo?",
            "How do I cook a chocolate cake recipe?",
            "Write a poem about sunshine and rainbows.",
            "Who won the 2022 World Cup finals?",
            "Hello, how are you doing today?",
            "What is the capital of France?",
        ]
        for query in out_of_scope_samples:
            res = classify_query_heuristic(query)
            assert res.mode == QueryIntent.OUT_OF_SCOPE, f"Failed for: {query}"
            assert res.confidence >= 0.90
            assert res.suggested_template_id is None

    def test_graph_only_queries(self) -> None:
        graph_queries = [
            (
                "Which services depend on the payment-gateway service?",
                TEMPLATE_MULTI_HOP_DEPENDENCY,
            ),
            ("What are the downstream dependencies of checkout?", TEMPLATE_MULTI_HOP_DEPENDENCY),
            (
                "List all services owned by Team Checkout Engineering",
                TEMPLATE_TEAM_OWNERSHIP_SERVICES,
            ),
            (
                "What is the blast radius of vulnerability CVE-2024-21626?",
                TEMPLATE_VULNERABILITY_BLAST_RADIUS,
            ),
            (
                "Which services read from or write to the orders database?",
                TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
            ),
        ]
        for query, expected_template in graph_queries:
            res = classify_query_heuristic(query)
            assert res.mode == QueryIntent.GRAPH_ONLY, f"Expected GRAPH_ONLY for: {query}"
            assert res.confidence >= 0.85
            assert res.suggested_template_id == expected_template

    def test_vector_only_queries(self) -> None:
        vector_queries = [
            "What is the retry policy described in the payment gateway spec?",
            "Explain the difference between optimistic and pessimistic locking in the docs.",
            "What is the timeout configuration and SLA for orders?",
            "How to configure setup guide for the local development environment?",
        ]
        for query in vector_queries:
            res = classify_query_heuristic(query)
            assert res.mode == QueryIntent.VECTOR_ONLY, f"Expected VECTOR_ONLY for: {query}"
            assert res.confidence >= 0.80
            assert res.suggested_template_id is None

    def test_hybrid_queries(self) -> None:
        hybrid_queries = [
            "Describe the checkout architecture and which teams are responsible for it",
            "Give a full summary of the payment architecture and dependencies",
            "Which services depend on checkout and what is its retry policy?",  # touches both graph & vector
        ]
        for query in hybrid_queries:
            res = classify_query_heuristic(query)
            assert res.mode == QueryIntent.HYBRID, f"Expected HYBRID for: {query}"
            assert res.confidence >= 0.75

    def test_ambiguous_query_fallback(self) -> None:
        # A vague enterprise query with no specific trigger words falls back to HYBRID
        res = classify_query_heuristic(
            "Tell me something about service:checkout-service", confidence_threshold=0.65
        )
        assert res.mode == QueryIntent.HYBRID
        assert res.confidence < 0.65
        assert "service:checkout-service" in res.extracted_entities


# ===========================================================================
# 2. LLM-Assisted Intent Classifier Unit Tests (With Mocks)
# ===========================================================================


class TestQueryIntentClassifierLLM:
    """Tests for QueryIntentClassifier using mocked Anthropic client."""

    @pytest.fixture
    def mock_anthropic_client(self) -> MagicMock:
        return MagicMock(spec=anthropic.Anthropic)

    @pytest.fixture
    def classifier(self, mock_anthropic_client: MagicMock) -> QueryIntentClassifier:
        settings = Settings(
            router_confidence_threshold=0.65,
            routing_model="claude-3-haiku-20240307",
        )
        return QueryIntentClassifier(
            settings=settings,
            anthropic_client=mock_anthropic_client,
            use_llm=True,
        )

    def test_llm_graph_only_classification(
        self, classifier: QueryIntentClassifier, mock_anthropic_client: MagicMock
    ) -> None:
        mock_payload = {
            "mode": "GRAPH_ONLY",
            "confidence": 0.95,
            "reasoning": "Query asks for multi-hop service call graph.",
            "suggested_template_id": "T-02",
            "extracted_entities": ["service:checkout", "service:payment"],
        }
        mock_anthropic_client.messages.create.return_value = _make_mock_llm_response(mock_payload)

        query = "Show me the call path between checkout and payment"
        result = classifier.classify(query)

        assert result.mode == QueryIntent.GRAPH_ONLY
        assert result.confidence == 0.95
        assert result.suggested_template_id == "T-02"
        assert result.extracted_entities == ["service:checkout", "service:payment"]
        assert result.method == ClassificationMethod.LLM

        # Ensure XML wrapping in prompt for injection defense
        call_args = mock_anthropic_client.messages.create.call_args[1]
        assert "<user_query>" in call_args["messages"][0]["content"]
        assert query in call_args["messages"][0]["content"]

    def test_llm_vector_only_classification(
        self, classifier: QueryIntentClassifier, mock_anthropic_client: MagicMock
    ) -> None:
        mock_payload = {
            "mode": "VECTOR_ONLY",
            "confidence": 0.90,
            "reasoning": "Query asks for retry policy documentation text.",
            "suggested_template_id": None,
            "extracted_entities": ["service:payment-gateway"],
        }
        mock_anthropic_client.messages.create.return_value = _make_mock_llm_response(mock_payload)

        result = classifier.classify("What is the retry policy for payment gateway?")
        assert result.mode == QueryIntent.VECTOR_ONLY
        assert result.confidence == 0.90
        assert result.suggested_template_id is None
        assert result.method == ClassificationMethod.LLM

    def test_llm_hybrid_classification(
        self, classifier: QueryIntentClassifier, mock_anthropic_client: MagicMock
    ) -> None:
        mock_payload = {
            "mode": "HYBRID",
            "confidence": 0.88,
            "reasoning": "Query covers both team ownership graph and architectural prose.",
            "suggested_template_id": "T-01",
            "extracted_entities": ["team:checkout-engineering"],
        }
        mock_anthropic_client.messages.create.return_value = _make_mock_llm_response(mock_payload)

        result = classifier.classify("Describe the checkout architecture and who owns it")
        assert result.mode == QueryIntent.HYBRID
        assert result.confidence == 0.88
        assert result.suggested_template_id == "T-01"
        assert result.method == ClassificationMethod.LLM

    def test_llm_low_confidence_fallback_to_hybrid(
        self, classifier: QueryIntentClassifier, mock_anthropic_client: MagicMock
    ) -> None:
        # LLM gave GRAPH_ONLY with confidence 0.55 (< threshold 0.65)
        mock_payload = {
            "mode": "GRAPH_ONLY",
            "confidence": 0.55,
            "reasoning": "Vague question about relations.",
            "suggested_template_id": "T-01",
            "extracted_entities": ["service:auth"],
        }
        mock_anthropic_client.messages.create.return_value = _make_mock_llm_response(mock_payload)

        result = classifier.classify("Tell me something about auth relations maybe?")
        assert result.mode == QueryIntent.HYBRID  # Automatically demoted to HYBRID
        assert result.confidence == 0.55
        assert result.method == ClassificationMethod.FALLBACK

    def test_llm_returns_markdown_code_fenced_json(
        self, classifier: QueryIntentClassifier, mock_anthropic_client: MagicMock
    ) -> None:
        mock_content = MagicMock()
        mock_content.text = (
            "```json\n"
            + json.dumps(
                {
                    "mode": "VECTOR_ONLY",
                    "confidence": 0.92,
                    "reasoning": "Document text lookup.",
                    "suggested_template_id": None,
                    "extracted_entities": [],
                }
            )
            + "\n```"
        )
        mock_resp = MagicMock()
        mock_resp.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_resp

        result = classifier.classify("What is the SLA timeout?")
        assert result.mode == QueryIntent.VECTOR_ONLY
        assert result.confidence == 0.92

    def test_llm_malformed_json_triggers_fallback(
        self, classifier: QueryIntentClassifier, mock_anthropic_client: MagicMock
    ) -> None:
        mock_content = MagicMock()
        mock_content.text = "This is not valid JSON at all!"
        mock_resp = MagicMock()
        mock_resp.content = [mock_content]
        mock_anthropic_client.messages.create.return_value = mock_resp

        result = classifier.classify("Which services depend on service:auth?")
        # Falls back to heuristic (which detects 'depend on' -> GRAPH_ONLY)
        assert result.mode == QueryIntent.GRAPH_ONLY
        assert result.method == ClassificationMethod.FALLBACK

    def test_llm_api_exception_triggers_fallback(
        self, classifier: QueryIntentClassifier, mock_anthropic_client: MagicMock
    ) -> None:
        mock_anthropic_client.messages.create.side_effect = RuntimeError(
            "Anthropic API network error"
        )

        result = classifier.classify("What is the retry policy for service:payment?")
        assert result.mode == QueryIntent.VECTOR_ONLY
        assert result.method == ClassificationMethod.FALLBACK


# ===========================================================================
# 3. Explicit Overrides & Edge Cases
# ===========================================================================


class TestOverridesAndEdgeCases:
    """Tests for explicit caller overrides and edge cases."""

    def test_caller_override_graph_only(self) -> None:
        classifier = QueryIntentClassifier(use_llm=False)
        result = classifier.classify("Any random query", retrieval_override="GRAPH_ONLY")
        assert result.mode == QueryIntent.GRAPH_ONLY
        assert result.confidence == 1.0
        assert result.method == ClassificationMethod.OVERRIDE

    def test_caller_override_with_enum(self) -> None:
        classifier = QueryIntentClassifier(use_llm=False)
        result = classifier.classify("Any query", retrieval_override=QueryIntent.VECTOR_ONLY)
        assert result.mode == QueryIntent.VECTOR_ONLY
        assert result.confidence == 1.0
        assert result.method == ClassificationMethod.OVERRIDE

    def test_offline_mode_without_llm(self) -> None:
        classifier = QueryIntentClassifier(use_llm=False)
        result = classifier.classify("Which services depend on checkout?")
        assert result.mode == QueryIntent.GRAPH_ONLY
        assert result.method == ClassificationMethod.HEURISTIC

    def test_schema_immutability(self) -> None:
        res = IntentClassificationResult(
            query="test",
            mode=QueryIntent.GRAPH_ONLY,
            confidence=0.9,
            reasoning="test",
        )
        with pytest.raises((AttributeError, TypeError, ValidationError)):
            res.confidence = 0.5
