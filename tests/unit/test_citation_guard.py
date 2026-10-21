"""Unit tests for Citation & Grounding Guardrail (TASK-403).

Verifies:
- Authoritative validation of cited chunk IDs against candidate chunk IDs.
- Detection of hallucinated / unknown citation IDs and automatic citation scrubbing.
- Grounding status classification: VALID, PARTIALLY_GROUNDED, UNGROUNDED, REFUSAL.
- Enriched citation provenance mapping (linking chunk ID to document ID and excerpt).
- Preservation of regular bracketed technical terms (e.g., [JSON-RPC], [0], [x]) during scrubbing.
- Graceful handling of empty answers, empty contexts, missing citations, and duplicate citations.
- Deterministic behavior and offline execution without paid LLM API or network calls.
"""

from __future__ import annotations

import pytest

from src.synthesis.citation_validator import (
    CitationValidator,
    GroundingStatus,
    GuardrailResult,
)
from src.synthesis.normalizer import (
    NormalizedChunkItem,
    NormalizedContext,
    NormalizedGraphTriple,
)
from src.synthesis.synthesizer import DraftAnswer, TokenUsageMetrics


@pytest.fixture
def validator() -> CitationValidator:
    return CitationValidator()


@pytest.fixture
def sample_normalized_context() -> NormalizedContext:
    chunk1 = NormalizedChunkItem(
        chunk_id="c1111111-1111-1111-1111-111111111111",
        document_id="doc-checkout-arch",
        content="The Checkout Service is owned and operated by the Checkout Team.",
        similarity_score=0.95,
        provenance_source="hybrid",
        token_count=15,
    )
    chunk2 = NormalizedChunkItem(
        chunk_id="c2222222-2222-2222-2222-222222222222",
        document_id="doc-payment-gateway-spec",
        content="Payment Gateway makes synchronous gRPC calls to the Checkout Service for order validation.",
        similarity_score=0.88,
        provenance_source="vector",
        token_count=18,
    )
    triple1 = NormalizedGraphTriple(
        source_id="service:payment-gateway",
        source_name="Payment Gateway",
        relationship_type="DEPENDS_ON",
        target_id="service:checkout-service",
        target_name="Checkout Service",
        source_chunk_ids=["c3333333-3333-3333-3333-333333333333"],
    )
    return NormalizedContext(
        formatted_context="Sample context text",
        chunks=[chunk1, chunk2],
        graph_triples=[triple1],
        graph_entities=[],
        total_tokens=100,
        max_budget_tokens=1000,
        candidate_chunk_ids=[
            "c1111111-1111-1111-1111-111111111111",
            "c2222222-2222-2222-2222-222222222222",
            "c3333333-3333-3333-3333-333333333333",
        ],
        all_entity_ids=["service:checkout-service", "service:payment-gateway"],
        is_truncated=False,
        truncated_chunks_count=0,
        truncated_triples_count=0,
    )


# ---------------------------------------------------------------------------
# Valid Citation & Provenance Tests
# ---------------------------------------------------------------------------


def test_validate_all_valid_citations(
    validator: CitationValidator, sample_normalized_context: NormalizedContext
) -> None:
    draft = DraftAnswer(
        question="Who owns Checkout and what depends on it?",
        raw_answer=(
            "The Checkout Service is maintained by Checkout Team [c1111111-1111-1111-1111-111111111111]. "
            "The Payment Gateway depends on it [c2222222-2222-2222-2222-222222222222]."
        ),
        cited_chunk_ids=[
            "c1111111-1111-1111-1111-111111111111",
            "c2222222-2222-2222-2222-222222222222",
        ],
        candidate_chunk_ids=sample_normalized_context.candidate_chunk_ids,
        token_usage=TokenUsageMetrics(prompt_tokens=100, completion_tokens=30, total_tokens=130),
        execution_time_ms=50.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    res: GuardrailResult = validator.validate(draft, sample_normalized_context)

    assert res.status == GroundingStatus.VALID
    assert res.is_grounded is True
    assert res.has_hallucinated_citations is False
    assert res.valid_chunk_ids == [
        "c1111111-1111-1111-1111-111111111111",
        "c2222222-2222-2222-2222-222222222222",
    ]
    assert res.invalid_chunk_ids == []
    assert res.sanitized_answer == draft.raw_answer
    assert len(res.valid_citations) == 2
    assert res.valid_citations[0].document_id == "doc-checkout-arch"
    assert "Checkout Team" in res.valid_citations[0].excerpt
    assert res.valid_citations[1].document_id == "doc-payment-gateway-spec"


def test_validate_graph_only_provenance_citation(
    validator: CitationValidator, sample_normalized_context: NormalizedContext
) -> None:
    draft = DraftAnswer(
        question="What is the relation between Payment Gateway and Checkout?",
        raw_answer="Payment Gateway depends on Checkout [c3333333-3333-3333-3333-333333333333].",
        cited_chunk_ids=["c3333333-3333-3333-3333-333333333333"],
        candidate_chunk_ids=sample_normalized_context.candidate_chunk_ids,
        token_usage=TokenUsageMetrics(prompt_tokens=50, completion_tokens=15, total_tokens=65),
        execution_time_ms=30.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    res = validator.validate(draft, sample_normalized_context)

    assert res.status == GroundingStatus.VALID
    assert res.is_grounded is True
    assert res.valid_chunk_ids == ["c3333333-3333-3333-3333-333333333333"]
    assert res.valid_citations[0].provenance_source == "graph"


# ---------------------------------------------------------------------------
# Hallucination Detection & Citation Scrubbing Tests
# ---------------------------------------------------------------------------


def test_validate_hallucinated_citation_scrubbing(
    validator: CitationValidator, sample_normalized_context: NormalizedContext
) -> None:
    draft = DraftAnswer(
        question="Who owns Checkout and what is its port?",
        raw_answer=(
            "The Checkout Service is maintained by Checkout Team [c1111111-1111-1111-1111-111111111111]. "
            "It runs on port 8080 [chunk-unknown-999]."
        ),
        cited_chunk_ids=[
            "c1111111-1111-1111-1111-111111111111",
            "chunk-unknown-999",
        ],
        candidate_chunk_ids=sample_normalized_context.candidate_chunk_ids,
        token_usage=TokenUsageMetrics(prompt_tokens=100, completion_tokens=30, total_tokens=130),
        execution_time_ms=45.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    res = validator.validate(draft, sample_normalized_context)

    assert res.status == GroundingStatus.PARTIALLY_GROUNDED
    assert res.is_grounded is True
    assert res.has_hallucinated_citations is True
    assert res.valid_chunk_ids == ["c1111111-1111-1111-1111-111111111111"]
    assert res.invalid_chunk_ids == ["chunk-unknown-999"]
    # Verify hallucinated citation was scrubbed from sanitized answer while valid citation was kept
    assert "[c1111111-1111-1111-1111-111111111111]" in res.sanitized_answer
    assert "[chunk-unknown-999]" not in res.sanitized_answer
    assert "It runs on port 8080." in res.sanitized_answer
    assert len(res.warnings) == 1
    assert res.warnings[0].code == "HALLUCINATED_CITATION"
    assert res.warnings[0].chunk_id == "chunk-unknown-999"


def test_validate_scrubbing_preserves_unrelated_bracketed_text(
    validator: CitationValidator, sample_normalized_context: NormalizedContext
) -> None:
    draft = DraftAnswer(
        question="What protocol and array index does Checkout use?",
        raw_answer=(
            "The Checkout Service uses protocol [HTTP-2] with payload items at index [0] "
            "[c1111111-1111-1111-1111-111111111111] and invalid reference [chunk-hallucinated-999]."
        ),
        cited_chunk_ids=[
            "c1111111-1111-1111-1111-111111111111",
            "chunk-hallucinated-999",
        ],
        candidate_chunk_ids=sample_normalized_context.candidate_chunk_ids,
        token_usage=TokenUsageMetrics(prompt_tokens=100, completion_tokens=30, total_tokens=130),
        execution_time_ms=40.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    res = validator.validate(draft, sample_normalized_context)

    # Verifies only invalid citation [chunk-hallucinated-999] was scrubbed;
    # non-cited bracketed phrases [HTTP-2] and [0] were preserved intact.
    assert "[HTTP-2]" in res.sanitized_answer
    assert "[0]" in res.sanitized_answer
    assert "[c1111111-1111-1111-1111-111111111111]" in res.sanitized_answer
    assert "[chunk-hallucinated-999]" not in res.sanitized_answer
    assert res.invalid_chunk_ids == ["chunk-hallucinated-999"]
    assert res.valid_chunk_ids == ["c1111111-1111-1111-1111-111111111111"]


def test_validate_completely_ungrounded_answer(
    validator: CitationValidator, sample_normalized_context: NormalizedContext
) -> None:
    draft = DraftAnswer(
        question="What is the internal database password?",
        raw_answer="The password is admin123 [chunk-fake-1] and stored at /etc/secret [chunk-fake-2].",
        cited_chunk_ids=["chunk-fake-1", "chunk-fake-2"],
        candidate_chunk_ids=sample_normalized_context.candidate_chunk_ids,
        token_usage=TokenUsageMetrics(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        execution_time_ms=40.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    res = validator.validate(draft, sample_normalized_context)

    assert res.status == GroundingStatus.UNGROUNDED
    assert res.is_grounded is False
    assert res.valid_chunk_ids == []
    assert res.invalid_chunk_ids == ["chunk-fake-1", "chunk-fake-2"]
    assert len(res.warnings) == 2


# ---------------------------------------------------------------------------
# Missing Citations & Refusal Handling Tests
# ---------------------------------------------------------------------------


def test_validate_missing_citations_when_context_available(
    validator: CitationValidator, sample_normalized_context: NormalizedContext
) -> None:
    draft = DraftAnswer(
        question="Who owns Checkout?",
        raw_answer="The Checkout Service is owned by Checkout Team.",
        cited_chunk_ids=[],
        candidate_chunk_ids=sample_normalized_context.candidate_chunk_ids,
        token_usage=TokenUsageMetrics(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        execution_time_ms=30.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    res = validator.validate(draft, sample_normalized_context)

    assert res.status == GroundingStatus.UNGROUNDED
    assert res.is_grounded is False
    assert res.valid_chunk_ids == []
    assert len(res.warnings) == 1
    assert res.warnings[0].code == "MISSING_CITATIONS"


def test_validate_explicit_refusal_with_context(
    validator: CitationValidator, sample_normalized_context: NormalizedContext
) -> None:
    draft = DraftAnswer(
        question="What is the weather forecast for tomorrow?",
        raw_answer="I could not find any relevant documentation or information to answer this question.",
        cited_chunk_ids=[],
        candidate_chunk_ids=sample_normalized_context.candidate_chunk_ids,
        token_usage=TokenUsageMetrics(prompt_tokens=100, completion_tokens=15, total_tokens=115),
        execution_time_ms=25.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    res = validator.validate(draft, sample_normalized_context)

    assert res.status == GroundingStatus.REFUSAL
    assert res.is_grounded is True
    assert res.valid_chunk_ids == []
    assert res.invalid_chunk_ids == []
    assert res.warnings == []


def test_validate_empty_context_refusal(validator: CitationValidator) -> None:
    empty_context = NormalizedContext(
        formatted_context="",
        chunks=[],
        graph_triples=[],
        graph_entities=[],
        total_tokens=0,
        max_budget_tokens=1000,
        candidate_chunk_ids=[],
        all_entity_ids=[],
        is_truncated=False,
        truncated_chunks_count=0,
        truncated_triples_count=0,
    )
    draft = DraftAnswer(
        question="Tell me about service X",
        raw_answer="I could not find any relevant documentation or graph relationships to answer your question.",
        cited_chunk_ids=[],
        candidate_chunk_ids=[],
        token_usage=TokenUsageMetrics(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        execution_time_ms=1.0,
        model_name="claude-3-5-sonnet-20241022",
        metadata={"empty_context": True},
    )

    res = validator.validate(draft, empty_context)

    assert res.status == GroundingStatus.REFUSAL
    assert res.is_grounded is True
    assert res.valid_chunk_ids == []
    assert res.metadata.get("empty_context") is True


def test_validate_empty_answer_string(
    validator: CitationValidator, sample_normalized_context: NormalizedContext
) -> None:
    draft = DraftAnswer(
        question="Who owns Checkout?",
        raw_answer="   ",
        cited_chunk_ids=[],
        candidate_chunk_ids=sample_normalized_context.candidate_chunk_ids,
        token_usage=TokenUsageMetrics(prompt_tokens=100, completion_tokens=0, total_tokens=100),
        execution_time_ms=10.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    res = validator.validate(draft, sample_normalized_context)

    assert res.status == GroundingStatus.UNGROUNDED
    assert res.is_grounded is False
    assert res.sanitized_answer == ""
    assert any(w.code == "EMPTY_ANSWER" for w in res.warnings)


# ---------------------------------------------------------------------------
# Duplicate Citations & Edge Cases Tests
# ---------------------------------------------------------------------------


def test_validate_duplicate_citations_handling(
    validator: CitationValidator, sample_normalized_context: NormalizedContext
) -> None:
    draft = DraftAnswer(
        question="Tell me about Checkout",
        raw_answer=(
            "Checkout is owned by Checkout Team [c1111111-1111-1111-1111-111111111111]. "
            "It is operated by Checkout Team [c1111111-1111-1111-1111-111111111111]."
        ),
        cited_chunk_ids=[
            "c1111111-1111-1111-1111-111111111111",
            "c1111111-1111-1111-1111-111111111111",
        ],
        candidate_chunk_ids=sample_normalized_context.candidate_chunk_ids,
        token_usage=TokenUsageMetrics(prompt_tokens=80, completion_tokens=25, total_tokens=105),
        execution_time_ms=35.0,
        model_name="claude-3-5-sonnet-20241022",
    )

    res = validator.validate(draft, sample_normalized_context)

    assert res.status == GroundingStatus.VALID
    assert res.is_grounded is True
    # Valid chunk IDs list is deduplicated
    assert res.valid_chunk_ids == ["c1111111-1111-1111-1111-111111111111"]
    assert len(res.valid_citations) == 1
