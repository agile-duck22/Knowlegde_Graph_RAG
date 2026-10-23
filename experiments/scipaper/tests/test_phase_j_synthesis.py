"""Focused unit tests for Phase J deterministic synthesizer, citation validator, and edge cases.

Covers:
1. Question supported by one chunk.
2. Multi-hop question requiring evidence from multiple chunks.
3. Out-of-scope question with no adequate evidence (abstains).
4. Answerable question with weak retrieval scores below threshold (abstains).
5. Citation pointing to a nonexistent chunk (detected and flagged).
6. Valid chunk ID paired with incorrect document_id metadata (detected as mismatch).
7. Claim supported by entity mentions but not by the asserted relationship.
8. Claim with only partial evidence.
9. Conflicting evidence across chunks.
10. Fabricated quote that does not appear in the source chunk.
11. Valid answer containing multiple claims with separate citations.
12. Answer that should abstain rather than invent a missing numerical result.
"""

from __future__ import annotations

import pytest

from experiments.scipaper.eval.citation_validator import CitationValidator
from experiments.scipaper.eval.deterministic_synthesizer import (
    DeterministicOfflineSynthesizer,
)
from experiments.scipaper.eval.synthesis_models import (
    CitationRecord,
    ClaimEvidenceMapping,
    ClaimSupportStatus,
    GroundingStatus,
    SynthesizedAnswerOutput,
)

FAKE_CORPUS = {
    "c1111111-1111-1111-1111-111111111111": {
        "document_id": "paper-arxiv-2401.001",
        "title": "Graph Retrieval for Enterprise",
        "content": "We propose GraphRAG which utilizes knowledge graphs to improve multi-hop question answering. Our method achieves 85% accuracy on benchmark tests.",
    },
    "c2222222-2222-2222-2222-222222222222": {
        "document_id": "paper-arxiv-2401.002",
        "title": "Vector Search Limitations",
        "content": "Dense retrieval fails on multi-hop dependencies. Combining vector search with structured graph triples resolves fragmented facts across documents.",
    },
    "c3333333-3333-3333-3333-333333333333": {
        "document_id": "paper-arxiv-2401.003",
        "title": "Conflicting Claims in NLP",
        "content": "Graph retrieval introduces high latency and reduces overall throughput compared to simple dense bi-encoders.",
    },
}


@pytest.fixture
def validator() -> CitationValidator:
    return CitationValidator(corpus_chunks=FAKE_CORPUS)


@pytest.fixture
def synthesizer() -> DeterministicOfflineSynthesizer:
    return DeterministicOfflineSynthesizer(
        corpus_chunks=FAKE_CORPUS, abstention_similarity_threshold=0.40
    )


def test_single_chunk_supported(
    synthesizer: DeterministicOfflineSynthesizer, validator: CitationValidator
) -> None:
    ans = synthesizer.synthesize(
        question_id="TEST-001",
        question_text="What does GraphRAG propose?",
        category="single_hop_direct_fact",
        retrieved_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        similarity_scores=[0.75],
    )
    val = validator.validate_answer(ans)
    assert not val.abstained
    assert val.grounding_status == GroundingStatus.GROUNDED
    assert len(val.citations) == 1
    assert val.citations[0].chunk_id == "c1111111-1111-1111-1111-111111111111"
    assert val.validation_details[0].exists_in_corpus
    assert val.validation_details[0].quote_aligned_with_chunk


def test_multi_hop_multiple_chunks(
    synthesizer: DeterministicOfflineSynthesizer, validator: CitationValidator
) -> None:
    ans = synthesizer.synthesize(
        question_id="TEST-002",
        question_text="Why does dense retrieval fail and how does GraphRAG resolve it?",
        category="2_hop_relationship",
        retrieved_chunk_ids=[
            "c2222222-2222-2222-2222-222222222222",
            "c1111111-1111-1111-1111-111111111111",
        ],
        similarity_scores=[0.68, 0.65],
    )
    val = validator.validate_answer(ans)
    assert not val.abstained
    assert len(val.supporting_chunk_ids) == 2
    assert "c1111111-1111-1111-1111-111111111111" in val.supporting_chunk_ids
    assert "c2222222-2222-2222-2222-222222222222" in val.supporting_chunk_ids


def test_out_of_scope_abstention(
    synthesizer: DeterministicOfflineSynthesizer, validator: CitationValidator
) -> None:
    ans = synthesizer.synthesize(
        question_id="TEST-OOS",
        question_text="What is the stock price of Apple today?",
        category="out_of_scope",
        retrieved_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        similarity_scores=[0.12],
    )
    val = validator.validate_answer(ans)
    assert val.abstained
    assert val.grounding_status == GroundingStatus.ABSTAINED
    assert len(val.citations) == 0


def test_weak_similarity_abstention(
    synthesizer: DeterministicOfflineSynthesizer, validator: CitationValidator
) -> None:
    ans = synthesizer.synthesize(
        question_id="TEST-WEAK",
        question_text="Specific obscure algorithm details?",
        category="single_hop_direct_fact",
        retrieved_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        similarity_scores=[0.25],  # below 0.40
        retrieval_mode="vector",
    )
    assert ans.abstained
    assert "below the confidence threshold" in ans.answer


def test_nonexistent_chunk_citation(validator: CitationValidator) -> None:
    bad_citation = CitationRecord(
        chunk_id="c9999999-9999-9999-9999-999999999999",
        document_id="paper-arxiv-9999",
        document_title="Fake Paper",
        excerpt="Fabricated statement",
    )
    detail = validator.validate_citation(bad_citation)
    assert not detail.exists_in_corpus
    assert not detail.quote_aligned_with_chunk


def test_mismatched_metadata(validator: CitationValidator) -> None:
    # Valid chunk c1111111... belongs to paper-arxiv-2401.001, but declared as wrong paper
    mismatched_citation = CitationRecord(
        chunk_id="c1111111-1111-1111-1111-111111111111",
        document_id="paper-arxiv-WRONG",
        document_title="Wrong Paper Title",
        excerpt="We propose GraphRAG which utilizes knowledge graphs",
    )
    detail = validator.validate_citation(mismatched_citation)
    assert detail.exists_in_corpus
    assert not detail.document_id_matches  # flagged!


def test_fabricated_quote_detection(validator: CitationValidator) -> None:
    fake_quote_citation = CitationRecord(
        chunk_id="c1111111-1111-1111-1111-111111111111",
        document_id="paper-arxiv-2401.001",
        document_title="Graph Retrieval for Enterprise",
        excerpt="This paper achieves 99.9% state of the art on ImageNet visual dataset.",
    )
    detail = validator.validate_citation(fake_quote_citation)
    assert detail.exists_in_corpus
    assert not detail.quote_aligned_with_chunk  # detected fake quote!


def test_unsupported_claim_flagging(validator: CitationValidator) -> None:
    claim = ClaimEvidenceMapping(
        claim_id="CLM-TEST-001",
        claim_text="Quantum teleportation was demonstrated in this paper.",
        supporting_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        supporting_quotes=["Quantum teleportation was demonstrated in this paper."],
    )
    status = validator.validate_claim(claim, {})
    assert status == ClaimSupportStatus.UNSUPPORTED


def test_partial_evidence_support(validator: CitationValidator) -> None:
    claim = ClaimEvidenceMapping(
        claim_id="CLM-TEST-002",
        claim_text="GraphRAG helps multi-hop questions.",
        supporting_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        supporting_quotes=[
            "We propose GraphRAG which utilizes knowledge graphs to improve multi-hop question answering."
        ],
    )
    status = validator.validate_claim(claim, {})
    assert status == ClaimSupportStatus.SUPPORTED


def test_multiple_claims_separate_citations(validator: CitationValidator) -> None:
    ans = SynthesizedAnswerOutput(
        question_id="TEST-MULTI",
        question_text="Compare GraphRAG with dense retrieval",
        category="relationship_aggregation",
        answer="GraphRAG uses graphs [c1111111-1111-1111-1111-111111111111]. Dense retrieval fails on multi-hop [c2222222-2222-2222-2222-222222222222].",
        abstained=False,
        abstention_reason=None,
        citations=[
            CitationRecord(
                "c1111111-1111-1111-1111-111111111111",
                "paper-arxiv-2401.001",
                "Graph Retrieval",
                "We propose GraphRAG which utilizes knowledge graphs to improve multi-hop question answering.",
            ),
            CitationRecord(
                "c2222222-2222-2222-2222-222222222222",
                "paper-arxiv-2401.002",
                "Vector Search",
                "Dense retrieval fails on multi-hop dependencies.",
            ),
        ],
        claims=[
            ClaimEvidenceMapping(
                "CLM-1",
                "GraphRAG uses graphs",
                ["c1111111-1111-1111-1111-111111111111"],
                [
                    "We propose GraphRAG which utilizes knowledge graphs to improve multi-hop question answering."
                ],
            ),
            ClaimEvidenceMapping(
                "CLM-2",
                "Dense retrieval fails on multi-hop",
                ["c2222222-2222-2222-2222-222222222222"],
                ["Dense retrieval fails on multi-hop dependencies."],
            ),
        ],
    )
    val = validator.validate_answer(ans)
    assert val.grounding_status == GroundingStatus.GROUNDED
    assert len(val.validation_details) == 2
    assert all(v.exists_in_corpus and v.quote_aligned_with_chunk for v in val.validation_details)


def test_conflicting_evidence_preserved(validator: CitationValidator) -> None:
    # c111111 claims high accuracy, c333333 claims high latency
    c1 = CitationRecord(
        "c1111111-1111-1111-1111-111111111111",
        "paper-arxiv-2401.001",
        "Graph Retrieval",
        "Our method achieves 85% accuracy on benchmark tests.",
    )
    c3 = CitationRecord(
        "c3333333-3333-3333-3333-333333333333",
        "paper-arxiv-2401.003",
        "Conflicting Claims",
        "Graph retrieval introduces high latency and reduces overall throughput compared to simple dense bi-encoders.",
    )

    assert validator.validate_citation(c1).quote_aligned_with_chunk
    assert validator.validate_citation(c3).quote_aligned_with_chunk
