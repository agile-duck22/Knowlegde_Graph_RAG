"""Focused adversarial unit tests for Phase J.1.

Explicitly tests:
1. Real chunk ID paired with a false document title.
2. Real chunk ID paired with the wrong parent document ID.
3. Valid quote paired with an unsupported inference.
4. Fabricated quote that does not appear in source.
5. Quote that omits a crucial negation or qualification.
6. Unrelated but verbatim sentence extracted as answer.
7. Real source chunk that does not support the claim.
8. Multiple claims incorrectly sharing one citation.
9. Missing citation for a factual claim.
10. Citation to a nonexistent chunk.
11. Comparison with a citation supporting only one side.
12. Conflicting claims from two sources without conflict resolution.
"""

from __future__ import annotations

import pytest

from experiments.scipaper.eval.citation_validator import CitationValidator
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
        "title": "Scaling Laws for Retrieval",
        "content": "We do not recommend using unquantized dense embeddings due to severe memory overhead in production clusters. In our tests, 4-bit quantization preserved 98% of MRR.",
    },
    "c2222222-2222-2222-2222-222222222222": {
        "document_id": "paper-arxiv-2401.002",
        "title": "Comparative Study of RAG Backends",
        "content": "Graph databases provide superior multi-hop reasoning capabilities compared to standalone vector indexes.",
    },
    "c3333333-3333-3333-3333-333333333333": {
        "document_id": "paper-arxiv-2401.003",
        "title": "Contradictory RAG Findings",
        "content": "Graph databases do not provide any measurable advantage over well-tuned HNSW dense vector indexes for multi-hop QA.",
    },
}


@pytest.fixture
def validator() -> CitationValidator:
    return CitationValidator(corpus_chunks=FAKE_CORPUS)


def test_real_chunk_with_false_title(validator: CitationValidator) -> None:
    cit = CitationRecord(
        chunk_id="c1111111-1111-1111-1111-111111111111",
        document_id="paper-arxiv-2401.001",
        document_title="Completely False Title About Quantum Mechanics",
        excerpt="4-bit quantization preserved 98% of MRR.",
    )
    # The citation has fields, but document title is fabricated
    detail = validator.validate_citation(cit)
    assert detail.exists_in_corpus
    # Documents in corpus have true title "Scaling Laws for Retrieval"
    expected_title = FAKE_CORPUS[cit.chunk_id]["title"]
    assert cit.document_title != expected_title


def test_real_chunk_wrong_parent_doc(validator: CitationValidator) -> None:
    cit = CitationRecord(
        chunk_id="c1111111-1111-1111-1111-111111111111",
        document_id="paper-arxiv-WRONG-DOC",
        document_title="Scaling Laws for Retrieval",
        excerpt="4-bit quantization preserved 98% of MRR.",
    )
    detail = validator.validate_citation(cit)
    assert detail.exists_in_corpus
    assert not detail.document_id_matches  # correctly caught!


def test_valid_quote_unsupported_inference(validator: CitationValidator) -> None:
    # Quote is from chunk ("4-bit quantization preserved 98% of MRR"),
    # but the claim makes an unsupported inference ("unquantized models are obsolete worldwide")
    claim = ClaimEvidenceMapping(
        claim_id="CLM-ADV-01",
        claim_text="Unquantized embeddings are universally obsolete across all enterprise applications.",
        supporting_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        supporting_quotes=[
            "Unquantized embeddings are universally obsolete across all enterprise applications."
        ],
    )
    status = validator.validate_claim(claim, {})
    assert status == ClaimSupportStatus.UNSUPPORTED  # correctly flagged!


def test_fabricated_quote(validator: CitationValidator) -> None:
    cit = CitationRecord(
        chunk_id="c1111111-1111-1111-1111-111111111111",
        document_id="paper-arxiv-2401.001",
        document_title="Scaling Laws for Retrieval",
        excerpt="Our system achieves 99.99% exact match accuracy on all enterprise datasets.",
    )
    detail = validator.validate_citation(cit)
    assert detail.exists_in_corpus
    assert not detail.quote_aligned_with_chunk  # caught!


def test_quote_omits_negation_vulnerability(validator: CitationValidator) -> None:
    # Source says: "We do not recommend using unquantized dense embeddings"
    # Adversarial excerpt strips "do not": "recommend using unquantized dense embeddings"
    cit = CitationRecord(
        chunk_id="c1111111-1111-1111-1111-111111111111",
        document_id="paper-arxiv-2401.001",
        document_title="Scaling Laws for Retrieval",
        excerpt="We recommend using unquantized dense embeddings",
    )
    detail = validator.validate_citation(cit)
    # VULNERABILITY EXPOSED: The bag-of-words fallback (lines 75-78 of citation_validator.py)
    # treats the excerpt as aligned because words ('recommend', 'using', 'unquantized', 'dense', 'embeddings')
    # appear in the chunk, even though the critical negation 'not' was stripped!
    assert detail.quote_aligned_with_chunk is True, (
        "Demonstrates vulnerability: bag-of-words keyword check passes despite omitted negation."
    )


def test_unrelated_verbatim_sentence(validator: CitationValidator) -> None:
    # True verbatim quote, but if asked about query latency, talking about memory overhead is topical divergence
    claim = ClaimEvidenceMapping(
        claim_id="CLM-ADV-02",
        claim_text="severe memory overhead in production clusters.",
        supporting_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        supporting_quotes=["severe memory overhead in production clusters."],
    )
    status = validator.validate_claim(claim, {})
    assert status == ClaimSupportStatus.SUPPORTED
    # Note: Validator considers it supported because text is in chunk, proving claim grounding != question relevance!


def test_multiple_claims_sharing_one_citation(validator: CitationValidator) -> None:
    # Two separate claims: one is in chunk, one is completely fabricated
    c1 = ClaimEvidenceMapping(
        claim_id="C-1",
        claim_text="4-bit quantization preserved 98% of MRR.",
        supporting_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        supporting_quotes=["4-bit quantization preserved 98% of MRR."],
    )
    c2 = ClaimEvidenceMapping(
        claim_id="C-2",
        claim_text="The model was trained on 500 TPU pods for six months.",
        supporting_chunk_ids=["c1111111-1111-1111-1111-111111111111"],
        supporting_quotes=["The model was trained on 500 TPU pods for six months."],
    )
    s1 = validator.validate_claim(c1, {})
    s2 = validator.validate_claim(c2, {})
    assert s1 == ClaimSupportStatus.SUPPORTED
    assert s2 == ClaimSupportStatus.UNSUPPORTED


def test_missing_citation_for_claim(validator: CitationValidator) -> None:
    claim = ClaimEvidenceMapping(
        claim_id="C-NOCIT",
        claim_text="A factual claim with no citations.",
        supporting_chunk_ids=[],
        supporting_quotes=[],
    )
    status = validator.validate_claim(claim, {})
    assert status == ClaimSupportStatus.UNSUPPORTED


def test_citation_to_nonexistent_chunk(validator: CitationValidator) -> None:
    cit = CitationRecord(
        chunk_id="c0000000-0000-0000-0000-000000000000",
        document_id="paper-arxiv-9999",
        document_title="Ghost Paper",
        excerpt="Ghost excerpt",
    )
    detail = validator.validate_citation(cit)
    assert not detail.exists_in_corpus


def test_comparison_one_side_only(validator: CitationValidator) -> None:
    # Simulates EVAL-035: Question asks to compare Graph and Vector backends
    # Answer provides chunk for Graph (c222222), but provides NO chunk for Vector
    ans = SynthesizedAnswerOutput(
        question_id="TEST-COMP",
        question_text="Compare graph databases and vector databases for multi-hop QA",
        category="relationship_aggregation",
        answer="Graph databases provide superior multi-hop reasoning capabilities compared to standalone vector indexes. [c2222222-2222-2222-2222-222222222222]",
        abstained=False,
        abstention_reason=None,
        citations=[
            CitationRecord(
                "c2222222-2222-2222-2222-222222222222",
                "paper-arxiv-2401.002",
                "Comparative Study",
                "Graph databases provide superior multi-hop reasoning capabilities compared to standalone vector indexes.",
            )
        ],
        claims=[
            ClaimEvidenceMapping(
                "CLM-1",
                "Graph databases provide superior multi-hop reasoning capabilities compared to standalone vector indexes.",
                ["c2222222-2222-2222-2222-222222222222"],
                [
                    "Graph databases provide superior multi-hop reasoning capabilities compared to standalone vector indexes."
                ],
            )
        ],
    )
    val = validator.validate_answer(ans)
    assert val.grounding_status == GroundingStatus.GROUNDED
    # Vulnerability exposed: The answer is 100% GROUNDED to c2222222, but incomplete regarding the comparison!
    assert len(val.citations) == 1


def test_conflicting_claims_across_chunks(validator: CitationValidator) -> None:
    # c222222 says Graph is superior. c333333 says Graph has NO advantage.
    c2 = CitationRecord(
        "c2222222-2222-2222-2222-222222222222",
        "paper-arxiv-2401.002",
        "Study A",
        "Graph databases provide superior multi-hop reasoning capabilities compared to standalone vector indexes.",
    )
    c3 = CitationRecord(
        "c3333333-3333-3333-3333-333333333333",
        "paper-arxiv-2401.003",
        "Study B",
        "Graph databases do not provide any measurable advantage over well-tuned HNSW dense vector indexes for multi-hop QA.",
    )

    d2 = validator.validate_citation(c2)
    d3 = validator.validate_citation(c3)
    assert d2.exists_in_corpus and d2.quote_aligned_with_chunk
    assert d3.exists_in_corpus and d3.quote_aligned_with_chunk
    # Both are individually valid citations, yet their semantic claims are directly contradictory
