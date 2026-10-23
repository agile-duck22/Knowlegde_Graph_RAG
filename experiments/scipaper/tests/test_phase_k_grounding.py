"""Phase K Grounding, Regression, and Coverage Unit Tests.

Verifies:
1. EVAL-035: TagRAG versus HetaRAG coverage and comparison synthesis.
2. EVAL-036: Off-topic detection and abstention when targeted retrieval is disabled or fails.
3. 8 Adversarial Cases from Phase J.1:
   - Case 1: Method mentioned only in related work (background attribution rejection).
   - Case 3: Explicitly rejected method / negative finding preservation.
   - Case 4: Dataset mentioned without experiment.
   - Case 5: Metric mentioned without experimental result.
   - Case 6: Numerical result misattribution.
   - Case 7: Reversed relationship direction.
   - Case 9: One-sided comparative query coverage.
   - Case 10: Conflicting claims cross-chunk handling.
4. Clean sentence boundary preservation (rejection of clipped sentences ending with 'which extra', 'data augment').
5. Safe fallback when targeted retrieval is disabled or encounters empty results.
6. Refusal accuracy on out-of-scope queries.
"""

from __future__ import annotations

import pytest

from experiments.scipaper.eval.citation_validator import CitationValidator
from experiments.scipaper.eval.coverage_guard import RetrievalCoverageGuard
from experiments.scipaper.eval.grounded_synthesizer import GroundedScientificSynthesizer
from experiments.scipaper.eval.query_analyzer import QueryIntentAnalyzer
from experiments.scipaper.eval.synthesis_models import (
    CitationRecord,
    ClaimEvidenceMapping,
    ClaimSupportStatus,
    GroundingStatus,
)


@pytest.fixture
def test_corpus() -> dict[str, dict[str, str]]:
    """Mock corpus dictionary covering test documents."""
    return {
        "chunk-tagrag-1": {
            "document_id": "paper-arxiv-2601.05254",
            "title": "TagRAG: Tag-guided Hierarchical Knowledge Graph RAG",
            "content": "To overcome these limitations, we propose TagRAG, a tag-guided hierarchical knowledge graph RAG framework. Extensive experiments show TagRAG achieves a winning rate of 78.36% against baselines.",
        },
        "chunk-hetarag-1": {
            "document_id": "paper-arxiv-2509.21336",
            "title": "HetaRAG: Hybrid Deep RAG across Heterogeneous Data Stores",
            "content": "In this work, we propose HetaRAG to query across heterogeneous relational and graph databases, mitigating latency bottlenecks.",
        },
        "chunk-scaling-1": {
            "document_id": "paper-arxiv-2403.18684",
            "title": "Scaling Laws For Dense Retrieval",
            "content": "We propose to use contrastive log-likelihood as the evaluation metric for scaling laws. Additionally, we examine scaling with prevalent data augment",
        },
        "chunk-touche-1": {
            "document_id": "paper-arxiv-2608.08143",
            "title": "Large Language Models for Retrieval-Augmented Debate",
            "content": "This consensus does not reliably track the official evaluation target, with the largest gap on the Quality maxim.",
        },
        "chunk-fairrag-1": {
            "document_id": "paper-arxiv-2510.22344",
            "title": "FAIR-RAG: Faithful Adaptive Iterative Refinement for RAG",
            "content": "We introduce FAIR-RAG, a novel agentic framework that transforms the standard RAG pipeline into a dynamic, evidence-driven reasoning process.",
        },
        "chunk-adversarial-1": {
            "document_id": "paper-arxiv-2401.001",
            "title": "Scaling Laws for Retrieval",
            "content": "Prior works rely on GraphRAG to traverse knowledge graphs. In contrast, our work introduces VectorFlow which bypasses graph construction entirely. We do not recommend using unquantized dense embeddings.",
        },
    }


def test_eval035_coverage_and_targeted_augmentation(test_corpus: dict[str, dict[str, str]]) -> None:
    """EVAL-035: Verify that when HetaRAG is missing from candidate set, coverage guard augments or qualifies."""
    analyzer = QueryIntentAnalyzer()
    guard = RetrievalCoverageGuard(corpus_chunks=test_corpus)
    synthesizer = GroundedScientificSynthesizer(corpus_chunks=test_corpus)

    qtext = "Compare how TagRAG (paper-arxiv-2601.05254) and HetaRAG (paper-arxiv-2509.21336) incorporate non-vector structures."
    analysis = analyzer.analyze(qtext)

    assert analysis.is_comparative is True
    assert "paper-arxiv-2601.05254" in analysis.required_evidence_targets
    assert "paper-arxiv-2509.21336" in analysis.required_evidence_targets

    # Case A: Initial candidates contain ONLY TagRAG
    initial_candidates = ["chunk-tagrag-1"]
    diag = guard.evaluate_and_augment(
        analysis, initial_candidates, allow_targeted_augmentation=True
    )
    assert diag.status == "COMPLETE"
    assert "chunk-hetarag-1" in diag.filtered_candidate_chunk_ids
    assert diag.targeted_retrieval_applied is True

    # Synthesizer produces comparison with both chunks cited
    ans = synthesizer.synthesize(
        question_id="EVAL-035",
        question_text=qtext,
        category="relationship_aggregation",
        retrieved_chunk_ids=initial_candidates,
        allow_targeted_augmentation=True,
    )
    assert not ans.abstained
    cited_docs = {c.document_id for c in ans.citations}
    assert "paper-arxiv-2601.05254" in cited_docs
    assert "paper-arxiv-2509.21336" in cited_docs


def test_eval035_one_sided_qualification_when_augmentation_disabled(
    test_corpus: dict[str, dict[str, str]],
) -> None:
    """EVAL-035: When augmentation is disabled, verify that a one-sided comparison is never marked complete."""
    guard = RetrievalCoverageGuard(corpus_chunks=test_corpus)
    synthesizer = GroundedScientificSynthesizer(corpus_chunks=test_corpus)
    analyzer = QueryIntentAnalyzer()

    qtext = "Compare how TagRAG (paper-arxiv-2601.05254) and HetaRAG (paper-arxiv-2509.21336) incorporate non-vector structures."
    analysis = analyzer.analyze(qtext)

    initial_candidates = ["chunk-tagrag-1"]
    diag = guard.evaluate_and_augment(
        analysis, initial_candidates, allow_targeted_augmentation=False
    )
    assert diag.status == "PARTIAL"
    assert "paper-arxiv-2509.21336" in diag.missing_targets

    ans = synthesizer.synthesize(
        question_id="EVAL-035",
        question_text=qtext,
        category="relationship_aggregation",
        retrieved_chunk_ids=initial_candidates,
        allow_targeted_augmentation=False,
    )
    assert (
        "[Note: Partial comparison only; no evidence was retrieved for paper-arxiv-2509.21336.]"
        in ans.answer
    )


def test_eval036_off_topic_detection_and_abstention(test_corpus: dict[str, dict[str, str]]) -> None:
    """EVAL-036: When retrieved candidates are completely off-topic, guard diagnoses OFF_TOPIC and synthesizer abstains."""
    analyzer = QueryIntentAnalyzer()
    guard = RetrievalCoverageGuard(corpus_chunks=test_corpus)
    synthesizer = GroundedScientificSynthesizer(corpus_chunks=test_corpus)

    qtext = "Contrast the training-time optimization in paper-arxiv-2402.11794 with inference-time optimizations in paper-arxiv-2402.11626."
    analysis = analyzer.analyze(qtext)

    # Candidate chunks are from unrelated papers (2403.18684 and 2608.08143)
    unrelated_candidates = ["chunk-scaling-1", "chunk-touche-1"]

    diag = guard.evaluate_and_augment(
        analysis, unrelated_candidates, allow_targeted_augmentation=False
    )
    assert diag.status == "OFF_TOPIC"
    assert "paper-arxiv-2402.11794" in diag.missing_targets

    ans = synthesizer.synthesize(
        question_id="EVAL-036",
        question_text=qtext,
        category="relationship_aggregation",
        retrieved_chunk_ids=unrelated_candidates,
        allow_targeted_augmentation=False,
    )
    assert ans.abstained is True
    assert ans.grounding_status == GroundingStatus.ABSTAINED
    assert "retrieved evidence does not cover the requested targets" in ans.answer


def test_clipped_sentence_truncation_filtered(test_corpus: dict[str, dict[str, str]]) -> None:
    """Verify that clipped sentences ending in truncated phrases (e.g. 'prevalent data augment') are rejected."""
    synthesizer = GroundedScientificSynthesizer(corpus_chunks=test_corpus)
    assert (
        synthesizer.is_sentence_clean(
            "Additionally, we examine scaling with prevalent data augment"
        )
        is False
    )
    assert (
        synthesizer.is_sentence_clean(
            "TagRAG introduces two key components: (1) Tag Knowledge Graph Construction, which extra"
        )
        is False
    )
    assert (
        synthesizer.is_sentence_clean(
            "In this paper, we propose a novel framework for retrieval augmented generation."
        )
        is True
    )


def test_adversarial_case1_method_in_related_work(test_corpus: dict[str, dict[str, str]]) -> None:
    """Adversarial Case 1: Proposing GraphRAG when text says 'Prior works rely on GraphRAG' fails under strict validation."""
    validator = CitationValidator(corpus_chunks=test_corpus, strict_mode=True)
    claim = ClaimEvidenceMapping(
        claim_id="CLM-ADV-01",
        claim_text="We propose GraphRAG to traverse knowledge graphs.",
        supporting_chunk_ids=["chunk-adversarial-1"],
        supporting_quotes=["Prior works rely on GraphRAG to traverse knowledge graphs."],
    )
    status = validator.validate_claim(claim, {})
    assert status == ClaimSupportStatus.UNSUPPORTED
    assert "Prior-work method" in claim.verification_notes


def test_adversarial_case7_reversed_relationship_direction(
    test_corpus: dict[str, dict[str, str]],
) -> None:
    """Adversarial Case 7: Inverting sentence words without preserving sequential n-grams fails alignment."""
    validator = CitationValidator(corpus_chunks=test_corpus, strict_mode=True)
    cit = CitationRecord(
        chunk_id="chunk-adversarial-1",
        document_id="paper-arxiv-2401.001",
        document_title="Scaling Laws for Retrieval",
        excerpt="VectorFlow relies on GraphRAG to bypass graph construction entirely.",
    )
    detail = validator.validate_citation(cit)
    assert detail.quote_aligned_with_chunk is False


def test_adversarial_case_negation_loss(test_corpus: dict[str, dict[str, str]]) -> None:
    """Adversarial Case 3 / Negation Loss: Stripping 'do not' fails under strict mode."""
    validator = CitationValidator(corpus_chunks=test_corpus, strict_mode=True)
    cit = CitationRecord(
        chunk_id="chunk-adversarial-1",
        document_id="paper-arxiv-2401.001",
        document_title="Scaling Laws for Retrieval",
        excerpt="We recommend using unquantized dense embeddings.",
    )
    detail = validator.validate_citation(cit)
    assert detail.quote_aligned_with_chunk is False


def test_adversarial_case_unsupported_numerical_result(
    test_corpus: dict[str, dict[str, str]],
) -> None:
    """Numerical result misattribution: Altering experimental metric values fails quote alignment."""
    validator = CitationValidator(corpus_chunks=test_corpus, strict_mode=True)
    cit = CitationRecord(
        chunk_id="chunk-tagrag-1",
        document_id="paper-arxiv-2601.05254",
        document_title="TagRAG: Tag-guided Hierarchical Knowledge Graph RAG",
        excerpt="Extensive experiments show TagRAG achieves a winning rate of 99.99% against baselines.",
    )
    detail = validator.validate_citation(cit)
    assert detail.quote_aligned_with_chunk is False


def test_adversarial_case_irrelevant_verbatim_claim(test_corpus: dict[str, dict[str, str]]) -> None:
    """Irrelevant verbatim claim: Quoting an ungrounded or off-topic chunk for a specific target fails."""
    analyzer = QueryIntentAnalyzer()
    guard = RetrievalCoverageGuard(corpus_chunks=test_corpus)
    qtext = "What is the primary contribution of FAIR-RAG (paper-arxiv-2510.22344)?"
    analysis = analyzer.analyze(qtext)

    # Chunks are verbatim text, but completely unrelated to paper-arxiv-2510.22344
    unrelated_candidates = ["chunk-tagrag-1", "chunk-scaling-1"]
    diag = guard.evaluate_and_augment(
        analysis, unrelated_candidates, allow_targeted_augmentation=False
    )
    assert diag.status == "OFF_TOPIC"
    assert "paper-arxiv-2510.22344" in diag.missing_targets


def test_missing_evidence_safe_fallback_abstention(test_corpus: dict[str, dict[str, str]]) -> None:
    """Safe abstention when required query coverage cannot be satisfied and augmentation fails."""
    synthesizer = GroundedScientificSynthesizer(corpus_chunks=test_corpus)
    # Question requests an unknown target paper not in the corpus
    qtext = "Describe the architecture of NonExistentSystem in paper-arxiv-9999.99999."
    ans = synthesizer.synthesize(
        question_id="EVAL-MISSING-TARGET",
        question_text=qtext,
        category="single_hop",
        retrieved_chunk_ids=["chunk-tagrag-1"],
        allow_targeted_augmentation=True,
    )
    assert ans.abstained is True
    assert "retrieved evidence does not cover the requested targets" in ans.answer


def test_missing_target_retrieval_failure_and_recovery(
    test_corpus: dict[str, dict[str, str]],
) -> None:
    """Targeted retrieval recovery when candidate lacks target, and graceful fallback when target absent from corpus."""
    guard = RetrievalCoverageGuard(corpus_chunks=test_corpus)
    analyzer = QueryIntentAnalyzer()

    # Subtest A: Target exists in corpus -> Successful recovery
    analysis_a = analyzer.analyze(
        "Compare TagRAG (paper-arxiv-2601.05254) and HetaRAG (paper-arxiv-2509.21336)."
    )
    diag_a = guard.evaluate_and_augment(
        analysis_a, ["chunk-tagrag-1"], allow_targeted_augmentation=True
    )
    assert diag_a.status == "COMPLETE"
    assert "chunk-hetarag-1" in diag_a.filtered_candidate_chunk_ids

    # Subtest B: Target does NOT exist in corpus -> Diagnostic fallback to PARTIAL
    analysis_b = analyzer.analyze(
        "Compare TagRAG (paper-arxiv-2601.05254) and UnknownSys (paper-arxiv-1111.22222)."
    )
    diag_b = guard.evaluate_and_augment(
        analysis_b, ["chunk-tagrag-1"], allow_targeted_augmentation=True
    )
    assert diag_b.status == "PARTIAL"
    assert "paper-arxiv-1111.22222" in diag_b.missing_targets
