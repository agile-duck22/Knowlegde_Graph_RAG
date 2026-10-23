"""Phase J Citation Validator: Comprehensive Grounding and Provenance Guardrail.

Verifies:
1. Every cited chunk ID exists in the authoritative PostgreSQL/local corpus.
2. Every cited chunk belongs to the declared document ID.
3. Citations contain complete metadata (document_id, title, excerpt).
4. Evidence text / quotes actually appear verbatim in the cited chunk content.
5. Factual claims are supported by the specific citations linked to them.
6. Catches fabricated quotes, unknown chunk IDs, and entity-only co-occurrence fallacies.
"""

from __future__ import annotations

import logging
from typing import Any

from experiments.scipaper.eval.synthesis_models import (
    CitationRecord,
    CitationValidationDetail,
    ClaimEvidenceMapping,
    ClaimSupportStatus,
    GroundingStatus,
    SynthesizedAnswerOutput,
)

logger = logging.getLogger(__name__)


class CitationValidator:
    """Offline Citation and Claim Evidence Validator."""

    def __init__(self, corpus_chunks: dict[str, dict[str, Any]], strict_mode: bool = False) -> None:
        """Initialize validator with authoritative chunk map.

        Args:
            corpus_chunks: Mapping of chunk_id (UUID string) -> dict with:
                           'document_id', 'content', 'title' or 'metadata'.
            strict_mode: If True, enforces strict negation matching, attribution checks,
                         and sequential ordering to reject adversarial modifications.
        """
        self.corpus_chunks = corpus_chunks
        self.strict_mode = strict_mode

    def validate_citation(self, citation: CitationRecord) -> CitationValidationDetail:
        """Validate an individual citation record against the corpus."""
        chunk = self.corpus_chunks.get(citation.chunk_id)
        if chunk is None:
            return CitationValidationDetail(
                chunk_id=citation.chunk_id,
                exists_in_corpus=False,
                document_id_matches=False,
                metadata_complete=False,
                quote_aligned_with_chunk=False,
                semantic_relationship_supported=False,
                notes=f"Chunk ID {citation.chunk_id} does not exist in corpus.",
            )

        # 1. Document ID match
        expected_doc_id = chunk.get("document_id", "")
        doc_match = bool(expected_doc_id and expected_doc_id == citation.document_id)

        # 2. Metadata completeness
        meta_complete = bool(
            citation.chunk_id
            and citation.document_id
            and citation.document_title
            and citation.excerpt
        )

        # 3. Quote alignment with negation & order preservation
        content = chunk.get("content", "")
        quote_clean = citation.excerpt.strip().lower()
        content_clean = content.lower()
        aligned = False

        if quote_clean and quote_clean in content_clean:
            aligned = True
        elif quote_clean:
            if self.strict_mode:
                # In strict mode, enforce negation consistency and sequential matching
                quote_has_not = (
                    " not " in f" {quote_clean} "
                    or "n't " in quote_clean
                    or " never " in quote_clean
                )
                content_has_not = (
                    " not " in f" {content_clean} "
                    or "n't " in content_clean
                    or " never " in content_clean
                )
                if quote_has_not != content_has_not:
                    aligned = False
                else:
                    words = [w for w in quote_clean.split() if len(w) > 3]
                    # Also check for numerical values/percentages in quote_clean
                    import re

                    quote_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", quote_clean))
                    content_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", content_clean))
                    if quote_numbers and not quote_numbers.issubset(content_numbers):
                        aligned = False
                    elif len(words) >= 3:
                        found_ngram = False
                        for i in range(len(words) - 2):
                            ngram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                            if ngram in content_clean:
                                found_ngram = True
                                break
                        if found_ngram:
                            aligned = True
            else:
                # Baseline Phase J bag-of-words fallback
                words = [w for w in quote_clean.split() if len(w) > 3]
                if words and all(w in content_clean for w in words[: min(5, len(words))]):
                    aligned = True

        return CitationValidationDetail(
            chunk_id=citation.chunk_id,
            exists_in_corpus=True,
            document_id_matches=doc_match,
            metadata_complete=meta_complete,
            quote_aligned_with_chunk=aligned,
            semantic_relationship_supported=True if (doc_match and aligned) else False,
            notes="Valid citation" if (doc_match and aligned) else "Metadata or excerpt mismatch",
        )

    def validate_claim(
        self, claim: ClaimEvidenceMapping, citations_by_id: dict[str, CitationRecord]
    ) -> ClaimSupportStatus:
        """Validate whether an atomic factual claim is supported by its cited chunks.

        Distinguishes:
        - Exact verbatim quotes from cited chunks (SUPPORTED)
        - Negation loss / reversed direction (UNSUPPORTED)
        - Attribution to prior works / background when claimed as current paper contribution (UNVERIFIED / UNSUPPORTED)
        """
        if not claim.supporting_chunk_ids:
            claim.support_status = ClaimSupportStatus.UNSUPPORTED
            claim.verification_notes = "No supporting chunk IDs provided for claim."
            return claim.support_status

        valid_chunk_found = False
        aligned_quote_found = False
        has_partial_overlap = False
        negation_mismatch = False
        background_attribution_conflict = False

        claim_clean = claim.claim_text.strip().lower()
        claim_has_not = (
            " not " in f" {claim_clean} "
            or "n't " in claim_clean
            or " never " in claim_clean
            or "abandon" in claim_clean
            or "without " in claim_clean
        )

        for chunk_id in claim.supporting_chunk_ids:
            chunk = self.corpus_chunks.get(chunk_id)
            if not chunk:
                continue
            valid_chunk_found = True
            content = chunk.get("content", "").lower()

            # Check if claim relates to prior work / background
            prior_work_markers = (
                "prior work",
                "prior works",
                "previous work",
                "existing method",
                "existing work",
            )
            has_prior_work_marker = any(pwm in content for pwm in prior_work_markers) or any(
                any(pwm in q.lower() for pwm in prior_work_markers) for q in claim.supporting_quotes
            )
            if has_prior_work_marker and (
                "we propose" in claim_clean
                or "our work" in claim_clean
                or "we introduce" in claim_clean
            ):
                background_attribution_conflict = True

            # Check each quote for this claim
            for quote in claim.supporting_quotes:
                q_clean = quote.strip().lower()
                if not q_clean:
                    continue

                quote_has_not = (
                    " not " in f" {q_clean} "
                    or "n't " in q_clean
                    or " never " in q_clean
                    or "abandon" in q_clean
                    or "without " in q_clean
                )
                if claim_has_not != quote_has_not:
                    negation_mismatch = True

                if q_clean in content:
                    aligned_quote_found = True
                    break
                else:
                    # Require sequential n-gram match, not just bag-of-words
                    words = [w for w in q_clean.split() if len(w) > 3]
                    if len(words) >= 3:
                        found_ngram = False
                        for i in range(len(words) - 2):
                            ngram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
                            if ngram in content:
                                found_ngram = True
                                break
                        if found_ngram:
                            aligned_quote_found = True
                            break
                        else:
                            # bag-of-words check for partial overlap only
                            overlap = sum(1 for k in words if k in content) / len(words)
                            if overlap >= 0.4:
                                has_partial_overlap = True

            if aligned_quote_found:
                break

        if not valid_chunk_found:
            claim.support_status = ClaimSupportStatus.UNSUPPORTED
            claim.verification_notes = "Cited chunk IDs do not exist in corpus."
        elif negation_mismatch:
            claim.support_status = ClaimSupportStatus.UNSUPPORTED
            claim.verification_notes = (
                "Negation mismatch detected between claim and source evidence."
            )
        elif background_attribution_conflict:
            claim.support_status = ClaimSupportStatus.UNSUPPORTED
            claim.verification_notes = (
                "Prior-work method erroneously claimed as current paper contribution."
            )
        elif aligned_quote_found:
            claim.support_status = ClaimSupportStatus.SUPPORTED
            claim.verification_notes = "Claim quote directly verified against cited chunk content."
        elif has_partial_overlap:
            claim.support_status = ClaimSupportStatus.PARTIALLY_SUPPORTED
            claim.verification_notes = "Chunk exists and has partial topical overlap, but exact phrase/meaning not verified."
        else:
            claim.support_status = ClaimSupportStatus.UNSUPPORTED
            claim.verification_notes = "Claim content is not substantiated by cited chunk text."

        return claim.support_status

    def validate_answer(self, answer_obj: SynthesizedAnswerOutput) -> SynthesizedAnswerOutput:
        """Validate all citations, claims, and overall grounding status of a synthesized answer."""
        if answer_obj.abstained:
            answer_obj.grounding_status = GroundingStatus.ABSTAINED
            return answer_obj

        # 1. Validate each citation
        val_details: list[CitationValidationDetail] = []
        cits_by_id: dict[str, CitationRecord] = {}
        for c in answer_obj.citations:
            detail = self.validate_citation(c)
            val_details.append(detail)
            cits_by_id[c.chunk_id] = c
        answer_obj.validation_details = val_details

        # 2. Validate each claim
        unsupported: list[str] = []
        for clm in answer_obj.claims:
            status = self.validate_claim(clm, cits_by_id)
            if status == ClaimSupportStatus.UNSUPPORTED:
                unsupported.append(clm.claim_text)
        answer_obj.unsupported_claims = unsupported

        # 3. Determine overall grounding status
        all_cits_exist = bool(val_details and all(v.exists_in_corpus for v in val_details))
        all_quotes_aligned = bool(
            val_details and all(v.quote_aligned_with_chunk for v in val_details)
        )
        has_unsupported_claims = bool(unsupported)

        if all_cits_exist and all_quotes_aligned and not has_unsupported_claims:
            answer_obj.grounding_status = GroundingStatus.GROUNDED
        elif (
            any(v.exists_in_corpus for v in val_details)
            and not all_cits_exist
            or has_unsupported_claims
        ):
            answer_obj.grounding_status = GroundingStatus.PARTIALLY_GROUNDED
        else:
            answer_obj.grounding_status = GroundingStatus.UNGROUNDED

        return answer_obj
