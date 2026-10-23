"""Phase J Data Models: Synthesis, Claims, Citations, and Validation Schemas.

Defines schemas for:
- ClaimEvidenceMapping: Atomic factual claims mapped to citations and exact quotes.
- CitationRecord: Structured citation referencing chunk ID, document ID, title, and excerpt.
- SynthesizedAnswerOutput: Complete structured answer response with abstention and claim provenance.
- ValidationResults: Results of citation existence, metadata, alignment, and semantic support checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class GroundingStatus(StrEnum):
    """Grounding outcome classification for synthesized answer."""

    GROUNDED = "GROUNDED"
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"
    UNGROUNDED = "UNGROUNDED"
    ABSTAINED = "ABSTAINED"


class ClaimSupportStatus(StrEnum):
    """Status of an individual factual claim."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class CitationRecord:
    """Individual citation attached to an answer or claim."""

    chunk_id: str
    document_id: str
    document_title: str
    excerpt: str
    similarity_score: float | None = None
    retrieval_source: str = "hybrid"  # vector, graph, or hybrid

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimEvidenceMapping:
    """An atomic factual statement extracted or composed in the answer, with supporting evidence."""

    claim_id: str
    claim_text: str
    supporting_chunk_ids: list[str]
    supporting_quotes: list[str]
    support_status: ClaimSupportStatus = ClaimSupportStatus.UNSUPPORTED
    verification_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["support_status"] = self.support_status.value
        return res


@dataclass
class CitationValidationDetail:
    """Detailed validation results for a citation."""

    chunk_id: str
    exists_in_corpus: bool
    document_id_matches: bool
    metadata_complete: bool
    quote_aligned_with_chunk: bool
    semantic_relationship_supported: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SynthesizedAnswerOutput:
    """Complete structured answer output for a benchmark question."""

    question_id: str
    question_text: str
    category: str
    answer: str
    abstained: bool
    abstention_reason: str | None
    citations: list[CitationRecord] = field(default_factory=list)
    supporting_chunk_ids: list[str] = field(default_factory=list)
    claims: list[ClaimEvidenceMapping] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    generation_mode: str = "deterministic_offline_extractive"
    grounding_status: GroundingStatus = GroundingStatus.GROUNDED
    validation_details: list[CitationValidationDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        res = asdict(self)
        res["grounding_status"] = self.grounding_status.value
        res["citations"] = [c.to_dict() for c in self.citations]
        res["claims"] = [c.to_dict() for c in self.claims]
        res["validation_details"] = [v.to_dict() for v in self.validation_details]
        return res
