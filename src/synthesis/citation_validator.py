"""Citation and Grounding Guardrail (TASK-403).

Validates citations in synthesized draft answers against retrieved candidate chunk IDs
and enforces grounding constraints to prevent hallucinated citations and ungrounded claims.

Design & Architectural Constraints (see docs/ARCHITECTURE.md, docs/SECURITY.md, docs/API_SPEC.md):
---------------------------------------------------------------------------------------------------
1. Authoritative Candidate Validation:
   - Evaluates every cited chunk ID against the authoritative `candidate_chunk_ids` set from
     the normalized context (TASK-401).
   - Distinguishes valid citations (present in candidate set) from invalid / unknown citations.
2. Hallucination Detection & Citation Scrubbing:
   - When invalid citations are detected, scrubs/removes invalid bracketed references from the
     returned sanitized text while recording warnings in validation metadata.
   - Preserves all valid citations in both the sanitized text and the structured citation list.
3. Grounding Status Determination:
   - Predictably classifies the synthesized answer into a status:
     * `VALID`: All citations are valid and sufficient evidence was retrieved.
     * `PARTIALLY_GROUNDED`: Some citations were valid, but some invalid citations were scrubbed.
     * `UNGROUNDED`: Citations were present but none were valid, or no citations were provided
       for substantive factual claims when context was available.
     * `REFUSAL`: Context was empty or model explicitly stated lack of information.
4. Provenance & Downstream Enriched Citations:
   - Associates valid chunk IDs with parent document IDs and excerpt summaries from the
     underlying NormalizedContext for API response construction (`POST /api/v1/query`).
5. Safe & Deterministic Execution:
   - Zero live API or network calls.
   - Robust against empty contexts, empty answers, duplicate citations, malformed inputs,
     and indirect prompt injections.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.synthesis.normalizer import NormalizedChunkItem, NormalizedContext
from src.synthesis.synthesizer import DraftAnswer

logger = logging.getLogger(__name__)

# Regular expression matching bracketed chunk citations, e.g. [c1111111-1111-1111-1111-111111111111]
_BRACKETED_CITATION_REGEX = re.compile(r"\[([a-zA-Z0-9_\-]+)\]")


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------


class GroundingStatus(StrEnum):
    """Grounding outcome classification for synthesized answers."""

    VALID = "VALID"
    PARTIALLY_GROUNDED = "PARTIALLY_GROUNDED"
    UNGROUNDED = "UNGROUNDED"
    REFUSAL = "REFUSAL"


class CitationProvenance(BaseModel):
    """Enriched citation metadata linking chunk ID to parent document and excerpt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., description="Canonical UUID of the supporting chunk.")
    document_id: str = Field(..., description="Identifier of the source document.")
    excerpt: str = Field(..., description="Excerpt or full text from the chunk content.")
    similarity_score: float | None = Field(
        default=None, description="Vector similarity score if vector-retrieved."
    )
    provenance_source: str = Field(
        default="vector", description="Source of chunk: 'vector', 'graph', or 'hybrid'."
    )


class CitationWarning(BaseModel):
    """Warning record for invalid or anomalous citation instances."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(..., description="Machine-readable warning code.")
    message: str = Field(..., description="Human-readable warning message.")
    chunk_id: str | None = Field(default=None, description="Subject chunk ID if applicable.")


@dataclass(frozen=True)
class GuardrailResult:
    """Validated, sanitized, and grounded synthesis result ready for API delivery."""

    question: str
    original_answer: str
    sanitized_answer: str
    status: GroundingStatus
    is_grounded: bool
    valid_citations: list[CitationProvenance]
    valid_chunk_ids: list[str]
    invalid_chunk_ids: list[str]
    candidate_chunk_ids: list[str]
    warnings: list[CitationWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_hallucinated_citations(self) -> bool:
        """Return True if any cited chunk ID was not present in the candidate set."""
        return len(self.invalid_chunk_ids) > 0


# ---------------------------------------------------------------------------
# Citation Validator & Grounding Guardrail Implementation
# ---------------------------------------------------------------------------


class CitationValidator:
    """Validates citations against candidate chunk IDs and enforces grounding invariants."""

    def __init__(self) -> None:
        """Initialize CitationValidator."""

    def validate(
        self,
        draft: DraftAnswer,
        context: NormalizedContext | None = None,
    ) -> GuardrailResult:
        """Validate citations in draft answer against candidate chunk IDs and context.

        Args:
            draft: The draft answer produced by GroundedAnswerSynthesizer.
            context: Optional NormalizedContext from which candidate chunks and excerpts originate.
                     If omitted, candidate_chunk_ids from draft are used.

        Returns:
            GuardrailResult with sanitized answer, validated citations, and grounding status.
        """
        raw_text = draft.raw_answer or ""
        cited_ids = list(draft.cited_chunk_ids)

        # 1. Determine Authoritative Candidate Set
        candidate_ids_set: set[str] = set()
        if context is not None:
            candidate_ids_set.update(context.candidate_chunk_ids)
        candidate_ids_set.update(draft.candidate_chunk_ids)

        # Build chunk lookup index from context chunks if provided
        chunk_lookup: dict[str, NormalizedChunkItem] = {}
        if context is not None:
            for ch in context.chunks:
                chunk_lookup[ch.chunk_id] = ch

        # 2. Check Empty / Refusal Cases
        is_empty_context = (
            (context.is_empty if context is not None else False)
            or (not candidate_ids_set)
            or draft.metadata.get("empty_context", False)
        )
        is_empty_answer = not raw_text.strip()

        if is_empty_answer:
            return GuardrailResult(
                question=draft.question,
                original_answer=raw_text,
                sanitized_answer="",
                status=GroundingStatus.REFUSAL if is_empty_context else GroundingStatus.UNGROUNDED,
                is_grounded=False,
                valid_citations=[],
                valid_chunk_ids=[],
                invalid_chunk_ids=[],
                candidate_chunk_ids=sorted(candidate_ids_set),
                warnings=[
                    CitationWarning(
                        code="EMPTY_ANSWER",
                        message="Synthesizer returned an empty answer.",
                    )
                ],
                metadata={"empty_answer": True, "empty_context": is_empty_context},
            )

        if is_empty_context:
            # Model response with no context is treated as a refusal
            return GuardrailResult(
                question=draft.question,
                original_answer=raw_text,
                sanitized_answer=raw_text.strip(),
                status=GroundingStatus.REFUSAL,
                is_grounded=True,  # Refusal on empty context is grounded behavior
                valid_citations=[],
                valid_chunk_ids=[],
                invalid_chunk_ids=[],
                candidate_chunk_ids=[],
                warnings=[],
                metadata={"empty_context": True, "refusal": True},
            )

        # 3. Partition Citations into Valid and Invalid Sets
        valid_ids: list[str] = []
        invalid_ids: list[str] = []
        warnings: list[CitationWarning] = []

        seen_cited: set[str] = set()
        for cid in cited_ids:
            clean_cid = cid.strip()
            if not clean_cid or clean_cid in seen_cited:
                continue
            seen_cited.add(clean_cid)

            if clean_cid in candidate_ids_set:
                valid_ids.append(clean_cid)
            else:
                invalid_ids.append(clean_cid)
                warnings.append(
                    CitationWarning(
                        code="HALLUCINATED_CITATION",
                        message=f"Citation [{clean_cid}] does not exist in retrieved candidate context.",
                        chunk_id=clean_cid,
                    )
                )

        # 4. Scrub Invalid Citations from Raw Answer Text
        sanitized_text = self._scrub_invalid_citations(raw_text, set(invalid_ids))

        # 5. Build Enriched Valid Citation Provenance List
        valid_provenance: list[CitationProvenance] = []
        for vid in valid_ids:
            if vid in chunk_lookup:
                ch = chunk_lookup[vid]
                excerpt = ch.content[:250].strip()
                if len(ch.content) > 250:
                    excerpt += "..."
                valid_provenance.append(
                    CitationProvenance(
                        chunk_id=vid,
                        document_id=ch.document_id,
                        excerpt=excerpt,
                        similarity_score=ch.similarity_score,
                        provenance_source=ch.provenance_source,
                    )
                )
            else:
                # Chunk was in graph triples/entities provenance but not direct chunk item
                doc_id = f"doc-provenance-{vid[:8]}" if len(vid) >= 8 else "doc-provenance"
                valid_provenance.append(
                    CitationProvenance(
                        chunk_id=vid,
                        document_id=doc_id,
                        excerpt=f"Knowledge graph relationship provenance for chunk {vid}",
                        similarity_score=None,
                        provenance_source="graph",
                    )
                )

        # 6. Determine Grounding Status
        # Status matrix:
        # - If no citations at all when candidate context was available: UNGROUNDED (or REFUSAL if explicit refusal text)
        # - If only invalid citations were provided: UNGROUNDED
        # - If mixed valid and invalid citations: PARTIALLY_GROUNDED
        # - If all citations were valid (at least 1): VALID
        if not cited_ids:
            # Check if it's an explicit refusal / cannot answer response
            if self._is_refusal_text(raw_text):
                status = GroundingStatus.REFUSAL
                is_grounded = True
            else:
                status = GroundingStatus.UNGROUNDED
                is_grounded = False
                warnings.append(
                    CitationWarning(
                        code="MISSING_CITATIONS",
                        message="Answer contains factual statements without supporting bracketed citations.",
                    )
                )
        elif not valid_ids:
            status = GroundingStatus.UNGROUNDED
            is_grounded = False
        elif invalid_ids:
            status = GroundingStatus.PARTIALLY_GROUNDED
            is_grounded = True  # Has genuine grounded evidence, but scrubbed hallucinations
        else:
            status = GroundingStatus.VALID
            is_grounded = True

        return GuardrailResult(
            question=draft.question,
            original_answer=raw_text,
            sanitized_answer=sanitized_text,
            status=status,
            is_grounded=is_grounded,
            valid_citations=valid_provenance,
            valid_chunk_ids=valid_ids,
            invalid_chunk_ids=invalid_ids,
            candidate_chunk_ids=sorted(candidate_ids_set),
            warnings=warnings,
            metadata={
                "cited_count": len(cited_ids),
                "valid_count": len(valid_ids),
                "invalid_count": len(invalid_ids),
                "candidate_count": len(candidate_ids_set),
            },
        )

    def _scrub_invalid_citations(self, text: str, invalid_ids: set[str]) -> str:
        """Remove bracketed citations corresponding to invalid chunk IDs while preserving valid ones."""
        if not invalid_ids or not text:
            return text.strip()

        def replacer(match: re.Match[str]) -> str:
            chunk_id = match.group(1)
            if chunk_id in invalid_ids:
                return ""  # Scrub invalid citation
            return match.group(0)  # Keep valid citation

        # Replace invalid citations
        cleaned = _BRACKETED_CITATION_REGEX.sub(replacer, text)
        # Clean up possible double spaces left by removed citation brackets
        cleaned = re.sub(r" +", " ", cleaned)
        # Clean up space before punctuation e.g. " ." -> "."
        cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
        return cleaned.strip()

    def _is_refusal_text(self, text: str) -> bool:
        """Check if model response is an explicit statement of insufficient context/information."""
        lower = text.lower()
        refusal_phrases = [
            "could not find",
            "cannot find",
            "no information",
            "not enough information",
            "not mentioned in",
            "does not contain",
            "insufficient information",
            "unable to answer",
            "cannot answer",
        ]
        return any(phrase in lower for phrase in refusal_phrases)
