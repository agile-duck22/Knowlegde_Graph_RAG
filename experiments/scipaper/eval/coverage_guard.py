"""Phase K Retrieval Coverage Guard.

Evaluates whether retrieved candidates satisfy query entity requirements:
1. Checks whether candidates contain evidence for all required query entities (e.g. all sides of a comparison).
2. Filters out candidates that are completely topically unrelated to the requested entities.
3. Triggers targeted retrieval when a required entity is missing from candidate set.
4. Avoids duplicate chunks and preserves provenance.
5. Records coverage diagnosis (COMPLETE, PARTIAL, MISSING).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from experiments.scipaper.eval.query_analyzer import QueryAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class CoverageDiagnosis:
    """Outcome of retrieval coverage check."""

    status: str  # COMPLETE, PARTIAL, OFF_TOPIC, UNRESTRICTED
    required_targets: list[str]
    covered_targets: list[str]
    missing_targets: list[str]
    filtered_candidate_chunk_ids: list[str] = field(default_factory=list)
    targeted_retrieval_applied: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


class RetrievalCoverageGuard:
    """Guard component ensuring candidate sets provide evidence for all query targets."""

    def __init__(self, corpus_chunks: dict[str, dict[str, Any]]) -> None:
        """Initialize guard with corpus chunk mapping."""
        self.corpus_chunks = corpus_chunks
        # Map document_id -> list of chunk_ids
        self.doc_to_chunks: dict[str, list[str]] = {}
        for cid, c in corpus_chunks.items():
            doc_id = c.get("document_id", "")
            if doc_id:
                self.doc_to_chunks.setdefault(doc_id, []).append(cid)

    def evaluate_and_augment(
        self,
        analysis: QueryAnalysisResult,
        candidate_chunk_ids: list[str],
        allow_targeted_augmentation: bool = True,
    ) -> CoverageDiagnosis:
        """Evaluate coverage of candidates against analysis requirements and augment if needed."""
        required = analysis.required_evidence_targets

        if not required:
            # Query does not target specific paper IDs; all candidates permitted
            return CoverageDiagnosis(
                status="UNRESTRICTED",
                required_targets=[],
                covered_targets=[],
                missing_targets=[],
                filtered_candidate_chunk_ids=list(candidate_chunk_ids),
                targeted_retrieval_applied=False,
                notes="Query has no explicit document targets; general semantic retrieval.",
            )

        # 1. Filter candidates to prioritize requested document targets
        targeted_candidates = []
        unrelated_candidates = []
        seen_targets = set()

        for cid in candidate_chunk_ids:
            chunk = self.corpus_chunks.get(cid)
            if not chunk:
                continue
            doc_id = chunk.get("document_id", "")
            if doc_id in required:
                targeted_candidates.append(cid)
                seen_targets.add(doc_id)
            else:
                unrelated_candidates.append(cid)

        # 2. Check for missing targets
        missing = [t for t in required if t not in seen_targets]
        targeted_applied = False

        # 3. Targeted Retrieval: If missing targets and augmentation allowed, fetch top chunks for missing docs
        final_candidates = list(targeted_candidates)

        if missing and allow_targeted_augmentation:
            for m_doc in missing:
                available_chunks = self.doc_to_chunks.get(m_doc, [])
                if available_chunks:
                    # Add up to 2 chunks for this missing document
                    for add_cid in available_chunks[:2]:
                        if add_cid not in final_candidates:
                            final_candidates.append(add_cid)
                            seen_targets.add(m_doc)
                            targeted_applied = True

            missing = [t for t in required if t not in seen_targets]

        # If after targeted search we still have candidates, keep them
        # Only fall back to unrelated candidates if zero targeted candidates exist
        if not final_candidates:
            # All candidates were off-topic!
            return CoverageDiagnosis(
                status="OFF_TOPIC",
                required_targets=required,
                covered_targets=[],
                missing_targets=required,
                filtered_candidate_chunk_ids=unrelated_candidates,
                targeted_retrieval_applied=targeted_applied,
                notes=f"Retrieved candidates do not cover any requested targets: {required}.",
            )

        covered = [t for t in required if t in seen_targets]

        if not missing:
            status = "COMPLETE"
            notes = "All required query targets covered in candidates."
        else:
            status = "PARTIAL"
            notes = (
                f"Partially covered targets ({len(covered)}/{len(required)}). Missing: {missing}."
            )

        return CoverageDiagnosis(
            status=status,
            required_targets=required,
            covered_targets=covered,
            missing_targets=missing,
            filtered_candidate_chunk_ids=final_candidates,
            targeted_retrieval_applied=targeted_applied,
            notes=notes,
        )
