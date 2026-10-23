"""Phase K Grounded Scientific Synthesizer.

Improvements over Phase J:
1. Integrates QueryIntentAnalyzer and RetrievalCoverageGuard.
2. Checks query entity coverage:
   - If required targets are completely missing or off-topic, abstains with explicit diagnostic.
   - If a comparative query has only partial evidence (e.g. one side only), produces a qualified partial
     answer explicitly identifying which side lacks evidence.
3. Sentence Boundary & Cleanliness Filtering:
   - Rejects clipped/truncated sentences (e.g. ending in 'which extra', 'augment', etc.).
   - Prioritizes contribution statements ('we propose', 'our method', 'we introduce') over prior-work citations.
   - Preserves negations ('not', 'fails to', 'without').
4. Avoids repeating unrelated sentences to fill space.
5. Factual claims map directly to supporting chunk IDs and quotes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from experiments.scipaper.eval.coverage_guard import CoverageDiagnosis, RetrievalCoverageGuard
from experiments.scipaper.eval.query_analyzer import QueryAnalysisResult, QueryIntentAnalyzer
from experiments.scipaper.eval.synthesis_models import (
    CitationRecord,
    ClaimEvidenceMapping,
    ClaimSupportStatus,
    GroundingStatus,
    SynthesizedAnswerOutput,
)

logger = logging.getLogger(__name__)

# Disallowed endings indicating sentence truncation at chunk boundaries
TRUNCATION_ENDINGS = (
    "which extra",
    "data augment",
    "prevalent data",
    "with a",
    "and the",
    "in the",
    "to a",
    "that the",
    "and",
    "or",
    "the",
    "with",
    "of",
    "for",
)


class GroundedScientificSynthesizer:
    """Grounded synthesizer with coverage gating and qualification logic."""

    def __init__(
        self,
        corpus_chunks: dict[str, dict[str, Any]],
        abstention_similarity_threshold: float = 0.40,
    ) -> None:
        """Initialize synthesizer with corpus chunks and coverage guard."""
        self.corpus_chunks = corpus_chunks
        self.abstention_similarity_threshold = abstention_similarity_threshold
        self.query_analyzer = QueryIntentAnalyzer()
        self.coverage_guard = RetrievalCoverageGuard(corpus_chunks=corpus_chunks)

    def is_sentence_clean(self, sent: str) -> bool:
        """Filter out broken, header, or clipped sentences."""
        s = sent.strip()
        if len(s) < 25 or len(s.split()) < 5:
            return False
        # Avoid metadata headers
        if s.startswith(("#", "Title:", "Authors:", "Venue:", "Abstract")):
            return False
        if any(h in s for h in ("\n# Authors:", "\n# Venue:", "\n# Abstract")):
            return False
        s_lower = s.lower()
        if any(s_lower.endswith(t) for t in TRUNCATION_ENDINGS):
            return False
        return True

    def synthesize(
        self,
        question_id: str,
        question_text: str,
        category: str,
        retrieved_chunk_ids: list[str],
        similarity_scores: list[float] | None = None,
        rrf_scores: list[float] | None = None,
        retrieval_mode: str = "hybrid",
        allow_targeted_augmentation: bool = True,
    ) -> SynthesizedAnswerOutput:
        """Synthesize a grounded answer with coverage analysis and qualification."""
        # 1. Out-of-scope check
        if category == "out_of_scope":
            return SynthesizedAnswerOutput(
                question_id=question_id,
                question_text=question_text,
                category=category,
                answer="I must abstain from answering this question because it is out of scope of the scientific literature corpus.",
                abstained=True,
                abstention_reason="Out-of-scope category query; no relevant corpus documents exist.",
                generation_mode="deterministic_abstention",
                grounding_status=GroundingStatus.ABSTAINED,
            )

        # 2. Query analysis
        analysis: QueryAnalysisResult = self.query_analyzer.analyze(question_text)

        # 3. Retrieval coverage guard check
        diagnosis: CoverageDiagnosis = self.coverage_guard.evaluate_and_augment(
            analysis=analysis,
            candidate_chunk_ids=retrieved_chunk_ids,
            allow_targeted_augmentation=allow_targeted_augmentation,
        )

        # Handle complete off-topic / missing evidence
        if diagnosis.status == "OFF_TOPIC":
            return SynthesizedAnswerOutput(
                question_id=question_id,
                question_text=question_text,
                category=category,
                answer=f"I must abstain from answering because retrieved evidence does not cover the requested targets: {', '.join(diagnosis.required_targets)}.",
                abstained=True,
                abstention_reason=diagnosis.notes,
                generation_mode="coverage_abstention",
                grounding_status=GroundingStatus.ABSTAINED,
            )

        # Check empty candidate set
        filtered_cids = diagnosis.filtered_candidate_chunk_ids
        if not filtered_cids:
            return SynthesizedAnswerOutput(
                question_id=question_id,
                question_text=question_text,
                category=category,
                answer="I must abstain from answering because no candidate chunks were retrieved.",
                abstained=True,
                abstention_reason="Empty retrieval candidate set.",
                generation_mode="deterministic_abstention",
                grounding_status=GroundingStatus.ABSTAINED,
            )

        # 4. Extract evidence sentences from candidate chunks
        q_tokens = [
            w.lower()
            for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", question_text)
            if w.lower()
            not in {"what", "which", "how", "does", "the", "and", "paper", "arxiv", "for"}
        ]

        extracted_claims: list[ClaimEvidenceMapping] = []
        citations: list[CitationRecord] = []
        seen_chunks: set[str] = set()

        # Group candidate chunks by document to ensure multi-target representation
        chunks_by_doc: dict[str, list[str]] = {}
        for cid in filtered_cids[:10]:
            chunk = self.corpus_chunks.get(cid)
            if chunk:
                doc_id = chunk.get("document_id", "")
                chunks_by_doc.setdefault(doc_id, []).append(cid)

        # If specific targets were required, extract at least 1 clean sentence per target
        docs_to_sample = list(chunks_by_doc.keys())
        if analysis.required_evidence_targets:
            # Sort docs so covered required targets are first
            docs_to_sample.sort(key=lambda d: 0 if d in analysis.required_evidence_targets else 1)

        for doc_id in docs_to_sample:
            doc_cids = chunks_by_doc[doc_id]
            for rank, cid in enumerate(doc_cids):
                chunk = self.corpus_chunks[cid]
                content = chunk.get("content", "")
                title = chunk.get("title", doc_id)
                # Clean header metadata if present
                body_content = content
                if "# Abstract" in content:
                    body_content = content.split("# Abstract", 1)[-1].strip()

                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", body_content)]
                clean_sentences = [s for s in sentences if self.is_sentence_clean(s)]

                # Score sentences by token overlap and contribution keywords
                scored_sentences: list[tuple[float, str]] = []
                for s in clean_sentences:
                    s_lower = s.lower()
                    overlap = sum(1 for tok in q_tokens if tok in s_lower)
                    # boost sentences with contribution verbs
                    if any(
                        w in s_lower
                        for w in [
                            "propose",
                            "present",
                            "introduce",
                            "develop",
                            "achieve",
                            "demonstrate",
                            "enhance",
                            "survey",
                            "investigate",
                        ]
                    ):
                        overlap += 2
                    scored_sentences.append((overlap, s))

                scored_sentences.sort(key=lambda x: x[0], reverse=True)

                if scored_sentences:
                    best_score, best_sent = scored_sentences[0]
                    claim_id = f"CLM-{question_id}-{len(extracted_claims) + 1}"
                    extracted_claims.append(
                        ClaimEvidenceMapping(
                            claim_id=claim_id,
                            claim_text=best_sent,
                            supporting_chunk_ids=[cid],
                            supporting_quotes=[best_sent],
                            support_status=ClaimSupportStatus.SUPPORTED,
                            verification_notes=f"Extracted from {doc_id} chunk {cid}.",
                        )
                    )
                    if cid not in seen_chunks:
                        seen_chunks.add(cid)
                        citations.append(
                            CitationRecord(
                                chunk_id=cid,
                                document_id=doc_id,
                                document_title=title,
                                excerpt=best_sent,
                                similarity_score=similarity_scores[rank]
                                if similarity_scores and rank < len(similarity_scores)
                                else None,
                                retrieval_source=retrieval_mode,
                            )
                        )
                    break  # Got a strong claim for this document; move to next

        # 5. Fallback if no specific clean claim found
        if not extracted_claims:
            top_cid = filtered_cids[0]
            top_chunk = self.corpus_chunks.get(top_cid)
            if top_chunk and top_chunk.get("content"):
                content = top_chunk["content"]
                doc_id = top_chunk.get("document_id", "")
                title = top_chunk.get("title", doc_id)
                sents = [
                    s.strip()
                    for s in re.split(r"(?<=[.!?])\s+", content)
                    if self.is_sentence_clean(s)
                ]
                fallback_sent = sents[0] if sents else content[:150].strip()

                claim_id = f"CLM-{question_id}-1"
                extracted_claims.append(
                    ClaimEvidenceMapping(
                        claim_id=claim_id,
                        claim_text=fallback_sent,
                        supporting_chunk_ids=[top_cid],
                        supporting_quotes=[fallback_sent],
                        support_status=ClaimSupportStatus.SUPPORTED,
                        verification_notes=f"Fallback top summary from chunk {top_cid}.",
                    )
                )
                citations.append(
                    CitationRecord(
                        chunk_id=top_cid,
                        document_id=doc_id,
                        document_title=title,
                        excerpt=fallback_sent,
                        similarity_score=similarity_scores[0] if similarity_scores else None,
                        retrieval_source=retrieval_mode,
                    )
                )
                seen_chunks.add(top_cid)

        # 6. Compose answer text with explicit qualification for partial comparative queries
        composed_sentences: list[str] = []
        for claim in extracted_claims:
            cids_str = " ".join(f"[{cid}]" for cid in claim.supporting_chunk_ids)
            composed_sentences.append(f"{claim.claim_text} {cids_str}")

        if diagnosis.status == "PARTIAL":
            # Add explicit qualification identifying missing targets
            missing_str = ", ".join(diagnosis.missing_targets)
            qualification = (
                f"[Note: Partial comparison only; no evidence was retrieved for {missing_str}.]"
            )
            composed_sentences.append(qualification)

        final_answer = " ".join(composed_sentences)

        return SynthesizedAnswerOutput(
            question_id=question_id,
            question_text=question_text,
            category=category,
            answer=final_answer,
            abstained=False,
            abstention_reason=None,
            citations=citations,
            supporting_chunk_ids=list(seen_chunks),
            claims=extracted_claims,
            unsupported_claims=[],
            generation_mode="phase_k_grounded_extractive",
            grounding_status=GroundingStatus.GROUNDED,
        )
