"""Phase K Query Entity & Intent Analysis Component.

Performs lightweight, deterministic analysis of queries to identify:
1. Target paper mentions (e.g. arXiv IDs like paper-arxiv-2402.11794, 2601.05254).
2. Method, model, and concept mentions.
3. Comparative or multi-part intent (e.g. "compare", "contrast", "versus", "vs").
4. Required evidence targets (each side of a comparison or multi-paper query).
5. Multi-paper requirement flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Match arXiv IDs in various formats: paper-arxiv-2401.001, arxiv:2401.001, 2401.001, etc.
ARXIV_ID_PATTERN = re.compile(r"(?:paper-)?(?:arxiv[:\-])?(\d{4}\.\d{4,5})", re.IGNORECASE)

COMPARATIVE_KEYWORDS = {
    "compare",
    "contrast",
    "versus",
    "vs",
    "vs.",
    "differ",
    "difference",
    "differences",
    "between",
}

MULTI_PART_KEYWORDS = {
    "both",
    "collective",
    "across",
    "multiple",
    "how do",
    "how does",
}


@dataclass
class QueryAnalysisResult:
    """Structured analysis of query intent and evidence requirements."""

    query: str
    target_paper_ids: list[str] = field(default_factory=list)
    target_methods: list[str] = field(default_factory=list)
    is_comparative: bool = False
    is_multi_paper: bool = False
    required_evidence_targets: list[str] = field(
        default_factory=list
    )  # e.g. list of paper_ids required


class QueryIntentAnalyzer:
    """Deterministic analyzer for scientific RAG queries."""

    def analyze(self, query: str) -> QueryAnalysisResult:
        """Analyze query to extract required paper entities and comparison structure."""
        lower_q = query.lower()

        # 1. Extract paper targets (normalized canonical format 'paper-arxiv-XXXX.YYYYY')
        raw_matches = ARXIV_ID_PATTERN.findall(query)
        seen_papers = set()
        paper_ids: list[str] = []
        for m in raw_matches:
            canonical_pid = f"paper-arxiv-{m}"
            if canonical_pid not in seen_papers:
                seen_papers.add(canonical_pid)
                paper_ids.append(canonical_pid)

        # 2. Check comparative intent
        tokens = set(re.findall(r"\b[a-z]+\b", lower_q))
        has_comp_kw = bool(tokens.intersection(COMPARATIVE_KEYWORDS))
        is_comparative = has_comp_kw or ("and" in tokens and len(paper_ids) >= 2)

        # 3. Check multi-paper intent
        is_multi_paper = len(paper_ids) > 1 or bool(tokens.intersection(MULTI_PART_KEYWORDS))

        # 4. Required evidence targets: if papers are specified, all of them are required
        required_targets = list(paper_ids)

        return QueryAnalysisResult(
            query=query,
            target_paper_ids=paper_ids,
            is_comparative=is_comparative,
            is_multi_paper=is_multi_paper,
            required_evidence_targets=required_targets,
        )
