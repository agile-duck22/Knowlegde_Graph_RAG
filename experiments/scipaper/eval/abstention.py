"""Local abstention and refusal evaluation experiment (Phase E, Task 5).

Evaluates whether a local cosine-similarity threshold can act as an abstention gate
on out-of-scope queries versus answerable queries.
Labels threshold selection as exploratory without claiming factual correctness guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AbstentionEvaluationResult:
    """Outcome of local abstention filter across benchmark."""

    threshold: float
    total_answerable: int
    total_out_of_scope: int
    answerable_refusals: int  # False refusals
    answerable_pass: int
    out_of_scope_refusals: int  # Correct refusals
    out_of_scope_pass: int  # Missed refusals (false positives)
    refusal_precision: float
    refusal_recall: float
    answerable_availability: float


def evaluate_local_abstention(
    evaluations: list[dict[str, Any]],
    thresholds: list[float] | None = None,
) -> list[AbstentionEvaluationResult]:
    """Evaluate abstention behavior across candidate similarity thresholds."""
    if thresholds is None:
        thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    results: list[AbstentionEvaluationResult] = []

    answerable = [e for e in evaluations if e["category"] != "out_of_scope"]
    out_of_scope = [e for e in evaluations if e["category"] == "out_of_scope"]

    for th in thresholds:
        # Check decision for answerable queries
        false_refusals = 0
        for e in answerable:
            top_sim = e.get("top_similarity", 0.0)
            if top_sim < th:
                false_refusals += 1

        # Check decision for out-of-scope queries
        correct_refusals = 0
        for e in out_of_scope:
            top_sim = e.get("top_similarity", 0.0)
            if top_sim < th:
                correct_refusals += 1

        missed_refusals = len(out_of_scope) - correct_refusals
        pass_answerable = len(answerable) - false_refusals

        total_refusals = correct_refusals + false_refusals
        prec = (correct_refusals / total_refusals) if total_refusals > 0 else 0.0
        rec = (correct_refusals / len(out_of_scope)) if out_of_scope else 0.0
        avail = (pass_answerable / len(answerable)) if answerable else 0.0

        results.append(
            AbstentionEvaluationResult(
                threshold=th,
                total_answerable=len(answerable),
                total_out_of_scope=len(out_of_scope),
                answerable_refusals=false_refusals,
                answerable_pass=pass_answerable,
                out_of_scope_refusals=correct_refusals,
                out_of_scope_pass=missed_refusals,
                refusal_precision=round(prec, 4),
                refusal_recall=round(rec, 4),
                answerable_availability=round(avail, 4),
            )
        )

    return results
