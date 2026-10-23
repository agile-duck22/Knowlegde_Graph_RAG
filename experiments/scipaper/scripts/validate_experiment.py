"""Offline validation CLI and utility functions for scientific paper experiment data (Phase A).

Validates:
1. Manifest records (PaperManifestEntry): unique paper IDs, licensing, source URLs, content checksums.
2. Corpus documents (RawDocument / ScientificPaperDocument): unique document IDs, non-empty fields,
   matching SHA-256 content hashes, and coverage count invariants (100–150 for completed experiment).
3. 50-question benchmark dataset (ScientificBenchmarkDataset): valid schema, globally unique IDs,
   exact category breakdown (10 direct fact, 10 2-hop, 10 3-hop, 10 aggregation, 10 out of scope),
   and cross-referencing against the source manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from experiments.scipaper.schemas.models import (
    PaperManifestEntry,
    ScientificBenchmarkDataset,
)
from src.eval.schema import BenchmarkCategory
from src.ingest.loader import DocumentLoader, RawDocument

logger = logging.getLogger("scipaper_validator")


class ValidationReport:
    """Tracks validation findings, errors, and warnings."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_manifest(
    manifest_path: Path,
    expected_count_range: tuple[int, int] | None = None,
) -> tuple[ValidationReport, list[PaperManifestEntry]]:
    """Validate JSONL paper manifest records."""
    report = ValidationReport()
    entries: list[PaperManifestEntry] = []
    seen_ids: set[str] = set()

    if not manifest_path.exists():
        report.add_error(f"Manifest file not found: {manifest_path.resolve()}")
        return report, entries

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    entry = PaperManifestEntry.model_validate(data)
                    if entry.paper_id in seen_ids:
                        report.add_error(f"Line {line_no}: Duplicate paper_id '{entry.paper_id}'")
                    seen_ids.add(entry.paper_id)
                    entries.append(entry)
                except json.JSONDecodeError as exc:
                    report.add_error(f"Line {line_no}: Invalid JSON ({exc})")
                except ValidationError as exc:
                    report.add_error(f"Line {line_no}: Schema validation failure:\n{exc}")
    except Exception as exc:
        report.add_error(f"Error reading manifest: {exc}")

    if expected_count_range:
        min_c, max_c = expected_count_range
        if not (min_c <= len(entries) <= max_c):
            report.add_error(
                f"Manifest contains {len(entries)} papers; expected between {min_c} and {max_c}"
            )

    return report, entries


def validate_corpus(
    corpus_path: Path,
    manifest_entries: list[PaperManifestEntry] | None = None,
    expected_count_range: tuple[int, int] | None = None,
) -> tuple[ValidationReport, list[RawDocument]]:
    """Validate JSONL scientific paper corpus documents against RawDocument data contract."""
    report = ValidationReport()
    documents: list[RawDocument] = []
    seen_ids: set[str] = set()
    manifest_map: dict[str, PaperManifestEntry] = (
        {m.paper_id: m for m in manifest_entries} if manifest_entries else {}
    )

    if not corpus_path.exists():
        report.add_error(f"Corpus file not found: {corpus_path.resolve()}")
        return report, documents

    try:
        loader = DocumentLoader(corpus_path)
        documents = loader.load()
    except Exception as exc:
        report.add_error(f"Failed to load corpus via canonical DocumentLoader: {exc}")
        return report, documents

    for doc in documents:
        if doc.document_id in seen_ids:
            report.add_error(f"Duplicate document_id found in corpus: '{doc.document_id}'")
        seen_ids.add(doc.document_id)

        # Check content presence
        if not doc.content.strip():
            report.add_error(f"Document '{doc.document_id}' has empty content")

        # Verify content SHA256 checksum consistency with computed property
        computed_hash = hashlib.sha256(doc.content.encode("utf-8")).hexdigest()
        if doc.content_hash != computed_hash:
            report.add_error(
                f"Document '{doc.document_id}' content_hash mismatch: expected {computed_hash}, got {doc.content_hash}"
            )

        # Cross-reference against manifest if provided
        if manifest_map:
            manifest_entry = manifest_map.get(doc.document_id)
            if not manifest_entry:
                report.add_error(
                    f"Document '{doc.document_id}' exists in corpus but has no matching entry in manifest"
                )
            else:
                if manifest_entry.sha256 != computed_hash:
                    report.add_error(
                        f"Document '{doc.document_id}' content sha256 ({computed_hash}) does not match manifest recorded sha256 ({manifest_entry.sha256})"
                    )

    if expected_count_range:
        min_c, max_c = expected_count_range
        if not (min_c <= len(documents) <= max_c):
            report.add_error(
                f"Corpus contains {len(documents)} documents; expected between {min_c} and {max_c}"
            )

    return report, documents


def validate_benchmark(
    benchmark_path: Path,
    manifest_entries: list[PaperManifestEntry] | None = None,
    enforce_exact_count: int | None = 50,
) -> tuple[ValidationReport, ScientificBenchmarkDataset | None]:
    """Validate benchmark question dataset schema, exact category distribution, and reference integrity."""
    report = ValidationReport()
    manifest_ids = {m.paper_id for m in manifest_entries} if manifest_entries else set()

    if not benchmark_path.exists():
        report.add_error(f"Benchmark file not found: {benchmark_path.resolve()}")
        return report, None

    try:
        with benchmark_path.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as exc:
        report.add_error(f"Failed to parse JSON in benchmark file: {exc}")
        return report, None

    try:
        dataset = ScientificBenchmarkDataset.model_validate(raw_data)
    except ValidationError as exc:
        report.add_error(f"Benchmark schema validation failed:\n{exc}")
        return report, None

    questions = dataset.questions
    if enforce_exact_count is not None and len(questions) != enforce_exact_count:
        report.add_error(
            f"Benchmark contains {len(questions)} questions; strictly requires exactly {enforce_exact_count}"
        )

    # Validate category taxonomy distribution
    category_counts = Counter(q.category for q in questions)
    expected_taxonomy: dict[BenchmarkCategory, int] = {
        BenchmarkCategory.SINGLE_HOP_DIRECT_FACT: 10,
        BenchmarkCategory.TWO_HOP_RELATIONSHIP: 10,
        BenchmarkCategory.THREE_HOP_DEPENDENCY: 10,
        BenchmarkCategory.RELATIONSHIP_AGGREGATION: 10,
        BenchmarkCategory.OUT_OF_SCOPE: 10,
    }

    if enforce_exact_count == 50:
        for cat, expected_qty in expected_taxonomy.items():
            actual_qty = category_counts.get(cat, 0)
            if actual_qty != expected_qty:
                report.add_error(
                    f"Category '{cat.value}' has {actual_qty} questions; expected exactly {expected_qty}"
                )

    # Validate individual questions and manifest references
    for q in questions:
        if q.category != BenchmarkCategory.OUT_OF_SCOPE:
            if not q.expected_facts:
                report.add_error(
                    f"Question [{q.id}] in category '{q.category}' requires at least one proposition in expected_facts"
                )
            if not q.reference_answer.strip():
                report.add_error(f"Question [{q.id}] requires non-empty reference_answer")

        # Check metadata source paper references
        source_papers = q.metadata.get("source_paper_ids", [])
        if manifest_ids and source_papers:
            for p_id in source_papers:
                if p_id not in manifest_ids:
                    report.add_error(
                        f"Question [{q.id}] references unknown paper_id '{p_id}' not found in manifest"
                    )

    return report, dataset


def main() -> int:
    """CLI entrypoint for offline validation."""
    parser = argparse.ArgumentParser(
        description="Offline validation tool for Scientific-Paper Generalization Experiment"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/scipaper/data/manifest.jsonl"),
        help="Path to manifest JSONL file",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("experiments/scipaper/data/corpus_scipaper.jsonl"),
        help="Path to corpus JSONL file",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("experiments/scipaper/eval/benchmark_scipaper_50.json"),
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--strict-count",
        action="store_true",
        help="Enforce full experiment counts (100-150 corpus docs, 50 benchmark questions)",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    count_range = (100, 150) if args.strict_count else None
    benchmark_exact_count = 50 if args.strict_count else None

    overall_success = True

    print("\n" + "=" * 65)
    print("SCIENTIFIC PAPER EXPERIMENT — OFFLINE VALIDATION AUDIT")
    print("=" * 65)

    # 1. Manifest
    print(f"\n1. Validating Manifest: {args.manifest}")
    m_report, manifest_entries = validate_manifest(args.manifest, expected_count_range=count_range)
    if m_report.is_valid:
        print(f"   [PASS] Manifest contains {len(manifest_entries)} valid records.")
    else:
        print(f"   [FAIL] Manifest failed validation ({len(m_report.errors)} errors):")
        for err in m_report.errors:
            print(f"     - {err}")
        overall_success = False

    # 2. Corpus
    print(f"\n2. Validating Corpus: {args.corpus}")
    c_report, corpus_docs = validate_corpus(
        args.corpus,
        manifest_entries=manifest_entries,
        expected_count_range=count_range,
    )
    if c_report.is_valid:
        print(f"   [PASS] Corpus contains {len(corpus_docs)} valid RawDocument records.")
    else:
        print(f"   [FAIL] Corpus failed validation ({len(c_report.errors)} errors):")
        for err in c_report.errors:
            print(f"     - {err}")
        overall_success = False

    # 3. Benchmark
    print(f"\n3. Validating Benchmark: {args.benchmark}")
    b_report, b_dataset = validate_benchmark(
        args.benchmark,
        manifest_entries=manifest_entries,
        enforce_exact_count=benchmark_exact_count,
    )
    if b_report.is_valid and b_dataset:
        print(f"   [PASS] Benchmark contains {len(b_dataset.questions)} valid questions.")
    else:
        print(f"   [FAIL] Benchmark failed validation ({len(b_report.errors)} errors):")
        for err in b_report.errors:
            print(f"     - {err}")
        overall_success = False

    print("\n" + "=" * 65)
    if overall_success:
        print("OVERALL RESULT: [PASS] All offline contracts verified successfully.")
        return 0
    else:
        print("OVERALL RESULT: [FAIL] Errors encountered in validation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
