"""Unit tests validating synthetic enterprise reference corpus integrity."""

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def corpus_records() -> list[dict[str, Any]]:
    """Load and parse all records from data/raw/corpus.jsonl."""
    corpus_path = Path(__file__).parents[2] / "data" / "raw" / "corpus.jsonl"
    assert corpus_path.exists(), f"Corpus file not found at {corpus_path}"

    records: list[dict[str, Any]] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON on line {line_num}: {e}")
    return records


def test_corpus_document_count(corpus_records: list[dict[str, Any]]) -> None:
    """Verify corpus contains at least 120 documents (meets >=100 requirement)."""
    assert len(corpus_records) >= 120, f"Expected >= 120 documents, found {len(corpus_records)}"


def test_unique_document_ids(corpus_records: list[dict[str, Any]]) -> None:
    """Verify all document_id values are unique and non-empty strings."""
    doc_ids = [r.get("document_id") for r in corpus_records]
    assert all(doc_ids), "All records must contain non-empty document_id"
    assert len(doc_ids) == len(set(doc_ids)), "Duplicate document_id detected in corpus"


def test_required_schema_fields(corpus_records: list[dict[str, Any]]) -> None:
    """Verify all required schema fields are present and non-empty."""
    required_fields = ["document_id", "source_uri", "title", "doc_type", "content", "metadata"]
    for record in corpus_records:
        for field in required_fields:
            assert field in record, (
                f"Missing field '{field}' in document {record.get('document_id')}"
            )
            assert record[field], (
                f"Field '{field}' cannot be empty in document {record.get('document_id')}"
            )
        assert isinstance(record["metadata"], dict), (
            f"metadata must be dict in {record.get('document_id')}"
        )


def test_canonical_entity_identifiers(corpus_records: list[dict[str, Any]]) -> None:
    """Verify referenced entity IDs adhere to kebab-case prefix conventions."""
    valid_prefixes = (
        "team:",
        "service:",
        "api:",
        "database:",
        "infra:",
        "repo:",
        "vuln:",
        "adr:",
        "incident:",
        "runbook:",
        "integration:",
    )
    for record in corpus_records:
        entity_id = record["metadata"].get("entity_id")
        if entity_id:
            assert any(entity_id.startswith(p) for p in valid_prefixes), (
                f"Entity ID '{entity_id}' in {record['document_id']} does not match valid prefixes"
            )


def test_doc_type_distribution(corpus_records: list[dict[str, Any]]) -> None:
    """Verify all expected domain document categories are represented."""
    doc_types = {r["doc_type"] for r in corpus_records}
    expected_types = {
        "team_charter",
        "service_spec",
        "api_contract",
        "database_spec",
        "infra_spec",
        "repository_manifest",
        "security_advisory",
        "architecture_decision_record",
        "post_mortem",
        "operational_runbook",
        "integration_spec",
    }
    assert expected_types.issubset(doc_types), (
        f"Missing document types: {expected_types - doc_types}"
    )


def test_cross_document_connections(corpus_records: list[dict[str, Any]]) -> None:
    """Verify multi-hop cross-document relational continuity."""
    # Find Checkout Service document
    checkout_doc = next(
        (r for r in corpus_records if r["document_id"] == "doc-service-checkout-service"), None
    )
    assert checkout_doc is not None
    assert "team:checkout-engineering" in checkout_doc["content"]
    assert "service:payment-gateway" in checkout_doc["content"]
    assert "database:customer-orders-pg" in checkout_doc["content"]

    # Verify Payment Gateway document connects to PCI VPC and CVE
    payment_doc = next(
        (r for r in corpus_records if r["document_id"] == "doc-service-payment-gateway"), None
    )
    assert payment_doc is not None
    assert "database:payment-vault-pg" in payment_doc["content"]
    assert "infra:pci-isolated-vpc" in payment_doc["content"]

    # Verify CVE document points back to payment gateway
    cve_doc = next(
        (r for r in corpus_records if r["document_id"] == "doc-vuln-cve-2024-21626"), None
    )
    assert cve_doc is not None
    assert "service:payment-gateway" in cve_doc["content"]
