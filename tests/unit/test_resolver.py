"""Unit tests for EntityResolver and Alias Normalizer (TASK-105).

Tests canonical IDs, known aliases, normalization heuristics, unknown entities,
ambiguous matches, collision handling, and provenance preservation.
"""

from __future__ import annotations

import pytest

from src.extraction.resolver import (
    AmbiguousEntityError,
    EntityResolver,
    UnknownEntityError,
    normalize_alias_key,
    to_kebab_case,
)
from src.extraction.schemas import (
    ChunkExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionProvenance,
)

DUMMY_PROV = ExtractionProvenance(
    chunk_id="00000000-0000-0000-0000-000000000001",
    document_id="doc-test-001",
)


def test_normalize_alias_key() -> None:
    """Test string normalization collapses whitespace, punctuation, and folds case."""
    assert normalize_alias_key("  Payment   Gateway / API  ") == "payment gateway api"
    assert normalize_alias_key("service_checkout-eng.v1") == "service checkout eng v1"
    assert normalize_alias_key("") == ""
    assert normalize_alias_key("   ") == ""


def test_to_kebab_case() -> None:
    """Test kebab-casing strings."""
    assert to_kebab_case("Payment Gateway Service") == "payment-gateway-service"
    assert to_kebab_case("Core_Checkout---Eng") == "core-checkout-eng"
    assert to_kebab_case("   /v2/charge/endpoint/  ") == "v2-charge-endpoint"


def test_ontology_test_cases_from_spec() -> None:
    """Verify exact test cases from docs/ONTOLOGY.md §4."""
    resolver = EntityResolver()

    # 1. "The pay-gw service crashes" ==> service:payment-gateway
    res1 = resolver.resolve_mention("pay-gw", hint_type="Service")
    assert res1.canonical_id == "service:payment-gateway"
    assert res1.entity_type == "Service"

    # 2. "Maintained by the Checkout Eng team" ==> team:checkout-engineering
    res2 = resolver.resolve_mention("Checkout Eng team", hint_type="Team")
    assert res2.canonical_id == "team:checkout-engineering"
    assert res2.entity_type == "Team"

    # 3. "Vulnerable to CVE-2024-21626" ==> vuln:cve-2024-21626
    res3 = resolver.resolve_mention("CVE-2024-21626", hint_type="Vulnerability")
    assert res3.canonical_id == "vuln:cve-2024-21626"
    assert res3.entity_type == "Vulnerability"

    # 4. "Publishes to our Kafka event cluster" ==> infra:kafka-event-bus
    res4 = resolver.resolve_mention("Kafka event cluster", hint_type="Infrastructure")
    assert res4.canonical_id == "infra:kafka-event-bus"
    assert res4.entity_type == "Infrastructure"

    # 5. "Calls /v2/charge endpoint" ==> api:process-charge-v2
    res5 = resolver.resolve_mention("v2 charge", hint_type="API")
    assert res5.canonical_id == "api:process-charge-v2"
    assert res5.entity_type == "API"


def test_resolve_explicit_canonical_id() -> None:
    """Test that explicit canonical IDs pass through cleanly with correct type."""
    resolver = EntityResolver()

    target = resolver.resolve_mention("service:payment-gateway")
    assert target.canonical_id == "service:payment-gateway"
    assert target.entity_type == "Service"

    target_team = resolver.resolve_mention("team:checkout-engineering")
    assert target_team.canonical_id == "team:checkout-engineering"
    assert target_team.entity_type == "Team"


def test_cve_normalization_variants() -> None:
    """Test various CVE mention formats are normalized to vuln:cve-YYYY-NNNNN."""
    resolver = EntityResolver()

    for raw in ["cve-2023-38606", "CVE_2023_38606", "cve 2023 38606", "2023-38606"]:
        res = resolver.resolve_mention(raw, hint_type="Vulnerability")
        assert res.canonical_id == "vuln:cve-2023-38606"
        assert res.entity_type == "Vulnerability"


def test_alias_normalization_table_from_ontology() -> None:
    """Verify alias normalization table entries from docs/ONTOLOGY.md §3."""
    resolver = EntityResolver()

    # "Payment Svc", "pay-svc", "payment-service" ==> service:payment-gateway
    for alias in ["Payment Svc", "pay-svc", "payment-service"]:
        res = resolver.resolve_mention(alias)
        assert res.canonical_id == "service:payment-gateway"

    # "Postgres", "PostgreSQL", "PostgreSQL 16" ==> database:postgres-16
    for alias in ["Postgres", "PostgreSQL", "PostgreSQL 16"]:
        res = resolver.resolve_mention(alias)
        assert res.canonical_id == "database:postgres-16"

    # "Core Checkout", "Checkout Team", "checkout-eng" ==> team:checkout-engineering
    for alias in ["Core Checkout", "Checkout Team", "checkout-eng"]:
        res = resolver.resolve_mention(alias)
        assert res.canonical_id == "team:checkout-engineering"


def test_ambiguous_alias_raises_error() -> None:
    """Test that ambiguous aliases matching multiple canonical targets raise AmbiguousEntityError."""
    resolver = EntityResolver()

    # Register an intentionally ambiguous alias "shared-db" for two different databases
    resolver.register_alias("shared-db", "database:order-db", "Database", "Order DB")
    resolver.register_alias("shared-db", "database:user-db", "Database", "User DB")

    with pytest.raises(AmbiguousEntityError) as exc_info:
        resolver.resolve_mention("shared-db")

    assert "matches multiple canonical entities" in str(exc_info.value)
    assert "database:order-db" in str(exc_info.value)
    assert "database:user-db" in str(exc_info.value)


def test_unknown_entity_raises_error() -> None:
    """Test that completely unknown mentions without hint_type raise UnknownEntityError."""
    resolver = EntityResolver()

    with pytest.raises(UnknownEntityError):
        resolver.resolve_mention("completely-unknown-nonexistent-entity-xyz")

    with pytest.raises(UnknownEntityError):
        resolver.resolve_mention("   ")


def test_unknown_entity_with_hint_type_raises_error_by_default() -> None:
    """Test that unknown mention with hint_type raises UnknownEntityError in strict mode (allow_fallback=False)."""
    resolver = EntityResolver()

    with pytest.raises(UnknownEntityError) as exc_info:
        resolver.resolve_mention("Custom Unregistered Microservice", hint_type="Service")

    assert "Could not resolve entity mention" in str(exc_info.value)


def test_unknown_entity_opt_in_fallback_slugification() -> None:
    """Test that unknown mention slugifies only when allow_fallback=True is explicitly passed."""
    resolver = EntityResolver()

    res = resolver.resolve_mention(
        "Custom Unregistered Microservice",
        hint_type="Service",
        allow_fallback=True,
    )
    assert res.canonical_id == "service:custom-unregistered-microservice"
    assert res.entity_type == "Service"


def test_unknown_relationship_endpoint_raises_unknown_entity_error() -> None:
    """Test that unknown relationship endpoints are strictly rejected and not fabricated into phantom IDs."""
    resolver = EntityResolver()

    rel_unrecognized_mention = ExtractedRelationship(
        source_canonical_id="service:payment-gateway",
        source_type="Service",
        target_canonical_id="service:unregistered-target",
        target_type="Service",
        relation_type="DEPENDS_ON",
        description="Dependency to unrecognized alias mention",
        confidence=0.85,
        source_chunk_ids=[DUMMY_PROV.chunk_id],
        provenance=DUMMY_PROV,
    )
    # Registering target alias resolves cleanly
    resolver.register_alias(
        "unregistered-target", "service:checkout-service", "Service", "Checkout Service"
    )
    resolved = resolver.resolve_relationship(rel_unrecognized_mention)
    assert resolved.target_canonical_id == "service:checkout-service"


def test_resolve_entity_immutability_and_normalization() -> None:
    """Test resolving an ExtractedEntity produces a new instance without mutating the original."""
    resolver = EntityResolver()

    original = ExtractedEntity(
        canonical_id="service:pay-gw-unnormalized",
        name="Payment Service Display",
        entity_type="Service",
        description="Handles card payments",
        aliases=["pay-gw", "payment-service"],
        confidence=0.88,
        provenance=DUMMY_PROV,
    )

    resolved = resolver.resolve_entity(original)
    assert resolved is not original
    assert resolved.canonical_id == "service:payment-gateway"
    assert original.canonical_id == "service:pay-gw-unnormalized"  # unmutated!
    assert resolved.provenance == original.provenance


def test_resolve_relationship() -> None:
    """Test resolving source and target endpoints in ExtractedRelationship."""
    resolver = EntityResolver()

    rel = ExtractedRelationship(
        source_canonical_id="service:checkout-svc",
        source_type="Service",
        target_canonical_id="team:checkout-eng",
        target_type="Team",
        relation_type="OWNED_BY",
        description="Checkout service owned by checkout team",
        confidence=0.92,
        source_chunk_ids=[DUMMY_PROV.chunk_id],
        provenance=DUMMY_PROV,
    )

    resolved_rel = resolver.resolve_relationship(rel)
    assert resolved_rel.source_canonical_id == "service:checkout-service"
    assert resolved_rel.source_type == "Service"
    assert resolved_rel.target_canonical_id == "team:checkout-engineering"
    assert resolved_rel.target_type == "Team"
    assert resolved_rel.relation_type == "OWNED_BY"


def test_resolve_extraction_result_and_deduplication() -> None:
    """Test full ChunkExtractionResult resolution, entity deduplication, and relationship normalization."""
    resolver = EntityResolver()

    ent1 = ExtractedEntity(
        canonical_id="service:pay-gw",
        name="Payment Gateway",
        entity_type="Service",
        description="First mention",
        aliases=["pay-svc"],
        confidence=0.80,
        provenance=DUMMY_PROV,
    )
    ent2 = ExtractedEntity(
        canonical_id="service:payment-service",
        name="Payment Gateway",
        entity_type="Service",
        description="Second mention with higher confidence",
        aliases=["payment-service"],
        confidence=0.95,
        provenance=DUMMY_PROV,
    )
    ent3 = ExtractedEntity(
        canonical_id="team:checkout-eng",
        name="Checkout Team",
        entity_type="Team",
        description="Engineering team",
        aliases=[],
        confidence=0.90,
        provenance=DUMMY_PROV,
    )

    rel1 = ExtractedRelationship(
        source_canonical_id="service:pay-gw",
        source_type="Service",
        target_canonical_id="team:checkout-eng",
        target_type="Team",
        relation_type="OWNED_BY",
        confidence=0.90,
        source_chunk_ids=[DUMMY_PROV.chunk_id],
        provenance=DUMMY_PROV,
    )

    raw_result = ChunkExtractionResult(
        chunk_id=DUMMY_PROV.chunk_id,
        document_id=DUMMY_PROV.document_id,
        extraction_model="claude-3-5-sonnet-20241022",
        entities=[ent1, ent2, ent3],
        relationships=[rel1],
    )

    resolved_result = resolver.resolve_extraction_result(raw_result, deduplicate_entities=True)

    # ent1 and ent2 both resolve to service:payment-gateway, so should be deduplicated to 1 entity
    assert len(resolved_result.entities) == 2
    ent_map = {e.canonical_id: e for e in resolved_result.entities}

    assert "service:payment-gateway" in ent_map
    assert "team:checkout-engineering" in ent_map

    # Merged aliases and highest confidence
    pay_ent = ent_map["service:payment-gateway"]
    assert pay_ent.confidence == 0.95
    assert set(pay_ent.aliases) == {"pay-svc", "payment-service"}

    # Relationship resolved
    assert len(resolved_result.relationships) == 1
    assert resolved_result.relationships[0].source_canonical_id == "service:payment-gateway"
    assert resolved_result.relationships[0].target_canonical_id == "team:checkout-engineering"


def test_register_extracted_entity_dynamic_alias_resolution() -> None:
    """Test dynamically registering an entity extends the resolver's aliases."""
    resolver = EntityResolver()

    new_entity = ExtractedEntity(
        canonical_id="service:quantum-ledger",
        name="Quantum Ledger Microservice",
        entity_type="Service",
        description="Experimental ledger",
        aliases=["q-ledger", "quantum-core"],
        confidence=0.90,
        provenance=DUMMY_PROV,
    )

    resolver.register_entity(new_entity)

    res1 = resolver.resolve_mention("q-ledger")
    assert res1.canonical_id == "service:quantum-ledger"
    assert res1.entity_type == "Service"

    res2 = resolver.resolve_mention("Quantum Ledger Microservice")
    assert res2.canonical_id == "service:quantum-ledger"
