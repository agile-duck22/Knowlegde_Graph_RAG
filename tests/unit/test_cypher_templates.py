"""Unit test suite for Audited Parameterized Cypher Query Template Catalog (TASK-301).

Validates:
1. All approved templates exist in the catalog (T-01 through T-08).
2. Each template has static, audited Cypher query string with zero string interpolation or format specifiers.
3. Every template correctly specifies and validates required parameters.
4. Limit validation enforces minimum (>= 1) and clamps to max_limit.
5. Missing or blank required parameters raise ValueError with informative messages.
6. Traversal depths in Cypher templates never exceed depth limit D <= 3.
7. Templates returning relationship traversals return `source_chunk_ids` for provenance integrity.
"""

import pytest

from src.retrieval.cypher_templates import (
    TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
    TEMPLATE_DIRECT_NEIGHBORHOOD,
    TEMPLATE_ENTITY_LOOKUP_BY_ID,
    TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS,
    TEMPLATE_MULTI_HOP_DEPENDENCY,
    TEMPLATE_SERVICE_DEPENDENCY_CHAIN,
    TEMPLATE_TEAM_OWNERSHIP_SERVICES,
    TEMPLATE_VULNERABILITY_BLAST_RADIUS,
    TEMPLATE_INCOMING_SERVICE_DEPENDENTS,
    CypherTemplateCatalog,
    CypherTemplateSpec,
)


class TestCypherTemplateCatalog:
    """Unit tests for CypherTemplateCatalog and CypherTemplateSpec."""

    def test_catalog_contains_all_approved_templates(self) -> None:
        """Verify all standard templates T-01 through T-08 are registered."""
        expected_ids = {
            TEMPLATE_DIRECT_NEIGHBORHOOD,
            TEMPLATE_MULTI_HOP_DEPENDENCY,
            TEMPLATE_TEAM_OWNERSHIP_SERVICES,
            TEMPLATE_VULNERABILITY_BLAST_RADIUS,
            TEMPLATE_SERVICE_DEPENDENCY_CHAIN,
            TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
            TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS,
            TEMPLATE_ENTITY_LOOKUP_BY_ID,
            TEMPLATE_INCOMING_SERVICE_DEPENDENTS,
        }
        catalog_ids = set(CypherTemplateCatalog.list_template_ids())
        assert expected_ids.issubset(catalog_ids)

        for tid in expected_ids:
            assert CypherTemplateCatalog.has_template(tid) is True
            spec = CypherTemplateCatalog.get_template(tid)
            assert isinstance(spec, CypherTemplateSpec)
            assert spec.template_id == tid
            assert len(spec.cypher.strip()) > 0
            assert len(spec.name.strip()) > 0
            assert len(spec.description.strip()) > 0

    def test_unknown_template_id_raises_key_error(self) -> None:
        """Requesting an unknown template ID raises KeyError."""
        with pytest.raises(KeyError, match="Unknown Cypher template ID 'T-999'"):
            CypherTemplateCatalog.get_template("T-999")

        assert CypherTemplateCatalog.has_template("T-999") is False

    def test_cypher_template_safety_invariants(self) -> None:
        """Ensure no dynamic formatting placeholders exist in Cypher template text."""
        for spec in CypherTemplateCatalog.list_templates():
            # Check for unsafe format string artifacts
            assert "%s" not in spec.cypher, f"Found %s in {spec.template_id}"
            assert "{0}" not in spec.cypher, f"Found {{0}} in {spec.template_id}"
            assert "{" in spec.cypher or "$" in spec.cypher

            # Verify that every query containing Cypher parameters uses $ prefix
            for req in spec.required_parameters:
                assert f"${req}" in spec.cypher, (
                    f"${req} not found in Cypher for {spec.template_id}"
                )

            # Verify LIMIT is present and parameterized or hardcoded
            assert "LIMIT" in spec.cypher, f"LIMIT clause missing in {spec.template_id}"

    def test_traversal_depth_boundary_enforcement(self) -> None:
        """Verify that multi-hop traversals never specify depth > 3."""
        for spec in CypherTemplateCatalog.list_templates():
            # Look for variable length pattern *1..N
            if "*1.." in spec.cypher:
                # Check maximum upper-bound in range
                import re

                matches = re.findall(r"\*1\.\.(\d+)", spec.cypher)
                for max_depth_str in matches:
                    max_depth = int(max_depth_str)
                    assert max_depth <= 3, (
                        f"Template {spec.template_id} specifies depth {max_depth} > 3"
                    )

    def test_provenance_retrieval_invariant(self) -> None:
        """Templates returning relationship edges or impact paths must include source_chunk_ids."""
        relationship_templates = [
            TEMPLATE_DIRECT_NEIGHBORHOOD,
            TEMPLATE_MULTI_HOP_DEPENDENCY,
            TEMPLATE_TEAM_OWNERSHIP_SERVICES,
            TEMPLATE_VULNERABILITY_BLAST_RADIUS,
            TEMPLATE_SERVICE_DEPENDENCY_CHAIN,
            TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
            TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS,
            TEMPLATE_INCOMING_SERVICE_DEPENDENTS,
        ]
        for tid in relationship_templates:
            spec = CypherTemplateCatalog.get_template(tid)
            assert "source_chunk_ids" in spec.cypher or "chunk_ids" in spec.cypher, (
                f"Template {tid} does not return chunk IDs for provenance"
            )

    def test_parameter_validation_happy_path(self) -> None:
        """Valid parameters build cleanly with default limit applied."""
        spec = CypherTemplateCatalog.get_template(TEMPLATE_DIRECT_NEIGHBORHOOD)
        params = spec.validate_and_build_parameters({"entity_id": "service:payment-gateway"})

        assert params["entity_id"] == "service:payment-gateway"
        assert params["limit"] == 50  # Default limit

    def test_parameter_validation_custom_limit(self) -> None:
        """Custom valid limit is preserved."""
        spec = CypherTemplateCatalog.get_template(TEMPLATE_DIRECT_NEIGHBORHOOD)
        params = spec.validate_and_build_parameters(
            {"entity_id": "service:payment-gateway", "limit": 25}
        )

        assert params["limit"] == 25

    def test_parameter_validation_clamps_to_max_limit(self) -> None:
        """Excessive limit values are safely clamped to max_limit."""
        spec = CypherTemplateCatalog.get_template(TEMPLATE_DIRECT_NEIGHBORHOOD)
        params = spec.validate_and_build_parameters(
            {"entity_id": "service:payment-gateway", "limit": 1000}
        )

        assert params["limit"] == spec.max_limit
        assert params["limit"] == 200

    def test_parameter_validation_missing_required_raises_error(self) -> None:
        """Missing required parameter raises ValueError."""
        spec = CypherTemplateCatalog.get_template(TEMPLATE_DIRECT_NEIGHBORHOOD)
        with pytest.raises(ValueError, match="Missing required parameter 'entity_id'"):
            spec.validate_and_build_parameters({})

    def test_parameter_validation_empty_required_raises_error(self) -> None:
        """Empty string for required parameter raises ValueError."""
        spec = CypherTemplateCatalog.get_template(TEMPLATE_DIRECT_NEIGHBORHOOD)
        with pytest.raises(ValueError, match="cannot be empty string"):
            spec.validate_and_build_parameters({"entity_id": "   "})

    def test_parameter_validation_invalid_limit_type_raises_error(self) -> None:
        """Non-integer limit raises ValueError."""
        spec = CypherTemplateCatalog.get_template(TEMPLATE_DIRECT_NEIGHBORHOOD)
        with pytest.raises(ValueError, match="Invalid 'limit' value"):
            spec.validate_and_build_parameters(
                {"entity_id": "service:payment-gateway", "limit": "invalid_number"}
            )

    def test_parameter_validation_zero_or_negative_limit_raises_error(self) -> None:
        """Non-positive limit raises ValueError."""
        spec = CypherTemplateCatalog.get_template(TEMPLATE_DIRECT_NEIGHBORHOOD)
        with pytest.raises(ValueError, match="'limit' must be >= 1"):
            spec.validate_and_build_parameters({"entity_id": "service:payment-gateway", "limit": 0})
        with pytest.raises(ValueError, match="'limit' must be >= 1"):
            spec.validate_and_build_parameters(
                {"entity_id": "service:payment-gateway", "limit": -5}
            )

    def test_entity_lookup_by_id_limit_locked(self) -> None:
        """Template T-08 exact entity lookup enforces default limit 1."""
        spec = CypherTemplateCatalog.get_template(TEMPLATE_ENTITY_LOOKUP_BY_ID)
        params = spec.validate_and_build_parameters({"entity_id": "service:payment-gateway"})
        assert params["limit"] == 1
        assert spec.max_limit == 1
