"""Pydantic schemas for structured entity and relationship extraction.

These schemas are the single source of truth for what Claude must produce.
They enforce the domain ontology defined in ``docs/ONTOLOGY.md`` and the
storage contracts defined in ``docs/DATA_MODEL.md``.

Design constraints
------------------
* Entity types and relationship types are locked to the ontology's 7 node
  labels and 10 directed edge types — no dynamic strings are accepted.
* Canonical IDs follow the ``<type_prefix>:<kebab-name>`` format required by
  Neo4j constraints and the ``entity_links`` PostgreSQL table.
* Provenance is mandatory: every extracted entity and relationship carries the
  originating ``chunk_id`` and ``document_id`` so the dual-store invariant can
  be validated downstream.
* ``confidence`` is a [0.0, 1.0] float — the LLM must justify each extraction
  with an explicit confidence score.
* These schemas are intentionally free of I/O logic (no API calls, no DB
  writes). TASK-104 will use them inside the extractor.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Ontology-locked literal types
# ---------------------------------------------------------------------------

EntityType = Literal[
    "Service",
    "Team",
    "API",
    "Database",
    "Repository",
    "Vulnerability",
    "Infrastructure",
]

RelationshipType = Literal[
    "OWNED_BY",
    "DEPENDS_ON",
    "EXPOSES_API",
    "CALLS_API",
    "READS_FROM",
    "WRITES_TO",
    "STORES_CODE",
    "AFFECTED_BY",
    "DEPLOYED_ON",
    "PRODUCES_EVENTS",
]

# Canonical ID prefixes keyed by entity type (mirrors ONTOLOGY.md §1)
_CANONICAL_ID_PREFIX: dict[EntityType, str] = {
    "Service": "service",
    "Team": "team",
    "API": "api",
    "Database": "database",
    "Repository": "repo",
    "Vulnerability": "vuln",
    "Infrastructure": "infra",
}

# Pre-compiled pattern for a valid canonical entity ID: "<prefix>:<kebab-slug>"
_CANONICAL_ID_RE = re.compile(
    r"^(service|team|api|database|repo|vuln|infra):[a-z0-9][a-z0-9\-\.]*$"
)

# UUID4 pattern for chunk_id and document_id provenance validation
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


# ---------------------------------------------------------------------------
# Provenance — shared by both Entity and Relationship extractions
# ---------------------------------------------------------------------------


class ExtractionProvenance(BaseModel):
    """Traceability record linking an extraction back to a specific chunk.

    Both ``chunk_id`` and ``document_id`` are mandatory so that downstream
    validators can verify the dual-store invariant without additional lookups.
    """

    chunk_id: str = Field(
        ...,
        description="UUID of the originating TextChunk (matches PostgreSQL chunks.id and Neo4j source_chunk_ids)",
    )
    document_id: str = Field(
        ...,
        description="Stable document identifier from RawDocument.document_id",
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("chunk_id", "document_id")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Provenance fields cannot be empty or whitespace")
        return v.strip()


# ---------------------------------------------------------------------------
# Extracted entity
# ---------------------------------------------------------------------------


class ExtractedEntity(BaseModel):
    """A single entity extracted from a document chunk.

    Canonical IDs must conform to the ontology's ``<prefix>:<kebab-name>``
    format.  The ``entity_type`` literal determines which prefix is valid.

    Example
    -------
    ::

        ExtractedEntity(
            canonical_id="service:payment-gateway",
            name="Payment Gateway",
            entity_type="Service",
            description="Processes card and ACH payments.",
            aliases=["pay-gw", "payment-svc"],
            confidence=0.95,
            provenance=ExtractionProvenance(
                chunk_id="a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
                document_id="doc-payment-gateway-spec",
            ),
        )
    """

    canonical_id: str = Field(
        ...,
        description="Canonical entity identifier: '<type_prefix>:<kebab-name>'",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable display name of the entity",
    )
    entity_type: EntityType = Field(
        ...,
        description="Ontology-defined entity category",
    )
    description: str = Field(
        default="",
        description="Short description extracted or inferred from the document",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names or abbreviations seen in the text",
    )
    confidence: Confidence = Field(
        ...,
        description="Extraction confidence in [0.0, 1.0]",
    )
    provenance: ExtractionProvenance = Field(
        ...,
        description="Originating chunk and document identifiers",
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("canonical_id")
    @classmethod
    def validate_canonical_id_format(cls, v: str) -> str:
        """Enforce ``<prefix>:<kebab-slug>`` format."""
        if not _CANONICAL_ID_RE.match(v):
            raise ValueError(
                f"canonical_id '{v}' does not match required format "
                "'<type_prefix>:<kebab-slug>' (e.g. 'service:payment-gateway'). "
                "Allowed prefixes: service, team, api, database, repo, vuln, infra."
            )
        return v

    @field_validator("name")
    @classmethod
    def validate_name_non_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Entity name cannot be empty or whitespace only")
        return v

    @model_validator(mode="after")
    def validate_prefix_matches_entity_type(self) -> ExtractedEntity:
        """Ensure the canonical ID prefix is consistent with the entity_type."""
        expected_prefix = _CANONICAL_ID_PREFIX[self.entity_type]
        actual_prefix = self.canonical_id.split(":")[0]
        if actual_prefix != expected_prefix:
            raise ValueError(
                f"canonical_id prefix '{actual_prefix}' does not match "
                f"entity_type '{self.entity_type}' (expected prefix '{expected_prefix}')"
            )
        return self


# ---------------------------------------------------------------------------
# Extracted relationship
# ---------------------------------------------------------------------------


class ExtractedRelationship(BaseModel):
    """A single directed relationship between two entities.

    Both endpoints are referenced by their canonical entity IDs.  The
    ``source_chunk_ids`` list must contain at least one chunk UUID — it mirrors
    the Neo4j relationship property of the same name.

    Example
    -------
    ::

        ExtractedRelationship(
            source_canonical_id="service:payment-gateway",
            source_type="Service",
            target_canonical_id="service:checkout-service",
            target_type="Service",
            relation_type="DEPENDS_ON",
            description="Synchronous gRPC calls during checkout flow.",
            confidence=0.92,
            source_chunk_ids=["a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"],
            provenance=ExtractionProvenance(
                chunk_id="a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
                document_id="doc-payment-gateway-spec",
            ),
        )
    """

    source_canonical_id: str = Field(
        ...,
        description="Canonical ID of the relationship source entity",
    )
    source_type: EntityType = Field(
        ...,
        description="Ontology entity type of the source node",
    )
    target_canonical_id: str = Field(
        ...,
        description="Canonical ID of the relationship target entity",
    )
    target_type: EntityType = Field(
        ...,
        description="Ontology entity type of the target node",
    )
    relation_type: RelationshipType = Field(
        ...,
        description="Ontology-defined directed edge label",
    )
    description: str = Field(
        default="",
        description="Brief description of the relationship from document evidence",
    )
    confidence: Confidence = Field(
        ...,
        description="Extraction confidence in [0.0, 1.0]",
    )
    source_chunk_ids: list[str] = Field(
        ...,
        min_length=1,
        description="UUID(s) of the chunks providing evidence for this relationship",
    )
    provenance: ExtractionProvenance = Field(
        ...,
        description="Primary originating chunk and document identifiers",
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("source_canonical_id", "target_canonical_id")
    @classmethod
    def validate_canonical_id_format(cls, v: str) -> str:
        if not _CANONICAL_ID_RE.match(v):
            raise ValueError(
                f"canonical_id '{v}' does not match required format '<type_prefix>:<kebab-slug>'"
            )
        return v

    @field_validator("source_chunk_ids")
    @classmethod
    def validate_chunk_ids_non_empty_strings(cls, v: list[str]) -> list[str]:
        for chunk_id in v:
            if not chunk_id.strip():
                raise ValueError("source_chunk_ids entries cannot be empty or whitespace")
        return v

    @model_validator(mode="after")
    def validate_source_prefix_matches_source_type(self) -> ExtractedRelationship:
        expected = _CANONICAL_ID_PREFIX[self.source_type]
        actual = self.source_canonical_id.split(":")[0]
        if actual != expected:
            raise ValueError(
                f"source_canonical_id prefix '{actual}' does not match "
                f"source_type '{self.source_type}' (expected '{expected}')"
            )
        return self

    @model_validator(mode="after")
    def validate_target_prefix_matches_target_type(self) -> ExtractedRelationship:
        expected = _CANONICAL_ID_PREFIX[self.target_type]
        actual = self.target_canonical_id.split(":")[0]
        if actual != expected:
            raise ValueError(
                f"target_canonical_id prefix '{actual}' does not match "
                f"target_type '{self.target_type}' (expected '{expected}')"
            )
        return self

    @model_validator(mode="after")
    def validate_source_and_target_differ(self) -> ExtractedRelationship:
        if self.source_canonical_id == self.target_canonical_id:
            raise ValueError(
                "source_canonical_id and target_canonical_id must be different — "
                "self-referential relationships are not permitted"
            )
        return self


# ---------------------------------------------------------------------------
# Aggregated extraction result for a single chunk
# ---------------------------------------------------------------------------


class ChunkExtractionResult(BaseModel):
    """Aggregated extraction output for one ``TextChunk``.

    This is the top-level object that TASK-104's extractor will produce and
    persist.  It bundles all entities and relationships found in one chunk with
    the chunk's own provenance so downstream writers can perform a single
    atomic upsert.

    The ``extraction_model`` field records which Claude model was used so
    result quality can be traced during evaluation.
    """

    chunk_id: str = Field(
        ...,
        description="UUID of the source TextChunk (must match provenance.chunk_id on all children)",
    )
    document_id: str = Field(
        ...,
        description="Parent document identifier",
    )
    extraction_model: str = Field(
        ...,
        min_length=1,
        description="Model identifier used for extraction (e.g. 'claude-3-5-sonnet-20241022')",
    )
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Entities identified in this chunk",
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list,
        description="Directed relationships identified in this chunk",
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("chunk_id", "document_id")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk_id and document_id must not be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_provenance_consistency(self) -> ChunkExtractionResult:
        """All child provenance records must reference this chunk's own IDs."""
        for entity in self.entities:
            if entity.provenance.chunk_id != self.chunk_id:
                raise ValueError(
                    f"Entity '{entity.canonical_id}' provenance.chunk_id "
                    f"'{entity.provenance.chunk_id}' does not match "
                    f"ChunkExtractionResult.chunk_id '{self.chunk_id}'"
                )
            if entity.provenance.document_id != self.document_id:
                raise ValueError(
                    f"Entity '{entity.canonical_id}' provenance.document_id "
                    f"'{entity.provenance.document_id}' does not match "
                    f"ChunkExtractionResult.document_id '{self.document_id}'"
                )
        for rel in self.relationships:
            if rel.provenance.chunk_id != self.chunk_id:
                raise ValueError(
                    f"Relationship '{rel.source_canonical_id}->{rel.target_canonical_id}' "
                    f"provenance.chunk_id '{rel.provenance.chunk_id}' does not match "
                    f"ChunkExtractionResult.chunk_id '{self.chunk_id}'"
                )
            if rel.provenance.document_id != self.document_id:
                raise ValueError(
                    f"Relationship '{rel.source_canonical_id}->{rel.target_canonical_id}' "
                    f"provenance.document_id '{rel.provenance.document_id}' does not match "
                    f"ChunkExtractionResult.document_id '{self.document_id}'"
                )
            if self.chunk_id not in rel.source_chunk_ids:
                raise ValueError(
                    f"Relationship '{rel.source_canonical_id}->{rel.target_canonical_id}' "
                    f"source_chunk_ids does not include the parent chunk_id '{self.chunk_id}'"
                )
        return self
