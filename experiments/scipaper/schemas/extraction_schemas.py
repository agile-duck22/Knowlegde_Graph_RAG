"""Scientific Domain Extraction and Graph Schemas.

Defines Pydantic models for:
- ScientificExtractedEntity: Typed entities for academic research literature.
- ScientificExtractedRelationship: Directed semantic relationships between scientific entities.
- ScientificExtractionResult: Full chunk-level extraction container.

Enforces ontology rules:
- 8 Entity Types: Paper, Author, Model, Dataset, Metric, Task, Method, ResearchOrganization.
- 7 Relationship Types: AUTHORED, PROPOSES, EVALUATED_ON, REPORTS_METRIC, USES_METHOD, AFFILIATED_WITH, CITES.
- Dual-store provenance: chunk_id and document_id must match PostgreSQL chunks.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from experiments.scipaper.schemas.models import (
    SCIENTIFIC_CANONICAL_PREFIXES,
    SCIENTIFIC_ID_REGEX,
    ScientificEntityType,
    ScientificRelationshipType,
)
from src.extraction.schemas import ExtractionProvenance

# Confidence float bounded in [0.0, 1.0]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]

_ALLOWED_SCIENTIFIC_ENTITY_TYPES: frozenset[str] = frozenset(SCIENTIFIC_CANONICAL_PREFIXES.keys())

_ALLOWED_SCIENTIFIC_REL_TYPES: frozenset[str] = frozenset(
    [
        "AUTHORED",
        "PROPOSES",
        "EVALUATED_ON",
        "REPORTS_METRIC",
        "USES_METHOD",
        "AFFILIATED_WITH",
        "CITES",
    ]
)


class ScientificExtractedEntity(BaseModel):
    """A single scientific entity extracted from an academic paper chunk."""

    canonical_id: str = Field(
        ...,
        description="Canonical ID format: '<prefix>:<kebab-name>' (e.g. 'paper:dense-passage-retrieval')",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable display name of the entity",
    )
    entity_type: ScientificEntityType = Field(
        ...,
        description="Scientific ontology entity category",
    )
    description: str = Field(
        default="",
        description="Short description or context extracted from the document",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names or abbreviations seen in the text",
    )
    confidence: Confidence = Field(
        default=1.0,
        description="Extraction confidence score in [0.0, 1.0]",
    )
    provenance: ExtractionProvenance = Field(
        ...,
        description="Originating chunk and document identifiers",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("canonical_id")
    @classmethod
    def validate_canonical_id_format(cls, v: str) -> str:
        """Enforce canonical ID prefix and format."""
        if not SCIENTIFIC_ID_REGEX.match(v):
            raise ValueError(
                f"canonical_id '{v}' does not match required scientific format "
                "'<prefix>:<kebab-slug>' (e.g. 'paper:rag-survey'). "
                f"Allowed prefixes: {sorted(SCIENTIFIC_CANONICAL_PREFIXES.values())}"
            )
        return v

    @field_validator("name")
    @classmethod
    def validate_name_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Entity name cannot be empty or whitespace only")
        return v.strip()

    @model_validator(mode="after")
    def validate_prefix_matches_entity_type(self) -> ScientificExtractedEntity:
        expected_prefix = SCIENTIFIC_CANONICAL_PREFIXES[self.entity_type]
        actual_prefix = self.canonical_id.split(":")[0]
        if actual_prefix != expected_prefix:
            raise ValueError(
                f"canonical_id prefix '{actual_prefix}' does not match "
                f"entity_type '{self.entity_type}' (expected '{expected_prefix}')"
            )
        return self


class ScientificExtractedRelationship(BaseModel):
    """A directed semantic relationship between two scientific entities."""

    source_canonical_id: str = Field(
        ...,
        description="Canonical ID of the source entity",
    )
    source_type: ScientificEntityType = Field(
        ...,
        description="Scientific entity type of the source node",
    )
    target_canonical_id: str = Field(
        ...,
        description="Canonical ID of the target entity",
    )
    target_type: ScientificEntityType = Field(
        ...,
        description="Scientific entity type of the target node",
    )
    relation_type: ScientificRelationshipType = Field(
        ...,
        description="Permitted scientific relationship type",
    )
    description: str = Field(
        default="",
        description="Brief evidence statement from the chunk text",
    )
    confidence: Confidence = Field(
        default=1.0,
        description="Relationship confidence score in [0.0, 1.0]",
    )
    source_chunk_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Originating chunk UUIDs providing evidence for this relationship",
    )
    provenance: ExtractionProvenance = Field(
        ...,
        description="Primary chunk and document provenance for this relationship extraction",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("source_canonical_id", "target_canonical_id")
    @classmethod
    def validate_endpoint_id_format(cls, v: str) -> str:
        if not SCIENTIFIC_ID_REGEX.match(v):
            raise ValueError(f"Relationship endpoint '{v}' does not match canonical format")
        return v

    @model_validator(mode="after")
    def validate_endpoints_and_types(self) -> ScientificExtractedRelationship:
        # Validate source prefix
        exp_src_prefix = SCIENTIFIC_CANONICAL_PREFIXES[self.source_type]
        act_src_prefix = self.source_canonical_id.split(":")[0]
        if act_src_prefix != exp_src_prefix:
            raise ValueError(
                f"source_canonical_id prefix '{act_src_prefix}' does not match source_type '{self.source_type}'"
            )

        # Validate target prefix
        exp_tgt_prefix = SCIENTIFIC_CANONICAL_PREFIXES[self.target_type]
        act_tgt_prefix = self.target_canonical_id.split(":")[0]
        if act_tgt_prefix != exp_tgt_prefix:
            raise ValueError(
                f"target_canonical_id prefix '{act_tgt_prefix}' does not match target_type '{self.target_type}'"
            )

        if self.source_canonical_id == self.target_canonical_id:
            raise ValueError(
                f"Self-referential relationships not permitted: '{self.source_canonical_id}'"
            )

        return self


class ScientificChunkExtractionResult(BaseModel):
    """Result of scientific entity and relationship extraction for a single text chunk."""

    chunk_id: str = Field(..., description="Canonical chunk UUID")
    document_id: str = Field(..., description="Parent document identifier")
    extraction_model: str = Field(
        default="scientific-mock-extractor-v1",
        description="Model identifier responsible for this extraction",
    )
    entities: list[ScientificExtractedEntity] = Field(
        default_factory=list,
        description="List of validated scientific entities extracted from the chunk",
    )
    relationships: list[ScientificExtractedRelationship] = Field(
        default_factory=list,
        description="List of validated scientific relationships extracted from the chunk",
    )

    model_config = ConfigDict(frozen=True)

    @field_validator("chunk_id", "document_id")
    @classmethod
    def validate_ids_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk_id and document_id cannot be empty")
        return v.strip()
