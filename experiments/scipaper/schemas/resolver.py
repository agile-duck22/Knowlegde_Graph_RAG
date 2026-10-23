"""Scientific Entity Resolver and Alias Normalizer.

Provides deterministic normalization and canonical ID assignment for scientific literature:
- 8 Entity Types: Paper, Author, Model, Dataset, Metric, Task, Method, ResearchOrganization.
- Normalizes punctuation, collapses whitespace, and applies kebab-casing.
- Explicit conflict & ambiguity detection (AmbiguousScientificEntityError).
- Pure offline resolution with zero LLM calls or database writes.
"""

from __future__ import annotations

import re

from experiments.scipaper.schemas.extraction_schemas import (
    ScientificChunkExtractionResult,
    ScientificExtractedEntity,
    ScientificExtractedRelationship,
)
from experiments.scipaper.schemas.models import (
    SCIENTIFIC_CANONICAL_PREFIXES,
    ScientificEntityType,
)


class ScientificResolverError(Exception):
    """Base exception for scientific entity resolution errors."""


class AmbiguousScientificEntityError(ScientificResolverError):
    """Raised when an alias or mention matches multiple distinct canonical entities."""


class UnknownScientificEntityError(ScientificResolverError):
    """Raised when an entity reference cannot be resolved."""


def slugify_scientific_name(name: str) -> str:
    """Normalize a display name or mention into a kebab-case slug.

    Handles diacritics and accented characters (e.g. 'Oğuz' -> 'oguz').

    Examples:
    - "Dense Passage Retrieval" -> "dense-passage-retrieval"
    - "GPT-4o (Omni)" -> "gpt-4o-omni"
    - "arXiv:2305.12345" -> "arxiv-2305.12345"
    - "Barlas Oğuz" -> "barlas-oguz"
    """
    import unicodedata

    norm = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    clean = norm.strip().lower()
    # Replace non-alphanumeric (except hyphens and periods) with hyphen
    clean = re.sub(r"[^\w\.\-]+", "-", clean)
    # Collapse multiple consecutive hyphens
    clean = re.sub(r"-+", "-", clean)
    # Strip leading/trailing hyphens or dots
    clean = clean.strip("-.")
    return clean or "unknown"


class ScientificEntityResolver:
    """Deterministic resolver for scientific literature entities and relationships."""

    def __init__(self, alias_map: dict[str, str] | None = None) -> None:
        """Initialize resolver with an optional alias -> canonical_id dictionary."""
        self._alias_to_canonical: dict[str, str] = {}
        self._alias_to_type: dict[str, ScientificEntityType] = {}
        if alias_map:
            for alias, canonical_id in alias_map.items():
                norm = slugify_scientific_name(alias)
                self._alias_to_canonical[norm] = canonical_id

    def build_canonical_id(self, entity_type: ScientificEntityType, name_or_slug: str) -> str:
        """Construct canonical ID with correct prefix."""
        prefix = SCIENTIFIC_CANONICAL_PREFIXES.get(entity_type)
        if not prefix:
            raise ValueError(f"Unknown entity type: {entity_type}")

        slug = slugify_scientific_name(name_or_slug)
        # If slug already has the prefix, avoid double prefixing
        if slug.startswith(f"{prefix}-"):
            slug = slug[len(prefix) + 1 :]
        elif slug.startswith(f"{prefix}:"):
            slug = slug[len(prefix) + 1 :]

        return f"{prefix}:{slug}"

    def register_entity(self, entity: ScientificExtractedEntity) -> None:
        """Register an entity's name and aliases for subsequent mention resolution."""
        norm_name = slugify_scientific_name(entity.name)
        if norm_name in self._alias_to_canonical:
            existing = self._alias_to_canonical[norm_name]
            if existing != entity.canonical_id:
                raise AmbiguousScientificEntityError(
                    f"Name '{entity.name}' maps to multiple canonical IDs: '{existing}' and '{entity.canonical_id}'"
                )
        self._alias_to_canonical[norm_name] = entity.canonical_id
        self._alias_to_type[norm_name] = entity.entity_type

        for alias in entity.aliases:
            norm_alias = slugify_scientific_name(alias)
            if norm_alias in self._alias_to_canonical:
                existing = self._alias_to_canonical[norm_alias]
                if existing != entity.canonical_id:
                    raise AmbiguousScientificEntityError(
                        f"Alias '{alias}' maps to multiple canonical IDs: '{existing}' and '{entity.canonical_id}'"
                    )
            self._alias_to_canonical[norm_alias] = entity.canonical_id
            self._alias_to_type[norm_alias] = entity.entity_type

    def resolve_entity(self, entity: ScientificExtractedEntity) -> ScientificExtractedEntity:
        """Resolve and standardize an entity's canonical ID."""
        norm_name = slugify_scientific_name(entity.name)
        if norm_name in self._alias_to_canonical:
            resolved_id = self._alias_to_canonical[norm_name]
        else:
            resolved_id = self.build_canonical_id(entity.entity_type, entity.name)

        if resolved_id == entity.canonical_id:
            return entity

        # Return new entity with resolved canonical ID
        return ScientificExtractedEntity(
            canonical_id=resolved_id,
            name=entity.name.strip(),
            entity_type=entity.entity_type,
            description=entity.description,
            aliases=list(entity.aliases),
            confidence=entity.confidence,
            provenance=entity.provenance,
        )

    def resolve_extraction_result(
        self, result: ScientificChunkExtractionResult
    ) -> ScientificChunkExtractionResult:
        """Resolve all entities and relationships in a chunk extraction result."""
        # 1. First pass: register all entities defined in this chunk
        for ent in result.entities:
            self.register_entity(ent)

        # 2. Second pass: resolve entities
        resolved_entities: list[ScientificExtractedEntity] = []
        id_remap: dict[str, str] = {}
        for ent in result.entities:
            resolved_ent = self.resolve_entity(ent)
            resolved_entities.append(resolved_ent)
            if resolved_ent.canonical_id != ent.canonical_id:
                id_remap[ent.canonical_id] = resolved_ent.canonical_id

        # 3. Third pass: update relationships if endpoints were remapped
        resolved_relationships: list[ScientificExtractedRelationship] = []
        for rel in result.relationships:
            new_src = id_remap.get(rel.source_canonical_id, rel.source_canonical_id)
            new_tgt = id_remap.get(rel.target_canonical_id, rel.target_canonical_id)

            if new_src == new_tgt:
                # Omit collapsed self-loops
                continue

            resolved_relationships.append(
                ScientificExtractedRelationship(
                    source_canonical_id=new_src,
                    source_type=rel.source_type,
                    target_canonical_id=new_tgt,
                    target_type=rel.target_type,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    confidence=rel.confidence,
                    source_chunk_ids=list(rel.source_chunk_ids),
                    provenance=rel.provenance,
                )
            )

        return ScientificChunkExtractionResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            extraction_model=result.extraction_model,
            entities=resolved_entities,
            relationships=resolved_relationships,
        )
