"""Context Normalizer and Token Budget Manager (TASK-401).

Coordinates deduplication, normalization, formatting, and token-budget enforcement
for hybrid context combining knowledge graph traversals and dense vector chunks.

Design Constraints & Invariants (see docs/RETRIEVAL_STRATEGY.md, docs/DATA_MODEL.md, docs/SECURITY.md):
--------------------------------------------------------------------------------------------------------
1. Dual-Store Invariant & Provenance Integrity:
   - Preserves canonical `chunk_id` (UUID format) and canonical `entity_id`s across all normalized structures.
   - Preserves all source chunk IDs on graph relationships and vector chunks for downstream citation validation.
2. Context Normalization & Deduplication:
   - Normalizes graph records into structured relationship triples and entity descriptors.
   - Deduplicates text chunks by canonical `chunk_id`. Chunks retrieved by both vector search and graph
     provenance are merged and given higher priority.
3. Deterministic Token Counting:
   - Uses exact whitespace/heuristic or injectable tokenizer callable (default: character-to-token ratio
     of 4 chars/token or exact word-level tokenization) without requiring live external tokenization APIs.
4. Deterministic Prioritization & Budget Enforcement:
   - Configurable `max_context_tokens` (sourced from Settings or override, default: 6000).
   - Prioritization order:
     1. Overlapping chunks (retrieved via both Graph provenance & Vector search).
     2. Graph structured triples / paths (high information density).
     3. High-similarity vector chunks (sorted by similarity score descending).
     4. Graph-only provenance chunks.
   - Budget enforcement is deterministic and safe against oversized items or malformed metadata.
5. Structured Output Contracts:
   - Returns `NormalizedContext` containing formatted context string, included chunk list, graph triples,
     token usage summary, and full candidate `chunk_id` set for Phase 4 citation validation.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings, get_settings
from src.retrieval.graph_retriever import GraphQueryResult
from src.retrieval.vector_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token Counting Strategy
# ---------------------------------------------------------------------------


def default_estimate_tokens(text: str) -> int:
    """Estimate token count for a text string using standard LLM character/word heuristics.

    Approximates tokens accurately:
    - Assumes ~4 characters per token for English text.
    - Uses max of character estimate (ceil(len / 4)) and word token estimate (len(words) * 1.3).
    - Minimum 1 token for non-empty text; 0 for empty text.
    """
    if not text:
        return 0
    clean_text = text.strip()
    if not clean_text:
        return 0
    char_estimate = math.ceil(len(clean_text) / 4.0)
    word_count = len(re.findall(r"\S+", clean_text))
    word_estimate = math.ceil(word_count * 1.3)
    return max(1, max(char_estimate, word_estimate))


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class NormalizedGraphTriple(BaseModel):
    """Structured graph relationship triple extracted from graph query results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(..., description="Canonical ID of the source entity.")
    source_name: str | None = Field(
        default=None, description="Human-readable name or label of source entity."
    )
    relationship_type: str = Field(
        ..., description="Type of relationship (e.g. DEPENDS_ON, OWNED_BY)."
    )
    target_id: str = Field(..., description="Canonical ID of the target entity.")
    target_name: str | None = Field(
        default=None, description="Human-readable name or label of target entity."
    )
    source_chunk_ids: list[str] = Field(
        default_factory=list, description="Provenance chunk IDs supporting this relationship."
    )
    properties: dict[str, Any] = Field(
        default_factory=dict, description="Additional relationship or path properties."
    )

    def to_markdown_row(self) -> str:
        """Render triple as a markdown table row."""
        src = f"`{self.source_id}`" + (f" ({self.source_name})" if self.source_name else "")
        tgt = f"`{self.target_id}`" + (f" ({self.target_name})" if self.target_name else "")
        rel = f"`{self.relationship_type}`"
        chunks = ", ".join(self.source_chunk_ids) if self.source_chunk_ids else "N/A"
        return f"| {src} | {rel} | {tgt} | {chunks} |"


class NormalizedGraphEntity(BaseModel):
    """Structured graph entity node details extracted from entity lookup query results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: str = Field(
        ..., description="Canonical ID of the entity (e.g. service:payment-gateway)."
    )
    name: str = Field(..., description="Entity primary name.")
    entity_type: str = Field(..., description="Entity ontology type (e.g. Service, Team).")
    description: str | None = Field(default=None, description="Entity descriptive summary.")
    aliases: list[str] = Field(default_factory=list, description="Known aliases.")
    source_chunk_ids: list[str] = Field(default_factory=list, description="Provenance chunk IDs.")

    def to_markdown_block(self) -> str:
        """Render entity details as markdown text."""
        desc = f": {self.description}" if self.description else ""
        chunks = (
            f" [Provenance: {', '.join(self.source_chunk_ids)}]" if self.source_chunk_ids else ""
        )
        return f"- **{self.name}** (`{self.entity_id}` - *{self.entity_type}*){desc}{chunks}"


class NormalizedChunkItem(BaseModel):
    """Normalized text chunk item ready for prompt inclusion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., description="Canonical UUID string of the chunk.")
    document_id: str = Field(..., description="Parent document identifier.")
    content: str = Field(..., description="Raw or formatted text content.")
    similarity_score: float | None = Field(
        default=None, description="Vector similarity score if vector-retrieved."
    )
    provenance_source: str = Field(
        default="vector",
        description="Source of chunk: 'vector', 'graph', or 'hybrid' (present in both).",
    )
    token_count: int = Field(default=0, description="Estimated token count of this chunk item.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary.")

    def to_formatted_block(self) -> str:
        """Format chunk with clear XML/Markdown tags for LLM citation injection."""
        header_parts = [f'id="{self.chunk_id}"', f'document="{self.document_id}"']
        if self.similarity_score is not None:
            header_parts.append(f'score="{self.similarity_score:.3f}"')
        header = " ".join(header_parts)
        return f"<chunk {header}>\n{self.content.strip()}\n</chunk>"


@dataclass(frozen=True)
class NormalizationInput:
    """Input container for context normalizer."""

    graph_results: Sequence[GraphQueryResult] = field(default_factory=list)
    vector_chunks: Sequence[RetrievedChunk] = field(default_factory=list)
    hydrated_graph_chunks: Sequence[RetrievedChunk] = field(default_factory=list)
    query: str = ""
    max_context_tokens: int | None = None


@dataclass(frozen=True)
class NormalizedContext:
    """Consolidated, deduplicated, and token-budgeted context ready for synthesis."""

    formatted_context: str
    chunks: list[NormalizedChunkItem]
    graph_triples: list[NormalizedGraphTriple]
    graph_entities: list[NormalizedGraphEntity]
    total_tokens: int
    max_budget_tokens: int
    candidate_chunk_ids: list[str]
    all_entity_ids: list[str]
    is_truncated: bool
    truncated_chunks_count: int
    truncated_triples_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Return True if no context was assembled."""
        return (
            len(self.chunks) == 0 and len(self.graph_triples) == 0 and len(self.graph_entities) == 0
        )


# ---------------------------------------------------------------------------
# Context Normalizer & Token Budget Manager Implementation
# ---------------------------------------------------------------------------


class ContextNormalizer:
    """Normalizes, deduplicates, and bounds graph and vector retrieval context within token budgets."""

    def __init__(
        self,
        settings: Settings | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        """Initialize ContextNormalizer with settings and optional token estimator."""
        self.settings = settings or get_settings()
        self._estimate_tokens = token_estimator or default_estimate_tokens

    def normalize(self, input_data: NormalizationInput) -> NormalizedContext:
        """Normalize, deduplicate, prioritize, and enforce token budget on retrieved context."""
        max_tokens = input_data.max_context_tokens or self.settings.max_context_tokens

        # 1. Extract and normalize graph triples and entity node records
        triples, entities = self._extract_graph_items(input_data.graph_results)

        # 2. Extract all referenced entity IDs
        all_entity_ids = self._extract_all_entity_ids(input_data.graph_results, triples, entities)

        # 3. Deduplicate and score text chunks
        normalized_chunks = self._deduplicate_chunks(
            vector_chunks=input_data.vector_chunks,
            hydrated_graph_chunks=input_data.hydrated_graph_chunks,
            graph_results=input_data.graph_results,
        )

        # 4. Sort and prioritize chunks deterministically
        prioritized_chunks = self._prioritize_chunks(normalized_chunks)

        # 5. Assemble and enforce token budget
        return self._assemble_and_budget(
            triples=triples,
            entities=entities,
            chunks=prioritized_chunks,
            max_tokens=max_tokens,
            all_entity_ids=all_entity_ids,
        )

    # -----------------------------------------------------------------------
    # Internal Extraction & Deduplication Helpers
    # -----------------------------------------------------------------------

    def _extract_graph_items(
        self, graph_results: Sequence[GraphQueryResult]
    ) -> tuple[list[NormalizedGraphTriple], list[NormalizedGraphEntity]]:
        """Extract structured relationship triples and entity details from raw GraphQueryResult records."""
        triples: list[NormalizedGraphTriple] = []
        entities: list[NormalizedGraphEntity] = []
        seen_triple_keys: set[tuple[str, str, str]] = set()
        seen_entity_ids: set[str] = set()

        for res in graph_results:
            for record in res.records:
                extracted_triples, extracted_entities = self._parse_single_record(
                    record, res.source_chunk_ids
                )
                for triple in extracted_triples:
                    key = (triple.source_id, triple.relationship_type, triple.target_id)
                    if key not in seen_triple_keys:
                        seen_triple_keys.add(key)
                        triples.append(triple)
                for entity in extracted_entities:
                    if entity.entity_id not in seen_entity_ids:
                        seen_entity_ids.add(entity.entity_id)
                        entities.append(entity)

        return triples, entities

    def _parse_single_record(
        self, record: dict[str, Any], default_chunk_ids: list[str]
    ) -> tuple[list[NormalizedGraphTriple], list[NormalizedGraphEntity]]:
        """Parse individual Neo4j result record into NormalizedGraphTriple and NormalizedGraphEntity objects."""
        triples: list[NormalizedGraphTriple] = []
        entities: list[NormalizedGraphEntity] = []

        # Case 1: Standard Cypher relationship return (source, relationship, target)
        src = record.get("source") or record.get("source_id") or record.get("service")
        rel = (
            record.get("relation")
            or record.get("relationship")
            or record.get("relationship_type")
            or record.get("type")
        )
        tgt = (
            record.get("target")
            or record.get("target_id")
            or record.get("dependency")
            or record.get("database")
            or record.get("infra")
        )

        # Check for extracted node dictionaries vs string IDs
        src_id = self._extract_node_id(src)
        src_name = self._extract_node_name(src)
        tgt_id = self._extract_node_id(tgt)
        tgt_name = self._extract_node_name(tgt)
        rel_type = str(rel) if rel is not None else "CONNECTED_TO"

        # Provenance chunk IDs from record or default
        rec_chunk_ids = (
            record.get("source_chunk_ids")
            or record.get("provenance_chunk_ids")
            or record.get("chunk_ids")
            or default_chunk_ids
        )
        if isinstance(rec_chunk_ids, str):
            rec_chunk_ids = [rec_chunk_ids]
        elif not isinstance(rec_chunk_ids, list):
            rec_chunk_ids = (
                list(rec_chunk_ids) if isinstance(rec_chunk_ids, Sequence) else list[Any]()
            )
        clean_chunk_ids = [str(c).strip() for c in rec_chunk_ids if str(c).strip()]

        if src_id and tgt_id:
            triples.append(
                NormalizedGraphTriple(
                    source_id=src_id,
                    source_name=src_name,
                    relationship_type=rel_type,
                    target_id=tgt_id,
                    target_name=tgt_name,
                    source_chunk_ids=clean_chunk_ids,
                    properties={
                        k: v
                        for k, v in record.items()
                        if k not in ("source", "relation", "target", "source_chunk_ids")
                    },
                )
            )

        # Case 1b: T-03 (Team Ownership & Service Impact: team_id -> service_id)
        team_id = record.get("team_id")
        svc_id = record.get("service_id")
        if team_id and svc_id and not (src_id and tgt_id):
            t_chunk_ids = (
                record.get("ownership_chunk_ids")
                or record.get("source_chunk_ids")
                or default_chunk_ids
            )
            clean_t_chunks = (
                [str(c).strip() for c in t_chunk_ids if str(c).strip()]
                if isinstance(t_chunk_ids, (list, tuple, set))
                else clean_chunk_ids
            )
            triples.append(
                NormalizedGraphTriple(
                    source_id=str(team_id).strip(),
                    source_name=str(record.get("team_name") or team_id).strip(),
                    relationship_type="OWNS",
                    target_id=str(svc_id).strip(),
                    target_name=str(record.get("service_name") or svc_id).strip(),
                    source_chunk_ids=clean_t_chunks,
                )
            )

        # Case 1c: T-02 (Multi-Hop Dependency Path: rel_start_id -> rel_end_id)
        r_start = record.get("rel_start_id")
        r_end = record.get("rel_end_id")
        if r_start and r_end and not (src_id and tgt_id):
            triples.append(
                NormalizedGraphTriple(
                    source_id=str(r_start).strip(),
                    source_name=None,
                    relationship_type=str(record.get("rel_type") or "DEPENDS_ON").strip(),
                    target_id=str(r_end).strip(),
                    target_name=None,
                    source_chunk_ids=clean_chunk_ids,
                )
            )

        # Case 1d: T-04 (Vulnerability Blast Radius: vulnerable_service_id -> vuln_id and repo -> service)
        vuln_id = record.get("vuln_id")
        vuln_svc_id = record.get("vulnerable_service_id")
        if vuln_id and vuln_svc_id and not (src_id and tgt_id):
            v_chunk_ids = (
                record.get("vuln_chunk_ids") or record.get("source_chunk_ids") or default_chunk_ids
            )
            clean_v_chunks = (
                [str(c).strip() for c in v_chunk_ids if str(c).strip()]
                if isinstance(v_chunk_ids, (list, tuple, set))
                else clean_chunk_ids
            )
            triples.append(
                NormalizedGraphTriple(
                    source_id=str(vuln_svc_id).strip(),
                    source_name=str(record.get("vulnerable_service_name") or vuln_svc_id).strip(),
                    relationship_type="AFFECTED_BY",
                    target_id=str(vuln_id).strip(),
                    target_name=str(record.get("vuln_name") or vuln_id).strip(),
                    source_chunk_ids=clean_v_chunks,
                )
            )
            # Process repositories storing code for the impacted service
            for repo in record.get("repositories") or []:
                if isinstance(repo, dict) and repo.get("id"):
                    repo_id = str(repo["id"]).strip()
                    repo_name = str(repo.get("name") or repo_id).strip()
                    r_chunks = [
                        str(c).strip()
                        for c in (repo.get("chunk_ids") or default_chunk_ids)
                        if str(c).strip()
                    ]
                    triples.append(
                        NormalizedGraphTriple(
                            source_id=repo_id,
                            source_name=repo_name,
                            relationship_type="STORES_CODE",
                            target_id=str(vuln_svc_id).strip(),
                            target_name=str(
                                record.get("vulnerable_service_name") or vuln_svc_id
                            ).strip(),
                            source_chunk_ids=r_chunks,
                        )
                    )

        # Case 1e: T-05 (Service Dependency Chain: root_service_id -> downstream_service_id)
        root_svc_id = record.get("root_service_id")
        down_svc_id = record.get("downstream_service_id")
        if root_svc_id and down_svc_id and not (src_id and tgt_id):
            triples.append(
                NormalizedGraphTriple(
                    source_id=str(root_svc_id).strip(),
                    source_name=None,
                    relationship_type=str(record.get("rel_type") or "DEPENDS_ON").strip(),
                    target_id=str(down_svc_id).strip(),
                    target_name=str(record.get("downstream_service_name") or down_svc_id).strip(),
                    source_chunk_ids=clean_chunk_ids,
                )
            )
            # Process downstream datastores
            for ds in record.get("datastores") or []:
                if isinstance(ds, dict) and ds.get("id"):
                    ds_id = str(ds["id"]).strip()
                    ds_name = str(ds.get("name") or ds_id).strip()
                    ds_rel = str(ds.get("rel") or "READS_FROM").strip()
                    ds_chunks = [
                        str(c).strip()
                        for c in (ds.get("chunk_ids") or default_chunk_ids)
                        if str(c).strip()
                    ]
                    triples.append(
                        NormalizedGraphTriple(
                            source_id=str(down_svc_id).strip(),
                            source_name=str(
                                record.get("downstream_service_name") or down_svc_id
                            ).strip(),
                            relationship_type=ds_rel,
                            target_id=ds_id,
                            target_name=ds_name,
                            source_chunk_ids=ds_chunks,
                        )
                    )
            # Process downstream infrastructure
            for inf in record.get("infrastructures") or []:
                if isinstance(inf, dict) and inf.get("id"):
                    down_inf_id = str(inf["id"]).strip()
                    down_inf_name = str(inf.get("name") or down_inf_id).strip()
                    down_inf_rel = str(inf.get("rel") or "DEPLOYED_ON").strip()
                    down_inf_chunks = [
                        str(c).strip()
                        for c in (inf.get("chunk_ids") or default_chunk_ids)
                        if str(c).strip()
                    ]
                    triples.append(
                        NormalizedGraphTriple(
                            source_id=str(down_svc_id).strip(),
                            source_name=str(
                                record.get("downstream_service_name") or down_svc_id
                            ).strip(),
                            relationship_type=down_inf_rel,
                            target_id=down_inf_id,
                            target_name=down_inf_name,
                            source_chunk_ids=down_inf_chunks,
                        )
                    )

        # Case 1h: T-09 (Incoming Service Dependents: caller_service_id -> service_id)
        caller_svc_id = record.get("caller_service_id")
        target_svc_id = record.get("service_id")
        if caller_svc_id and target_svc_id and not (src_id and tgt_id):
            caller_chunks_nested = record.get("caller_chunk_ids") or []
            caller_chunks_flat: list[str] = []
            if isinstance(caller_chunks_nested, list):
                for item in caller_chunks_nested:
                    if isinstance(item, list):
                        caller_chunks_flat.extend(str(c).strip() for c in item if str(c).strip())
                    elif str(item).strip():
                        caller_chunks_flat.append(str(item).strip())
            caller_chunks = caller_chunks_flat or clean_chunk_ids
            triples.append(
                NormalizedGraphTriple(
                    source_id=str(caller_svc_id).strip(),
                    source_name=str(record.get("caller_service_name") or caller_svc_id).strip(),
                    relationship_type="DEPENDS_ON",
                    target_id=str(target_svc_id).strip(),
                    target_name=None,
                    source_chunk_ids=caller_chunks,
                )
            )
            # Process caller datastores
            for ds in record.get("datastores") or []:
                if isinstance(ds, dict) and ds.get("id"):
                    ds_id = str(ds["id"]).strip()
                    ds_name = str(ds.get("name") or ds_id).strip()
                    ds_rel = str(ds.get("rel") or "WRITES_TO").strip()
                    ds_chunks = [
                        str(c).strip()
                        for c in (ds.get("chunk_ids") or default_chunk_ids)
                        if str(c).strip()
                    ]
                    triples.append(
                        NormalizedGraphTriple(
                            source_id=str(caller_svc_id).strip(),
                            source_name=str(
                                record.get("caller_service_name") or caller_svc_id
                            ).strip(),
                            relationship_type=ds_rel,
                            target_id=ds_id,
                            target_name=ds_name,
                            source_chunk_ids=ds_chunks,
                        )
                    )

        # Case 1f: T-06 (Database Producers & Consumers: readers / writers -> database_id)
        db_id = record.get("database_id")
        if db_id and not (src_id and tgt_id):
            db_name = str(record.get("database_name") or db_id).strip()
            for r in record.get("readers") or []:
                if isinstance(r, dict) and r.get("service_id"):
                    r_chunks = [
                        str(c).strip()
                        for c in (r.get("chunk_ids") or default_chunk_ids)
                        if str(c).strip()
                    ]
                    triples.append(
                        NormalizedGraphTriple(
                            source_id=str(r["service_id"]).strip(),
                            source_name=str(r.get("service_name") or r["service_id"]).strip(),
                            relationship_type="READS_FROM",
                            target_id=str(db_id).strip(),
                            target_name=db_name,
                            source_chunk_ids=r_chunks,
                        )
                    )
            for w in record.get("writers") or []:
                if isinstance(w, dict) and w.get("service_id"):
                    w_chunks = [
                        str(c).strip()
                        for c in (w.get("chunk_ids") or default_chunk_ids)
                        if str(c).strip()
                    ]
                    triples.append(
                        NormalizedGraphTriple(
                            source_id=str(w["service_id"]).strip(),
                            source_name=str(w.get("service_name") or w["service_id"]).strip(),
                            relationship_type="WRITES_TO",
                            target_id=str(db_id).strip(),
                            target_name=db_name,
                            source_chunk_ids=w_chunks,
                        )
                    )

        # Case 1g: T-07 (Infrastructure Deployments: deployed_components / event_producers / owners / connected_workloads)
        inf_id = record.get("infra_id")
        if inf_id and not (src_id and tgt_id):
            inf_name = str(record.get("infra_name") or inf_id).strip()
            # 1. Infra owner teams
            for o in record.get("owners") or []:
                if isinstance(o, dict) and o.get("team_id"):
                    o_team_id = str(o["team_id"]).strip()
                    o_team_name = str(o.get("team_name") or o_team_id).strip()
                    o_chunks = [
                        str(c).strip()
                        for c in (o.get("chunk_ids") or default_chunk_ids)
                        if str(c).strip()
                    ]
                    triples.append(
                        NormalizedGraphTriple(
                            source_id=str(inf_id).strip(),
                            source_name=inf_name,
                            relationship_type="OWNED_BY",
                            target_id=o_team_id,
                            target_name=o_team_name,
                            source_chunk_ids=o_chunks,
                        )
                    )
            # 2. Deployed components
            for d in record.get("deployed_components") or []:
                if isinstance(d, dict) and (d.get("entity_id") or d.get("entity_name")):
                    comp_id = str(d.get("entity_id") or d.get("entity_name")).strip()
                    d_chunks = [
                        str(c).strip()
                        for c in (d.get("chunk_ids") or default_chunk_ids)
                        if str(c).strip()
                    ]
                    triples.append(
                        NormalizedGraphTriple(
                            source_id=comp_id,
                            source_name=str(d.get("entity_name") or comp_id).strip(),
                            relationship_type="DEPLOYED_ON",
                            target_id=str(inf_id).strip(),
                            target_name=inf_name,
                            source_chunk_ids=d_chunks,
                        )
                    )
            # 3. Connected workloads (client services accessing deployed datastores and their owning teams)
            for w in record.get("connected_workloads") or []:
                if isinstance(w, dict) and w.get("service_id"):
                    svc_id = str(w["service_id"]).strip()
                    svc_name = str(w.get("service_name") or svc_id).strip()
                    tgt_db_id = str(w.get("target_id") or "").strip()
                    tgt_db_name = str(w.get("target_name") or tgt_db_id).strip()
                    rel_type_db = str(w.get("rel") or "READS_FROM").strip()
                    w_chunks = [
                        str(c).strip()
                        for c in (w.get("chunk_ids") or default_chunk_ids)
                        if str(c).strip()
                    ]
                    if tgt_db_id:
                        triples.append(
                            NormalizedGraphTriple(
                                source_id=svc_id,
                                source_name=svc_name,
                                relationship_type=rel_type_db,
                                target_id=tgt_db_id,
                                target_name=tgt_db_name,
                                source_chunk_ids=w_chunks,
                            )
                        )
                    if w.get("team_id"):
                        team_id = str(w["team_id"]).strip()
                        team_name = str(w.get("team_name") or team_id).strip()
                        triples.append(
                            NormalizedGraphTriple(
                                source_id=svc_id,
                                source_name=svc_name,
                                relationship_type="OWNED_BY",
                                target_id=team_id,
                                target_name=team_name,
                                source_chunk_ids=w_chunks,
                            )
                        )

        # Case 2: Multi-hop paths or lists of dependencies
        deps = (
            record.get("dependencies")
            or record.get("affected_dependents")
            or record.get("producers")
            or record.get("consumers")
        )
        if src_id and isinstance(deps, list):
            for dep in deps:
                dep_id = self._extract_node_id(dep)
                dep_name = self._extract_node_name(dep)
                if dep_id and dep_id != src_id:
                    triples.append(
                        NormalizedGraphTriple(
                            source_id=src_id,
                            source_name=src_name,
                            relationship_type=rel_type,
                            target_id=dep_id,
                            target_name=dep_name,
                            source_chunk_ids=clean_chunk_ids,
                        )
                    )

        # Case 3: Entity Node Lookup (T-08 exact entity lookup with entity_id, name, entity_type)
        ent_id = record.get("entity_id") or (
            record.get("id") if not src_id and not tgt_id else None
        )
        if ent_id and not tgt_id:
            clean_ent_id = str(ent_id).strip()
            ent_name = str(record.get("name") or clean_ent_id).strip()
            ent_type = str(record.get("entity_type") or record.get("type") or "Entity").strip()
            ent_desc = str(record.get("description")).strip() if record.get("description") else None
            aliases = record.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            elif not isinstance(aliases, list):
                aliases = list(aliases) if isinstance(aliases, Sequence) else list[Any]()

            entities.append(
                NormalizedGraphEntity(
                    entity_id=clean_ent_id,
                    name=ent_name,
                    entity_type=ent_type,
                    description=ent_desc,
                    aliases=[str(a).strip() for a in aliases if str(a).strip()],
                    source_chunk_ids=clean_chunk_ids,
                )
            )

        return triples, entities

    def _extract_node_id(self, node: Any) -> str | None:
        """Extract canonical ID string from a node dictionary, entity model, or string."""
        if not node:
            return None
        if isinstance(node, str):
            return node.strip()
        if isinstance(node, dict):
            val = (
                node.get("id")
                or node.get("canonical_id")
                or node.get("entity_id")
                or node.get("name")
            )
            return str(val).strip() if val else None
        if hasattr(node, "id"):
            return str(node.id).strip()
        if hasattr(node, "canonical_id"):
            return str(node.canonical_id).strip()
        return None

    def _extract_node_name(self, node: Any) -> str | None:
        """Extract name string from a node dictionary or entity model."""
        if not node:
            return None
        if isinstance(node, dict):
            name = node.get("name")
            return str(name).strip() if name else None
        if hasattr(node, "name"):
            name_val = node.name
            return str(name_val).strip() if name_val else None
        return None

    def _extract_all_entity_ids(
        self,
        graph_results: Sequence[GraphQueryResult],
        triples: Sequence[NormalizedGraphTriple],
        entities: Sequence[NormalizedGraphEntity],
    ) -> list[str]:
        """Aggregate all distinct canonical entity IDs present across results."""
        entity_ids: set[str] = set()
        for res in graph_results:
            entity_ids.update(res.entity_ids)
        for t in triples:
            if t.source_id:
                entity_ids.add(t.source_id)
            if t.target_id:
                entity_ids.add(t.target_id)
        for e in entities:
            if e.entity_id:
                entity_ids.add(e.entity_id)
        return sorted(entity_ids)

    def _deduplicate_chunks(
        self,
        vector_chunks: Sequence[RetrievedChunk],
        hydrated_graph_chunks: Sequence[RetrievedChunk],
        graph_results: Sequence[GraphQueryResult],
    ) -> dict[str, NormalizedChunkItem]:
        """Deduplicate chunks from all sources and compute combined provenance tags and token counts."""
        chunk_map: dict[str, NormalizedChunkItem] = {}

        # Collect graph provenance chunk IDs
        graph_provenance_ids: set[str] = set()
        for res in graph_results:
            graph_provenance_ids.update(res.source_chunk_ids)

        # 1. Process vector chunks
        for chunk in vector_chunks:
            cid = chunk.chunk_id.strip()
            if not cid:
                continue
            is_in_graph = cid in graph_provenance_ids
            prov_source = "hybrid" if is_in_graph else "vector"

            content = chunk.content or ""
            tokens = self._estimate_tokens(content)

            chunk_map[cid] = NormalizedChunkItem(
                chunk_id=cid,
                document_id=chunk.document_id,
                content=content,
                similarity_score=chunk.similarity_score,
                provenance_source=prov_source,
                token_count=tokens,
                metadata=dict(chunk.metadata) if chunk.metadata else {},
            )

        # 2. Process hydrated graph chunks
        for chunk in hydrated_graph_chunks:
            cid = chunk.chunk_id.strip()
            if not cid:
                continue
            if cid in chunk_map:
                existing = chunk_map[cid]
                # Merge into hybrid status with preserved best similarity score
                chunk_map[cid] = NormalizedChunkItem(
                    chunk_id=cid,
                    document_id=existing.document_id,
                    content=existing.content or chunk.content,
                    similarity_score=existing.similarity_score or chunk.similarity_score,
                    provenance_source="hybrid",
                    token_count=existing.token_count or self._estimate_tokens(chunk.content),
                    metadata={**existing.metadata, **(chunk.metadata or {})},
                )
            else:
                tokens = self._estimate_tokens(chunk.content)
                chunk_map[cid] = NormalizedChunkItem(
                    chunk_id=cid,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    similarity_score=chunk.similarity_score,
                    provenance_source="graph",
                    token_count=tokens,
                    metadata=dict(chunk.metadata) if chunk.metadata else {},
                )

        return chunk_map

    def _prioritize_chunks(
        self, chunk_map: dict[str, NormalizedChunkItem]
    ) -> list[NormalizedChunkItem]:
        """Sort normalized chunks with deterministic priority order."""

        # Priority sort order:
        # Tier 0: hybrid chunks (highest relevance)
        # Tier 1: vector chunks with high similarity score
        # Tier 2: graph provenance chunks
        def sort_key(item: NormalizedChunkItem) -> tuple[int, float, str]:
            if item.provenance_source == "hybrid":
                tier = 0
            elif item.provenance_source == "vector":
                tier = 1
            else:
                tier = 2

            # Negate score so higher scores sort earlier; default 0.0 if None
            score = -(item.similarity_score if item.similarity_score is not None else 0.0)
            return (tier, score, item.chunk_id)

        return sorted(chunk_map.values(), key=sort_key)

    # -----------------------------------------------------------------------
    # Budget Assembly & Truncation
    # -----------------------------------------------------------------------

    def _assemble_and_budget(
        self,
        triples: list[NormalizedGraphTriple],
        entities: list[NormalizedGraphEntity],
        chunks: list[NormalizedChunkItem],
        max_tokens: int,
        all_entity_ids: list[str],
    ) -> NormalizedContext:
        """Fit graph entities, triples, and text chunks into the context token budget."""
        included_triples: list[NormalizedGraphTriple] = []
        included_entities: list[NormalizedGraphEntity] = []
        included_chunks: list[NormalizedChunkItem] = []

        current_token_count = 0
        truncated_triples_count = 0
        truncated_chunks_count = 0

        # Header templates
        graph_header = "### Knowledge Graph Context\n| Source | Relation | Target | Provenance Chunks |\n| :--- | :--- | :--- | :--- |\n"
        graph_header_tokens = self._estimate_tokens(graph_header)

        entity_header = "### Entity Descriptions\n"
        entity_header_tokens = self._estimate_tokens(entity_header)

        chunks_header = "\n### Retrieved Document Chunks\n"
        chunks_header_tokens = self._estimate_tokens(chunks_header)

        # 1. Fit Graph Entities
        temp_entity_rows: list[str] = []
        if entities:
            if current_token_count + entity_header_tokens <= max_tokens:
                current_token_count += entity_header_tokens
                has_entity_header = True
            else:
                has_entity_header = False

            for entity in entities:
                block = entity.to_markdown_block()
                block_tokens = self._estimate_tokens(block + "\n")
                if current_token_count + block_tokens <= max_tokens:
                    included_entities.append(entity)
                    temp_entity_rows.append(block)
                    current_token_count += block_tokens
                else:
                    truncated_triples_count += 1

            if not included_entities and has_entity_header:
                current_token_count -= entity_header_tokens

        # 2. Fit Graph Triples (high information density)
        temp_triple_rows: list[str] = []
        if triples:
            if current_token_count + graph_header_tokens <= max_tokens:
                current_token_count += graph_header_tokens
                has_graph_header = True
            else:
                has_graph_header = False

            for triple in triples:
                row = triple.to_markdown_row()
                row_tokens = self._estimate_tokens(row + "\n")
                if current_token_count + row_tokens <= max_tokens:
                    included_triples.append(triple)
                    temp_triple_rows.append(row)
                    current_token_count += row_tokens
                else:
                    truncated_triples_count += 1

            if not included_triples and has_graph_header:
                current_token_count -= graph_header_tokens

        # 3. Fit Text Chunks
        temp_chunk_blocks: list[str] = []
        if chunks:
            if current_token_count + chunks_header_tokens <= max_tokens:
                current_token_count += chunks_header_tokens
                has_chunk_header = True
            else:
                has_chunk_header = False

            for chunk in chunks:
                block = chunk.to_formatted_block()
                block_tokens = self._estimate_tokens(block + "\n\n")
                if current_token_count + block_tokens <= max_tokens:
                    included_chunks.append(chunk)
                    temp_chunk_blocks.append(block)
                    current_token_count += block_tokens
                else:
                    truncated_chunks_count += 1

            if not included_chunks and has_chunk_header:
                current_token_count -= chunks_header_tokens

        # 4. Build Formatted String
        sections: list[str] = []
        if included_entities:
            sections.append(entity_header + "\n".join(temp_entity_rows))
        if included_triples:
            sections.append(graph_header + "\n".join(temp_triple_rows))
        if included_chunks:
            sections.append(chunks_header + "\n\n".join(temp_chunk_blocks))

        formatted_context = "\n\n".join(sections).strip()
        final_token_count = self._estimate_tokens(formatted_context)

        # Candidate chunk IDs for citation validation includes all chunks presented in context
        # plus provenance chunk IDs attached to included triples and entities
        candidate_chunk_ids_set: set[str] = {c.chunk_id for c in included_chunks}
        for t in included_triples:
            candidate_chunk_ids_set.update(t.source_chunk_ids)
        for e in included_entities:
            candidate_chunk_ids_set.update(e.source_chunk_ids)
        candidate_chunk_ids = sorted(candidate_chunk_ids_set)

        is_truncated = (truncated_triples_count > 0) or (truncated_chunks_count > 0)

        return NormalizedContext(
            formatted_context=formatted_context,
            chunks=included_chunks,
            graph_triples=included_triples,
            graph_entities=included_entities,
            total_tokens=final_token_count,
            max_budget_tokens=max_tokens,
            candidate_chunk_ids=candidate_chunk_ids,
            all_entity_ids=all_entity_ids,
            is_truncated=is_truncated,
            truncated_chunks_count=truncated_chunks_count,
            truncated_triples_count=truncated_triples_count,
            metadata={
                "initial_triples_count": len(triples),
                "initial_entities_count": len(entities),
                "initial_chunks_count": len(chunks),
                "included_triples_count": len(included_triples),
                "included_entities_count": len(included_entities),
                "included_chunks_count": len(included_chunks),
            },
        )
