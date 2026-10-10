"""Deterministic Entity Resolver and Alias Normalizer.

This module resolves raw mentions, abbreviations, and aliases to canonical
entity identifiers conforming to the ontology defined in ``docs/ONTOLOGY.md``.

Design & Invariants
-------------------
- Strict Ontology Compliance: Maps entities to the 7 core entity types:
  Service, Team, API, Database, Repository, Vulnerability, Infrastructure.
- Deterministic Normalization: Case folding, whitespace collapse, punctuation cleanup,
  and kebab-casing without fuzzy randomness.
- Explicit Conflict & Ambiguity Handling: If a mention or alias matches multiple
  distinct canonical entities, raises ``AmbiguousEntityError`` rather than guessing.
- Provenance Preservation: Does not mutate input extraction objects in place.
- Pure Offline Resolution: No LLM calls or database writes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.extraction.schemas import (
    _CANONICAL_ID_PREFIX,
    _CANONICAL_ID_RE,
    ChunkExtractionResult,
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ResolverError(Exception):
    """Base exception for entity resolution errors."""


class AmbiguousEntityError(ResolverError):
    """Raised when an alias or mention matches multiple distinct canonical entities."""


class UnknownEntityError(ResolverError):
    """Raised when an entity reference cannot be resolved and allow_unresolved is False."""


# Suffix patterns for Team mentions (e.g., " Eng team", " Team", " Engineering")
_TEAM_SUFFIX_RE = re.compile(
    r"\s+\b(engineering\s+team|eng\s+team|team|engineering|tech|sre|secops|devops)\b.*$",
    re.IGNORECASE,
)

# Suffix patterns for Service mentions (e.g., " Service", " Svc", " Core")
_SERVICE_SUFFIX_RE = re.compile(
    r"\s+\b(service|svc|core|engine|dispatcher|gateway|relay|adapter|indexer|aggregator)\b.*$",
    re.IGNORECASE,
)

# CVE regex pattern for Vulnerability normalization
_CVE_RE = re.compile(r"(?:cve[-_\s]*)?(\d{4}[-_]\d{4,7})", re.IGNORECASE)

# Prefix to EntityType map (inverse of _CANONICAL_ID_PREFIX)
_PREFIX_TO_ENTITY_TYPE: dict[str, EntityType] = {
    v: k for k, v in _CANONICAL_ID_PREFIX.items()  # type: ignore[misc]
}


def normalize_alias_key(raw_name: str) -> str:
    """Normalize a raw mention string into a deterministic lookup key.

    Performs case folding, replaces underscores/slashes/dots with spaces,
    collapses multiple spaces, and strips leading/trailing whitespace.
    """
    if not raw_name:
        return ""
    # Lowercase and replace punctuation separators with single spaces
    text = raw_name.lower().strip()
    text = re.sub(r"[\t\n\r_\-/.:]+", " ", text)
    # Collapse multiple whitespace characters
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_kebab_case(text: str) -> str:
    """Convert text into standard lowercase kebab-case slug."""
    text = text.strip().lower()
    # Replace non-alphanumeric chars with hyphen
    text = re.sub(r"[^a-z0-9]+", "-", text)
    # Strip leading and trailing hyphens
    text = re.sub(r"^-+|-+$", "", text)
    return text


@dataclass(frozen=True)
class ResolvedEntityTarget:
    """Target resolution metadata."""

    canonical_id: str
    entity_type: EntityType
    primary_name: str


# Default seed alias dictionary based on ONTOLOGY.md §3 & §4
DEFAULT_ALIAS_MAP: dict[str, tuple[str, EntityType, str]] = {
    # Services
    "payment svc": ("service:payment-gateway", "Service", "Payment Gateway"),
    "pay svc": ("service:payment-gateway", "Service", "Payment Gateway"),
    "pay gw": ("service:payment-gateway", "Service", "Payment Gateway"),
    "payment gateway": ("service:payment-gateway", "Service", "Payment Gateway"),
    "payment service": ("service:payment-gateway", "Service", "Payment Gateway"),
    "checkout svc": ("service:checkout-service", "Service", "Checkout Service"),
    "checkout service": ("service:checkout-service", "Service", "Checkout Service"),
    "checkout core": ("service:checkout-service", "Service", "Checkout Service"),
    "auth svc": ("service:auth-service", "Service", "Authentication Service"),
    "auth service": ("service:auth-service", "Service", "Authentication Service"),
    "iam core": ("service:auth-service", "Service", "Authentication Service"),
    "oauth service": ("service:auth-service", "Service", "Authentication Service"),
    "inventory svc": ("service:inventory-service", "Service", "Inventory Reservation Service"),
    "inventory service": ("service:inventory-service", "Service", "Inventory Reservation Service"),
    "stock reservation service": ("service:inventory-service", "Service", "Inventory Reservation Service"),
    "catalog svc": ("service:catalog-service", "Service", "Product Catalog Service"),
    "catalog service": ("service:catalog-service", "Service", "Product Catalog Service"),
    "product registry": ("service:catalog-service", "Service", "Product Catalog Service"),
    "notification svc": ("service:notification-service", "Service", "Notification Dispatch Service"),
    "notification service": ("service:notification-service", "Service", "Notification Dispatch Service"),
    "comms dispatcher": ("service:notification-service", "Service", "Notification Dispatch Service"),
    "billing svc": ("service:billing-service", "Service", "Billing and Invoicing Service"),
    "billing service": ("service:billing-service", "Service", "Billing and Invoicing Service"),
    "ledger engine": ("service:billing-service", "Service", "Billing and Invoicing Service"),
    "warehouse svc": ("service:warehouse-service", "Service", "Warehouse Management Service"),
    "warehouse service": ("service:warehouse-service", "Service", "Warehouse Management Service"),
    "wms core": ("service:warehouse-service", "Service", "Warehouse Management Service"),
    "sms relay": ("service:sms-relay-service", "Service", "SMS Relay Gateway"),
    "sms relay service": ("service:sms-relay-service", "Service", "SMS Relay Gateway"),
    "twilio gateway": ("service:sms-relay-service", "Service", "SMS Relay Gateway"),
    "tax svc": ("service:tax-calculator-service", "Service", "Tax Calculation Service"),
    "tax calculator service": ("service:tax-calculator-service", "Service", "Tax Calculation Service"),
    "avalara adapter": ("service:tax-calculator-service", "Service", "Tax Calculation Service"),
    "user dir": ("service:user-directory-service", "Service", "User Directory Service"),
    "user directory service": ("service:user-directory-service", "Service", "User Directory Service"),
    "account registry": ("service:user-directory-service", "Service", "User Directory Service"),
    "cart svc": ("service:cart-service", "Service", "Shopping Cart Service"),
    "cart service": ("service:cart-service", "Service", "Shopping Cart Service"),
    "basket engine": ("service:cart-service", "Service", "Shopping Cart Service"),
    "search indexer": ("service:search-indexer", "Service", "Search Pipeline Indexer"),
    "elastic ingestion": ("service:search-indexer", "Service", "Search Pipeline Indexer"),
    "data pipeline": ("service:data-pipeline-service", "Service", "Data Pipeline Ingestion Engine"),
    "data pipeline service": ("service:data-pipeline-service", "Service", "Data Pipeline Ingestion Engine"),
    "etl engine": ("service:data-pipeline-service", "Service", "Data Pipeline Ingestion Engine"),
    "analytics aggregator": ("service:analytics-aggregator", "Service", "Analytics Aggregator Service"),
    "bi metric engine": ("service:analytics-aggregator", "Service", "Analytics Aggregator Service"),
    # Teams
    "core checkout": ("team:checkout-engineering", "Team", "Checkout Engineering Team"),
    "checkout team": ("team:checkout-engineering", "Team", "Checkout Engineering Team"),
    "checkout eng": ("team:checkout-engineering", "Team", "Checkout Engineering Team"),
    "checkout engineering": ("team:checkout-engineering", "Team", "Checkout Engineering Team"),
    "checkout engineering team": ("team:checkout-engineering", "Team", "Checkout Engineering Team"),
    "iam team": ("team:identity-access", "Team", "Identity and Access Management Team"),
    "auth team": ("team:identity-access", "Team", "Identity and Access Management Team"),
    "identity team": ("team:identity-access", "Team", "Identity and Access Management Team"),
    "identity and access management team": ("team:identity-access", "Team", "Identity and Access Management Team"),
    "data infra": ("team:data-platform", "Team", "Data Platform Team"),
    "data platform eng": ("team:data-platform", "Team", "Data Platform Team"),
    "data platform team": ("team:data-platform", "Team", "Data Platform Team"),
    "notification team": ("team:notification-messaging", "Team", "Notification and Messaging Team"),
    "comms team": ("team:notification-messaging", "Team", "Notification and Messaging Team"),
    "messaging eng": ("team:notification-messaging", "Team", "Notification and Messaging Team"),
    "notification and messaging team": ("team:notification-messaging", "Team", "Notification and Messaging Team"),
    "warehouse team": ("team:inventory-fulfillment", "Team", "Inventory and Fulfillment Team"),
    "inventory eng": ("team:inventory-fulfillment", "Team", "Inventory and Fulfillment Team"),
    "logistics team": ("team:inventory-fulfillment", "Team", "Inventory and Fulfillment Team"),
    "inventory and fulfillment team": ("team:inventory-fulfillment", "Team", "Inventory and Fulfillment Team"),
    "search team": ("team:catalog-search", "Team", "Catalog and Search Team"),
    "catalog eng": ("team:catalog-search", "Team", "Catalog and Search Team"),
    "discovery team": ("team:catalog-search", "Team", "Catalog and Search Team"),
    "catalog and search team": ("team:catalog-search", "Team", "Catalog and Search Team"),
    "security team": ("team:security-secops", "Team", "Enterprise Security and SecOps Team"),
    "infosec": ("team:security-secops", "Team", "Enterprise Security and SecOps Team"),
    "secops": ("team:security-secops", "Team", "Enterprise Security and SecOps Team"),
    "enterprise security and secops team": ("team:security-secops", "Team", "Enterprise Security and SecOps Team"),
    "support eng": ("team:customer-support-eng", "Team", "Customer Support Engineering Team"),
    "care tech": ("team:customer-support-eng", "Team", "Customer Support Engineering Team"),
    "customer support engineering team": ("team:customer-support-eng", "Team", "Customer Support Engineering Team"),
    "finance tech": ("team:billing-finance", "Team", "Billing and Finance Tech Team"),
    "billing eng": ("team:billing-finance", "Team", "Billing and Finance Tech Team"),
    "ledger team": ("team:billing-finance", "Team", "Billing and Finance Tech Team"),
    "billing and finance tech team": ("team:billing-finance", "Team", "Billing and Finance Tech Team"),
    "infra team": ("team:cloud-infrastructure", "Team", "Cloud Infrastructure and SRE Team"),
    "sre": ("team:cloud-infrastructure", "Team", "Cloud Infrastructure and SRE Team"),
    "devops": ("team:cloud-infrastructure", "Team", "Cloud Infrastructure and SRE Team"),
    "platform sre": ("team:cloud-infrastructure", "Team", "Cloud Infrastructure and SRE Team"),
    "cloud infrastructure and sre team": ("team:cloud-infrastructure", "Team", "Cloud Infrastructure and SRE Team"),
    # Databases
    "postgres": ("database:postgres-16", "Database", "PostgreSQL Database"),
    "postgresql": ("database:postgres-16", "Database", "PostgreSQL Database"),
    "postgresql 16": ("database:postgres-16", "Database", "PostgreSQL Database"),
    "customer orders pg": ("database:customer-orders-pg", "Database", "Customer Orders Database"),
    "payment vault pg": ("database:payment-vault-pg", "Database", "Payment Vault Database"),
    "user directory pg": ("database:user-directory-pg", "Database", "User Directory Database"),
    "inventory pg": ("database:inventory-pg", "Database", "Inventory Database"),
    "financial ledger pg": ("database:financial-ledger-pg", "Database", "Financial Ledger Database"),
    "notification pg": ("database:notification-pg", "Database", "Notification Database"),
    "auth session redis": ("database:auth-session-redis", "Database", "Auth Session Redis Cache"),
    "cart redis": ("database:cart-redis", "Database", "Cart Redis Cache"),
    "catalog mongodb": ("database:catalog-mongodb", "Database", "Catalog MongoDB Store"),
    "search elasticsearch": ("database:search-elasticsearch", "Database", "Search Elasticsearch Cluster"),
    "analytics clickhouse": ("database:analytics-clickhouse", "Database", "Analytics ClickHouse Warehouse"),
    # Infrastructure
    "kafka": ("infra:kafka-event-bus", "Infrastructure", "Kafka Event Bus"),
    "kafka event bus": ("infra:kafka-event-bus", "Infrastructure", "Kafka Event Bus"),
    "kafka event cluster": ("infra:kafka-event-bus", "Infrastructure", "Kafka Event Bus"),
    "k8s prod cluster us east": ("infra:k8s-prod-cluster-us-east", "Infrastructure", "K8s Prod Cluster US East"),
    "k8s prod cluster us west": ("infra:k8s-prod-cluster-us-west", "Infrastructure", "K8s Prod Cluster US West"),
    "pci isolated vpc": ("infra:pci-isolated-vpc", "Infrastructure", "PCI Isolated VPC"),
    # APIs
    "process charge v2": ("api:process-charge-v2", "API", "Process Charge API v2"),
    "v2 charge": ("api:process-charge-v2", "API", "Process Charge API v2"),
    "create order v2": ("api:create-order-v2", "API", "Create Order API v2"),
    "validate token v1": ("api:validate-token-v1", "API", "Validate Token API v1"),
    "reserve inventory v1": ("api:reserve-inventory-v1", "API", "Reserve Inventory API v1"),
}


class EntityResolver:
    """Resolves entity mentions, aliases, and IDs to canonical ontology-compliant identifiers."""

    def __init__(
        self,
        custom_aliases: dict[str, tuple[str, EntityType, str]] | None = None,
    ) -> None:
        """Initialize resolver with alias dictionary."""
        # Key: normalized_alias -> set of (canonical_id, entity_type, primary_name)
        self._alias_registry: dict[str, set[ResolvedEntityTarget]] = {}

        # Load default aliases
        for alias, (canon_id, ent_type, name) in DEFAULT_ALIAS_MAP.items():
            self.register_alias(alias, canon_id, ent_type, name)

        # Load custom aliases if provided
        if custom_aliases:
            for alias, (canon_id, ent_type, name) in custom_aliases.items():
                self.register_alias(alias, canon_id, ent_type, name)

    def register_alias(
        self, alias: str, canonical_id: str, entity_type: EntityType, primary_name: str
    ) -> None:
        """Register a new alias for a canonical entity."""
        norm_key = normalize_alias_key(alias)
        if not norm_key:
            return

        target = ResolvedEntityTarget(
            canonical_id=canonical_id,
            entity_type=entity_type,
            primary_name=primary_name,
        )
        if norm_key not in self._alias_registry:
            self._alias_registry[norm_key] = set()
        self._alias_registry[norm_key].add(target)

    def register_entity(self, entity: ExtractedEntity) -> None:
        """Register an extracted entity along with all its declared aliases."""
        # Register canonical_id itself
        self.register_alias(entity.canonical_id, entity.canonical_id, entity.entity_type, entity.name)
        # Register display name
        self.register_alias(entity.name, entity.canonical_id, entity.entity_type, entity.name)
        # Register all aliases
        for alias in entity.aliases:
            self.register_alias(alias, entity.canonical_id, entity.entity_type, entity.name)

    def resolve_mention(
        self,
        mention: str,
        hint_type: EntityType | None = None,
        allow_fallback: bool = False,
    ) -> ResolvedEntityTarget:
        """Resolve a raw mention or alias to its canonical entity target.

        Parameters
        ----------
        mention:
            The raw entity mention, alias, or candidate ID.
        hint_type:
            Optional expected EntityType to disambiguate or guide fallback inference.
        allow_fallback:
            If True, enables slugification fallback for unknown mentions. Defaults to False (strict).

        Returns
        -------
        ResolvedEntityTarget
            The resolved canonical ID, entity type, and primary name.

        Raises
        ------
        AmbiguousEntityError
            If multiple distinct canonical entities match the mention.
        UnknownEntityError
            If the mention is empty, unresolvable, or not recognized and allow_fallback is False.
        """
        raw = mention.strip()
        if not raw:
            raise UnknownEntityError("Cannot resolve empty or whitespace mention")

        # 1. Check alias registry first using raw mention or stripped prefix if applicable
        norm_key = normalize_alias_key(raw)
        candidates = self._alias_registry.get(norm_key, set())

        # If mention is in 'prefix:slug' form, also check if the slug is an alias
        if ":" in raw:
            parts = raw.split(":", 1)
            prefix_part = parts[0].strip().lower()
            slug_part = parts[1].strip()
            if prefix_part in _PREFIX_TO_ENTITY_TYPE:
                if not hint_type:
                    hint_type = _PREFIX_TO_ENTITY_TYPE[prefix_part]
                slug_norm = normalize_alias_key(slug_part)
                slug_cands = self._alias_registry.get(slug_norm, set())
                if slug_cands:
                    candidates = candidates | slug_cands

        if hint_type:
            filtered = {c for c in candidates if c.entity_type == hint_type}
            if filtered:
                candidates = filtered

        if len(candidates) == 1:
            return next(iter(candidates))
        elif len(candidates) > 1:
            ids = sorted({c.canonical_id for c in candidates})
            raise AmbiguousEntityError(
                f"Mention '{mention}' is ambiguous and matches multiple canonical entities: {ids}"
            )

        # 2. Check for CVE Vulnerability pattern
        cve_match = _CVE_RE.search(raw)
        if cve_match or (hint_type == "Vulnerability" and "cve" in raw.lower()):
            if cve_match:
                cve_digits = cve_match.group(1).replace("_", "-")
            else:
                cve_digits = to_kebab_case(raw).replace("cve-", "").replace("cve", "")
            canon_id = f"vuln:cve-{cve_digits.lower()}"
            # Ensure valid format
            if _CANONICAL_ID_RE.match(canon_id):
                return ResolvedEntityTarget(
                    canonical_id=canon_id,
                    entity_type="Vulnerability",
                    primary_name=f"CVE-{cve_digits.upper()}",
                )

        # 3. Suffix/Prefix stripping heuristics if not matched
        # A. Team suffix stripping
        for suffix_pat in [
            re.compile(r"\s+\b(team|engineering|tech|sre|secops|devops)\b$", re.IGNORECASE),
            re.compile(r"\s+\b(engineering\s+team|eng\s+team)\b$", re.IGNORECASE),
            _TEAM_SUFFIX_RE,
        ]:
            team_stripped = suffix_pat.sub("", raw).strip()
            if team_stripped and team_stripped != raw:
                team_norm = normalize_alias_key(team_stripped)
                team_cands = self._alias_registry.get(team_norm, set())
                team_cands = {c for c in team_cands if c.entity_type == "Team"}
                if len(team_cands) == 1:
                    return next(iter(team_cands))
                elif len(team_cands) > 1:
                    ids = sorted({c.canonical_id for c in team_cands})
                    raise AmbiguousEntityError(
                        f"Team mention '{mention}' is ambiguous after suffix stripping: {ids}"
                    )

        # B. Service suffix stripping
        for suffix_pat in [
            re.compile(r"\s+\b(service|svc|core|engine|dispatcher|gateway|relay|adapter|indexer|aggregator)\b$", re.IGNORECASE),
            _SERVICE_SUFFIX_RE,
        ]:
            svc_stripped = suffix_pat.sub("", raw).strip()
            if svc_stripped and svc_stripped != raw:
                svc_norm = normalize_alias_key(svc_stripped)
                svc_cands = self._alias_registry.get(svc_norm, set())
                svc_cands = {c for c in svc_cands if c.entity_type == "Service"}
                if len(svc_cands) == 1:
                    return next(iter(svc_cands))
                elif len(svc_cands) > 1:
                    ids = sorted({c.canonical_id for c in svc_cands})
                    raise AmbiguousEntityError(
                        f"Service mention '{mention}' is ambiguous after suffix stripping: {ids}"
                    )

        # 4. If it's already a valid canonical ID format (e.g. 'service:payment-gateway')
        if _CANONICAL_ID_RE.match(raw.lower()):
            clean_canon_id = raw.lower()
            prefix = clean_canon_id.split(":")[0]
            inferred_type = _PREFIX_TO_ENTITY_TYPE[prefix]
            return ResolvedEntityTarget(
                canonical_id=clean_canon_id,
                entity_type=inferred_type,
                primary_name=raw,
            )

        # 5. Deterministic fallback generation only if explicitly opted-in (allow_fallback=True)
        if allow_fallback and hint_type:
            prefix = _CANONICAL_ID_PREFIX[hint_type]
            slug = to_kebab_case(raw)
            if slug:
                canon_id = f"{prefix}:{slug}"
                if _CANONICAL_ID_RE.match(canon_id):
                    return ResolvedEntityTarget(
                        canonical_id=canon_id,
                        entity_type=hint_type,
                        primary_name=raw,
                    )

        raise UnknownEntityError(
            f"Could not resolve entity mention '{mention}' to a canonical identifier."
        )

    def resolve_entity(self, entity: ExtractedEntity) -> ExtractedEntity:
        """Resolve and normalize a single ExtractedEntity, returning a new instance without mutating in place."""
        # 1. Try aliases first (explicit known alias mentions)
        target: ResolvedEntityTarget | None = None
        for alias in list(entity.aliases):
            norm = normalize_alias_key(alias)
            if norm in self._alias_registry:
                try:
                    alt_target = self.resolve_mention(alias, hint_type=entity.entity_type)
                    target = alt_target
                    break
                except (UnknownEntityError, AmbiguousEntityError):
                    continue

        # 2. Try entity display name if it's registered in alias registry
        if target is None:
            norm_name = normalize_alias_key(entity.name)
            if norm_name in self._alias_registry:
                try:
                    target = self.resolve_mention(entity.name, hint_type=entity.entity_type)
                except (UnknownEntityError, AmbiguousEntityError):
                    target = None

        # 3. Fallback to resolving canonical_id
        if target is None:
            target = self.resolve_mention(entity.canonical_id, hint_type=entity.entity_type)

        # Normalize any aliases within the entity as well
        normalized_aliases = sorted(
            {a.strip() for a in entity.aliases if a.strip() and a.strip().lower() != target.canonical_id}
        )

        return ExtractedEntity(
            canonical_id=target.canonical_id,
            name=entity.name,
            entity_type=target.entity_type,
            description=entity.description,
            aliases=normalized_aliases,
            confidence=entity.confidence,
            provenance=entity.provenance,
        )

    def resolve_relationship(self, rel: ExtractedRelationship) -> ExtractedRelationship:
        """Resolve endpoints of an ExtractedRelationship, returning a new instance."""
        source_target = self.resolve_mention(rel.source_canonical_id, hint_type=rel.source_type)
        target_target = self.resolve_mention(rel.target_canonical_id, hint_type=rel.target_type)

        return ExtractedRelationship(
            source_canonical_id=source_target.canonical_id,
            source_type=source_target.entity_type,
            target_canonical_id=target_target.canonical_id,
            target_type=target_target.entity_type,
            relation_type=rel.relation_type,
            description=rel.description,
            confidence=rel.confidence,
            source_chunk_ids=list(rel.source_chunk_ids),
            provenance=rel.provenance,
        )

    def resolve_extraction_result(
        self,
        result: ChunkExtractionResult,
        deduplicate_entities: bool = True,
    ) -> ChunkExtractionResult:
        """Resolve all entities and relationships in a ChunkExtractionResult immutably.

        Parameters
        ----------
        result:
            Input ``ChunkExtractionResult`` from TASK-104 extraction.
        deduplicate_entities:
            If True, merges multiple entity records in the same chunk that resolve
            to the same canonical_id (taking max confidence, merged aliases, etc.).

        Returns
        -------
        ChunkExtractionResult
            New, validated extraction result with normalized canonical IDs and intact provenance.
        """
        # Resolve entities
        resolved_entities_map: dict[str, ExtractedEntity] = {}
        for raw_ent in result.entities:
            resolved_ent = self.resolve_entity(raw_ent)
            cid = resolved_ent.canonical_id

            if deduplicate_entities and cid in resolved_entities_map:
                existing = resolved_entities_map[cid]
                # Merge aliases and retain highest confidence
                merged_aliases = sorted(set(existing.aliases) | set(resolved_ent.aliases))
                max_conf = max(existing.confidence, resolved_ent.confidence)
                desc = existing.description or resolved_ent.description

                resolved_entities_map[cid] = ExtractedEntity(
                    canonical_id=cid,
                    name=existing.name,
                    entity_type=existing.entity_type,
                    description=desc,
                    aliases=merged_aliases,
                    confidence=max_conf,
                    provenance=existing.provenance,
                )
            else:
                resolved_entities_map[cid] = resolved_ent

        resolved_entities = list(resolved_entities_map.values())

        # Resolve relationships
        resolved_relationships: list[ExtractedRelationship] = []
        seen_rels: set[tuple[str, str, str]] = set()

        for raw_rel in result.relationships:
            resolved_rel = self.resolve_relationship(raw_rel)
            rel_key = (
                resolved_rel.source_canonical_id,
                resolved_rel.target_canonical_id,
                resolved_rel.relation_type,
            )
            # Skip duplicate identical relationships within same chunk
            if rel_key in seen_rels:
                continue
            # Reject self-loops if source resolved to same ID as target
            if resolved_rel.source_canonical_id == resolved_rel.target_canonical_id:
                logger.warning(
                    "Skipping self-referential relationship after resolution: %s",
                    resolved_rel.source_canonical_id,
                )
                continue
            seen_rels.add(rel_key)
            resolved_relationships.append(resolved_rel)

        return ChunkExtractionResult(
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            extraction_model=result.extraction_model,
            entities=resolved_entities,
            relationships=resolved_relationships,
        )
