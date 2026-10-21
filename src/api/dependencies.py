"""FastAPI Dependency Injection and Pipeline Service Assembly (TASK-404).

Provides injectable services for health checks, query execution, routing,
normalization, synthesis, and citation validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import anthropic

from src.config import Settings, get_settings
from src.db.neo4j import Neo4jClient
from src.db.postgres import PostgresClient
from src.embeddings.client import EmbeddingGenerator
from src.retrieval.graph_retriever import SafeGraphRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.routing.entity_linker import EntityLinker
from src.routing.intent_classifier import QueryIntentClassifier
from src.routing.router import RouterEngine
from src.synthesis.citation_validator import CitationValidator
from src.synthesis.normalizer import ContextNormalizer
from src.synthesis.synthesizer import GroundedAnswerSynthesizer

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """Return current UTC time formatted as an ISO-8601 string."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ServiceContainer:
    """Holds shared lifecycle dependencies and pipeline components."""

    settings: Settings
    neo4j_client: Neo4jClient | None = None
    postgres_client: PostgresClient | None = None
    embedding_generator: EmbeddingGenerator | None = None
    intent_classifier: QueryIntentClassifier | None = None
    entity_linker: EntityLinker | None = None
    router: RouterEngine | None = None
    normalizer: ContextNormalizer | None = None
    synthesizer: GroundedAnswerSynthesizer | None = None
    validator: CitationValidator | None = None
    anthropic_client: anthropic.Anthropic | None = None

    def get_router(self) -> RouterEngine:
        """Lazily initialize and return the RouterEngine."""
        if self.router is None:
            classifier = self.intent_classifier or QueryIntentClassifier()
            entity_linker = self.entity_linker or EntityLinker(
                neo4j_client=self.neo4j_client,
                verify_graph=False,
            )
            graph_retriever = (
                SafeGraphRetriever(client=self.neo4j_client) if self.neo4j_client else None
            )
            vector_retriever = (
                VectorRetriever(client=self.postgres_client, settings=self.settings)
                if self.postgres_client
                else None
            )
            self.router = RouterEngine(
                classifier=classifier,
                entity_linker=entity_linker,
                graph_retriever=graph_retriever,
                vector_retriever=vector_retriever,
                embedding_generator=self.embedding_generator,
                settings=self.settings,
            )
        return self.router

    def get_normalizer(self) -> ContextNormalizer:
        """Lazily initialize and return ContextNormalizer."""
        if self.normalizer is None:
            self.normalizer = ContextNormalizer(settings=self.settings)
        return self.normalizer

    def get_synthesizer(self) -> GroundedAnswerSynthesizer:
        """Lazily initialize and return GroundedAnswerSynthesizer."""
        if self.synthesizer is None:
            self.synthesizer = GroundedAnswerSynthesizer(
                client=self.anthropic_client,
                settings=self.settings,
            )
        return self.synthesizer

    def get_validator(self) -> CitationValidator:
        """Lazily initialize and return CitationValidator."""
        if self.validator is None:
            self.validator = CitationValidator()
        return self.validator

    async def check_databases_readiness(self) -> tuple[bool, dict[str, str], str | None]:
        """Verify Neo4j and PostgreSQL database connectivity.

        Returns:
            Tuple of (is_ready, database_status_dict, error_message).
        """
        neo4j_status = "disconnected"
        postgres_status = "disconnected"
        errors: list[str] = []

        # Check Neo4j
        try:
            if self.neo4j_client is None:
                self.neo4j_client = Neo4jClient(settings=self.settings)
            self.neo4j_client.verify_connectivity()
            neo4j_status = "connected"
        except Exception as exc:
            logger.warning("Neo4j readiness probe failed: %s", exc)
            errors.append(f"Failed to connect to Neo4j database: {exc}")

        # Check PostgreSQL
        try:
            if self.postgres_client is None:
                self.postgres_client = PostgresClient(settings=self.settings)
            await self.postgres_client.verify_connectivity()
            postgres_status = "connected"
        except Exception as exc:
            logger.warning("PostgreSQL readiness probe failed: %s", exc)
            errors.append(f"Failed to connect to PostgreSQL database: {exc}")

        is_ready = (neo4j_status == "connected") and (postgres_status == "connected")
        error_msg = "; ".join(errors) if errors else None
        return is_ready, {"neo4j": neo4j_status, "postgres": postgres_status}, error_msg

    async def close(self) -> None:
        """Clean up managed database connections and pools."""
        if self.postgres_client is not None:
            try:
                await self.postgres_client.close()
            except Exception as exc:
                logger.warning("Error closing PostgreSQL client: %s", exc)
        if self.neo4j_client is not None:
            try:
                self.neo4j_client.close()
            except Exception as exc:
                logger.warning("Error closing Neo4j client: %s", exc)


_GLOBAL_CONTAINER: ServiceContainer | None = None


def create_service_container(
    settings: Settings | None = None,
    *,
    init_clients: bool = True,
) -> ServiceContainer:
    """Factory creating a fully initialized ServiceContainer.

    When init_clients is True, initializes managed PostgresClient, Neo4jClient,
    and EmbeddingGenerator instances so retrieval components are fully active.
    """
    cfg = settings or get_settings()
    postgres_client = PostgresClient(settings=cfg) if init_clients else None
    neo4j_client = Neo4jClient(settings=cfg) if init_clients else None
    embedding_generator = EmbeddingGenerator(settings=cfg) if init_clients else None

    return ServiceContainer(
        settings=cfg,
        postgres_client=postgres_client,
        neo4j_client=neo4j_client,
        embedding_generator=embedding_generator,
    )


def get_service_container() -> ServiceContainer:
    """Dependency provider for FastAPI route handlers."""
    global _GLOBAL_CONTAINER
    if _GLOBAL_CONTAINER is None:
        _GLOBAL_CONTAINER = ServiceContainer(settings=get_settings())
    return _GLOBAL_CONTAINER


def set_service_container(container: ServiceContainer | None) -> None:
    """Override the global service container for unit tests."""
    global _GLOBAL_CONTAINER
    _GLOBAL_CONTAINER = container

