"""Query intent classification, entity linking, and routing modules (TASK-303, TASK-304, TASK-305).

Provides:
- Deterministic and LLM-assisted query intent classification into GRAPH_ONLY, VECTOR_ONLY, HYBRID, OUT_OF_SCOPE.
- Entity linking mapping mentions and question terms to canonical Neo4j entity IDs.
- Router Engine coordinating intent classification, entity linking, and retrieval delegation.
"""

from __future__ import annotations

from src.routing.entity_linker import (
    EntityLink,
    EntityLinker,
    EntityLinkingResult,
    LinkResolutionStatus,
)
from src.routing.intent_classifier import (
    ClassificationMethod,
    IntentClassificationResult,
    QueryIntent,
    QueryIntentClassifier,
    classify_query_heuristic,
)
from src.routing.router import (
    RouterEngine,
    RoutingRequest,
    RoutingResult,
)

__all__ = [
    "ClassificationMethod",
    "EntityLink",
    "EntityLinker",
    "EntityLinkingResult",
    "IntentClassificationResult",
    "LinkResolutionStatus",
    "QueryIntent",
    "QueryIntentClassifier",
    "RouterEngine",
    "RoutingRequest",
    "RoutingResult",
    "classify_query_heuristic",
]
