r"""Query Intent Classifier for Knowledge Graph RAG (TASK-303).

Design Constraints & Routing Taxonomy (see docs/RETRIEVAL_STRATEGY.md, docs/SECURITY.md, docs/API_SPEC.md):
---------------------------------------------------------------------------------------------------------
* **Four Execution Modes**:
  1. `GRAPH_ONLY`: Queries requiring multi-hop graph traversal, direct entity neighborhood,
     structural dependency paths, ownership lookups, vulnerability blast radiuses, or relationship aggregations.
  2. `VECTOR_ONLY`: Direct semantic lookups, definitions, policy explanations, or factual details
     contained within isolated text chunks.
  3. `HYBRID`: Broad architectural questions, multi-faceted inquiries touching both relationships
     and textual descriptions, or any classification whose confidence falls below `router_confidence_threshold` (default: 0.65).
  4. `OUT_OF_SCOPE`: Non-enterprise, general knowledge (weather, sports, general chit-chat), or malicious/jailbreak prompts.
* **Deterministic Fallback & Fast-Path**:
  - Implements a fast, deterministic regex- and keyword-based heuristic classifier (`classify_query_heuristic`)
    that accurately flags out-of-scope inquiries and clear graph/vector patterns offline without LLM invocations.
* **LLM-Assisted Intent Classification**:
  - Integrates with Anthropic Claude (using `routing_model`, e.g. `claude-3-haiku-20240307`) using structured JSON
    prompts with explicit XML delimiters (<user_query>...</user_query>) to prevent prompt injection.
  - Automatically falls back to `HYBRID` mode or deterministic heuristic if the LLM response is malformed,
    or if confidence is lower than the configured threshold.
* **Predictable, Typed Result Schema**:
  - Emits `IntentClassificationResult` with `mode`, `confidence`, `reasoning`, `suggested_template_id`, `extracted_entities`,
    and `method`.
* **Zero Database & Paid API Requirements in Tests**:
  - Supports dependency injection of client, offline heuristic execution, and mock responses.
"""

from __future__ import annotations

import json
import logging
import re
from enum import StrEnum
from typing import Final

import anthropic
from pydantic import BaseModel, ConfigDict, Field

from src.config import Settings
from src.retrieval.cypher_templates import (
    TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
    TEMPLATE_DIRECT_NEIGHBORHOOD,
    TEMPLATE_ENTITY_LOOKUP_BY_ID,
    TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS,
    TEMPLATE_MULTI_HOP_DEPENDENCY,
    TEMPLATE_SERVICE_DEPENDENCY_CHAIN,
    TEMPLATE_TEAM_OWNERSHIP_SERVICES,
    TEMPLATE_VULNERABILITY_BLAST_RADIUS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Result Schemas
# ---------------------------------------------------------------------------


class QueryIntent(StrEnum):
    """The four authoritative routing execution modes (docs/RETRIEVAL_STRATEGY.md §1)."""

    GRAPH_ONLY = "GRAPH_ONLY"
    VECTOR_ONLY = "VECTOR_ONLY"
    HYBRID = "HYBRID"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class ClassificationMethod(StrEnum):
    """Method used to classify the query."""

    HEURISTIC = "heuristic"
    LLM = "llm"
    OVERRIDE = "override"
    FALLBACK = "fallback"


class IntentClassificationResult(BaseModel):
    """Predictable, typed classification result returned by the Intent Classifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(..., description="The original raw or trimmed query string.")
    mode: QueryIntent = Field(
        ...,
        description="The authoritative execution mode (GRAPH_ONLY, VECTOR_ONLY, HYBRID, OUT_OF_SCOPE).",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score of the classification [0.0, 1.0]."
    )
    reasoning: str = Field(..., description="Brief rationale explaining the routing decision.")
    suggested_template_id: str | None = Field(
        default=None,
        description="Suggested Cypher template ID (e.g., T-01..T-08) if mode involves graph retrieval.",
    )
    extracted_entities: list[str] = Field(
        default_factory=list,
        description="Key entities, aliases, or identifiers extracted from the query.",
    )
    method: ClassificationMethod = Field(
        default=ClassificationMethod.HEURISTIC,
        description="The mechanism that produced this classification.",
    )


# ---------------------------------------------------------------------------
# System Prompt for Structured LLM Classification
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM_PROMPT: Final[
    str
] = """You are an expert Query Intent Classifier for an Enterprise Architecture & Software Systems Knowledge Graph RAG platform.

Your task is to analyze the user's query and classify it into exactly ONE of four execution modes:

1. `GRAPH_ONLY`:
   - Queries focused on structural relationships, multi-hop dependencies, call graphs, team ownership, service blast radius, database producers/consumers, or entity neighborhood lookups.
   - Examples:
     * "Which services depend on a service owned by the Checkout team?"
     * "List all databases written to by services vulnerable to CVE-2024-21626."
     * "What services are owned by Team Core?"
     * "Show me all downstream services that call the Payment Gateway."

2. `VECTOR_ONLY`:
   - Queries focused on direct semantic lookup, text definitions, retry policies, configuration parameters, SLA descriptions, or conceptual explanations contained in document prose.
   - Examples:
     * "What is the retry policy described in the payment gateway spec?"
     * "Explain the difference between optimistic and pessimistic locking in our order service."
     * "What does the SLA section say about database failover time?"

3. `HYBRID`:
   - Broad architecture questions, multi-faceted inquiries touching both relationship topologies and prose explanations, or questions where neither graph nor vector alone provides a complete answer.
   - Also used when the classification confidence is ambiguous (< 0.65).
   - Examples:
     * "Describe the payment architecture and which teams are responsible for it."
     * "Give an overview of the checkout system, its dependencies, and its deployment guide."
     * "How does order fulfillment work and which services and databases are involved?"

4. `OUT_OF_SCOPE`:
   - General trivia, weather, sports, personal questions, recipes, casual chit-chat, or queries completely unrelated to enterprise software architecture, services, teams, databases, APIs, repositories, infrastructure, or security vulnerabilities.
   - Examples:
     * "What is the weather in Tokyo?"
     * "Write a poem about dogs."
     * "Who won the 2022 World Cup?"

## Approved Cypher Templates for Graph Routing
If the query matches or benefits from a graph traversal, select the best matching template ID:
- `T-01`: Direct Entity Neighborhood (1-hop connected neighbors, relationships, descriptions)
- `T-02`: Multi-Hop Dependency Path (2 to 3 hop dependency or API call chains)
- `T-03`: Team Ownership & Services (services owned by a team and their vulnerabilities)
- `T-04`: Vulnerability Blast Radius (services affected by a CVE and their upstream dependents)
- `T-05`: Service Dependency Chain (upstream callers and downstream dependencies of a service)
- `T-06`: Database Producers & Consumers (services reading from or writing to a database)
- `T-07`: Infrastructure Deployments (components deployed on or producing events to infrastructure)
- `T-08`: Entity Lookup by ID or Name (exact canonical entity lookup)

## Output Format
Respond ONLY with a valid JSON object matching this schema. Do not include markdown code blocks, backticks, or any additional text:
{
  "mode": "GRAPH_ONLY" | "VECTOR_ONLY" | "HYBRID" | "OUT_OF_SCOPE",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<short concise explanation>",
  "suggested_template_id": "<T-01..T-08 or null>",
  "extracted_entities": ["<entity_name_or_identifier_1>", ...]
}
"""


# ---------------------------------------------------------------------------
# Deterministic Heuristic Classifier (Offline / Fast Path)
# ---------------------------------------------------------------------------

# Patterns indicating obvious out-of-scope non-enterprise queries
_OUT_OF_SCOPE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"\b(weather|forecast|temperature|rain|snow)\b", re.IGNORECASE),
    re.compile(r"\b(recipe|cook|bake|ingredients|dinner|lunch|breakfast)\b", re.IGNORECASE),
    re.compile(r"\b(poem|rhyme|song|lyrics|joke|riddle|story|fiction)\b", re.IGNORECASE),
    re.compile(r"\b(world cup|super bowl|olympics|nba|nfl|fifa|score|sports)\b", re.IGNORECASE),
    re.compile(
        r"\b(president|prime minister|election|politician|celebrity|actor|movie)\b", re.IGNORECASE
    ),
    re.compile(r"\b(capital of|population of|distance to the moon)\b", re.IGNORECASE),
    re.compile(r"^\s*(hi|hello|hey|howdy|greetings)\b", re.IGNORECASE),
    re.compile(r"\b(how are you|good morning|good evening|good afternoon)\b", re.IGNORECASE),
    re.compile(r"\b(who are you|what is your name|tell me a joke)\b", re.IGNORECASE),
]

# Patterns indicating graph-centric structural queries
_GRAPH_PATTERNS: Final[list[tuple[re.Pattern[str], str | None]]] = [
    # Multi-hop & Dependency Paths (T-02 / T-05)
    (
        re.compile(
            r"\b(depend(s|ed|ing)? on|dependency|dependencies|downstream|upstream|call chain|call path|calls api)\b",
            re.IGNORECASE,
        ),
        TEMPLATE_MULTI_HOP_DEPENDENCY,
    ),
    # Vulnerability blast radius (T-04)
    (
        re.compile(
            r"\b(cve-\d{4}-\d+|vulnerability|vulnerabilities|blast radius|affected by vuln|impacted services)\b",
            re.IGNORECASE,
        ),
        TEMPLATE_VULNERABILITY_BLAST_RADIUS,
    ),
    # Team ownership (T-03)
    (
        re.compile(
            r"\b(owned by|team ownership|maintain(ed|s)? by|team owns|which team)\b", re.IGNORECASE
        ),
        TEMPLATE_TEAM_OWNERSHIP_SERVICES,
    ),
    # Database reading/writing (T-06)
    (
        re.compile(
            r"\b(read(s|ing)? from|write(s|ing)? to|database producers|database consumers|tables?|datastore)\b",
            re.IGNORECASE,
        ),
        TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
    ),
    # Infrastructure deployments (T-07)
    (
        re.compile(
            r"\b(deployed on|infrastructure|cluster|kubernetes|k8s|kafka topic|event bus|broker)\b",
            re.IGNORECASE,
        ),
        TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS,
    ),
    # 1-hop neighborhood / relationships (T-01)
    (
        re.compile(
            r"\b(connected to|relations(hip)?s of|neighbors? of|linked to)\b", re.IGNORECASE
        ),
        TEMPLATE_DIRECT_NEIGHBORHOOD,
    ),
]

# Patterns indicating vector-centric semantic prose queries
_VECTOR_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"\b(retry policy|retry strategy|timeout configuration|sla|specification)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(how to configure|setup guide|documentation for|step[- ]by[- ]step|overview of the concept)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(explain the difference between|definition of|meaning of|what does .* mean)\b",
        re.IGNORECASE,
    ),
]

# Patterns indicating broad hybrid questions
_HYBRID_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"\b(describe the .* architecture and|overview of .* and which teams|architecture and dependencies)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(comprehensive overview|full summary of|how does .* work end to end)\b", re.IGNORECASE
    ),
]


def classify_query_heuristic(
    query: str,
    confidence_threshold: float = 0.65,
) -> IntentClassificationResult:
    """Classify a query using deterministic heuristic rules.

    This function operates offline without external API calls or network access.
    It serves as a fast-path for clear queries, a safety filter for malformed/empty
    or obviously out-of-scope inputs, and a reliable fallback when LLM routing is unavailable.

    Parameters
    ----------
    query:
        The raw question string to classify.
    confidence_threshold:
        Confidence score threshold (from settings) below which queries fall back to HYBRID.

    Returns
    -------
    IntentClassificationResult:
        Typed classification output.
    """
    clean_query = query.strip()

    # 1. Empty, whitespace-only, or very short garbage queries
    if not clean_query or len(clean_query) < 3:
        return IntentClassificationResult(
            query=clean_query,
            mode=QueryIntent.OUT_OF_SCOPE,
            confidence=1.0,
            reasoning="Empty or excessively short query with no enterprise semantic content.",
            suggested_template_id=None,
            extracted_entities=[],
            method=ClassificationMethod.HEURISTIC,
        )

    # 2. Check for explicit Out of Scope patterns
    for pattern in _OUT_OF_SCOPE_PATTERNS:
        if pattern.search(clean_query):
            return IntentClassificationResult(
                query=clean_query,
                mode=QueryIntent.OUT_OF_SCOPE,
                confidence=0.95,
                reasoning=f"Query matched out-of-scope non-enterprise domain pattern: '{pattern.pattern}'.",
                suggested_template_id=None,
                extracted_entities=[],
                method=ClassificationMethod.HEURISTIC,
            )

    # 3. Extract candidate entities / identifiers from query
    extracted_entities: list[str] = []
    # Match canonical ID patterns like service:payment-gateway, team:checkout, cve-2024-21626
    cve_matches = re.findall(r"\b(?:cve-\d{4}-\d+|vuln:[a-z0-9-]+)\b", clean_query, re.IGNORECASE)
    extracted_entities.extend([m.lower() for m in cve_matches])

    canonical_matches = re.findall(
        r"\b(?:service|team|database|api|repo|infra):[a-z0-9-]+\b", clean_query, re.IGNORECASE
    )
    extracted_entities.extend([m.lower() for m in canonical_matches])

    # 4. Check for Hybrid patterns
    for pattern in _HYBRID_PATTERNS:
        if pattern.search(clean_query):
            return IntentClassificationResult(
                query=clean_query,
                mode=QueryIntent.HYBRID,
                confidence=0.85,
                reasoning="Query asks for a broad multi-faceted architectural overview combining prose and relationships.",
                suggested_template_id=TEMPLATE_DIRECT_NEIGHBORHOOD,
                extracted_entities=extracted_entities,
                method=ClassificationMethod.HEURISTIC,
            )

    # 5. Check for Graph patterns
    matched_template: str | None = None
    graph_match_count = 0
    for pattern, template_id in _GRAPH_PATTERNS:
        if pattern.search(clean_query):
            graph_match_count += 1
            if matched_template is None:
                matched_template = template_id

    # 6. Check for Vector patterns
    vector_match_count = 0
    for pattern in _VECTOR_PATTERNS:
        if pattern.search(clean_query):
            vector_match_count += 1

    # 7. Evaluate match combinations
    if graph_match_count > 0 and vector_match_count > 0:
        return IntentClassificationResult(
            query=clean_query,
            mode=QueryIntent.HYBRID,
            confidence=0.80,
            reasoning="Query contains both structural graph traversal cues and semantic text lookup cues.",
            suggested_template_id=matched_template or TEMPLATE_DIRECT_NEIGHBORHOOD,
            extracted_entities=extracted_entities,
            method=ClassificationMethod.HEURISTIC,
        )

    if graph_match_count > 0:
        return IntentClassificationResult(
            query=clean_query,
            mode=QueryIntent.GRAPH_ONLY,
            confidence=0.90,
            reasoning="Query asks for relationship topology, dependency path, ownership, or vulnerability blast radius.",
            suggested_template_id=matched_template or TEMPLATE_DIRECT_NEIGHBORHOOD,
            extracted_entities=extracted_entities,
            method=ClassificationMethod.HEURISTIC,
        )

    if vector_match_count > 0:
        return IntentClassificationResult(
            query=clean_query,
            mode=QueryIntent.VECTOR_ONLY,
            confidence=0.88,
            reasoning="Query asks for semantic text explanations, policies, specifications, or definitions.",
            suggested_template_id=None,
            extracted_entities=extracted_entities,
            method=ClassificationMethod.HEURISTIC,
        )

    # 8. Ambiguous enterprise query fallback
    # If the query seems enterprise-related but lacks specific structural or prose triggers
    confidence = 0.50
    if confidence < confidence_threshold:
        return IntentClassificationResult(
            query=clean_query,
            mode=QueryIntent.HYBRID,
            confidence=confidence,
            reasoning=f"Ambiguous query intent with confidence ({confidence:.2f}) below threshold ({confidence_threshold:.2f}); falling back to HYBRID dual-retrieval.",
            suggested_template_id=TEMPLATE_DIRECT_NEIGHBORHOOD if extracted_entities else None,
            extracted_entities=extracted_entities,
            method=ClassificationMethod.FALLBACK,
        )

    return IntentClassificationResult(
        query=clean_query,
        mode=QueryIntent.HYBRID,
        confidence=confidence,
        reasoning="Default fallback to hybrid retrieval mode.",
        suggested_template_id=None,
        extracted_entities=extracted_entities,
        method=ClassificationMethod.HEURISTIC,
    )


# ---------------------------------------------------------------------------
# Query Intent Classifier Class
# ---------------------------------------------------------------------------


class QueryIntentClassifier:
    """Classifies user queries into GRAPH_ONLY, VECTOR_ONLY, HYBRID, or OUT_OF_SCOPE.

    Features:
    - Safe fast-path heuristic evaluation for clear offline routing.
    - LLM-assisted classification using structured Anthropic Claude prompts with security delimiters.
    - Confidence threshold evaluation: routes ambiguous queries (< threshold) to HYBRID mode.
    - Graceful degradation: falls back to heuristic/HYBRID on LLM errors or malformed outputs.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        anthropic_client: anthropic.Anthropic | None = None,
        use_llm: bool = True,
    ) -> None:
        """Initialize the QueryIntentClassifier.

        Parameters
        ----------
        settings:
            Application configuration settings. If None, default settings are loaded.
        anthropic_client:
            Optional Anthropic client interface for testing/dependency injection.
        use_llm:
            Whether to attempt LLM-based classification before heuristic fallback.
        """
        self._settings = settings or Settings()
        self._client = anthropic_client
        self._use_llm = use_llm
        self._confidence_threshold = self._settings.router_confidence_threshold

    @property
    def confidence_threshold(self) -> float:
        """Configured confidence threshold for routing decisions."""
        return self._confidence_threshold

    def classify(
        self,
        query: str,
        retrieval_override: QueryIntent | str | None = None,
    ) -> IntentClassificationResult:
        """Classify an incoming query into an authoritative execution mode.

        Parameters
        ----------
        query:
            The raw question or prompt string.
        retrieval_override:
            Optional explicit override mode ('GRAPH_ONLY', 'VECTOR_ONLY', 'HYBRID', 'OUT_OF_SCOPE').
            If provided, bypasses classification.

        Returns
        -------
        IntentClassificationResult:
            Predictable, typed classification result.
        """
        clean_query = query.strip() if query else ""

        # 1. Check for manual caller override
        if retrieval_override is not None:
            override_mode = (
                retrieval_override
                if isinstance(retrieval_override, QueryIntent)
                else QueryIntent(retrieval_override)
            )
            return IntentClassificationResult(
                query=clean_query,
                mode=override_mode,
                confidence=1.0,
                reasoning=f"Explicit caller override requested: '{override_mode.value}'.",
                suggested_template_id=TEMPLATE_DIRECT_NEIGHBORHOOD
                if override_mode in (QueryIntent.GRAPH_ONLY, QueryIntent.HYBRID)
                else None,
                extracted_entities=[],
                method=ClassificationMethod.OVERRIDE,
            )

        # 2. Fast check: Empty or trivially invalid queries
        if not clean_query or len(clean_query) < 3:
            return classify_query_heuristic(clean_query, self._confidence_threshold)

        # 3. Fast check: Clear out-of-scope queries (saves LLM tokens & latency)
        heuristic_res = classify_query_heuristic(clean_query, self._confidence_threshold)
        if heuristic_res.mode == QueryIntent.OUT_OF_SCOPE and heuristic_res.confidence >= 0.90:
            return heuristic_res

        # 4. If LLM is disabled or no client / API key available, return heuristic result
        if not self._use_llm or (self._client is None and self._settings.anthropic_api_key is None):
            return heuristic_res

        # 5. Attempt LLM-assisted classification with prompt injection defense
        try:
            return self._classify_with_llm(clean_query)
        except Exception as exc:
            logger.warning(
                "LLM intent classification failed (%s); falling back to heuristic routing.",
                exc,
            )
            # Re-run heuristic with fallback marker
            fallback_res = classify_query_heuristic(clean_query, self._confidence_threshold)
            return IntentClassificationResult(
                query=fallback_res.query,
                mode=fallback_res.mode,
                confidence=fallback_res.confidence,
                reasoning=f"Heuristic fallback after LLM classification exception: {fallback_res.reasoning}",
                suggested_template_id=fallback_res.suggested_template_id,
                extracted_entities=fallback_res.extracted_entities,
                method=ClassificationMethod.FALLBACK,
            )

    def _get_client(self) -> anthropic.Anthropic:
        """Get or initialize the Anthropic client."""
        if self._client is not None:
            return self._client

        api_key = self._settings.anthropic_api_key
        if api_key is None:
            raise ValueError("Anthropic API key is not configured in settings.")

        self._client = anthropic.Anthropic(
            api_key=api_key.get_secret_value(),
            timeout=self._settings.request_timeout_seconds,
        )
        return self._client

    def _classify_with_llm(self, query: str) -> IntentClassificationResult:
        """Execute LLM classification request against Anthropic Claude."""
        client = self._get_client()

        # Wrap untrusted user query in XML tags for injection defense
        user_message = f"<user_query>\n{query}\n</user_query>"

        response = client.messages.create(
            model=self._settings.routing_model,
            max_tokens=self._settings.max_completion_tokens,
            system=CLASSIFICATION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        content_block = response.content[0]
        response_text = getattr(content_block, "text", "")

        return self._parse_llm_response(query, response_text)

    def _parse_llm_response(self, query: str, response_text: str) -> IntentClassificationResult:
        """Parse and validate the JSON output from Claude."""
        clean_text = response_text.strip()
        # Remove any markdown code fences if model accidentally emitted them
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
            clean_text = re.sub(r"\s*```$", "", clean_text)
            clean_text = clean_text.strip()

        try:
            data = json.loads(clean_text)
        except json.JSONDecodeError as err:
            logger.warning("Failed to decode JSON from LLM classifier response: %s", err)
            fallback = classify_query_heuristic(query, self._confidence_threshold)
            return IntentClassificationResult(
                query=query,
                mode=fallback.mode,
                confidence=fallback.confidence,
                reasoning=f"Malformed JSON from LLM; defaulted to heuristic ({fallback.reasoning})",
                suggested_template_id=fallback.suggested_template_id,
                extracted_entities=fallback.extracted_entities,
                method=ClassificationMethod.FALLBACK,
            )

        # Parse and validate fields
        raw_mode = data.get("mode")
        try:
            mode = QueryIntent(str(raw_mode).strip().upper())
        except (ValueError, AttributeError):
            mode = QueryIntent.HYBRID

        raw_confidence = data.get("confidence", 0.5)
        try:
            confidence = float(raw_confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        reasoning = str(data.get("reasoning", "LLM-classified query intent.")).strip()
        suggested_template_id = data.get("suggested_template_id")
        if suggested_template_id is not None:
            suggested_template_id = str(suggested_template_id).strip().upper()
            if suggested_template_id not in (
                TEMPLATE_DIRECT_NEIGHBORHOOD,
                TEMPLATE_MULTI_HOP_DEPENDENCY,
                TEMPLATE_TEAM_OWNERSHIP_SERVICES,
                TEMPLATE_VULNERABILITY_BLAST_RADIUS,
                TEMPLATE_SERVICE_DEPENDENCY_CHAIN,
                TEMPLATE_DATABASE_PRODUCERS_CONSUMERS,
                TEMPLATE_INFRASTRUCTURE_DEPLOYMENTS,
                TEMPLATE_ENTITY_LOOKUP_BY_ID,
            ):
                suggested_template_id = None

        extracted_entities = data.get("extracted_entities", [])
        if not isinstance(extracted_entities, list):
            extracted_entities = []
        extracted_entities = [str(e).strip() for e in extracted_entities if e]

        # Enforce confidence threshold rule: if confidence < threshold, route to HYBRID
        if mode != QueryIntent.OUT_OF_SCOPE and confidence < self._confidence_threshold:
            logger.info(
                "Classification confidence (%.2f) below threshold (%.2f). Demoting mode '%s' to HYBRID.",
                confidence,
                self._confidence_threshold,
                mode.value,
            )
            return IntentClassificationResult(
                query=query,
                mode=QueryIntent.HYBRID,
                confidence=confidence,
                reasoning=f"LLM confidence ({confidence:.2f}) below threshold ({self._confidence_threshold:.2f}); routing to HYBRID. Original reasoning: {reasoning}",
                suggested_template_id=suggested_template_id or TEMPLATE_DIRECT_NEIGHBORHOOD,
                extracted_entities=extracted_entities,
                method=ClassificationMethod.FALLBACK,
            )

        return IntentClassificationResult(
            query=query,
            mode=mode,
            confidence=confidence,
            reasoning=reasoning,
            suggested_template_id=suggested_template_id,
            extracted_entities=extracted_entities,
            method=ClassificationMethod.LLM,
        )
