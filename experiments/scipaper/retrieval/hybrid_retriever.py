"""Experiment-only Scientific Hybrid Retriever with Reciprocal Rank Fusion (RRF).

Combines:
1. Vector retrieval against isolated PostgreSQL (scipaper_db).
2. Graph retrieval against isolated Neo4j (scipaper graph).
3. RRF Rank Merging:
   RRF_score(chunk) = sum( 1.0 / (rrf_k + rank_i) ) for each retrieval list i where chunk appears.
4. Output format normalization preserving provenance, scores, ranks, and document IDs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from experiments.scipaper.retrieval.cypher_templates import (
    TEMPLATE_FIND_AUTHORS_OF_PAPER,
    TEMPLATE_FIND_METHODS_OF_PAPER,
    TEMPLATE_FIND_PAPERS_BY_METHOD,
    TEMPLATE_TRAVERSE_PAPER_NEIGHBORHOOD,
)
from experiments.scipaper.retrieval.graph_retriever import (
    ScientificGraphQueryResult,
    ScientificGraphRetriever,
)
from src.retrieval.vector_retriever import RetrievedChunk, VectorRetriever

logger = logging.getLogger(__name__)

# Regular expressions to detect scientific entities in benchmark questions
ARXIV_ID_PATTERN = re.compile(r"(?:paper-)?(?:arxiv-)?(\d{4}\.\d{4,5})", re.IGNORECASE)
KNOWN_METHODS = [
    (
        "method:retrieval-augmented-generation",
        "Retrieval-Augmented Generation",
        ["rag", "retrieval-augmented generation"],
    ),
    ("method:knowledge-graph", "Knowledge Graph", ["knowledge graph", "kg", "knowledge graphs"]),
    (
        "method:dense-retrieval",
        "Dense Retrieval",
        ["dense retrieval", "dense vector", "dense search"],
    ),
    ("method:bm25", "BM25", ["bm25"]),
    ("method:vector-search", "Vector Search", ["vector search", "vector retrieval"]),
]


@dataclass(frozen=True)
class HybridRetrievedChunk:
    """Unified normalized result item for hybrid retrieval."""

    chunk_id: str
    document_id: str
    content: str
    vector_rank: int | None = None
    vector_similarity: float | None = None
    graph_rank: int | None = None
    graph_provenance_sources: list[str] = field(default_factory=list)
    rrf_score: float = 0.0
    final_rank: int = 1


@dataclass(frozen=True)
class HybridRetrievalOutput:
    """Complete retrieval traces across vector-only, graph-only, and fused hybrid."""

    query: str
    vector_results: list[RetrievedChunk]
    graph_result: ScientificGraphQueryResult | None
    graph_hydrated_chunks: list[RetrievedChunk]
    hybrid_results: list[HybridRetrievedChunk]
    graph_coverage: bool
    detected_entities: dict[str, Any]


class ScientificHybridRetriever:
    """Hybrid Retriever combining vector and graph retrieval via RRF."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        graph_retriever: ScientificGraphRetriever,
        rrf_k: int = 60,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.rrf_k = rrf_k

    def detect_entities(self, query: str) -> dict[str, Any]:
        """Detect mentions of papers, authors, or methods in scientific query."""
        detected_papers: list[str] = []
        for match in ARXIV_ID_PATTERN.finditer(query):
            arxiv_num = match.group(1)
            detected_papers.append(f"paper:arxiv-{arxiv_num}")

        detected_methods: list[str] = []
        lower_q = query.lower()
        for mid, _mname, keywords in KNOWN_METHODS:
            for kw in keywords:
                if kw in lower_q:
                    detected_methods.append(mid)
                    break

        return {
            "papers": detected_papers,
            "methods": detected_methods,
        }

    async def execute_graph_search(
        self,
        query: str,
        detected_entities: dict[str, Any],
        limit: int = 20,
    ) -> ScientificGraphQueryResult | None:
        """Run graph retrieval if recognized entities or relationships exist."""
        papers = detected_entities.get("papers", [])
        methods = detected_entities.get("methods", [])

        # If a paper is present in query
        if papers:
            paper_id = papers[0]
            paper_name = paper_id.replace("paper:", "paper-")
            lower_q = query.lower()

            # Check if query asks about authors or methods
            if "author" in lower_q or "who wrote" in lower_q:
                return self.graph_retriever.execute_template(
                    TEMPLATE_FIND_AUTHORS_OF_PAPER,
                    {"paper_id": paper_id, "paper_name": paper_name},
                    limit=limit,
                )
            elif "method" in lower_q or "technique" in lower_q:
                return self.graph_retriever.execute_template(
                    TEMPLATE_FIND_METHODS_OF_PAPER,
                    {"paper_id": paper_id, "paper_name": paper_name},
                    limit=limit,
                )
            else:
                return self.graph_retriever.execute_template(
                    TEMPLATE_TRAVERSE_PAPER_NEIGHBORHOOD,
                    {"paper_id": paper_id, "paper_name": paper_name},
                    limit=limit,
                )

        # If a method is present
        if methods:
            method_id = methods[0]
            method_name = method_id.replace("method:", "").replace("-", " ")
            return self.graph_retriever.execute_template(
                TEMPLATE_FIND_PAPERS_BY_METHOD,
                {"method_id": method_id, "method_name": method_name},
                limit=limit,
            )

        return None

    async def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        similarity_threshold: float = 0.0,
    ) -> HybridRetrievalOutput:
        """Execute vector search, graph search, and merge with RRF."""
        # 1. Vector Search
        vector_results: list[RetrievedChunk] = await self.vector_retriever.search(
            query_vector=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        # 2. Graph Search
        detected = self.detect_entities(query)
        graph_res: ScientificGraphQueryResult | None = await self.execute_graph_search(
            query=query,
            detected_entities=detected,
            limit=top_k,
        )

        # 3. Hydrate Graph Provenance Chunks from PostgreSQL
        graph_hydrated: list[RetrievedChunk] = []
        if graph_res and graph_res.provenance_chunk_ids:
            graph_hydrated = await self.vector_retriever.get_chunks_by_ids(
                chunk_ids=graph_res.provenance_chunk_ids
            )

        # 4. Reciprocal Rank Fusion (RRF)
        # RRF Score = sum( 1 / (rrf_k + rank_i) )
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}
        vec_ranks: dict[str, int] = {}
        graph_ranks: dict[str, int] = {}

        # Tally vector ranks (1-indexed)
        for rank, chk in enumerate(vector_results, start=1):
            scores[chk.chunk_id] = scores.get(chk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            chunk_map[chk.chunk_id] = chk
            vec_ranks[chk.chunk_id] = rank

        # Tally graph ranks (1-indexed)
        for rank, chk in enumerate(graph_hydrated, start=1):
            scores[chk.chunk_id] = scores.get(chk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))
            chunk_map[chk.chunk_id] = chk
            graph_ranks[chk.chunk_id] = rank

        # Sort chunks by fused RRF score descending
        sorted_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        hybrid_items: list[HybridRetrievedChunk] = []
        for final_rank, cid in enumerate(sorted_chunk_ids[:top_k], start=1):
            chk = chunk_map[cid]
            hybrid_items.append(
                HybridRetrievedChunk(
                    chunk_id=cid,
                    document_id=chk.document_id,
                    content=chk.content,
                    vector_rank=vec_ranks.get(cid),
                    vector_similarity=chk.similarity_score if cid in vec_ranks else None,
                    graph_rank=graph_ranks.get(cid),
                    graph_provenance_sources=[graph_res.template_id]
                    if graph_res and cid in graph_ranks
                    else [],
                    rrf_score=scores[cid],
                    final_rank=final_rank,
                )
            )

        return HybridRetrievalOutput(
            query=query,
            vector_results=vector_results,
            graph_result=graph_res,
            graph_hydrated_chunks=graph_hydrated,
            hybrid_results=hybrid_items,
            graph_coverage=(graph_res is not None and not graph_res.is_empty),
            detected_entities=detected,
        )
