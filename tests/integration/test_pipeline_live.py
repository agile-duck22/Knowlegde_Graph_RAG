"""Live integration tests for the end-to-end IngestionPipeline against local Neo4j.

Uses a synthetic fixture with a mocked extraction phase (no external paid Claude calls)
to test the complete pipeline: Document -> Chunker -> Extractor -> Resolver -> Live Neo4j.
"""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.db.neo4j import Neo4jClient
from src.extraction.extractor import ChunkExtractor
from src.extraction.resolver import EntityResolver
from src.extraction.schemas import (
    ChunkExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionProvenance,
)
from src.ingest.chunker import TextChunk
from src.ingest.loader import RawDocument
from src.ingest.pipeline import IngestionPipeline

_TEST_IDS = [
    "service:billing-service",
    "database:financial-ledger-pg",
    "service:checkout-service",
    "api:process-charge-v2",
]


@pytest.fixture(scope="module")
def live_neo4j() -> Generator[Neo4jClient, None, None]:
    """Provide a live Neo4jClient connected to the local Neo4j container."""
    settings = Settings(
        neo4j_uri="bolt://127.0.0.1:7687",
        neo4j_user="neo4j",
        neo4j_password=SecretStr("neo4j_password"),
        anthropic_api_key=SecretStr("sk-ant-test-key-12345"),
        openai_api_key=SecretStr("sk-test-openai-key-12345"),
        postgres_password=SecretStr("postgres_password"),
    )
    client = Neo4jClient(settings)
    yield client
    client.close()


@pytest.fixture(autouse=True)
def cleanup_pipeline_test_nodes(live_neo4j: Neo4jClient) -> Generator[None, None, None]:
    """Clean up any nodes created by the pipeline integration test."""
    _clean(live_neo4j)
    yield
    _clean(live_neo4j)


def _clean(client: Neo4jClient) -> None:
    with client.driver.session() as session:
        session.run(
            "MATCH (n:Entity) WHERE n.id IN $ids DETACH DELETE n",
            {"ids": _TEST_IDS},
        )


@pytest.mark.integration
class TestPipelineLiveIntegration:
    """End-to-end pipeline run against live local Neo4j instance."""

    def test_pipeline_end_to_end_live_neo4j(
        self,
        live_neo4j: Neo4jClient,
        tmp_path: Path,
    ) -> None:
        # 1. Create synthetic document corpus
        doc1 = RawDocument(
            document_id="doc-integ-billing-01",
            source_uri="docs/billing_spec.md",
            title="Billing Spec",
            doc_type="technical_doc",
            content="Billing Service writes to Financial Ledger PG database and is owned by Billing and Finance Tech Team.",
        )
        doc2 = RawDocument(
            document_id="doc-integ-orders-02",
            source_uri="docs/orders_spec.md",
            title="Orders Spec",
            doc_type="technical_doc",
            content="Checkout Service calls Process Charge API v2.",
        )

        # 2. Mock extractor to return un-normalized mentions (testing resolver -> neo4j)
        mock_extractor = MagicMock(spec=ChunkExtractor)

        def mock_extract(chunk: TextChunk) -> ChunkExtractionResult:
            prov = ExtractionProvenance(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
            )
            if chunk.document_id == "doc-integ-billing-01":
                e1 = ExtractedEntity(
                    canonical_id="service:billing-service",
                    name="Billing Service",
                    entity_type="Service",
                    description="Handles billing",
                    aliases=["billing svc"],
                    confidence=0.95,
                    provenance=prov,
                )
                e2 = ExtractedEntity(
                    canonical_id="database:financial-ledger-pg",
                    name="Financial Ledger PG",
                    entity_type="Database",
                    description="Financial DB",
                    aliases=["financial ledger pg"],
                    confidence=0.92,
                    provenance=prov,
                )
                rel = ExtractedRelationship(
                    source_canonical_id="service:billing-service",
                    source_type="Service",
                    target_canonical_id="database:financial-ledger-pg",
                    target_type="Database",
                    relation_type="WRITES_TO",
                    description="Writes ledger entries",
                    confidence=0.95,
                    source_chunk_ids=[chunk.chunk_id],
                    provenance=prov,
                )
                return ChunkExtractionResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    extraction_model="claude-mock",
                    entities=[e1, e2],
                    relationships=[rel],
                )
            else:
                e1 = ExtractedEntity(
                    canonical_id="service:checkout-service",
                    name="Checkout Service",
                    entity_type="Service",
                    description="Processes checkout",
                    aliases=["checkout svc"],
                    confidence=0.95,
                    provenance=prov,
                )
                e2 = ExtractedEntity(
                    canonical_id="api:process-charge-v2",
                    name="Process Charge API v2",
                    entity_type="API",
                    description="Payment API",
                    aliases=["process charge v2"],
                    confidence=0.9,
                    provenance=prov,
                )
                rel = ExtractedRelationship(
                    source_canonical_id="service:checkout-service",
                    source_type="Service",
                    target_canonical_id="api:process-charge-v2",
                    target_type="API",
                    relation_type="CALLS_API",
                    description="Invokes payment charge",
                    confidence=0.9,
                    source_chunk_ids=[chunk.chunk_id],
                    provenance=prov,
                )
                return ChunkExtractionResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    extraction_model="claude-mock",
                    entities=[e1, e2],
                    relationships=[rel],
                )

        mock_extractor.extract.side_effect = mock_extract

        pipeline = IngestionPipeline(
            settings=None,
            extractor=mock_extractor,
            resolver=EntityResolver(),
            neo4j_client=live_neo4j,
        )

        # 3. Run pipeline
        report = pipeline.run(documents=[doc1, doc2], dry_run=False)

        # 4. Verify report metrics
        assert report.is_success is True
        assert report.total_documents == 2
        assert report.successful_documents == 2
        assert report.failed_documents == 0
        assert report.total_chunks == 2
        assert report.total_entities_written == 4
        assert report.total_relationships_written == 2

        # 5. Verify graph state directly in Neo4j
        with live_neo4j.driver.session() as session:
            # Check specific entities created
            res_nodes = session.run(
                "MATCH (n:Entity) WHERE n.id IN ['service:billing-service', 'database:financial-ledger-pg', 'service:checkout-service', 'api:process-charge-v2'] RETURN count(n) AS cnt"
            )
            rec_nodes = res_nodes.single()
            assert rec_nodes is not None
            assert rec_nodes["cnt"] == 4

            # Check relationship created
            res_rel = session.run(
                """
                MATCH (s:Entity {id: 'service:billing-service'})-[r:WRITES_TO]->(t:Entity {id: 'database:financial-ledger-pg'})
                RETURN s.name AS s_name, t.name AS t_name, r.relation_type AS rel_type
                """
            )
            rec_rel = res_rel.single()
            assert rec_rel is not None
            assert rec_rel["s_name"] == "Billing Service"
            assert rec_rel["t_name"] == "Financial Ledger PG"
            assert rec_rel["rel_type"] == "WRITES_TO"
