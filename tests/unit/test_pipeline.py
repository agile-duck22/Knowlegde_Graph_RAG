"""Unit tests for the end-to-end graph store IngestionPipeline (TASK-107).

Tests cover:
1. Happy path end-to-end run with mocked dependencies.
2. Dry-run mode (no Neo4j writes).
3. Empty corpus / document handling.
4. Document chunking integration with deterministic chunk IDs.
5. Document limit argument handling.
6. Extractor failures on a document (graceful document-level failure reporting).
7. Resolver failures (ambiguous or unknown entity handling).
8. Neo4j batch persistence failure handling and rollback reporting.
9. Pipeline-owned Neo4jClient lifecycle and exception cleanup.
10. CLI arguments parsing and main() function exit codes.
"""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from src.config import Settings
from src.extraction.extractor import ChunkExtractor
from src.extraction.resolver import AmbiguousEntityError, EntityResolver
from src.extraction.schemas import (
    ChunkExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionProvenance,
)
from src.ingest.chunker import TextChunk
from src.ingest.loader import RawDocument
from src.ingest.pipeline import (
    IngestionPipeline,
    IngestionPipelineReport,
    build_parser,
    main,
)
from src.llm.client import MultiProviderClient


@pytest.fixture
def mock_pipeline_settings() -> Settings:
    return Settings(
        llm_provider="anthropic",
        environment="test",
        chunk_size=200,
        chunk_overlap=20,
        anthropic_api_key=SecretStr("sk-ant-mock-key"),
        neo4j_password=SecretStr("mock_password"),
        postgres_password=SecretStr("mock_postgres_password"),
    )


@pytest.fixture
def sample_documents() -> list[RawDocument]:
    return [
        RawDocument(
            document_id="doc-arch-001",
            source_uri="docs/arch.md",
            title="Architecture Overview",
            doc_type="architecture_spec",
            content="Payment Gateway service depends on Checkout Service and writes to Payment Vault PG.",
        ),
        RawDocument(
            document_id="doc-team-002",
            source_uri="docs/teams.md",
            title="Team Directory",
            doc_type="team_directory",
            content="Checkout Engineering Team owns Checkout Service.",
        ),
    ]


@pytest.fixture
def mock_extractor() -> MagicMock:
    extractor = MagicMock(spec=ChunkExtractor)

    def side_effect(chunk: TextChunk) -> ChunkExtractionResult:
        prov = ExtractionProvenance(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
        )
        if "Payment Gateway" in chunk.content:
            e1 = ExtractedEntity(
                canonical_id="service:payment-gateway",
                name="Payment Gateway",
                entity_type="Service",
                description="Processes payments",
                aliases=["PaySvc"],
                confidence=0.95,
                provenance=prov,
            )
            e2 = ExtractedEntity(
                canonical_id="service:checkout-service",
                name="Checkout Service",
                entity_type="Service",
                description="Processes orders",
                aliases=[],
                confidence=0.95,
                provenance=prov,
            )
            rel = ExtractedRelationship(
                source_canonical_id="service:payment-gateway",
                source_type="Service",
                target_canonical_id="service:checkout-service",
                target_type="Service",
                relation_type="DEPENDS_ON",
                description="Calls checkout",
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
        else:
            e1 = ExtractedEntity(
                canonical_id="team:checkout-engineering",
                name="Checkout Engineering",
                entity_type="Team",
                description="Engineering team",
                aliases=[],
                confidence=0.9,
                provenance=prov,
            )
            return ChunkExtractionResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                extraction_model="claude-mock",
                entities=[e1],
                relationships=[],
            )

    extractor.extract.side_effect = side_effect
    return extractor


@pytest.fixture
def mock_neo4j_client() -> MagicMock:
    client = MagicMock()
    client.verify_connectivity.return_value = None
    client.initialise_schema.return_value = None
    client.write_chunk_result.return_value = (2, 1)
    return client


@pytest.fixture
def mock_postgres_client() -> MagicMock:
    client = MagicMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_postgres_writer() -> MagicMock:
    writer = MagicMock()
    writer.write_document_bundle = AsyncMock(return_value=(1, 2))
    return writer


@pytest.fixture
def mock_embedding_generator() -> MagicMock:
    gen = MagicMock()
    gen.generate_batch.return_value = [[0.1] * 384]
    return gen


class TestIngestionPipeline:
    """Unit test suite for IngestionPipeline orchestration."""

    def test_run_successful_end_to_end(
        self,
        mock_pipeline_settings: Settings,
        sample_documents: list[RawDocument],
        mock_extractor: MagicMock,
        mock_neo4j_client: MagicMock,
        mock_postgres_client: MagicMock,
        mock_postgres_writer: MagicMock,
        mock_embedding_generator: MagicMock,
    ) -> None:
        pipeline = IngestionPipeline(
            settings=mock_pipeline_settings,
            extractor=mock_extractor,
            neo4j_client=mock_neo4j_client,
            postgres_client=mock_postgres_client,
            postgres_writer=mock_postgres_writer,
            embedding_generator=mock_embedding_generator,
        )

        report = pipeline.run(documents=sample_documents, dry_run=False)

        assert isinstance(report, IngestionPipelineReport)
        assert report.is_success is True
        assert report.total_documents == 2
        assert report.successful_documents == 2
        assert report.failed_documents == 0
        assert report.total_chunks == 2
        assert report.total_entities_extracted > 0
        assert report.total_relationships_extracted > 0
        assert report.total_entities_written > 0
        assert report.total_relationships_written > 0
        assert report.total_postgres_chunks_written == 2
        assert report.total_postgres_links_written == 4
        assert report.dry_run is False
        assert report.duration_seconds > 0

        # Verify Neo4j calls
        mock_neo4j_client.verify_connectivity.assert_called_once()
        mock_neo4j_client.initialise_schema.assert_called_once()
        assert mock_neo4j_client.write_chunk_result.call_count == 2

        # Verify Postgres & Embedding calls
        assert mock_embedding_generator.generate_batch.call_count == 2
        assert mock_postgres_writer.write_document_bundle.call_count == 2

    def test_run_dry_run_skips_writes(
        self,
        mock_pipeline_settings: Settings,
        sample_documents: list[RawDocument],
        mock_extractor: MagicMock,
        mock_neo4j_client: MagicMock,
        mock_postgres_client: MagicMock,
        mock_postgres_writer: MagicMock,
        mock_embedding_generator: MagicMock,
    ) -> None:
        pipeline = IngestionPipeline(
            settings=mock_pipeline_settings,
            extractor=mock_extractor,
            neo4j_client=mock_neo4j_client,
            postgres_client=mock_postgres_client,
            postgres_writer=mock_postgres_writer,
            embedding_generator=mock_embedding_generator,
        )

        report = pipeline.run(documents=sample_documents, dry_run=True)

        assert report.is_success is True
        assert report.dry_run is True
        assert report.total_entities_written == 0
        assert report.total_relationships_written == 0
        assert report.total_postgres_chunks_written == 0
        assert report.total_postgres_links_written == 0
        mock_neo4j_client.verify_connectivity.assert_not_called()
        mock_neo4j_client.initialise_schema.assert_not_called()
        mock_neo4j_client.write_chunk_result.assert_not_called()
        mock_postgres_writer.write_document_bundle.assert_not_called()
        mock_embedding_generator.generate_batch.assert_not_called()

    def test_run_empty_documents_sequence(
        self,
        mock_pipeline_settings: Settings,
        mock_extractor: MagicMock,
        mock_neo4j_client: MagicMock,
    ) -> None:
        pipeline = IngestionPipeline(
            settings=mock_pipeline_settings,
            extractor=mock_extractor,
            neo4j_client=mock_neo4j_client,
        )

        report = pipeline.run(documents=[])

        assert report.total_documents == 0
        assert report.successful_documents == 0
        assert report.failed_documents == 0
        assert report.is_success is True
        mock_neo4j_client.verify_connectivity.assert_not_called()

    def test_run_document_limit(
        self,
        mock_pipeline_settings: Settings,
        sample_documents: list[RawDocument],
        mock_extractor: MagicMock,
        mock_neo4j_client: MagicMock,
        mock_postgres_client: MagicMock,
        mock_postgres_writer: MagicMock,
        mock_embedding_generator: MagicMock,
    ) -> None:
        pipeline = IngestionPipeline(
            settings=mock_pipeline_settings,
            extractor=mock_extractor,
            neo4j_client=mock_neo4j_client,
            postgres_client=mock_postgres_client,
            postgres_writer=mock_postgres_writer,
            embedding_generator=mock_embedding_generator,
        )

        report = pipeline.run(documents=sample_documents, limit=1)

        assert report.total_documents == 1
        assert report.successful_documents == 1
        assert len(report.document_results) == 1
        assert report.document_results[0].document_id == "doc-arch-001"

    def test_run_no_source_raises_value_error(
        self,
        mock_pipeline_settings: Settings,
    ) -> None:
        pipeline = IngestionPipeline(settings=mock_pipeline_settings)
        with pytest.raises(ValueError, match="No corpus path, documents, or DocumentLoader"):
            pipeline.run()

    def test_run_extractor_failure_records_failed_document(
        self,
        mock_pipeline_settings: Settings,
        sample_documents: list[RawDocument],
        mock_neo4j_client: MagicMock,
    ) -> None:
        bad_extractor = MagicMock(spec=ChunkExtractor)
        bad_extractor.extract.side_effect = RuntimeError("Claude API Timeout")

        pipeline = IngestionPipeline(
            settings=mock_pipeline_settings,
            extractor=bad_extractor,
            neo4j_client=mock_neo4j_client,
        )

        report = pipeline.run(documents=sample_documents)

        assert report.is_success is False
        assert report.total_documents == 2
        assert report.successful_documents == 0
        assert report.failed_documents == 2
        assert len(report.errors) == 2
        assert "Claude API Timeout" in report.errors[0]["error"]
        assert report.document_results[0].status == "FAILED"

    def test_run_resolver_failure_records_failed_document(
        self,
        mock_pipeline_settings: Settings,
        sample_documents: list[RawDocument],
        mock_extractor: MagicMock,
        mock_neo4j_client: MagicMock,
    ) -> None:
        mock_resolver = MagicMock(spec=EntityResolver)
        mock_resolver.resolve_extraction_result.side_effect = AmbiguousEntityError(
            "Ambiguous mention matched multiple targets"
        )

        pipeline = IngestionPipeline(
            settings=mock_pipeline_settings,
            extractor=mock_extractor,
            resolver=mock_resolver,
            neo4j_client=mock_neo4j_client,
        )

        report = pipeline.run(documents=sample_documents)

        assert report.is_success is False
        assert report.failed_documents == 2
        assert "Ambiguous mention" in report.errors[0]["error"]

    def test_run_neo4j_write_failure_records_failed_document(
        self,
        mock_pipeline_settings: Settings,
        sample_documents: list[RawDocument],
        mock_extractor: MagicMock,
        mock_postgres_writer: MagicMock,
        mock_embedding_generator: MagicMock,
    ) -> None:
        failing_neo4j = MagicMock()
        failing_neo4j.verify_connectivity.return_value = None
        failing_neo4j.initialise_schema.return_value = None
        failing_neo4j.write_chunk_result.side_effect = RuntimeError("Cypher execution failed")

        pipeline = IngestionPipeline(
            settings=mock_pipeline_settings,
            extractor=mock_extractor,
            neo4j_client=failing_neo4j,
            postgres_writer=mock_postgres_writer,
            embedding_generator=mock_embedding_generator,
        )

        report = pipeline.run(documents=sample_documents)

        assert report.is_success is False
        assert report.failed_documents == 2
        assert "Cypher execution failed" in report.errors[0]["error"]

    def test_run_postgres_write_failure_records_failed_document(
        self,
        mock_pipeline_settings: Settings,
        sample_documents: list[RawDocument],
        mock_extractor: MagicMock,
        mock_neo4j_client: MagicMock,
        mock_embedding_generator: MagicMock,
    ) -> None:
        failing_pg_writer = MagicMock()
        failing_pg_writer.write_document_bundle = AsyncMock(
            side_effect=RuntimeError("PostgreSQL connection lost")
        )

        pipeline = IngestionPipeline(
            settings=mock_pipeline_settings,
            extractor=mock_extractor,
            neo4j_client=mock_neo4j_client,
            postgres_writer=failing_pg_writer,
            embedding_generator=mock_embedding_generator,
        )

        report = pipeline.run(documents=sample_documents)

        assert report.is_success is False
        assert report.failed_documents == 2
        assert "PostgreSQL connection lost" in report.errors[0]["error"]
        assert report.document_results[0].status == "FAILED"

    def test_pipeline_closes_owned_clients(
        self,
        mock_pipeline_settings: Settings,
        sample_documents: list[RawDocument],
        mock_extractor: MagicMock,
        mock_embedding_generator: MagicMock,
    ) -> None:
        mock_neo4j_instance = MagicMock()
        mock_neo4j_instance.write_chunk_result.return_value = (1, 1)

        mock_pg_instance = MagicMock()
        mock_pg_instance.close = AsyncMock()

        mock_pg_writer_instance = MagicMock()
        mock_pg_writer_instance.write_document_bundle = AsyncMock(return_value=(1, 1))

        with (
            patch("src.ingest.pipeline.Neo4jClient", return_value=mock_neo4j_instance),
            patch("src.ingest.pipeline.PostgresClient", return_value=mock_pg_instance),
            patch("src.ingest.pipeline.PostgresWriter", return_value=mock_pg_writer_instance),
        ):
            pipeline = IngestionPipeline(
                settings=mock_pipeline_settings,
                extractor=mock_extractor,
                neo4j_client=None,  # Pipeline should create and own Neo4jClient
                postgres_client=None,  # Pipeline should create and own PostgresClient
                postgres_writer=None,
                embedding_generator=mock_embedding_generator,
            )
            report = pipeline.run(documents=sample_documents, dry_run=False)
            assert report.is_success is True
            mock_neo4j_instance.close.assert_called_once()
            mock_pg_instance.close.assert_called_once()

    def test_run_reuses_same_event_loop_across_multiple_documents(
        self,
        mock_pipeline_settings: Settings,
        sample_documents: list[RawDocument],
        mock_extractor: MagicMock,
        mock_neo4j_client: MagicMock,
        mock_postgres_client: MagicMock,
        mock_embedding_generator: MagicMock,
    ) -> None:
        """Verify that all async bundle writes run on the same persistent event loop."""
        observed_loops: list[asyncio.AbstractEventLoop] = []

        mock_writer = MagicMock()

        async def fake_write(
            doc: RawDocument,
            chunk_items: list[Any],
        ) -> tuple[int, int]:
            loop = asyncio.get_running_loop()
            observed_loops.append(loop)
            return (len(chunk_items), 1)

        mock_writer.write_document_bundle = fake_write

        pipeline = IngestionPipeline(
            settings=mock_pipeline_settings,
            extractor=mock_extractor,
            neo4j_client=mock_neo4j_client,
            postgres_client=mock_postgres_client,
            postgres_writer=mock_writer,
            embedding_generator=mock_embedding_generator,
        )

        report = pipeline.run(documents=sample_documents, dry_run=False)
        assert report.is_success is True
        assert len(observed_loops) == 2
        # Both documents must have been written using the exact same active event loop
        assert observed_loops[0] is observed_loops[1]

    def test_run_cleanup_handles_exceptions_without_masking(
        self,
        mock_pipeline_settings: Settings,
        sample_documents: list[RawDocument],
        mock_extractor: MagicMock,
        mock_embedding_generator: MagicMock,
    ) -> None:
        """Verify that exceptions in close() are logged and do not mask pipeline errors."""
        mock_neo4j_instance = MagicMock()
        mock_neo4j_instance.verify_connectivity.return_value = None
        mock_neo4j_instance.initialise_schema.return_value = None
        mock_neo4j_instance.write_chunk_result.return_value = (2, 1)
        mock_neo4j_instance.close.side_effect = RuntimeError("Neo4j driver close failed")

        mock_pg_instance = MagicMock()
        mock_pg_instance.close = AsyncMock(side_effect=RuntimeError("PG pool close failed"))

        mock_pg_writer_instance = MagicMock()
        mock_pg_writer_instance.write_document_bundle = AsyncMock(
            side_effect=RuntimeError("Database failure during write")
        )

        with (
            patch("src.ingest.pipeline.Neo4jClient", return_value=mock_neo4j_instance),
            patch("src.ingest.pipeline.PostgresClient", return_value=mock_pg_instance),
            patch("src.ingest.pipeline.PostgresWriter", return_value=mock_pg_writer_instance),
        ):
            pipeline = IngestionPipeline(
                settings=mock_pipeline_settings,
                extractor=mock_extractor,
                embedding_generator=mock_embedding_generator,
            )
            report = pipeline.run(documents=sample_documents, dry_run=False)
            assert report.is_success is False
            assert report.failed_documents == 2
            assert "Database failure during write" in report.errors[0]["error"]
            mock_neo4j_instance.close.assert_called_once()
            mock_pg_instance.close.assert_called_once()

    def test_lazy_extractor_initialisation_with_dummy_key(
        self,
        mock_pipeline_settings: Settings,
    ) -> None:
        pipeline = IngestionPipeline(settings=mock_pipeline_settings)
        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            extractor = pipeline._get_extractor()
            assert isinstance(extractor, ChunkExtractor)
            assert isinstance(extractor.client, MultiProviderClient)
            assert extractor.client._anthropic_client is None
            extractor.client._get_anthropic_client()
            mock_anthropic_cls.assert_called_once()


class TestIngestionCLI:
    """Unit tests for the CLI entrypoint."""

    def test_build_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--corpus", "custom/corpus.jsonl", "--dry-run", "--limit", "5"])
        assert args.corpus == "custom/corpus.jsonl"
        assert args.dry_run is True
        assert args.limit == 5

    def test_main_missing_corpus_file(self) -> None:
        exit_code = main(["--corpus", "non_existent_file_xyz.jsonl"])
        assert exit_code == 1

    def test_main_success_dry_run(
        self,
        tmp_path: Path,
    ) -> None:
        corpus_file = tmp_path / "test_corpus.jsonl"
        corpus_file.write_text(
            '{"document_id": "doc-01", "source_uri": "doc1.md", "title": "Doc 1", "content": "Sample content"}\n',
            encoding="utf-8",
        )

        with patch.object(IngestionPipeline, "run") as mock_run:
            mock_run.return_value = IngestionPipelineReport(
                total_documents=1,
                successful_documents=1,
                failed_documents=0,
                dry_run=True,
            )
            exit_code = main(["--corpus", str(corpus_file), "--dry-run"])
            assert exit_code == 0
            mock_run.assert_called_once()
