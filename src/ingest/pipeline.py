"""End-to-end ingestion pipeline and CLI for graph store.

Workflow:
1. Load raw documents from JSONL (DocumentLoader)
2. Chunk documents into deterministic TextChunk objects (DocumentChunker)
3. Extract structured entities and relationships via Claude (ChunkExtractor)
4. Resolve mentions and normalize aliases into canonical IDs (EntityResolver)
5. Persist schema constraints and extracted graph structures to Neo4j (Neo4jClient)

Design & Invariants:
- Zero dynamic Cypher formatting.
- Strict ontology & schema adherence.
- Structured document-level error tracking (no silent failure).
- Safe logging (no secret tokens or full sensitive document content).
- Deterministic chunk and canonical entity IDs preserved across pipeline.
- Resource safety: Neo4j client closed reliably on all code paths.
- Dry-run mode: executes loading, chunking, extraction, and resolution without database writes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.db.neo4j import Neo4jClient
from src.extraction.extractor import ChunkExtractor
from src.extraction.resolver import EntityResolver
from src.extraction.schemas import ChunkExtractionResult
from src.ingest.chunker import DocumentChunker, TextChunk
from src.ingest.loader import DocumentLoader, RawDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentIngestionResult:
    """Summary record for single document processing."""

    document_id: str
    chunks_count: int = 0
    entities_count: int = 0
    relationships_count: int = 0
    status: str = "SUCCESS"  # SUCCESS, FAILED, SKIPPED
    error_message: str | None = None
    chunk_results: list[ChunkExtractionResult] = field(default_factory=list)


@dataclass
class IngestionPipelineReport:
    """Aggregated execution summary for the ingestion run."""

    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_chunks: int = 0
    total_entities_extracted: int = 0
    total_relationships_extracted: int = 0
    total_entities_written: int = 0
    total_relationships_written: int = 0
    dry_run: bool = False
    duration_seconds: float = 0.0
    document_results: list[DocumentIngestionResult] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Pipeline is successful if all attempted documents succeeded."""
        return self.failed_documents == 0


class IngestionPipeline:
    """Coordinates end-to-end loading, chunking, extraction, resolution, and graph persistence."""

    def __init__(
        self,
        settings: Settings | None = None,
        loader: DocumentLoader | None = None,
        chunker: DocumentChunker | None = None,
        extractor: ChunkExtractor | None = None,
        resolver: EntityResolver | None = None,
        neo4j_client: Neo4jClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.loader = loader
        self.chunker = chunker or DocumentChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.extractor = extractor
        self.resolver = resolver or EntityResolver()
        self.neo4j_client = neo4j_client

    def _get_extractor(self) -> ChunkExtractor:
        """Lazily initialize ChunkExtractor if not provided."""
        if self.extractor is not None:
            return self.extractor

        import anthropic

        api_key = (
            self.settings.anthropic_api_key.get_secret_value()
            if self.settings.anthropic_api_key
            else ""
        )
        client = anthropic.Anthropic(
            api_key=api_key or "sk-ant-dummy",
            timeout=self.settings.request_timeout_seconds,
        )
        self.extractor = ChunkExtractor(client=client, settings=self.settings)
        return self.extractor

    def process_document(
        self,
        document: RawDocument,
        extractor: ChunkExtractor,
    ) -> tuple[DocumentIngestionResult, list[ChunkExtractionResult]]:
        """Chunk a document, extract entities/relationships, and resolve aliases."""
        chunks: list[TextChunk] = self.chunker.chunk(document)
        doc_res = DocumentIngestionResult(
            document_id=document.document_id,
            chunks_count=len(chunks),
        )

        resolved_results: list[ChunkExtractionResult] = []
        ent_count = 0
        rel_count = 0

        for chunk in chunks:
            raw_extraction = extractor.extract(chunk)
            resolved_extraction = self.resolver.resolve_extraction_result(raw_extraction)
            resolved_results.append(resolved_extraction)
            ent_count += len(resolved_extraction.entities)
            rel_count += len(resolved_extraction.relationships)

        doc_res.entities_count = ent_count
        doc_res.relationships_count = rel_count
        doc_res.chunk_results = resolved_results
        return doc_res, resolved_results

    def run(
        self,
        corpus_path: str | Path | None = None,
        documents: Sequence[RawDocument] | None = None,
        *,
        dry_run: bool = False,
        limit: int | None = None,
    ) -> IngestionPipelineReport:
        """Execute the ingestion pipeline over a corpus path or document sequence.

        Parameters
        ----------
        corpus_path:
            Path to JSONL corpus file.
        documents:
            Optional direct sequence of RawDocument objects (overrides corpus_path).
        dry_run:
            If True, skips schema initialisation and Neo4j database writes.
        limit:
            Optional max number of documents to ingest.

        Returns
        -------
        IngestionPipelineReport
            Detailed pipeline execution report.
        """
        start_time = time.perf_counter()
        report = IngestionPipelineReport(dry_run=dry_run)

        # 1. Load documents
        docs_to_process: list[RawDocument] = []
        if documents is not None:
            docs_to_process = list(documents)
        elif corpus_path is not None:
            loader = DocumentLoader(corpus_path)
            docs_to_process = loader.load()
        elif self.loader is not None:
            docs_to_process = self.loader.load()
        else:
            raise ValueError("No corpus path, documents, or DocumentLoader provided to pipeline.")

        if limit is not None and limit > 0:
            docs_to_process = docs_to_process[:limit]

        report.total_documents = len(docs_to_process)
        if not docs_to_process:
            report.duration_seconds = time.perf_counter() - start_time
            logger.info("Ingestion pipeline finished: 0 documents to process.")
            return report

        extractor = self._get_extractor()

        # 2. Setup Neo4j client if writing
        client_to_use: Neo4jClient | None = None
        owns_client = False
        if not dry_run:
            if self.neo4j_client is not None:
                client_to_use = self.neo4j_client
            else:
                client_to_use = Neo4jClient(self.settings)
                owns_client = True

        try:
            if not dry_run and client_to_use is not None:
                logger.info("Initialising Neo4j schema constraints and indexes...")
                client_to_use.verify_connectivity()
                client_to_use.initialise_schema()

            # 3. Process each document
            for idx, doc in enumerate(docs_to_process, start=1):
                logger.info(
                    "Processing document [%d/%d] id='%s' title='%s'",
                    idx,
                    report.total_documents,
                    doc.document_id,
                    doc.title,
                )
                try:
                    doc_res, resolved_chunk_results = self.process_document(doc, extractor)
                    report.total_chunks += doc_res.chunks_count
                    report.total_entities_extracted += doc_res.entities_count
                    report.total_relationships_extracted += doc_res.relationships_count

                    # Persist to Neo4j if not dry_run
                    if not dry_run and client_to_use is not None:
                        for chunk_res in resolved_chunk_results:
                            e_cnt, r_cnt = client_to_use.write_chunk_result(chunk_res)
                            report.total_entities_written += e_cnt
                            report.total_relationships_written += r_cnt

                    doc_res.status = "SUCCESS"
                    report.successful_documents += 1
                    report.document_results.append(doc_res)

                except Exception as exc:
                    logger.error(
                        "Failed processing document id='%s': %s",
                        doc.document_id,
                        exc,
                        exc_info=True,
                    )
                    doc_res = DocumentIngestionResult(
                        document_id=doc.document_id,
                        status="FAILED",
                        error_message=str(exc),
                    )
                    report.failed_documents += 1
                    report.document_results.append(doc_res)
                    report.errors.append(
                        {
                            "document_id": doc.document_id,
                            "error": str(exc),
                        }
                    )

        finally:
            if owns_client and client_to_use is not None:
                logger.debug("Closing pipeline-owned Neo4jClient driver.")
                client_to_use.close()

        report.duration_seconds = time.perf_counter() - start_time
        logger.info(
            "Ingestion complete: %d/%d documents succeeded in %.2fs. "
            "(Chunks: %d, Entities Extracted: %d, Relationships Extracted: %d, Entities Written: %d, Relationships Written: %d, DryRun: %s)",
            report.successful_documents,
            report.total_documents,
            report.duration_seconds,
            report.total_chunks,
            report.total_entities_extracted,
            report.total_relationships_extracted,
            report.total_entities_written,
            report.total_relationships_written,
            report.dry_run,
        )
        return report


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for graph ingestion."""
    parser = argparse.ArgumentParser(
        description="Knowledge Graph RAG - End-to-End Graph Store Ingestion Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/raw/corpus.jsonl",
        help="Path to the JSONL corpus file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform extraction and resolution without writing to Neo4j",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to ingest",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override logging verbosity level",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for graph store ingestion."""
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    log_level = args.log_level or settings.log_level
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error("Corpus file not found: %s", corpus_path.resolve())
        return 1

    pipeline = IngestionPipeline(settings=settings)
    logger.info(
        "Starting ingestion from '%s' (dry_run=%s, limit=%s)", corpus_path, args.dry_run, args.limit
    )

    try:
        report = pipeline.run(
            corpus_path=corpus_path,
            dry_run=args.dry_run,
            limit=args.limit,
        )

        print("\n" + "=" * 60)
        print("INGESTION PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Status:                    {'SUCCESS' if report.is_success else 'FAILED'}")
        print(f"Total Documents:           {report.total_documents}")
        print(f"Successful Documents:      {report.successful_documents}")
        print(f"Failed Documents:          {report.failed_documents}")
        print(f"Total Chunks:              {report.total_chunks}")
        print(f"Entities Extracted:        {report.total_entities_extracted}")
        print(f"Relationships Extracted:   {report.total_relationships_extracted}")
        print(f"Entities Written (Neo4j):  {report.total_entities_written}")
        print(f"Rels Written (Neo4j):      {report.total_relationships_written}")
        print(f"Dry Run:                   {report.dry_run}")
        print(f"Duration:                  {report.duration_seconds:.2f}s")
        print("=" * 60)

        if report.failed_documents > 0:
            print("\nDocument Failure Details:")
            for err in report.errors:
                print(f" - Doc [{err['document_id']}]: {err['error']}")
            return 1

        return 0

    except Exception as exc:
        logger.error("Pipeline fatal error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
