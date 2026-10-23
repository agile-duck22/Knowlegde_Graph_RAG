"""Data contracts and schemas for Scientific-Paper Generalization Experiment.

Defines schemas for:
1. PaperManifestEntry: Paper provenance, license, source identifier, and checksum tracking.
2. ScientificPaperDocument: Extension and specialization of the canonical RawDocument contract.
3. ScientificOntology: Domain entity and relationship definitions tailored for academic research.
4. ScientificBenchmarkDataset: 50-question benchmark validation and category allocation contracts.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.eval.schema import BenchmarkQuestion
from src.ingest.loader import RawDocument


class ContentType(StrEnum):
    """Document text coverage category."""

    ABSTRACT = "abstract"
    FULL_TEXT = "full_text"
    EXTRACTED_SECTIONS = "extracted_sections"


class PaperManifestEntry(BaseModel):
    """Provenance and licensing metadata for an ingested scientific paper."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    paper_id: str = Field(
        ...,
        pattern=r"^paper-[a-z0-9\.\-_]+$",
        description="Unique canonical identifier for the paper (e.g. paper-arxiv-2305.12345)",
    )
    title: str = Field(..., min_length=5, description="Official paper title")
    authors: list[str] = Field(..., min_length=1, description="List of contributing authors")
    year: int = Field(..., ge=1950, le=2030, description="Publication year")
    venue: str = Field(..., min_length=1, description="Conference, journal, or preprint repository")
    source_url: str = Field(..., pattern=r"^https?://", description="Authoritative public URL")
    license: str = Field(
        ..., min_length=1, description="Open-access license identifier (e.g. CC-BY-4.0)"
    )
    content_type: ContentType = Field(..., description="Abstract vs full text content level")
    sha256: str = Field(
        ..., pattern=r"^[a-f0-9]{64}$", description="SHA-256 content checksum of extracted text"
    )
    doi: str | None = Field(default=None, description="Optional Digital Object Identifier")
    arxiv_id: str | None = Field(
        default=None, description="Optional arXiv identifier (e.g. 2305.12345)"
    )


class ScientificPaperDocument(RawDocument):
    """Specialized RawDocument adhering to the core ingestion data contract."""

    doc_type: str = Field(
        default="scientific_paper",
        description="Domain document category matching RawDocument requirements",
    )


class ScientificBenchmarkDataset(BaseModel):
    """50-question benchmark dataset specification for the scientific paper experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(default="1.0.0", description="Benchmark version string")
    domain: str | None = Field(default=None, description="Optional benchmark subject domain")
    total_questions: int = Field(
        default=50, description="Total questions count (must be exactly 50)"
    )
    questions: list[BenchmarkQuestion] = Field(
        ..., min_length=1, description="List of benchmark questions"
    )

    @field_validator("questions")
    @classmethod
    def validate_benchmark_composition(
        cls, questions: list[BenchmarkQuestion]
    ) -> list[BenchmarkQuestion]:
        """Validate question ID uniqueness."""
        ids = [q.id for q in questions]
        if len(ids) != len(set(ids)):
            raise ValueError("All benchmark question IDs must be globally unique")
        return questions


# ---------------------------------------------------------------------------
# Scientific Domain Ontology Specifications
# ---------------------------------------------------------------------------

ScientificEntityType = Literal[
    "Paper",
    "Author",
    "Model",
    "Dataset",
    "Metric",
    "Task",
    "Method",
    "ResearchOrganization",
]

ScientificRelationshipType = Literal[
    "AUTHORED",
    "PROPOSES",
    "EVALUATED_ON",
    "REPORTS_METRIC",
    "USES_METHOD",
    "AFFILIATED_WITH",
    "CITES",
]

SCIENTIFIC_CANONICAL_PREFIXES: dict[ScientificEntityType, str] = {
    "Paper": "paper",
    "Author": "author",
    "Model": "model",
    "Dataset": "dataset",
    "Metric": "metric",
    "Task": "task",
    "Method": "method",
    "ResearchOrganization": "org",
}

SCIENTIFIC_ID_REGEX = re.compile(
    r"^(paper|author|model|dataset|metric|task|method|org):[a-z0-9][a-z0-9\-\.]*$"
)
