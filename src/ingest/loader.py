"""Document schema models and JSONL dataset loader.

Loads raw enterprise document records from JSONL files, validates fields against
Pydantic schemas, and provides structured RawDocument instances.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RawDocument(BaseModel):
    """Raw document record matching the ingestion data contract."""

    document_id: str = Field(
        ..., min_length=1, description="Unique, stable document identifier"
    )
    source_uri: str = Field(
        ..., min_length=1, description="Relative or absolute source document path/URI"
    )
    title: str = Field(..., min_length=1, description="Document title")
    doc_type: str = Field(
        default="technical_doc", min_length=1, description="Domain document category"
    )
    content: str = Field(default="", description="Raw markdown or text document content")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary domain metadata dictionary"
    )

    model_config = ConfigDict(frozen=True, extra="ignore")

    @field_validator("document_id", "source_uri", "title", "doc_type")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info: Any) -> str:
        """Ensure core identifiers are not blank whitespace."""
        if not v.strip():
            raise ValueError(f"Field '{info.field_name}' cannot be empty or whitespace only")
        return v.strip()

    @property
    def content_hash(self) -> str:
        """Compute SHA-256 hash of the document content for change detection."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class DocumentLoader:
    """Deterministic loader for JSONL document corpora."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(self) -> list[RawDocument]:
        """Load and validate all document records from the JSONL file.

        Raises:
            FileNotFoundError: If the corpus file does not exist.
            ValueError: If JSON is malformed, required fields are missing, or duplicate IDs exist.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Corpus file not found at: {self.file_path.resolve()}")

        documents: list[RawDocument] = []
        seen_ids: set[str] = set()

        with self.file_path.open("r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue  # Skip blank lines safely

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON on line {line_number} in {self.file_path.name}: {exc}"
                    ) from exc

                if not isinstance(data, dict):
                    raise ValueError(
                        f"Invalid JSON on line {line_number} in {self.file_path.name}: Expected JSON object, got {type(data).__name__}"
                    )

                try:
                    doc = RawDocument.model_validate(data)
                except Exception as exc:
                    raise ValueError(
                        f"Schema validation error on line {line_number} in {self.file_path.name}: {exc}"
                    ) from exc

                if doc.document_id in seen_ids:
                    raise ValueError(
                        f"Duplicate document_id '{doc.document_id}' detected on line {line_number} in {self.file_path.name}"
                    )

                seen_ids.add(doc.document_id)
                documents.append(doc)

        return documents

    def iter_load(self) -> Iterator[RawDocument]:
        """Stream documents lazily as an iterator."""
        yield from self.load()
