"""Unit tests for Phase H resumable batch coordinator (Gate D)."""

from __future__ import annotations

from pathlib import Path

from experiments.scipaper.scripts.run_phase_h_migration import (
    load_or_create_checkpoint,
    save_checkpoint,
)
from src.ingest.loader import RawDocument


def test_checkpoint_creation_and_resume(tmp_path: Path) -> None:
    checkpoint_file = tmp_path / "checkpoint.json"
    docs = [
        RawDocument(
            document_id=f"doc-{i:03d}",
            source_uri=f"uri-{i}",
            title=f"Paper {i}",
            doc_type="scientific_paper",
            content="Content",
            metadata={},
        )
        for i in range(25)
    ]

    cp = load_or_create_checkpoint(checkpoint_file, docs, batch_size=10)
    assert cp.total_papers == 25
    assert cp.total_batches == 3
    assert len(cp.batches) == 3
    assert cp.completed_batches_count == 0
    assert len(cp.batches[0].paper_ids) == 10
    assert len(cp.batches[1].paper_ids) == 10
    assert len(cp.batches[2].paper_ids) == 5

    # Simulate completion of batch 0
    cp.batches[0].status = "COMPLETED"
    cp.completed_batches_count = 1
    cp.completed_paper_ids = cp.batches[0].paper_ids
    save_checkpoint(checkpoint_file, cp)

    # Reload checkpoint
    cp_reloaded = load_or_create_checkpoint(checkpoint_file, docs, batch_size=10)
    assert cp_reloaded.completed_batches_count == 1
    assert cp_reloaded.batches[0].status == "COMPLETED"
    assert cp_reloaded.batches[1].status == "PENDING"
