"""Unit tests verifying package metadata and offline test scaffolding."""

from typing import Any

import src


def test_package_version() -> None:
    """Verify package version is defined and conforms to semver."""
    assert src.__version__ == "0.1.0"


def test_scaffolding_fixture(sample_chunk_record: dict[str, Any]) -> None:
    """Verify base test fixture loading works offline."""
    assert "id" in sample_chunk_record
    assert sample_chunk_record["token_count"] > 0
