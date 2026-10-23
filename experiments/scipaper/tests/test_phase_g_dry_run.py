"""Tests for Phase G Gate B dry-run structural guarantees."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from experiments.scipaper.scripts.preflight_check import parse_env_file_isolated
from experiments.scipaper.scripts.run_phase_g_integration import (
    DEFAULT_ENV,
    TARGET_PAPER_ID,
    run_dry_run_analysis,
)


@pytest.mark.asyncio
async def test_dry_run_makes_zero_writes() -> None:
    """Verify that run_dry_run_analysis queries read-only and never calls write/execute."""
    env_vars = parse_env_file_isolated(DEFAULT_ENV)

    # Patch asyncpg connection execute / executemany to ensure no writes happen
    with patch("asyncpg.connect") as mock_pg_connect:
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {"id": "be7ff9bf-3730-57d7-a2d6-fc30758eb313", "chunk_index": 0},
            {"id": "d55eb84a-1e7d-5d8b-9f4f-192c731940ee", "chunk_index": 1},
            {"id": "16344bbf-6d04-5f0e-8c48-06166d96a98a", "chunk_index": 2},
            {"id": "bfee6412-060c-5026-aa8c-ff369d74bb7a", "chunk_index": 3},
        ]
        mock_pg_connect.return_value = mock_conn

        # Patch GraphDatabase to inspect Cypher queries and ensure no CREATE / MERGE writes
        with patch(
            "experiments.scipaper.scripts.run_phase_g_integration.GraphDatabase.driver"
        ) as mock_driver:
            mock_session = MagicMock()
            mock_driver.return_value.session.return_value.__enter__.return_value = mock_session
            mock_session.run.return_value.single.return_value = {"cnt": 1}

            manifest = await run_dry_run_analysis(env_vars, target_doc_id=TARGET_PAPER_ID)

            # Assert PG execute was never called
            assert not mock_conn.execute.called
            assert not mock_conn.executemany.called

            # Inspect Neo4j run calls to ensure all queries are MATCH ... RETURN (no MERGE or CREATE)
            for call in mock_session.run.call_args_list:
                query = call[0][0].strip().upper()
                assert query.startswith("MATCH")
                assert "MERGE" not in query
                assert "CREATE" not in query
                assert "DELETE" not in query
                assert "SET" not in query

            assert manifest.document_id == TARGET_PAPER_ID
            assert manifest.postgres_chunks_verified is True
            assert len(manifest.chunk_ids) == 4
