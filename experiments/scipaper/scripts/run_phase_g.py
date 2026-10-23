"""CLI runner for Phase G Dry Run and Integration."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
from pathlib import Path

from experiments.scipaper.scripts.preflight_check import (
    parse_env_file_isolated,
    validate_environment_dict,
)
from experiments.scipaper.scripts.run_phase_g_integration import (
    DEFAULT_ENV,
    DEFAULT_RESULTS_DIR,
    TARGET_PAPER_ID,
    execute_single_paper_integration,
    run_dry_run_analysis,
)

logger = logging.getLogger("phase_g_cli")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Phase G Dry Run and Single-Paper Integration")
    parser.add_argument("--execute", action="store_true", help="Execute single paper integration")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV, help="Path to .env.scipaper")
    parser.add_argument("--target-id", type=str, default=TARGET_PAPER_ID, help="Target document ID")
    args = parser.parse_args()

    env_vars = parse_env_file_isolated(args.env_file)
    validate_environment_dict(env_vars)

    # Always run dry run first
    manifest = await run_dry_run_analysis(env_vars, target_doc_id=args.target_id)
    out_dir = DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "single_paper_dry_run_manifest.json"
    manifest_path.write_text(json.dumps(dataclasses.asdict(manifest), indent=2), encoding="utf-8")

    print("=" * 60)
    print("GATE B: DRY RUN COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Target Paper: {manifest.document_id} ('{manifest.title}')")
    print(
        f"PostgreSQL Chunks Verified: {manifest.postgres_chunks_verified} ({len(manifest.chunk_ids)} chunks)"
    )
    print(f"Entities Extracted (Unique): {manifest.unique_entities_count}")
    print(f"  Existing in Neo4j: {manifest.existing_nodes_in_neo4j}")
    print(f"  Proposed New Nodes: {manifest.proposed_new_nodes}")
    print(f"Relationships Extracted (Unique across chunks): {manifest.unique_relationships_count}")
    print(f"  Existing in Neo4j: {manifest.existing_rels_in_neo4j}")
    print(f"  Proposed New Relationships: {manifest.proposed_new_rels}")
    print(f"Manifest saved to: {manifest_path}")

    if not args.execute:
        print("\nDry-run mode only. Zero database writes performed.")
        return

    print("\n" + "=" * 60)
    print("GATE C: EXECUTING SINGLE PAPER INTEGRATION")
    print("=" * 60)
    res = await execute_single_paper_integration(env_vars, target_doc_id=args.target_id)
    res_path = out_dir / "single_paper_integration_verification.json"
    res_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("Integration complete and verified.")
    print(f"PostgreSQL entity_links added: {res['pg_elinks_added']}")
    print(f"Neo4j post-integration total nodes: {res['post_integration_neo4j']['total_nodes']}")
    print(
        f"Neo4j post-integration total relationships: {res['post_integration_neo4j']['total_relationships']}"
    )
    print(f"Verification saved to: {res_path}")


if __name__ == "__main__":
    asyncio.run(main())
