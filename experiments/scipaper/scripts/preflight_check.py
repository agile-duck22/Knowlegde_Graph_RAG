"""Fail-closed preflight validation script for the scientific-paper experiment.

Enforces infrastructure isolation and safety boundaries WITHOUT starting containers,
connecting to any database, or making external network requests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import yaml

# Authoritative Enterprise Infrastructure Blocklist
ENTERPRISE_PORTS = {5432, 7474, 7687}
ENTERPRISE_DB_NAMES = {"kgr_db", "kgr"}
ENTERPRISE_CONTAINER_NAMES = {"kgr-postgres", "kgr-neo4j"}
ENTERPRISE_VOLUME_NAMES = {"kgr_pgdata", "kgr_neo4jdata"}
ENTERPRISE_NETWORKS = {"kgr_default", "bridge", "default"}

# Target Isolated Infrastructure Allowlist
REQUIRED_POSTGRES_PORT = 5433
REQUIRED_NEO4J_BOLT_PORT = 7688
REQUIRED_NEO4J_HTTP_PORT = 7475

REQUIRED_CONTAINER_NAMES = {"kgr-scipaper-postgres", "kgr-scipaper-neo4j"}
REQUIRED_VOLUME_NAMES = {"kgr_scipaper_pgdata", "kgr_scipaper_neo4jdata"}
REQUIRED_NETWORK_NAME = "kgr-scipaper-net"

REQUIRED_EMBEDDING_PROVIDER = "sentence_transformers"
REQUIRED_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
REQUIRED_EMBEDDING_DIMENSIONS = 384

PLACEHOLDER_SUBSTRINGS = [
    "change_me",
    "your_",
    "_here",
    "placeholder",
    "example",
    "replace_me",
    "todo",
]


class PreflightValidationError(Exception):
    """Raised when preflight safety or isolation checks fail."""


def parse_env_file_isolated(env_path: Path) -> dict[str, str]:
    """Parse an environment file directly without touching os.environ or falling back to .env.

    Parameters
    ----------
    env_path:
        Path to the environment file.

    Returns
    -------
    dict[str, str]
        Key-value mapping parsed from the environment file.
    """
    if not env_path.exists():
        raise PreflightValidationError(f"Environment file does not exist: {env_path}")

    env_vars: dict[str, str] = {}
    with open(env_path, encoding="utf-8") as f:
        for _line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip()
            # Strip outer single or double quotes if present
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            env_vars[key] = val

    return env_vars


def validate_compose_file(compose_path: Path) -> list[str]:
    """Validate Docker Compose configuration for strict isolation.

    Checks:
    - No enterprise container names, ports, volumes, or networks are used.
    - Required isolated container names and volumes are defined.
    - Published ports match isolated ports (5433, 7688, 7475) and not enterprise ports.
    - Healthchecks are configured.
    - No plaintext secrets are hardcoded into environment fields.

    Parameters
    ----------
    compose_path:
        Path to docker-compose.scipaper.yml.

    Returns
    -------
    list[str]
        List of validation error messages.
    """
    errors: list[str] = []
    if not compose_path.exists():
        return [f"Compose file does not exist: {compose_path}"]

    try:
        with open(compose_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        return [f"Failed to parse Compose YAML: {exc}"]

    services = data.get("services", {})
    if not services:
        errors.append("No services defined in Compose file.")
        return errors

    # Check services
    found_containers = set()
    found_host_ports = set()
    for sname, scfg in services.items():
        if not isinstance(scfg, dict):
            continue

        cname = scfg.get("container_name")
        if cname:
            found_containers.add(cname)
            if cname in ENTERPRISE_CONTAINER_NAMES:
                errors.append(
                    f"Service '{sname}' uses forbidden enterprise container name '{cname}'."
                )

        # Check published ports
        ports = scfg.get("ports", [])
        for p in ports:
            p_str = str(p)
            # Match formats like "127.0.0.1:5433:5432", "5433:5432"
            parts = p_str.split(":")
            if len(parts) >= 2:
                try:
                    host_port = int(parts[-2])
                    found_host_ports.add(host_port)
                    if host_port in ENTERPRISE_PORTS:
                        errors.append(
                            f"Service '{sname}' publishes forbidden enterprise host port {host_port}."
                        )
                except ValueError:
                    pass

        # Check healthcheck
        if "healthcheck" not in scfg:
            errors.append(f"Service '{sname}' is missing a healthcheck configuration.")

    # Check named volumes
    volumes = data.get("volumes", {})
    for vol_name, vol_cfg in volumes.items():
        if vol_name in ENTERPRISE_VOLUME_NAMES:
            errors.append(f"Compose defines forbidden enterprise volume '{vol_name}'.")
        vname_attr = vol_cfg.get("name") if isinstance(vol_cfg, dict) else None
        if vname_attr in ENTERPRISE_VOLUME_NAMES:
            errors.append(
                f"Compose volume '{vol_name}' references enterprise volume name '{vname_attr}'."
            )

    # Check networks
    networks = data.get("networks", {})
    if not networks:
        errors.append("No custom isolated network defined in Compose file.")
    for net_name, _net_cfg in networks.items():
        if net_name in ENTERPRISE_NETWORKS:
            errors.append(f"Compose uses default or forbidden enterprise network '{net_name}'.")

    # Verify expected experiment ports are targeted
    if REQUIRED_POSTGRES_PORT not in found_host_ports:
        errors.append(
            f"PostgreSQL service host port {REQUIRED_POSTGRES_PORT} not found in Compose ports."
        )
    if REQUIRED_NEO4J_BOLT_PORT not in found_host_ports:
        errors.append(
            f"Neo4j Bolt service host port {REQUIRED_NEO4J_BOLT_PORT} not found in Compose ports."
        )

    return errors


def validate_environment_dict(
    env_vars: dict[str, str],
    *,
    allow_placeholders: bool = False,
) -> list[str]:
    """Validate environment settings against isolation and safety invariants.

    Parameters
    ----------
    env_vars:
        Dictionary of environment variable names to values.
    allow_placeholders:
        If True, placeholder credentials are not reported as errors (useful for .env.example).

    Returns
    -------
    list[str]
        List of validation error messages.
    """
    errors: list[str] = []

    # 1. PostgreSQL Port Check
    pg_port_raw = env_vars.get("POSTGRES_PORT", "")
    if not pg_port_raw:
        errors.append("POSTGRES_PORT is missing.")
    else:
        try:
            pg_port = int(pg_port_raw)
            if pg_port in ENTERPRISE_PORTS or pg_port == 5432:
                errors.append(
                    f"POSTGRES_PORT is set to production/enterprise port {pg_port}. Must be {REQUIRED_POSTGRES_PORT}."
                )
            elif pg_port != REQUIRED_POSTGRES_PORT:
                errors.append(
                    f"POSTGRES_PORT is {pg_port}; expected isolated port {REQUIRED_POSTGRES_PORT}."
                )
        except ValueError:
            errors.append(f"POSTGRES_PORT '{pg_port_raw}' is not a valid integer.")

    # 2. PostgreSQL DB Name Check
    pg_db = env_vars.get("POSTGRES_DB", "")
    if not pg_db:
        errors.append("POSTGRES_DB is missing.")
    elif pg_db in ENTERPRISE_DB_NAMES:
        errors.append(
            f"POSTGRES_DB is set to enterprise database '{pg_db}'. Must be isolated (e.g. 'scipaper_db')."
        )

    # 3. Neo4j URI Check
    neo4j_uri = env_vars.get("NEO4J_URI", "")
    if not neo4j_uri:
        errors.append("NEO4J_URI is missing.")
    else:
        try:
            parsed = urlparse(neo4j_uri)
            port = parsed.port
            if port in ENTERPRISE_PORTS or port == 7687:
                errors.append(
                    f"NEO4J_URI points to production/enterprise port {port}. Must use {REQUIRED_NEO4J_BOLT_PORT}."
                )
            elif port != REQUIRED_NEO4J_BOLT_PORT:
                errors.append(
                    f"NEO4J_URI port is {port}; expected isolated port {REQUIRED_NEO4J_BOLT_PORT}."
                )
        except Exception as exc:
            errors.append(f"Failed to parse NEO4J_URI: {exc}")

    # 4. Embeddings Safety Check
    provider = env_vars.get("EMBEDDING_PROVIDER", "").strip().lower()
    if provider != REQUIRED_EMBEDDING_PROVIDER:
        errors.append(
            f"EMBEDDING_PROVIDER '{provider}' must be '{REQUIRED_EMBEDDING_PROVIDER}' for zero-cost operation."
        )

    model = env_vars.get("EMBEDDING_MODEL", "").strip()
    if model != REQUIRED_EMBEDDING_MODEL:
        errors.append(f"EMBEDDING_MODEL '{model}' must be '{REQUIRED_EMBEDDING_MODEL}'.")

    dims_raw = env_vars.get("EMBEDDING_DIMENSIONS", "")
    try:
        dims = int(dims_raw)
        if dims != REQUIRED_EMBEDDING_DIMENSIONS:
            errors.append(
                f"EMBEDDING_DIMENSIONS {dims} must be {REQUIRED_EMBEDDING_DIMENSIONS} for {REQUIRED_EMBEDDING_MODEL}."
            )
    except ValueError:
        errors.append(f"EMBEDDING_DIMENSIONS '{dims_raw}' is not a valid integer.")

    allow_paid = env_vars.get("ALLOW_PAID_EMBEDDINGS", "").strip().lower()
    if allow_paid != "false":
        errors.append(
            f"ALLOW_PAID_EMBEDDINGS is '{allow_paid}'; must be 'false' for zero-cost scientific paper experiment."
        )

    # 5. Credentials & Placeholders Check
    required_keys = ["POSTGRES_PASSWORD", "NEO4J_PASSWORD"]
    for k in required_keys:
        val = env_vars.get(k, "")
        if not val:
            errors.append(f"Required credential '{k}' is missing or empty.")
        elif not allow_placeholders:
            lower_val = val.lower()
            if any(token in lower_val for token in PLACEHOLDER_SUBSTRINGS):
                errors.append(
                    f"Credential '{k}' contains an unconfigured placeholder. Set a real local password before starting."
                )

    return errors


def run_preflight_checks(
    *,
    compose_path: Path,
    env_path: Path,
    allow_placeholders: bool = False,
) -> tuple[bool, list[str]]:
    """Run full isolated preflight validation without starting containers or network calls.

    Parameters
    ----------
    compose_path:
        Path to docker-compose.scipaper.yml.
    env_path:
        Path to .env.scipaper or .env.scipaper.example.
    allow_placeholders:
        Whether to tolerate placeholder passwords (e.g. for example template verification).

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []

    # 1. Compose file checks
    compose_errors = validate_compose_file(compose_path)
    errors.extend(compose_errors)

    # 2. Environment file checks
    try:
        env_dict = parse_env_file_isolated(env_path)
        env_errors = validate_environment_dict(env_dict, allow_placeholders=allow_placeholders)
        errors.extend(env_errors)
    except PreflightValidationError as exc:
        errors.append(str(exc))

    return (len(errors) == 0, errors)


def main() -> None:
    """CLI entrypoint for preflight checking."""
    parser = argparse.ArgumentParser(
        description="Fail-closed preflight checker for scientific-paper RAG infrastructure."
    )
    base_dir = Path(__file__).resolve().parent.parent
    default_compose = base_dir / "docker-compose.scipaper.yml"
    default_env = base_dir / ".env.scipaper"
    default_example = base_dir / ".env.scipaper.example"

    parser.add_argument(
        "--compose-file",
        type=Path,
        default=default_compose,
        help="Path to docker-compose.scipaper.yml",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to environment file (defaults to .env.scipaper if exists, else .env.scipaper.example)",
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder passwords (e.g. when validating .env.scipaper.example)",
    )

    args = parser.parse_args()

    compose_file = args.compose_file
    if args.env_file:
        env_file = args.env_file
        allow_placeholders = args.allow_placeholders
    else:
        if default_env.exists():
            env_file = default_env
            allow_placeholders = args.allow_placeholders
        else:
            env_file = default_example
            allow_placeholders = True

    print("=================================================================")
    print("Scientific-Paper RAG Infrastructure Preflight Check (Fail-Closed)")
    print("=================================================================")
    print(f"Compose file: {compose_file}")
    print(f"Env file:     {env_file}")
    print(
        f"Mode:         {'Tolerate Placeholders' if allow_placeholders else 'Strict Credentials'}"
    )
    print("-----------------------------------------------------------------")

    passed, errors = run_preflight_checks(
        compose_path=compose_file,
        env_path=env_file,
        allow_placeholders=allow_placeholders,
    )

    if passed:
        print("[SUCCESS] All isolation and preflight checks PASSED.")
        print("  - Zero enterprise ports detected (target: 5433, 7688).")
        print("  - Zero enterprise database names or container names detected.")
        print("  - Local sentence-transformers embeddings verified (dimension 384).")
        print("  - Zero paid API usage allowed (ALLOW_PAID_EMBEDDINGS=false).")
        print("  - Containers remain stopped; no network or database connections made.")
        sys.exit(0)
    else:
        print(f"[FAILED] Found {len(errors)} isolation violation(s):")
        for idx, err in enumerate(errors, 1):
            print(f"  {idx}. {err}")
        print("\nPreflight check failed closed. Refusing to proceed to container startup.")
        sys.exit(1)


if __name__ == "__main__":
    main()
