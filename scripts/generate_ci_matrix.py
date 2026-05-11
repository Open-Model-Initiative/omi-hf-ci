#!/usr/bin/env python3
"""Generate a hardware-aware CI matrix for Hugging Face style libraries.

Reads backend and suite definitions from ci/*.json and emits a JSON matrix
consumable by GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}")
    return data


def has_required_capabilities(backend: dict[str, Any], suite: dict[str, Any]) -> bool:
    """Check if a backend satisfies a suite's capability requirements."""
    caps = set(backend.get("capabilities", []))
    requires_all = set(suite.get("requires_all", []))
    requires_any = set(suite.get("requires_any", []))

    if requires_all and not requires_all.issubset(caps):
        return False
    if requires_any and caps.isdisjoint(requires_any):
        return False
    return True


def _build_env_list(env_dict: dict[str, str]) -> list[str]:
    """Convert env dict to KEY=VALUE list for GitHub Actions env block."""
    return [f"{k}={v}" for k, v in env_dict.items()]


def generate_matrix(
    backends_cfg: dict[str, Any],
    suites_cfg: dict[str, Any],
    backend_filter: set[str] | None = None,
    suite_filter: set[str] | None = None,
) -> dict[str, Any]:
    """Generate a CI matrix by matching suites to compatible backends.

    Args:
        backends_cfg: Parsed backends.json
        suites_cfg: Parsed test_suites.json
        backend_filter: If set, only include these backend names
        suite_filter: If set, only include these suite names
    """
    include: list[dict[str, Any]] = []

    for backend_name, backend in backends_cfg.get("backends", {}).items():
        if backend_filter and backend_name not in backend_filter:
            continue

        for suite_name, suite in suites_cfg.get("suites", {}).items():
            if suite_filter and suite_name not in suite_filter:
                continue

            if has_required_capabilities(backend, suite):
                setup = backend.get("setup", {})
                env_dict = backend.get("env", {})

                include.append(
                    {
                        "backend": backend_name,
                        "suite": suite_name,
                        "package": suite.get("package", "common"),
                        "command": suite["command"],
                        "runs_on": backend.get("labels", ["ubuntu-latest"]),
                        "python": setup.get("python", "3.11"),
                        "extras": setup.get("extras", []),
                        "install_cmd": setup.get("install_cmd", ""),
                        "torch_device": backend.get("torch_device", "cpu"),
                        "env": _build_env_list(env_dict),
                        "timeout_minutes": suite.get("timeout_minutes", 60),
                        "tags": suite.get("tags", []),
                    }
                )

    return {"include": include}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a hardware-aware CI matrix"
    )
    parser.add_argument(
        "--backends",
        default=str(ROOT / "ci" / "backends.json"),
        help="Path to backends config",
    )
    parser.add_argument(
        "--suites",
        default=str(ROOT / "ci" / "test_suites.json"),
        help="Path to suites config",
    )
    parser.add_argument(
        "--backend-filter",
        default=None,
        help="Comma-separated backend names to include",
    )
    parser.add_argument(
        "--suite-filter",
        default=None,
        help="Comma-separated suite names to include",
    )
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    backends_cfg = load_json(Path(args.backends))
    suites_cfg = load_json(Path(args.suites))

    backend_filter = set(args.backend_filter.split(",")) if args.backend_filter else None
    suite_filter = set(args.suite_filter.split(",")) if args.suite_filter else None

    matrix = generate_matrix(
        backends_cfg, suites_cfg,
        backend_filter=backend_filter,
        suite_filter=suite_filter,
    )

    if args.pretty:
        print(json.dumps(matrix, indent=2))
    else:
        print(json.dumps(matrix))


if __name__ == "__main__":
    main()
