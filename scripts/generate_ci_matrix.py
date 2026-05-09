#!/usr/bin/env python3
"""Generate a hardware-aware CI matrix for Hugging Face style libraries.

Reads backend and suite definitions from ci/*.json and emits a JSON matrix
consumable by GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
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
    caps = set(backend.get("capabilities", []))
    requires_all = set(suite.get("requires_all", []))
    requires_any = set(suite.get("requires_any", []))

    if requires_all and not requires_all.issubset(caps):
        return False
    if requires_any and caps.isdisjoint(requires_any):
        return False
    return True


def generate_matrix(backends_cfg: dict[str, Any], suites_cfg: dict[str, Any]) -> dict[str, Any]:
    include: list[dict[str, Any]] = []
    for backend_name, backend in backends_cfg.get("backends", {}).items():
        for suite_name, suite in suites_cfg.get("suites", {}).items():
            if has_required_capabilities(backend, suite):
                include.append(
                    {
                        "backend": backend_name,
                        "suite": suite_name,
                        "package": suite.get("package", "common"),
                        "command": suite["command"],
                        "runs_on": backend.get("labels", ["ubuntu-latest"]),
                        "python": backend.get("setup", {}).get("python", "3.11"),
                        "extras": backend.get("setup", {}).get("extras", []),
                        "torch_device": backend.get("torch_device", "cpu"),
                    }
                )
    return {"include": include}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default=str(ROOT / "ci" / "backends.json"))
    parser.add_argument("--suites", default=str(ROOT / "ci" / "test_suites.json"))
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    backends_cfg = load_json(Path(args.backends))
    suites_cfg = load_json(Path(args.suites))
    matrix = generate_matrix(backends_cfg, suites_cfg)

    if args.pretty:
        print(json.dumps(matrix, indent=2))
    else:
        print(json.dumps(matrix))


if __name__ == "__main__":
    main()
