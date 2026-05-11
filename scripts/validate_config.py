#!/usr/bin/env python3
"""Validate CI config files against their JSON schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


SCHEMAS = {
    "backends": ROOT / "ci" / "backends.schema.json",
    "suites": ROOT / "ci" / "suites.schema.json",
}


def _fallback_validate_backends(data: dict) -> list[str]:
    """Minimal validation when jsonschema is not installed."""
    errors = []
    if "backends" not in data:
        return ["Missing required key 'backends'"]
    for name, backend in data["backends"].items():
        for field in ("type", "labels", "torch_device", "capabilities"):
            if field not in backend:
                errors.append(f"backends.{name}: missing required field '{field}'")
    return errors


def _fallback_validate_suites(data: dict) -> list[str]:
    """Minimal validation when jsonschema is not installed."""
    errors = []
    if "suites" not in data:
        return ["Missing required key 'suites'"]
    for name, suite in data["suites"].items():
        for field in ("package", "command"):
            if field not in suite:
                errors.append(f"suites.{name}: missing required field '{field}'")
    return errors


def validate(data: dict, schema_path: Path, config_type: str) -> list[str]:
    """Validate data against a JSON schema. Returns list of error strings."""
    if HAS_JSONSCHEMA:
        schema = json.loads(schema_path.read_text())
        validator = jsonschema.Draft7Validator(schema)
        return [str(e.message) for e in validator.iter_errors(data)]
    # Fallback validation
    if config_type == "backends":
        return _fallback_validate_backends(data)
    return _fallback_validate_suites(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CI config files")
    parser.add_argument(
        "--config",
        required=True,
        choices=["backends", "suites"],
        help="Which config to validate",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to config file, or '-' for stdin. Defaults to ci/<config>.json",
    )
    args = parser.parse_args()

    if args.input == "-":
        data = json.load(sys.stdin)
    elif args.input:
        data = json.loads(Path(args.input).read_text())
    else:
        filename = "backends.json" if args.config == "backends" else "test_suites.json"
        default_path = ROOT / "ci" / filename
        data = json.loads(default_path.read_text())

    errors = validate(data, SCHEMAS[args.config], args.config)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {args.config} config is valid")


if __name__ == "__main__":
    main()
