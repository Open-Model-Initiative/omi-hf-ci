"""Tests for the config validation script."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_validate(config: str, stdin_data: str | None = None, input_path: str | None = None):
    cmd = [sys.executable, "scripts/validate_config.py", "--config", config]
    if input_path:
        cmd += ["--input", input_path]
    elif stdin_data is not None:
        cmd += ["--input", "-"]
    return subprocess.run(
        cmd,
        cwd=ROOT,
        input=stdin_data,
        capture_output=True,
        text=True,
    )


class TestValidateBackends:
    def test_accepts_valid_default_config(self):
        proc = _run_validate("backends")
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout

    def test_rejects_missing_required_fields(self):
        bad = {"backends": {"broken": {"type": "gpu"}}}
        proc = _run_validate("backends", stdin_data=json.dumps(bad))
        assert proc.returncode != 0
        assert "ERROR" in proc.stderr

    def test_rejects_invalid_type_enum(self):
        bad = {
            "backends": {
                "bad": {
                    "type": "tpu",
                    "labels": ["x"],
                    "torch_device": "tpu",
                    "capabilities": [],
                }
            }
        }
        proc = _run_validate("backends", stdin_data=json.dumps(bad))
        assert proc.returncode != 0

    def test_rejects_empty_labels(self):
        bad = {
            "backends": {
                "bad": {
                    "type": "gpu",
                    "labels": [],
                    "torch_device": "cuda",
                    "capabilities": [],
                }
            }
        }
        proc = _run_validate("backends", stdin_data=json.dumps(bad))
        assert proc.returncode != 0


class TestValidateSuites:
    def test_accepts_valid_default_config(self):
        proc = _run_validate("suites")
        assert proc.returncode == 0, proc.stderr
        assert "OK" in proc.stdout

    def test_rejects_missing_package(self):
        bad = {"suites": {"bad": {"command": "pytest"}}}
        proc = _run_validate("suites", stdin_data=json.dumps(bad))
        assert proc.returncode != 0

    def test_rejects_missing_command(self):
        bad = {"suites": {"bad": {"package": "transformers"}}}
        proc = _run_validate("suites", stdin_data=json.dumps(bad))
        assert proc.returncode != 0

    def test_rejects_invalid_timeout(self):
        bad = {
            "suites": {
                "bad": {
                    "package": "transformers",
                    "command": "pytest",
                    "timeout_minutes": 0,
                }
            }
        }
        proc = _run_validate("suites", stdin_data=json.dumps(bad))
        assert proc.returncode != 0
