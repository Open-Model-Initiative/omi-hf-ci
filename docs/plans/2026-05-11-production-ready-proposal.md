# Production-Ready Hardware-Agnostic CI Proposal

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Transform the initial draft into a production-ready, proposal-grade hardware-agnostic CI system for HuggingFace libraries that can be presented to the HF team.

**Architecture:** Capability-driven CI that decouples test intent from hardware vendor. Backends declare capabilities (bf16, flash_attention_2, etc.), suites declare requirements, a matrix generator matches them, and GitHub Actions orchestrates execution on self-hosted runners.

**Tech Stack:** Python 3.11+, GitHub Actions, JSON schema validation, pytest

---

## Current State Assessment

What exists:
- Basic backend catalog (4 vendors: nvidia, amd, gaudi, ascend)
- Basic suite catalog (3 suites, all pointing to placeholder tests)
- Matrix generator script
- GitHub Actions workflow skeleton
- Minimal tests

What's missing for production-ready proposal:
- Real test suite commands (currently all point to matrix generator tests)
- Backend-specific SDK installation logic
- JSON schema validation
- Timeouts, caching, artifact collection
- Nightly vs PR differentiation
- Result aggregation and reporting
- Comprehensive documentation for HF team
- Contributing guide for vendor teams
- Multi-GPU distributed test support

---

## Phase 1: Foundation Hardening

### Task 1: Add JSON Schema Validation for Configs

**Objective:** Validate backends.json and test_suites.json at generation time to catch config errors early.

**Files:**
- Create: `scripts/validate_config.py`
- Create: `ci/backends.schema.json`
- Create: `ci/suites.schema.json`
- Modify: `scripts/generate_ci_matrix.py`
- Create: `tests/test_validate_config.py`

**Step 1: Write failing test for schema validation**

```python
# tests/test_validate_config.py
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_validate_accepts_valid_backends():
    proc = subprocess.run(
        ["python", "scripts/validate_config.py", "--config", "backends"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_validate_accepts_valid_suites():
    proc = subprocess.run(
        ["python", "scripts/validate_config.py", "--config", "suites"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_validate_rejects_missing_required_field():
    bad = {"backends": {"broken": {"type": "gpu"}}}  # missing labels, capabilities
    proc = subprocess.run(
        ["python", "scripts/validate_config.py", "--config", "backends", "--input", "-"],
        cwd=ROOT,
        input=json.dumps(bad),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
```

**Step 2: Run tests to verify failure**

Run: `cd /Users/zhipeng/Workspace/omi-hf-ci && python -m pytest tests/test_validate_config.py -v`
Expected: FAIL — validate_config.py doesn't exist

**Step 3: Create JSON schemas**

```json
// ci/backends.schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["backends"],
  "properties": {
    "backends": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["type", "labels", "torch_device", "capabilities"],
        "properties": {
          "type": {"type": "string", "enum": ["gpu", "accelerator", "cpu"]},
          "labels": {"type": "array", "items": {"type": "string"}, "minItems": 1},
          "torch_device": {"type": "string"},
          "setup": {
            "type": "object",
            "properties": {
              "python": {"type": "string"},
              "extras": {"type": "array", "items": {"type": "string"}}
            }
          },
          "capabilities": {"type": "array", "items": {"type": "string"}},
          "env": {"type": "object", "additionalProperties": {"type": "string"}}
        },
        "additionalProperties": false
      }
    }
  }
}
```

```json
// ci/suites.schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["suites"],
  "properties": {
    "suites": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "required": ["package", "command"],
        "properties": {
          "package": {"type": "string"},
          "command": {"type": "string"},
          "requires_any": {"type": "array", "items": {"type": "string"}},
          "requires_all": {"type": "array", "items": {"type": "string"}},
          "timeout_minutes": {"type": "integer", "minimum": 1},
          "multi_gpu": {"type": "boolean"},
          "tags": {"type": "array", "items": {"type": "string"}}
        },
        "additionalProperties": false
      }
    }
  }
}
```

**Step 4: Implement validate_config.py**

```python
#!/usr/bin/env python3
"""Validate CI config files against their JSON schemas."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import jsonschema
except ImportError:
    jsonschema = None


SCHEMAS = {
    "backends": ROOT / "ci" / "backends.schema.json",
    "suites": ROOT / "ci" / "suites.schema.json",
}


def validate(data: dict, schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text())
    if jsonschema:
        validator = jsonschema.Draft7Validator(schema)
        return [str(e) for e in validator.iter_errors(data)]
    # Fallback: basic required-field check
    errors = []
    required = schema.get("properties", {}).get("backends", {}).get("additionalProperties", {}).get("required", [])
    if "backends" in data:
        for name, backend in data["backends"].items():
            for field in required:
                if field not in backend:
                    errors.append(f"backends.{name}: missing required field '{field}'")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=["backends", "suites"])
    parser.add_argument("--input", default=None, help="File path or '-' for stdin")
    args = parser.parse_args()

    if args.input == "-":
        data = json.load(sys.stdin)
    elif args.input:
        data = json.loads(Path(args.input).read_text())
    else:
        default_path = ROOT / "ci" / ("backends.json" if args.config == "backends" else "test_suites.json")
        data = json.loads(default_path.read_text())

    errors = validate(data, SCHEMAS[args.config])
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {args.config} config is valid")


if __name__ == "__main__":
    main()
```

**Step 5: Run tests to verify pass**

Run: `cd /Users/zhipeng/Workspace/omi-hf-ci && python -m pytest tests/test_validate_config.py -v`
Expected: PASS

**Step 6: Integrate validation into matrix generator**

Add a `--validate` flag to generate_ci_matrix.py that runs schema validation before generating the matrix.

**Step 7: Commit**

```bash
git add ci/backends.schema.json ci/suites.schema.json scripts/validate_config.py tests/test_validate_config.py
git commit -m "feat: add JSON schema validation for CI configs"
```

---

### Task 2: Enrich Backend Definitions with Env Vars and Setup Hooks

**Objective:** Each backend needs vendor-specific environment variables and setup commands for real-world CI.

**Files:**
- Modify: `ci/backends.json`
- Modify: `scripts/generate_ci_matrix.py`

**Step 1: Update backends.json with env vars and setup commands**

```json
{
  "backends": {
    "nvidia": {
      "type": "gpu",
      "labels": ["self-hosted", "linux", "x64", "nvidia"],
      "torch_device": "cuda",
      "setup": {
        "python": "3.11",
        "extras": ["cuda"],
        "install_cmd": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
      },
      "env": {
        "NVIDIA_VISIBLE_DEVICES": "all",
        "CUDA_VISIBLE_DEVICES": "0"
      },
      "capabilities": ["fp16", "bf16", "flash_attention_2", "bitsandbytes", "torch_compile"]
    },
    "amd": {
      "type": "gpu",
      "labels": ["self-hosted", "linux", "x64", "amd"],
      "torch_device": "cuda",
      "setup": {
        "python": "3.11",
        "extras": ["rocm"],
        "install_cmd": "pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0"
      },
      "env": {
        "HIP_VISIBLE_DEVICES": "0",
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"
      },
      "capabilities": ["fp16", "bf16"]
    },
    "gaudi": {
      "type": "accelerator",
      "labels": ["self-hosted", "linux", "x64", "gaudi"],
      "torch_device": "hpu",
      "setup": {
        "python": "3.11",
        "extras": ["gaudi"],
        "install_cmd": "pip install optimum-habana"
      },
      "env": {
        "PT_HPU_LAZY_MODE": "0",
        "HABANA_VISIBLE_DEVICES": "all"
      },
      "capabilities": ["bf16", "hpu_graph"]
    },
    "ascend": {
      "type": "accelerator",
      "labels": ["self-hosted", "linux", "x64", "ascend"],
      "torch_device": "npu",
      "setup": {
        "python": "3.10",
        "extras": ["ascend"],
        "install_cmd": "pip install torch_npu"
      },
      "env": {
        "ASCEND_VISIBLE_DEVICES": "0"
      },
      "capabilities": ["fp16", "bf16"]
    }
  }
}
```

**Step 2: Update generate_ci_matrix.py to emit env and install_cmd**

Add `env` and `install_cmd` fields to the matrix output.

**Step 3: Update tests**

Add assertions for the new fields in `tests/test_generate_ci_matrix.py`.

**Step 4: Commit**

```bash
git add ci/backends.json scripts/generate_ci_matrix.py tests/test_generate_ci_matrix.py
git commit -m "feat: add vendor-specific env vars and install commands to backends"
```

---

### Task 3: Define Real Test Suites

**Objective:** Replace placeholder test commands with realistic HF library test scenarios.

**Files:**
- Modify: `ci/test_suites.json`

**Step 1: Update test_suites.json with realistic suites**

```json
{
  "suites": {
    "transformers_smoke": {
      "package": "transformers",
      "command": "python -m pytest tests/test_modeling_common.py -x -q --timeout=300",
      "requires_any": ["fp16", "bf16"],
      "timeout_minutes": 30,
      "tags": ["smoke", "core"]
    },
    "transformers_training": {
      "package": "transformers",
      "command": "python -m pytest tests/trainer/ -x -q --timeout=600 -k 'not slow'",
      "requires_any": ["bf16"],
      "requires_all": [],
      "timeout_minutes": 60,
      "tags": ["training"]
    },
    "diffusers_smoke": {
      "package": "diffusers",
      "command": "python -m pytest tests/test_models_unet_2d.py -x -q --timeout=300",
      "requires_any": ["fp16", "bf16"],
      "timeout_minutes": 30,
      "tags": ["smoke", "core"]
    },
    "diffusers_pipelines": {
      "package": "diffusers",
      "command": "python -m pytest tests/test_pipelines.py -x -q --timeout=600 -k 'not slow'",
      "requires_any": ["bf16"],
      "timeout_minutes": 60,
      "tags": ["pipelines"]
    },
    "distributed_sanity": {
      "package": "common",
      "command": "python -m pytest tests/distributed/ -x -q --timeout=300",
      "requires_all": [],
      "multi_gpu": true,
      "timeout_minutes": 20,
      "tags": ["distributed"]
    },
    "quantization_bitsandbytes": {
      "package": "transformers",
      "command": "python -m pytest tests/quantization/ -x -q --timeout=300 -k bnb",
      "requires_any": ["bitsandbytes"],
      "timeout_minutes": 30,
      "tags": ["quantization"]
    }
  }
}
```

**Step 2: Run matrix generator to verify**

Run: `python scripts/generate_ci_matrix.py --pretty`
Expected: Matrix includes all 6 suites across compatible backends.

**Step 3: Commit**

```bash
git add ci/test_suites.json
git commit -m "feat: define realistic HF test suites with timeouts and tags"
```

---

## Phase 2: Workflow Productionization

### Task 4: Rewrite GitHub Actions Workflow with Production Features

**Objective:** Add timeouts, caching, artifact collection, env injection, and failure handling.

**Files:**
- Modify: `.github/workflows/hf-hardware-agnostic-ci.yml`

**Step 1: Rewrite the workflow**

```yaml
name: HF Hardware Agnostic CI

on:
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      suite_filter:
        description: "Comma-separated suite names to run (empty = all)"
        required: false
        default: ""
      backend_filter:
        description: "Comma-separated backend names to run (empty = all)"
        required: false
        default: ""

concurrency:
  group: "hf-ci-${{ github.ref }}"
  cancel-in-progress: true

jobs:
  matrix:
    name: Build matrix
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.mk.outputs.matrix }}
      suite_count: ${{ steps.mk.outputs.suite_count }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install validation deps
        run: pip install jsonschema
      - name: Validate configs
        run: |
          python scripts/validate_config.py --config backends
          python scripts/validate_config.py --config suites
      - id: mk
        run: |
          ARGS=""
          if [ -n "${{ github.event.inputs.suite_filter }}" ]; then
            ARGS="$ARGS --suite-filter ${{ github.event.inputs.suite_filter }}"
          fi
          if [ -n "${{ github.event.inputs.backend_filter }}" ]; then
            ARGS="$ARGS --backend-filter ${{ github.event.inputs.backend_filter }}"
          fi
          matrix=$(python scripts/generate_ci_matrix.py $ARGS)
          echo "matrix=$matrix" >> "$GITHUB_OUTPUT"
          count=$(echo "$matrix" | python -c "import sys,json; print(len(json.load(sys.stdin)['include']))")
          echo "suite_count=$count" >> "$GITHUB_OUTPUT"

  test:
    name: "${{ matrix.backend }} / ${{ matrix.suite }}"
    needs: [matrix]
    if: needs.matrix.outputs.suite_count != '0'
    runs-on: ${{ toJson(matrix.runs_on) }}
    timeout-minutes: ${{ matrix.timeout_minutes || 60 }}
    strategy:
      fail-fast: false
      matrix: ${{ fromJSON(needs.matrix.outputs.matrix) }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}

      - name: Install PyTorch (vendor-specific)
        run: ${{ matrix.install_cmd }}

      - name: Install HF libraries
        run: |
          pip install -U pip
          pip install "transformers[torch]" "diffusers[torch]" pytest pytest-timeout

      - name: Run suite
        env:
          HF_CI_BACKEND: ${{ matrix.backend }}
          HF_CI_DEVICE: ${{ matrix.torch_device }}
          HF_CI_EXTRAS: ${{ join(matrix.extras, ',') }}
          ${{ matrix.env }}
        run: ${{ matrix.command }}

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: "results-${{ matrix.backend }}-${{ matrix.suite }}"
          path: |
            test-results/
            *.xml
          retention-days: 14

  summary:
    name: CI Summary
    needs: [test]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Report status
        run: |
          echo "## CI Results" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "| Backend | Suite | Status |" >> $GITHUB_STEP_SUMMARY
          echo "|---------|-------|--------|" >> $GITHUB_STEP_SUMMARY
          # Status is derived from job context
          echo "Matrix had ${{ needs.matrix.outputs.suite_count }} combinations" >> $GITHUB_STEP_SUMMARY
```

**Step 2: Commit**

```bash
git add .github/workflows/hf-hardware-agnostic-ci.yml
git commit -m "feat: productionize CI workflow with timeouts, caching, artifacts"
```

---

### Task 5: Add Suite/Backend Filtering to Matrix Generator

**Objective:** Support `--suite-filter` and `--backend-filter` flags for targeted runs.

**Files:**
- Modify: `scripts/generate_ci_matrix.py`
- Modify: `tests/test_generate_ci_matrix.py`

**Step 1: Write failing test**

```python
def test_matrix_filter_by_backend():
    proc = subprocess.run(
        ["python", "scripts/generate_ci_matrix.py", "--backend-filter", "nvidia"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    backends = {e["backend"] for e in payload["include"]}
    assert backends == {"nvidia"}


def test_matrix_filter_by_suite():
    proc = subprocess.run(
        ["python", "scripts/generate_ci_matrix.py", "--suite-filter", "transformers_smoke"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    suites = {e["suite"] for e in payload["include"]}
    assert suites == {"transformers_smoke"}
```

**Step 2: Implement filtering in generate_ci_matrix.py**

Add `--suite-filter` and `--backend-filter` comma-separated args that restrict the matrix.

**Step 3: Run tests**

Run: `cd /Users/zhipeng/Workspace/omi-hf-ci && python -m pytest tests/test_generate_ci_matrix.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add scripts/generate_ci_matrix.py tests/test_generate_ci_matrix.py
git commit -m "feat: add suite and backend filtering to matrix generator"
```

---

## Phase 3: Documentation & Proposal Quality

### Task 6: Rewrite Architecture Documentation

**Objective:** Create comprehensive architecture doc suitable for HF team review.

**Files:**
- Modify: `docs/architecture.md`

Write a thorough architecture document covering:
- Problem statement (why hardware-agnostic CI matters for HF)
- Design principles (capability-based, vendor-neutral, extensible)
- Component diagram (ASCII art)
- Data flow (how a PR triggers tests across hardware)
- Adding a new backend (step-by-step guide)
- Adding a new test suite (step-by-step guide)
- Configuration reference (all fields in backends.json and test_suites.json)
- Runner requirements per vendor
- Security considerations

**Step 1: Write the document**

(See full content in implementation — approximately 200 lines of markdown)

**Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: rewrite architecture doc for HF team proposal"
```

---

### Task 7: Create Contributing Guide for Vendor Teams

**Objective:** Provide a clear guide for hardware vendors to onboard their accelerators.

**Files:**
- Create: `docs/contributing-backend.md`

Covers:
- Prerequisites (self-hosted runner setup)
- Backend JSON definition walkthrough
- Capability tags reference
- Testing your backend locally
- Submitting a PR
- CI runner labeling convention

**Step 1: Write the document**

**Step 2: Commit**

```bash
git add docs/contributing-backend.md
git commit -m "docs: add contributing guide for hardware vendor teams"
```

---

### Task 8: Rewrite README as Proposal Document

**Objective:** Make the README a compelling proposal document for the HF team.

**Files:**
- Modify: `README.md`

Structure:
- Title and tagline
- Problem statement
- Solution overview (with ASCII diagram)
- Supported hardware (table)
- How it works (3-step explanation)
- Quick start
- Configuration reference
- Extending the system
- FAQ
- License

**Step 1: Write the README**

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as HF team proposal document"
```

---

### Task 9: Add Comprehensive Tests

**Objective:** Ensure all components are well-tested.

**Files:**
- Modify: `tests/test_generate_ci_matrix.py`
- Create: `tests/test_validate_config.py` (already in Task 1)
- Create: `tests/conftest.py`

Add tests for:
- Capability matching edge cases (empty requires_all, empty requires_any, both set)
- Filter combinations
- Invalid config handling
- Schema validation pass/fail
- Matrix output structure integrity

**Step 1: Write tests**

**Step 2: Run full test suite**

Run: `cd /Users/zhipeng/Workspace/omi-hf-ci && python -m pytest tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/
git commit -m "test: add comprehensive tests for all CI components"
```

---

## Phase 4: Polish & Finalization

### Task 10: Add Nightly Full-Run Workflow

**Objective:** Separate fast PR checks from comprehensive nightly runs.

**Files:**
- Create: `.github/workflows/hf-nightly-ci.yml`

A nightly workflow that:
- Runs all suites on all backends
- Sends results to a summary issue or Slack
- Has longer timeouts
- Includes slow tests

**Step 1: Create workflow**

**Step 2: Commit**

```bash
git add .github/workflows/hf-nightly-ci.yml
git commit -m "feat: add nightly full-run CI workflow"
```

---

### Task 11: Add pyproject.toml and Dev Tooling

**Objective:** Proper Python project setup with linting and formatting.

**Files:**
- Create: `pyproject.toml`

Configure:
- pytest settings
- ruff linting
- project metadata

**Step 1: Create pyproject.toml**

**Step 2: Run linting and tests**

Run: `cd /Users/zhipeng/Workspace/omi-hf-ci && python -m pytest tests/ -v && python -m ruff check .`
Expected: All PASS

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyproject.toml with pytest and ruff config"
```

---

## Execution Order

1. Task 1: Schema validation (foundation)
2. Task 2: Enrich backends (data)
3. Task 3: Real test suites (data)
4. Task 4: Production workflow (infra)
5. Task 5: Filtering (features)
6. Task 6: Architecture docs (proposal)
7. Task 7: Contributing guide (proposal)
8. Task 8: README rewrite (proposal)
9. Task 9: Comprehensive tests (quality)
10. Task 10: Nightly workflow (completeness)
11. Task 11: pyproject.toml (polish)

## Success Criteria

- All tests pass: `python -m pytest tests/ -v`
- Config validation works: `python scripts/validate_config.py --config backends`
- Matrix generates correctly: `python scripts/generate_ci_matrix.py --pretty`
- Workflow YAML is valid
- Documentation is clear and comprehensive
- A hardware vendor can follow contributing guide to add their backend
