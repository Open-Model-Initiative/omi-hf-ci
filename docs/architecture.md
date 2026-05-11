# Hardware-agnostic CI design for Hugging Face libraries

This repository implements a capability-driven CI system that decouples **test intent**
from **hardware vendor implementation**.

## Goals

- Support NVIDIA GPUs, AMD GPUs, Intel Gaudi accelerators, and Huawei Ascend NPUs.
- Run a shared suite catalog for libraries such as `transformers` and `diffusers`.
- Keep vendor-specific logic in one place (`ci/backends.json`).
- Avoid hardcoding per-vendor test workflows.

## Components

1. `ci/backends.json`
   - Declares executor labels, device type, runtime extras, and capabilities.
2. `ci/test_suites.json`
   - Declares suites and capability requirements (`requires_any`, `requires_all`).
3. `scripts/generate_ci_matrix.py`
   - Matches suites to backends based on capabilities and emits a JSON matrix.
4. `.github/workflows/hf-hardware-agnostic-ci.yml`
   - Builds the matrix and executes each eligible suite on matching runner labels.

## Execution model

- A suite only runs on backends that satisfy its capability requirements.
- Hardware abstraction is represented as capabilities (`bf16`, `fp16`, etc.) rather
  than vendor checks.
- Backend-specific environment details are exposed through generic env vars:
  - `HF_CI_BACKEND`
  - `HF_CI_DEVICE`

## Extending the system

- Add a new hardware backend by appending to `ci/backends.json`.
- Add a new test suite by appending to `ci/test_suites.json`.
- Add richer feature gates by introducing new capability tags.

## Notes

- The workflow assumes availability of appropriately labeled self-hosted runners.
- Real projects can split installation logic per backend via script hooks keyed by
  the `extras` field from the generated matrix.
