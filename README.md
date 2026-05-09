# omi-hf-ci

Experimental hardware-agnostic CI for Hugging Face libraries (`transformers`, `diffusers`).

## What this repo now includes

- Capability-based backend catalog for:
  - NVIDIA GPU
  - AMD GPU (ROCm)
  - Intel Gaudi (HPU)
  - Huawei Ascend (NPU)
- Suite catalog for Hugging Face smoke/distributed tests.
- Matrix generator to map suites to compatible hardware.
- GitHub Actions workflow that runs tests against compatible runners.

## Quick start

```bash
python scripts/generate_ci_matrix.py --pretty
python -m pytest -q tests/test_generate_ci_matrix.py
```

See `docs/architecture.md` for design details.
