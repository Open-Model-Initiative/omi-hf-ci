# Hardware-Agnostic CI for Hugging Face

A capability-driven continuous integration system that runs Hugging Face library tests
across NVIDIA, AMD, Intel Gaudi, and Huawei Ascend hardware — from a single workflow.

## The Problem

Hugging Face libraries (`transformers`, `diffusers`, `accelerate`) support multiple
hardware backends. Today, testing across these backends requires:

- Separate CI pipelines per vendor
- Duplicated test logic
- Inconsistent coverage (a test may pass on NVIDIA but never run on AMD)
- High maintenance cost for vendor teams

## The Solution

A single, declarative CI system where:

1. **Backends declare capabilities** (`bf16`, `flash_attention_2`, etc.)
2. **Test suites declare requirements** (needs `bf16` OR `fp16`)
3. **A matrix generator matches them automatically**
4. **One GitHub Actions workflow runs everything**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  backends   │     │    suites    │     │   matrix    │
│  .json      │────▶│   .json     │────▶│  generator  │
│             │     │             │     │             │
│ nvidia:     │     │ smoke:      │     │ nvidia +    │
│  [bf16,     │     │  needs:     │     │  smoke  ✓   │
│   fp16,     │     │  [bf16 OR   │     │             │
│   flash_attn│     │   fp16]     │     │ amd +       │
│  ]          │     │             │     │  smoke  ✓   │
│             │     │ quantize:   │     │             │
│ amd:        │     │  needs:     │     │ nvidia +    │
│  [bf16,     │     │  [bitsand   │     │  quant  ✓   │
│   fp16]     │     │   bytes]    │     │             │
│             │     │             │     │ amd +       │
│ ascend:     │     │             │     │  quant  ✗   │
│  [bf16,     │     │             │     │  (no bnb)   │
│   fp16]     │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Supported Hardware

| Backend | Device | Capabilities | Runner Labels |
|---------|--------|--------------|---------------|
| NVIDIA | GPU (CUDA) | fp16, bf16, flash_attention_2, bitsandbytes, torch_compile | `self-hosted, linux, x64, nvidia` |
| AMD | GPU (ROCm) | fp16, bf16 | `self-hosted, linux, x64, amd` |
| Intel Gaudi | HPU | bf16, hpu_graph | `self-hosted, linux, x64, gaudi` |
| Huawei Ascend | NPU | fp16, bf16 | `self-hosted, linux, x64, ascend` |

## How It Works

### 1. Validate

Before any runner spins up, configs are validated against JSON schemas:

```bash
python scripts/validate_config.py --config backends
python scripts/validate_config.py --config suites
```

### 2. Generate Matrix

The generator matches suites to compatible backends:

```bash
python scripts/generate_ci_matrix.py --pretty
```

Output:
```json
{
  "include": [
    {
      "backend": "nvidia",
      "suite": "transformers_smoke",
      "package": "transformers",
      "command": "python -m pytest tests/test_modeling_common.py -x -q --timeout=300",
      "runs_on": ["self-hosted", "linux", "x64", "nvidia"],
      "python": "3.11",
      "install_cmd": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121",
      "torch_device": "cuda",
      "timeout_minutes": 30
    }
  ]
}
```

### 3. Execute

GitHub Actions runs each matrix entry on the matching self-hosted runner.

## Test Suites

| Suite | Package | What it tests | Requires |
|-------|---------|---------------|----------|
| `transformers_smoke` | transformers | Core modeling | fp16 OR bf16 |
| `transformers_training` | transformers | Trainer | bf16 |
| `diffusers_smoke` | diffusers | UNet models | fp16 OR bf16 |
| `diffusers_pipelines` | diffusers | Pipeline inference | bf16 |
| `distributed_sanity` | common | Multi-GPU | (any) |
| `quantization_bitsandbytes` | transformers | 8-bit quantization | bitsandbytes |

## Quick Start

```bash
# Validate configs
python scripts/validate_config.py --config backends
python scripts/validate_config.py --config suites

# Generate matrix
python scripts/generate_ci_matrix.py --pretty

# Filter by backend or suite
python scripts/generate_ci_matrix.py --backend-filter nvidia --pretty
python scripts/generate_ci_matrix.py --suite-filter transformers_smoke --pretty

# Run tests
python -m pytest tests/ -v
```

## Extending the System

### Add a new hardware backend

Append to `ci/backends.json`. See [docs/contributing-backend.md](docs/contributing-backend.md).

### Add a new test suite

Append to `ci/test_suites.json`. The suite will automatically run on all backends
that satisfy its capability requirements.

### Add a new capability

Introduce a new tag (e.g., `int8`, `multi_gpu`) and declare it in the relevant
backends and suites.

## Project Structure

```
omi-hf-ci/
├── ci/
│   ├── backends.json          # Hardware backend definitions
│   ├── backends.schema.json   # JSON schema for backends
│   ├── test_suites.json       # Test suite definitions
│   └── suites.schema.json     # JSON schema for suites
├── scripts/
│   ├── generate_ci_matrix.py  # Matrix generator
│   └── validate_config.py     # Config validator
├── tests/
│   ├── test_generate_ci_matrix.py
│   └── test_validate_config.py
├── docs/
│   ├── architecture.md        # Detailed architecture
│   └── contributing-backend.md # Vendor onboarding guide
└── .github/workflows/
    └── hf-hardware-agnostic-ci.yml
```

## For Hardware Vendors

Want to add your accelerator? See the [Contributing Guide](docs/contributing-backend.md).

**TL;DR:**
1. Add your backend to `ci/backends.json`
2. Set up a self-hosted runner with matching labels
3. Submit a PR

## License

Apache 2.0
