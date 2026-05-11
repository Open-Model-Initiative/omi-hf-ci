# Hardware-Agnostic CI for Hugging Face Libraries

## Architecture Document

### 1. Problem Statement

Hugging Face libraries (`transformers`, `diffusers`, `accelerate`) run on diverse
hardware: NVIDIA GPUs, AMD GPUs (ROCm), Intel Gaudi accelerators, and Huawei Ascend
NPUs. Today, each vendor maintains separate CI pipelines with duplicated test logic,
inconsistent coverage, and fragmented failure reporting.

This creates three problems:

1. **Duplicated effort** — Each vendor reimplements the same test workflows.
2. **Inconsistent coverage** — A test may pass on NVIDIA but never run on AMD.
3. **Hard to extend** — Adding a new accelerator requires copying and modifying
   entire workflow files.

### 2. Design Principles

| Principle | Meaning |
|-----------|---------|
| **Capability-based** | Tests declare *what* they need (`bf16`, `flash_attention_2`), not *where* they run. |
| **Vendor-neutral** | No vendor-specific logic in test definitions. Vendor details live in one config file. |
| **Extensible** | Adding a backend = one JSON entry. Adding a suite = one JSON entry. |
| **Fail-fast** | Config validation catches errors before any runner spins up. |

### 3. Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions Workflow                    │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   Validate    │───▶│ Build Matrix │───▶│  Run Tests   │   │
│  │   Configs     │    │              │    │              │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                    │           │
│         ▼                   ▼                    ▼           │
│  ci/backends.json    generate_ci_matrix.py   Self-hosted     │
│  ci/test_suites.json                          runners        │
│  ci/*.schema.json                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. Data Flow

```
PR opened or workflow_dispatch
        │
        ▼
┌─ Validate Configs ──────────────────────────────────┐
│  1. Load ci/backends.json + ci/test_suites.json     │
│  2. Validate against JSON schemas                   │
│  3. Fail early if invalid                           │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─ Build Matrix ──────────────────────────────────────┐
│  1. For each (backend, suite) pair:                 │
│     - Check: backend.capabilities ∩ suite.requires  │
│     - If match → add to matrix                      │
│  2. Apply optional filters (--backend-filter, etc.) │
│  3. Emit JSON matrix                                │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─ Run Tests ─────────────────────────────────────────┐
│  For each matrix entry:                             │
│    1. Spin up runner with matching labels           │
│    2. Install PyTorch via vendor install_cmd        │
│    3. Install HF libraries                          │
│    4. Set HF_CI_BACKEND, HF_CI_DEVICE env vars      │
│    5. Run test command                              │
│    6. Upload artifacts                              │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─ Summary ───────────────────────────────────────────┐
│  Aggregate results, report pass/fail                │
└─────────────────────────────────────────────────────┘
```

### 5. Configuration Reference

#### 5.1 `ci/backends.json`

Each backend declares:

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | `"gpu"`, `"accelerator"`, or `"cpu"` |
| `labels` | `string[]` | GitHub Actions runner labels |
| `torch_device` | `string` | PyTorch device name (`cuda`, `hpu`, `npu`) |
| `setup.python` | `string` | Python version |
| `setup.extras` | `string[]` | Tags for documentation (e.g., `["cuda"]`) |
| `setup.install_cmd` | `string` | Vendor-specific PyTorch install command |
| `env` | `object` | Environment variables for the runner |
| `capabilities` | `string[]` | What this hardware supports |

#### 5.2 `ci/test_suites.json`

Each suite declares:

| Field | Type | Description |
|-------|------|-------------|
| `package` | `string` | HF package name (`transformers`, `diffusers`, `common`) |
| `command` | `string` | pytest command to run |
| `requires_any` | `string[]` | Need at least one of these capabilities (OR) |
| `requires_all` | `string[]` | Need all of these capabilities (AND) |
| `timeout_minutes` | `int` | Job timeout (default: 60) |
| `multi_gpu` | `bool` | Whether suite needs multiple GPUs |
| `tags` | `string[]` | Metadata tags (`smoke`, `training`, etc.) |

#### 5.3 Capability Tags

| Tag | Meaning | Who has it |
|-----|---------|------------|
| `fp16` | Half-precision float | nvidia, amd, ascend |
| `bf16` | BFloat16 | nvidia, amd, gaudi, ascend |
| `flash_attention_2` | Flash Attention v2 | nvidia |
| `bitsandbytes` | 8-bit quantization | nvidia |
| `torch_compile` | torch.compile() | nvidia |
| `hpu_graph` | HPU graph mode | gaudi |

### 6. Adding a New Backend

**Step 1:** Add an entry to `ci/backends.json`:

```json
{
  "my_vendor": {
    "type": "accelerator",
    "labels": ["self-hosted", "linux", "x64", "my-vendor"],
    "torch_device": "mvp",
    "setup": {
      "python": "3.11",
      "extras": ["my-vendor"],
      "install_cmd": "pip install torch-my-vendor"
    },
    "env": {
      "MY_VENDOR_VISIBLE_DEVICES": "0"
    },
    "capabilities": ["fp16", "bf16"]
  }
}
```

**Step 2:** Validate:

```bash
python scripts/validate_config.py --config backends
```

**Step 3:** Verify matrix includes your backend:

```bash
python scripts/generate_ci_matrix.py --backend-filter my_vendor --pretty
```

**Step 4:** Set up a self-hosted runner with the matching labels.

**Step 5:** Submit a PR.

### 7. Adding a New Test Suite

**Step 1:** Add an entry to `ci/test_suites.json`:

```json
{
  "my_suite": {
    "package": "transformers",
    "command": "python -m pytest tests/my_tests/ -x -q",
    "requires_any": ["bf16"],
    "timeout_minutes": 30,
    "tags": ["custom"]
  }
}
```

**Step 2:** Validate and verify:

```bash
python scripts/validate_config.py --config suites
python scripts/generate_ci_matrix.py --suite-filter my_suite --pretty
```

### 8. Runner Requirements

| Backend | Labels | OS | GPU/Accelerator |
|---------|--------|----|-----------------|
| nvidia | `self-hosted, linux, x64, nvidia` | Ubuntu 22.04+ | NVIDIA GPU (A100/H100 recommended) |
| amd | `self-hosted, linux, x64, amd` | Ubuntu 22.04+ | AMD GPU (MI250/MI300) |
| gaudi | `self-hosted, linux, x64, gaudi` | Ubuntu 22.04+ | Intel Gaudi 2/3 |
| ascend | `self-hosted, linux, x64, ascend` | Ubuntu 22.04+ | Huawei Ascend 910B |

### 9. Security Considerations

- **Self-hosted runners** must be scoped to the repository or organization.
- **Secrets** (API keys, registry credentials) are managed via GitHub Secrets.
- **Config validation** prevents malformed matrix entries from reaching runners.
- **Concurrency control** cancels in-progress runs when new commits arrive.

### 10. Extending the System

Future directions:

- **Multi-GPU suites** — Use `multi_gpu: true` flag + runner GPU count.
- **Model caching** — Shared cache volume for downloaded models.
- **Result dashboard** — Aggregate results into a web UI.
- **Cost tracking** — Monitor runner usage per vendor.
- **Nightly full runs** — Separate workflow for comprehensive testing.
