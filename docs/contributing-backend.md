# Contributing a New Hardware Backend

This guide is for hardware vendors (AMD, Intel, Huawei, etc.) who want to add
their accelerator to the Hugging Face CI system.

## Prerequisites

Before you begin, ensure you have:

1. **A self-hosted GitHub Actions runner** with your hardware installed.
2. **PyTorch support** for your device (either upstream or via a plugin).
3. **A working `pip install` command** that sets up PyTorch for your hardware.

## Step-by-Step Guide

### Step 1: Fork and Clone

```bash
git clone https://github.com/<your-org>/omi-hf-ci.git
cd omi-hf-ci
```

### Step 2: Define Your Backend

Edit `ci/backends.json` and add your entry:

```json
{
  "my_accelerator": {
    "type": "accelerator",
    "labels": ["self-hosted", "linux", "x64", "my-accelerator"],
    "torch_device": "my_device",
    "setup": {
      "python": "3.11",
      "extras": ["my-accelerator"],
      "install_cmd": "pip install torch-my-plugin"
    },
    "env": {
      "MY_DEVICE_VISIBLE_DEVICES": "0",
      "MY_PLUGIN_SETTING": "value"
    },
    "capabilities": ["fp16", "bf16"]
  }
}
```

**Field reference:**

| Field | What to put |
|-------|-------------|
| `type` | `"gpu"` for GPUs, `"accelerator"` for specialized hardware |
| `labels` | Must include `"self-hosted"`, `"linux"`, `"x64"`, and your vendor tag |
| `torch_device` | The `torch.device` string your hardware uses |
| `setup.python` | Python version on your runner |
| `setup.extras` | Tags for documentation |
| `setup.install_cmd` | Shell command to install PyTorch for your hardware |
| `env` | Environment variables needed by your runtime |
| `capabilities` | What your hardware supports (see below) |

### Step 3: Declare Capabilities

Capabilities tell the system which test suites can run on your hardware.

**Available capabilities:**

| Capability | Description | Required for |
|------------|-------------|--------------|
| `fp16` | Half-precision float support | Most smoke tests |
| `bf16` | BFloat16 support | Training tests |
| `flash_attention_2` | Flash Attention v2 | Attention-heavy tests |
| `bitsandbytes` | 8-bit quantization | Quantization tests |
| `torch_compile` | torch.compile() support | Compilation tests |
| `hpu_graph` | Graph mode (Intel Gaudi) | Gaudi-specific tests |

**Only declare capabilities your hardware actually supports.** If you claim
`bitsandbytes` but it doesn't work, the CI will fail.

### Step 4: Validate Your Config

```bash
pip install jsonschema
python scripts/validate_config.py --config backends
```

Expected output: `OK: backends config is valid`

### Step 5: Verify Matrix Generation

```bash
python scripts/generate_ci_matrix.py --backend-filter my_accelerator --pretty
```

This shows exactly which test suites will run on your hardware.

### Step 6: Test Locally

Run one of the matched suites on your hardware:

```bash
export HF_CI_BACKEND=my_accelerator
export HF_CI_DEVICE=my_device
# Run the command from the matrix output
python -m pytest tests/test_modeling_common.py -x -q --timeout=300
```

### Step 7: Set Up the Runner

Follow the [GitHub Actions self-hosted runner guide](https://docs.github.com/en/actions/hosting-your-own-runners).

**Critical:** Apply the same labels you used in `backends.json`:

```bash
./config.sh --labels self-hosted,linux,x64,my-accelerator
```

### Step 8: Submit a PR

1. Commit your changes to `ci/backends.json`.
2. Open a PR against the main branch.
3. The CI will automatically validate your config.
4. A maintainer will review and merge.

## Runner Requirements

Your self-hosted runner must have:

- Ubuntu 22.04 or later
- Python 3.10+
- Your hardware drivers and runtime installed
- Network access to PyPI
- At least 50GB free disk space

## Troubleshooting

**Q: My backend shows 0 suites in the matrix.**
A: Check that your `capabilities` array includes at least one tag that a suite
requires. Run `python scripts/generate_ci_matrix.py --pretty` to see all
requirements.

**Q: The install command fails in CI.**
A: Test it locally first. Ensure all dependencies (drivers, CUDA toolkit, etc.)
are pre-installed on your runner.

**Q: How do I add a new capability tag?**
A: Open an issue describing the capability. If approved, it will be added to the
schema and you can use it in your backend definition.

## Questions?

Open an issue or reach out to the maintainers.
