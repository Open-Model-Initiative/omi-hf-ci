# Pilot Testing Guide

This guide shows how to test the hardware-agnostic CI system on real GPU hardware.

## Option 1: Vast.ai (Recommended — Cheapest)

Vast.ai offers GPU instances starting at ~$0.10/hr with a simple API.

### Setup

1. Create an account at https://cloud.vast.ai/
2. Add credits ($5-10 is enough for testing)
3. Get your API key at https://cloud.vast.ai/account/
4. Set the API key:
   ```bash
   export VAST_API_KEY="your-api-key-here"
   ```

### Run Tests

```bash
# Test on NVIDIA GPU (RTX 4090, ~$0.30/hr)
python scripts/vastai_test.py --backend nvidia

# Test on AMD GPU (MI250/MI300, ~$0.80/hr)
python scripts/vastai_test.py --backend amd

# Just launch an instance for manual testing
python scripts/vastai_test.py --backend nvidia --launch-only

# Terminate an instance
python scripts/vastai_test.py --terminate <instance-id>
```

### Cost Estimate

- NVIDIA RTX 4090: ~$0.30/hr (10 min test = ~$0.05)
- AMD MI250: ~$0.80/hr (10 min test = ~$0.13)

## Option 2: RunPod

RunPod offers similar pricing with a web UI.

### Setup

1. Create an account at https://www.runpod.io/
2. Add credits ($10 minimum)
3. Create a GPU pod with:
   - Template: PyTorch 2.2
   - GPU: RTX 4090 (NVIDIA) or MI250 (AMD)
   - Disk: 50GB

### Run Tests

SSH into the pod and run:

```bash
# Clone the repo
git clone https://github.com/Open-Model-Initiative/omi-hf-ci.git
cd omi-hf-ci

# Install dependencies
pip install jsonschema pytest

# Run pilot test
python scripts/pilot_test.py --backend nvidia
```

## Option 3: Lambda Labs

Lambda Labs offers ML-optimized instances.

### Setup

1. Create an account at https://lambdalabs.com/
2. Launch an instance (GPU: A10 or A100)
3. SSH into the instance

### Run Tests

```bash
git clone https://github.com/Open-Model-Initiative/omi-hf-ci.git
cd omi-hf-ci
pip install jsonschema pytest
python scripts/pilot_test.py --backend nvidia
```

## Option 4: Google Colab (Free — NVIDIA Only)

Google Colab provides free GPU access (T4, limited hours).

### Steps

1. Go to https://colab.research.google.com/
2. Create a new notebook
3. Change runtime to GPU (Runtime > Change runtime type > T4)
4. Run these cells:

```python
# Cell 1: Clone repo
!git clone https://github.com/Open-Model-Initiative/omi-hf-ci.git
%cd omi-hf-ci

# Cell 2: Install dependencies
!pip install jsonschema pytest

# Cell 3: Run pilot test
!python scripts/pilot_test.py --backend nvidia --skip-install
```

## Option 5: Self-Hosted Runner (For Production)

For actual CI integration, set up a self-hosted GitHub Actions runner.

### NVIDIA Runner

```bash
# On a machine with NVIDIA GPU
# 1. Install GitHub Actions runner
# https://docs.github.com/en/actions/hosting-your-own-runners

# 2. Configure with labels
./config.sh --labels self-hosted,linux,x64,nvidia

# 3. Install dependencies
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install jsonschema pytest

# 4. Start the runner
./run.sh
```

### AMD Runner

```bash
# On a machine with AMD GPU (ROCm installed)
./config.sh --labels self-hosted,linux,x64,amd
./run.sh
```

## Manual Testing

If you already have GPU access, run the pilot test directly:

```bash
# Clone the repo
git clone https://github.com/Open-Model-Initiative/omi-hf-ci.git
cd omi-hf-ci

# Install dependencies
pip install jsonschema pytest

# Auto-detect GPU and run tests
python scripts/pilot_test.py --backend auto

# Or specify backend explicitly
python scripts/pilot_test.py --backend nvidia
python scripts/pilot_test.py --backend amd

# Skip PyTorch installation (if already installed)
python scripts/pilot_test.py --backend nvidia --skip-install

# Run with custom test command
python scripts/pilot_test.py --backend nvidia --test-command "python -m pytest tests/ -v"
```

## What the Pilot Test Checks

1. **Config Validation** — JSON schemas are valid
2. **Matrix Generation** — Correct backend/suite matching
3. **Environment Variables** — CI vars set correctly
4. **PyTorch Installation** — Vendor-specific install works
5. **GPU Detection** — PyTorch sees the GPU
6. **HF Libraries** — transformers/diffusers import correctly
7. **Smoke Test** — Basic functionality works

## Expected Output

```
============================================================
  Hardware-Agnostic CI Pilot Test
============================================================

Python: 3.10.x
Platform: linux
Working dir: /tmp/hf-test

============================================================
  Step 1: Hardware Detection
============================================================

GPU Type: nvidia
GPU Name: NVIDIA GeForce RTX 4090
Driver: 535.xx.xx
Auto-detected backend: nvidia

Backend config:
  Device: cuda
  Capabilities: ['fp16', 'bf16', 'flash_attention_2', 'bitsandbytes', 'torch_compile']

...

============================================================
  Pilot Test Summary
============================================================

Results: 7/7 passed

✓ All tests passed! The CI pipeline works on this hardware.
```

## Troubleshooting

**"No GPU detected"**
- Ensure GPU drivers are installed
- Check `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD)

**"PyTorch CUDA not available"**
- Install correct PyTorch version for your CUDA/ROCm
- Check driver compatibility

**"transformers not installed"**
- Run: `pip install transformers diffusers`

**"Instance too expensive"**
- Use smaller GPUs (RTX 3090 instead of A100)
- Use spot/preemptible instances
