#!/usr/bin/env python3
"""Pilot test script for hardware-agnostic CI system.

Run this on a GPU instance to validate the CI pipeline works end-to-end.
Tests: config validation, matrix generation, PyTorch install, HF library smoke test.

Usage:
  python scripts/pilot_test.py --backend nvidia
  python scripts/pilot_test.py --backend amd
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def run_step(description: str, cmd: list[str], cwd: str | None = None, env: dict | None = None) -> bool:
    """Run a command and report success/failure."""
    print(f"▶ {description}")
    print(f"  Command: {' '.join(cmd)}")

    merged_env = {**os.environ}
    if env:
        merged_env.update(env)

    start = time.time()
    result = subprocess.run(
        cmd,
        cwd=cwd or str(ROOT),
        env=merged_env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"  ✓ PASS ({elapsed:.1f}s)")
        if result.stdout.strip():
            # Show last few lines of output
            lines = result.stdout.strip().split('\n')
            for line in lines[-3:]:
                print(f"    {line}")
        return True
    else:
        print(f"  ✗ FAIL ({elapsed:.1f}s)")
        if result.stderr.strip():
            lines = result.stderr.strip().split('\n')
            for line in lines[-5:]:
                print(f"    {line}")
        return False


def detect_gpu() -> dict:
    """Detect available GPU hardware."""
    info = {"type": "unknown", "name": "unknown", "driver": "unknown"}

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            info["type"] = "nvidia"
            info["name"] = parts[0].strip()
            info["driver"] = parts[1].strip() if len(parts) > 1 else "unknown"
            return info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try rocm-smi
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            info["type"] = "amd"
            info["name"] = "AMD GPU (ROCm)"
            return info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try npu-smi (Ascend)
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info["type"] = "ascend"
            info["name"] = "Huawei Ascend NPU"
            return info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return info


def get_backend_config(backend_name: str) -> dict | None:
    """Load backend config from backends.json."""
    config_path = ROOT / "ci" / "backends.json"
    data = json.loads(config_path.read_text())
    return data.get("backends", {}).get(backend_name)


def main():
    parser = argparse.ArgumentParser(description="Pilot test for hardware-agnostic CI")
    parser.add_argument(
        "--backend",
        choices=["nvidia", "amd", "gaudi", "ascend", "auto"],
        default="auto",
        help="Backend to test (auto=detect from hardware)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip PyTorch installation (if already installed)",
    )
    parser.add_argument(
        "--test-command",
        default=None,
        help="Override the test command to run",
    )
    parser.add_argument(
        "--workdir",
        default="/tmp/hf-test",
        help="Working directory for test runs",
    )
    args = parser.parse_args()

    banner("Hardware-Agnostic CI Pilot Test")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Working dir: {args.workdir}")

    # ──────────────────────────────────────────────
    # Step 1: Detect hardware
    # ──────────────────────────────────────────────
    banner("Step 1: Hardware Detection")
    gpu_info = detect_gpu()
    print(f"GPU Type: {gpu_info['type']}")
    print(f"GPU Name: {gpu_info['name']}")
    print(f"Driver: {gpu_info['driver']}")

    backend_name = args.backend
    if backend_name == "auto":
        backend_name = gpu_info["type"]
        if backend_name == "unknown":
            print("ERROR: Could not detect GPU hardware. Use --backend to specify.")
            sys.exit(1)
        print(f"Auto-detected backend: {backend_name}")

    backend = get_backend_config(backend_name)
    if not backend:
        print(f"ERROR: Backend '{backend_name}' not found in ci/backends.json")
        sys.exit(1)

    print(f"\nBackend config:")
    print(f"  Device: {backend['torch_device']}")
    print(f"  Capabilities: {backend.get('capabilities', [])}")

    # ──────────────────────────────────────────────
    # Step 2: Validate configs
    # ──────────────────────────────────────────────
    banner("Step 2: Config Validation")
    results = []

    results.append(run_step(
        "Validate backends config",
        [sys.executable, "scripts/validate_config.py", "--config", "backends"],
    ))

    results.append(run_step(
        "Validate suites config",
        [sys.executable, "scripts/validate_config.py", "--config", "suites"],
    ))

    # ──────────────────────────────────────────────
    # Step 3: Generate matrix
    # ──────────────────────────────────────────────
    banner("Step 3: Matrix Generation")
    results.append(run_step(
        f"Generate matrix for {backend_name}",
        [sys.executable, "scripts/generate_ci_matrix.py", "--backend-filter", backend_name, "--pretty"],
    ))

    # ──────────────────────────────────────────────
    # Step 4: Set up environment
    # ──────────────────────────────────────────────
    banner("Step 4: Environment Setup")

    # Set CI environment variables
    env = {
        "HF_CI_BACKEND": backend_name,
        "HF_CI_DEVICE": backend["torch_device"],
        "HF_CI_EXTRAS": ",".join(backend.get("setup", {}).get("extras", [])),
    }
    for k, v in backend.get("env", {}).items():
        env[k] = v
        os.environ[k] = v

    print(f"Environment variables set:")
    for k, v in env.items():
        print(f"  {k}={v}")

    # ──────────────────────────────────────────────
    # Step 5: Install PyTorch
    # ──────────────────────────────────────────────
    if not args.skip_install:
        banner("Step 5: PyTorch Installation")
        install_cmd = backend.get("setup", {}).get("install_cmd", "")
        if install_cmd:
            results.append(run_step(
                f"Install PyTorch for {backend_name}",
                ["bash", "-c", install_cmd],
            ))
        else:
            print("No install_cmd specified, skipping PyTorch installation")

        # Install HF libraries
        results.append(run_step(
            "Install Hugging Face libraries",
            ["bash", "-c", "pip install -U pip && pip install transformers diffusers pytest pytest-timeout"],
        ))
    else:
        banner("Step 5: Skipping PyTorch Installation (--skip-install)")

    # ──────────────────────────────────────────────
    # Step 6: Verify PyTorch sees GPU
    # ──────────────────────────────────────────────
    banner("Step 6: Verify PyTorch GPU Access")
    verify_cmd = f"""
import torch
print(f"PyTorch version: {{torch.__version__}}")
print(f"CUDA available: {{torch.cuda.is_available()}}")
device = torch.device('{backend["torch_device"]}')
print(f"Device: {{device}}")
if torch.cuda.is_available():
    print(f"GPU count: {{torch.cuda.device_count()}}")
    print(f"GPU name: {{torch.cuda.get_device_name(0)}}")
    x = torch.randn(100, 100, device=device)
    y = torch.matmul(x, x)
    print(f"Matmul test: OK (shape={{y.shape}})")
else:
    print("WARNING: CUDA not available, running on CPU")
    x = torch.randn(100, 100)
    y = torch.matmul(x, x)
    print(f"CPU matmul test: OK (shape={{y.shape}})")
"""
    results.append(run_step(
        "Verify PyTorch GPU access",
        [sys.executable, "-c", verify_cmd],
    ))

    # ──────────────────────────────────────────────
    # Step 7: Run smoke test
    # ──────────────────────────────────────────────
    banner("Step 7: HF Library Smoke Test")

    # Create a minimal smoke test
    smoke_test = """
import sys
print(f"Python: {sys.version}")

try:
    import transformers
    print(f"transformers: {transformers.__version__}")
except ImportError:
    print("transformers: not installed")
    sys.exit(1)

try:
    import diffusers
    print(f"diffusers: {diffusers.__version__}")
except ImportError:
    print("diffusers: not installed")
    sys.exit(1)

import torch
print(f"torch: {torch.__version__}")
print(f"device: {torch.device('{backend["torch_device"]}')}")
print("Smoke test PASSED")
"""
    results.append(run_step(
        "HF library smoke test",
        [sys.executable, "-c", smoke_test],
    ))

    # ──────────────────────────────────────────────
    # Step 8: Run actual test suite (optional)
    # ──────────────────────────────────────────────
    if args.test_command:
        banner("Step 8: Custom Test Command")
        results.append(run_step(
            "Run custom test",
            ["bash", "-c", args.test_command],
        ))
    else:
        banner("Step 8: Skipping Custom Tests (use --test-command to specify)")

    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────
    banner("Pilot Test Summary")
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed == total:
        print("\n✓ All tests passed! The CI pipeline works on this hardware.")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} test(s) failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
