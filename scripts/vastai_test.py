#!/usr/bin/env python3
"""Provision GPU instances on Vast.ai for pilot testing.

Usage:
  # Set your Vast.ai API key
  export VAST_API_KEY="your-api-key-here"

  # Launch NVIDIA instance and run pilot test
  python scripts/vastai_test.py --backend nvidia

  # Launch AMD instance and run pilot test
  python scripts/vastai_test.py --backend amd

  # Just launch an instance (manual testing)
  python scripts/vastai_test.py --backend nvidia --launch-only
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

# Vast.ai API configuration
VAST_API_BASE = "https://console.vast.ai/api/v0"

# Instance templates per backend
TEMPLATES = {
    "nvidia": {
        "search_filter": {
            "gpu_name": {"eq": "RTX_4090"},  # Cheapest good GPU
            "num_gpus": {"eq": 1},
            "inet_up_cost": {"lte": 0.1},
            "inet_down_cost": {"lte": 0.1},
            "dph_total": {"lte": 0.50},  # Max $0.50/hr
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
        },
        "image": "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        "disk": 50,
    },
    "amd": {
        "search_filter": {
            "gpu_name": {"in": ["MI250", "MI300X", "RX_7900_XTX"]},
            "num_gpus": {"eq": 1},
            "dph_total": {"lte": 1.0},
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
        },
        "image": "rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.2",
        "disk": 50,
    },
}


def vastai_api(method: str, endpoint: str, data: dict | None = None) -> dict:
    """Make a Vast.ai API call."""
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        print("ERROR: Set VAST_API_KEY environment variable")
        print("Get your API key at: https://cloud.vast.ai/account/")
        sys.exit(1)

    url = f"{VAST_API_BASE}{endpoint}"
    cmd = [
        "curl", "-s", "-X", method,
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
    ]
    if data:
        cmd += ["-d", json.dumps(data)]
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"API call failed: {result.stderr}")
        sys.exit(1)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Invalid API response: {result.stdout}")
        sys.exit(1)


def search_instances(backend: str) -> list[dict]:
    """Search for available GPU instances."""
    template = TEMPLATES[backend]
    print(f"Searching for {backend} GPU instances...")

    result = vastai_api("GET", "/bundles?q=" + json.dumps(template["search_filter"]))
    offers = result.get("offers", [])

    if not offers:
        print(f"No {backend} instances available within budget")
        return []

    # Show top 3 options
    print(f"\nTop {min(3, len(offers))} options:")
    for i, offer in enumerate(offers[:3]):
        print(f"  {i+1}. {offer.get('gpu_name', 'unknown')} "
              f"- ${offer.get('dph_total', 0):.3f}/hr "
              f"- {offer.get('num_gpus', 0)} GPU "
              f"- {offer.get('cpu_ram', 0):.0f}GB RAM "
              f"- ID: {offer.get('id', 'unknown')}")

    return offers


def launch_instance(backend: str, offer_id: int | None = None) -> dict:
    """Launch a GPU instance."""
    template = TEMPLATES[backend]

    if offer_id is None:
        offers = search_instances(backend)
        if not offers:
            sys.exit(1)
        offer_id = offers[0]["id"]

    print(f"\nLaunching instance {offer_id}...")

    # Create instance
    data = {
        "client_id": "me",
        "image": template["image"],
        "disk": template["disk"],
        "onstart": "bash -c 'apt-get update && apt-get install -y git python3-pip && pip3 install transformers diffusers pytest'",
        "runtype": "ssh",
        "cancel_unavail": True,
    }

    result = vastai_api("POST", f"/asks/{offer_id}/", data)
    instance_id = result.get("new_contract")
    if not instance_id:
        print(f"Failed to launch: {result}")
        sys.exit(1)

    print(f"Instance launched: {instance_id}")
    print(f"Waiting for instance to be ready...")

    # Wait for instance to be ready
    for _ in range(30):  # 5 minutes max
        time.sleep(10)
        status = vastai_api("GET", f"/instances/{instance_id}")
        instance = status.get("instances", [{}])[0]
        state = instance.get("actual_status", "unknown")

        if state == "running":
            print(f"Instance ready!")
            # Get SSH connection info
            ssh_host = instance.get("ssh_host", "unknown")
            ssh_port = instance.get("ssh_port", 22)
            print(f"SSH: ssh -p {ssh_port} root@{ssh_host}")
            return instance

        print(f"  Status: {state}")

    print("Timeout waiting for instance")
    return {}


def terminate_instance(instance_id: int) -> None:
    """Terminate a GPU instance."""
    print(f"Terminating instance {instance_id}...")
    vastai_api("DELETE", f"/instances/{instance_id}/")
    print("Instance terminated")


def run_pilot_test(instance: dict, backend: str) -> bool:
    """Run the pilot test on a remote instance."""
    ssh_host = instance.get("ssh_host", "unknown")
    ssh_port = instance.get("ssh_port", 22)

    print(f"\nRunning pilot test on {ssh_host}:{ssh_port}...")

    # Copy pilot test script to instance
    scp_cmd = [
        "scp", "-P", str(ssh_port), "-o", "StrictHostKeyChecking=no",
        str(ROOT / "scripts" / "pilot_test.py"),
        f"root@{ssh_host}:/tmp/pilot_test.py",
    ]
    subprocess.run(scp_cmd, check=True)

    # Run pilot test
    ssh_cmd = [
        "ssh", "-p", str(ssh_port), "-o", "StrictHostKeyChecking=no",
        f"root@{ssh_host}",
        f"cd /tmp && python3 pilot_test.py --backend {backend}",
    ]
    result = subprocess.run(ssh_cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Vast.ai GPU instance manager")
    parser.add_argument(
        "--backend",
        choices=["nvidia", "amd"],
        required=True,
        help="Backend to test",
    )
    parser.add_argument(
        "--offer-id",
        type=int,
        default=None,
        help="Specific Vast.ai offer ID to use",
    )
    parser.add_argument(
        "--launch-only",
        action="store_true",
        help="Just launch the instance, don't run tests",
    )
    parser.add_argument(
        "--terminate",
        type=int,
        default=None,
        help="Terminate a specific instance ID",
    )
    args = parser.parse_args()

    if args.terminate:
        terminate_instance(args.terminate)
        return

    # Launch instance
    instance = launch_instance(args.backend, args.offer_id)
    if not instance:
        sys.exit(1)

    instance_id = instance.get("id")
    if not instance_id:
        print("ERROR: Could not get instance ID")
        sys.exit(1)

    if args.launch_only:
        print(f"\nInstance launched. SSH into it to run tests manually.")
        print(f"Remember to terminate when done: python scripts/vastai_test.py --terminate {instance_id}")
        return

    try:
        # Run pilot test
        success = run_pilot_test(instance, args.backend)
        if success:
            print("\n✓ Pilot test completed successfully!")
        else:
            print("\n✗ Pilot test failed")
    finally:
        # Always terminate instance to avoid charges
        print("\nTerminating instance to avoid charges...")
        terminate_instance(instance_id)


if __name__ == "__main__":
    main()
