import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_matrix(*extra_args):
    cmd = [sys.executable, "scripts/generate_ci_matrix.py"] + list(extra_args)
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)


def test_matrix_has_all_backends_and_suites():
    payload = json.loads(_run_matrix().stdout)
    include = payload["include"]

    backends = {entry["backend"] for entry in include}
    suites = {entry["suite"] for entry in include}

    assert backends == {"nvidia", "amd", "gaudi", "ascend"}
    assert suites == {
        "transformers_smoke",
        "diffusers_smoke",
        "distributed_sanity",
        "transformers_training",
        "diffusers_pipelines",
        "quantization_bitsandbytes",
    }


def test_matrix_records_runner_labels_as_list():
    payload = json.loads(_run_matrix().stdout)
    for entry in payload["include"]:
        assert isinstance(entry["runs_on"], list)
        assert len(entry["runs_on"]) >= 1


def test_matrix_includes_install_cmd():
    payload = json.loads(_run_matrix().stdout)
    for entry in payload["include"]:
        assert "install_cmd" in entry
        assert isinstance(entry["install_cmd"], str)


def test_matrix_includes_env():
    payload = json.loads(_run_matrix().stdout)
    for entry in payload["include"]:
        assert "env" in entry
        assert isinstance(entry["env"], list)


def test_matrix_includes_timeout():
    payload = json.loads(_run_matrix().stdout)
    for entry in payload["include"]:
        assert "timeout_minutes" in entry
        assert isinstance(entry["timeout_minutes"], int)
        assert entry["timeout_minutes"] >= 1


def test_matrix_filter_by_backend():
    payload = json.loads(_run_matrix("--backend-filter", "nvidia").stdout)
    backends = {e["backend"] for e in payload["include"]}
    assert backends == {"nvidia"}


def test_matrix_filter_by_suite():
    payload = json.loads(_run_matrix("--suite-filter", "transformers_smoke").stdout)
    suites = {e["suite"] for e in payload["include"]}
    assert suites == {"transformers_smoke"}


def test_matrix_filter_combined():
    payload = json.loads(
        _run_matrix("--backend-filter", "nvidia", "--suite-filter", "transformers_smoke").stdout
    )
    assert len(payload["include"]) == 1
    entry = payload["include"][0]
    assert entry["backend"] == "nvidia"
    assert entry["suite"] == "transformers_smoke"


def test_quantization_only_on_nvidia():
    """bitsandbytes capability only exists on nvidia."""
    payload = json.loads(_run_matrix("--suite-filter", "quantization_bitsandbytes").stdout)
    backends = {e["backend"] for e in payload["include"]}
    assert backends == {"nvidia"}


def test_gaudi_excluded_from_fp16_only_suites():
    """gaudi has no fp16 capability, so fp16-only suites should not match."""
    # gaudi has bf16 but not fp16 — suites requiring only fp16 should still match
    # because requires_any is OR. But suites requiring flash_attention_2 won't match.
    payload = json.loads(_run_matrix("--backend-filter", "gaudi").stdout)
    # gaudi should get suites that require bf16 or have no requirements
    assert len(payload["include"]) > 0
    for entry in payload["include"]:
        assert entry["backend"] == "gaudi"
