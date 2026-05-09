import json
import subprocess


def test_matrix_has_all_backends_and_suites() -> None:
    proc = subprocess.run(
        ["python", "scripts/generate_ci_matrix.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    include = payload["include"]

    backends = {entry["backend"] for entry in include}
    suites = {entry["suite"] for entry in include}

    assert backends == {"nvidia", "amd", "gaudi", "ascend"}
    assert suites == {"transformers_smoke", "diffusers_smoke", "distributed_sanity"}


def test_matrix_records_runner_labels_as_list() -> None:
    proc = subprocess.run(
        ["python", "scripts/generate_ci_matrix.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    for entry in payload["include"]:
        assert isinstance(entry["runs_on"], list)
        assert len(entry["runs_on"]) >= 1
