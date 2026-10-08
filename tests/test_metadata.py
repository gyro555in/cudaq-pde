import json
import subprocess
from pathlib import Path

import pytest

from cudaq_pde import metadata

REQUIRED_KEYS = {
    "cudaq_version",
    "cudaq_version_raw",
    "target",
    "gpu_name",
    "driver_version",
    "driver_max_cuda",
    "cuda_runtime",
    "python",
    "numpy",
    "scipy",
    "hostname",
    "slurm_job_id",
    "omp_num_threads",
    "openblas_num_threads",
    "slurm_cpus_per_task",
    "cpus_available",
    "git_commit",
    "git_dirty",
}


def test_required_keys_present() -> None:
    assert REQUIRED_KEYS <= set(metadata.collect_metadata())


def test_degrades_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd[0] == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(metadata.subprocess, "run", fake_run)
    meta = metadata.collect_metadata()
    assert meta["gpu_name"] is None
    assert meta["driver_version"] is None
    assert meta["driver_max_cuda"] is None


def test_versions_present() -> None:
    meta = metadata.collect_metadata()
    assert meta["python"]
    assert meta["numpy"]
    assert meta["scipy"]


def test_write_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    written = metadata.write_metadata(path)
    assert json.loads(path.read_text()) == written


@pytest.mark.gpu
def test_gpu_fields_populated() -> None:
    meta = metadata.collect_metadata()
    assert meta["gpu_name"]
    assert meta["driver_version"]
