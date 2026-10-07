"""Run metadata for reproducibility.

Every benchmark or result should write the dictionary from
:func:`collect_metadata` next to its data. Collection never raises: any field
that cannot be determined (no GPU on the login node, no cudaq, not a git
checkout) is ``None``.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

_RUNTIME_DISTS = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-runtime-cu13",
    "nvidia-cuda-runtime",
)


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    """Return stripped stdout of ``cmd`` or None on any failure."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=20, cwd=cwd, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _dist_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _cudaq_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cudaq_version": None,
        "cudaq_version_raw": None,
        "target": None,
    }
    try:
        import cudaq
    except Exception:
        return info
    raw = str(getattr(cudaq, "__version__", ""))
    match = re.search(r"\d+\.\d+\.\d+", raw)
    info["cudaq_version_raw"] = raw or None
    info["cudaq_version"] = match.group(0) if match else None
    try:
        info["target"] = cudaq.get_target().name
    except Exception:
        pass
    return info


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu_name": None,
        "driver_version": None,
        "driver_max_cuda": None,
    }
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    if query:
        first = query.splitlines()[0]
        name, _, driver = (part.strip() for part in first.partition(","))
        info["gpu_name"] = name or None
        info["driver_version"] = driver or None
    header = _run(["nvidia-smi"])
    if header:
        match = re.search(r"CUDA Version:\s*([\d.]+)", header)
        if match:
            info["driver_max_cuda"] = match.group(1)
    return info


def _cuda_runtime_version() -> str | None:
    """Version of the CUDA runtime wheel that CUDA-Q actually loads."""
    for dist in _RUNTIME_DISTS:
        found = _dist_version(dist)
        if found:
            return found
    return None


def _git_info() -> dict[str, Any]:
    repo = Path(__file__).resolve().parent
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    dirty = None
    if commit:
        status = _run(["git", "status", "--porcelain"], cwd=repo)
        dirty = bool(status) if status is not None else None
    return {"git_commit": commit, "git_dirty": dirty}


def collect_metadata() -> dict[str, Any]:
    """Collect provenance for a run.

    Returns
    -------
    dict
        JSON-serializable. Keys: ``cudaq_version``, ``cudaq_version_raw``,
        ``target``, ``gpu_name``, ``driver_version``, ``driver_max_cuda``
        (highest CUDA the driver supports, not the runtime in use),
        ``cuda_runtime`` (bundled runtime wheel version), ``python``,
        ``numpy``, ``scipy``, ``hostname``, ``slurm_job_id``, thread settings
        (``omp_num_threads``, ``openblas_num_threads``, ``slurm_cpus_per_task``,
        ``cpus_available``), ``git_commit``, ``git_dirty``. Unknown values are ``None``.
    """
    meta: dict[str, Any] = {}
    meta.update(_cudaq_info())
    meta.update(_gpu_info())
    meta["cuda_runtime"] = _cuda_runtime_version()
    meta["python"] = platform.python_version()
    meta["numpy"] = _dist_version("numpy")
    meta["scipy"] = _dist_version("scipy")
    meta["hostname"] = socket.gethostname()
    meta["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    meta["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
    meta["openblas_num_threads"] = os.environ.get("OPENBLAS_NUM_THREADS")
    meta["slurm_cpus_per_task"] = os.environ.get("SLURM_CPUS_PER_TASK")
    meta["cpus_available"] = len(os.sched_getaffinity(0))
    meta.update(_git_info())
    return meta


def write_metadata(path: str | Path) -> dict[str, Any]:
    """Write :func:`collect_metadata` to ``path`` as JSON and return it."""
    meta = collect_metadata()
    Path(path).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return meta
