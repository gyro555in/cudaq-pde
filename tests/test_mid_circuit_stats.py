"""Joint statistics of mid-circuit records: ``cudaq.run`` is correct, while
``cudaq.sample(..., explicit_measurements=True)`` returns impossible records (0.16).

The circuit and the exact distribution come from
``benchmarks/mid_circuit_probe_stats.py`` (derived by branch enumeration, not hard
coded). Both experiments use 100000 shots at a fixed seed and run once per module (about
15 s of CPU), so the tests that need them are marked slow; the exact-distribution tests
need no simulation and stay in the default run.
"""

import importlib.util
import json
from pathlib import Path

import cudaq
import numpy as np
import pytest

BENCH = Path(__file__).resolve().parent.parent / "benchmarks"


def load():
    import sys

    sys.path.insert(0, str(BENCH))
    spec = importlib.util.spec_from_file_location(
        "mid_circuit_probe_stats", BENCH / "mid_circuit_probe_stats.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return load()


@pytest.fixture(scope="module")
def stats(mod):
    out = mod.run_stats()
    cudaq.reset_target()
    return out


# ------------------------------------------------------------------ exact derivation


def test_exact_distribution_is_derived_from_the_circuit(mod) -> None:
    dist = mod.exact_record_distribution(mod.A, mod.B, mod.T)
    assert sum(dist.values()) == pytest.approx(1.0, abs=1e-15)
    assert mod.impossible_mass(dist) == pytest.approx(0.0, abs=1e-15)
    # out = 0 forces q0 = 0, the anc is never rotated: joint all-zero = cos^2(a/2)
    assert dist[(0, 0, 0)] == pytest.approx(np.cos(mod.A / 2) ** 2, abs=1e-15)
    # q0 = 1 (probability sin^2(a/2)) always reads out = 1
    out1 = sum(p for (_, _, out), p in dist.items() if out == 1)
    assert out1 == pytest.approx(np.sin(mod.A / 2) ** 2, abs=1e-15)
    # first anc outcome m1 = 1 only with q0 = 1: P = sin^2(a/2) sin^2(b/2)
    m1 = sum(p for (m, _, _), p in dist.items() if m == 1)
    assert m1 == pytest.approx(
        np.sin(mod.A / 2) ** 2 * np.sin(mod.B / 2) ** 2, abs=1e-15
    )


def test_exact_distribution_for_other_angles(mod) -> None:
    for a, b, t in [(0.3, 2.0, 1.1), (2.5, 0.4, 0.2)]:
        dist = mod.exact_record_distribution(a, b, t)
        assert dist[(0, 0, 0)] == pytest.approx(np.cos(a / 2) ** 2, abs=1e-15)
        assert mod.impossible_mass(dist) == pytest.approx(0.0, abs=1e-15)


# ------------------------------------------------------------------ the two readouts


@pytest.mark.slow
def test_run_has_no_impossible_records_and_the_right_joint_frequency(stats) -> None:
    run = stats["run"]
    assert run["shots"] == 100_000
    assert run["impossible_fraction"] == 0.0
    assert run["joint_all_zero_exact"] == pytest.approx(run["cos2_a_over_2"], abs=1e-15)
    assert (
        abs(run["joint_all_zero"] - run["joint_all_zero_exact"])
        < 4 * run["binomial_sigma"]
    )


@pytest.mark.slow
def test_explicit_measurements_sample_returns_impossible_records(stats) -> None:
    """Documents a CUDA-Q 0.16 defect (about 8% of the records). The repository pins
    ``cudaq<0.17``; if a future release fixes it this test fails and the README and
    CLAUDE.md statements about ``explicit_measurements`` must be revisited."""
    if not cudaq.__version__.split("Version")[-1].strip().startswith("0.16"):
        pytest.skip("the finding is for CUDA-Q 0.16")
    ex = stats["sample_explicit_measurements"]
    assert ex["impossible_fraction"] > 0.05
    assert (
        abs(ex["joint_all_zero"] - ex["joint_all_zero_exact"])
        > 4 * ex["binomial_sigma"]
    )


@pytest.mark.slow
def test_stats_json_is_written_with_metadata(stats, mod, tmp_path: Path) -> None:
    path = mod.write_stats(stats, tmp_path / "out" / "mid_circuit_probe_stats.json")
    data = json.loads(path.read_text())
    assert {
        "metadata",
        "params",
        "exact",
        "run",
        "sample_explicit_measurements",
    } <= set(data)
    assert data["metadata"]["cudaq_version"] and "git_commit" in data["metadata"]
    assert data["exact_impossible_mass"] == pytest.approx(0.0, abs=1e-15)
