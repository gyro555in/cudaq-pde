"""Smoke tests for the examples: they run, write outputs, and meet the error
thresholds already used in the solver tests (small settings, qpp-cpu fp64)."""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path

import cudaq
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
META_KEYS = {"cudaq_version", "target", "python", "numpy", "scipy", "git_commit"}


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    yield
    cudaq.reset_target()


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spectral_example(tmp_path: Path, capsys) -> None:
    ex = load("advection_spectral")
    out = ex.run(n=4, T=0.5, outdir=tmp_path)
    assert (tmp_path / "advection_spectral.png").stat().st_size > 1000
    saved = json.loads((tmp_path / "advection_spectral.json").read_text())
    assert META_KEYS <= set(saved["metadata"])
    err = out["errors"]
    assert err["circuit_vs_evolve_spectral_max_abs"] < 1e-12
    assert err["swaps_vs_swap_free_max_abs"] < 1e-13
    assert err["stepped_vs_one_shot_max_abs"] < 1e-12
    assert set(out["resources_native_per_step"]) == {"swaps=False", "swaps=True"}
    assert "circuit vs evolve_spectral" in capsys.readouterr().out


@pytest.mark.parametrize("nyquist", ["zero", "negative"])
def test_spectral_example_cli(tmp_path: Path, nyquist: str) -> None:
    ex = load("advection_spectral")
    out = ex.main(
        ["--n", "3", "--T", "0.7", "--nyquist", nyquist, "--outdir", str(tmp_path)]
    )
    assert out["params"]["nyquist"] == nyquist
    assert out["errors"]["circuit_vs_evolve_spectral_max_abs"] < 1e-12


def test_variational_example(tmp_path: Path, capsys) -> None:
    ex = load("advection_diffusion_variational")
    out = ex.run(n=3, steps=3, outdir=tmp_path)
    assert (tmp_path / "advection_diffusion_variational.png").stat().st_size > 1000
    saved = json.loads((tmp_path / "advection_diffusion_variational.json").read_text())
    assert META_KEYS <= set(saved["metadata"])
    per = out["per_step"]
    assert max(per["infidelity_vs_fd_euler"]) < 1e-9  # CP4 threshold, n = 3
    assert max(per["l2_vs_fd_euler"]) < 1e-4
    final = out["final"]
    # discretization error is reported separately and dominates the optimizer's
    assert final["discretization_error_l2"] > 10 * final["optimization_error_l2"]
    text = capsys.readouterr().out
    assert "optimization error" in text and "discretization error" in text


def test_variational_example_requires_verified_depth(tmp_path: Path) -> None:
    ex = load("advection_diffusion_variational")
    with pytest.raises(SystemExit):
        ex.run(n=5, steps=1, outdir=tmp_path)
