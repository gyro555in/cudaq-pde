"""Structural guard: the hardware path never reads a state vector.

Scans the AST of every module in the package. ``cudaq.get_state`` may be called only in
the explicitly SIMULATOR-ONLY modules listed below, and ``cudaq.sample`` only in the
sampling wrapper.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "cudaq_pde"

# modules that read a state vector, each documented as SIMULATOR-ONLY
SIMULATOR_ONLY = {
    "solvers/variational.py",  # simulator_cost_and_state
    "solvers/spectral.py",  # evolve_state
}
SAMPLING_MODULE = "solvers/sampling.py"
HARDWARE_PATH = [
    "solvers/measurable.py",
    "solvers/measurable_kernels.py",
    "solvers/measurable_resources.py",
    "solvers/cost_statistics.py",
    "solvers/sampling.py",
    "solvers/lcu.py",
    "solvers/lcu_kernels.py",
    "solvers/shift_kernels.py",
    "solvers/spectral_measure.py",
    "solvers/spectral_kernels.py",
    "solvers/variational_kernels.py",
]


def called_names(path: Path) -> set[str]:
    """Names of functions/attributes called (or imported) in a module."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                names.add(f.attr)
            elif isinstance(f, ast.Name):
                names.add(f.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names


def all_modules() -> list[str]:
    return sorted(str(p.relative_to(SRC)) for p in SRC.rglob("*.py"))


@pytest.mark.parametrize("module", HARDWARE_PATH)
def test_hardware_path_module_has_no_state_readout(module: str) -> None:
    names = called_names(SRC / module)
    assert "get_state" not in names and "get_state_async" not in names, module


def test_only_the_documented_simulator_modules_call_get_state() -> None:
    callers = {m for m in all_modules() if "get_state" in called_names(SRC / m)}
    assert callers == SIMULATOR_ONLY


@pytest.mark.parametrize("module", sorted(SIMULATOR_ONLY))
def test_simulator_only_modules_say_so(module: str) -> None:
    assert "SIMULATOR-ONLY" in (SRC / module).read_text()


def test_cudaq_sample_has_a_single_call_site() -> None:
    callers = {
        m
        for m in all_modules()
        for node in ast.walk(ast.parse((SRC / m).read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("sample", "sample_async", "observe")
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cudaq"
    }
    assert callers == {SAMPLING_MODULE}


def test_the_scanner_would_catch_a_violation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("import cudaq\n\ndef f(k):\n    return cudaq.get_state(k)\n")
    assert "get_state" in called_names(bad)
    sneaky = tmp_path / "sneaky.py"
    sneaky.write_text("from cudaq import get_state\n")
    assert "get_state" in called_names(sneaky)
