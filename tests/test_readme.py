"""The README cannot drift from the code: snippets run, the table is recomputed."""

import re
from collections.abc import Iterator
from pathlib import Path

import cudaq
import numpy as np
import pytest

from cudaq_pde.solvers import oneshot
from cudaq_pde.solvers import oneshot_resources as rr
from cudaq_pde.solvers import spectral_resources as sr

README = (Path(__file__).resolve().parent.parent / "README.md").read_text()


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    yield
    cudaq.reset_target()


def python_blocks() -> list[str]:
    return re.findall(r"```python\n(.*?)```", README, flags=re.DOTALL)


def test_two_quick_start_snippets_exist() -> None:
    assert len(python_blocks()) == 2


def test_spectral_snippet_runs_and_meets_its_claim() -> None:
    ns: dict = {}
    exec(compile(python_blocks()[0], "README spectral", "exec"), ns)
    assert np.max(np.abs(ns["u"] - ns["ref"])) < 1e-12


def test_variational_snippet_runs_and_meets_its_claim() -> None:
    ns: dict = {}
    exec(compile(python_blocks()[1], "README variational", "exec"), ns)
    assert ns["l2"][-1] < 1e-4  # CP4 threshold for n = 3
    assert int(ns["res"].nfev.sum()) > 0


def test_resource_table_matches_the_code() -> None:
    cudaq.set_target("qpp-cpu")
    rows = re.findall(
        r"^\| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$",
        README,
        flags=re.MULTILINE,
    )
    assert [int(r[0]) for r in rows] == [4, 6, 8, 10]
    for n, gates, two_q, low_cx, x_gates, x_cx, x_anc in (map(int, r) for r in rows):
        neg = sr.resource_report(n, "negative", False)
        zero = sr.resource_report(n, "zero", False)
        assert gates == neg["native"]["total_gates"] == n * n + 2 * n
        assert two_q == neg["native"]["two_qubit_gates"] == n * (n - 1)
        assert low_cx == neg["lowered"]["cx"] == 2 * n * (n - 1)
        assert x_gates == zero["native"]["total_gates"] - neg["native"]["total_gates"]
        assert x_cx == zero["lowered"]["cx"] - neg["lowered"]["cx"] == 12 * n - 22
        assert x_anc == zero["lowered"]["qubits"] - neg["lowered"]["qubits"] == n - 2


def test_oneshot_resource_table_matches_the_code() -> None:
    cudaq.set_target("qpp-cpu")
    rows = re.findall(
        r"^\| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$",
        README,
        flags=re.MULTILINE,
    )
    assert [int(r[0]) for r in rows] == [4, 6, 8, 10]
    sigma = 4.0  # grid cells, the value stated in the README
    for n, cx_a, cx_b, anc_b, cx_c, q in (map(int, r) for r in rows):
        a = 2 * np.pi**2 * sigma**2 / 4**n
        assert rr.lowered_counts("A", n, a)["cx"] == cx_a
        assert rr.lowered_counts("B2", n, a)["cx"] == cx_b
        assert rr.source_ancillas("B2", n) == anc_b
        plan = oneshot.gaussian_plan(n, a)
        assert plan.q == q
        assert rr.lowered_counts("C", n, a)["cx"] == cx_c


def test_swaps_claim_in_readme() -> None:
    cudaq.set_target("qpp-cpu")
    for n in (4, 7):
        a = sr.native_counts(n, "negative", False)["total_gates"]
        b = sr.native_counts(n, "negative", True)["total_gates"]
        assert b - a == 2 * (n // 2)


def test_no_em_dashes_in_docs() -> None:
    root = Path(__file__).resolve().parent.parent
    for name in ("README.md", "CHANGELOG.md", "CITATION.cff", "CLAUDE.md"):
        path = root / name
        if path.exists():
            assert "—" not in path.read_text(), name
