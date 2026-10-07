"""Resource counts: closed forms, real kernel vs builder mirror, lowering."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde.classical import TWO_PI
from cudaq_pde.solvers import spectral_resources as sr
from cudaq_pde.solvers.spectral import phase_angles
from cudaq_pde.solvers.spectral_kernels import spectral_advection_circuit


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


@pytest.mark.parametrize("swaps", [False, True])
@pytest.mark.parametrize("nyquist", ["negative", "zero"])
@pytest.mark.parametrize("n", range(2, 11))
def test_native_counts_match_closed_form(n: int, nyquist: str, swaps: bool) -> None:
    got = sr.native_counts(n, nyquist, swaps)["by_gate"]
    assert got == sr.expected_native_counts(n, nyquist, swaps)


@pytest.mark.parametrize("n", [3, 6, 9])
def test_swap_free_circuit_has_no_swaps_and_negative_has_no_two_qubit_phases(
    n: int,
) -> None:
    nat = sr.native_counts(n, "negative", False)
    assert "swap" not in nat["by_gate"]
    assert nat["two_qubit_gates"] == n * (n - 1)  # only the QFT controlled phases
    assert nat["total_gates"] == n * n + 2 * n


@pytest.mark.parametrize("n", range(2, 9))
def test_swaps_add_exactly_the_bit_reversals(n: int) -> None:
    a = sr.native_counts(n, "negative", False)
    b = sr.native_counts(n, "negative", True)
    assert b["by_gate"]["swap"] == 2 * (n // 2)
    assert b["total_gates"] - a["total_gates"] == 2 * (n // 2)


@pytest.mark.parametrize("swaps", [False, True])
@pytest.mark.parametrize("nyquist", ["negative", "zero"])
@pytest.mark.parametrize("n", [2, 3, 5])
def test_lowered_cx_matches_construction(n: int, nyquist: str, swaps: bool) -> None:
    low = sr.lowered_counts(n, nyquist, swaps)
    cx_expected = 2 * n * (n - 1) + (6 * (n // 2) if swaps else 0)
    if nyquist == "negative":
        assert low["cx"] == cx_expected  # cr1 = 2 CX, swap = 3 CX
        assert low["qubits"] == n
    else:
        assert low["cx"] > cx_expected  # plus the multi-controlled phase


@pytest.mark.parametrize("swaps", [False, True])
def test_mirror_has_same_output_state_as_real_kernel(swaps: bool) -> None:
    n = 3
    angles, phi = phase_angles(n, 1.0, TWO_PI, 0.1)
    real = np.array(
        cudaq.get_state(spectral_advection_circuit, n, angles, phi, True, swaps, 1)
    )
    mirror = np.array(cudaq.get_state(sr.build_mirror(n, "zero", swaps)))
    np.testing.assert_allclose(mirror, real, atol=1e-12)
    assert np.max(np.abs(real - np.eye(2**n)[0])) > 1e-3  # circuit is non-trivial


def test_qasm_stats_parser() -> None:
    qasm = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
        "h q[0];\ncx q[0], q[1];\nrz(0.5) q[1];\n"
    )
    assert sr.qasm_stats(qasm) == {
        "total_gates": 3,
        "cx": 1,
        "depth": 3,
        "qubits": 2,
    }
