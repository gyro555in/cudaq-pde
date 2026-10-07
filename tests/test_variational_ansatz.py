"""Ansatz kernel vs an independent numpy gate simulation (qpp-cpu, fp64)."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde.solvers.ansatz_numpy import apply_cx, numpy_ansatz
from cudaq_pde.solvers.variational_kernels import hea_state

TOL = 1e-12


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_kernel_matches_numpy_simulation(n: int, depth: int) -> None:
    rng = np.random.default_rng(10 * n + depth)
    thetas = rng.uniform(-np.pi, np.pi, n * (depth + 1))
    state = np.array(cudaq.get_state(hea_state, n, thetas.tolist(), depth))
    assert state.dtype == np.complex128
    np.testing.assert_allclose(state, numpy_ansatz(n, thetas, depth), atol=TOL)
    assert np.max(np.abs(state.imag)) < 1e-15  # real amplitudes
    assert abs(np.linalg.norm(state) - 1.0) < 1e-14


def test_numpy_reference_cx_is_a_permutation() -> None:
    n = 3
    for control, target in [(0, 1), (1, 0), (0, 2), (2, 1)]:
        for j in range(2**n):
            basis = np.eye(2**n)[j]
            out = apply_cx(basis, n, control, target)
            bits = [(j >> k) & 1 for k in range(n)]
            if bits[control]:
                bits[target] ^= 1
            assert out[sum(b << k for k, b in enumerate(bits))] == 1.0


def test_zero_angles_give_zero_state() -> None:
    state = np.array(cudaq.get_state(hea_state, 3, [0.0] * 9, 2))
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1.0
    np.testing.assert_allclose(state, expected, atol=1e-15)


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("n", [3, 4])
def test_gate_counts_match_closed_form(n: int, depth: int) -> None:
    thetas = [0.1] * (n * (depth + 1))
    res = cudaq.estimate_resources(hea_state, n, thetas, depth)
    counts = dict(res.to_dict())
    assert counts.get("ry", 0) == n * (depth + 1)
    assert counts.get("cx", 0) == (n - 1) * depth
    assert sum(counts.values()) == n * (depth + 1) + (n - 1) * depth


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_adjoint_undoes_the_ansatz(n: int, depth: int) -> None:
    from ansatz_test_kernels import ansatz_then_adjoint

    thetas = np.random.default_rng(7 * n + depth).uniform(-3, 3, n * (depth + 1))
    state = np.array(cudaq.get_state(ansatz_then_adjoint, n, thetas.tolist(), depth))
    expected = np.zeros(2**n, dtype=complex)
    expected[0] = 1.0
    np.testing.assert_allclose(state, expected, atol=1e-12)


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("n", [3, 4])
def test_adjoint_amplitude_is_the_state_overlap(n: int, depth: int) -> None:
    """<0| A(other)^dagger A(theta) |0> equals <psi(other)|psi(theta)> (numpy)."""
    from ansatz_test_kernels import ansatz_with_other_adjoint

    rng = np.random.default_rng(n + 10 * depth)
    a, b = rng.uniform(-3, 3, (2, n * (depth + 1)))
    state = np.array(
        cudaq.get_state(ansatz_with_other_adjoint, n, a.tolist(), b.tolist(), depth)
    )
    overlap = numpy_ansatz(n, b, depth) @ numpy_ansatz(n, a, depth)
    assert state[0] == pytest.approx(overlap, abs=1e-12)
    assert abs(overlap) < 0.999  # the two states really differ
