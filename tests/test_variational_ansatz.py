"""Ansatz kernel vs an independent numpy gate simulation (qpp-cpu, fp64)."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from ansatz_reference import apply_cx, numpy_ansatz
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
