"""increment/decrement (QFT form) equal the CP3 spectral circuit with c * t = dx.

STATEVECTOR CHECKS (test only). The shift is advection by one grid cell, so the
kernel must coincide with ``spectral.evolve_state`` (the CP3 circuit) for
``c * dt = +-dx = +-L / N`` with the "negative" Nyquist convention, on every basis
state, and with the numpy-defined roll.
"""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde import classical
from cudaq_pde.solvers import spectral
from test_shift_kernels import shift_matrix

NS = range(2, 7)
TOL = 1e-12


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def spectral_matrix(n: int, c_dt: float, nyquist: str, swaps: bool = False):
    """Matrix of the CP3 circuit for a shift of c * dt (all basis states)."""
    N, L = 2**n, 6.5  # L is arbitrary: the shift by one cell is independent of it
    cols = [
        spectral.evolve_state(
            np.eye(N)[j].astype(complex),
            1.0,
            L,
            c_dt * L / N,
            nyquist=nyquist,
            swaps=swaps,
        )
        for j in range(N)
    ]
    return np.array(cols).T


@pytest.mark.parametrize("n", NS)
def test_increment_equals_spectral_circuit_with_ct_equal_dx(n: int) -> None:
    inc = shift_matrix(n, 0)  # increment, QFT form
    np.testing.assert_allclose(inc, spectral_matrix(n, +1.0, "negative"), atol=TOL)
    np.testing.assert_allclose(
        inc, spectral_matrix(n, +1.0, "negative", swaps=True), atol=TOL
    )
    for j in range(2**n):  # and the numpy definition
        np.testing.assert_allclose(
            inc[:, j], classical.increment(np.eye(2**n)[j]), atol=TOL
        )


@pytest.mark.parametrize("n", NS)
def test_decrement_equals_spectral_circuit_with_ct_equal_minus_dx(n: int) -> None:
    dec = shift_matrix(n, 1)  # decrement, QFT form
    np.testing.assert_allclose(dec, spectral_matrix(n, -1.0, "negative"), atol=TOL)
    np.testing.assert_allclose(dec, shift_matrix(n, 0).T, atol=TOL)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_zero_nyquist_convention_is_not_a_permutation(n: int) -> None:
    """Negative control: with "zero" the Nyquist factor is 1 not -1, so the circuit
    is unitary but no longer the cyclic shift."""
    zero = spectral_matrix(n, +1.0, "zero")
    assert np.max(np.abs(zero - shift_matrix(n, 0))) > 1e-3
    np.testing.assert_allclose(zero.conj().T @ zero, np.eye(2**n), atol=TOL)


def test_single_phase_layer_implementation() -> None:
    """The shift kernels contain no phase code of their own."""
    import inspect

    from cudaq_pde.solvers import lcu_kernels, shift_kernels

    shift_src = inspect.getsource(shift_kernels)
    lcu_src = inspect.getsource(lcu_kernels)
    assert "phase_layer_mirrored(q, angles)" in shift_src
    assert "controlled_phase_layer_mirrored" in lcu_src
    assert "r1(" not in shift_src and "r1.ctrl(" not in lcu_src
    assert "TWO_PI" not in shift_src and "TWO_PI" not in lcu_src
