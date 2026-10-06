"""Gate-level spectral advection vs classical.evolve_spectral (nu = 0), qpp-cpu fp64."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde import classical
from cudaq_pde.solvers import spectral

TOL = 1e-12
NS = range(2, 9)
CONVENTIONS = ["zero", "negative"]


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def _real_input(n: int, seed: int) -> np.ndarray:
    u = np.random.default_rng(seed).standard_normal(2**n)
    assert abs(np.fft.fft(u)[2 ** (n - 1)]) > 1e-3  # has Nyquist content
    return u


def _complex_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("swaps", [False, True])
@pytest.mark.parametrize("nyquist", CONVENTIONS)
@pytest.mark.parametrize("T", [0.3, 2.9])
@pytest.mark.parametrize("L", [classical.TWO_PI, 3.7])
@pytest.mark.parametrize("c", [-0.7, 1.3])
@pytest.mark.parametrize("n", NS)
def test_complex_state_matches_classical(
    n: int, c: float, L: float, T: float, nyquist: str, swaps: bool
) -> None:
    psi = _complex_state(n, 1000 + n)
    out = spectral.evolve_state(psi, c, L, T, nyquist=nyquist, swaps=swaps)
    ref = classical.evolve_spectral(psi, T, c, 0.0, L, nyquist=nyquist)
    np.testing.assert_allclose(out, ref, atol=TOL)


@pytest.mark.parametrize("swaps", [False, True])
@pytest.mark.parametrize("nyquist", CONVENTIONS)
@pytest.mark.parametrize("T", [0.3, 2.9])
@pytest.mark.parametrize("L", [classical.TWO_PI, 3.7])
@pytest.mark.parametrize("c", [-0.7, 1.3])
@pytest.mark.parametrize("n", NS)
def test_real_input_matches_classical(
    n: int, c: float, L: float, T: float, nyquist: str, swaps: bool
) -> None:
    u0 = 2.5 * _real_input(n, 2000 + n)
    out = spectral.evolve(u0, c, L, T, nyquist=nyquist, swaps=swaps)
    ref = classical.evolve_spectral(u0, T, c, 0.0, L, nyquist=nyquist)
    assert out.dtype == (np.float64 if nyquist == "zero" else np.complex128)
    np.testing.assert_allclose(out, ref, atol=TOL)


@pytest.mark.parametrize("swaps", [False, True])
@pytest.mark.parametrize("n", [2, 4, 6])
def test_zero_convention_real_input_stays_real(n: int, swaps: bool) -> None:
    u0 = _real_input(n, 3000 + n)
    psi = u0 / np.linalg.norm(u0)
    out = spectral.evolve_state(psi, 1.1, 3.7, 1.7, nyquist="zero", swaps=swaps)
    assert np.max(np.abs(out.imag)) < 1e-14


@pytest.mark.parametrize("nyquist", CONVENTIONS)
@pytest.mark.parametrize("n", NS)
def test_swap_free_equals_with_swaps(n: int, nyquist: str) -> None:
    psi = _complex_state(n, 4000 + n)
    a = spectral.evolve_state(psi, 0.9, 3.7, 1.9, nyquist=nyquist, swaps=False)
    b = spectral.evolve_state(psi, 0.9, 3.7, 1.9, nyquist=nyquist, swaps=True)
    np.testing.assert_allclose(a, b, atol=1e-13)


@pytest.mark.parametrize("swaps", [False, True])
@pytest.mark.parametrize("nyquist", CONVENTIONS)
@pytest.mark.parametrize("n", [3, 5, 7])
def test_fast_forward_equals_repeated_steps(n: int, nyquist: str, swaps: bool) -> None:
    m, dt, c, L = 7, 0.23, 1.3, 3.7
    psi = _complex_state(n, 5000 + n)
    stepped = spectral.evolve_state(
        psi, c, L, dt, steps=m, nyquist=nyquist, swaps=swaps
    )
    one_shot = spectral.evolve_state(
        psi, c, L, m * dt, steps=1, nyquist=nyquist, swaps=swaps
    )
    ref = classical.evolve_spectral(psi, m * dt, c, 0.0, L, nyquist=nyquist)
    np.testing.assert_allclose(stepped, one_shot, atol=TOL)
    np.testing.assert_allclose(one_shot, ref, atol=TOL)


def test_zero_steps_is_identity() -> None:
    psi = _complex_state(4, 6000)
    out = spectral.evolve_state(psi, 1.0, steps=0)
    np.testing.assert_allclose(out, psi, atol=1e-14)


def test_rejects_unnormalized_state() -> None:
    with pytest.raises(ValueError):
        spectral.evolve_state(2.0 * _complex_state(3, 7000), 1.0)
