"""Spectral advection on the nvidia fp64 target. Run with --run-gpu on a GPU node."""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde import classical
from cudaq_pde.solvers import spectral

pytestmark = pytest.mark.gpu

TOL = 1e-12


@pytest.fixture(autouse=True)
def gpu_fp64_target() -> Iterator[None]:
    cudaq.set_target("nvidia", option="fp64")
    yield
    cudaq.reset_target()


def _complex_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(2**n) + 1j * rng.standard_normal(2**n)
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("swaps", [False, True])
@pytest.mark.parametrize("nyquist", ["zero", "negative"])
@pytest.mark.parametrize("T", [0.3, 2.9])
@pytest.mark.parametrize("L", [classical.TWO_PI, 3.7])
@pytest.mark.parametrize("c", [-0.7, 1.3])
@pytest.mark.parametrize("n", range(2, 7))
def test_matches_classical_on_gpu(
    n: int, c: float, L: float, T: float, nyquist: str, swaps: bool
) -> None:
    psi = _complex_state(n, 1000 + n)
    out = spectral.evolve_state(psi, c, L, T, nyquist=nyquist, swaps=swaps)
    ref = classical.evolve_spectral(psi, T, c, 0.0, L, nyquist=nyquist)
    np.testing.assert_allclose(out, ref, atol=TOL)


@pytest.mark.parametrize("nyquist", ["zero", "negative"])
def test_fast_forward_on_gpu(nyquist: str) -> None:
    n, m, dt, c, L = 5, 7, 0.23, 1.3, 3.7
    psi = _complex_state(n, 5000 + n)
    stepped = spectral.evolve_state(psi, c, L, dt, steps=m, nyquist=nyquist)
    one_shot = spectral.evolve_state(psi, c, L, m * dt, nyquist=nyquist)
    np.testing.assert_allclose(stepped, one_shot, atol=TOL)
