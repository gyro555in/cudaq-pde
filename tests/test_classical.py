import numpy as np
import pytest

from cudaq_pde import classical

TOL = 1e-12


@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_single_mode_matches_analytic(n: int, m: int) -> None:
    L, c, nu, t = 2.0, 0.7, 0.05, 0.9
    x = classical.grid(n, L)
    k = 2 * np.pi * m / L
    u0 = np.sin(k * x)
    exact = np.exp(-nu * k**2 * t) * np.sin(k * (x - c * t))
    np.testing.assert_allclose(
        classical.evolve_spectral(u0, t, c, nu, L), exact, atol=TOL
    )


def test_wavenumber_layout() -> None:
    k = classical.wavenumbers(3)
    np.testing.assert_array_equal(k, [0, 1, 2, 3, -4, -3, -2, -1])
    k = classical.wavenumbers(2, L=4.0 * np.pi)
    np.testing.assert_allclose(k, [0, 0.5, -1.0, -0.5])


def test_grid_excludes_endpoint() -> None:
    x = classical.grid(3, L=8.0)
    np.testing.assert_allclose(x, np.arange(8.0))


def test_mean_conserved() -> None:
    rng = np.random.default_rng(1)
    u0 = rng.standard_normal(32)
    u = classical.evolve_spectral(u0, 1.3, 0.4, 0.1)
    assert abs(u.mean() - u0.mean()) < TOL


def _without_nyquist(u: np.ndarray) -> np.ndarray:
    """Drop the Nyquist mode, whose real-part projection is not unitary."""
    spec = np.fft.fft(u)
    spec[u.size // 2] = 0
    return np.fft.ifft(spec).real


def test_energy_conserved_without_diffusion() -> None:
    rng = np.random.default_rng(2)
    u0 = _without_nyquist(rng.standard_normal(32))
    u = classical.evolve_spectral(u0, 2.1, 0.9, 0.0)
    assert abs(np.linalg.norm(u) - np.linalg.norm(u0)) < TOL


def test_reversible_without_diffusion() -> None:
    rng = np.random.default_rng(3)
    u0 = _without_nyquist(rng.standard_normal(16))
    fwd = classical.evolve_spectral(u0, 1.7, 0.8, 0.0)
    back = classical.evolve_spectral(fwd, -1.7, 0.8, 0.0)
    np.testing.assert_allclose(back, u0, atol=TOL)


def test_semigroup_property() -> None:
    rng = np.random.default_rng(4)
    u0 = rng.standard_normal(16)
    t1, t2, c, nu = 0.3, 0.55, 0.6, 0.02
    two_steps = classical.evolve_spectral(
        classical.evolve_spectral(u0, t1, c, nu, real=False), t2, c, nu, real=False
    )
    one_step = classical.evolve_spectral(u0, t1 + t2, c, nu, real=False)
    np.testing.assert_allclose(two_steps, one_step, atol=TOL)


def test_l2_norm_decays_with_diffusion() -> None:
    rng = np.random.default_rng(5)
    u0 = rng.standard_normal(32)
    norms = [
        np.linalg.norm(classical.evolve_spectral(u0, t, 0.5, 0.1))
        for t in (0.0, 0.2, 0.4, 0.8)
    ]
    assert all(b < a for a, b in zip(norms, norms[1:], strict=False))


def test_nyquist_real_flag() -> None:
    n = 3
    u0 = np.cos(np.pi * np.arange(2**n))  # pure Nyquist mode (-1)^j
    raw = classical.evolve_spectral(u0, 0.3, 1.0, 0.0, real=False)
    assert np.max(np.abs(raw.imag)) > 1e-3
    real = classical.evolve_spectral(u0, 0.3, 1.0, 0.0, real=True)
    np.testing.assert_allclose(real, raw.real, atol=TOL)


@pytest.mark.parametrize("bad_size", [1, 3, 6, 12])
def test_rejects_non_power_of_two(bad_size: int) -> None:
    with pytest.raises(ValueError):
        classical.evolve_spectral(np.ones(bad_size), 0.1, 1.0)


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        classical.evolve_spectral(np.ones(8), 0.1, 1.0, nu=-0.1)
    with pytest.raises(ValueError):
        classical.evolve_spectral(np.array([1.0, np.nan, 0, 0]), 0.1, 1.0)
    with pytest.raises(ValueError):
        classical.evolve_spectral(np.ones((4, 4)), 0.1, 1.0)
