import numpy as np
import pytest

from cudaq_pde import classical

TOL = 1e-12
CONVENTIONS = ["zero", "negative"]


def _generic(n: int, seed: int, complex_valued: bool = False) -> np.ndarray:
    """Random data with full spectral content, including the Nyquist mode."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(2**n)
    if complex_valued:
        u = u + 1j * rng.standard_normal(2**n)
    assert abs(np.fft.fft(u)[2 ** (n - 1)]) > 1e-3
    return u


@pytest.mark.parametrize("nyquist", CONVENTIONS)
@pytest.mark.parametrize("n", [3, 5, 7])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_single_mode_matches_analytic(n: int, m: int, nyquist: str) -> None:
    L, c, nu, t = 2.0, 0.7, 0.05, 0.9
    x = classical.grid(n, L)
    k = 2 * np.pi * m / L
    u0 = np.sin(k * x)
    exact = np.exp(-nu * k**2 * t) * np.sin(k * (x - c * t))
    u = classical.evolve_spectral(u0, t, c, nu, L, nyquist=nyquist)
    np.testing.assert_allclose(u, exact, atol=TOL)


@pytest.mark.parametrize("n", [2, 4])
def test_wavenumber_layout_and_L_scaling(n: int) -> None:
    N = 2**n
    m = np.concatenate([np.arange(N // 2), np.arange(-N // 2, 0)])
    np.testing.assert_array_equal(classical.wavenumbers(n), m)
    for L in (1.0, 4.0 * np.pi, 10.0):
        np.testing.assert_allclose(
            classical.wavenumbers(n, L), 2 * np.pi / L * m, rtol=1e-15
        )


def test_advection_wavenumbers_conventions() -> None:
    neg = classical.advection_wavenumbers(3, nyquist="negative")
    np.testing.assert_array_equal(neg, [0, 1, 2, 3, -4, -3, -2, -1])
    zero = classical.advection_wavenumbers(3, nyquist="zero")
    np.testing.assert_array_equal(zero, [0, 1, 2, 3, 0, -3, -2, -1])
    with pytest.raises(ValueError):
        classical.advection_wavenumbers(3, nyquist="positive")


def test_grid_excludes_endpoint() -> None:
    x = classical.grid(3, L=8.0)
    np.testing.assert_allclose(x, np.arange(8.0))


@pytest.mark.parametrize("nyquist", CONVENTIONS)
def test_mean_conserved(nyquist: str) -> None:
    u0 = _generic(5, 1)
    u = classical.evolve_spectral(u0, 1.3, 0.4, 0.1, nyquist=nyquist)
    assert abs(u.mean() - u0.mean()) < TOL


@pytest.mark.parametrize("complex_valued", [False, True])
@pytest.mark.parametrize("L", [classical.TWO_PI, 3.7])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("nyquist", CONVENTIONS)
def test_unitary_without_diffusion(
    nyquist: str, n: int, L: float, complex_valued: bool
) -> None:
    u0 = _generic(n, 10 + n, complex_valued)
    u = classical.evolve_spectral(u0, 2.1, 0.9, 0.0, L, nyquist=nyquist)
    assert abs(np.linalg.norm(u) - np.linalg.norm(u0)) < TOL


@pytest.mark.parametrize("complex_valued", [False, True])
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("nyquist", CONVENTIONS)
def test_reversible_without_diffusion(
    nyquist: str, n: int, complex_valued: bool
) -> None:
    u0 = _generic(n, 20 + n, complex_valued)
    fwd = classical.evolve_spectral(u0, 1.7, 0.8, 0.0, nyquist=nyquist)
    back = classical.evolve_spectral(fwd, -1.7, 0.8, 0.0, nyquist=nyquist)
    np.testing.assert_allclose(back, u0, atol=TOL)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_zero_convention_real_input_stays_real(n: int) -> None:
    u0 = _generic(n, 30 + n)
    out = classical.evolve_spectral(u0, 1.9, 0.8, 0.03, nyquist="zero")
    assert out.dtype == np.float64
    # The returned array is real by construction, so also check the underlying
    # complex evolution of the same real data has no imaginary part.
    raw = classical.evolve_spectral(
        u0.astype(np.complex128), 1.9, 0.8, 0.03, nyquist="zero"
    )
    assert np.max(np.abs(raw.imag)) < 1e-14
    np.testing.assert_allclose(raw.real, out, atol=1e-14)


def test_negative_convention_real_input_becomes_complex() -> None:
    u0 = _generic(4, 40)
    out = classical.evolve_spectral(u0, 0.3, 1.0, 0.0, nyquist="negative")
    assert np.iscomplexobj(out)
    assert np.max(np.abs(out.imag)) > 1e-3


@pytest.mark.parametrize("L", [classical.TWO_PI, 5.0])
def test_pure_nyquist_mode_by_convention(L: float) -> None:
    n, c, nu, t = 4, 0.8, 0.01, 0.37
    N = 2**n
    u0 = np.cos(np.pi * np.arange(N))  # (-1)^j, the Nyquist mode
    k_nyq = np.pi * N / L
    decay = np.exp(-nu * k_nyq**2 * t)
    zero = classical.evolve_spectral(u0, t, c, nu, L, nyquist="zero")
    np.testing.assert_allclose(zero, decay * u0, atol=TOL)
    neg = classical.evolve_spectral(u0, t, c, nu, L, nyquist="negative")
    np.testing.assert_allclose(neg, decay * np.exp(1j * c * k_nyq * t) * u0, atol=TOL)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_conventions_agree_without_nyquist_content(n: int) -> None:
    spec = np.fft.fft(_generic(n, 50 + n))
    spec[2 ** (n - 1)] = 0
    u0 = np.fft.ifft(spec).real
    a = classical.evolve_spectral(u0, 1.1, 0.7, 0.02, nyquist="zero")
    b = classical.evolve_spectral(u0, 1.1, 0.7, 0.02, nyquist="negative")
    np.testing.assert_allclose(b.imag, 0, atol=TOL)
    np.testing.assert_allclose(b.real, a, atol=TOL)


@pytest.mark.parametrize("nyquist", CONVENTIONS)
def test_semigroup_property(nyquist: str) -> None:
    u0 = _generic(4, 4, complex_valued=True)
    t1, t2, c, nu = 0.3, 0.55, 0.6, 0.02
    two_steps = classical.evolve_spectral(
        classical.evolve_spectral(u0, t1, c, nu, nyquist=nyquist),
        t2,
        c,
        nu,
        nyquist=nyquist,
    )
    one_step = classical.evolve_spectral(u0, t1 + t2, c, nu, nyquist=nyquist)
    np.testing.assert_allclose(two_steps, one_step, atol=TOL)


@pytest.mark.parametrize("nyquist", CONVENTIONS)
def test_l2_norm_decays_with_diffusion(nyquist: str) -> None:
    u0 = _generic(5, 5)
    norms = [
        np.linalg.norm(classical.evolve_spectral(u0, t, 0.5, 0.1, nyquist=nyquist))
        for t in (0.0, 0.2, 0.4, 0.8)
    ]
    assert all(b < a for a, b in zip(norms, norms[1:], strict=False))


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
    with pytest.raises(ValueError):
        classical.evolve_spectral(np.ones(8), 0.1, 1.0, nyquist="positive")


@pytest.mark.parametrize("L", [classical.TWO_PI, 3.7])
@pytest.mark.parametrize("n", [2, 3, 4, 5])
@pytest.mark.parametrize("nyquist", CONVENTIONS)
def test_every_mode_incl_nyquist_analytic(nyquist: str, n: int, L: float) -> None:
    """Each Fourier mode gets exp(-i c k t - nu k^2 t); oracle uses integers m."""
    N, c, nu, t = 2**n, 0.8, 0.04, 0.37
    x = classical.grid(n, L)
    for m in range(-N // 2, N // 2):  # includes the Nyquist mode m = -N/2
        k = 2 * np.pi * m / L
        k_adv = 0.0 if (nyquist == "zero" and m == -N // 2) else k
        u0 = np.exp(1j * k * x)
        expected = np.exp(-1j * c * k_adv * t - nu * k**2 * t) * u0
        u = classical.evolve_spectral(u0, t, c, nu, L, nyquist=nyquist)
        np.testing.assert_allclose(u, expected, atol=TOL)
