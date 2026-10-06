"""Explicit-Euler finite-difference reference: correctness, stability, errors."""

import numpy as np
import pytest
from scipy.linalg import expm

from cudaq_pde import classical
from fd_helpers import dense_L

TOL = 1e-13


def _smooth(n: int, L: float = classical.TWO_PI) -> np.ndarray:
    x = classical.grid(n, L)
    k = 2 * np.pi / L
    return 1.0 + 0.5 * np.sin(k * x) + 0.25 * np.cos(2 * k * x)


@pytest.mark.parametrize("L", [classical.TWO_PI, 3.7])
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_matches_dense_matrix_power(n: int, L: float) -> None:
    c, nu = 0.8, 0.4
    dt = 0.5 * classical.fd_euler_dt_max(n, c, nu, L)
    u0 = np.random.default_rng(n).standard_normal(2**n)
    traj = classical.evolve_fd_euler(u0, dt, 6, c, nu, L)
    step = np.eye(2**n) + dt * dense_L(n, c, nu, L)
    expected = u0.copy()
    assert traj.shape == (7, 2**n)
    np.testing.assert_array_equal(traj[0], u0)
    for s in range(1, 7):
        expected = step @ expected
        np.testing.assert_allclose(traj[s], expected, atol=TOL)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_amplification_is_largest_eigenvalue_modulus(n: int) -> None:
    c, nu, L, dt = 1.0, 0.4, 3.7, 0.05
    step = np.eye(2**n) + dt * dense_L(n, c, nu, L)
    eig = np.max(np.abs(np.linalg.eigvals(step)))
    assert classical.fd_euler_amplification(n, dt, c, nu, L) == pytest.approx(
        eig, abs=1e-12
    )


def test_sum_is_conserved() -> None:
    u0 = _smooth(4)
    traj = classical.evolve_fd_euler(u0, 0.05, 20, 1.0, 0.4)
    np.testing.assert_allclose(traj.sum(axis=1), u0.sum(), atol=1e-12)


@pytest.mark.parametrize("c", [0.0, 0.5, 1.0, 3.0])
@pytest.mark.parametrize("nu", [0.05, 0.4, 2.0])
@pytest.mark.parametrize("L", [classical.TWO_PI, 3.7])
@pytest.mark.parametrize("n", [3, 4, 6])
def test_closed_form_limit(n: int, L: float, nu: float, c: float) -> None:
    """dt_max is stable for every grid; clearly beyond it the scheme is unstable.

    On a finite grid the discrete wavenumbers never reach theta -> 0, so the
    grid stays stable slightly beyond the ``b^2 <= a`` part of the closed form
    (the limit is exact only as N -> infinity), while the ``a <= 1`` part is
    exact for even N (theta = pi is present). Hence "just inside is stable" is
    asserted everywhere, and instability is asserted at 10x the limit.
    """
    dt_max = classical.fd_euler_dt_max(n, c, nu, L)
    assert dt_max > 0
    assert classical.fd_euler_amplification(n, dt_max, c, nu, L) <= 1 + 1e-12
    assert classical.fd_euler_amplification(n, 10.0 * dt_max, c, nu, L) > 1 + 1e-6


@pytest.mark.parametrize("n", [3, 4, 6])
def test_diffusion_limit_is_exact_when_it_is_the_binding_one(n: int) -> None:
    nu, c, L = 0.4, 0.1, classical.TWO_PI  # tiny c: h^2/(2 nu) binds
    dt_max = classical.fd_euler_dt_max(n, c, nu, L)
    assert dt_max == pytest.approx((L / 2**n) ** 2 / (2 * nu))
    assert classical.fd_euler_amplification(n, 1.001 * dt_max, c, nu, L) > 1 + 1e-9


def test_unstable_step_raises() -> None:
    u0 = _smooth(3)
    dt_max = classical.fd_euler_dt_max(3, 1.0, 0.4)
    classical.evolve_fd_euler(u0, 0.99 * dt_max, 3, 1.0, 0.4)
    with pytest.raises(ValueError, match="unstable"):
        classical.evolve_fd_euler(u0, 10.0 * dt_max, 3, 1.0, 0.4)


def test_pure_advection_has_no_stable_step() -> None:
    assert classical.fd_euler_dt_max(4, 1.0, 0.0) == 0.0
    with pytest.raises(ValueError, match="unstable"):
        classical.evolve_fd_euler(_smooth(4), 1e-4, 2, 1.0, 0.0)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        classical.evolve_fd_euler(np.ones(6), 0.01, 1, 1.0, 0.4)
    with pytest.raises(ValueError):
        classical.evolve_fd_euler(np.ones(8), 0.0, 1, 1.0, 0.4)
    with pytest.raises(ValueError):
        classical.evolve_fd_euler(np.ones(8), 0.01, -1, 1.0, 0.4)
    with pytest.raises(ValueError):
        classical.evolve_fd_euler(np.ones(8), 0.01, 1, 1.0, -0.4)
    with pytest.raises(ValueError):
        classical.evolve_fd_euler(np.array([1.0, np.nan, 0, 0]), 0.01, 1, 1.0, 0.4)


def _discretization_errors(n: int, T: float, dt: float, c: float, nu: float):
    """(space error, time error, total error) in the relative L2 norm."""
    L = classical.TWO_PI
    u0 = _smooth(n, L)
    steps = round(T / dt)
    fd = classical.evolve_fd_euler(u0, dt, steps, c, nu, L)[-1]
    semi = expm(T * dense_L(n, c, nu, L)) @ u0  # exact in time, FD in space
    spec = classical.evolve_spectral(u0, T, c, nu, L)  # exact in both
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)  # noqa: E731
    return rel(semi, spec), rel(fd, semi), rel(fd, spec)


def test_space_error_is_second_order_and_shrinks_with_n() -> None:
    errs = [_discretization_errors(n, 0.4, 0.4 / 400, 1.0, 0.4)[0] for n in (4, 5, 6)]
    assert errs[0] > errs[1] > errs[2]
    for coarse, fine in zip(errs, errs[1:], strict=False):
        assert 3.0 < coarse / fine < 5.0  # ~4 for a second-order scheme


def test_time_error_is_first_order_in_dt() -> None:
    n, T = 4, 0.4
    e1 = _discretization_errors(n, T, T / 100, 1.0, 0.4)[1]
    e2 = _discretization_errors(n, T, T / 200, 1.0, 0.4)[1]
    assert 1.8 < e1 / e2 < 2.2  # ~2 for explicit Euler


def test_total_discretization_error_is_reported_and_small() -> None:
    n, T, dt = 5, 0.4, 0.001
    space, time, total = _discretization_errors(n, T, dt, 1.0, 0.4)
    print(f"n={n} T={T} dt={dt}: space {space:.2e} time {time:.2e} total {total:.2e}")
    assert total < 1e-2
