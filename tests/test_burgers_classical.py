"""Classical Burgers references: Cole-Hopf and the pseudo-spectral integrator must agree
before any quantum code is built, plus the zero-mean (Galilean) handling.

Tolerances are derived, not fitted: the exact Cole-Hopf solution is converged to ~1e-15
(fine grid doubling), so any disagreement is the integrator's, whose size is measured by
its own N -> 2N difference; ``agree`` allows 3 times that plus a 1e-12 floor.
"""

import numpy as np
import pytest

from cudaq_pde import burgers as B
from cudaq_pde import classical

L = 2 * np.pi
XS = np.linspace(0, L, 41)[:-1] + 0.0137  # off-grid points
CASES = {
    "sin": np.sin,
    "asym": lambda x: np.sin(x) + 0.5 * np.sin(2 * x),
}


def agreement(f, nu, T):
    exact = B.exact_cole_hopf(XS, f, nu, T)
    u1 = B.evaluate_fourier(B.pseudospectral_burgers(f, nu, T, N=1024), XS)
    u2 = B.evaluate_fourier(B.pseudospectral_burgers(f, nu, T, N=2048), XS)
    err = float(np.max(np.abs(exact - u1)))
    conv = float(np.max(np.abs(u1 - u2)))
    return err, conv


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("nu", [0.5, 0.1])
def test_references_agree_on_smooth_cases(name: str, nu: float) -> None:
    err, conv = agreement(CASES[name], nu, 0.5)
    assert err <= 3 * conv + 1e-12, (err, conv)
    assert err < 1e-11  # and both are far below any quantum-side error


@pytest.mark.slow
@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("nu,T", [(0.5, 1.0), (0.1, 1.0), (0.05, 1.0), (0.02, 1.0)])
def test_references_agree_through_and_after_shock_formation(
    name: str, nu: float, T: float
) -> None:
    err, conv = agreement(CASES[name], nu, T)
    assert err <= 3 * conv + 1e-12, (err, conv)


def test_exact_cole_hopf_is_converged_in_its_fine_grid() -> None:
    a = B.exact_cole_hopf(XS, np.sin, 0.05, 1.0, n_fine=2048)
    b = B.exact_cole_hopf(XS, np.sin, 0.05, 1.0, n_fine=4096)
    assert np.max(np.abs(a - b)) < 1e-12


def test_exact_cole_hopf_at_t_zero_is_the_initial_field() -> None:
    np.testing.assert_allclose(B.exact_cole_hopf(XS, np.sin, 0.1, 0.0), np.sin(XS))


def test_pseudospectral_follows_the_linear_decay_at_large_viscosity() -> None:
    nu, T, N = 2.0, 3.0, 256
    u = B.pseudospectral_burgers(np.sin, nu, T, N=N)
    linear = np.exp(-nu * T) * np.sin(classical.grid(8))  # nonlinearity is O(u^2)
    assert np.max(np.abs(u - linear)) < 1e-2 * np.exp(-nu * T)


# ------------------------------------------------------------------ zero mean


def test_galilean_split_removes_and_asserts_the_mean() -> None:
    x = classical.grid(6)
    u0 = 0.8 + np.sin(x)
    w0, U0 = B.galilean_split(u0)
    assert U0 == pytest.approx(0.8, abs=1e-15)
    assert abs(np.mean(w0)) <= B.zero_mean_tolerance(u0)
    np.testing.assert_allclose(w0, np.sin(x), atol=1e-15)


def test_cole_hopf_refuses_a_nonzero_mean() -> None:
    x = classical.grid(5)
    with pytest.raises(ValueError, match="int u0 = 0"):
        B.cole_hopf_phi0(0.3 + np.sin(x), 0.5)
    with pytest.raises(ValueError, match="int u0 = 0"):
        B.antiderivative(np.ones(32))


def test_cole_hopf_refuses_a_valley_that_underflows() -> None:
    with pytest.raises(ValueError, match="float64 cannot hold"):
        B.cole_hopf_phi0(np.sin(classical.grid(5)), 0.001)


@pytest.mark.parametrize("nu", [0.5, 0.1])
def test_galilean_shift_matches_the_integrator_on_a_nonzero_mean(nu: float) -> None:
    T, U0 = 1.0, 0.8
    full = B.pseudospectral_burgers(lambda x: U0 + np.sin(x), nu, T, N=1024)
    x_lab = XS
    # pointwise: lab value at x equals U0 + w(x - U0 T) with w the zero-mean solution
    w = B.exact_cole_hopf(np.mod(x_lab - U0 * T, L), np.sin, nu, T)
    assert np.max(np.abs(B.evaluate_fourier(full, x_lab) - (U0 + w))) < 1e-10
    # field level: existing spectral advection with c = U0
    w_grid = B.pseudospectral_burgers(np.sin, nu, T, N=1024)
    assert np.max(np.abs(B.shift_back(w_grid, U0, T) - full)) < 1e-10


def test_lab_positions_are_exact_shifts() -> None:
    xp = np.array([0.1, 2.0, 6.0])
    np.testing.assert_allclose(B.lab_positions(xp, 0.8, 1.0), np.mod(xp + 0.8, L))


def test_negative_controls_for_the_shift() -> None:
    nu, T, U0 = 0.5, 1.0, 0.8
    full = B.pseudospectral_burgers(lambda x: U0 + np.sin(x), nu, T, N=1024)
    w_grid = B.pseudospectral_burgers(np.sin, nu, T, N=1024)
    good = np.max(np.abs(B.shift_back(w_grid, U0, T) - full))
    assert good < 1e-10
    assert np.max(np.abs(B.shift_back(w_grid, -U0, T) - full)) > 0.1  # wrong sign
    assert np.max(np.abs(B.shift_back(w_grid, U0, 2 * T) - full)) > 0.1  # wrong T
    assert np.max(np.abs(w_grid - full)) > 0.5  # no shift, no mean at all


# ------------------------------------------------------------------ discrete Cole-Hopf


def test_phi0_encodes_the_initial_velocity() -> None:
    x = classical.grid(7)
    w0 = np.sin(x) + 0.5 * np.sin(2 * x)
    phi0 = B.cole_hopf_phi0(w0, 0.5)
    assert phi0.max() == 1.0 and np.all(phi0 > 0)
    np.testing.assert_allclose(B.velocity_spectral(phi0, 0.5), w0, atol=1e-9)


def test_fd_velocity_is_second_order_and_spectral_is_far_better() -> None:
    nu, T = 0.5, 1.0
    errs_fd, errs_sp = [], []
    for n in (5, 6, 7):
        u0 = np.sin(classical.grid(n))
        d = B.discrete_burgers(u0, nu, T)
        ex = B.exact_cole_hopf(d.x, np.sin, nu, T)
        errs_fd.append(np.max(np.abs(d.u_fd - ex)))
        errs_sp.append(np.max(np.abs(d.u_spectral - ex)))
    ratios = [errs_fd[i] / errs_fd[i + 1] for i in range(2)]
    assert all(3.5 < r < 4.5 for r in ratios), ratios  # O(h^2)
    assert errs_sp[-1] < 1e-7 and errs_sp[-1] < 1e-3 * errs_fd[-1]


def test_discrete_burgers_bookkeeping() -> None:
    u0 = 0.8 + np.sin(classical.grid(5))
    d = B.discrete_burgers(u0, 0.5, 1.0)
    assert d.U0 == pytest.approx(0.8)
    assert d.p.sum() == pytest.approx(1.0) and np.all(d.p > 0)
    # P_succ is the norm ratio of the heat step
    assert d.p_succ == pytest.approx(np.sum(d.phiT**2) / np.sum(d.phi0**2))
    # an inverted ratio would give -u
    inv = -(0.5 / (2 * np.pi / 32)) * np.log(np.roll(d.phiT, 1) / np.roll(d.phiT, -1))
    np.testing.assert_allclose(inv, -d.u_fd, atol=1e-12)


def test_evaluate_fourier_is_exact_for_trigonometric_polynomials() -> None:
    n = 5
    x = classical.grid(n)
    f = np.cos(3 * x) + 0.2 * np.sin(x) + 0.1 * np.cos(16 * x)  # includes Nyquist
    q = np.array([0.3, 1.7, 5.1])
    np.testing.assert_allclose(
        B.evaluate_fourier(f, q),
        np.cos(3 * q) + 0.2 * np.sin(q) + 0.1 * np.cos(16 * q),
        atol=1e-13,
    )
