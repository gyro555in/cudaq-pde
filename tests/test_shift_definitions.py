"""The definition of the shift names: numpy roll. Everything else must agree."""

import numpy as np
import pytest

from cudaq_pde import classical
from fd_helpers import dense_L


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_names_are_defined_by_np_roll(n: int) -> None:
    u = np.random.default_rng(n).standard_normal(2**n)
    np.testing.assert_array_equal(classical.increment(u), np.roll(u, 1))
    np.testing.assert_array_equal(classical.decrement(u), np.roll(u, -1))
    # element-wise meaning
    inc, dec = classical.increment(u), classical.decrement(u)
    for j in range(2**n):
        assert inc[j] == u[(j - 1) % 2**n]
        assert dec[j] == u[(j + 1) % 2**n]


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_action_on_basis_states(n: int) -> None:
    """increment: |j> -> |j+1>, decrement: |j> -> |j-1> (amplitudes move)."""
    N = 2**n
    for j in range(N):
        e = np.eye(N)[j]
        assert np.argmax(classical.increment(e)) == (j + 1) % N
        assert np.argmax(classical.decrement(e)) == (j - 1) % N


@pytest.mark.parametrize("n", [3, 4])
def test_they_are_inverse_transposes(n: int) -> None:
    u = np.random.default_rng(0).standard_normal(2**n)
    np.testing.assert_array_equal(classical.decrement(classical.increment(u)), u)
    np.testing.assert_array_equal(classical.increment(classical.decrement(u)), u)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_fd_euler_step_is_a0_a_dec_a_inc(n: int) -> None:
    """I + dt L = a0 I + a_dec * decrement + a_inc * increment, vs the stencil."""
    c, nu, L = 0.8, 0.4, classical.TWO_PI
    dt = 0.5 * classical.fd_euler_dt_max(n, c, nu, L)
    u = np.random.default_rng(n).standard_normal(2**n)
    a0, a_dec, a_inc = classical.fd_euler_coefficients(n, dt, c, nu, L)
    step = a0 * u + a_dec * classical.decrement(u) + a_inc * classical.increment(u)
    dense = (np.eye(2**n) + dt * dense_L(n, c, nu, L)) @ u
    np.testing.assert_allclose(step, dense, atol=1e-13)
    # a_dec weights u_{j+1}: advection to the right (c > 0) lowers a_dec, raises a_inc
    h = L / 2**n
    s_adv = c * dt / (2 * h)
    r = nu * dt / h**2
    assert a_dec == pytest.approx(r - s_adv) and a_inc == pytest.approx(r + s_adv)
    assert a_inc > a_dec
