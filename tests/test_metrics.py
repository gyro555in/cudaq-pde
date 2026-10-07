"""Stable angles and infidelities vs the old ``arcsin(sqrt(1 - F))`` pattern.

The reference is EXACT rational arithmetic on the stored float64 values, so the "known
angle" is the true angle of the inputs actually passed, not of the real numbers they
approximate. Two kinds of inputs, because float64 cannot give more than this:

* exactly representable pairs (axis aligned): the stable form is correct to ~eps
  RELATIVE, down to angle 1e-8 (the old form loses everything);
* generic random states: components carry ~eps rounding of their own, so the best
  possible is ~eps ABSOLUTE (relative 1e-8 at angle 1e-8). The stable form reaches it
  (observed 0.1 eps); the old form has an absolute error floor ``sqrt(eps) = 1.5e-8``.
"""

import math
from collections.abc import Iterator
from fractions import Fraction

import cudaq
import numpy as np
import pytest

from cudaq_pde import metrics
from cudaq_pde.solvers import variational as V
from cudaq_pde.solvers.ansatz_numpy import numpy_ansatz

EPS = np.finfo(float).eps
DELTAS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]


def old_angle(psi, phi) -> float:
    """The pattern replaced everywhere: ``arcsin(sqrt(1 - F))``."""
    f = (psi @ phi) ** 2 / ((psi @ psi) * (phi @ phi))
    return float(np.arcsin(np.sqrt(max(0.0, 1.0 - f))))


def old_infidelity(psi, phi) -> float:
    return float(1.0 - (psi @ phi) ** 2 / ((psi @ psi) * (phi @ phi)))


def exact_sin2(psi, phi) -> float:
    """``sin^2`` of the angle of the stored float64 vectors, in exact arithmetic."""
    p = [Fraction(float(x)) for x in psi]
    q = [Fraction(float(x)) for x in phi]
    dot = sum(a * b for a, b in zip(p, q, strict=True))
    pp = sum(a * a for a in p)
    qq = sum(b * b for b in q)
    return float(1 - dot * dot / (pp * qq))  # correctly rounded: relative eps


def exact_angle(psi, phi) -> float:
    return math.asin(math.sqrt(exact_sin2(psi, phi)))


def axis_pair(delta: float):
    psi = np.zeros(8)
    psi[0] = 1.0
    phi = np.zeros(8)
    phi[0], phi[3] = np.cos(delta), np.sin(delta)
    return psi, phi


def generic_pair(delta: float, seed: int, dim: int = 16):
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(dim)
    psi /= np.linalg.norm(psi)
    w = rng.standard_normal(dim)
    w -= (w @ psi) * psi
    w /= np.linalg.norm(w)
    return psi, np.cos(delta) * psi + np.sin(delta) * w


@pytest.mark.parametrize("delta", DELTAS)
def test_exactly_representable_pair_is_recovered_to_eps_relative(delta) -> None:
    psi, phi = axis_pair(delta)
    exact = exact_angle(psi, phi)
    assert exact == pytest.approx(delta, rel=1e-12)  # sanity: the intended angle
    stable = metrics.state_angle(psi, phi)
    assert abs(stable - exact) <= 2 * EPS * exact  # observed 0
    # the old form: relative error 1e-9 at 1e-4 growing as delta^-2, 100% at 1e-8
    old_rel = abs(old_angle(psi, phi) - exact) / exact
    assert old_rel > 1e-10 * (1e-4 / delta) ** 2
    assert old_rel > 1e4 * abs(stable - exact) / exact or stable == exact


@pytest.mark.parametrize("delta", DELTAS)
def test_infidelity_is_accurate_where_one_minus_f_is_not(delta) -> None:
    psi, phi = axis_pair(delta)
    exact = exact_sin2(psi, phi)
    assert abs(metrics.infidelity(psi, phi) - exact) <= 8 * EPS * exact  # observed ~eps
    # the old 1 - F has an ABSOLUTE error ~1e-16, i.e. relative ~1e-16 / delta^2
    old_rel = abs(old_infidelity(psi, phi) - exact) / exact
    assert old_rel > 1e-10 * (1e-4 / delta) ** 2  # observed 6e-9 (1e-4) .. 1.0 (1e-8)


@pytest.mark.parametrize("delta", DELTAS)
def test_generic_states_reach_the_float64_limit(delta) -> None:
    worst_stable = worst_old = 0.0
    for seed in range(20):
        psi, phi = generic_pair(delta, seed)
        exact = exact_angle(psi, phi)
        worst_stable = max(worst_stable, abs(metrics.state_angle(psi, phi) - exact))
        worst_old = max(worst_old, abs(old_angle(psi, phi) - exact))
    assert worst_stable <= 1.0 * EPS  # observed 0.1 eps: backward-stable level
    assert worst_old > 100 * EPS  # observed 5.6e3 eps at 1e-4 up to 7e7 eps at 1e-8
    assert worst_old > 20 * worst_stable


def test_old_angle_has_a_sqrt_eps_floor_and_the_stable_one_does_not() -> None:
    errs_old, errs_stable = [], []
    for seed in range(20):
        psi, phi = generic_pair(1e-8, seed)
        exact = exact_angle(psi, phi)
        errs_old.append(abs(old_angle(psi, phi) - exact))
        errs_stable.append(abs(metrics.state_angle(psi, phi) - exact))
    assert max(errs_old) > 1e-9  # ~ sqrt(eps) = 1.5e-8
    assert max(errs_stable) < 1e-15
    assert max(errs_stable) / 1e-8 < 1e-7  # relative error at the limit of float64


def test_agrees_with_the_asin_form_for_sign_aligned_unit_states() -> None:
    for delta in DELTAS:
        psi, phi = generic_pair(delta, 3)
        asin_form = 2 * math.asin(float(np.linalg.norm(psi - phi)) / 2)
        assert abs(metrics.state_angle(psi, phi) - asin_form) <= 4 * EPS


def test_invariances_complex_scale_and_phase() -> None:
    psi, phi = generic_pair(1e-5, 7)
    angle = metrics.state_angle(psi, phi)
    assert metrics.state_angle(3.7 * psi, 0.2 * phi) == pytest.approx(angle, abs=EPS)
    assert metrics.state_angle(-psi, phi) == pytest.approx(angle, abs=EPS)  # same ray
    rng = np.random.default_rng(1)
    p = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    q = p + 1e-6 * (rng.standard_normal(8) + 1j * rng.standard_normal(8))
    a = metrics.state_angle(p, q)
    assert metrics.state_angle(np.exp(0.7j) * p, q) == pytest.approx(a, abs=EPS)
    assert 0 < a < 1e-5


def test_generic_angle_matches_one_minus_f_where_that_is_fine() -> None:
    psi, phi = generic_pair(0.4, 5)
    assert metrics.infidelity(psi, phi) == pytest.approx(
        old_infidelity(psi, phi), rel=1e-13
    )
    assert metrics.state_angle(psi, phi) == pytest.approx(0.4, abs=1e-15)
    orthogonal = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    assert metrics.state_angle(*orthogonal) == pytest.approx(math.pi / 2)


def test_stacks_match_row_by_row_and_are_deterministic() -> None:
    pairs = [generic_pair(d, 9 + i) for i, d in enumerate(DELTAS)]
    psis = np.array([p for p, _ in pairs])
    phis = np.array([q for _, q in pairs])
    stack = metrics.state_angles(psis, phis)
    for i, (p, q) in enumerate(pairs):
        assert stack[i] == metrics.state_angle(p, q)  # bitwise: pairwise sums
    np.testing.assert_array_equal(stack, metrics.state_angles(psis, phis))


# ---------------------------------------------------- the solver's cost near zero


@pytest.fixture
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


@pytest.mark.parametrize("delta", [1e-5, 1e-6, 1e-7])
def test_solver_cost_is_accurate_near_the_optimum(cpu_target, delta) -> None:
    """``simulator_cost_and_state`` at C ~ 1e-10 to 1e-14 (state plus a small kick)."""
    n, depth = 3, 2
    theta = np.random.default_rng(4).uniform(-1, 1, V.num_params(n, depth))
    psi_np = numpy_ansatz(n, theta, depth)
    rng = np.random.default_rng(5)
    w = rng.standard_normal(psi_np.size)
    w -= (w @ psi_np) * psi_np
    w /= np.linalg.norm(w)
    b = 3.7 * (np.cos(delta) * psi_np + np.sin(delta) * w)  # not normalized
    cost, psi, overlap = V.simulator_cost_and_state(theta, b, n, depth)
    expected = exact_sin2(psi_np, b)
    assert cost == pytest.approx(expected, rel=1e-6)  # observed ~1e-9
    old = 1.0 - overlap**2 / float(b @ b)  # the replaced formula
    if delta <= 1e-7:  # C ~ 1e-14: the old formula is noise (observed >1e-2 relative)
        assert abs(old - expected) / expected > 1e-4
    assert overlap == pytest.approx(3.7 * np.cos(delta), rel=1e-12)
