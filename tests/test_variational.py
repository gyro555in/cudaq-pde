"""Variational advection-diffusion vs the explicit-Euler finite-difference solution.

Setup used throughout: ``c = 1``, ``nu = 0.4``, ``dt = 0.05``, ``L = 2 pi``,
``u0 = 1 + 0.5 sin x + 0.25 cos 2x``. For n = 3 and 4 the step satisfies the
explicit-Euler stability limit by a wide margin (``a = 2 nu dt / h^2 = 0.065``
and ``0.26``, ``b^2 = 0.004`` and ``0.016``).

Tolerances and why
------------------
* Local cost: ``C_s = 1 - <psi_s|b_s>^2 / ||b_s||^2`` is exactly the infidelity of
  the step's result to its exact Euler target, so it measures the optimizer and
  the ansatz, nothing else. With L-BFGS-B and finite-difference gradients,
  ``C`` at convergence scales like ``tol^2`` (observed ``C ~ 0.4 tol^2`` to
  ``40 tol^2``) until the finite-difference gradient floor (``eps = 1e-8``) is
  reached, so ``tol`` below ``1e-8`` gives no further gain.
* Global error: with ``kappa`` the condition number of ``I + dt L`` and
  ``phi_s`` the angle between ``psi_s`` and the exact FD state,
  ``sin(angle after the map) <= kappa sin(angle before)``, hence
  ``phi_s <= arcsin(kappa sin phi_{s-1}) + arcsin(sqrt(C_s))`` (the second term is
  the local angle). ``test_global_error_obeys_angle_accumulation_bound`` asserts
  this with the measured ``C_s``; it holds with no free parameter.
* Absolute thresholds are set from the observed results with a margin (observed
  values in comments next to each constant) so they catch regressions without
  being tuned to the last digit. An L2 error of ``x`` corresponds to an
  infidelity of about ``x^2 / 4`` (angle ~ L2 for a normalized field), which is
  why the L2 thresholds are much larger than the infidelity thresholds.
* Everything runs on qpp-cpu fp64. On the login node run pytest with
  ``taskset -c 0`` so runtime threads do not multiply CPU time.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache
from typing import NamedTuple

import cudaq
import numpy as np
import pytest

from cudaq_pde import classical
from cudaq_pde.solvers import variational as V
from cudaq_pde.solvers.ansatz_numpy import numpy_ansatz
from fd_helpers import dense_L

C_ADV, NU, DT, L = 1.0, 0.4, 0.05, classical.TWO_PI

# n -> (depth, tol, steps, max infidelity, max L2). Observed: n=3 2e-12 / 1.4e-6,
# n=4 9e-10 / 3e-5 over 5 steps.
MAIN = {3: (2, 1e-6, 5, 1e-9, 1e-4), 4: (3, 1e-6, 5, 1e-7, 1e-3)}


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def u0_of(n: int) -> np.ndarray:
    x = classical.grid(n, L)
    return 1.0 + 0.5 * np.sin(x) + 0.25 * np.cos(2 * x)


class Run(NamedTuple):
    res: V.VariationalResult
    ref: np.ndarray
    angle: np.ndarray  # stable angle to the FD-Euler state (cudaq_pde.metrics)
    infid: np.ndarray  # sin^2(angle): never computed as 1 - F
    l2: np.ndarray


@cache
def run(n, depth, tol, steps, seed=0, method="L-BFGS-B", maxiter=500):
    ref = classical.evolve_fd_euler(u0_of(n), DT, steps, C_ADV, NU, L)
    cudaq.set_target("qpp-cpu")
    res = V.evolve_variational(
        u0_of(n),
        DT,
        steps,
        C_ADV,
        NU,
        L,
        depth=depth,
        seed=seed,
        method=method,
        tol=tol,
        maxiter=maxiter,
    )
    angle, infid, l2 = V.reference_errors(res, ref)
    return Run(res, ref, angle, infid, l2)


def _table(label: str, res, infid, l2) -> str:
    rows = [f"{label}: step  1-F        L2         cost       nfev"]
    for s in range(len(infid)):
        rows.append(
            f"      {s:4d}  {infid[s]:.2e}  {l2[s]:.2e}  "
            f"{res.costs[s]:.2e}  {res.nfev[s]:5d}"
        )
    return "\n".join(rows)


def _random_state(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(2**n)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- algebra


@pytest.mark.parametrize("n", [3, 4])
def test_fd_target_matches_dense_operator(n: int) -> None:
    psi = _random_state(n, n)
    b = V.fd_target(psi, DT, C_ADV, NU, L)
    dense = (np.eye(2**n) + DT * dense_L(n, C_ADV, NU, L)) @ psi
    np.testing.assert_allclose(b, dense, atol=1e-14)


@pytest.mark.parametrize("n", [3, 4])
def test_shift_decomposition_of_overlap_and_norm(n: int) -> None:
    """Everything the hardware cost needs: 3 overlaps + 2 per-step expectations."""
    N = 2**n
    # matrices of the numpy-defined shifts: dec @ u == classical.decrement(u)
    dec = np.array([classical.decrement(e) for e in np.eye(N)]).T
    inc = np.array([classical.increment(e) for e in np.eye(N)]).T
    np.testing.assert_array_equal(dec @ inc, np.eye(N))  # unitary, inverse pair
    np.testing.assert_array_equal(inc, dec.T)
    psi_prev, psi = _random_state(n, 1), _random_state(n, 2)
    a0, a_dec, a_inc = classical.fd_euler_coefficients(n, DT, C_ADV, NU, L)
    b = V.fd_target(psi_prev, DT, C_ADV, NU, L)

    overlap = (
        a0 * psi @ psi_prev
        + a_dec * psi @ dec @ psi_prev
        + a_inc * psi @ inc @ psi_prev
    )
    assert overlap == pytest.approx(psi @ b, abs=1e-14)

    e1 = psi_prev @ dec @ psi_prev  # <decrement> (equals <increment>)
    e2 = psi_prev @ dec @ dec @ psi_prev  # <decrement^2> (equals <increment^2>)
    assert e1 == pytest.approx(psi_prev @ inc @ psi_prev, abs=1e-14)
    norm_sq = (
        a0**2
        + a_dec**2
        + a_inc**2
        + 2 * a0 * (a_dec + a_inc) * e1
        + 2 * a_dec * a_inc * e2
    )
    assert norm_sq == pytest.approx(b @ b, abs=1e-14)

    cost_dense = 1.0 - (psi @ b) ** 2 / (b @ b)
    cost_shifts = 1.0 - overlap**2 / norm_sq
    assert cost_shifts == pytest.approx(cost_dense, abs=1e-14)


@pytest.mark.parametrize("n", [3, 4])
def test_norm_tracking_algebra_is_exact_for_a_perfect_ansatz(n: int) -> None:
    u_prev = u0_of(n) * 1.7
    norm_prev = np.linalg.norm(u_prev)
    psi_prev = u_prev / norm_prev
    b = V.fd_target(psi_prev, DT, C_ADV, NU, L)
    psi = b / np.linalg.norm(b)  # a perfect ansatz: C = 0
    overlap = psi @ b
    u_next = classical.evolve_fd_euler(u_prev, DT, 1, C_ADV, NU, L)[1]
    assert norm_prev * overlap == pytest.approx(np.linalg.norm(u_next), rel=1e-14)
    np.testing.assert_allclose(norm_prev * overlap * psi, u_next, atol=1e-13)
    # a global sign flip of psi changes the sign of the overlap and nothing else
    np.testing.assert_allclose(
        norm_prev * (-psi @ b) * (-psi), norm_prev * overlap * psi, atol=1e-14
    )


@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("n", [3, 4])
def test_simulator_cost_matches_numpy_and_is_sign_invariant(n: int, depth: int) -> None:
    rng = np.random.default_rng(n + depth)
    theta = rng.uniform(-np.pi, np.pi, V.num_params(n, depth))
    b = V.fd_target(_random_state(n, 7), DT, C_ADV, NU, L)
    cost, psi, overlap = V.simulator_cost_and_state(theta, b, n, depth)
    ref_psi = numpy_ansatz(n, theta, depth)
    np.testing.assert_allclose(psi, ref_psi, atol=1e-12)
    assert overlap == pytest.approx(ref_psi @ b, abs=1e-12)
    assert cost == pytest.approx(1.0 - (ref_psi @ b) ** 2 / (b @ b), abs=1e-12)
    assert 0.0 <= cost <= 1.0
    cost_flipped, _, overlap_flipped = V.simulator_cost_and_state(theta, -b, n, depth)
    assert cost_flipped == pytest.approx(cost, abs=1e-14)
    assert overlap_flipped == pytest.approx(-overlap, abs=1e-14)


# ------------------------------------------------------------ per-step error


@pytest.mark.slow
@pytest.mark.parametrize("n", [3, 4])
def test_per_step_fidelity_and_l2_vs_fd_euler(n: int, capsys) -> None:
    depth, tol, steps, max_infid, max_l2 = MAIN[n]
    res, _, _, infid, l2 = run(n, depth, tol, steps)
    print(_table(f"n={n} depth={depth} tol={tol:g}", res, infid, l2))
    assert infid.shape == l2.shape == (steps + 1,)
    assert np.all(infid < max_infid), infid
    assert np.all(l2 < max_l2), l2
    assert res.costs[0] < max_infid  # initial-state fit (preparation error)


@pytest.mark.slow
@pytest.mark.parametrize("n", [3, 4])
def test_global_error_obeys_angle_accumulation_bound(n: int) -> None:
    depth, tol, steps, *_ = MAIN[n]
    res, _, angle, _, _ = run(n, depth, tol, steps)
    kappa = np.linalg.cond(np.eye(2**n) + DT * dense_L(n, C_ADV, NU, L))
    # kappa = max|g| / min|g| over Fourier modes. It is NOT close to 1: the
    # highest-frequency mode is damped by |1 - 4 nu dt / h^2| (min|g| = 0.870 for
    # n = 3, 0.481 for n = 4), so kappa = 1.149 and 2.078 (np.linalg.cond, checked
    # by the assertion below). The bound below is valid but loose; smooth fields
    # do not excite that mode.
    a0, a_dec, a_inc = classical.fd_euler_coefficients(n, DT, C_ADV, NU, L)
    theta = 2 * np.pi * np.arange(2**n) / 2**n
    g = np.abs(a0 + a_dec * np.exp(1j * theta) + a_inc * np.exp(-1j * theta))
    assert kappa == pytest.approx(g.max() / g.min(), rel=1e-9)
    print(f"n={n}: kappa(I + dt L) = {kappa:.3f}")

    # Tolerance from the arithmetic, not by hand. Both sides come from stable forms
    # (cudaq_pde.metrics): the angle is atan2 of the orthogonal residual and the local
    # angle is arcsin(sqrt(C)) with C = sin^2(angle) from the same residual. Each has an
    # ABSOLUTE error of a few eps (backward stable; the components of the inputs carry
    # ~eps rounding): E = 4 eps per quantity. The recursion
    #   bound_s = arcsin(kappa sin bound_{s-1}) + a_s
    # multiplies an earlier error by at most kappa per step, so bound_s has error
    # <= E sum_{k<=s} kappa^k, and the compared angle adds E. At step 0 the two sides
    # are mathematically equal and must agree to within tol_0 = 2 E.
    eps = np.finfo(float).eps
    big_e = 4 * eps

    def tol_at(s: int) -> float:
        return big_e * (1.0 + sum(kappa**k for k in range(s + 1)))

    bound = np.arcsin(np.sqrt(res.costs[0]))
    assert abs(angle[0] - bound) <= tol_at(0), (angle[0], bound)  # equality case
    for s in range(steps + 1):
        if s > 0:
            bound = np.arcsin(min(1.0, kappa * np.sin(bound))) + np.arcsin(
                np.sqrt(res.costs[s])
            )
        assert angle[s] <= bound + tol_at(s), (s, angle[s], bound)


@pytest.mark.slow
@pytest.mark.parametrize("n", [3, 4])
def test_norm_is_tracked_classically(n: int) -> None:
    depth, tol, steps, _, max_l2 = MAIN[n]
    res, ref = run(n, depth, tol, steps)[:2]
    ref_norm = np.linalg.norm(ref, axis=1)
    # The optimizer may land on psi or -psi (same cost, a global phase), so the
    # tracked norm is signed: observed positive for n = 3 and negative (-4.3) for
    # n = 4. What is guaranteed is a constant sign and a correct field norm * psi.
    assert np.all(np.sign(res.norms) == np.sign(res.norms[0]))
    assert np.all(res.fields > 0)  # smooth positive solution
    assert np.all(np.abs(np.abs(res.norms) - ref_norm) / ref_norm < max_l2)
    # the FD scheme is dissipative here, so the norm must decrease
    assert np.all(np.diff(np.abs(res.norms)) < 0)


@pytest.mark.slow
@pytest.mark.parametrize("n", [3, 4])
def test_optimization_error_is_below_discretization_error(n: int, capsys) -> None:
    """Report FD-Euler vs spectral (discretization) separately from the optimizer."""
    depth, tol, steps, *_ = MAIN[n]
    res, ref, _, _, l2 = run(n, depth, tol, steps)
    T = steps * DT
    spec = classical.evolve_spectral(u0_of(n), T, C_ADV, NU, L)
    rel = lambda a: np.linalg.norm(a - spec) / np.linalg.norm(spec)  # noqa: E731
    disc, var_vs_spec = rel(ref[-1]), rel(res.fields[-1])
    print(
        f"n={n} T={T}: FD-Euler vs spectral (discretization) {disc:.3e}; "
        f"variational vs FD-Euler (optimization) {l2[-1]:.3e}; "
        f"variational vs spectral {var_vs_spec:.3e}"
    )
    assert disc > 10 * l2[-1]
    assert abs(var_vs_spec - disc) <= l2[-1] * (1 + 1e-9)  # triangle inequality


@pytest.mark.slow
@pytest.mark.parametrize("n", [3, 4])
def test_cost_evaluations_are_reported_and_counted(n: int, monkeypatch) -> None:
    depth, tol, steps, *_ = MAIN[n]
    calls = 0
    real = V.simulator_cost_and_state

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(V, "simulator_cost_and_state", counting)
    res = V.evolve_variational(
        u0_of(n), DT, 2, C_ADV, NU, L, depth=depth, seed=0, tol=tol
    )
    P = V.num_params(n, depth)
    assert np.all(res.nfev >= P + 1)  # at least one finite-difference gradient
    assert calls == int(res.nfev.sum()) + 3  # + one final state read per solve


# -------------------------------------------------------------- convergence


def _final_infidelity(n, depth, tol, steps) -> float:
    return float(run(n, depth, tol, steps).infid[-1])


@pytest.mark.slow
def test_convergence_vs_ansatz_depth_n3(capsys) -> None:
    """Expressibility: observed 5e-2, 3e-3, then <= 1e-11 for depth 0, 1, >= 2."""
    inf = {d: _final_infidelity(3, d, 1e-6, 3) for d in range(5)}
    print("n=3 final infidelity by depth:", {d: f"{v:.1e}" for d, v in inf.items()})
    assert inf[0] > 1e-2 > 1e-3 > 1e-1 * 1e-3
    assert inf[1] > 1e-3
    assert inf[0] > inf[1] > 100 * inf[2]
    assert all(inf[d] < 1e-9 for d in (2, 3, 4))  # saturated at the optimizer floor


@pytest.mark.slow
def test_convergence_vs_ansatz_depth_n4(capsys) -> None:
    """Observed 8e-3, 9e-4, 5e-8, 2e-9 for depth 1..4 (n = 4 has 15 amplitudes)."""
    inf = {d: _final_infidelity(4, d, 1e-5, 2) for d in range(1, 5)}
    print("n=4 final infidelity by depth:", {d: f"{v:.1e}" for d, v in inf.items()})
    assert inf[1] > 1e-3 and inf[2] > 1e-4
    assert inf[1] > inf[2] > 100 * inf[3]
    assert inf[3] < 1e-6 and inf[4] < 1e-6


@pytest.mark.slow
def test_convergence_vs_optimizer_tolerance_n3(capsys) -> None:
    """Observed final infidelity 1e-3, 1e-4, 2e-8, 1e-12, 1e-14, 1e-14 for tol
    1e-2 .. 1e-10; the last two coincide (finite-difference gradient floor)."""
    tols = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
    inf = [_final_infidelity(3, 2, t, 3) for t in tols]
    cost = [float(run(3, 2, t, 3).res.costs[-1]) for t in tols]
    for t, i, c in zip(tols, inf, cost, strict=True):
        print(f"tol={t:g}: final infidelity {i:.1e}, final cost {c:.1e}")
    assert inf[0] > 1e-5
    for coarse, fine in zip(inf[:4], inf[1:5], strict=False):
        assert fine < coarse  # strictly better until the floor
    assert inf[4] < 1e-10
    assert inf[5] < 1e-10  # no further gain below the floor, but no loss either
    assert inf[0] > 1e4 * inf[4]


# ------------------------------------------------- determinism, methods, input


@pytest.mark.slow
def test_same_seed_is_deterministic_and_other_seed_agrees_physically() -> None:
    a = run(3, 2, 1e-6, 3, seed=0).res
    again = V.evolve_variational(
        u0_of(3), DT, 3, C_ADV, NU, L, depth=2, seed=0, tol=1e-6
    )
    np.testing.assert_allclose(again.thetas, a.thetas, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(again.nfev, a.nfev)
    b = run(3, 2, 1e-6, 3, seed=5).res
    assert not np.allclose(b.thetas, a.thetas, atol=1e-3)  # different path ...
    assert np.max(np.abs(b.fields - a.fields)) < 1e-4  # ... same physics


def test_init_theta_is_seeded() -> None:
    a, b = V.init_theta(4, 3, 11), V.init_theta(4, 3, 11)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, V.init_theta(4, 3, 12))
    assert a.shape == (V.num_params(4, 3),) == (16,)


@pytest.mark.slow
def test_cobyla_works_but_is_much_weaker(capsys) -> None:
    """Observed: COBYLA stops at maxiter (300) every step, infidelity ~1e-4."""
    res, _, _, infid, l2 = run(3, 2, 1e-6, 3, method="COBYLA", maxiter=300)
    print(_table("n=3 COBYLA", res, infid, l2))
    assert np.all(res.nfev <= 300 + 1)
    assert np.all(infid < 1e-2)
    lbfgs = run(3, 2, 1e-6, 3).infid
    assert infid[-1] > 100 * lbfgs[-1]
    again = run(3, 2, 1e-6, 3, method="COBYLA", maxiter=300)[0]
    np.testing.assert_allclose(again.thetas, res.thetas, atol=1e-12)


@pytest.mark.slow
def test_restarts_never_worsen_the_initial_fit() -> None:
    one = V.evolve_variational(u0_of(3), DT, 0, C_ADV, NU, L, depth=1, seed=0)
    many = V.evolve_variational(
        u0_of(3), DT, 0, C_ADV, NU, L, depth=1, seed=0, restarts=3
    )
    assert many.costs[0] <= one.costs[0] + 1e-15
    assert many.nfev[0] > one.nfev[0]


def test_stability_limit_is_asserted() -> None:
    u0 = u0_of(3)
    dt_max = classical.fd_euler_dt_max(3, C_ADV, NU, L)
    with pytest.raises(ValueError, match="unstable"):
        V.evolve_variational(u0, 10 * dt_max, 1, C_ADV, NU, L)
    with pytest.raises(ValueError, match="unstable"):
        V.evolve_variational(u0, 1e-4, 1, C_ADV, 0.0, L)  # nu = 0: no stable step


def test_input_validation() -> None:
    u0 = u0_of(3)
    with pytest.raises(ValueError):
        V.evolve_variational(u0, DT, 1, C_ADV, NU, L, method="BFGS")
    with pytest.raises(ValueError):
        V.evolve_variational(u0, DT, -1, C_ADV, NU, L)
    with pytest.raises(ValueError):
        V.evolve_variational(u0, DT, 1, C_ADV, NU, L, restarts=0)
    with pytest.raises(ValueError):
        V.evolve_variational(np.zeros(8), DT, 1, C_ADV, NU, L)
