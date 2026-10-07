"""Cost circuits (a) Hadamard test and (b) LCU: mirrors tied to kernels, statistics.

STATEVECTOR CHECKS (test only) are marked.
"""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest
from scipy.optimize import minimize

from cudaq_pde import classical, encoding
from cudaq_pde.solvers import cost_statistics as cs
from cudaq_pde.solvers import lcu, measurable
from cudaq_pde.solvers import measurable_resources as mr
from cudaq_pde.solvers import variational as V
from cudaq_pde.solvers.ansatz_numpy import exact_cost_and_overlap, numpy_ansatz
from cudaq_pde.solvers.measurable_kernels import lcu_cost_circuit

L = classical.TWO_PI


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def make_plan(n: int):
    dt, c, nu = 0.5 * classical.fd_euler_dt_max(n, 1.0, 1.0, L), 1.0, 1.0
    return lcu.lcu_plan(n, dt, c, nu, L), (dt, c, nu)


# ------------------------------------------------ (b) mirror tied to the kernel


@pytest.mark.parametrize("select", ["qft", "ladder"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("n", [3, 4, 5])
def test_cost_b_mirror_has_the_kernel_gate_counts(n, depth, select) -> None:
    plan, _ = make_plan(n)
    rng = np.random.default_rng(n)
    th_prev, th = rng.uniform(-1, 1, (2, n * (depth + 1)))
    kernel = cudaq.estimate_resources(
        lcu_cost_circuit,
        *measurable.circuit_arguments(plan, n, depth, th_prev, th, select),
    )
    mirror = cudaq.estimate_resources(
        mr.build_cost_b_mirror(n, depth, plan, select, thetas=(th_prev, th))
    )
    assert dict(mirror.to_dict()) == dict(kernel.to_dict())


@pytest.mark.parametrize("select", ["qft", "ladder"])
def test_cost_b_mirror_state_equals_kernel_state(select: str) -> None:
    n, depth = 3, 2
    plan, _ = make_plan(n)
    rng = np.random.default_rng(1)
    th_prev, th = rng.uniform(-1, 1, (2, n * (depth + 1)))
    args = measurable.circuit_arguments(plan, n, depth, th_prev, th, select)
    # STATEVECTOR CHECK (test only)
    kernel_state = np.array(cudaq.get_state(lcu_cost_circuit, *args))
    mirror = mr.build_cost_b_mirror(n, depth, plan, select, thetas=(th_prev, th))
    np.testing.assert_allclose(
        np.array(cudaq.get_state(mirror)), kernel_state, atol=1e-12
    )


def test_cost_b_two_qubit_count_closed_form() -> None:
    """(b): 2(n-1)d CX + 2 CRY + n(n-1) + 2n cr1 (QFT SELECT), no Toffoli."""
    for n, depth in ((3, 2), (4, 3), (5, 2)):
        plan, _ = make_plan(n)
        rep = mr._native(
            cudaq.estimate_resources(mr.build_cost_b_mirror(n, depth, plan, "qft"))
        )
        assert rep["two_qubit_gates"] == 2 * (n - 1) * depth + 2 + n * (n - 1) + 2 * n
        assert rep["multi_qubit_gates"] == rep["two_qubit_gates"]  # no 3-qubit gates


# ------------------------------------- (a) mirror: the Hadamard-test identity


@pytest.mark.parametrize("shift", ["I", "dec", "inc"])
def test_a_mirror_probability_is_one_plus_overlap_over_two(shift: str) -> None:
    n, depth = 3, 1
    plan, _ = make_plan(n)
    rng = np.random.default_rng(5)
    th_prev, th = rng.uniform(-1, 1, (2, n * (depth + 1)))
    mirror = mr.build_cost_a_mirror(n, depth, plan, shift, thetas=(th_prev, th))
    # STATEVECTOR CHECK (test only): ancilla is the highest qubit
    state = np.array(cudaq.get_state(mirror))
    p0 = np.linalg.norm(state[: 2**n]) ** 2
    psi, prev = numpy_ansatz(n, th, depth), numpy_ansatz(n, th_prev, depth)
    shifted = {
        "I": prev,
        "dec": classical.decrement(prev),
        "inc": classical.increment(prev),
    }
    overlap = psi @ shifted[shift]
    assert p0 == pytest.approx((1 + overlap) / 2, abs=1e-12)
    assert abs(overlap) < 0.99  # a non-trivial test point


# --------------------------------------------------------------- statistics


def near_optimum(n=3, depth=2):
    plan, (dt, c, nu) = make_plan(n)
    x = classical.grid(n, L)
    u0 = 1 + 0.5 * np.sin(x) + 0.25 * np.cos(2 * x)
    psi0, _ = encoding.encode(u0)
    th_prev = V.optimize_to_target(
        psi0.real, n, depth, V.init_theta(n, depth, 0), "L-BFGS-B", 1e-10, 500
    ).theta
    b = V.fd_target(numpy_ansatz(n, th_prev, depth), dt, c, nu, L)
    th_opt = minimize(
        lambda t: exact_cost_and_overlap(n, t, depth, b)[0],
        th_prev,
        method="L-BFGS-B",
        options={"gtol": 1e-12, "ftol": 1e-16},
    ).x
    return n, depth, plan, th_prev, th_opt


def test_estimator_stds_match_their_analytic_formulas() -> None:
    n, depth, plan, th_prev, th_opt = near_optimum()
    for theta in (
        th_prev,
        th_opt + np.random.default_rng(1).normal(0, 0.3, th_opt.size),
    ):
        q = cs.exact_quantities(n, depth, plan, th_prev, theta)
        r = cs.monte_carlo(plan, q, 6000, 20000, seed=2)
        assert r["a_std"] == pytest.approx(r["a_std_analytic"], rel=0.05)
        assert r["b_std"] == pytest.approx(r["b_std_analytic"], rel=0.05)
        assert abs(r["a_mean"] - r["exact_cost"]) < 5 * r["a_std"] / 100  # unbiased
        assert abs(r["b_mean"] - r["exact_cost"]) < 5 * r["b_std"] / 100


def test_b_noise_vanishes_at_the_optimum_while_a_does_not() -> None:
    n, depth, plan, th_prev, th_opt = near_optimum()
    q = cs.exact_quantities(n, depth, plan, th_prev, th_opt)
    assert q["cost"] < 1e-10
    r = cs.monte_carlo(plan, q, 3000, 20000, seed=3)
    assert r["b_std"] < 1e-6  # observed 0: C(1-C) -> 0
    assert r["a_std"] > 1e-3  # observed 1.9e-3, set by (1 - o_shift^2) / N


@pytest.mark.parametrize("sigma", [0.0, 0.3])
def test_b_has_lower_variance_at_equal_total_shots(sigma: float) -> None:
    n, depth, plan, th_prev, th_opt = near_optimum()
    theta = (
        th_prev
        if sigma == 0
        else th_opt + np.random.default_rng(1).normal(0, sigma, th_opt.size)
    )
    q = cs.exact_quantities(n, depth, plan, th_prev, theta)
    r = cs.monte_carlo(plan, q, 6000, 20000, seed=4)
    assert r["a_std"] / r["b_std"] > 2.5  # observed about 3.3 at both points


def test_circuits_per_cost_evaluation() -> None:
    n, depth = 3, 2
    plan, (dt, c, nu) = make_plan(n)
    rep = mr.cost_circuit_report(n, depth, dt, c, nu)
    assert rep["a_totals_per_cost_evaluation"]["circuits"] == 3
    assert set(rep["a_circuits"]) == {"I", "dec", "inc"}
    assert set(rep["a_step_constants"]) == {"D^1", "D^2"}  # 2 more circuits per step
