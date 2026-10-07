"""Measurable cost circuit, sampled estimates and parameter-shift gradients (qpp-cpu).

STATEVECTOR CHECKS (test only) are marked; the solver code under test never reads a
state vector, and ``test_hardware_path_runs_with_get_state_disabled`` proves it.
"""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde import classical
from cudaq_pde.solvers import lcu, measurable
from cudaq_pde.solvers import variational as V
from cudaq_pde.solvers.ansatz_numpy import exact_cost_and_overlap, numpy_ansatz
from cudaq_pde.solvers.measurable_kernels import lcu_cost_circuit
from cudaq_pde.solvers.sampling import Sampler

L = classical.TWO_PI


def plan_params(n: int, negative: bool) -> tuple[float, float, float]:
    c, nu = (1.2, 0.05) if negative else (1.0, 0.4)
    return 0.5 * classical.fd_euler_dt_max(n, c, nu, L), c, nu


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def setup(n: int, depth: int, negative: bool, seed: int = 0):
    dt, c, nu = plan_params(n, negative)
    rng = np.random.default_rng(seed)
    theta_prev = rng.uniform(-1.5, 1.5, V.num_params(n, depth))
    theta = theta_prev + rng.normal(0, 0.15, theta_prev.size)
    plan = lcu.lcu_plan(n, dt, c, nu, L)
    b = V.fd_target(numpy_ansatz(n, theta_prev, depth), dt, c, nu, L)
    return plan, theta_prev, theta, b


# ------------------------------------------------- circuit vs numpy (statevector)


@pytest.mark.parametrize("select", ["qft", "ladder"])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("n", [3, 4])
def test_postselected_amplitude_is_overlap_over_lambda(n, negative, select) -> None:
    depth = 2
    plan, theta_prev, theta, b = setup(n, depth, negative, seed=n)
    args = measurable.circuit_arguments(plan, n, depth, theta_prev, theta, select)
    # STATEVECTOR CHECK (test only)
    state = np.array(cudaq.get_state(lcu_cost_circuit, *args))
    block = state[: 2**n]  # anc = 00, work = 0
    _, overlap = exact_cost_and_overlap(n, theta, depth, b)
    assert block[0] == pytest.approx(overlap / plan.lam, abs=1e-12)  # signed
    assert np.linalg.norm(block) ** 2 == pytest.approx(
        plan.success_probability(b @ b), abs=1e-12
    )


# ------------------------------------------------------------ counts reduction


def test_counts_to_estimate() -> None:
    counts = {"00000": 60, "10000": 30, "00010": 7, "01100": 3}  # n = 3
    est = measurable.counts_to_estimate(counts, 3, 100)
    assert est.n_succ == 93 and est.shots == 100
    assert est.p_joint == pytest.approx(0.60)
    assert est.cost == pytest.approx(1 - 60 / 93)
    assert est.p_succ == pytest.approx(0.93)


def test_counts_with_work_qubits_and_no_success() -> None:
    # n = 3: 3 system bits, 2 LCU ancilla bits, then 2 work qubits
    est = measurable.counts_to_estimate({"0000000": 5, "0000001": 1}, 3, 6)
    assert est.n_succ == 6 and est.cost == 0.0  # work qubits are not post-selected
    none = measurable.counts_to_estimate({"00010": 4}, 3, 4)
    assert none.n_succ == 0 and none.cost == 1.0 and none.p_joint == 0.0


# ----------------------------------------------- sampled estimates vs exact, 5 sigma


@pytest.mark.parametrize("select", ["qft", "ladder"])
@pytest.mark.parametrize("negative", [False, True])
def test_sampled_cost_matches_exact_within_5_sigma(negative, select) -> None:
    n, depth, shots = 3, 2, 400_000
    plan, theta_prev, theta, b = setup(n, depth, negative, seed=3)
    cost, overlap = exact_cost_and_overlap(n, theta, depth, b)
    p_joint = overlap**2 / plan.lam**2
    p_succ = plan.success_probability(b @ b)
    est = measurable.sampled_cost(
        Sampler(base_seed=1), plan, n, depth, theta_prev, theta, shots, select
    )
    assert 0.01 < cost < 0.99  # a non-trivial test point
    assert abs(est.p_succ - p_succ) < 5 * np.sqrt(p_succ * (1 - p_succ) / shots)
    assert abs(est.p_joint - p_joint) < 5 * np.sqrt(p_joint * (1 - p_joint) / shots)
    assert abs(est.cost - cost) < 5 * np.sqrt(cost * (1 - cost) / est.n_succ)
    # the norm update magnitude lambda sqrt(P_joint) estimates |<psi|b>|
    assert abs(plan.lam * np.sqrt(est.p_joint) - abs(overlap)) < 0.01


def test_estimates_are_reproducible_from_the_base_seed() -> None:
    n, depth = 3, 1
    plan, theta_prev, theta, _ = setup(n, depth, False)
    a = measurable.sampled_cost(Sampler(7), plan, n, depth, theta_prev, theta, 1000)
    b = measurable.sampled_cost(Sampler(7), plan, n, depth, theta_prev, theta, 1000)
    c = measurable.sampled_cost(Sampler(8), plan, n, depth, theta_prev, theta, 1000)
    assert a == b and a != c


# ----------------------------------------------------------- parameter shift


def exact_cost_fn(n, depth, b):
    return lambda th: exact_cost_and_overlap(n, th, depth, b)[0]


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("n", [3, 4])
def test_parameter_shift_equals_finite_differences_on_the_exact_cost(n, depth) -> None:
    _, _, theta, b = setup(n, depth, False, seed=10 * n + depth)
    f = exact_cost_fn(n, depth, b)
    grad = measurable.parameter_shift_gradient(f, theta)
    eps = 1e-6
    fd = np.array(
        [
            (f(theta + eps * e) - f(theta - eps * e)) / (2 * eps)
            for e in np.eye(theta.size)
        ]
    )
    np.testing.assert_allclose(grad, fd, atol=1e-8)
    assert np.max(np.abs(grad)) > 1e-3  # a non-trivial gradient


def test_a_wrong_shift_would_not_match() -> None:
    """Negative control: a pi/4 shift disagrees with the finite differences."""
    _, _, theta, b = setup(3, 2, False, seed=5)
    f = exact_cost_fn(3, 2, b)
    good = measurable.parameter_shift_gradient(f, theta)
    bad = measurable.parameter_shift_gradient(f, theta, shift=np.pi / 4)
    assert np.max(np.abs(good - bad)) > 1e-3


@pytest.mark.slow
def test_sampled_gradient_is_unbiased() -> None:
    """Mean of repeated sampled gradients within 4 standard errors of the exact one."""
    n, depth, shots, repeats = 3, 1, 2000, 40
    plan, theta_prev, theta, b = setup(n, depth, False, seed=2)
    f = exact_cost_fn(n, depth, b)
    exact = measurable.parameter_shift_gradient(f, theta)
    sampler = Sampler(base_seed=100)

    def cost(th):
        return measurable.sampled_cost(
            sampler, plan, n, depth, theta_prev, th, shots
        ).cost

    grads = np.array(
        [measurable.parameter_shift_gradient(cost, theta) for _ in range(repeats)]
    )
    mean, sem = grads.mean(axis=0), grads.std(axis=0, ddof=1) / np.sqrt(repeats)
    assert np.all(np.abs(mean - exact) < 4 * sem + 1e-3)
    assert np.all(grads.std(axis=0) > 0)  # it really is noisy


# ----------------------------------------------------------- momentum descent


def test_momentum_descent_minimizes_a_quadratic_deterministically() -> None:
    target = np.array([0.3, -0.2, 0.5])

    def grad(th):
        return 2 * (th - target)

    a = measurable.momentum_descent(grad, np.zeros(3), 300, 0.3, 0.8, 0.1)
    b = measurable.momentum_descent(grad, np.zeros(3), 300, 0.3, 0.8, 0.1)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, target, atol=1e-6)


def test_momentum_descent_converges_on_the_exact_variational_cost() -> None:
    """The settings used by the solver reach ~1e-10 on exact gradients (tuning runs)."""
    n, depth = 3, 2
    plan, theta_prev, _, b = setup(n, depth, False, seed=4)
    f = exact_cost_fn(n, depth, b)
    theta = measurable.momentum_descent(
        lambda t: measurable.parameter_shift_gradient(f, t), theta_prev, 80, 1.0, 0.8
    )
    assert f(theta) < 1e-6 * max(f(theta_prev), 1e-3)


# ---------------------------------------------------- the hardware path purity


def test_hardware_path_runs_with_get_state_disabled(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("cudaq.get_state called in the hardware path")

    monkeypatch.setattr(cudaq, "get_state", forbidden)
    n, depth = 3, 1
    theta0 = np.random.default_rng(0).uniform(-1, 1, V.num_params(n, depth))
    res = measurable.evolve_measurable(
        theta0,
        1.0,
        0.05,
        2,
        1.0,
        0.4,
        L,
        n,
        depth,
        shots=300,
        seed=1,
        iterations=2,
        lr=0.5,
    )
    assert res.thetas.shape == (3, V.num_params(n, depth))
    assert np.all(res.circuits_per_step[1:] == 2 * 2 * V.num_params(n, depth) + 1)
    assert np.all(np.isfinite(res.norms_abs))
