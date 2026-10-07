"""LCU preparation of b_hat vs one explicit-Euler step (qpp-cpu fp64).

STATEVECTOR CHECKS (test only): the post-selected state is read from the full
state vector, which a device cannot do. The kernels never read a state.
"""

from collections.abc import Iterator

import cudaq
import numpy as np
import pytest

from cudaq_pde import classical
from cudaq_pde.solvers import lcu
from cudaq_pde.solvers import variational as V
from cudaq_pde.solvers.ansatz_numpy import numpy_ansatz
from cudaq_pde.solvers.lcu_kernels import lcu_target

L = classical.TWO_PI
TOL = 1e-12
# (n, c, nu, dt-fraction-of-limit): the last case has a negative a_dec (a+ = r - s)
CASES = [(3, 1.0, 0.4, 0.05), (4, 1.0, 0.4, 0.05), (5, 0.7, 0.3, 0.05)]
NEG_CASE = (3, 1.2, 0.05, 0.05)


@pytest.fixture(autouse=True)
def cpu_target() -> Iterator[None]:
    cudaq.set_target("qpp-cpu")
    yield
    cudaq.reset_target()


def run_lcu(n, dt, c, nu, depth, theta, kind):
    plan = lcu.lcu_plan(n, dt, c, nu, L)
    state = np.array(
        cudaq.get_state(
            lcu_target,
            n,
            [float(t) for t in theta],
            depth,
            plan.prep,
            plan.neg,
            lcu.SELECT_KINDS[kind],
            lcu.n_work(kind, n),
            plan.dec_angles,
            plan.inc_angles,
        )
    )
    # STATEVECTOR CHECK: ancilla qubits n, n+1 are the high bits of the index (work
    # qubits are higher still), so the |00> post-selected block with the work qubits
    # in |0> is the first 2**n amplitudes; the rest must hold no weight there.
    post = state[: 2**n]
    work_block = state[2 ** (n + 2) :]
    assert work_block.size == 0 or np.max(np.abs(work_block)) < TOL
    return plan, post


@pytest.mark.parametrize("kind", ["qft", "ladder"])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("case", CASES + [NEG_CASE])
def test_postselected_state_equals_fd_euler_step(case, depth: int, kind: str) -> None:
    n, c, nu, dt = case
    theta = np.random.default_rng(n * 10 + depth).uniform(
        -np.pi, np.pi, V.num_params(n, depth)
    )
    plan, post = run_lcu(n, dt, c, nu, depth, theta, kind)
    psi_prev = numpy_ansatz(n, theta, depth)
    step = classical.evolve_fd_euler(psi_prev, dt, 1, c, nu, L)[1]
    b_hat = step / np.linalg.norm(step)
    assert np.max(np.abs(post.imag)) < TOL
    np.testing.assert_allclose(post.real / np.linalg.norm(post), b_hat, atol=TOL)
    # success probability = ||b||^2 / (sum |coefficients|)^2
    b = V.fd_target(psi_prev, dt, c, nu, L)
    assert np.linalg.norm(post) ** 2 == pytest.approx(
        plan.success_probability(b @ b), abs=TOL
    )
    # the unnormalized block is exactly b / lambda
    np.testing.assert_allclose(post.real, b / plan.lam, atol=TOL)


def test_negative_coefficient_case_is_really_negative() -> None:
    n, c, nu, dt = NEG_CASE
    plan = lcu.lcu_plan(n, dt, c, nu, L)
    assert plan.a_dec < 0 < plan.a_inc and plan.neg == [1, 0]
    assert plan.lam == pytest.approx(abs(plan.a0) + abs(plan.a_dec) + abs(plan.a_inc))
    assert plan.lam > 1.0  # negative coefficient costs success probability


def test_all_positive_case_has_lambda_one() -> None:
    plan = lcu.lcu_plan(3, 0.05, 1.0, 0.4, L)
    assert plan.lam == pytest.approx(1.0, abs=1e-14)  # a0 + a+ + a- = 1
    assert plan.neg == [0, 0]


def test_wrong_direction_would_be_detected() -> None:
    """Negative control: swapping the decrement and increment weights must fail."""
    n, c, nu, dt = 3, 1.0, 0.4, 0.05
    theta = np.random.default_rng(1).uniform(-np.pi, np.pi, V.num_params(n, 1))
    psi_prev = numpy_ansatz(n, theta, 1)
    plan = lcu.lcu_plan(n, dt, c, nu, L)
    swapped = (
        plan.a0 * psi_prev
        + plan.a_inc * np.roll(psi_prev, -1)  # wrong: increment weight on u_{j+1}
        + plan.a_dec * np.roll(psi_prev, 1)
    )
    step = classical.evolve_fd_euler(psi_prev, dt, 1, c, nu, L)[1]
    assert np.max(np.abs(swapped - step)) > 1e-4


def test_plan_requires_stability() -> None:
    with pytest.raises(ValueError, match="unstable"):
        lcu.lcu_plan(3, 10.0, 1.0, 0.4, L)
