"""Shot-based (hardware-measurable) variational advection-diffusion.

Nothing here reads a state vector: every quantity comes from ``cudaq.sample`` counts
through :mod:`cudaq_pde.solvers.sampling`.

Cost and norm
-------------
With ``b = (I + dt L) psi_prev`` and ``lambda = sum |c_i|`` of the LCU, the circuit of
:mod:`measurable_kernels` gives the two probabilities

    P_joint = P(sys = 0, anc = 00) = <psi|b>^2 / lambda^2,
    P_succ  = P(anc = 00)          = ||b||^2 / lambda^2,

so ``C = 1 - P_joint / P_succ`` and ``|<psi|b>| = lambda sqrt(P_joint)``. The classical
norm is tracked as ``|norm_new| = |norm_prev| lambda sqrt(P_joint)`` (from
``norm_new = norm_prev <psi(theta*)|b>``). The sign of the overlap is a global phase of
the state and not observable, so this path tracks magnitudes and keeps the sign of the
initial state.

Gradients are exact parameter-shift gradients (every RY angle appears once and
``C = <psi|(I - |b_hat><b_hat|)|psi>`` is an expectation value):
``dC/dtheta_i = [C(theta_i + pi/2) - C(theta_i - pi/2)] / 2``, i.e. ``2 P`` circuits per
gradient. The optimizer is heavy-ball (momentum) descent with a linearly decaying
learning rate and a fixed number of iterations (:func:`momentum_descent`; L-BFGS-B and
COBYLA do not cope with shot noise, and Adam reached only 3e-6 on exact gradients
against 4e-11 for heavy ball).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cudaq_pde.solvers import lcu
from cudaq_pde.solvers.measurable_kernels import lcu_cost_circuit
from cudaq_pde.solvers.sampling import Sampler


@dataclass(frozen=True)
class SampledCost:
    """Estimates from one sampled cost circuit."""

    cost: float  # 1 - N(sys=0, anc=00) / N(anc=00); 1.0 if no shot post-selects
    p_joint: float  # N(sys=0, anc=00) / shots
    n_succ: int  # shots with the LCU ancillas in |00>
    shots: int

    @property
    def p_succ(self) -> float:
        return self.n_succ / self.shots


def counts_to_estimate(counts: dict[str, int], n: int, shots: int) -> SampledCost:
    """Reduce counts of ``lcu_cost_circuit`` to the post-selected cost estimate.

    A bitstring is ``n`` system bits, 2 LCU ancilla bits, then the work qubits (always
    ``0``); character ``k`` is qubit ``k``.
    """
    n_succ = 0
    n_zero = 0
    for bits, count in counts.items():
        if bits[n : n + 2] == "00":
            n_succ += count
            if bits[:n] == "0" * n:
                n_zero += count
    cost = (n_succ - n_zero) / n_succ if n_succ else 1.0  # exact integer difference
    return SampledCost(cost=cost, p_joint=n_zero / shots, n_succ=n_succ, shots=shots)


def circuit_arguments(
    plan: lcu.LCUPlan,
    n: int,
    depth: int,
    theta_prev: NDArray[np.float64],
    theta: NDArray[np.float64],
    select: str,
) -> tuple:
    """Arguments of :func:`lcu_cost_circuit` for one cost evaluation."""
    return (
        n,
        [float(t) for t in theta_prev],
        [float(t) for t in theta],
        depth,
        plan.prep,
        plan.neg,
        lcu.SELECT_KINDS[select],
        lcu.n_work(select, n),
        plan.dec_angles,
        plan.inc_angles,
    )


def sampled_cost(
    sampler: Sampler,
    plan: lcu.LCUPlan,
    n: int,
    depth: int,
    theta_prev: NDArray[np.float64],
    theta: NDArray[np.float64],
    shots: int,
    select: str = "qft",
) -> SampledCost:
    """Estimate ``C(theta)`` against ``b(theta_prev)`` with ``shots`` shots."""
    args = circuit_arguments(plan, n, depth, theta_prev, theta, select)
    counts = sampler.counts(lcu_cost_circuit, *args, shots=shots)
    return counts_to_estimate(counts, n, shots)


def parameter_shift_gradient(
    cost_fn: Callable[[NDArray[np.float64]], float],
    theta: NDArray[np.float64],
    shift: float = np.pi / 2,
) -> NDArray[np.float64]:
    """``dC/dtheta_i = [C(theta_i + shift) - C(theta_i - shift)] / 2`` (``2 P`` calls).

    Exact for a cost that is an expectation value in a state prepared with each angle in
    one RY gate (shift ``pi/2``). ``cost_fn`` may be exact or sampled.
    """
    theta = np.asarray(theta, dtype=np.float64)
    grad = np.empty_like(theta)
    for i in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += shift
        minus[i] -= shift
        grad[i] = 0.5 * (cost_fn(plus) - cost_fn(minus))
    return grad


def momentum_descent(
    grad_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    theta0: NDArray[np.float64],
    iterations: int,
    lr: float,
    momentum: float = 0.8,
    lr_final_fraction: float = 1.0,
) -> NDArray[np.float64]:
    """Heavy-ball gradient descent (deterministic for a given ``grad_fn``).

    ``v <- momentum v - rate grad; theta <- theta + v`` with the rate decaying linearly
    from ``lr`` to ``lr * lr_final_fraction``. The cost is a smooth, nearly quadratic
    bowl whose Hessian has largest eigenvalue about 1.4 (n = 3, depth 2), so
    ``lr = 1`` with momentum 0.8 is stable (limit ``2 (1 + momentum) / lambda_max``)
    and converged to 1e-10 in 80 exact-gradient iterations in the tuning runs. The
    last iterate is returned (a noisy cost gives no reliable "best").
    """
    theta = np.array(theta0, dtype=np.float64)
    v = np.zeros_like(theta)
    for k in range(iterations):
        frac = k / max(iterations - 1, 1)
        rate = lr * (1.0 - (1.0 - lr_final_fraction) * frac)
        v = momentum * v - rate * grad_fn(theta)
        theta = theta + v
    return theta


@dataclass
class MeasurableStep:
    """Result of one measurable time step."""

    theta: NDArray[np.float64]
    norm_abs: float  # |norm_new| = |norm_prev| lambda sqrt(P_joint)
    cost_est: float  # sampled C at theta* (final_shots)
    p_joint_est: float
    p_succ_est: float
    circuits: int
    shots: int


def measurable_step(
    sampler: Sampler,
    plan: lcu.LCUPlan,
    n: int,
    depth: int,
    theta_prev: NDArray[np.float64],
    norm_prev_abs: float,
    shots: int,
    iterations: int,
    lr: float,
    final_shots: int,
    momentum: float = 0.8,
    lr_final_fraction: float = 1.0,
    select: str = "qft",
) -> MeasurableStep:
    """One Euler step: momentum descent on parameter-shift gradients, warm start.

    Circuits: ``iterations * 2P`` gradient circuits of ``shots`` shots each, plus one
    final circuit of ``final_shots`` shots that estimates the cost and the success
    probability used for norm tracking.
    """
    circuits0, shots0 = sampler.circuits, sampler.shots

    def cost_fn(theta: NDArray[np.float64]) -> float:
        return sampled_cost(
            sampler, plan, n, depth, theta_prev, theta, shots, select
        ).cost

    theta = momentum_descent(
        lambda th: parameter_shift_gradient(cost_fn, th),
        theta_prev,
        iterations,
        lr,
        momentum,
        lr_final_fraction,
    )
    final = sampled_cost(
        sampler, plan, n, depth, theta_prev, theta, final_shots, select
    )
    return MeasurableStep(
        theta=theta,
        norm_abs=norm_prev_abs * plan.lam * float(np.sqrt(final.p_joint)),
        cost_est=final.cost,
        p_joint_est=final.p_joint,
        p_succ_est=final.p_succ,
        circuits=sampler.circuits - circuits0,
        shots=sampler.shots - shots0,
    )


@dataclass
class MeasurableResult:
    """Trajectory of the measurable solver; row 0 is the starting state."""

    thetas: NDArray[np.float64]
    norms_abs: NDArray[np.float64]
    costs_est: NDArray[np.float64]
    p_joint_est: NDArray[np.float64]
    p_succ_est: NDArray[np.float64]
    circuits_per_step: NDArray[np.int64]
    shots_per_step: NDArray[np.int64]


def evolve_measurable(
    theta0: NDArray[np.float64],
    norm0_abs: float,
    dt: float,
    steps: int,
    c: float,
    nu: float,
    L: float,
    n: int,
    depth: int,
    shots: int,
    seed: int,
    iterations: int = 60,
    lr: float = 1.0,
    final_shots: int | None = None,
    momentum: float = 0.8,
    lr_final_fraction: float = 0.1,
    select: str = "qft",
) -> MeasurableResult:
    """Shot-based evolution from ansatz angles ``theta0`` (classical preprocessing).

    ``theta0`` and ``norm0_abs`` describe the initial state (for example from the CP4
    simulator fit of the initial data); loading data on hardware is exponential and
    outside this path. The explicit-Euler stability limit is asserted by
    :func:`lcu.lcu_plan`.
    """
    plan = lcu.lcu_plan(n, dt, c, nu, L)
    sampler = Sampler(base_seed=seed)
    final_shots = shots if final_shots is None else final_shots
    thetas = [np.asarray(theta0, dtype=np.float64)]
    norms = [float(norm0_abs)]
    costs, pj, ps = [np.nan], [np.nan], [np.nan]
    circuits, shot_counts = [0], [0]
    for _ in range(steps):
        step = measurable_step(
            sampler,
            plan,
            n,
            depth,
            thetas[-1],
            norms[-1],
            shots,
            iterations,
            lr,
            final_shots,
            momentum,
            lr_final_fraction,
            select,
        )
        thetas.append(step.theta)
        norms.append(step.norm_abs)
        costs.append(step.cost_est)
        pj.append(step.p_joint_est)
        ps.append(step.p_succ_est)
        circuits.append(step.circuits)
        shot_counts.append(step.shots)
    return MeasurableResult(
        thetas=np.array(thetas),
        norms_abs=np.array(norms),
        costs_est=np.array(costs),
        p_joint_est=np.array(pj),
        p_succ_est=np.array(ps),
        circuits_per_step=np.array(circuits, dtype=np.int64),
        shots_per_step=np.array(shot_counts, dtype=np.int64),
    )
