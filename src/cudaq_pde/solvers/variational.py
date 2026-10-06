"""Variational time stepping for 1D periodic advection-diffusion.

Solves ``u_t + c u_x = nu u_xx`` with explicit-Euler time stepping on the
central-difference operator, but replaces the exact matrix-vector product by a
variational projection onto a real-amplitude hardware-efficient ansatz
(Lubasch et al., PRA 101, 010301, 2020 style). The validation target is
:func:`cudaq_pde.classical.evolve_fd_euler`, which has the same discretization,
so any difference is optimization or expressibility error.

Per step
--------
The field is ``u_s = norm_s * psi_s`` with ``psi_s`` normalized. The Euler
target is ``u_{s+1} = (I + dt Lfd) u_s = norm_s * b`` with the unnormalized
``b = (I + dt Lfd) psi_s``. The best amplitude ``alpha`` for a candidate
``psi(theta)`` minimizes ``|| alpha psi - norm_s b ||^2``, giving
``alpha = norm_s <psi|b>`` (psi is real) and the residual
``norm_s^2 (||b||^2 - <psi|b>^2)``. Dividing by ``norm_s^2 ||b||^2``::

    C(theta) = 1 - <psi(theta)|b>^2 / ||b||^2

which is exactly the infidelity of ``psi(theta)`` to ``b / ||b||``. The norm is
tracked classically: ``norm_{s+1} = norm_s * <psi(theta*)|b>``. The cost does
not change under ``psi -> -psi`` (a global phase), so ``norm`` is kept as a
signed scalar whose sign carries that phase; ``|norm|`` is
``|norm_s| |<psi|b>|`` and the reconstructed field ``norm * psi`` is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from cudaq_pde import classical, encoding

TWO_PI = classical.TWO_PI
METHODS = ("L-BFGS-B", "COBYLA")


def num_params(n: int, depth: int) -> int:
    """Number of ansatz angles: ``n * (depth + 1)``."""
    return n * (depth + 1)


def init_theta(
    n: int, depth: int, seed: int, scale: float = 0.5
) -> NDArray[np.float64]:
    """Seeded initial angles ``~ N(0, scale^2)`` (deterministic for a given seed)."""
    return np.random.default_rng(seed).normal(0.0, scale, num_params(n, depth))


def fd_target(
    psi_prev: NDArray[np.float64], dt: float, c: float, nu: float, L: float
) -> NDArray[np.float64]:
    """Unnormalized ``b = (I + dt Lfd) psi_prev`` via identity, decrement, increment."""
    n = encoding.num_qubits(psi_prev.size)
    a0, a_dec, a_inc = classical.fd_euler_coefficients(n, dt, c, nu, L)
    return (
        a0 * psi_prev
        + a_dec * classical.decrement(psi_prev)
        + a_inc * classical.increment(psi_prev)
    )


def simulator_cost_and_state(
    theta: NDArray[np.float64], b: NDArray[np.float64], n: int, depth: int
) -> tuple[float, NDArray[np.float64], float]:
    """SIMULATOR-ONLY: cost, state and overlap of the ansatz against target ``b``.

    This is the single place that reads a state vector (``cudaq.get_state``),
    which costs exponentially many shots on hardware. It returns
    ``C = 1 - <psi|b>^2 / ||b||^2``, ``psi(theta)`` and the signed overlap
    ``<psi|b>``.

    Hardware version
    ----------------
    ``I + dt Lfd = a0 I + a_dec * decrement + a_inc * increment`` with the unitary
    cyclic shifts of :func:`cudaq_pde.classical.decrement` (``u_{j+1}``) and
    :func:`cudaq_pde.classical.increment` (``u_{j-1}``); ``a0 = 1-2r``,
    ``a_dec = r-s``, ``a_inc = r+s``, ``r = nu dt/h^2``, ``s = c dt/(2h)``. So for
    real states::

        <psi(theta)|b> = a0 <psi|psi_prev> + a_dec <psi|decrement|psi_prev>
                         + a_inc <psi|increment|psi_prev>.

    Each term is the real part of ``<0|A(theta)^dagger V A(theta_prev)|0>`` with
    ``V in {I, decrement, increment}`` and ``A`` the ansatz, estimated by a
    Hadamard test (ancilla in |+>, controlled ``A^dagger V A_prev``, H, measure Z;
    ``P(0) - P(1)``). All amplitudes are real, so the imaginary part vanishes and
    one circuit per overlap suffices.

    * Distinct overlap circuits per cost evaluation: **3**
      (V = I, decrement, increment).
    * ``||b||^2`` depends on theta not at all. It needs
      ``<psi_prev|decrement|psi_prev>`` and ``<psi_prev|decrement^2|psi_prev>``
      (``increment = decrement^T``), i.e. **2 further circuits per step**,
      evaluated once per step.
    * A finite-difference gradient costs ``P + 1`` cost evaluations with
      ``P = n (depth + 1)``.

    The shifts are the CP3 spectral circuit with phase ``exp(-+ 2 pi i k / N)``
    and the "negative" Nyquist convention (exact for a permutation), costing
    ``n^2 + 2n`` gates; controlling it only needs the n phase gates controlled
    because ``c-(QFT D QFT^dagger) = QFT (c-D) QFT^dagger``.
    """
    import cudaq

    from cudaq_pde.solvers.variational_kernels import hea_state

    raw = np.array(cudaq.get_state(hea_state, n, [float(t) for t in theta], depth))
    if np.max(np.abs(raw.imag)) > 1e-12:
        raise RuntimeError("ansatz state is not real; RY/CX must keep amplitudes real")
    psi = raw.real.copy()
    overlap = float(psi @ b)
    cost = 1.0 - overlap**2 / float(b @ b)
    return cost, psi, overlap


@dataclass
class FitResult:
    """Outcome of one optimization (a time step or the initial fit)."""

    theta: NDArray[np.float64]
    psi: NDArray[np.float64]
    overlap: float
    cost: float
    nfev: int
    nit: int
    success: bool


def optimize_to_target(
    b: NDArray[np.float64],
    n: int,
    depth: int,
    theta0: NDArray[np.float64],
    method: str = "L-BFGS-B",
    tol: float = 1e-8,
    maxiter: int = 500,
) -> FitResult:
    """Minimize ``C(theta)`` against ``b`` with scipy from ``theta0``.

    ``method="L-BFGS-B"`` uses finite-difference gradients (step 1e-8); ``tol``
    is the projected-gradient tolerance (``ftol`` is set to 1e-15 because a
    relative function tolerance is meaningless as ``C -> 0``). ``"COBYLA"`` is
    derivative free; ``tol`` is its final trust-region radius. Both are
    deterministic for fixed ``theta0``. ``nfev`` counts every cost evaluation,
    finite-difference evaluations included.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    calls = 0

    def objective(theta: NDArray[np.float64]) -> float:
        nonlocal calls
        calls += 1
        return simulator_cost_and_state(theta, b, n, depth)[0]

    if method == "L-BFGS-B":
        res = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            options={"gtol": tol, "ftol": 1e-15, "maxiter": maxiter, "eps": 1e-8},
        )
    else:
        res = minimize(
            objective,
            theta0,
            method="COBYLA",
            tol=tol,
            options={"rhobeg": 0.1, "maxiter": maxiter},
        )
    cost, psi, overlap = simulator_cost_and_state(res.x, b, n, depth)
    return FitResult(
        theta=np.asarray(res.x, dtype=np.float64),
        psi=psi,
        overlap=overlap,
        cost=cost,
        nfev=calls,
        nit=int(getattr(res, "nit", 0)),
        success=bool(res.success),
    )


@dataclass
class VariationalResult:
    """Trajectory of the variational solver; row 0 is the initial-state fit.

    ``fields[s] = norms[s] * psis[s]`` is the reconstructed field. ``costs[s]``
    is the final cost of step ``s`` (for ``s = 0`` the infidelity of the fit to
    the encoded initial data), ``nfev[s]`` the cost evaluations spent.
    """

    thetas: NDArray[np.float64]
    psis: NDArray[np.float64]
    norms: NDArray[np.float64]
    fields: NDArray[np.float64]
    costs: NDArray[np.float64]
    nfev: NDArray[np.int64]
    nit: NDArray[np.int64]
    dt: float
    depth: int
    seed: int
    method: str
    tol: float


def evolve_variational(
    u0: NDArray[np.floating],
    dt: float,
    steps: int,
    c: float,
    nu: float,
    L: float = TWO_PI,
    depth: int = 3,
    seed: int = 0,
    method: str = "L-BFGS-B",
    tol: float = 1e-8,
    maxiter: int = 500,
    restarts: int = 1,
) -> VariationalResult:
    """Variational explicit-Euler evolution of ``u0`` for ``steps`` steps.

    SIMULATOR-ONLY (see :func:`simulator_cost_and_state`); the caller selects the
    CUDA-Q target. Row 0 of the result is the ansatz fit of ``encode(u0)``
    (initial-state preparation error), rows ``1..steps`` are Euler steps warm
    started from the previous angles. ``restarts`` only affects the initial fit
    (seeds ``seed + r``, best cost kept).
    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}")
    psi0, norm0 = encoding.encode(np.asarray(u0, dtype=np.float64))
    psi0 = psi0.real
    n = encoding.num_qubits(psi0.size)
    classical.check_fd_euler_stable(n, dt, c, nu, L)

    fits = [
        optimize_to_target(
            psi0, n, depth, init_theta(n, depth, seed + r), method, tol, maxiter
        )
        for r in range(restarts)
    ]
    fit = min(fits, key=lambda f: f.cost)
    nfev0 = sum(f.nfev for f in fits)

    thetas = [fit.theta]
    psis = [fit.psi]
    norms = [norm0 * fit.overlap]
    costs = [fit.cost]
    nfevs = [nfev0]
    nits = [fit.nit]
    for _ in range(steps):
        b = fd_target(psis[-1], dt, c, nu, L)
        step = optimize_to_target(b, n, depth, thetas[-1], method, tol, maxiter)
        thetas.append(step.theta)
        psis.append(step.psi)
        norms.append(norms[-1] * step.overlap)
        costs.append(step.cost)
        nfevs.append(step.nfev)
        nits.append(step.nit)

    norms_arr = np.array(norms)
    psis_arr = np.array(psis)
    return VariationalResult(
        thetas=np.array(thetas),
        psis=psis_arr,
        norms=norms_arr,
        fields=norms_arr[:, None] * psis_arr,
        costs=np.array(costs),
        nfev=np.array(nfevs, dtype=np.int64),
        nit=np.array(nits, dtype=np.int64),
        dt=dt,
        depth=depth,
        seed=seed,
        method=method,
        tol=tol,
    )


def compare_to_reference(
    result: VariationalResult, reference: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-step fidelity and relative L2 error against a reference trajectory.

    Parameters
    ----------
    reference : ndarray, shape (steps + 1, N)
        E.g. the output of :func:`cudaq_pde.classical.evolve_fd_euler`.

    Returns
    -------
    fidelity, l2_error : ndarray, shape (steps + 1,)
        ``|<psi_s|u_ref_s>|^2 / ||u_ref_s||^2`` and
        ``||norm_s psi_s - u_ref_s|| / ||u_ref_s||``.
    """
    ref_norm = np.linalg.norm(reference, axis=1)
    overlap = np.einsum("sj,sj->s", result.psis, reference)
    fidelity = overlap**2 / ref_norm**2
    l2 = np.linalg.norm(result.fields - reference, axis=1) / ref_norm
    return fidelity, l2
