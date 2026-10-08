"""Dynamic-range and spectrum studies for the Cole-Hopf Burgers pipeline.

``phi = exp(-int u / 2nu)`` spans ``exp(max|int u| / 2nu)`` in amplitude, so
``p = phi^2`` spans its square, and the shots needed for a pointwise velocity grow with
``1/p_min`` (see :mod:`burgers_estimator`). :func:`study_row` quantifies, for one
``(case, nu, n, T)``:

* ``min_p_true`` from the EXACT ``ln phi`` (log domain) against ``min_p`` of the
  double-precision grid solution; ``n_unresolved`` counts grid points where the grid
  ``p`` is non-positive or off by more than 10% in ``ln p`` (aliasing and roundoff: the
  heat kernel cannot damp mode ``N/2`` enough to produce a valley of ``e^{-1/nu}``);
* the error terms at the probes: grid/aliasing (spectral ``u`` of the grid ``phi``
  against the exact solution) and finite difference (FD against spectral); the shot
  noise is the delta-method scale of :mod:`burgers_estimator`;
* shots for a target standard error ``eps`` at each probe (delta method), and
  ``shots_to_see_all``: shots for ``c_min`` expected counts in the rarest bin.

Cases are zero-mean initial velocities on ``[0, 2 pi)``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from cudaq_pde import burgers as B
from cudaq_pde import classical
from cudaq_pde.classical import TWO_PI
from cudaq_pde.solvers import burgers_estimator as be
from cudaq_pde.solvers import burgers_quantum as bq
from cudaq_pde.solvers.spectral_measure import (
    classical_power_spectrum,
    expected_rms_error,
)

CASES: dict[str, Callable[[NDArray], NDArray]] = {
    "sin": np.sin,
    "asym": lambda x: np.sin(x) + 0.5 * np.sin(2 * x),
}
FP64_AMPLITUDE_FLOOR = 1e-15  # sqrt(p) below this is below double-precision resolution
NOISE_FLOOR = 1e-3
"""Default hardware noise floor on a bin probability: a MODELLING ASSUMPTION for the
combined gate and readout error per shot, to be replaced by device calibration data. A
probe whose smaller neighbour probability ``min(p_{j+1}, p_{j-1})`` is below it is
unresolvable regardless of the number of shots (the bin is dominated by error
counts)."""


def probe_indices(n: int) -> NDArray[np.int64]:
    """Probes at ``x = pi/4, pi/2, 3 pi/4`` (never the shock centre ``pi``)."""
    N = 2**n
    return np.array([N // 8, N // 4, 3 * N // 8])


def true_probabilities(
    w0: Callable[[NDArray], NDArray], nu: float, T: float, n: int, L: float = TWO_PI
) -> NDArray[np.float64]:
    """Normalized exact ``p_j = phi(x_j, T)^2 / sum`` from the log-domain solution."""
    x = classical.grid(n, L)
    lp = 2.0 * B.exact_log_phi(x, w0, nu, T, L)
    lp -= lp.max()
    p = np.exp(lp)
    return p / p.sum()


def study_row(
    case: str,
    nu: float,
    n: int,
    T: float = 1.0,
    eps: float = 0.05,
    c_min: int = 30,
    L: float = TWO_PI,
    noise_floor: float = NOISE_FLOOR,
) -> dict:
    """One row of the dynamic-range table (classical formulas, no sampling).

    ``resolvable_on_hardware`` is per probe: ``min(p_{j+1}, p_{j-1}) >= noise_floor``
    with ``p`` the distribution the circuit samples (see :data:`NOISE_FLOOR`; a
    modelling assumption, not a device measurement).
    """
    w0f = CASES[case]
    x = classical.grid(n, L)
    d = B.discrete_burgers(w0f(x), nu, T, L)
    p_true = true_probabilities(w0f, nu, T, n, L)
    probes = probe_indices(n)
    h = L / 2**n
    with np.errstate(divide="ignore", invalid="ignore"):
        log_off = np.abs(np.log(np.where(d.p > 0, d.p, np.nan)) - np.log(p_true))
    unresolved = ~(log_off <= 0.1)  # NaN (p <= 0) counts as unresolved
    u_exact = B.exact_cole_hopf(x[probes], w0f, nu, T, L)
    qp, qm = be.probe_probabilities(d.p, d.p_succ, probes)
    with np.errstate(divide="ignore", invalid="ignore"):
        shots = be.shots_for_error(qp, qm, eps, nu, h)
    jp, jm = be.neighbours(probes, 2**n)
    p_nb = np.minimum(d.p[jp], d.p[jm])
    q_min = d.p_succ * float(np.min(d.p[d.p > 0])) if np.any(d.p > 0) else 0.0
    return {
        "case": case, "nu": nu, "n": n, "T": T, "eps": eps,
        "p_succ": d.p_succ,
        "min_p_grid": float(d.p.min()),
        "min_p_true": float(p_true.min()),
        "sqrt_min_p_true": float(np.sqrt(p_true.min())),
        "below_fp64_floor": bool(np.sqrt(p_true.min()) < FP64_AMPLITUDE_FLOOR),
        "n_unresolved": int(unresolved.sum()),
        "resolved": bool(unresolved.sum() == 0),
        "probes": probes.tolist(),
        "u_exact": u_exact.tolist(),
        "err_grid": np.abs(d.u_spectral[probes] - u_exact).tolist(),
        "err_fd": np.abs(d.u_fd[probes] - d.u_spectral[probes]).tolist(),
        "err_total": np.abs(d.u_fd[probes] - u_exact).tolist(),
        "shots_per_probe": shots.tolist(),
        "shots_best": float(np.nanmin(shots)),
        "shots_worst": float(np.nanmax(shots)),
        "noise_floor": noise_floor,
        "p_neighbour_min": p_nb.tolist(),
        "resolvable_on_hardware": (p_nb >= noise_floor).tolist(),
        "shots_to_see_all": c_min / q_min if q_min > 0 else float("inf"),
    }  # fmt: skip


def sampled_confirmation(
    case: str,
    nu: float,
    n: int,
    T: float,
    probe: int,
    shots: int,
    seeds: int = 5,
    route: str = "A",
    L: float = TWO_PI,
    base_seed: int = 1000,
) -> dict:
    """Real CUDA-Q shots: scatter of ``u_hat`` at one probe vs the delta prediction.

    The reference is the statevector-equivalent classical ``u_fd`` of the grid solution
    (what an ideal circuit samples), so the z-scores isolate shot noise.
    """
    w0f = CASES[case]
    x = classical.grid(n, L)
    d = B.discrete_burgers(w0f(x), nu, T, L)
    h = L / 2**n
    qp, qm = be.probe_probabilities(d.p, d.p_succ, [probe])
    sd = float(np.sqrt(be.velocity_variance(qp, qm, shots, nu, h))[0])
    errs = []
    for k in range(seeds):
        s = bq.sample_heat(d.phi0, nu, T, L, route, shots, seed=base_seed + k)
        est = be.velocity_estimate(s.grid_counts, [probe], nu, L)
        errs.append(float(est.u_plain[0] - d.u_fd[probe]))
    errs = np.array(errs)
    return {
        "probe": probe, "shots": shots, "predicted_sigma": sd,
        "rms_error": float(np.sqrt(np.mean(errs**2))),
        "max_abs_z": float(np.max(np.abs(errs)) / sd),
    }  # fmt: skip


def spectrum_row(
    case: str,
    nu: float,
    n: int,
    T: float,
    shots: int,
    seed: int = 0,
    route: str = "A",
    L: float = TWO_PI,
) -> dict:
    """Sampled power spectrum of ``phi(T)`` against the classical one."""
    w0f = CASES[case]
    x = classical.grid(n, L)
    d = B.discrete_burgers(w0f(x), nu, T, L)
    s = bq.sample_heat(d.phi0, nu, T, L, route, shots, seed=seed, fourier=True)
    P_hat = s.fft_counts() / s.accepted
    P_cl = classical_power_spectrum(d.phiT)
    m = np.fft.fftfreq(2**n, 1.0 / 2**n)
    k2 = (TWO_PI / L * m) ** 2
    return {
        "case": case, "nu": nu, "n": n, "T": T, "shots": shots,
        "accepted": s.accepted,
        "rms_error": float(np.sqrt(np.sum((P_hat - P_cl) ** 2))),
        "rms_predicted": expected_rms_error(P_cl, s.accepted),
        "tv_distance": float(0.5 * np.sum(np.abs(P_hat - P_cl))),
        "weighted_msq_velocity_hat": float(4 * nu**2 * np.sum(k2 * P_hat)),
        "weighted_msq_velocity_classical": float(4 * nu**2 * np.sum(k2 * P_cl)),
    }  # fmt: skip
