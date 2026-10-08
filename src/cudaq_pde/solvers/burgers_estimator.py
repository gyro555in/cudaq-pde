"""Pointwise velocity from sampled bin counts (numpy only, no CUDA-Q).

``phi > 0`` so ``phi_j = sqrt(p_j)`` and ``u = -2 nu (ln phi)_x`` with the central
difference on the grid gives

    u_j = -(nu / 2h) ln(p_{j+1} / p_{j-1}) = -(nu / 2h) ln(c_{j+1} / c_{j-1}),

where ``c`` are raw accepted counts: the norm of ``phi`` AND the acceptance probability
cancel in the ratio. Statistics (multinomial over ``S`` TOTAL shots; ``q_a = P_succ
p_a`` is the probability of "accepted and bin a", ``lambda_a = S q_a`` the expected
count):

* ``Var(ln c_a - ln c_b) = (1 - q_a)/(S q_a) + (1 - q_b)/(S q_b) + 2/S
  = 1/(S q_a) + 1/(S q_b)``  (delta method; the covariance ``-1/S`` of the two logs
  cancels the ``-1/S`` terms), so ``Var(u_hat) = (nu/2h)^2 (1/lambda_a + 1/lambda_b)``.
* ``E ln c ~ ln lambda - (1 - q)/(2 lambda)``, so the bias of ``u_hat`` is
  ``(nu / 4 h S)(1/q_{j+1} - 1/q_{j-1})``.
* A zero count makes ``ln c`` undefined. The plain estimator then returns NaN and is
  counted in ``n_undefined``; the Haldane-Anscombe estimator (``c + 1/2``) is always
  defined and removes the leading bias, but is not trustworthy at tiny counts either.
  Every estimate therefore carries ``valid = min(c_a, c_b) >= c_min`` and a plug-in
  standard deviation. Monte Carlo with exact multinomial draws (tests) puts the
  delta-method variance within 5% of the truth from a smaller expected count of 30 (16%
  off at 10, wrong at 1), hence ``c_min = 30``.

The shots needed for a standard error ``eps`` are
``S = (nu/2h)^2 (1/q_a + 1/q_b) / eps^2``: with ``q = P_succ p``, the cost of a probe is
set by the smaller ``p`` next to it, which is the dynamic range of ``phi``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cudaq_pde.classical import TWO_PI


def neighbours(probes: NDArray[np.integer] | list[int], N: int):
    """Indices ``j+1`` and ``j-1`` (periodic) of the probe points."""
    j = np.asarray(probes, dtype=np.int64)
    return (j + 1) % N, (j - 1) % N


def velocity_variance(
    q_plus: NDArray, q_minus: NDArray, shots: float, nu: float, h: float
) -> NDArray[np.float64]:
    """Delta-method variance of ``u_hat``: ``(nu/2h)^2 (1/(S q+) + 1/(S q-))``."""
    qp, qm = np.asarray(q_plus, float), np.asarray(q_minus, float)
    return (nu / (2.0 * h)) ** 2 * (1.0 / (shots * qp) + 1.0 / (shots * qm))


def velocity_bias(
    q_plus: NDArray, q_minus: NDArray, shots: float, nu: float, h: float
) -> NDArray[np.float64]:
    """Second-order bias of the plain estimator: ``(nu/4hS)(1/q+ - 1/q-)``."""
    qp, qm = np.asarray(q_plus, float), np.asarray(q_minus, float)
    return nu / (4.0 * h * shots) * (1.0 / qp - 1.0 / qm)


def shots_for_error(
    q_plus: NDArray, q_minus: NDArray, eps: float, nu: float, h: float
) -> NDArray[np.float64]:
    """Total shots for a standard error ``eps`` of ``u_hat`` (delta method)."""
    qp, qm = np.asarray(q_plus, float), np.asarray(q_minus, float)
    return (nu / (2.0 * h)) ** 2 * (1.0 / qp + 1.0 / qm) / eps**2


def probe_probabilities(
    p: NDArray, p_succ: float, probes: NDArray | list[int]
) -> tuple[NDArray, NDArray]:
    """Joint probabilities ``q+ = P_succ p_{j+1}``, ``q- = P_succ p_{j-1}``."""
    jp, jm = neighbours(probes, np.size(p))
    p = np.asarray(p, float)
    return p_succ * p[jp], p_succ * p[jm]


@dataclass(frozen=True)
class VelocityEstimate:
    """Estimates at probe points (moving frame for a nonzero-mean problem)."""

    probes: NDArray[np.int64]
    u_plain: NDArray[np.float64]  # NaN where either neighbour count is zero
    u_haldane: NDArray[np.float64]  # (c + 1/2): always defined
    sigma_plugin: NDArray[np.float64]  # (nu/2h) sqrt(1/c+ + 1/c-), inf if a count is 0
    c_plus: NDArray[np.int64]
    c_minus: NDArray[np.int64]
    valid: NDArray[np.bool_]  # min count >= c_min
    n_undefined: int  # probes where the plain estimate is undefined (a zero count)


def velocity_estimate(
    bin_counts: NDArray[np.integer],
    probes: NDArray[np.integer] | list[int],
    nu: float,
    L: float = TWO_PI,
    c_min: int = 30,
) -> VelocityEstimate:
    """``u_hat`` at ``probes`` from accepted counts per grid bin."""
    counts = np.asarray(bin_counts, dtype=np.int64)
    h = L / counts.size
    jp, jm = neighbours(probes, counts.size)
    cp, cm = counts[jp], counts[jm]
    with np.errstate(divide="ignore", invalid="ignore"):
        plain = np.where(
            (cp > 0) & (cm > 0),
            -(nu / (2.0 * h)) * (np.log(np.maximum(cp, 1)) - np.log(np.maximum(cm, 1))),
            np.nan,
        )
        sigma = np.where(
            (cp > 0) & (cm > 0),
            (nu / (2.0 * h))
            * np.sqrt(1.0 / np.maximum(cp, 1) + 1.0 / np.maximum(cm, 1)),
            np.inf,
        )
    haldane = -(nu / (2.0 * h)) * (np.log(cp + 0.5) - np.log(cm + 0.5))
    return VelocityEstimate(
        probes=np.asarray(probes, dtype=np.int64),
        u_plain=plain,
        u_haldane=haldane,
        sigma_plugin=sigma,
        c_plus=cp,
        c_minus=cm,
        valid=np.minimum(cp, cm) >= c_min,
        n_undefined=int(np.sum(~np.isfinite(plain))),
    )
