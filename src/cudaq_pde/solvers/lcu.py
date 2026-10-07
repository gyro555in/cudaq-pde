"""Classical bookkeeping for the LCU preparation of ``b_hat`` (no state vectors).

For ``I + dt L = a0 I + a_dec D + a_inc U`` (see :mod:`lcu_kernels`) this module
computes the coefficients, the PREPARE angles and the sign flags handed to the
kernels, and the success probability formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cudaq_pde import classical
from cudaq_pde.solvers.spectral import shift_angles

SELECT_KINDS = {"qft": 0, "ladder": 1}


def n_work(kind: str, n: int) -> int:
    """Work ancillas the SELECT form needs beyond the 2 LCU ancillas."""
    return {"qft": 0, "ladder": n - 1}[kind]


@dataclass(frozen=True)
class LCUPlan:
    """Coefficients and kernel arguments for one explicit-Euler step."""

    a0: float
    a_dec: float  # coefficient of the decrement (u_{j+1}), r - s
    a_inc: float  # coefficient of the increment (u_{j-1}), r + s
    lam: float  # sum of |coefficients|
    prep: list[float]  # [theta_h, theta_0]
    neg: list[int]  # [a_dec < 0, a_inc < 0]
    dec_angles: list[float]  # phase layer of the decrement (QFT SELECT)
    inc_angles: list[float]  # phase layer of the increment (QFT SELECT)

    def success_probability(self, b_norm_sq: float) -> float:
        """``||b||^2 / lambda^2`` for the unnormalized target ``b``."""
        return b_norm_sq / self.lam**2


def lcu_plan(n: int, dt: float, c: float, nu: float, L: float) -> LCUPlan:
    """LCU plan for ``I + dt L`` on ``N = 2**n`` points (explicit Euler, stable)."""
    classical.check_fd_euler_stable(n, dt, c, nu, L)
    a0, a_dec, a_inc = classical.fd_euler_coefficients(n, dt, c, nu, L)
    if a0 < 0.0:
        raise ValueError(f"a0 = {a0} < 0; outside the explicit-Euler stable range")
    lam = abs(a0) + abs(a_dec) + abs(a_inc)
    alpha = np.sqrt(np.array([abs(a0), abs(a_dec), abs(a_inc)]) / lam)
    theta_h = 2.0 * np.arctan2(alpha[2], np.hypot(alpha[0], alpha[1]))
    theta_0 = 2.0 * np.arctan2(alpha[1], alpha[0])
    return LCUPlan(
        a0=a0,
        a_dec=a_dec,
        a_inc=a_inc,
        lam=float(lam),
        prep=[float(theta_h), float(theta_0)],
        neg=[int(a_dec < 0.0), int(a_inc < 0.0)],
        dec_angles=shift_angles(n, -1),
        inc_angles=shift_angles(n, +1),
    )
