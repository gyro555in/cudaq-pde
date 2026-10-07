"""Numerically stable angles and infidelities between nearly equal states.

``1 - F`` with ``F = |<psi|phi>|^2`` subtracts two numbers that agree to within
``~eps`` when the states are close: its absolute error is ``~eps``, so at an angle
``theta`` the relative error is ``eps / theta^2`` and ``arccos(sqrt(F))`` or
``arcsin(sqrt(1 - F))`` has an absolute error FLOOR of ``sqrt(eps) ~ 1.5e-8``. At
``1 - F ~ 1e-10`` (angle 1e-5) that is a relative error of 1e-3 in the angle.

The forms below never form ``1 - F``. For the angle between the rays of ``psi`` and
``phi``::

    theta = atan2( || phi - <psi|phi> psi / ||psi||^2 || ,  |<psi|phi>| / ||psi|| )

the first argument is the component of ``phi`` orthogonal to ``psi``. Its error is the
backward-stable level: components of the inputs carry ``~eps`` rounding of their own, so
the returned angle is exact for inputs perturbed by ``~eps`` per component, i.e. an
ABSOLUTE error of a few ``eps`` (not ``sqrt(eps)``). An error of the dot product itself
lies along ``psi`` and enters the residual only in quadrature. For sign-aligned unit
real states ``2 asin(||psi - phi|| / 2)`` is equivalent. Sums use ``np.sum`` (pairwise,
deterministic) rather than BLAS dot products, whose summation order varies with threads.

What is NOT possible in float64: relative accuracy ``~eps`` of the angle between two
GENERIC stored vectors. Their components are only known to ``~eps`` absolute, so an
angle of 1e-8 has at best a relative error of ``eps / 1e-8 ~ 1e-8``. Relative ``~eps``
needs inputs that are exactly representable (for example axis-aligned pairs).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def deterministic_dot(a: NDArray, b: NDArray) -> NDArray:
    """``<a|b>`` along the last axis with a deterministic (pairwise) summation order."""
    return np.sum(np.conj(a) * b, axis=-1)


def state_angles(psi: NDArray, phi: NDArray) -> NDArray[np.float64]:
    """Angle between the rays of ``psi`` and ``phi`` (last axis is the vector axis).

    Real or complex, normalized or not; invariant under scaling and a global phase of
    either argument. Works on single vectors and on stacks of vectors.
    """
    psi, phi = np.asarray(psi), np.asarray(phi)
    norm_sq = np.sum(np.abs(psi) ** 2, axis=-1)
    inner = deterministic_dot(psi, phi)
    residual = phi - (inner / norm_sq)[..., None] * psi
    perp = np.sqrt(np.sum(np.abs(residual) ** 2, axis=-1))
    return np.arctan2(perp, np.abs(inner) / np.sqrt(norm_sq))


def state_angle(psi: NDArray, phi: NDArray) -> float:
    """Scalar form of :func:`state_angles` for two vectors."""
    return float(state_angles(psi, phi))


def infidelities(psi: NDArray, phi: NDArray) -> NDArray[np.float64]:
    """``1 - |<psi|phi>|^2 / (||psi||^2 ||phi||^2) = sin^2(theta)``, no subtraction."""
    return np.sin(state_angles(psi, phi)) ** 2


def infidelity(psi: NDArray, phi: NDArray) -> float:
    """Scalar form of :func:`infidelities` for two vectors."""
    return float(infidelities(psi, phi))
