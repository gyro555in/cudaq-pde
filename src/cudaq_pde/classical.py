"""Classical reference solutions (ground truth for all tests).

Equation: ``u_t + c u_x = nu u_xx`` on a periodic domain ``[0, L)`` sampled at
``N = 2**n`` points ``x_j = j L / N``. For periodic data the solution is exact
in Fourier space: each mode is multiplied by ``exp(-i c k t - nu k^2 t)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

TWO_PI = 2.0 * np.pi


def _check_size(N: int) -> int:
    if N < 2 or N & (N - 1):
        raise ValueError(f"grid size must be a power of two >= 2, got {N}")
    return N.bit_length() - 1


def grid(n: int, L: float = TWO_PI) -> NDArray[np.float64]:
    """Grid points ``x_j = j L / N`` for ``N = 2**n`` (endpoint excluded)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return np.arange(2**n) * (L / 2**n)


def wavenumbers(n: int, L: float = TWO_PI) -> NDArray[np.float64]:
    """Wavenumbers ``k_j`` in FFT array order for ``N = 2**n`` points.

    ``k_j = 2 pi m_j / L`` with integer ``m_j = fftfreq`` order:
    ``0, 1, ..., N/2 - 1, -N/2, ..., -1``. Array index ``j`` of the Fourier
    transform therefore carries ``m = j`` for ``j < N/2`` and ``m = j - N`` for
    ``j >= N/2``. The Nyquist mode is ``m = -N/2``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    N = 2**n
    m = np.fft.fftfreq(N, d=1.0 / N)
    return TWO_PI / L * m


def evolve_spectral(
    u0: NDArray[np.floating],
    t: float,
    c: float,
    nu: float = 0.0,
    L: float = TWO_PI,
    real: bool = True,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Exact solution of ``u_t + c u_x = nu u_xx`` with periodic boundaries.

    Parameters
    ----------
    u0 : ndarray of float, shape (2**n,)
        Initial values at ``x_j = j L / N``.
    t : float
        Final time (may be negative when ``nu == 0``).
    c : float
        Advection speed.
    nu : float
        Diffusion coefficient, must be >= 0.
    L : float
        Domain length.
    real : bool
        The Nyquist mode ``m = -N/2`` has no distinct conjugate partner, so for
        ``c != 0`` its phase makes the raw result slightly complex. If True
        (default) return the real part, which equals averaging the ``+N/2`` and
        ``-N/2`` conventions. If False return the complex array, so that a
        circuit that applies the same phase to every basis state can be
        compared with an identical convention.

    Returns
    -------
    ndarray, shape (2**n,)
        ``u(x_j, t)``.
    """
    u0 = np.asarray(u0)
    if u0.ndim != 1:
        raise ValueError("u0 must be one-dimensional")
    if not np.all(np.isfinite(u0)):
        raise ValueError("u0 must be finite")
    if nu < 0:
        raise ValueError(f"nu must be >= 0, got {nu}")
    n = _check_size(u0.size)
    k = wavenumbers(n, L)
    factor = np.exp(-1j * c * k * t - nu * k**2 * t)
    u = np.fft.ifft(factor * np.fft.fft(u0))
    return u.real if real else u
