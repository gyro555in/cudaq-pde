"""Classical reference solutions (ground truth for all tests).

Equation: ``u_t + c u_x = nu u_xx`` on a periodic domain ``[0, L)`` sampled at
``N = 2**n`` points ``x_j = j L / N``. For periodic data the solution is exact
in Fourier space: each mode is multiplied by ``exp(-i c k t - nu k^2 t)``.

Nyquist convention
------------------
On ``N = 2**n`` points the integer modes are ``m = -N/2, ..., N/2 - 1``. The
mode ``m = -N/2`` is its own complex conjugate on the grid (``+N/2`` and
``-N/2`` are the same samples), so its wavenumber is a convention. Two
conventions are supported, both exactly unitary when ``nu = 0``:

``"zero"`` (default)
    Odd derivatives (advection) use ``k = 0`` at the Nyquist mode; even
    derivatives (diffusion) keep ``k^2 = (pi N / L)^2``. This is the standard
    spectral-method choice. The phase factor then has the Hermitian symmetry
    ``f(-k) = conj(f(k))`` including the Nyquist mode, so real data stays real.

``"negative"``
    The Nyquist mode uses ``k = -N/2`` (times ``2 pi / L``) like every other
    mode, so ``k(j) = j`` for ``j < N/2`` and ``j - N`` for ``j >= N/2``. This
    is what a circuit built from one phase per qubit applies. Real data in
    general becomes complex whenever the Nyquist component is nonzero.

The two conventions agree exactly on data with no Nyquist content.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

TWO_PI = 2.0 * np.pi

Nyquist = Literal["zero", "negative"]


def _check_nyquist(nyquist: str) -> None:
    if nyquist not in ("zero", "negative"):
        raise ValueError(f"nyquist must be 'zero' or 'negative', got {nyquist!r}")


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


def advection_wavenumbers(
    n: int, L: float = TWO_PI, nyquist: Nyquist = "zero"
) -> NDArray[np.float64]:
    """Wavenumbers used for odd derivatives (advection) under a Nyquist convention.

    Equal to :func:`wavenumbers` except that for ``nyquist="zero"`` the Nyquist
    entry (array index ``N/2``) is set to 0. Even derivatives (diffusion) always
    use ``k^2 = (pi N / L)^2`` at the Nyquist mode and need no such helper.
    """
    _check_nyquist(nyquist)
    k = wavenumbers(n, L)
    if nyquist == "zero":
        k[2 ** (n - 1)] = 0.0
    return k


def evolve_spectral(
    u0: NDArray[np.floating] | NDArray[np.complexfloating],
    t: float,
    c: float,
    nu: float = 0.0,
    L: float = TWO_PI,
    nyquist: Nyquist = "zero",
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Exact solution of ``u_t + c u_x = nu u_xx`` with periodic boundaries.

    Each Fourier mode is multiplied by ``exp(-i c k t - nu k^2 t)``; the only
    ambiguity is the Nyquist mode, see the module docstring.

    Parameters
    ----------
    u0 : ndarray, shape (2**n,)
        Initial values at ``x_j = j L / N``. Real or complex.
    t : float
        Final time (may be negative when ``nu == 0``).
    c : float
        Advection speed.
    nu : float
        Diffusion coefficient, must be >= 0.
    L : float
        Domain length. Wavenumbers scale as ``2 pi / L``.
    nyquist : {"zero", "negative"}
        Nyquist convention, see the module docstring.

    Returns
    -------
    ndarray, shape (2**n,)
        ``u(x_j, t)``. Real (float64) if ``u0`` is real and
        ``nyquist == "zero"``; otherwise complex128 (with ``"negative"`` real
        data generally becomes complex).
    """
    u0 = np.asarray(u0)
    if u0.ndim != 1:
        raise ValueError("u0 must be one-dimensional")
    if not np.all(np.isfinite(u0)):
        raise ValueError("u0 must be finite")
    if nu < 0:
        raise ValueError(f"nu must be >= 0, got {nu}")
    n = _check_size(u0.size)
    k_adv = advection_wavenumbers(n, L, nyquist)
    k_dif = wavenumbers(n, L)
    factor = np.exp(-1j * c * k_adv * t - nu * k_dif**2 * t)
    u = np.fft.ifft(factor * np.fft.fft(u0))
    if nyquist == "zero" and not np.iscomplexobj(u0):
        return u.real
    return u
