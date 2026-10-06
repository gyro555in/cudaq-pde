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

Wavenumber layout
-----------------
Array index ``j`` of the FFT carries integer mode ``m(j) = j`` for
``j < N/2`` and ``m(j) = j - N`` for ``j >= N/2``, so the Nyquist mode sits at
``j = N/2`` with ``m = -N/2``. The wavenumber is ``k = 2 pi m / L``. With
CUDA-Q qubit 0 as the least significant bit this index ``j`` is the
computational basis state ``|j>``.

No real-part projection
-----------------------
An earlier version returned the real part of the result for real input. That
is a non-unitary step a circuit cannot perform, so a reference containing it
cannot be matched by a unitary kernel. The reference now applies only the
diagonal phase/decay factor in Fourier space. For ``"zero"`` with real input
the result is exactly real by symmetry (the cast to float64 drops an imaginary
part of order 1e-17 that is round-off, verified in the tests), not a
projection of a genuinely complex result.
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
        ``nyquist == "zero"`` (exactly real by Hermitian symmetry, the cast
        only drops round-off); otherwise complex128 (with ``"negative"`` real
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


def increment(u: NDArray) -> NDArray:
    """Cyclic shift ``(increment u)_j = u_{j-1}``, i.e. ``np.roll(u, +1)``.

    On computational basis states ``|j> -> |j + 1 mod N>`` (amplitudes move to the
    next index), which is what the CUDA-Q ``increment`` kernels implement. This
    function is the definition of the name; every shift in the package follows it.
    """
    return np.roll(u, 1)


def decrement(u: NDArray) -> NDArray:
    """Cyclic shift ``(decrement u)_j = u_{j+1}``, i.e. ``np.roll(u, -1)``.

    On basis states ``|j> -> |j - 1 mod N>``; the inverse (and transpose) of
    :func:`increment`.
    """
    return np.roll(u, -1)


def fd_euler_coefficients(
    n: int, dt: float, c: float, nu: float, L: float = TWO_PI
) -> tuple[float, float, float]:
    """Coefficients ``(a0, a_dec, a_inc)`` of ``I + dt*Lfd``.

    ``I + dt Lfd = a0 I + a_dec * decrement + a_inc * increment`` where, for the
    periodic central-difference operator on ``N = 2**n`` points, ``h = L/N``,
    ``Lfd = -c (decrement - increment)/(2h) + nu (decrement - 2 I + increment)/h^2``.
    With ``r = nu dt / h^2`` and ``s = c dt / (2 h)``:
    ``a0 = 1 - 2r``, ``a_dec = r - s`` (weight of ``u_{j+1}``) and
    ``a_inc = r + s`` (weight of ``u_{j-1}``).
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    h = L / 2**n
    r = nu * dt / h**2
    s = c * dt / (2.0 * h)
    return 1.0 - 2.0 * r, r - s, r + s


def fd_euler_amplification(
    n: int, dt: float, c: float, nu: float, L: float = TWO_PI
) -> float:
    """Largest Euler amplification ``max_m |g_m|`` over the N Fourier modes.

    Mode ``e^{i j theta}`` with ``theta = 2 pi m / N`` has
    ``g = a0 + a_dec e^{i theta} + a_inc e^{-i theta}``. Explicit Euler is
    stable (no growing mode) iff this is <= 1.
    """
    a0, a_dec, a_inc = fd_euler_coefficients(n, dt, c, nu, L)
    theta = TWO_PI * np.arange(2**n) / 2**n
    g = a0 + a_dec * np.exp(1j * theta) + a_inc * np.exp(-1j * theta)
    return float(np.max(np.abs(g)))


def fd_euler_dt_max(n: int, c: float, nu: float, L: float = TWO_PI) -> float:
    """Largest stable explicit-Euler step for central-difference advection-diffusion.

    For a continuum of wavenumbers ``|g| <= 1`` iff ``b^2 <= a <= 1`` with
    ``a = 2 nu dt / h^2`` and ``b = c dt / h``, i.e.
    ``dt <= min(2 nu / c^2, h^2 / (2 nu))``. This is returned. It is stable on
    every grid; the ``h^2 / (2 nu)`` part is exact for even N, while on a finite
    grid the ``2 nu / c^2`` part is slightly conservative (the smallest discrete
    wavenumber is not 0). :func:`check_fd_euler_stable`, which the solvers use,
    tests the exact discrete amplification instead. For ``nu = 0`` no step is
    stable (central differences with Euler always grow), and 0.0 is returned.
    """
    if nu <= 0.0:
        return 0.0
    h = L / 2**n
    limit = h**2 / (2.0 * nu)
    if c != 0.0:
        limit = min(limit, 2.0 * nu / c**2)
    return limit


STABILITY_TOL = 1e-12


def check_fd_euler_stable(
    n: int, dt: float, c: float, nu: float, L: float = TWO_PI
) -> None:
    """Raise ``ValueError`` if explicit Euler would amplify any mode."""
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if nu < 0.0:
        raise ValueError(f"nu must be >= 0, got {nu}")
    amp = fd_euler_amplification(n, dt, c, nu, L)
    if amp > 1.0 + STABILITY_TOL:
        raise ValueError(
            f"explicit Euler unstable: max amplification {amp:.6f} > 1 "
            f"(dt={dt}, dt_max={fd_euler_dt_max(n, c, nu, L):.6g}; "
            "nu = 0 has no stable step)"
        )


def evolve_fd_euler(
    u0: NDArray[np.floating],
    dt: float,
    steps: int,
    c: float,
    nu: float,
    L: float = TWO_PI,
) -> NDArray[np.float64]:
    """Explicit Euler on the periodic central-difference operator.

    ``u^{s+1} = (I + dt Lfd) u^s``, see :func:`fd_euler_coefficients`. This is
    the validation target of the variational solver: it contains the spatial
    and time discretization error of the scheme but no optimization error.
    Compare with :func:`evolve_spectral` to measure the discretization error.

    Returns
    -------
    ndarray, shape (steps + 1, N)
        Row ``s`` is the field after ``s`` steps (row 0 is ``u0``).

    Raises
    ------
    ValueError
        If the step violates the explicit-Euler stability limit.
    """
    u0 = np.asarray(u0, dtype=np.float64)
    if u0.ndim != 1:
        raise ValueError("u0 must be one-dimensional")
    if not np.all(np.isfinite(u0)):
        raise ValueError("u0 must be finite")
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    n = _check_size(u0.size)
    check_fd_euler_stable(n, dt, c, nu, L)
    a0, a_dec, a_inc = fd_euler_coefficients(n, dt, c, nu, L)
    out = np.empty((steps + 1, u0.size))
    out[0] = u0
    for s in range(steps):
        u = out[s]
        out[s + 1] = a0 * u + a_dec * decrement(u) + a_inc * increment(u)
    return out
