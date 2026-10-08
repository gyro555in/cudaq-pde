"""Classical references for viscous Burgers, ``u_t + u u_x = nu u_xx``, periodic.

Two independent references, which must agree before anything quantum is trusted:

* **Exact Cole-Hopf** (:func:`exact_cole_hopf`): ``u = -2 nu (ln phi)_x`` with ``phi``
  the heat-equation solution of ``phi0 = exp(-(1/2nu) int u0)``. Evaluated in closed
  form in the log domain (softmax weights over a fine grid and periodic images), so it
  stays stable when ``phi`` spans ``exp(1/nu)`` in amplitude (no FFT of such a
  function).
* **Pseudo-spectral integrator** (:func:`pseudospectral_burgers`): integrating-factor
  RK4, 2/3 dealiasing, independent of Cole-Hopf; accepts a nonzero mean.

and the **discrete Cole-Hopf** (:func:`discrete_burgers`), what the ``n``-qubit circuit
computes on ``N = 2**n`` points: ``phi0`` sampled, spectral heat step, then the
finite-difference velocity ``u_j = -(nu/h)(ln phi_{j+1} - ln phi_{j-1})``.

Zero mean
---------
A periodic ``phi0`` needs ``int u0 = 0``. :func:`galilean_split` removes the mean ``U0``
(and asserts it is gone); Burgers is Galilean invariant, ``u(x, t) = U0 + w(x - U0 t,
t)`` with ``w`` the zero-mean problem, so :func:`shift_back` applies the existing
spectral advection with ``c = U0`` to the evolved ``w``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from cudaq_pde import classical
from cudaq_pde.classical import TWO_PI

# exp(-range) must stay representable: the valley of phi0 is exp(-(max W - min W)/2nu)
MAX_LOG_RANGE = 600.0


def zero_mean_tolerance(u0: NDArray[np.floating]) -> float:
    """Roundoff bound on the mean of a mean-subtracted vector: ``8 eps max|u0|``.

    The mean of ``N`` doubles is accurate to about ``eps max|u0|`` (pairwise sum); the
    factor 8 is headroom, not a fitted number.
    """
    return 8.0 * np.finfo(np.float64).eps * float(np.max(np.abs(u0)))


def galilean_split(u0: NDArray[np.floating]) -> tuple[NDArray[np.float64], float]:
    """Return ``(w0, U0)`` with ``w0 = u0 - U0`` of zero mean (asserted)."""
    u0 = np.asarray(u0, dtype=np.float64)
    U0 = float(np.mean(u0))
    w0 = u0 - U0
    if abs(float(np.mean(w0))) > zero_mean_tolerance(u0):
        raise AssertionError(f"mean after subtraction is {np.mean(w0):.3e}")
    return w0, U0


def antiderivative(w: NDArray[np.floating], L: float = TWO_PI) -> NDArray[np.float64]:
    """Spectral antiderivative of a zero-mean periodic vector (zero-mean result).

    The Nyquist mode of the integral is set to zero (the "zero" convention of
    :mod:`cudaq_pde.classical` for odd operators).
    """
    w = np.asarray(w, dtype=np.float64)
    n = classical._check_size(w.size)
    if abs(float(np.mean(w))) > 1e3 * zero_mean_tolerance(w):
        raise ValueError(
            "periodic Cole-Hopf needs int u0 = 0; subtract the mean first "
            f"(mean is {np.mean(w):.3e}, see galilean_split)"
        )
    k = classical.wavenumbers(n, L)
    k[w.size // 2] = 0.0  # Nyquist
    what = np.fft.fft(w)
    out = np.zeros_like(what)
    nz = k != 0.0
    out[nz] = what[nz] / (1j * k[nz])
    return np.fft.ifft(out).real


def cole_hopf_phi0(
    w0: NDArray[np.floating], nu: float, L: float = TWO_PI
) -> NDArray[np.float64]:
    """``phi0 = exp(-(W - min W)/2nu)`` with ``W = int w0``: positive, maximum 1.

    Only the direction of ``phi`` matters (the norm cancels in ``u``), so the constant
    of integration is fixed to make the maximum 1. Raises if the valley underflows.
    """
    if nu <= 0:
        raise ValueError(f"nu must be > 0, got {nu}")
    W = antiderivative(w0, L)
    log_range = float((W.max() - W.min()) / (2.0 * nu))
    if log_range > MAX_LOG_RANGE:
        raise ValueError(
            f"phi0 spans exp({log_range:.0f}) in amplitude; float64 cannot hold it"
        )
    return np.exp(-(W - W.min()) / (2.0 * nu))


def heat_phi(
    phi0: NDArray[np.floating], nu: float, T: float, L: float = TWO_PI
) -> NDArray[np.float64]:
    """Spectral heat step ``phi_t = nu phi_xx`` (exactly what the circuit applies)."""
    out = classical.evolve_spectral(phi0, T, 0.0, nu, L, "zero")
    return np.asarray(out, dtype=np.float64)


def velocity_fd(
    phi: NDArray[np.floating], nu: float, L: float = TWO_PI
) -> NDArray[np.float64]:
    """``u_j = -(nu/h)(ln phi_{j+1} - ln phi_{j-1})``; NaN where ``phi <= 0``."""
    phi = np.asarray(phi, dtype=np.float64)
    h = L / phi.size
    with np.errstate(invalid="ignore", divide="ignore"):
        lp = np.log(np.where(phi > 0, phi, np.nan))
    return -(nu / h) * (np.roll(lp, -1) - np.roll(lp, 1))


def velocity_spectral(
    phi: NDArray[np.floating], nu: float, L: float = TWO_PI
) -> NDArray[np.float64]:
    """``u = -2 nu phi_x / phi`` with a spectral derivative (Nyquist set to zero)."""
    phi = np.asarray(phi, dtype=np.float64)
    n = classical._check_size(phi.size)
    k = classical.wavenumbers(n, L)
    k[phi.size // 2] = 0.0
    dphi = np.fft.ifft(1j * k * np.fft.fft(phi)).real
    with np.errstate(invalid="ignore", divide="ignore"):
        return -2.0 * nu * dphi / np.where(phi > 0, phi, np.nan)


@dataclass(frozen=True)
class DiscreteBurgers:
    """Discrete Cole-Hopf on ``2**n`` points, in the moving frame of the mean ``U0``."""

    x: NDArray[np.float64]
    U0: float
    w0: NDArray[np.float64]
    phi0: NDArray[np.float64]
    phiT: NDArray[np.float64]
    p: NDArray[np.float64]  # normalized phiT**2, the distribution the circuit samples
    p_succ: float  # ||phiT||^2 / ||phi0||^2
    u_fd: NDArray[np.float64]  # moving-frame velocity w(x_j, T), FD formula
    u_spectral: NDArray[np.float64]  # same with a spectral derivative


def discrete_burgers(
    u0: NDArray[np.floating], nu: float, T: float, L: float = TWO_PI
) -> DiscreteBurgers:
    """Cole-Hopf on the grid of ``u0``: what the circuit computes (double precision)."""
    w0, U0 = galilean_split(u0)
    phi0 = cole_hopf_phi0(w0, nu, L)
    phiT = heat_phi(phi0, nu, T, L)
    p = phiT**2 / np.sum(phiT**2)
    return DiscreteBurgers(
        x=classical.grid(classical._check_size(w0.size), L),
        U0=U0,
        w0=w0,
        phi0=phi0,
        phiT=phiT,
        p=p,
        p_succ=float(np.sum(phiT**2) / np.sum(phi0**2)),
        u_fd=velocity_fd(phiT, nu, L),
        u_spectral=velocity_spectral(phiT, nu, L),
    )


def shift_back(
    w: NDArray[np.floating], U0: float, T: float, L: float = TWO_PI
) -> NDArray[np.float64]:
    """Lab-frame field ``U0 + w(x - U0 T)`` using the existing spectral advection.

    ``c = U0`` advection of the evolved zero-mean field, then the mean is added back.
    Meant for a full reconstructed field (``w`` must be finite everywhere); pointwise
    probes need no interpolation, see :func:`lab_positions`.
    """
    w = np.asarray(w, dtype=np.float64)
    shifted = classical.evolve_spectral(w, T, U0, 0.0, L, "zero")
    return U0 + np.asarray(shifted, dtype=np.float64)


def lab_positions(
    x_moving: NDArray[np.floating], U0: float, T: float, L: float = TWO_PI
) -> NDArray[np.float64]:
    """Lab-frame positions ``x + U0 T`` (mod L) of a moving-frame probe value."""
    return np.mod(np.asarray(x_moving, dtype=np.float64) + U0 * T, L)


def _log_weights(x, w0, nu, t, L, n_fine, images):
    """Log integrand of the periodic heat kernel: ``(logw, z)``, shape (x, y, m)."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.arange(n_fine) * (L / n_fine)
    W = antiderivative(w0(y), L)
    m = np.arange(-images, images + 1) * L
    z = x[:, None, None] - y[None, :, None] + m[None, None, :]
    return -W[None, :, None] / (2.0 * nu) - z**2 / (4.0 * nu * t), z


def exact_log_phi(
    x: NDArray[np.floating],
    w0: Callable[[NDArray], NDArray],
    nu: float,
    t: float,
    L: float = TWO_PI,
    n_fine: int = 4096,
    images: int = 3,
) -> NDArray[np.float64]:
    """``ln phi(x, t)`` of the exact Cole-Hopf solution, up to a constant in x.

    Needed where ``phi`` itself is far below what the double-precision grid solution
    can represent (the valley at small ``nu``); ``t > 0``.
    """
    if t <= 0.0:
        raise ValueError("exact_log_phi needs t > 0")
    logw, _ = _log_weights(x, w0, nu, t, L, n_fine, images)
    return logsumexp(logw, axis=(1, 2))


def exact_cole_hopf(
    x: NDArray[np.floating],
    w0: Callable[[NDArray], NDArray],
    nu: float,
    t: float,
    L: float = TWO_PI,
    n_fine: int = 4096,
    images: int = 3,
) -> NDArray[np.float64]:
    """Exact zero-mean Burgers solution ``w(x, t)`` by Cole-Hopf, at arbitrary ``x``.

    ``u(x,t) = sum_y ((x - y)/t) W_y / sum_y W_y`` with
    ``log W_y = -int w0 (y)/2nu - (x - y + mL)^2/(4 nu t)``, summed over a fine periodic
    grid ``y`` (trapezoid, spectrally accurate) and ``2*images + 1`` images ``m``; the
    weights are a log-sum-exp softmax. ``w0`` is a callable sampled on the fine grid and
    must have zero mean.
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    if t == 0.0:
        return np.asarray(w0(x), dtype=np.float64)
    logw, z = _log_weights(x, w0, nu, t, L, n_fine, images)
    norm = logsumexp(logw, axis=(1, 2))
    weights = np.exp(logw - norm[:, None, None])
    return np.sum(weights * z / t, axis=(1, 2))


def pseudospectral_burgers(
    u0: Callable[[NDArray], NDArray] | NDArray,
    nu: float,
    T: float,
    L: float = TWO_PI,
    N: int = 1024,
    steps: int | None = None,
) -> NDArray[np.float64]:
    """Integrating-factor RK4 with 2/3 dealiasing; returns ``u(x_j, T)``, ``N`` points.

    Independent of Cole-Hopf. Accepts a nonzero mean. ``steps`` defaults to a step with
    ``dt max|u| k_max <= 0.5`` (``k_max`` the largest retained wavenumber), well inside
    the RK4 stability limit of about 2.8.
    """
    n = classical._check_size(N)
    x = np.arange(N) * (L / N)
    u = np.asarray(u0(x) if callable(u0) else u0, dtype=np.float64)
    if u.size != N:
        raise ValueError(f"u0 must have {N} points, got {u.size}")
    k = classical.wavenumbers(n, L)
    keep = np.abs(np.fft.fftfreq(N, 1.0 / N)) <= N / 3.0
    kmax = float(np.max(np.abs(k[keep])))
    if steps is None:
        umax = float(np.max(np.abs(u))) * 1.5 + 1e-12  # allow some growth
        steps = max(1, int(np.ceil(T * umax * kmax / 0.5)))
    dt = T / steps
    E = np.exp(-nu * k**2 * dt)
    E2 = np.exp(-nu * k**2 * dt / 2.0)

    def nonlinear(uh: NDArray) -> NDArray:
        uh = uh * keep
        ur = np.fft.ifft(uh).real
        return -0.5j * k * (np.fft.fft(ur * ur) * keep)

    uh = np.fft.fft(u) * keep
    for _ in range(steps):
        a = dt * nonlinear(uh)
        b = dt * nonlinear(E2 * (uh + a / 2.0))
        c = dt * nonlinear(E2 * uh + b / 2.0)
        d = dt * nonlinear(E * uh + E2 * c)
        uh = E * uh + (E * a + 2.0 * E2 * (b + c) + d) / 6.0
    return np.fft.ifft(uh).real


def evaluate_fourier(
    field: NDArray[np.floating], x: NDArray[np.floating], L: float = TWO_PI
) -> NDArray[np.float64]:
    """Spectral interpolation of a periodic grid field at arbitrary positions ``x``."""
    field = np.asarray(field, dtype=np.float64)
    classical._check_size(field.size)
    m = np.fft.fftfreq(field.size, 1.0 / field.size)  # Nyquist coefficient is real
    fh = np.fft.fft(field) / field.size
    phase = np.exp(1j * (TWO_PI / L) * np.outer(np.atleast_1d(x), m))
    return (phase @ fh).real
