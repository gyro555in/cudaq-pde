"""Python driver for the gate-level spectral advection solver.

Solves ``u_t + c u_x = 0`` on a periodic domain of length ``L`` with
``N = 2**n`` points by ``QFT-dagger -> diagonal phases -> QFT``, where the
CUDA-Q kernels live in :mod:`cudaq_pde.solvers.spectral_kernels`.

The QFT built here has the + sign (it is ``sqrt(N) * numpy.fft.ifft``), so the
transform to Fourier space used by ``numpy.fft.fft`` is the QFT-dagger and the
phase is ``exp(-i c (2 pi / L) k t)``. ``k`` is the two's-complement integer of
the register (top qubit weight ``-2**(n-1)``), which makes the phase layer
n single-qubit ``R1`` gates. See ``classical.py`` for the Nyquist conventions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cudaq_pde import encoding
from cudaq_pde.classical import TWO_PI, Nyquist, _check_nyquist


def phase_angles(
    n: int, c: float, L: float = TWO_PI, dt: float = 1.0
) -> tuple[list[float], float]:
    """R1 angles for one step of length ``dt``.

    Returns
    -------
    angles : list of float, length n
        ``angles[b]`` multiplies the state by ``exp(i angles[b])`` when bit b
        of the Fourier index is 1: ``-c (2 pi / L) dt * w_b`` with
        ``w_b = 2**b`` for ``b < n-1`` and ``w_{n-1} = -2**(n-1)``. This is the
        "negative" Nyquist convention (k = -N/2).
    phi : float
        Extra phase on the Nyquist state that turns "negative" into "zero":
        ``-c (2 pi / L) dt * N/2``.
    """
    if n < 2:
        raise ValueError(f"need n >= 2 qubits, got {n}")
    ct = c * (TWO_PI / L) * dt
    weights = [2**b for b in range(n - 1)] + [-(2 ** (n - 1))]
    return [-ct * w for w in weights], -ct * 2 ** (n - 1)


def shift_angles(n: int, direction: int) -> list[float]:
    """R1 angles of a cyclic shift by one grid cell, as a spectral circuit.

    A shift is advection by one cell: ``c * dt = direction * L / N``, so the phase
    factor ``exp(-i c (2 pi / L) dt k) = exp(-+ 2 pi i k / N)`` and the angles are
    :func:`phase_angles` with that ``c * dt``. ``direction = +1`` is the
    ``increment`` (``np.roll(u, +1)``) and ``-1`` the ``decrement``. With the
    "negative" Nyquist convention (the one these angles implement) the result is an
    exact permutation. The angles are independent of ``L``.
    """
    if direction not in (+1, -1):
        raise ValueError(f"direction must be +1 or -1, got {direction}")
    angles, _ = phase_angles(n, 1.0, TWO_PI, direction * TWO_PI / 2**n)
    return angles


def evolve_state(
    psi: NDArray[np.complexfloating],
    c: float,
    L: float = TWO_PI,
    dt: float = 1.0,
    steps: int = 1,
    nyquist: Nyquist = "zero",
    swaps: bool = False,
) -> NDArray[np.complex128]:
    """Evolve a normalized state by ``steps`` steps of ``dt`` (total time ``steps*dt``).

    SIMULATOR-ONLY: the result is read with ``cudaq.get_state``. On hardware
    one would sample the circuit or measure observables instead. The caller
    selects the CUDA-Q target (``qpp-cpu`` or ``nvidia`` with ``fp64``).

    Parameters
    ----------
    psi : ndarray, shape (2**n,)
        Normalized amplitudes in the ordering of :mod:`cudaq_pde.encoding`.
    swaps : bool
        Default False: the QFT bit reversals are eliminated and the phases are
        applied to the mirrored qubits (exactly the same unitary). True keeps
        the explicit swap layers.
    """
    import cudaq

    from cudaq_pde.solvers.spectral_kernels import spectral_advection

    _check_nyquist(nyquist)
    psi = np.asarray(psi, dtype=np.complex128)
    n = encoding.num_qubits(psi.size)
    if abs(np.linalg.norm(psi) - 1.0) > 1e-9:
        raise ValueError("psi must be normalized")
    if steps < 0:
        raise ValueError("steps must be >= 0")
    angles, phi = phase_angles(n, c, L, dt)
    state = cudaq.get_state(
        spectral_advection,
        psi.tolist(),
        angles,
        phi,
        nyquist == "zero",
        swaps,
        steps,
    )
    return np.array(state, dtype=np.complex128)


def evolve(
    u0: NDArray[np.floating],
    c: float,
    L: float = TWO_PI,
    dt: float = 1.0,
    steps: int = 1,
    nyquist: Nyquist = "zero",
    swaps: bool = False,
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Evolve real grid values; the norm is tracked classically.

    Returns float64 for ``nyquist="zero"`` (real stays real, checked) and
    complex128 for ``"negative"``. SIMULATOR-ONLY, see :func:`evolve_state`.
    """
    psi, norm = encoding.encode(u0)
    out = evolve_state(psi, c, L, dt, steps, nyquist, swaps)
    if nyquist == "zero":
        return encoding.decode(out, norm, atol=1e-10 * max(1.0, norm))
    return norm * out
