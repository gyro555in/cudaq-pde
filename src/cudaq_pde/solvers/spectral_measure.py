"""Measurable power spectrum of the evolved field (shots only, no state vector).

After the spectral evolution the circuit applies the swap-free QFT-dagger
(:func:`spectral_kernels.spectrum_circuit`) and samples. With ``psi = u / ||u||`` the
outcome distribution is the normalized power spectrum

    P[k] = |u_hat_k|^2 / (N ||u||^2),   u_hat = numpy.fft.fft(u),   sum_k P[k] = 1.

Ordering: ``iqft_b`` equals ``R F-`` with ``R`` the bit reversal, so a sampled string
``s`` (character ``q`` is qubit ``q``) has register index ``r = sum_q s[q] 2**q``,
which is the bit-reversed FFT index: ``k_fft = rev(r)``. The wavenumber is
``m = k_fft`` for ``k_fft < N/2`` and ``k_fft - N`` otherwise (the layout of
:func:`cudaq_pde.classical.wavenumbers`).

Advection multiplies each ``u_hat_k`` by a unit phase, so the power spectrum of the
evolved field equals that of the initial field; the readout therefore also checks the
evolution kernels and the ordering end to end.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cudaq_pde import encoding
from cudaq_pde.solvers.sampling import Sampler
from cudaq_pde.solvers.spectral import phase_angles


def bit_reverse_index(r: int, n: int) -> int:
    """Reverse the ``n`` bits of ``r``."""
    return int(format(r, f"0{n}b")[::-1], 2)


def counts_to_power_spectrum(
    counts: dict[str, int], n: int, shots: int
) -> NDArray[np.float64]:
    """Estimate ``P[k_fft]`` from counts; bitstring character ``q`` is qubit ``q``."""
    p = np.zeros(2**n)
    for bits, count in counts.items():
        r = encoding.bitstring_to_index(bits)
        p[bit_reverse_index(r, n)] += count
    return p / shots


def classical_power_spectrum(u: NDArray) -> NDArray[np.float64]:
    """``|fft(u)|^2 / (N ||u||^2)``, the quantity the circuit samples."""
    u = np.asarray(u)
    return np.abs(np.fft.fft(u)) ** 2 / (u.size * np.linalg.norm(u) ** 2)


def expected_rms_error(p: NDArray[np.float64], shots: int) -> float:
    """RMS L2 error of a multinomial estimate: ``sqrt((1 - sum p^2) / N)``."""
    return float(np.sqrt((1.0 - np.sum(p**2)) / shots))


def estimate_power_spectrum(
    u0: NDArray[np.floating],
    c: float,
    L: float,
    dt: float,
    shots: int,
    seed: int,
    steps: int = 1,
    nyquist: str = "zero",
    swaps: bool = False,
    fused: bool = False,
    sampler: Sampler | None = None,
) -> NDArray[np.float64]:
    """Sampled power spectrum of the field after ``steps`` advection steps of ``dt``.

    ``fused=True`` uses the circuit without the cancelling QFT pair (same
    distribution, fewer gates). Returns ``P_hat`` in numpy FFT index order.
    """
    from cudaq_pde.solvers.spectral_kernels import (
        spectrum_circuit,
        spectrum_circuit_fused,
    )

    psi, _ = encoding.encode(np.asarray(u0, dtype=np.float64))
    n = encoding.num_qubits(psi.size)
    angles, phi = phase_angles(n, c, L, dt)
    sampler = sampler or Sampler(base_seed=seed)
    amps = psi.tolist()
    if fused:
        counts = sampler.counts(
            spectrum_circuit_fused,
            amps,
            angles,
            phi,
            nyquist == "zero",
            steps,
            shots=shots,
        )
    else:
        counts = sampler.counts(
            spectrum_circuit,
            amps,
            angles,
            phi,
            nyquist == "zero",
            swaps,
            steps,
            shots=shots,
        )
    return counts_to_power_spectrum(counts, n, shots)
