"""Amplitude encoding of grid values into an n-qubit register.

Ordering convention (measured on CUDA-Q 0.16, verified in the tests)
-------------------------------------------------------------------
Grid index ``j = sum_k b_k 2**k``; qubit ``k`` holds bit ``b_k``. Qubit 0 is
the least significant bit of ``j`` and qubit ``n-1`` is the most significant.
The encoded state is ``psi[j] = u[j] / ||u||``, which is exactly the layout of
``cudaq.get_state``, so no bit reversal is needed.

``cudaq.sample`` bitstrings list qubits in order, so character ``k`` of the
string is qubit ``k`` and the string reads least significant bit first, the
reverse of ordinary binary notation. Use :func:`bitstring_to_index`.

Norm
----
Amplitude encoding stores only the direction of ``u``. The norm
``||u||_2`` is tracked classically and returned by :func:`encode`; pass it back
to :func:`decode` to recover ``u``. A quantum device never sees the norm.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def num_qubits(N: int) -> int:
    """Number of qubits for ``N = 2**n`` grid values."""
    if N < 2 or N & (N - 1):
        raise ValueError(f"grid size must be a power of two >= 2, got {N}")
    return N.bit_length() - 1


def encode(u: NDArray[np.floating]) -> tuple[NDArray[np.complex128], float]:
    """Amplitude-encode a real vector.

    Parameters
    ----------
    u : ndarray of float, shape (2**n,)
        Real grid values, not identically zero.

    Returns
    -------
    psi : ndarray of complex128, shape (2**n,)
        Normalized amplitudes ``u / ||u||``, ordered as described in the module
        docstring.
    norm : float
        ``||u||_2``, tracked classically.
    """
    u = np.asarray(u)
    if u.ndim != 1:
        raise ValueError("u must be one-dimensional")
    if np.iscomplexobj(u):
        raise ValueError("u must be real")
    num_qubits(u.size)
    if not np.all(np.isfinite(u)):
        raise ValueError("u must be finite")
    norm = float(np.linalg.norm(u))
    if norm == 0.0:
        raise ValueError("cannot amplitude-encode the zero vector")
    return (u / norm).astype(np.complex128), norm


def decode(
    psi: NDArray[np.complexfloating], norm: float, atol: float = 1e-12
) -> NDArray[np.float64]:
    """Recover the real vector ``norm * psi``.

    Raises ``ValueError`` if the imaginary part of ``norm * psi`` exceeds
    ``atol`` (the state does not encode a real vector).
    """
    psi = np.asarray(psi, dtype=np.complex128)
    num_qubits(psi.size)
    scaled = norm * psi
    if np.max(np.abs(scaled.imag)) > atol:
        raise ValueError("state has a non-negligible imaginary part")
    return scaled.real.copy()


def index_to_bits(j: int, n: int) -> tuple[int, ...]:
    """Bits of grid index ``j``; element ``k`` is the value of qubit ``k``."""
    if not 0 <= j < 2**n:
        raise ValueError(f"index {j} out of range for {n} qubits")
    return tuple((j >> k) & 1 for k in range(n))


def bits_to_index(bits: tuple[int, ...] | list[int]) -> int:
    """Inverse of :func:`index_to_bits`."""
    if any(b not in (0, 1) for b in bits):
        raise ValueError("bits must be 0 or 1")
    return sum(int(b) << k for k, b in enumerate(bits))


def bitstring_to_index(s: str) -> int:
    """Grid index of a ``cudaq.sample`` bitstring (character k is qubit k)."""
    if not s or set(s) - {"0", "1"}:
        raise ValueError(f"not a bitstring: {s!r}")
    return bits_to_index([int(ch) for ch in s])
