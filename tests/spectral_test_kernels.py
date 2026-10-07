"""Test-only wrappers that apply QFT pieces to an arbitrary input state."""

import cudaq

from cudaq_pde.solvers.spectral_kernels import iqft_b, qft_b, reverse_qubits


@cudaq.kernel
def apply_qft_piece(amps: list[complex], mode: int):
    """mode 0: B, 1: B-dagger, 2: B after R (the QFT), 3: R after B-dagger."""
    q = cudaq.qvector(amps)
    if mode == 0:
        qft_b(q)
    if mode == 1:
        iqft_b(q)
    if mode == 2:
        reverse_qubits(q)
        qft_b(q)
    if mode == 3:
        iqft_b(q)
        reverse_qubits(q)
