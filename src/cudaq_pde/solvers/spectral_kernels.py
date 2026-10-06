"""Gate-level kernels for the spectral advection circuit.

Conventions (qubit 0 is the least significant bit of the grid index, see
:mod:`cudaq_pde.encoding`; measured against numpy in the tests):

* ``qft_b`` is the textbook QFT that starts with H on qubit 0 and has no final
  swaps. As a matrix it equals ``F+ R`` with ``F+ = sqrt(N) * ifft`` (sign +,
  the inverse DFT up to normalization) and ``R`` the bit reversal.
* The true QFT is therefore ``qft_b`` after ``reverse_qubits``, and the QFT
  dagger is ``reverse_qubits`` after ``iqft_b``.
"""

import cudaq


@cudaq.kernel
def qft_b(q: cudaq.qview):
    """QFT circuit B (H first on qubit 0), without swaps: equals ``F+ R``."""
    n = q.size()
    for a in range(n):
        h(q[a])
        theta = 3.141592653589793
        for b in range(a + 1, n):
            theta = theta / 2.0
            r1.ctrl(theta, q[b], q[a])


@cudaq.kernel
def iqft_b(q: cudaq.qview):
    """Dagger of :func:`qft_b`: equals ``R F-``, ``F-`` being ``F+`` conjugated."""
    n = q.size()
    for i in range(n):
        a = n - 1 - i
        for j in range(n - 1 - a):
            b = n - 1 - j
            theta = 3.141592653589793
            for _ in range(b - a):
                theta = theta / 2.0
            r1.ctrl(-theta, q[b], q[a])
        h(q[a])


@cudaq.kernel
def reverse_qubits(q: cudaq.qview):
    """Bit reversal ``R`` of the register as floor(n/2) swaps."""
    n = q.size()
    for i in range(n // 2):
        swap(q[i], q[n - 1 - i])
