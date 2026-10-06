"""Cyclic shift kernels on ``N = 2**n`` basis states (qubit 0 is the LSB).

Names describe the action on computational basis states and are defined by numpy:

* ``increment``: ``|j> -> |j + 1 mod N>``, the matrix of ``np.roll(u, +1)``
  (:func:`cudaq_pde.classical.increment`)
* ``decrement``: ``|j> -> |j - 1 mod N>``, the matrix of ``np.roll(u, -1)``
  (:func:`cudaq_pde.classical.decrement`), the inverse of ``increment``

The finite-difference code, the LCU preparation and these kernels all use this one
vocabulary; the tests compare the kernels with ``classical.increment`` and
``classical.decrement`` on every basis state. Three implementations:

``*_qft``
    ``QFT (diagonal phases) QFT-dagger`` with the spectral circuit of
    :mod:`spectral_kernels` (phases ``exp(-+ 2 pi i k / N)``, "negative" Nyquist
    convention, which is exact for a permutation). No ancilla.
``*_mcx``
    Cascade of multi-controlled X: bit ``i`` flips iff all lower bits are 1,
    applied from the top bit down. No ancilla in the source (the compiler may add
    its own when lowering).
``*_ladder``
    The same cascade with ``n - 2`` explicit clean ancillas ``work`` holding the
    prefix ANDs, so only Toffoli and CX gates remain (linear gate count).
"""

import cudaq

from cudaq_pde.solvers.spectral_kernels import iqft_b, qft_b

TWO_PI = 6.283185307179586


@cudaq.kernel
def increment_qft(q: cudaq.qview):
    """``|j> -> |j+1>`` as ``QFT D QFT-dagger`` (swap-free, mirrored phases)."""
    n = q.size()
    iqft_b(q)
    ang = -TWO_PI
    for _ in range(n):
        ang = ang / 2.0
    for b in range(n):
        a = ang
        if b == n - 1:
            a = -ang
        r1(a, q[n - 1 - b])
        ang = ang * 2.0
    qft_b(q)


@cudaq.kernel
def decrement_qft(q: cudaq.qview):
    """``|j> -> |j-1>``: the same circuit with all phases negated."""
    n = q.size()
    iqft_b(q)
    ang = TWO_PI
    for _ in range(n):
        ang = ang / 2.0
    for b in range(n):
        a = ang
        if b == n - 1:
            a = -ang
        r1(a, q[n - 1 - b])
        ang = ang * 2.0
    qft_b(q)


@cudaq.kernel
def increment_mcx(q: cudaq.qview):
    """``|j> -> |j+1>`` with a top-down cascade of multi-controlled X."""
    n = q.size()
    for k in range(n - 1):
        i = n - 1 - k
        x.ctrl(q.front(i), q[i])
    x(q[0])


@cudaq.kernel
def decrement_mcx(q: cudaq.qview):
    """``|j> -> |j-1>``: the cascade in reverse order."""
    n = q.size()
    x(q[0])
    for i in range(1, n):
        x.ctrl(q.front(i), q[i])


@cudaq.kernel
def increment_ladder(q: cudaq.qview, work: cudaq.qview):
    """``|j> -> |j+1>`` using ``n - 2`` clean ancillas in ``work`` (size >= n-2).

    ``work[i-2]`` holds ``q[0] and ... and q[i-1]`` for ``i = 2 .. n-1`` and is
    uncomputed before the next lower bit is touched.
    """
    n = q.size()
    if n > 2:
        x.ctrl(q[0], q[1], work[0])
        for i in range(3, n):
            x.ctrl(work[i - 3], q[i - 1], work[i - 2])
        for k in range(n - 2):
            i = n - 1 - k
            x.ctrl(work[i - 2], q[i])
            if i > 2:
                x.ctrl(work[i - 3], q[i - 1], work[i - 2])
            else:
                x.ctrl(q[0], q[1], work[0])
    x.ctrl(q[0], q[1])
    x(q[0])


@cudaq.kernel
def decrement_ladder(q: cudaq.qview, work: cudaq.qview):
    """Inverse of :func:`increment_ladder` (exact gate-by-gate reversal)."""
    n = q.size()
    x(q[0])
    x.ctrl(q[0], q[1])
    if n > 2:
        for k in range(n - 2):
            i = 2 + k
            if i > 2:
                x.ctrl(work[i - 3], q[i - 1], work[i - 2])
            else:
                x.ctrl(q[0], q[1], work[0])
            x.ctrl(work[i - 2], q[i])
        for k in range(n - 3):
            i = n - 1 - k
            x.ctrl(work[i - 3], q[i - 1], work[i - 2])
        x.ctrl(q[0], q[1], work[0])


@cudaq.kernel
def shift_circuit(n: int, m: int, kind: int):
    """One shift on ``n`` fresh qubits plus ``m`` work qubits (resource counting).

    ``kind``: 0/1 increment/decrement QFT, 2/3 MCX cascade, 4/5 ancilla ladder.
    """
    q = cudaq.qvector(n + m)
    sys = q.front(n)
    work = q.back(m)
    if kind == 0:
        increment_qft(sys)
    if kind == 1:
        decrement_qft(sys)
    if kind == 2:
        increment_mcx(sys)
    if kind == 3:
        decrement_mcx(sys)
    if kind == 4:
        increment_ladder(sys, work)
    if kind == 5:
        decrement_ladder(sys, work)
