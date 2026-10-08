"""Kernel for the mid-circuit measurement probe (see mid_circuit_probe.py)."""

import cudaq


@cudaq.kernel
def two_factor_probe(a: float, b: float, t: float) -> list[bool]:
    """ry on q0, two controlled rotations onto one ancilla, each followed by mz + reset.

    Returns ``[m1, m2, out]``; the joint all-zero probability is ``cos(a/2)**2``
    (``out = 0`` forces q0 = 0, so the ancilla is never rotated).
    """
    q = cudaq.qvector(3)
    ry(a, q[0])
    ry.ctrl(b, q[0], q[1])
    m1 = mz(q[1])
    reset(q[1])
    ry.ctrl(t, q[0], q[1])
    m2 = mz(q[1])
    reset(q[1])
    out = mz(q[0])
    return [m1, m2, out]
