"""Kernels for the variational advection-diffusion solver.

The ansatz is real-amplitude hardware-efficient: RY and CX are real gates, so
the prepared state has real amplitudes, matching the real PDE solution. Qubit
ordering follows :mod:`cudaq_pde.encoding` (qubit 0 is the least significant
bit of the grid index).
"""

import cudaq


@cudaq.kernel
def hea_ry_on(q: cudaq.qview, thetas: list[float], depth: int):
    """Apply the ansatz to ``q``: RY layer, then ``depth`` x (CX ladder, RY layer).

    Parameter layout: ``thetas[l*n + i]`` is the RY angle of qubit ``i`` in
    layer ``l`` (``n * (depth + 1)`` angles). The CX ladder is
    ``cx(q[i], q[i+1])`` for ``i = 0 .. n-2``.
    """
    n = q.size()
    for i in range(n):
        ry(thetas[i], q[i])
    for d in range(depth):
        for i in range(n - 1):
            x.ctrl(q[i], q[i + 1])
        for i in range(n):
            ry(thetas[(d + 1) * n + i], q[i])


@cudaq.kernel
def hea_state(n: int, thetas: list[float], depth: int):
    """Prepare the ansatz state on ``n`` fresh qubits starting from |0...0>."""
    q = cudaq.qvector(n)
    hea_ry_on(q, thetas, depth)
