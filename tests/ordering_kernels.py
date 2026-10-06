"""Test-only kernels used to pin the qubit-ordering convention."""

import cudaq


@cudaq.kernel
def basis_state(n: int, k: int):
    q = cudaq.qvector(n)
    x(q[k])


@cudaq.kernel
def product_ry(angles: list[float]):
    q = cudaq.qvector(len(angles))
    for i in range(len(angles)):
        ry(angles[i], q[i])
