"""Test-only wrappers for the ansatz and its adjoint."""

import cudaq

from cudaq_pde.solvers.variational_kernels import hea_ry_adjoint_on, hea_ry_on


@cudaq.kernel
def ansatz_then_adjoint(n: int, thetas: list[float], depth: int):
    q = cudaq.qvector(n)
    hea_ry_on(q, thetas, depth)
    hea_ry_adjoint_on(q, thetas, depth)


@cudaq.kernel
def ansatz_with_other_adjoint(
    n: int, thetas: list[float], others: list[float], depth: int
):
    q = cudaq.qvector(n)
    hea_ry_on(q, thetas, depth)
    hea_ry_adjoint_on(q, others, depth)
