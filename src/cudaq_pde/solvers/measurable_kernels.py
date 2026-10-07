"""Measurable cost circuit for the variational advection-diffusion step.

``lcu_cost_circuit`` prepares ``A(theta_prev)|0>`` on the system register, applies the
LCU block (so that the system holds ``b / lambda`` when the two LCU ancillas are
``|00>``, see :mod:`lcu_kernels`) and then ``A(theta)^dagger``. Sampling every qubit,

    P(sys = 0..0 and anc = 00) = <psi(theta)|b>^2 / lambda^2,
    P(anc = 00)                = ||b||^2 / lambda^2,

so ``C = 1 - P(sys = 0 | anc = 00) = 1 - <psi|b>^2 / ||b||^2``. One circuit per cost
evaluation, an uncontrolled ansatz, no state-vector readout.
"""

import cudaq

from cudaq_pde.solvers.lcu_kernels import lcu_block
from cudaq_pde.solvers.variational_kernels import hea_ry_adjoint_on, hea_ry_on


@cudaq.kernel
def lcu_cost_circuit(
    n: int,
    theta_prev: list[float],
    theta: list[float],
    depth: int,
    prep: list[float],
    neg: list[int],
    select_kind: int,
    n_work: int,
    dec_angles: list[float],
    inc_angles: list[float],
):
    """Register: ``n`` system qubits, 2 LCU ancillas, ``n_work`` work ancillas."""
    q = cudaq.qvector(n + 2 + n_work)
    sys = q.front(n)
    hea_ry_on(sys, theta_prev, depth)
    lcu_block(q, n, prep, neg, select_kind, dec_angles, inc_angles)
    hea_ry_adjoint_on(sys, theta, depth)
