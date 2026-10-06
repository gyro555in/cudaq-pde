"""Kernels preparing ``b_hat = (I + dt L) |psi_prev> / ||b||`` by LCU.

``I + dt L = a0 I + a_dec * decrement + a_inc * increment`` with the shifts defined
by ``classical.decrement`` / ``classical.increment`` (``np.roll(u, -1)`` and
``np.roll(u, +1)``). Two ancilla
qubits ``anc``: slot ``|a0 a1> = |00>`` is the identity, ``a0 = 1`` the
decrement, ``a1 = 1`` the increment, slot ``|11>`` has zero weight, so each
controlled shift has a single ancilla control.

PREPARE puts amplitudes ``sqrt(|c_i| / lambda)`` on the slots, SELECT applies
the controlled shifts, PREPARE-dagger undoes the preparation, and the system
register holds ``b / lambda`` when the ancillas are measured in ``|00>``
(success probability ``||b||^2 / lambda^2``, ``lambda = sum |c_i|``). Negative
coefficients are a Z on the matching ancilla (their slot is the only one with
that ancilla bit set).
"""

import cudaq

from cudaq_pde.solvers.shift_kernels import TWO_PI
from cudaq_pde.solvers.spectral_kernels import iqft_b, qft_b
from cudaq_pde.solvers.variational_kernels import hea_ry_on


@cudaq.kernel
def lcu_prepare(anc: cudaq.qview, theta_h: float, theta_0: float):
    """Real 3-slot PREPARE: RY on a1, then RY on a0 controlled on a1 = 0."""
    ry(theta_h, anc[1])
    x(anc[1])
    ry.ctrl(theta_0, anc[1], anc[0])
    x(anc[1])


@cudaq.kernel
def lcu_unprepare(anc: cudaq.qview, theta_h: float, theta_0: float):
    """PREPARE-dagger (gate-by-gate reversal with negated angles)."""
    x(anc[1])
    ry.ctrl(-theta_0, anc[1], anc[0])
    x(anc[1])
    ry(-theta_h, anc[1])


@cudaq.kernel
def select_shifts_qft(sys: cudaq.qview, anc: cudaq.qview):
    """Controlled decrement (anc[0]) and increment (anc[1]) in one QFT sandwich.

    ``c-(QFT D QFT-dagger) = QFT (c-D) QFT-dagger``, so both controlled shifts share
    the two QFTs and only the ``2n`` phase gates are controlled.
    """
    n = sys.size()
    iqft_b(sys)
    ang = TWO_PI
    for _ in range(n):
        ang = ang / 2.0
    for b in range(n):
        a = ang
        if b == n - 1:
            a = -ang
        r1.ctrl(a, anc[0], sys[n - 1 - b])
        r1.ctrl(-a, anc[1], sys[n - 1 - b])
        ang = ang * 2.0
    qft_b(sys)


@cudaq.kernel
def controlled_increment_ladder(ctl: cudaq.qubit, sys: cudaq.qview, work: cudaq.qview):
    """Increment controlled on ``ctl``; ``work`` (size >= n-1) are clean ancillas.

    CUDA-Q 0.16 kernels cannot mix one qubit and a qview as control operands, so the
    multi-controlled X is built from Toffoli gates: ``work[i-1]`` holds
    ``ctl and q[0] and ... and q[i-1]`` and is uncomputed before the next lower bit.
    """
    n = sys.size()
    x.ctrl(ctl, sys[0], work[0])
    for i in range(2, n):
        x.ctrl(work[i - 2], sys[i - 1], work[i - 1])
    for k in range(n - 1):
        i = n - 1 - k
        x.ctrl(work[i - 1], sys[i])
        if i > 1:
            x.ctrl(work[i - 2], sys[i - 1], work[i - 1])
        else:
            x.ctrl(ctl, sys[0], work[0])
    x.ctrl(ctl, sys[0])


@cudaq.kernel
def controlled_decrement_ladder(ctl: cudaq.qubit, sys: cudaq.qview, work: cudaq.qview):
    """Inverse of :func:`controlled_increment_ladder` (exact gate reversal)."""
    n = sys.size()
    x.ctrl(ctl, sys[0])
    for k in range(n - 1):
        i = 1 + k
        if i > 1:
            x.ctrl(work[i - 2], sys[i - 1], work[i - 1])
        else:
            x.ctrl(ctl, sys[0], work[0])
        x.ctrl(work[i - 1], sys[i])
    for k in range(n - 2):
        i = n - 1 - k
        x.ctrl(work[i - 2], sys[i - 1], work[i - 1])
    x.ctrl(ctl, sys[0], work[0])


@cudaq.kernel
def select_shifts_ladder(sys: cudaq.qview, anc: cudaq.qview, work: cudaq.qview):
    """Controlled decrement (anc[0]) and increment (anc[1]) with Toffoli ladders."""
    controlled_decrement_ladder(anc[0], sys, work)
    controlled_increment_ladder(anc[1], sys, work)


@cudaq.kernel
def lcu_block(
    q: cudaq.qview,
    n: int,
    prep: list[float],
    neg: list[int],
    select_kind: int,
):
    """PREPARE, sign Zs, SELECT, PREPARE-dagger on the register ``q``.

    Layout of ``q``: qubits ``0..n-1`` system, ``n, n+1`` ancillas (slot qubits
    ``a0, a1``), then (only for ``select_kind == 1``) ``n - 1`` work ancillas.
    ``select_kind``: 0 = QFT sandwich, 1 = Toffoli ladder.
    """
    sys = q.front(n)
    anc = q[n : n + 2]
    lcu_prepare(anc, prep[0], prep[1])
    if neg[0] == 1:
        z(anc[0])
    if neg[1] == 1:
        z(anc[1])
    if select_kind == 0:
        select_shifts_qft(sys, anc)
    if select_kind == 1:
        select_shifts_ladder(sys, anc, q[n + 2 : 2 * n + 1])
    lcu_unprepare(anc, prep[0], prep[1])


@cudaq.kernel
def lcu_target(
    n: int,
    theta_prev: list[float],
    depth: int,
    prep: list[float],
    neg: list[int],
    select_kind: int,
    n_work: int,
):
    """Ansatz state ``A(theta_prev)|0>`` followed by the LCU block.

    Register: ``n`` system qubits, 2 ancillas (qubits ``n`` and ``n + 1``, post-select
    on ``|00>``), then ``n_work`` work ancillas (0 for the QFT form, ``n - 1`` for
    the ladder form), which return to ``|0>``.
    """
    q = cudaq.qvector(n + 2 + n_work)
    hea_ry_on(q.front(n), theta_prev, depth)
    lcu_block(q, n, prep, neg, select_kind)


@cudaq.kernel
def lcu_block_circuit(
    n: int,
    prep: list[float],
    neg: list[int],
    select_kind: int,
    n_work: int,
):
    """The LCU block alone on fresh qubits (resource counting, no ansatz)."""
    q = cudaq.qvector(n + 2 + n_work)
    lcu_block(q, n, prep, neg, select_kind)
