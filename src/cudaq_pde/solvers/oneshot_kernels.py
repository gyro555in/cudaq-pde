"""Kernels for the one-shot advection-diffusion circuits (see :mod:`oneshot`).

All routes share ``QFT-dagger -> advection phases -> diagonal block -> QFT`` on the
system register ``sys`` (``n`` qubits, the swap-free mirrored layout of CP3: Fourier bit
``b`` on qubit ``n-1-b``). Ancillas live in their own register and are post-selected on
``|0>``; they are separate qviews because CUDA-Q 0.16 cannot mix operand kinds freely.

Route A: ``anc`` is one qubit.
Route B2: ``anc`` has ``n(n+1)/2`` qubits, factor ``i`` rotates ``anc[i]``.
Route B1: one ancilla, ``mz`` then ``reset`` after every factor (:func:`oneshot_b1_run`,
needs ``cudaq.run``; the record holds the factor outcomes, then the system bits).
Route C: ``w`` is the width register holding the shift ``s`` (two's complement).
"""

import cudaq

from cudaq_pde.solvers.spectral_kernels import (
    iqft_b,
    nyquist_phase,
    phase_layer_mirrored,
    qft_b,
)


@cudaq.kernel
def ry_cx_sequence(w: cudaq.qview, tgt: list[int], alphas: list[float], ctl: list[int]):
    """``RY(alphas[i])`` on ``w[tgt[i]]``, then CX from ``w[ctl[i]]`` if >= 0."""
    for i in range(len(alphas)):
        ry(alphas[i], w[tgt[i]])
        if ctl[i] >= 0:
            x.ctrl(w[ctl[i]], w[tgt[i]])


@cudaq.kernel
def cx_ry_sequence(w: cudaq.qview, tgt: list[int], alphas: list[float], ctl: list[int]):
    """CX before RY: inverts :func:`ry_cx_sequence` for ``RySequence.reversed``."""
    for i in range(len(alphas)):
        if ctl[i] >= 0:
            x.ctrl(w[ctl[i]], w[tgt[i]])
        ry(alphas[i], w[tgt[i]])


@cudaq.kernel
def advection_on(sys: cudaq.qview, angles: list[float], phi: float, zero: bool):
    """Advection phases in the mirrored Fourier layout, Nyquist convention included."""
    phase_layer_mirrored(sys, angles)
    if zero:
        nyquist_phase(sys, phi, True)


# ------------------------------------------------------------------ route A


@cudaq.kernel
def multiplexed_on_ancilla(
    sys: cudaq.qview,
    anc: cudaq.qubit,
    alphas: list[float],
    ctl: list[int],
):
    """Gray-code multiplexed RY on one ancilla, controls ``sys[ctl[i]]``."""
    for i in range(len(alphas)):
        ry(alphas[i], anc)
        x.ctrl(sys[ctl[i]], anc)


@cudaq.kernel
def oneshot_a_on(
    sys: cudaq.qview,
    anc: cudaq.qubit,
    angles: list[float],
    phi: float,
    zero: bool,
    alphas: list[float],
    ctl: list[int],
):
    """Route A: ancilla ``anc`` is post-selected on 0."""
    iqft_b(sys)
    advection_on(sys, angles, phi, zero)
    multiplexed_on_ancilla(sys, anc, alphas, ctl)
    qft_b(sys)


@cudaq.kernel
def oneshot_a(
    amps: list[complex],
    angles: list[float],
    phi: float,
    zero: bool,
    alphas: list[float],
    ctl: list[int],
):
    """Load ``amps`` (simulator convenience), then route A."""
    sys = cudaq.qvector(amps)
    anc = cudaq.qubit()
    oneshot_a_on(sys, anc, angles, phi, zero, alphas, ctl)


@cudaq.kernel
def oneshot_a_circuit(
    n: int,
    angles: list[float],
    phi: float,
    zero: bool,
    alphas: list[float],
    ctl: list[int],
):
    """Route A on fresh qubits (resource counting, no data loading)."""
    sys = cudaq.qvector(n)
    anc = cudaq.qubit()
    oneshot_a_on(sys, anc, angles, phi, zero, alphas, ctl)


# ------------------------------------------------------------------ route B


@cudaq.kernel
def sign_fold(sys: cudaq.qview):
    """CX from the sign qubit 0 onto qubits ``1..n-1`` (self-inverse): ``low -> v``."""
    for j in range(1, sys.size()):
        x.ctrl(sys[0], sys[j])


@cudaq.kernel
def factor_to(
    sys: cudaq.qview,
    target: cudaq.qubit,
    c1: int,
    c2: int,
    theta: float,
):
    """``RY(theta)`` on ``target``, controls ``sys[c1]`` and ``sys[c2]`` if >= 0."""
    if c2 >= 0:
        ry.ctrl(theta, sys[c1], sys[c2], target)
    else:
        ry.ctrl(theta, sys[c1], target)


@cudaq.kernel
def oneshot_b2_on(
    sys: cudaq.qview,
    anc: cudaq.qview,
    angles: list[float],
    phi: float,
    zero: bool,
    c1: list[int],
    c2: list[int],
    theta: list[float],
):
    """Route B2: ``len(theta)`` explicit ancillas, all post-selected on 0."""
    iqft_b(sys)
    advection_on(sys, angles, phi, zero)
    sign_fold(sys)
    for i in range(len(theta)):
        factor_to(sys, anc[i], c1[i], c2[i], theta[i])
    sign_fold(sys)
    qft_b(sys)


@cudaq.kernel
def oneshot_b2(
    amps: list[complex],
    n_anc: int,
    angles: list[float],
    phi: float,
    zero: bool,
    c1: list[int],
    c2: list[int],
    theta: list[float],
):
    """Load ``amps`` (simulator convenience), then route B2."""
    sys = cudaq.qvector(amps)
    anc = cudaq.qvector(n_anc)
    oneshot_b2_on(sys, anc, angles, phi, zero, c1, c2, theta)


@cudaq.kernel
def oneshot_b2_circuit(
    n: int,
    n_anc: int,
    angles: list[float],
    phi: float,
    zero: bool,
    c1: list[int],
    c2: list[int],
    theta: list[float],
):
    """Route B2 on fresh qubits (resource counting)."""
    sys = cudaq.qvector(n)
    anc = cudaq.qvector(n_anc)
    oneshot_b2_on(sys, anc, angles, phi, zero, c1, c2, theta)


@cudaq.kernel
def oneshot_b1_run(
    amps: list[complex],
    angles: list[float],
    phi: float,
    zero: bool,
    c1: list[int],
    c2: list[int],
    theta: list[float],
) -> list[bool]:
    """Route B1: ONE ancilla, ``mz`` then ``reset`` after every factor.

    Returns the ``len(theta)`` factor outcomes followed by the ``n`` measured system
    bits (qubit ``k`` at position ``len(theta) + k``). The run succeeded when all factor
    outcomes are 0; the system bits of those shots are the post-selected distribution.
    Use through ``cudaq.run`` (see :func:`sampling.run_records`).
    """
    sys = cudaq.qvector(amps)
    anc = cudaq.qubit()
    n = sys.size()
    iqft_b(sys)
    advection_on(sys, angles, phi, zero)
    sign_fold(sys)
    rec = [False for _ in range(len(theta) + n)]
    for i in range(len(theta)):
        factor_to(sys, anc, c1[i], c2[i], theta[i])
        rec[i] = mz(anc)
        reset(anc)
    sign_fold(sys)
    qft_b(sys)
    for k in range(n):
        rec[len(theta) + k] = mz(sys[k])
    return rec


# ------------------------------------------------------------------ route C


@cudaq.kernel
def controlled_shift_phases(
    ctl: cudaq.qubit, sys: cudaq.qview, shift_angles: list[float], offset: int
):
    """Phase layer of a shift by ``2**b`` cells, controlled on ``ctl``."""
    n = sys.size()
    for b in range(n):
        r1.ctrl(shift_angles[offset + b], ctl, sys[n - 1 - b])


@cudaq.kernel
def oneshot_c_on(
    sys: cudaq.qview,
    w: cudaq.qview,
    angles: list[float],
    phi: float,
    zero: bool,
    shift_angles: list[float],
    r_tgt: list[int],
    r_alpha: list[float],
    r_ctl: list[int],
    l_tgt: list[int],
    l_alpha: list[float],
    l_ctl: list[int],
):
    """Route C: ``A_L^dagger [QFT (advection x controlled shifts) QFT-dagger] A_R``.

    ``A_R`` prepares the signed weights on the width register ``w``, ``A_L^dagger`` is
    the reversed (CX then RY) list of the unsigned preparation. Shifts and advection
    are diagonal in the Fourier basis, so they share one QFT sandwich. Post-select ``w``
    on ``|0..0>``; the surviving system state is ``(sum_s w_s S^s) u / lambda``.
    """
    n = sys.size()
    ry_cx_sequence(w, r_tgt, r_alpha, r_ctl)
    iqft_b(sys)
    advection_on(sys, angles, phi, zero)
    for b in range(w.size()):
        controlled_shift_phases(w[b], sys, shift_angles, b * n)
    qft_b(sys)
    cx_ry_sequence(w, l_tgt, l_alpha, l_ctl)


@cudaq.kernel
def oneshot_c(
    amps: list[complex],
    q: int,
    angles: list[float],
    phi: float,
    zero: bool,
    shift_angles: list[float],
    r_tgt: list[int],
    r_alpha: list[float],
    r_ctl: list[int],
    l_tgt: list[int],
    l_alpha: list[float],
    l_ctl: list[int],
):
    """Load ``amps`` (simulator convenience), then route C with a ``q``-qubit width."""
    sys = cudaq.qvector(amps)
    w = cudaq.qvector(q)
    oneshot_c_on(
        sys,
        w,
        angles,
        phi,
        zero,
        shift_angles,
        r_tgt,
        r_alpha,
        r_ctl,
        l_tgt,
        l_alpha,
        l_ctl,
    )


@cudaq.kernel
def oneshot_c_circuit(
    n: int,
    q: int,
    angles: list[float],
    phi: float,
    zero: bool,
    shift_angles: list[float],
    r_tgt: list[int],
    r_alpha: list[float],
    r_ctl: list[int],
    l_tgt: list[int],
    l_alpha: list[float],
    l_ctl: list[int],
):
    """Route C on fresh qubits (resource counting)."""
    sys = cudaq.qvector(n)
    w = cudaq.qvector(q)
    oneshot_c_on(
        sys,
        w,
        angles,
        phi,
        zero,
        shift_angles,
        r_tgt,
        r_alpha,
        r_ctl,
        l_tgt,
        l_alpha,
        l_ctl,
    )
