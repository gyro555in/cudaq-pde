"""Heat-step kernels for the Cole-Hopf Burgers pipeline (see :mod:`burgers_quantum`).

The Cole-Hopf field ``phi`` obeys the heat equation, which is the CP7 one-shot circuit
with ``c = 0``: the advection phases are the identity, so they are omitted here. Every
``*_on`` kernel is ``QFT-dagger -> diagonal block -> QFT`` on ``sys`` with the block's
ancillas post-selected on 0; ``fourier=True`` drops the final QFT so that measuring
``sys`` reads the Fourier amplitudes ``phi_hat_k g_k`` (power spectrum, bit-reversed
register index, see :mod:`spectral_measure`).

The ``*_measured`` kernels load the real positive ``phi0`` by gates
(:func:`oneshot.prepare_real`, ``2**n - 1`` RY and ``2**n - 2`` CX: an exponential
convenience for small ``n``, not a scalable loader) and measure the system then the
ancillas, so bitstring character ``k`` is qubit ``k`` and the shots whose ancilla
characters are all ``0`` are the accepted ones.
"""

import cudaq

from cudaq_pde.solvers.oneshot_kernels import (
    controlled_shift_phases,
    cx_ry_sequence,
    factor_to,
    multiplexed_on_ancilla,
    ry_cx_sequence,
    sign_fold,
)
from cudaq_pde.solvers.spectral_kernels import iqft_b, qft_b

# ------------------------------------------------------------------ route A


@cudaq.kernel
def heat_a_on(
    sys: cudaq.qview,
    anc: cudaq.qubit,
    fourier: bool,
    alphas: list[float],
    ctl: list[int],
):
    """Route A heat block: multiplexed ``RY(2 arccos g)`` on one ancilla."""
    iqft_b(sys)
    multiplexed_on_ancilla(sys, anc, alphas, ctl)
    if not fourier:
        qft_b(sys)


@cudaq.kernel
def heat_a_measured(
    n: int,
    fourier: bool,
    ld_tgt: list[int],
    ld_alpha: list[float],
    ld_ctl: list[int],
    alphas: list[float],
    ctl: list[int],
):
    sys = cudaq.qvector(n)
    anc = cudaq.qubit()
    ry_cx_sequence(sys, ld_tgt, ld_alpha, ld_ctl)
    heat_a_on(sys, anc, fourier, alphas, ctl)
    mz(sys)
    mz(anc)


# ------------------------------------------------------------------ route B2


@cudaq.kernel
def heat_b2_on(
    sys: cudaq.qview,
    anc: cudaq.qview,
    fourier: bool,
    c1: list[int],
    c2: list[int],
    theta: list[float],
):
    """Route B2 heat block: ``n(n+1)/2`` explicit ancilla contractions."""
    iqft_b(sys)
    sign_fold(sys)
    for i in range(len(theta)):
        factor_to(sys, anc[i], c1[i], c2[i], theta[i])
    sign_fold(sys)
    if not fourier:
        qft_b(sys)


@cudaq.kernel
def heat_b2_measured(
    n: int,
    n_anc: int,
    fourier: bool,
    ld_tgt: list[int],
    ld_alpha: list[float],
    ld_ctl: list[int],
    c1: list[int],
    c2: list[int],
    theta: list[float],
):
    sys = cudaq.qvector(n)
    anc = cudaq.qvector(n_anc)
    ry_cx_sequence(sys, ld_tgt, ld_alpha, ld_ctl)
    heat_b2_on(sys, anc, fourier, c1, c2, theta)
    mz(sys)
    mz(anc)


# ------------------------------------------------------------------ route C


@cudaq.kernel
def heat_c_on(
    sys: cudaq.qview,
    w: cudaq.qview,
    fourier: bool,
    shift_angles: list[float],
    r_tgt: list[int],
    r_alpha: list[float],
    r_ctl: list[int],
    l_tgt: list[int],
    l_alpha: list[float],
    l_ctl: list[int],
):
    """Route C heat block: ``A_L^dagger [QFT (controlled shifts) QFT-dagger] A_R``.

    The unprepare ``A_L^dagger`` acts on ``w`` only, so it commutes with the system QFT
    and the ``fourier`` variant may simply skip that QFT.
    """
    n = sys.size()
    ry_cx_sequence(w, r_tgt, r_alpha, r_ctl)
    iqft_b(sys)
    for b in range(w.size()):
        controlled_shift_phases(w[b], sys, shift_angles, b * n)
    if not fourier:
        qft_b(sys)
    cx_ry_sequence(w, l_tgt, l_alpha, l_ctl)


@cudaq.kernel
def heat_c_measured(
    n: int,
    q: int,
    fourier: bool,
    ld_tgt: list[int],
    ld_alpha: list[float],
    ld_ctl: list[int],
    shift_angles: list[float],
    r_tgt: list[int],
    r_alpha: list[float],
    r_ctl: list[int],
    l_tgt: list[int],
    l_alpha: list[float],
    l_ctl: list[int],
):
    sys = cudaq.qvector(n)
    w = cudaq.qvector(q)
    ry_cx_sequence(sys, ld_tgt, ld_alpha, ld_ctl)
    heat_c_on(
        sys, w, fourier, shift_angles, r_tgt, r_alpha, r_ctl, l_tgt, l_alpha, l_ctl
    )
    mz(sys)
    mz(w)
