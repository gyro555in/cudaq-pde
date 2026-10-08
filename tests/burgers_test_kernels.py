"""Test-only kernels: load by gates, apply a CP8 heat block, no measurement.

STATEVECTOR CHECK (test only): ``cudaq.get_state`` of these gives the unmeasured state.
"""

import cudaq

from cudaq_pde.solvers.burgers_kernels import heat_a_on, heat_b2_on, heat_c_on
from cudaq_pde.solvers.oneshot_kernels import ry_cx_sequence


@cudaq.kernel
def state_a(
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


@cudaq.kernel
def state_b2(
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


@cudaq.kernel
def state_c(
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


STATE_KERNELS = {"A": state_a, "B2": state_b2, "C": state_c}
