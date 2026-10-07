"""One-shot advection-diffusion drivers that read the state vector.

SIMULATOR-ONLY: every function here calls ``cudaq.get_state`` and post-selects the
ancillas on the statevector. On hardware one samples the circuits of
:mod:`oneshot_kernels` instead (see :func:`oneshot.success_probability` and the sampled
route B1 in :mod:`sampling`). The caller selects the CUDA-Q target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cudaq_pde import encoding
from cudaq_pde.classical import TWO_PI, Nyquist
from cudaq_pde.solvers import oneshot


@dataclass(frozen=True)
class OneShotResult:
    """Post-selected one-shot result.

    ``u`` is the field at time ``T`` (norm applied). ``p_succ`` is the probability that
    all ancillas read 0; for routes A and B it equals ``||u(T)||^2 / ||u0||^2``, for
    route C ``||u(T)||^2 / (lam^2 ||u0||^2)``.
    """

    u: NDArray
    p_succ: float
    norm: float
    lam: float = 1.0


def evolve_oneshot(
    u0: NDArray[np.floating],
    c: float,
    nu: float,
    L: float = TWO_PI,
    T: float = 1.0,
    route: str = "B",
    nyquist: Nyquist = "zero",
    tail_tol: float = 1e-13,
) -> OneShotResult:
    """Evolve real data to time ``T`` in one circuit and post-select the ancillas.

    SIMULATOR-ONLY (state-vector post-selection). ``route`` is ``"A"`` (multiplexed RY,
    small ``n``), ``"B"`` (explicit-ancilla contraction product, route B2) or ``"C"``
    (Gaussian LCU, ``tail_tol`` sets the width register).
    """
    import cudaq

    from cudaq_pde.solvers import oneshot_kernels as k

    psi, norm = encoding.encode(u0)
    n = encoding.num_qubits(psi.size)
    plan = oneshot.oneshot_plan(n, c, nu, L, T, nyquist)
    amps = psi.tolist()
    lam = 1.0
    if route == "A":
        seq = oneshot.route_a_sequence(n, plan.a)
        state = cudaq.get_state(
            k.oneshot_a, amps, plan.angles, plan.phi, plan.zero, seq.alpha, seq.ctl
        )
    elif route == "B":
        fac = oneshot.contractions(n, plan.a)
        state = cudaq.get_state(
            k.oneshot_b2, amps, fac.count, plan.angles, plan.phi, plan.zero,
            fac.c1, fac.c2, fac.theta,
        )  # fmt: skip
    elif route == "C":
        g = oneshot.gaussian_plan(n, plan.a, tail_tol)
        lam = g.lam
        state = cudaq.get_state(
            k.oneshot_c, amps, g.q, plan.angles, plan.phi, plan.zero, g.shift_angles,
            g.prep_r.tgt, g.prep_r.alpha, g.prep_r.ctl,
            g.prep_l.reversed().tgt, g.prep_l.reversed().alpha, g.prep_l.reversed().ctl,
        )  # fmt: skip
    else:
        raise ValueError(f"route must be one of {oneshot.ROUTES}, got {route!r}")
    block = np.array(state, dtype=np.complex128)[: psi.size]  # ancillas all 0
    p = float(np.vdot(block, block).real)
    out = lam * block
    if nyquist == "zero":
        u = encoding.decode(out, norm, atol=1e-10 * max(1.0, norm))
    else:
        u = norm * out
    return OneShotResult(u=u, p_succ=p, norm=norm, lam=lam)
