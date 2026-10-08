"""Resource accounting for the Burgers heat step (loader + one-shot block).

No advection layer. Same convention as :mod:`oneshot_resources`: ``lowered`` numbers
are MEASURED from CUDA-Q 0.16's OpenQASM 2 lowering of a builder mirror (not
hardware-native); the closed forms are CONSTRUCTED and asserted equal in the tests. The
gate-loaded ``phi0`` costs ``2**n - 2`` CX (``prepare_real``), exponential in ``n``: a
convenience for small ``n``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cudaq_pde import encoding
from cudaq_pde.classical import TWO_PI
from cudaq_pde.solvers import oneshot
from cudaq_pde.solvers import oneshot_resources as rr

ROUTES = ("A", "B2", "C")
DEVICE_QUBITS = (20, 24)


def loader_for(phi0: NDArray[np.floating]) -> oneshot.RySequence:
    psi, _ = encoding.encode(np.asarray(phi0, dtype=np.float64))
    return oneshot.prepare_real(psi.real, list(range(encoding.num_qubits(psi.size))))


def total_qubits(route: str, n: int, q: int = 0) -> int:
    """System plus post-selected ancillas (route C: the width register ``q``)."""
    return n + rr.source_ancillas(route, n, q)


def constructed_cx(route: str, n: int, q: int = 0) -> int:
    """Lowered CX: loader ``2**n - 2`` plus the one-shot block (advection has no CX)."""
    return (2**n - 2) + rr.constructed_lowered_cx(route, n, "negative", q)


def route_row(
    route: str, phi0: NDArray[np.floating], nu: float, T: float, L: float = TWO_PI
) -> dict:
    """Measured and constructed counts for one route, with device-fit flags."""
    n = encoding.num_qubits(np.size(phi0))
    a = oneshot.diffusion_exponent(nu, T, L)
    q = oneshot.gaussian_plan(n, a).q if route == "C" else 0
    low = rr.lowered_counts(
        route, n, a, "negative", advection=False, loader=loader_for(phi0)
    )
    qubits = total_qubits(route, n, q)
    return {
        "route": route, "n": n, "nu": nu, "T": T, "q": q,
        "qubits": qubits,
        "ancillas": rr.source_ancillas(route, n, q),
        "loader_cx": 2**n - 2,
        "cx_lowered": low["cx"],
        "cx_constructed": constructed_cx(route, n, q),
        "depth_lowered": low["depth"],
        "lowered_qubits": low["qubits"],
        "fits": {str(d): qubits <= d for d in DEVICE_QUBITS},
    }  # fmt: skip
