"""Resource accounting for the one-shot advection-diffusion routes.

Three kinds of numbers, kept apart (same convention as :mod:`spectral_resources`):

* ``native``: ``cudaq.estimate_resources`` on the real ``*_circuit`` kernel (MEASURED
  from the kernel; ``cr1``, ``cry`` count as one gate each, depth is logical depth).
* ``lowered``: CUDA-Q 0.16's OpenQASM 2 lowering (``cudaq.translate``) of a builder-API
  mirror of the same gate sequence (MEASURED from the lowering; 0.16 cannot translate
  kernels with arguments). Not hardware-native. ``lowered.qubits`` includes ancillas the
  compiler adds itself, kept apart from the ancillas written in the source.
* ``constructed``: closed forms derived from the kernel structure (CONSTRUCTED). The
  tests check them against the measured values.

The mirrors are tied to the kernels by tests (equal native counts, equal output state).
Route B1 (one ancilla, mid-circuit ``mz`` + ``reset``) has the same gates as B2 plus
``n(n+1)/2`` measurements and resets; its lowering is measured on its own mirror.
"""

from __future__ import annotations

import math

import cudaq
import numpy as np

from cudaq_pde.classical import TWO_PI
from cudaq_pde.solvers import oneshot
from cudaq_pde.solvers import oneshot_kernels as ok
from cudaq_pde.solvers.spectral_resources import qasm_stats

ROUTE_LABELS = ("A", "B1", "B2", "C")


def n_factors(n: int) -> int:
    """Number of route-B contractions: ``n`` CRY and ``n(n-1)/2`` CCRY."""
    return n * (n + 1) // 2


def source_ancillas(route: str, n: int, q: int = 0) -> int:
    """Ancilla qubits written in the source (post-selected on 0)."""
    return {"A": 1, "B1": 1, "B2": n_factors(n), "C": q}[route]


# --------------------------------------------------------------------- constructed


def _sandwich_native(n: int, zero: bool) -> dict[str, int]:
    counts = {"h": 2 * n, "cr1": n * (n - 1), "r1": n}
    if zero:
        counts["x"] = 2 * (n - 1)
        mc = "c" * (n - 1) + "r1"
        counts[mc] = counts.get(mc, 0) + 1
    return counts


def _add(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


def constructed_native(
    route: str, n: int, nyquist: str = "negative", q: int = 0
) -> dict[str, int]:
    """Closed-form native gate counts keyed like ``estimate_resources`` (CONSTRUCTED).

    ``q`` is the width register of route C. Route B1 counts exclude the ``n(n+1)/2``
    ``mz`` and ``reset`` operations (see :func:`mid_circuit_ops`).
    """
    base = _sandwich_native(n, nyquist == "zero")
    if route == "A":
        return _add(base, {"ry": 2**n, "cx": 2**n})
    if route in ("B1", "B2"):
        pairs = n * (n - 1) // 2
        return _add(base, {"cry": n + 3 * pairs, "cx": 2 * (n - 1) + 2 * pairs})
    if route == "C":
        return _add(
            base,
            {
                "ry": 2 * (2**q - 1),
                "cx": 2 * (2**q - 2),
                "cr1": q * n,
            },
        )
    raise ValueError(f"unknown route {route!r}")


def mid_circuit_ops(route: str, n: int) -> int:
    """Mid-circuit ``mz`` (each followed by ``reset``) of route B1; 0 otherwise."""
    return n_factors(n) if route == "B1" else 0


def constructed_lowered_cx(
    route: str, n: int, nyquist: str = "negative", q: int = 0
) -> int:
    """Lowered CX from CUDA-Q's rules (cr1 = 2, cry = 2) (CONSTRUCTED).

    ``"zero"`` adds the lowering of the ``(n-1)``-controlled phase, ``12n - 22`` CX for
    ``n >= 3`` (see ``spectral_resources``), plus compiler ancillas.
    """
    cx = 2 * n * (n - 1)
    if nyquist == "zero":
        cx += 12 * n - 22 if n >= 3 else 2
    if route == "A":
        return cx + 2**n
    if route in ("B1", "B2"):
        return cx + 2 * n + 4 * n * (n - 1) + 2 * (n - 1)
    if route == "C":
        return cx + 2 * (2**q - 2) + 2 * q * n
    raise ValueError(f"unknown route {route!r}")


# ------------------------------------------------------------------------ measured


def kernel_args(
    route: str,
    n: int,
    a: float,
    c: float = 1.0,
    nyquist: str = "negative",
    tail_tol=1e-13,
):
    """``(kernel, args, q)`` of the real ``*_circuit`` kernel (route B1 uses B2's)."""
    angles, phi = oneshot.phase_angles(n, c, TWO_PI, 0.1)
    zero = nyquist == "zero"
    if route == "A":
        seq = oneshot.route_a_sequence(n, a)
        return ok.oneshot_a_circuit, (n, angles, phi, zero, seq.alpha, seq.ctl), 0
    if route in ("B1", "B2"):
        fac = oneshot.contractions(n, a)
        args = (n, fac.count, angles, phi, zero, fac.c1, fac.c2, fac.theta)
        return ok.oneshot_b2_circuit, args, 0
    g = oneshot.gaussian_plan(n, a, tail_tol)
    lr = g.prep_l.reversed()
    args = (
        n, g.q, angles, phi, zero, g.shift_angles,
        g.prep_r.tgt, g.prep_r.alpha, g.prep_r.ctl, lr.tgt, lr.alpha, lr.ctl,
    )  # fmt: skip
    return ok.oneshot_c_circuit, args, g.q


def native_counts(
    route: str, n: int, a: float, nyquist: str = "negative", **kw
) -> dict:
    """MEASURED native counts and logical depth from ``estimate_resources``."""
    kernel, args, q = kernel_args(route, n, a, nyquist=nyquist, **kw)
    res = cudaq.estimate_resources(kernel, *args)
    by_gate = dict(res.to_dict())
    by_arity = dict(res.gate_count_by_arity)
    return {
        "by_gate": by_gate,
        "total_gates": sum(by_gate.values()),
        "two_qubit_gates": by_arity.get(2, 0),
        "multi_qubit_gates": res.multi_qubit_gate_count,
        "depth": res.depth,
        "q": q,
    }


def build_mirror(
    route: str,
    n: int,
    a: float,
    nyquist: str = "negative",
    *,
    advection: bool = True,
    loader: oneshot.RySequence | None = None,
    **kw,
):
    """Builder-API kernel with the same gate sequence as the real kernel.

    ``advection=False`` drops the advection layer (the heat step of Burgers, ``c = 0``);
    ``loader`` is an RY/CX state-preparation list applied to the system first.
    """
    angles, phi = oneshot.phase_angles(n, 1.0, TWO_PI, 0.1)
    zero = nyquist == "zero"
    k = cudaq.make_kernel()
    sys = k.qalloc(n)

    def iqft() -> None:
        for b in range(n - 1, -1, -1):
            for j in range(n - 1, b, -1):
                k.cr1(-np.pi / 2 ** (j - b), sys[j], sys[b])
            k.h(sys[b])

    def qft() -> None:
        for b in range(n):
            k.h(sys[b])
            theta = np.pi
            for j in range(b + 1, n):
                theta /= 2.0
                k.cr1(theta, sys[j], sys[b])

    def advection_layer() -> None:
        if not advection:
            return
        for b in range(n):
            k.r1(angles[b], sys[n - 1 - b])
        if zero:
            clear = [sys[i] for i in range(1, n)]
            for i in range(1, n):
                k.x(sys[i])
            k.cr1(phi, clear, sys[0])
            for i in range(1, n):
                k.x(sys[i])

    def ry_cx(w, seq: oneshot.RySequence, cx_first: bool = False) -> None:
        for t, al, c in zip(seq.tgt, seq.alpha, seq.ctl, strict=True):
            if cx_first:
                if c >= 0:
                    k.cx(w[c], w[t])
                k.ry(al, w[t])
            else:
                k.ry(al, w[t])
                if c >= 0:
                    k.cx(w[c], w[t])

    if loader is not None:
        ry_cx(sys, loader)
    if route == "A":
        anc = k.qalloc(1)
        seq = oneshot.route_a_sequence(n, a)
        iqft()
        advection_layer()
        for al, c in zip(seq.alpha, seq.ctl, strict=True):
            k.ry(al, anc[0])
            k.cx(sys[c], anc[0])
        qft()
    elif route in ("B1", "B2"):
        fac = oneshot.contractions(n, a)
        anc = k.qalloc(fac.count if route == "B2" else 1)
        iqft()
        advection_layer()
        for j in range(1, n):
            k.cx(sys[0], sys[j])
        for i in range(fac.count):
            t = anc[i if route == "B2" else 0]
            c1, c2, th = fac.c1[i], fac.c2[i], fac.theta[i]
            if c2 >= 0:
                k.cry(0.5 * th, sys[c2], t)
                k.cx(sys[c1], sys[c2])
                k.cry(-0.5 * th, sys[c2], t)
                k.cx(sys[c1], sys[c2])
                k.cry(0.5 * th, sys[c1], t)
            else:
                k.cry(th, sys[c1], t)
            if route == "B1":
                k.mz(t)
                k.reset(t)
        for j in range(1, n):
            k.cx(sys[0], sys[j])
        qft()
    elif route == "C":
        g = oneshot.gaussian_plan(n, a, kw.get("tail_tol", 1e-13))
        w = k.qalloc(g.q)
        ry_cx(w, g.prep_r)
        iqft()
        advection_layer()
        for b in range(g.q):
            for j in range(n):
                k.cr1(g.shift_angles[b * n + j], w[b], sys[n - 1 - j])
        qft()
        ry_cx(w, g.prep_l.reversed(), cx_first=True)
    else:
        raise ValueError(f"unknown route {route!r}")
    return k


def lowered_counts(
    route: str, n: int, a: float, nyquist: str = "negative", **kw
) -> dict:
    """MEASURED: CUDA-Q's OpenQASM 2 lowering of the mirror circuit."""
    qasm = cudaq.translate(build_mirror(route, n, a, nyquist, **kw), format="openqasm2")
    return qasm_stats(qasm)


# ------------------------------------------------------------------------- shots


def expected_shots(p_succ: float, eps: float = 0.01) -> float:
    """Total shots to read a post-selected bin probability to absolute error ``eps``.

    A bin probability from ``M`` accepted shots has standard error at most
    ``1 / (2 sqrt(M))``, so ``M = 1 / (4 eps**2)``; only a fraction ``p_succ`` of the
    shots is accepted, hence ``M / p_succ`` shots (an estimate, not a measurement).
    """
    return 1.0 / (4.0 * eps**2 * p_succ)


def route_report(
    route: str, n: int, a: float, u0: np.ndarray, nyquist: str = "negative"
) -> dict:
    """Everything for one table row (see the module docstring for M and C)."""
    native = native_counts(route, n, a, nyquist)
    low = lowered_counts(route, n, a, nyquist)
    q = native["q"]
    lam = oneshot.gaussian_plan(n, a).lam if route == "C" else 1.0
    p = oneshot.success_probability(u0, a, lam=lam)
    return {
        "route": route,
        "n": n,
        "a": a,
        "sigma": math.sqrt(a) * 2**n / (math.pi * math.sqrt(2.0)),
        "q": q,
        "source_ancillas": source_ancillas(route, n, q),
        "mid_circuit_ops": mid_circuit_ops(route, n),
        "native": native,
        "lowered": low,
        "constructed_cx": constructed_lowered_cx(route, n, nyquist, q),
        "p_succ": p,
        "shots_eps_0.01": expected_shots(p),
    }
