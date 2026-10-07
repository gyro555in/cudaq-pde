"""Gate counts for the shift kernels and the LCU block.

Same two kinds of numbers as :mod:`spectral_resources`:

* ``native``: ``cudaq.estimate_resources`` on the real kernel; ``cr1``, ``cry`` and
  every multi-controlled X count as ONE gate each, depth is logical depth.
* ``lowered``: CUDA-Q's OpenQASM 2 lowering of a builder-API mirror of the same gate
  sequence (0.16 cannot translate kernels that take arguments). The mirrors are tied to
  the kernels by the tests (identical native counts and identical output state).
  Lowered qubit counts include ancillas the compiler adds itself, which is reported
  separately from the ancillas written in the source.
"""

from __future__ import annotations

import cudaq
import numpy as np

from cudaq_pde.solvers import lcu
from cudaq_pde.solvers.lcu_kernels import lcu_block_circuit
from cudaq_pde.solvers.shift_kernels import shift_circuit
from cudaq_pde.solvers.spectral import shift_angles
from cudaq_pde.solvers.spectral_resources import qasm_stats

SHIFT_KINDS = {
    ("qft", +1): 0,
    ("qft", -1): 1,
    ("mcx", +1): 2,
    ("mcx", -1): 3,
    ("ladder", +1): 4,
    ("ladder", -1): 5,
}


def source_ancillas(variant: str, n: int) -> int:
    """Ancilla qubits written in the source for an uncontrolled shift."""
    return max(n - 2, 0) if variant == "ladder" else 0


def _work(variant: str, n: int) -> int:
    return max(n - 2, 1) if variant == "ladder" else 1  # >= 1 so back(m) is valid


def _mc(controls: int) -> str:
    return "c" * controls + "x"


def expected_shift_native(n: int, variant: str) -> dict:
    """Closed-form native counts of an uncontrolled shift, keyed like CUDA-Q."""
    if variant == "qft":
        return {"h": 2 * n, "cr1": n * (n - 1), "r1": n}
    if variant == "mcx":
        counts: dict[str, int] = {"x": 1}
        for i in range(1, n):
            counts[_mc(i)] = counts.get(_mc(i), 0) + 1
        return counts
    counts = {"x": 1, "cx": n - 1}
    if n > 2:
        counts["ccx"] = 2 * (n - 2)
    return counts


def expected_lcu_native(n: int, select: str, n_neg: int = 0) -> dict:
    """Closed-form native counts of the LCU block (PREPARE, SELECT, PREPARE-dagger)."""
    counts = {"ry": 2, "cry": 2, "x": 4}
    if n_neg:
        counts["z"] = n_neg
    if select == "qft":
        counts["h"] = 2 * n
        counts["cr1"] = n * (n - 1) + 2 * n
    else:
        counts["cx"] = 2 * n
        counts["ccx"] = 4 * (n - 1)
    return counts


def _native(res) -> dict:
    by_gate = dict(res.to_dict())
    by_arity = dict(res.gate_count_by_arity)
    return {
        "by_gate": by_gate,
        "total_gates": sum(by_gate.values()),
        "two_qubit_gates": by_arity.get(2, 0),
        "multi_qubit_gates": res.multi_qubit_gate_count,
        "depth": res.depth,
    }


def shift_native(n: int, variant: str, direction: int = +1) -> dict:
    kind = SHIFT_KINDS[(variant, direction)]
    angles = shift_angles(n, direction)
    return _native(
        cudaq.estimate_resources(shift_circuit, n, _work(variant, n), kind, angles)
    )


def lcu_native(n: int, dt: float, c: float, nu: float, select: str) -> dict:
    plan = lcu.lcu_plan(n, dt, c, nu, 2 * np.pi)
    return _native(
        cudaq.estimate_resources(
            lcu_block_circuit,
            n,
            plan.prep,
            plan.neg,
            lcu.SELECT_KINDS[select],
            lcu.n_work(select, n),
            plan.dec_angles,
            plan.inc_angles,
        )
    )


# ---------------------------------------------------------------- mirrors


def _cx(k, controls, target) -> None:
    k.cx(controls[0] if len(controls) == 1 else list(controls), target)


def _qft_b(k, q) -> None:
    n = len(q)
    for a in range(n):
        k.h(q[a])
        theta = np.pi
        for b in range(a + 1, n):
            theta /= 2.0
            k.cr1(theta, q[b], q[a])


def _iqft_b(k, q) -> None:
    n = len(q)
    for a in range(n - 1, -1, -1):
        for b in range(n - 1, a, -1):
            k.cr1(-np.pi / 2 ** (b - a), q[b], q[a])
        k.h(q[a])


def _increment_mirror(k, q, variant: str, work=None) -> None:
    n = len(q)
    if variant == "qft":
        _iqft_b(k, q)
        for b, a in enumerate(shift_angles(n, +1)):
            k.r1(a, q[n - 1 - b])
        _qft_b(k, q)
    elif variant == "mcx":
        for j in range(n - 1):
            i = n - 1 - j
            _cx(k, q[:i], q[i])
        k.x(q[0])
    else:
        if n > 2:
            k.cx([q[0], q[1]], work[0])
            for i in range(3, n):
                k.cx([work[i - 3], q[i - 1]], work[i - 2])
            for j in range(n - 2):
                i = n - 1 - j
                k.cx(work[i - 2], q[i])
                if i > 2:
                    k.cx([work[i - 3], q[i - 1]], work[i - 2])
                else:
                    k.cx([q[0], q[1]], work[0])
        k.cx(q[0], q[1])
        k.x(q[0])


def build_shift_mirror(n: int, variant: str, with_input_bits: list[int] | None = None):
    """Builder kernel of the increment (the decrement is its exact reversal)."""
    k = cudaq.make_kernel()
    q = k.qalloc(n)
    work = k.qalloc(max(n - 2, 1)) if variant == "ladder" else None
    for b, bit in enumerate(with_input_bits or []):
        if bit:
            k.x(q[b])
    _increment_mirror(k, [q[i] for i in range(n)], variant, work)
    return k


def _controlled_increment_ladder_mirror(k, ctl, q, work) -> None:
    n = len(q)
    k.cx([ctl, q[0]], work[0])
    for i in range(2, n):
        k.cx([work[i - 2], q[i - 1]], work[i - 1])
    for j in range(n - 1):
        i = n - 1 - j
        k.cx(work[i - 1], q[i])
        if i > 1:
            k.cx([work[i - 2], q[i - 1]], work[i - 1])
        else:
            k.cx([ctl, q[0]], work[0])
    k.cx(ctl, q[0])


def _controlled_decrement_ladder_mirror(k, ctl, q, work) -> None:
    n = len(q)
    k.cx(ctl, q[0])
    for j in range(n - 1):
        i = 1 + j
        if i > 1:
            k.cx([work[i - 2], q[i - 1]], work[i - 1])
        else:
            k.cx([ctl, q[0]], work[0])
        k.cx(work[i - 1], q[i])
    for j in range(n - 2):
        i = n - 1 - j
        k.cx([work[i - 2], q[i - 1]], work[i - 1])
    k.cx([ctl, q[0]], work[0])


def build_lcu_mirror(n: int, dt: float, c: float, nu: float, select: str):
    """Builder kernel of the LCU block (system starts in |0...0>)."""
    plan = lcu.lcu_plan(n, dt, c, nu, 2 * np.pi)
    k = cudaq.make_kernel()
    q = k.qalloc(n + 2 + lcu.n_work(select, n))
    sys = [q[i] for i in range(n)]
    a0, a1 = q[n], q[n + 1]
    th_h, th_0 = plan.prep

    def prepare(sign: int) -> None:
        if sign > 0:
            k.ry(th_h, a1)
            k.x(a1)
            k.cry(th_0, a1, a0)
            k.x(a1)
        else:
            k.x(a1)
            k.cry(-th_0, a1, a0)
            k.x(a1)
            k.ry(-th_h, a1)

    prepare(+1)
    if plan.neg[0]:
        k.z(a0)
    if plan.neg[1]:
        k.z(a1)
    if select == "qft":
        _iqft_b(k, sys)
        for b, ang in enumerate(plan.dec_angles):  # decrement phases, ctl a0
            k.cr1(ang, a0, sys[n - 1 - b])
        for b, ang in enumerate(plan.inc_angles):  # increment phases, ctl a1
            k.cr1(ang, a1, sys[n - 1 - b])
        _qft_b(k, sys)
    else:
        work = [q[n + 2 + i] for i in range(n - 1)]
        _controlled_decrement_ladder_mirror(k, a0, sys, work)
        _controlled_increment_ladder_mirror(k, a1, sys, work)
    prepare(-1)
    return k


def lowered(kernel) -> dict:
    """CUDA-Q OpenQASM 2 lowering stats of a builder kernel."""
    return qasm_stats(cudaq.translate(kernel, format="openqasm2"))


def shift_report(n: int, variant: str) -> dict:
    return {
        "n": n,
        "variant": variant,
        "source_ancillas": source_ancillas(variant, n),
        "native": shift_native(n, variant),
        "lowered": lowered(build_shift_mirror(n, variant)),
    }


def lcu_report(n: int, dt: float, c: float, nu: float, select: str) -> dict:
    return {
        "n": n,
        "select": select,
        "source_ancillas": 2 + lcu.n_work(select, n),
        "native": lcu_native(n, dt, c, nu, select),
        "lowered": lowered(build_lcu_mirror(n, dt, c, nu, select)),
    }
