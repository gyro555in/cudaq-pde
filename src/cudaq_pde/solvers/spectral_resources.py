"""Resource accounting for the spectral advection circuit.

Two kinds of numbers are reported and kept apart:

* ``native``: from ``cudaq.estimate_resources`` on the real kernel. Gates are
  counted as the kernel writes them (``cr1``, ``swap``, the (n-1)-controlled
  ``r1`` each count as ONE gate). Depth is this logical depth.
* ``lowered``: CUDA-Q's own lowering to OpenQASM 2 (``cudaq.translate``). 0.16
  cannot translate kernels that take arguments, so a builder-API mirror of the
  same gate sequence is lowered instead. The tests tie the mirror to the real
  kernel (identical native gate counts and identical output state).
"""

from __future__ import annotations

import re
from typing import Any

import cudaq
import numpy as np

from cudaq_pde.classical import TWO_PI
from cudaq_pde.solvers.spectral import phase_angles
from cudaq_pde.solvers.spectral_kernels import spectral_advection_circuit


def expected_native_counts(n: int, nyquist: str, swaps: bool, steps: int = 1) -> dict:
    """Closed-form native gate counts, keyed like ``estimate_resources``.

    The (n-1)-controlled phase is named ``"c" * (n-1) + "r1"`` by CUDA-Q, which
    for n = 2 coincides with ``cr1`` and is then added to that entry.
    """
    zero = nyquist == "zero"
    counts = {"h": 2 * n * steps, "cr1": n * (n - 1) * steps, "r1": n * steps}
    if zero:
        counts["x"] = 2 * (n - 1) * steps
        mc = "c" * (n - 1) + "r1"
        counts[mc] = counts.get(mc, 0) + steps
    if swaps:
        counts["swap"] = 2 * (n // 2) * steps
    return {k: v for k, v in counts.items() if v}


def native_counts(n: int, nyquist: str, swaps: bool, steps: int = 1) -> dict:
    """Per-gate counts and logical depths from ``cudaq.estimate_resources``."""
    angles, phi = phase_angles(n, 1.0, TWO_PI, 0.1)
    res = cudaq.estimate_resources(
        spectral_advection_circuit, n, angles, phi, nyquist == "zero", swaps, steps
    )
    by_gate = dict(res.to_dict())
    by_arity = dict(res.gate_count_by_arity)
    return {
        "by_gate": by_gate,
        "total_gates": sum(by_gate.values()),
        "two_qubit_gates": by_arity.get(2, 0),
        "multi_qubit_gates": res.multi_qubit_gate_count,
        "depth": res.depth,
        "multi_qubit_depth": res.multi_qubit_depth,
    }


def build_mirror(n: int, nyquist: str, swaps: bool, steps: int = 1):
    """Builder-API kernel with the same gate sequence as the real kernel."""
    angles, phi = phase_angles(n, 1.0, TWO_PI, 0.1)
    zero = nyquist == "zero"
    k = cudaq.make_kernel()
    q = k.qalloc(n)

    def b_forward() -> None:
        for a in range(n):
            k.h(q[a])
            theta = np.pi
            for b in range(a + 1, n):
                theta /= 2.0
                k.cr1(theta, q[b], q[a])

    def b_dagger() -> None:
        for a in range(n - 1, -1, -1):
            for b in range(n - 1, a, -1):
                k.cr1(-np.pi / 2 ** (b - a), q[b], q[a])
            k.h(q[a])

    def reverse() -> None:
        for i in range(n // 2):
            k.swap(q[i], q[n - 1 - i])

    def nyquist_phase(mirrored: bool) -> None:
        clear = list(range(1, n)) if mirrored else list(range(n - 1))
        controls = [q[i] for i in clear]
        target = q[0] if mirrored else q[n - 1]
        for i in clear:
            k.x(q[i])
        k.cr1(phi, controls, target)
        for i in clear:
            k.x(q[i])

    for _ in range(steps):
        b_dagger()
        if swaps:
            reverse()
            for b in range(n):
                k.r1(angles[b], q[b])
            if zero:
                nyquist_phase(False)
            reverse()
        else:
            for b in range(n):
                k.r1(angles[b], q[n - 1 - b])
            if zero:
                nyquist_phase(True)
        b_forward()
    return k


def qasm_stats(qasm: str) -> dict[str, Any]:
    """Gate total, CX count, ASAP depth and qubit count of an OpenQASM 2 string."""
    qubits = 0
    depth_of: dict[int, int] = {}
    total = cx = 0
    for line in qasm.splitlines():
        line = line.strip().rstrip(";")
        if not line or line.startswith(("//", "OPENQASM", "include")):
            continue
        if line.startswith("qreg"):
            qubits += int(re.search(r"\[(\d+)\]", line).group(1))
            continue
        name = re.split(r"[ (]", line, maxsplit=1)[0]
        wires = [int(m) for m in re.findall(r"\[(\d+)\]", line.split(")")[-1])]
        total += 1
        cx += name == "cx"
        layer = 1 + max((depth_of.get(w, 0) for w in wires), default=0)
        for w in wires:
            depth_of[w] = layer
    return {
        "total_gates": total,
        "cx": cx,
        "depth": max(depth_of.values(), default=0),
        "qubits": qubits,
    }


def lowered_counts(n: int, nyquist: str, swaps: bool, steps: int = 1) -> dict:
    """CUDA-Q's OpenQASM 2 lowering of the mirror circuit (see module docstring)."""
    qasm = cudaq.translate(build_mirror(n, nyquist, swaps, steps), format="openqasm2")
    return qasm_stats(qasm)


def resource_report(n: int, nyquist: str = "zero", swaps: bool = False) -> dict:
    """Native and lowered resources for one step of the circuit on ``n`` qubits."""
    return {
        "n": n,
        "nyquist": nyquist,
        "swaps": swaps,
        "native": native_counts(n, nyquist, swaps),
        "expected_native": expected_native_counts(n, nyquist, swaps),
        "lowered": lowered_counts(n, nyquist, swaps),
    }
