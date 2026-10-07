"""Gate-count tables for the shift kernels and the LCU block, n = 2..6.

Run on qpp-cpu, pinned on the login node (only gates are counted):

    taskset -c 0 python benchmarks/measurable_resources.py

Writes benchmarks/output/measurable_resources.json including run metadata.
native = cudaq.estimate_resources on the real kernel (cr1, cry and every
multi-controlled X count as one gate, depth is logical). low.* = CUDA-Q 0.16
OpenQASM 2 lowering of a builder mirror tied to the kernel by the tests; these are
compiler-lowering counts, not hardware-native, and low.q includes ancillas the
compiler adds itself. anc = ancilla qubits written in the source.
"""

import json
from pathlib import Path

import cudaq
import numpy as np

from cudaq_pde import classical
from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import measurable_resources as mr

NS = range(2, 7)
HEADER = (
    f"{'n':>2} {'variant':>8} {'anc':>3} | {'native':>6} {'2q':>3} {'multi':>5} "
    f"{'depth':>5} | {'low.CX':>6} {'low.tot':>7} {'low.dep':>7} {'low.q':>5}"
)


def row(label: str, rep: dict) -> str:
    nat, low = rep["native"], rep["lowered"]
    return (
        f"{rep['n']:>2} {label:>8} {rep['source_ancillas']:>3} | "
        f"{nat['total_gates']:>6} {nat['two_qubit_gates']:>3} "
        f"{nat['multi_qubit_gates']:>5} {nat['depth']:>5} | {low['cx']:>6} "
        f"{low['total_gates']:>7} {low['depth']:>7} {low['qubits']:>5}"
    )


def main() -> None:
    cudaq.set_target("qpp-cpu")
    out: dict = {"metadata": collect_metadata(), "shifts": [], "lcu": []}
    print("Shifts (increment; the decrement is its exact reversal, same counts)")
    print(HEADER)
    print("-" * len(HEADER))
    for variant in ("qft", "mcx", "ladder"):
        for n in NS:
            rep = mr.shift_report(n, variant)
            print(row(variant, rep))
            out["shifts"].append(rep)
        print()
    print("LCU block (PREPARE, SELECT, PREPARE-dagger; ancilla qubits include the 2")
    print("slot qubits), step at half the explicit-Euler limit, c=1, nu=1.0")
    print(HEADER)
    print("-" * len(HEADER))
    for select in ("qft", "ladder"):
        for n in NS:
            dt = 0.5 * classical.fd_euler_dt_max(n, 1.0, 1.0, 2 * np.pi)
            rep = mr.lcu_report(n, dt, 1.0, 1.0, select)
            print(row(select, rep))
            out["lcu"].append(rep)
        print()
    path = Path(__file__).parent / "output" / "measurable_resources.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
