"""Resource table for the spectral advection circuit, n = 2..10.

Run on qpp-cpu (no state vector is simulated, only gates are counted):

    python benchmarks/spectral_resources.py

Writes benchmarks/output/spectral_resources.json with run metadata.
"""

import json
import sys
from pathlib import Path

import cudaq

from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers.spectral_resources import resource_report

NS = range(2, 11)

LOWERING_NOTE = """
Notes
  native: cudaq.estimate_resources on the real kernel; cr1, swap and the
    (n-1)-controlled r1 each count as one gate; depth is logical depth.
    2q = arity-2 gates, multi = all gates on >= 2 qubits.
  low.*: CUDA-Q 0.16 OpenQASM 2 lowering (cudaq.translate, target qpp-cpu) of a
    builder mirror of the circuit. CX counts are specific to this compiler
    lowering (cr1 = 2 CX, swap = 3 CX, multi-controlled phase uses ancilla
    qubits, see low.q > n). They are NOT hardware-native counts; those depend
    on the target's basis gates and connectivity and come later.
"""


def main() -> None:
    cudaq.set_target("qpp-cpu")
    rows = [
        resource_report(n, nyq, swaps)
        for nyq in ("negative", "zero")
        for swaps in (False, True)
        for n in NS
    ]
    header = (
        f"{'n':>2} {'nyquist':>8} {'swaps':>5} | {'native':>6} {'2q':>4} {'multi':>5} "
        f"{'depth':>5} {'mdepth':>6} | {'low.CX':>6} {'low.tot':>7} {'low.dep':>7} "
        f"{'low.q':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        nat, low = r["native"], r["lowered"]
        print(
            f"{r['n']:>2} {r['nyquist']:>8} {r['swaps']!s:>5} | "
            f"{nat['total_gates']:>6} {nat['two_qubit_gates']:>4} "
            f"{nat['multi_qubit_gates']:>5} {nat['depth']:>5} "
            f"{nat['multi_qubit_depth']:>6} | {low['cx']:>6} "
            f"{low['total_gates']:>7} {low['depth']:>7} {low['qubits']:>5}"
        )
    print("\nCost of the Nyquist convention: zero minus negative (extra, per step)")
    delta_header = (
        f"{'n':>2} {'swaps':>5} | {'+native':>7} {'+2q':>4} {'+multi':>6} "
        f"{'+depth':>6} | {'+low.CX':>7} {'+low.tot':>8} {'+low.dep':>8} "
        f"{'+low.q':>6}"
    )
    print(delta_header)
    print("-" * len(delta_header))
    by_key = {(r["n"], r["nyquist"], r["swaps"]): r for r in rows}
    for swaps in (False, True):
        for n in NS:
            neg, zer = by_key[(n, "negative", swaps)], by_key[(n, "zero", swaps)]
            dn = {
                k: zer["native"][k] - neg["native"][k]
                for k in neg["native"]
                if k != "by_gate"
            }
            dl = {k: zer["lowered"][k] - neg["lowered"][k] for k in neg["lowered"]}
            print(
                f"{n:>2} {swaps!s:>5} | {dn['total_gates']:>7} "
                f"{dn['two_qubit_gates']:>4} {dn['multi_qubit_gates']:>6} "
                f"{dn['depth']:>6} | {dl['cx']:>7} "
                f"{dl['total_gates']:>8} {dl['depth']:>8} {dl['qubits']:>6}"
            )
    print(LOWERING_NOTE)
    out = Path(__file__).parent / "output" / "spectral_resources.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "metadata": collect_metadata(),
                "lowering": "cudaq 0.16 translate(openqasm2), target qpp-cpu, "
                "builder mirror of the kernel",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
