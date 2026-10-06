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
    out = Path(__file__).parent / "output" / "spectral_resources.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps({"metadata": collect_metadata(), "rows": rows}, indent=2) + "\n"
    )
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
