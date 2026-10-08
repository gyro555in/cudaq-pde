"""Resource tables for the one-shot advection-diffusion routes, n = 3..10.

Run pinned on the login node (only gates are counted, no state vector):

    taskset -c 0 python benchmarks/oneshot_resources.py

Writes benchmarks/output/oneshot_resources.json with run metadata.

Tags: M = measured (CUDA-Q lowering of a builder mirror of the circuit, or
cudaq.estimate_resources on the real kernel); C = constructed (closed form or estimate).
  CX      M  OpenQASM 2 lowering (cr1 = 2 CX, cry = 2 CX, doubly controlled RY written
             as 3 CRY + 2 CX = 8). NOT hardware-native; "zero" Nyquist adds 12n - 22 CX.
  anc     source ancillas written in the circuit (exact); lowered qubits are reported
          by the tests (route C and A/B add no compiler ancillas for "negative").
  depth   M  ASAP depth of the lowered circuit (route B1: factors share one ancilla).
  P       C  success probability sum |c_m|^2 g(m)^2 of the Gaussian pulse below
             (route C divided by lambda^2)
  shots   C  1 / (4 eps^2 P), eps = 0.01: accepted shots 1/(4 eps^2) for a bin
             probability at absolute error eps, divided by the acceptance P.
Route B1 needs mid-circuit mz + reset (verified on qpp-cpu only); B2 does not.
Route C: width register q ~ log2(sigma) + 4 for sigma >= 4; its cost ~ 2**q grows like
sigma, which at fixed physical time doubles with every qubit (EXPONENTIAL in n). For
sigma below about 2.5 the Nyquist cut of g leaves a slowly decaying ripple in real space
and the 1e-13 window is the whole grid (q = n, rows with q = n at small sigma).
"""

import json
import sys
from pathlib import Path

import cudaq
import numpy as np

from cudaq_pde import classical
from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import oneshot_resources as rr

NS = range(3, 11)
SIGMAS = (1.0, 2.0, 4.0, 8.0)
L = 2 * np.pi
A_FIXED = 0.04  # nu T (2 pi / L)^2 at fixed physical time


def pulse(n: int) -> np.ndarray:
    x = classical.grid(n, L)
    return np.exp(-0.5 * ((x - np.pi) / 0.3) ** 2)


def a_of_sigma(n: int, sigma: float) -> float:
    """``a`` for a Gaussian of ``sigma`` grid cells: ``2 pi^2 sigma^2 / N^2``."""
    return 2 * np.pi**2 * sigma**2 / 4**n


def table_gates(rows: list[dict]) -> None:
    print("Table 1: routes A, B1, B2 (independent of sigma), negative Nyquist")
    head = (
        f"{'n':>2} {'route':>5} {'anc':>3} {'mid-ckt':>7} | "
        f"{'CX(M)':>6} {'depth(M)':>8}"
    )
    print(head + f" | {'native gates(M)':>15} {'2q':>4}")
    print("-" * (len(head) + 24))
    for n in NS:
        for route in ("A", "B1", "B2"):
            r = rr.route_report(route, n, a_of_sigma(n, 1.0), pulse(n))
            rows.append(r)
            print(
                f"{n:>2} {route:>5} {r['source_ancillas']:>3} "
                f"{r['mid_circuit_ops']:>7} | "
                f"{r['lowered']['cx']:>6} {r['lowered']['depth']:>8} | "
                f"{r['native']['total_gates']:>15} {r['native']['two_qubit_gates']:>4}"
            )
            assert r["lowered"]["cx"] == r["constructed_cx"], (route, n)
        print()


def table_sigma(rows: list[dict]) -> None:
    print("Table 2: P (C), expected shots for eps = 0.01 (C), and route C, per sigma")
    head = (
        f"{'n':>2} {'sigma':>5} | {'P(A,B)':>9} {'shots(A,B)':>11} | "
        f"{'q':>2} {'anc':>3} {'CX(M)':>7} {'depth(M)':>8} {'P(C)':>9} {'shots(C)':>11}"
    )
    print(head)
    print("-" * len(head))
    for n in NS:
        for sigma in SIGMAS:
            a = a_of_sigma(n, sigma)
            pb = rr.route_report("B2", n, a, pulse(n))
            c = rr.route_report("C", n, a, pulse(n))
            rows.append(c)
            print(
                f"{n:>2} {sigma:>5.0f} | {pb['p_succ']:>9.3e} "
                f"{pb['shots_eps_0.01']:>11.3e} | "
                f"{c['q']:>2} {c['source_ancillas']:>3} {c['lowered']['cx']:>7} "
                f"{c['lowered']['depth']:>8} {c['p_succ']:>9.3e} "
                f"{c['shots_eps_0.01']:>11.3e}"
            )
            assert c["lowered"]["cx"] == c["constructed_cx"], ("C", n, sigma)
        print()


def table_fixed_time(rows: list[dict]) -> None:
    print(f"Table 3: fixed physical time, a = nu T (2 pi / L)^2 = {A_FIXED}")
    head = (
        f"{'n':>2} {'sigma':>8} | {'B CX(M)':>8} {'C: q':>5} {'C CX(M)':>9} "
        f"{'C/B':>7} {'P':>9}"
    )
    print(head)
    print("-" * len(head))
    for n in NS:
        b = rr.route_report("B2", n, A_FIXED, pulse(n))
        c = rr.route_report("C", n, A_FIXED, pulse(n))
        rows += [b, c]
        sigma = c["sigma"]
        print(
            f"{n:>2} {sigma:>8.2f} | {b['lowered']['cx']:>8} {c['q']:>5} "
            f"{c['lowered']['cx']:>9} {c['lowered']['cx'] / b['lowered']['cx']:>7.1f} "
            f"{b['p_succ']:>9.3e}"
        )


def main() -> None:
    cudaq.set_target("qpp-cpu")
    rows: list[dict] = []
    table_gates(rows)
    table_sigma(rows)
    table_fixed_time(rows)
    out = Path("benchmarks/output")
    out.mkdir(parents=True, exist_ok=True)
    (out / "oneshot_resources.json").write_text(
        json.dumps(
            {"metadata": collect_metadata(), "rows": rows}, indent=1, default=float
        )
    )
    print("\nwrote", out / "oneshot_resources.json", file=sys.stderr)


if __name__ == "__main__":
    main()
