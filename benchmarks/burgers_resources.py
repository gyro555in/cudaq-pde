"""Resource table for the Burgers heat step: qubits, CX, P_succ, shots per route and nu.

    taskset -c 0 python benchmarks/burgers_resources.py

Writes benchmarks/output/burgers_resources.json with run metadata.

M = measured (CUDA-Q 0.16 OpenQASM 2 lowering of a builder mirror, loader included, no
advection layer; not hardware-native), C = constructed (closed form, asserted equal to M
here; shots and P_succ are estimates). "fit20/fit24" compare the TOTAL qubit count
with a
20- and a 24-qubit device; the CX count, not the qubit count, limits route A.
Case: u0 = sin x + 0.5 sin 2x, T = 1, target standard error 0.05 of u_hat at the probes.
"""

import json
import sys
from pathlib import Path

import cudaq
import numpy as np

from cudaq_pde import burgers as B
from cudaq_pde import classical
from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import burgers_estimator as be
from cudaq_pde.solvers import burgers_resources as br
from cudaq_pde.solvers import burgers_study as st
from cudaq_pde.solvers import oneshot

NS = range(3, 8)
NUS = (0.5, 0.1, 0.05, 0.02)
T, EPS, L = 1.0, 0.05, 2 * np.pi


def phi0(n: int, nu: float) -> np.ndarray:
    return B.discrete_burgers(st.CASES["asym"](classical.grid(n)), nu, T).phi0


def table_circuits(rows: list) -> None:
    print(
        "Table 1: circuit resources at nu = 0.5 (counts do not depend on nu for A, B2)"
    )
    head = (
        f"{'n':>2} {'route':>5} {'qubits':>6} {'anc':>4} | {'loader CX':>9} "
        f"{'total CX(M)':>11} "
        f"{'=C?':>4} {'depth(M)':>8} | {'fit20':>5} {'fit24':>5}"
    )
    print(head)
    print("-" * len(head))
    for n in NS:
        for route in br.ROUTES:
            r = br.route_row(route, phi0(n, 0.5), 0.5, T)
            rows.append(r)
            assert r["cx_lowered"] == r["cx_constructed"], (route, n)
            print(
                f"{n:>2} {route:>5} {r['qubits']:>6} {r['ancillas']:>4} | "
                f"{r['loader_cx']:>9} "
                f"{r['cx_lowered']:>11} {'yes':>4} {r['depth_lowered']:>8} | "
                f"{'yes' if r['fits']['20'] else 'no':>5} "
                f"{'yes' if r['fits']['24'] else 'no':>5}"
            )
        print()


def table_cost(rows: list) -> None:
    print(f"Table 2: P_succ and shots for standard error {EPS} (C = constructed)")
    head = (
        f"{'nu':>5} {'n':>2} {'route':>5} {'P_succ':>8} | {'shots best':>10} "
        f"{'shots worst':>11} "
        f"{'resolved':>8}"
    )
    print(head)
    print("-" * len(head))
    for nu in NUS:
        for n in (5, 6):
            row = st.study_row("asym", nu, n, T, EPS)
            d = B.discrete_burgers(st.CASES["asym"](classical.grid(n)), nu, T)
            a = oneshot.diffusion_exponent(nu, T, L)
            probes = np.array(row["probes"])
            for route in br.ROUTES:
                lam = oneshot.gaussian_plan(n, a).lam if route == "C" else 1.0
                p_succ = d.p_succ / lam**2
                qp, qm = be.probe_probabilities(d.p, p_succ, probes)
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = be.shots_for_error(qp, qm, EPS, nu, L / 2**n)
                rows.append({"nu": nu, "n": n, "route": route, "p_succ": p_succ,
                             "shots_best": float(np.nanmin(s)),
                             "shots_worst": float(np.nanmax(s)),
                             "resolved": row["resolved"]})  # fmt: skip
                print(
                    f"{nu:>5} {n:>2} {route:>5} {p_succ:8.4f} | {np.nanmin(s):10.2e} "
                    f"{np.nanmax(s):11.2e} {'yes' if row['resolved'] else 'NO':>8}"
                )
        print()


def main() -> None:
    cudaq.set_target("qpp-cpu")
    out: dict = {"metadata": collect_metadata(), "circuits": [], "cost": []}
    table_circuits(out["circuits"])
    table_cost(out["cost"])
    path = Path("benchmarks/output")
    path.mkdir(parents=True, exist_ok=True)
    (path / "burgers_resources.json").write_text(
        json.dumps(out, indent=1, default=float)
    )
    print("wrote", path / "burgers_resources.json", file=sys.stderr)


if __name__ == "__main__":
    main()
