"""Compare the two measurable cost circuits: gate counts and estimator statistics.

(a) Hadamard test with the ancilla-controlled ansatz: 3 overlap circuits per cost
    evaluation (V = I, decrement, increment) plus 2 per-step circuits for ||b||^2.
(b) LCU b_hat preparation -> adjoint ansatz -> sample: 1 circuit per cost evaluation.

Run on qpp-cpu, pinned (only gates are counted; the statistics use numpy):

    taskset -c 0 python benchmarks/cost_circuit_comparison.py

native = cudaq.estimate_resources on a builder mirror (cr1, cry count as one gate,
2q = arity 2, multi = arity >= 2); low.CX = CUDA-Q 0.16 OpenQASM lowering (compiler
lowering, not hardware-native; cry = 2 CX, Toffoli = 6 CX). The statistics table is a
numpy Monte Carlo of the estimators from exact overlaps at equal TOTAL shots.
"""

import json
from pathlib import Path

import cudaq
import numpy as np
from scipy.optimize import minimize

from cudaq_pde import classical, encoding
from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import cost_statistics as cs
from cudaq_pde.solvers import lcu
from cudaq_pde.solvers import measurable_resources as mr
from cudaq_pde.solvers import variational as V
from cudaq_pde.solvers.ansatz_numpy import exact_cost_and_overlap, numpy_ansatz

L = classical.TWO_PI
CASES = [(3, 2), (4, 3), (5, 3), (6, 3)]  # (n, depth)


def counts_table() -> list[dict]:
    rows = []
    print("Gate counts per circuit (n qubits, ansatz depth d)")
    header = (
        f"{'n':>2} {'d':>2} | {'(b) 2q':>6} {'low.CX':>6} {'qub':>3} {'circ/eval':>9}"
    )
    header += f" | {'(a) 2q I/dec/inc':>18} {'low.CX I/dec/inc':>20}"
    print(header + f" | {'(a)/(b) eval CX':>15}")
    for n, depth in CASES:
        dt = 0.5 * classical.fd_euler_dt_max(n, 1.0, 1.0, L)
        rep = mr.cost_circuit_report(n, depth, dt, 1.0, 1.0, "qft")
        b_nat, b_low = rep["b"]["native"], rep["b"]["lowered"]
        a = rep["a_circuits"]
        a2q = "/".join(
            str(a[v]["native"]["two_qubit_gates"]) for v in ("I", "dec", "inc")
        )
        acx = "/".join(str(a[v]["lowered"]["cx"]) for v in ("I", "dec", "inc"))
        a_tot = rep["a_totals_per_cost_evaluation"]["lowered_cx"]
        print(
            f"{n:>2} {depth:>2} | {b_nat['two_qubit_gates']:>6} {b_low['cx']:>6} "
            f"{b_low['qubits']:>3} {1:>9} | {a2q:>18} {acx:>20} | "
            f"{a_tot / b_low['cx']:>15.1f}"
        )
        rows.append(rep)
    return rows


def statistics_table() -> list[dict]:
    n, depth = 3, 2
    dt = 0.5 * classical.fd_euler_dt_max(n, 1.0, 1.0, L)
    plan = lcu.lcu_plan(n, 0.05, 1.0, 0.4, L)
    x = classical.grid(n, L)
    u0 = 1 + 0.5 * np.sin(x) + 0.25 * np.cos(2 * x)
    psi0, _ = encoding.encode(u0)
    th_prev = V.optimize_to_target(
        psi0.real, n, depth, V.init_theta(n, depth, 0), "L-BFGS-B", 1e-10, 500
    ).theta
    b = V.fd_target(numpy_ansatz(n, th_prev, depth), 0.05, 1.0, 0.4, L)
    th_opt = minimize(
        lambda t: exact_cost_and_overlap(n, t, depth, b)[0],
        th_prev,
        method="L-BFGS-B",
        options={"gtol": 1e-12, "ftol": 1e-16},
    ).x
    rng = np.random.default_rng(1)
    points = {
        "near optimum": th_opt,
        "warm start (theta_prev)": th_prev,
        "generic (+0.3 rad)": th_opt + rng.normal(0, 0.3, th_opt.size),
    }
    print(
        f"\nEstimator std of C at equal total shots (n={n}, depth={depth}, 20000 draws)"
    )
    head = f"{'point':>24} {'N':>6} | {'exact C':>9} | "
    print(head + f"{'std (a)':>9} {'std (b)':>9} | ratio")
    rows = []
    for label, th in points.items():
        q = cs.exact_quantities(n, depth, plan, th_prev, th)
        for total in (3000, 30000):
            r = cs.monte_carlo(plan, q, total, 20000, seed=5)
            ratio = r["a_std"] / r["b_std"] if r["b_std"] > 0 else float("inf")
            line = f"{label:>24} {total:>6} | {r['exact_cost']:>9.2e} | "
            print(line + f"{r['a_std']:>9.2e} {r['b_std']:>9.2e} | {ratio:>5.1f}")
            rows.append({"point": label, "total_shots": total, **r})
    del dt
    return rows


def main() -> None:
    cudaq.set_target("qpp-cpu")
    out = {
        "metadata": collect_metadata(),
        "counts": counts_table(),
        "statistics": statistics_table(),
    }
    path = Path(__file__).parent / "output" / "cost_circuit_comparison.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
