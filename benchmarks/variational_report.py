"""Per-step accuracy and convergence tables for the variational solver.

Compares ``evolve_variational`` with explicit-Euler FD (optimization error) and
FD-Euler with the exact spectral solution (discretization error, reported
separately). Run on qpp-cpu; on the login node pin to one core so CUDA-Q runtime
threads do not multiply CPU time:

    taskset -c 0 python benchmarks/variational_report.py --ns 3 4

Larger n or longer sweeps belong in an sbatch job. Writes
benchmarks/output/variational_report.json including run metadata.
"""

import argparse
import json
from pathlib import Path

import cudaq
import numpy as np

from cudaq_pde import classical
from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import variational as V

C_ADV, NU, DT, L = 1.0, 0.4, 0.05, classical.TWO_PI
DEPTH = {3: 2, 4: 3}


def u0_of(n: int) -> np.ndarray:
    x = classical.grid(n, L)
    return 1.0 + 0.5 * np.sin(x) + 0.25 * np.cos(2 * x)


def run(n: int, depth: int, tol: float, steps: int, method: str = "L-BFGS-B"):
    ref = classical.evolve_fd_euler(u0_of(n), DT, steps, C_ADV, NU, L)
    res = V.evolve_variational(
        u0_of(n), DT, steps, C_ADV, NU, L, depth=depth, seed=0, tol=tol, method=method
    )
    _, infid, l2 = V.reference_errors(res, ref)
    return res, ref, infid, l2


def per_step_table(n: int, depth: int, tol: float, steps: int) -> dict:
    res, ref, infid, l2 = run(n, depth, tol, steps)
    print(f"\nn={n} depth={depth} tol={tol:g} params={V.num_params(n, depth)}")
    print(" step       1-F        L2       cost    nfev   nit        norm")
    for s in range(steps + 1):
        print(
            f" {s:4d}  {infid[s]:.3e}  {l2[s]:.3e}  {res.costs[s]:.3e}  "
            f"{res.nfev[s]:6d}  {res.nit[s]:4d}  {res.norms[s]:+.6f}"
        )
    T = steps * DT
    spec = classical.evolve_spectral(u0_of(n), T, C_ADV, NU, L)
    rel = lambda a: float(np.linalg.norm(a - spec) / np.linalg.norm(spec))  # noqa: E731
    disc = {
        "T": T,
        "fd_euler_vs_spectral": rel(ref[-1]),
        "variational_vs_fd_euler": float(l2[-1]),
        "variational_vs_spectral": rel(res.fields[-1]),
    }
    print(
        f" discretization (FD-Euler vs spectral, T={T}): "
        f"{disc['fd_euler_vs_spectral']:.3e} | optimization (variational vs "
        f"FD-Euler): {disc['variational_vs_fd_euler']:.3e}"
    )
    return {
        "n": n,
        "depth": depth,
        "tol": tol,
        "infidelity": infid.tolist(),
        "l2": l2.tolist(),
        "cost": res.costs.tolist(),
        "nfev": res.nfev.tolist(),
        "nit": res.nit.tolist(),
        "norms": res.norms.tolist(),
        "discretization": disc,
    }


def depth_sweep(n: int, depths: list[int], tol: float, steps: int) -> list[dict]:
    print(f"\nDepth sweep n={n} tol={tol:g} ({steps} steps): final-step values")
    print(" depth params  1-F(final)  L2(final)  nfev(total)")
    rows = []
    for d in depths:
        res, _, infid, l2 = run(n, d, tol, steps)
        print(
            f" {d:5d} {V.num_params(n, d):6d}  {infid[-1]:.3e}  {l2[-1]:.3e}"
            f"  {int(res.nfev.sum()):8d}"
        )
        rows.append(
            {
                "depth": d,
                "infidelity": float(infid[-1]),
                "l2": float(l2[-1]),
                "nfev": int(res.nfev.sum()),
            }
        )
    return rows


def tol_sweep(n: int, depth: int, tols: list[float], steps: int) -> list[dict]:
    print(f"\nTolerance sweep n={n} depth={depth} ({steps} steps)")
    print("      tol  1-F(final)  L2(final)  cost(final)  nfev(total)")
    rows = []
    for t in tols:
        res, _, infid, l2 = run(n, depth, t, steps)
        print(
            f" {t:8.0e}  {infid[-1]:.3e}  {l2[-1]:.3e}  {res.costs[-1]:.3e}"
            f"  {int(res.nfev.sum()):8d}"
        )
        rows.append(
            {
                "tol": t,
                "infidelity": float(infid[-1]),
                "l2": float(l2[-1]),
                "cost": float(res.costs[-1]),
                "nfev": int(res.nfev.sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ns", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    cudaq.set_target("qpp-cpu")
    out: dict = {
        "metadata": collect_metadata(),
        "params": {"c": C_ADV, "nu": NU, "dt": DT, "L": L, "steps": args.steps},
        "per_step": [],
        "depth_sweeps": {},
        "tol_sweeps": {},
    }
    for n in args.ns:
        out["per_step"].append(per_step_table(n, DEPTH[n], 1e-6, args.steps))
        out["depth_sweeps"][n] = depth_sweep(n, list(range(5)), 1e-5, 3)
        out["tol_sweeps"][n] = tol_sweep(
            n, DEPTH[n], [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8], 3
        )
    path = Path(__file__).parent / "output" / "variational_report.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
