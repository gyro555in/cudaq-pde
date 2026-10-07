"""Shot-noise study of the measurable variational solver.

For each shots-per-circuit setting and each seed, evolve ``--steps`` explicit-Euler
steps with ``cudaq.sample`` only (no state vector), starting from the CP4 simulator fit
of the initial data (classical preprocessing), and compare with explicit-Euler FD:

* final and worst-step infidelity ``1 - F`` to the FD state and relative L2 error;
* total circuits and shots per time step: ``iterations * 2P`` gradient circuits of
  ``shots`` shots plus one final circuit (``P = n (depth + 1)`` angles);
* norm-tracking error vs the FD norm, split into the statistical error of the sampled
  success probability (vs a noise-free "oracle" chain of exact overlaps computed in
  numpy from the same angles) and the error from the imperfect state (oracle vs FD).

Long runs go through benchmarks/shot_study.sbatch. Quick local check, pinned:

    taskset -c 0 python benchmarks/shot_study.py --shots 100 1000 --seeds 2
"""

import argparse
import json
import time
from pathlib import Path

import cudaq
import numpy as np

from cudaq_pde import classical, encoding
from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import measurable
from cudaq_pde.solvers import variational as V
from cudaq_pde.solvers.ansatz_numpy import numpy_ansatz

C_ADV, NU, DT, L = 1.0, 0.4, 0.05, classical.TWO_PI


def prepare(n: int, depth: int, steps: int):
    """FD-Euler reference and the simulator-fit starting angles (preprocessing)."""
    x = classical.grid(n, L)
    u0 = 1.0 + 0.5 * np.sin(x) + 0.25 * np.cos(2 * x)
    ref = classical.evolve_fd_euler(u0, DT, steps, C_ADV, NU, L)
    psi0, norm0 = encoding.encode(u0)
    fit = V.optimize_to_target(
        psi0.real, n, depth, V.init_theta(n, depth, 0), "L-BFGS-B", 1e-8, 500
    )
    return ref, fit.theta, norm0 * abs(fit.overlap), float(np.sign(fit.overlap))


def oracle_norms(n, depth, thetas, norm0, plan):
    """Noise-free norm chain from exact overlaps ``|<psi_s|b_s>|`` (numpy)."""
    norms = [norm0]
    for s in range(1, len(thetas)):
        b = V.fd_target(numpy_ansatz(n, thetas[s - 1], depth), DT, C_ADV, NU, L)
        norms.append(norms[-1] * abs(numpy_ansatz(n, thetas[s], depth) @ b))
    return np.array(norms)


def run_one(n, depth, steps, shots, seed, iterations, lr, prep):
    ref, theta0, norm0, sign0 = prep
    from cudaq_pde.solvers import lcu

    plan = lcu.lcu_plan(n, DT, C_ADV, NU, L)
    t0 = time.perf_counter()
    res = measurable.evolve_measurable(
        theta0,
        norm0,
        DT,
        steps,
        C_ADV,
        NU,
        L,
        n,
        depth,
        shots=shots,
        seed=seed,
        iterations=iterations,
        lr=lr,
    )
    wall = time.perf_counter() - t0
    ref_norm = np.linalg.norm(ref, axis=1)
    psis = np.array([numpy_ansatz(n, th, depth) for th in res.thetas])
    infid = 1.0 - (np.einsum("sj,sj->s", psis, ref / ref_norm[:, None])) ** 2
    fields = sign0 * res.norms_abs[:, None] * psis
    l2 = np.linalg.norm(fields - ref, axis=1) / ref_norm
    oracle = oracle_norms(n, depth, res.thetas, norm0, plan)
    return {
        "infidelity": infid.tolist(),
        "l2": l2.tolist(),
        "norm_err_vs_fd": (np.abs(res.norms_abs - ref_norm) / ref_norm).tolist(),
        "norm_err_statistical": (np.abs(res.norms_abs - oracle) / oracle).tolist(),
        "norm_err_state": (np.abs(oracle - ref_norm) / ref_norm).tolist(),
        "circuits_per_step": int(res.circuits_per_step[1]),
        "shots_per_step": int(res.shots_per_step[1]),
        "wall_s": wall,
    }


def run_study(n, depth, steps, shots_list, seeds, iterations, lr, log=print):
    cudaq.set_target("qpp-cpu")
    prep = prepare(n, depth, steps)
    runs = {}
    for shots in shots_list:
        runs[shots] = []
        for s in range(seeds):
            r = run_one(n, depth, steps, shots, 1000 * s + 17, iterations, lr, prep)
            runs[shots].append(r)
            log(
                f"  shots={shots:>7} seed {s}: final 1-F {r['infidelity'][-1]:.2e}  "
                f"wall {r['wall_s']:.1f}s"
            )
    return runs


def summarize(runs: dict) -> list[dict]:
    rows = []
    for shots, rs in runs.items():
        fin = np.array([r["infidelity"][-1] for r in rs])
        worst = np.array([max(r["infidelity"][1:]) for r in rs])
        l2 = np.array([r["l2"][-1] for r in rs])
        nfd = np.array([r["norm_err_vs_fd"][-1] for r in rs])
        nst = np.array([r["norm_err_statistical"][-1] for r in rs])
        nsa = np.array([r["norm_err_state"][-1] for r in rs])
        rows.append(
            {
                "shots": shots,
                "shots_per_step": rs[0]["shots_per_step"],
                "circuits_per_step": rs[0]["circuits_per_step"],
                "final_infidelity_mean": float(fin.mean()),
                "final_infidelity_std": float(fin.std(ddof=1)) if len(fin) > 1 else 0.0,
                "worst_step_infidelity_mean": float(worst.mean()),
                "final_l2_mean": float(l2.mean()),
                "norm_err_vs_fd_mean": float(nfd.mean()),
                "norm_err_statistical_mean": float(nst.mean()),
                "norm_err_state_mean": float(nsa.mean()),
                "seeds": len(rs),
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    print(
        "\nshots/circ | circ/step shots/step |  final 1-F (mean +- std)  worst-step | "
        "final L2 | norm err: vs FD  statistical  state"
    )
    for r in rows:
        print(
            f"{r['shots']:>10} | {r['circuits_per_step']:>9} "
            f"{r['shots_per_step']:>10.2e} | "
            f"{r['final_infidelity_mean']:.2e} +- {r['final_infidelity_std']:.1e}  "
            f"{r['worst_step_infidelity_mean']:.2e} | {r['final_l2_mean']:.2e} | "
            f"{r['norm_err_vs_fd_mean']:.2e} {r['norm_err_statistical_mean']:.2e} "
            f"{r['norm_err_state_mean']:.2e}"
        )


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--shots", type=int, nargs="+", default=[100, 1000, 10000, 100000])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--lr", type=float, default=1.0)
    p.add_argument(
        "--outdir", default=str(Path(__file__).resolve().parent.parent / "results")
    )
    a = p.parse_args(argv)
    print(
        f"Shot study: n={a.n} depth={a.depth} steps={a.steps} seeds={a.seeds} "
        f"iterations={a.iterations} lr={a.lr} P={a.n * (a.depth + 1)}"
    )
    metadata = collect_metadata()  # at the start: provenance of the code that runs
    runs = run_study(a.n, a.depth, a.steps, a.shots, a.seeds, a.iterations, a.lr)
    rows = summarize(runs)
    print_table(rows)
    out = {
        "metadata": metadata,
        "params": vars(a),
        "summary": rows,
        "runs": {str(k): v for k, v in runs.items()},
    }
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "shot_study.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    main()
