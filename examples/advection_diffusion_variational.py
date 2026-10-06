"""Variational advection-diffusion with a hardware-efficient RY/CX ansatz.

Per explicit-Euler step the ansatz is optimized to match (I + dt L) applied to
the previous state. Two errors are reported separately:

* optimization error: variational solver vs ``classical.evolve_fd_euler`` (the
  same discretization, so the difference is the optimizer and the ansatz);
* discretization error: ``evolve_fd_euler`` vs the exact ``evolve_spectral``.

The cost is evaluated on the simulator from the state vector (SIMULATOR-ONLY);
the hardware version uses Hadamard tests, see ``cudaq_pde.solvers.variational``.

Run on the login node pinned to one core:

    taskset -c 0 python examples/advection_diffusion_variational.py --n 3

Writes results/advection_diffusion_variational.png and .json.
"""

import argparse
import json
from pathlib import Path

import cudaq
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cudaq_pde import classical  # noqa: E402
from cudaq_pde.metadata import collect_metadata  # noqa: E402
from cudaq_pde.solvers import variational as V  # noqa: E402

DEFAULT_OUTDIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_DEPTH = {3: 2, 4: 3}  # depths verified in the CP4 tests


def run(
    n: int = 3,
    depth: int | None = None,
    steps: int = 10,
    c: float = 1.0,
    nu: float = 0.4,
    dt: float = 0.05,
    L: float = classical.TWO_PI,
    tol: float = 1e-6,
    seed: int = 0,
    outdir: Path | str = DEFAULT_OUTDIR,
) -> dict:
    if depth is None:
        if n not in DEFAULT_DEPTH:
            raise SystemExit(f"no verified default depth for n={n}; pass --depth")
        depth = DEFAULT_DEPTH[n]
    cudaq.set_target("qpp-cpu")
    x = classical.grid(n, L)
    u0 = 1.0 + 0.5 * np.sin(x) + 0.25 * np.cos(2 * x)

    ref_fd = classical.evolve_fd_euler(u0, dt, steps, c, nu, L)
    res = V.evolve_variational(u0, dt, steps, c, nu, L, depth=depth, seed=seed, tol=tol)
    fid, l2 = V.compare_to_reference(res, ref_fd)
    spec = np.array(
        [classical.evolve_spectral(u0, s * dt, c, nu, L) for s in range(steps + 1)]
    )
    disc = np.linalg.norm(ref_fd - spec, axis=1) / np.linalg.norm(spec, axis=1)

    print(
        f"Variational advection-diffusion: n={n}, depth={depth} "
        f"({V.num_params(n, depth)} angles), c={c}, nu={nu}, dt={dt}, tol={tol:g}"
    )
    header = "  step  1-F(vs FD)  L2(vs FD)   cost      nfev   norm      "
    print(header + "disc(FD vs spectral)")
    for s in range(steps + 1):
        print(
            f"  {s:4d}  {1 - fid[s]:.2e}   {l2[s]:.2e}   {res.costs[s]:.1e}  "
            f"{res.nfev[s]:5d}  {res.norms[s]:+.5f}  {disc[s]:.2e}"
        )
    print(
        f"  optimization error (variational vs FD-Euler) at T={steps * dt:g}: "
        f"{l2[-1]:.2e}"
    )
    print(
        f"  discretization error (FD-Euler vs spectral) at T={steps * dt:g}:   "
        f"{disc[-1]:.2e}"
    )
    print(f"  total cost evaluations: {int(res.nfev.sum())}")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(x, spec[-1], label="evolve_spectral", color="C0")
    ax1.plot(x, ref_fd[-1], "--", label="FD-Euler", color="C2")
    ax1.plot(x, res.fields[-1], ".", label="variational", color="C3")
    ax1.set(xlabel="x", ylabel="u", title=f"n={n}, T={steps * dt:g}")
    ax1.legend()
    k = np.arange(steps + 1)
    ax2.semilogy(k, l2 + 1e-18, "o-", label="optimization (variational vs FD)")
    ax2.semilogy(k, disc + 1e-18, "s-", label="discretization (FD vs spectral)")
    ax2.set(xlabel="step", ylabel="relative L2 error", title="errors per step")
    ax2.legend()
    fig.tight_layout()
    png = outdir / "advection_diffusion_variational.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    payload = {
        "metadata": collect_metadata(),
        "params": {
            "n": n,
            "depth": depth,
            "steps": steps,
            "c": c,
            "nu": nu,
            "dt": dt,
            "L": L,
            "tol": tol,
            "seed": seed,
            "method": res.method,
        },
        "per_step": {
            "infidelity_vs_fd_euler": (1 - fid).tolist(),
            "l2_vs_fd_euler": l2.tolist(),
            "cost": res.costs.tolist(),
            "nfev": res.nfev.tolist(),
            "norm": res.norms.tolist(),
            "discretization_fd_vs_spectral": disc.tolist(),
        },
        "final": {
            "optimization_error_l2": float(l2[-1]),
            "discretization_error_l2": float(disc[-1]),
            "total_cost_evaluations": int(res.nfev.sum()),
        },
    }
    js = outdir / "advection_diffusion_variational.json"
    js.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {png} and {js}")
    return payload


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--nu", type=float, default=0.4)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    a = p.parse_args(argv)
    return run(
        a.n,
        a.depth,
        a.steps,
        a.c,
        a.nu,
        a.dt,
        classical.TWO_PI,
        a.tol,
        a.seed,
        a.outdir,
    )


if __name__ == "__main__":
    main()
