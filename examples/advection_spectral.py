"""Gate-level spectral advection of a periodic Gaussian pulse.

Solves u_t + c u_x = 0 on N = 2**n points with the CUDA-Q circuit
QFT-dagger -> n single-qubit phase gates -> QFT and compares it with the exact
spectral solution ``classical.evolve_spectral`` (same Nyquist convention). The
state vector is read with ``cudaq.get_state``, which is SIMULATOR-ONLY: on
hardware the full field would need exponentially many shots.

Run on the login node pinned to one core (CUDA-Q runtime threads otherwise
multiply CPU time):

    taskset -c 0 python examples/advection_spectral.py --n 6

Writes results/advection_spectral.png and results/advection_spectral.json.
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
from cudaq_pde.solvers import spectral, spectral_resources  # noqa: E402

DEFAULT_OUTDIR = Path(__file__).resolve().parent.parent / "results"


def periodic_gaussian(x: np.ndarray, center: float, width: float, L: float):
    d = (x - center + L / 2) % L - L / 2  # shortest periodic distance
    return np.exp(-0.5 * (d / width) ** 2)


def run(
    n: int = 6,
    c: float = 1.0,
    L: float = classical.TWO_PI,
    T: float = 1.5,
    nyquist: str = "zero",
    outdir: Path | str = DEFAULT_OUTDIR,
) -> dict:
    cudaq.set_target("qpp-cpu")
    x = classical.grid(n, L)
    u0 = periodic_gaussian(x, L / 2, L / 20, L)

    # One-shot circuit for time T (angles are linear in t, so this is exact)
    circuit = spectral.evolve(u0, c, L, dt=T, nyquist=nyquist, swaps=False)
    with_swaps = spectral.evolve(u0, c, L, dt=T, nyquist=nyquist, swaps=True)
    # Fast-forward check: 10 steps of T/10 equal one shot of T
    stepped = spectral.evolve(u0, c, L, dt=T / 10, steps=10, nyquist=nyquist)

    # Right reference: the exact spectral solution with the same convention
    ref = classical.evolve_spectral(u0, T, c, 0.0, L, nyquist=nyquist)
    # Informational: how well the band-limited sampling represents the true
    # translated pulse (a property of the grid, not of the circuit)
    translated = periodic_gaussian(x, (L / 2 + c * T) % L, L / 20, L)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))

    errors = {
        "circuit_vs_evolve_spectral_max_abs": float(np.max(np.abs(circuit - ref))),
        "circuit_vs_evolve_spectral_rel_l2": rel(circuit, ref),
        "swaps_vs_swap_free_max_abs": float(np.max(np.abs(with_swaps - circuit))),
        "stepped_vs_one_shot_max_abs": float(np.max(np.abs(stepped - circuit))),
        "grid_vs_exact_translation_rel_l2": rel(ref, translated),
    }
    counts = {
        f"swaps={s}": spectral_resources.native_counts(n, nyquist, s)
        for s in (False, True)
    }

    print(f"Spectral advection: n={n} (N={2**n}), c={c}, L={L:.4f}, T={T}, {nyquist}")
    print("  circuit vs evolve_spectral (the reference):")
    print(f"    max abs error {errors['circuit_vs_evolve_spectral_max_abs']:.2e}")
    print(f"    relative L2   {errors['circuit_vs_evolve_spectral_rel_l2']:.2e}")
    swap_err = errors["swaps_vs_swap_free_max_abs"]
    step_err = errors["stepped_vs_one_shot_max_abs"]
    print(f"  swap layers vs swap-free circuit: {swap_err:.2e}")
    print(f"  10 steps of T/10 vs one shot of T:  {step_err:.2e}")
    print(
        "  grid representation of the translated pulse (not a circuit error): "
        f"{errors['grid_vs_exact_translation_rel_l2']:.2e}"
    )
    for label, cnt in counts.items():
        print(
            f"  gates per step ({label}): {cnt['total_gates']} total, "
            f"{cnt['two_qubit_gates']} two-qubit, logical depth {cnt['depth']}"
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(x, u0, label="initial", color="0.6")
    ax1.plot(x, ref.real, label="evolve_spectral", color="C0")
    ax1.plot(x, circuit.real, ".", label="CUDA-Q circuit", color="C3")
    ax1.set(xlabel="x", ylabel="u", title=f"n={n}, T={T}, Nyquist {nyquist}")
    ax1.legend()
    ax2.semilogy(x, np.abs(circuit - ref) + 1e-18)
    ax2.set(xlabel="x", ylabel="|circuit - evolve_spectral|", title="error")
    fig.tight_layout()
    png = outdir / "advection_spectral.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    payload = {
        "metadata": collect_metadata(),
        "params": {"n": n, "c": c, "L": L, "T": T, "nyquist": nyquist},
        "errors": errors,
        "resources_native_per_step": counts,
    }
    js = outdir / "advection_spectral.json"
    js.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {png} and {js}")
    return payload


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--L", type=float, default=classical.TWO_PI)
    p.add_argument("--T", type=float, default=1.5)
    p.add_argument("--nyquist", choices=["zero", "negative"], default="zero")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    a = p.parse_args(argv)
    return run(a.n, a.c, a.L, a.T, a.nyquist, a.outdir)


if __name__ == "__main__":
    main()
