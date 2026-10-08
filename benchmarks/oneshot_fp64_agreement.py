"""fp64 state-vector agreement of the one-shot routes A, B2, C with the reference.

GPU node (nvidia, fp64) and login node (qpp-cpu, pinned):

    python benchmarks/oneshot_fp64_agreement.py --target nvidia --option fp64
    taskset -c 0 python benchmarks/oneshot_fp64_agreement.py --target qpp-cpu

Runs ``oneshot_sim.evolve_oneshot`` (SIMULATOR-ONLY state-vector post-selection) for
n = 3, 4, both Nyquist conventions and all three routes, and compares with
``classical.evolve_spectral``. Prints the log lines that ``mid_circuit_probe.sbatch``
has always printed, and writes benchmarks/output/oneshot_fp64_agreement.json with run
metadata (``collect_metadata``: target, GPU name, driver, commit, ``git_dirty``).
Exit code 0 only if the worst error is below 1e-12.
"""

import argparse
import json
import sys
from pathlib import Path

import cudaq
import numpy as np

from cudaq_pde import classical
from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import oneshot_sim

C, NU, L, T = 0.7, 0.05, 2 * np.pi, 0.8
NS = (3, 4)
TOLERANCE = 1e-12
OUT = Path("benchmarks/output/oneshot_fp64_agreement.json")


def run(seed: int = 0) -> list[dict]:
    """One record per (n, Nyquist convention, route): max error vs the reference."""
    rng = np.random.default_rng(seed)
    records = []
    for n in NS:
        u0 = rng.normal(size=2**n)
        for nyq in ("zero", "negative"):
            ref = classical.evolve_spectral(u0, T, C, NU, L, nyq)
            for route in "ABC":
                r = oneshot_sim.evolve_oneshot(u0, C, NU, L, T, route, nyq)
                records.append(
                    {
                        "n": n,
                        "nyquist": nyq,
                        "route": route,
                        "max_error": float(np.max(np.abs(r.u - ref))),
                        "p_succ": float(r.p_succ),
                    }
                )
    return records


def write(records: list[dict], path: Path = OUT) -> Path:
    worst = max(r["max_error"] for r in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "metadata": collect_metadata(),
                "params": {"c": C, "nu": NU, "L": L, "T": T, "ns": list(NS), "seed": 0},
                "records": records,
                "worst": worst,
                "tolerance": TOLERANCE,
                "passed": worst < TOLERANCE,
            },
            indent=1,
        )
    )
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="nvidia")
    ap.add_argument("--option", default=None, help='target option, e.g. "fp64"')
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    if a.option:
        cudaq.set_target(a.target, option=a.option)
    else:
        cudaq.set_target(a.target)
    print(
        "fp64 state-vector agreement of the one-shot routes with "
        "classical.evolve_spectral"
    )
    records = run()
    for r in records:
        head = f"n={r['n']} {r['nyquist']:8s} route {r['route']}"
        print(f"{head}: max error {r['max_error']:.2e}")
    worst = max(r["max_error"] for r in records)
    print("worst:", worst, "PASS" if worst < TOLERANCE else "FAIL")
    print("wrote", write(records, a.out), file=sys.stderr)
    return 0 if worst < TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main())
