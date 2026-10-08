"""Dynamic-range and spectrum tables for Cole-Hopf Burgers (n = 5, 6).

Run pinned on the login node (a few minutes of CPU):

    taskset -c 0 python benchmarks/burgers_dynamic_range.py

Writes benchmarks/output/burgers_dynamic_range.json with run metadata.

Table 1  per (case, nu, n, T): P_succ of the heat step, min p of the grid solution and
         of the EXACT solution (log domain), whether the grid solution is resolved
         (n_unres = grid points with p <= 0 or off by > 10% in ln p), the error terms of
         u at three probes (grid/aliasing, finite difference), and the shots for a
         standard error EPS of u_hat at each probe (best, worst) and to see the rarest
         bin (30 expected counts). Shots are CONSTRUCTED (delta method). "hw-res" has
         one letter per probe (pi/4, pi/2, 3pi/4): Y when min(p_{j+1}, p_{j-1}) is at
         least the noise floor (1e-3, a MODELLING ASSUMPTION for combined gate and
         readout error per shot, to be replaced by device calibration data), n when the
         probe is unresolvable regardless of shots.
Table 2  real CUDA-Q shots: scatter of u_hat at probes whose required shots are feasible
         (<= 3e6) against the delta-method prediction (z = max |error| / sigma).
Table 3  phi power spectrum (cheap global observable): RMS error vs shots against the
         multinomial prediction, and the p-weighted mean square velocity
         4 nu^2 sum k^2 P_k, the p-weighted mean of u^2, against the classical value.
"""

import json
import sys
from pathlib import Path

import cudaq
import numpy as np

from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import burgers_study as st

NUS = (0.5, 0.1, 0.05, 0.02)
NS = (5, 6)
EPS = 0.05
SHOT_CAP = 3_000_000


def table1(rows: list[dict]) -> None:
    print(f"Table 1: dynamic range, target standard error of u_hat = {EPS}")
    head = (
        f"{'case':>5} {'nu':>5} {'n':>2} {'T':>3} {'P_succ':>7} {'min p grid':>10} "
        f"{'min p true':>10} {'<fp64':>5} {'unres':>5} | {'err_grid(max)':>13} "
        f"{'err_fd(max)':>11} | {'shots best':>10} {'shots worst':>11} "
        f"{'see rarest':>10} | {'hw-res':>6}"
    )
    print(head)
    print("-" * len(head))
    for case in st.CASES:
        for T in (1.0, 0.5):
            for nu in NUS:
                for n in NS:
                    r = st.study_row(case, nu, n, T, EPS)
                    rows.append(r)
                    hw = "".join("Y" if v else "n" for v in r["resolvable_on_hardware"])
                    print(
                        f"{case:>5} {nu:>5} {n:>2} {T:>3} {r['p_succ']:7.3f} "
                        f"{r['min_p_grid']:10.2e} {r['min_p_true']:10.2e} "
                        f"{'Y' if r['below_fp64_floor'] else 'n':>5} "
                        f"{r['n_unresolved']:>5} | "
                        f"{max(r['err_grid']):13.1e} {max(r['err_fd']):11.1e} | "
                        f"{r['shots_best']:10.2e} {r['shots_worst']:11.2e} "
                        f"{r['shots_to_see_all']:10.2e} | "
                        f"{hw:>6}"
                    )
            print()


def table2(rows: list[dict]) -> None:
    print("Table 2: sampled confirmation (case sin, T = 1, route A, 5 seeds)")
    head = (
        f"{'nu':>5} {'n':>2} {'probe x':>8} {'shots':>9} {'pred sigma':>10} "
        f"{'rms err':>9} {'max|z|':>7}"
    )
    print(head)
    print("-" * len(head))
    for nu in (0.5, 0.1):
        for n in NS:
            r = st.study_row("sin", nu, n, 1.0, EPS)
            for probe, need in zip(r["probes"], r["shots_per_probe"], strict=True):
                x = probe * 2 * np.pi / 2**n
                if not np.isfinite(need) or need > SHOT_CAP:
                    needs = format(need, ".1e")
                    print(
                        f"{nu:>5} {n:>2} {x:8.3f} needs {needs:>9} shots: skipped "
                        f"(> {SHOT_CAP:.0e})"
                    )
                    continue
                out = st.sampled_confirmation(
                    "sin", nu, n, 1.0, probe, int(need), seeds=5
                )
                out.update(nu=nu, n=n, x=x)
                rows.append(out)
                print(
                    f"{nu:>5} {n:>2} {x:8.3f} {out['shots']:9d} "
                    f"{out['predicted_sigma']:10.3e} "
                    f"{out['rms_error']:9.2e} {out['max_abs_z']:7.2f}"
                )
    print()


def table3(rows: list[dict]) -> None:
    print("Table 3: phi power spectrum (case asym, T = 1, route A)")
    head = (
        f"{'nu':>5} {'n':>2} {'shots':>8} {'accepted':>8} | {'rms err':>9} "
        f"{'predicted':>9} "
        f"{'TV':>7} | {'<u^2>_p hat':>11} {'classical':>10}"
    )
    print(head)
    print("-" * len(head))
    for nu in NUS:
        for n in NS:
            for shots in (1_000, 10_000, 100_000):
                r = st.spectrum_row("asym", nu, n, 1.0, shots, seed=11)
                rows.append(r)
                print(
                    f"{nu:>5} {n:>2} {shots:8d} {r['accepted']:8d} | "
                    f"{r['rms_error']:9.2e} "
                    f"{r['rms_predicted']:9.2e} {r['tv_distance']:7.4f} | "
                    f"{r['weighted_msq_velocity_hat']:11.4f} "
                    f"{r['weighted_msq_velocity_classical']:10.4f}"
                )
        print()


def main() -> None:
    cudaq.set_target("qpp-cpu")
    out: dict = {"metadata": collect_metadata(), "eps": EPS, "dynamic_range": [],
                 "sampled": [], "spectrum": []}  # fmt: skip
    table1(out["dynamic_range"])
    table2(out["sampled"])
    table3(out["spectrum"])
    path = Path("benchmarks/output")
    path.mkdir(parents=True, exist_ok=True)
    (path / "burgers_dynamic_range.json").write_text(
        json.dumps(out, indent=1, default=float)
    )
    print("wrote", path / "burgers_dynamic_range.json", file=sys.stderr)


if __name__ == "__main__":
    main()
