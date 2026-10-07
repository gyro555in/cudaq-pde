"""Mid-circuit measurement + reset probe on one CUDA-Q target.

    python benchmarks/mid_circuit_probe.py --target qpp-cpu
    python benchmarks/mid_circuit_probe.py --target nvidia     # fp64, GPU node only
    python benchmarks/mid_circuit_probe.py --target ionq --emulate  # no credentials

Four checks, each printed with its pass/fail and the numbers:
  1. two_factor_probe through cudaq.run: joint all-zero probability vs analytic value;
  2. route B1 at n = 3 (one ancilla, mz + reset per factor, input loaded by gates):
     sampled P_succ and the post-selected distribution vs the exact values (4 sigma
     and chi-square);
  3. route B1 through cudaq.sample (``b1sample``): compile and run only, the counts of
     all shots are stored for comparison with another target;
  4. route B2 at n = 3 compiled and sampled (explicit ancillas, no mid-circuit
     measurement): sampled distribution of all qubits, ancillas post-selected on 0.
Exit code 0 only when every check that ran passed; a check that fails to compile
prints the error and counts as failed. Results go to
benchmarks/output/mid_circuit_<target>.json.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import cudaq
import numpy as np
from scipy.stats import chi2

from cudaq_pde import classical, encoding
from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import oneshot, sampling
from cudaq_pde.solvers import oneshot_kernels as ok

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mid_circuit_probe_kernels import two_factor_probe  # noqa: E402

C, NU, L, T, N_QUBITS = 0.7, 0.05, 2 * np.pi, 0.8, 3


def set_target(name: str, emulate: bool, extra: dict) -> None:
    kwargs = dict(extra)
    if name == "nvidia":
        kwargs["option"] = "fp64"
    if emulate:
        kwargs["emulate"] = True
    cudaq.set_target(name, **kwargs)


def check_joint(shots: int, seed: int) -> dict:
    a, b, t = 1.0, 1.2, 0.9
    rec = sampling.run_records(two_factor_probe, a, b, t, shots=shots, seed=seed)
    p = float(np.mean(np.all(rec == 0, axis=1)))
    exact = float(np.cos(a / 2) ** 2)
    sigma = float(np.sqrt(exact * (1 - exact) / shots))
    return {
        "name": "joint all-zero (cudaq.run)",
        "value": p, "exact": exact, "sigma": sigma,
        "passed": bool(abs(p - exact) < 4 * sigma),
    }  # fmt: skip


def field() -> np.ndarray:
    return np.random.default_rng(7).normal(size=2**N_QUBITS)


def check_b1(shots: int, seed: int) -> dict:
    n = N_QUBITS
    u0 = field()
    psi, _ = encoding.encode(u0)
    ld = oneshot.prepare_real(psi.real, list(range(n)))  # input loaded by gates
    plan = oneshot.oneshot_plan(n, C, NU, L, T, "zero")
    fac = oneshot.contractions(n, plan.a)
    args = (n, ld.tgt, ld.alpha, ld.ctl, plan.angles, plan.phi, plan.zero,
            fac.c1, fac.c2, fac.theta)  # fmt: skip
    rec = sampling.run_records(ok.oneshot_b1_run_loaded, *args, shots=shots, seed=seed)
    F = fac.count
    acc = rec[:, :F].sum(axis=1) == 0
    p_exp = oneshot.success_probability(u0, plan.a)
    sigma = float(np.sqrt(p_exp * (1 - p_exp) / shots))
    idx = (rec[acc][:, F:] * (1 << np.arange(n))).sum(axis=1)
    counts = np.bincount(idx, minlength=2**n)
    ref = classical.evolve_spectral(u0, T, C, NU, L, "zero")
    p = ref**2 / np.sum(ref**2)
    stat = float(np.sum((counts - counts.sum() * p) ** 2 / (counts.sum() * p)))
    crit = float(chi2.ppf(1 - 1e-4, 2**n - 1))
    return {
        "name": "route B1 n=3 (mid-circuit mz+reset)",
        "p_succ": float(acc.mean()), "exact": p_exp, "sigma": sigma,
        "chi2": stat, "chi2_critical_1e-4": crit,
        "passed": bool(abs(acc.mean() - p_exp) < 4 * sigma and stat < crit),
    }  # fmt: skip


def check_b2(shots: int, seed: int) -> dict:
    n = N_QUBITS
    u0 = field()
    psi, _ = encoding.encode(u0)
    ld = oneshot.prepare_real(psi.real, list(range(n)))
    plan = oneshot.oneshot_plan(n, C, NU, L, T, "zero")
    fac = oneshot.contractions(n, plan.a)
    args = (n, fac.count, ld.tgt, ld.alpha, ld.ctl, plan.angles, plan.phi, plan.zero,
            fac.c1, fac.c2, fac.theta)  # fmt: skip
    counts = sampling.sample_counts(
        ok.oneshot_b2_measured_loaded, *args, shots=shots, seed=seed
    )
    total = sum(counts.values())
    ok_counts = np.zeros(2**n)
    for bits, c in counts.items():
        if set(bits[n:]) == {"0"}:  # ancillas all 0
            ok_counts[encoding.bitstring_to_index(bits[:n])] += c
    p_exp = oneshot.success_probability(u0, plan.a)
    p_hat = ok_counts.sum() / total
    sigma = float(np.sqrt(p_exp * (1 - p_exp) / total))
    ref = classical.evolve_spectral(u0, T, C, NU, L, "zero")
    p = ref**2 / np.sum(ref**2)
    stat = float(np.sum((ok_counts - ok_counts.sum() * p) ** 2 / (ok_counts.sum() * p)))
    crit = float(chi2.ppf(1 - 1e-4, 2**n - 1))
    return {
        "name": "route B2 n=3 (explicit ancillas, sampled)",
        "p_succ": float(p_hat), "exact": p_exp, "sigma": sigma,
        "chi2": stat, "chi2_critical_1e-4": crit,
        "passed": bool(abs(p_hat - p_exp) < 4 * sigma and stat < crit),
    }  # fmt: skip


def check_b1_sample(shots: int, seed: int) -> dict:
    """Compile-and-run check of B1 through ``cudaq.sample`` (no joint statistic).

    Records the system-bit counts of ALL shots (factor outcomes are not kept by
    ``sample``); ``passed`` only means it compiled and ran. Compare the counts with the
    qpp-cpu run of the same kernel (see the two-sample test in the report).
    """
    n = N_QUBITS
    psi, _ = encoding.encode(field())
    ld = oneshot.prepare_real(psi.real, list(range(n)))
    plan = oneshot.oneshot_plan(n, C, NU, L, T, "zero")
    fac = oneshot.contractions(n, plan.a)
    args = (n, ld.tgt, ld.alpha, ld.ctl, plan.angles, plan.phi, plan.zero,
            fac.c1, fac.c2, fac.theta)  # fmt: skip
    counts = sampling.sample_counts(
        ok.oneshot_b1_sample_loaded, *args, shots=shots, seed=seed
    )
    return {
        "name": "route B1 n=3 via cudaq.sample (compile check, all shots)",
        "counts": counts,
        "passed": True,
    }


CHECKS = {
    "joint": check_joint,
    "b1": check_b1,
    "b1sample": check_b1_sample,
    "b2": check_b2,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="qpp-cpu")
    ap.add_argument("--emulate", action="store_true")
    ap.add_argument(
        "--opt", action="append", default=[], help="key=value target option"
    )
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--checks", nargs="+", default=list(CHECKS), choices=list(CHECKS))
    a = ap.parse_args()
    extra = dict(o.split("=", 1) for o in a.opt)
    set_target(a.target, a.emulate, extra)
    meta = collect_metadata()
    print("target:", a.target, "emulate:", a.emulate, "options:", extra)
    results = []
    for name in a.checks:
        try:
            res = CHECKS[name](a.shots, a.seed)
        except Exception as exc:  # report, do not hide
            traceback.print_exc(limit=2)
            res = {
                "name": name,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(res)
        print(json.dumps(res, indent=1, default=float))
    out = Path("benchmarks/output")
    out.mkdir(parents=True, exist_ok=True)
    tag = a.target + ("_emulated" if a.emulate else "")
    (out / f"mid_circuit_{tag}.json").write_text(
        json.dumps(
            {"metadata": meta, "options": extra, "results": results},
            indent=1,
            default=float,
        )
    )
    allok = all(r["passed"] for r in results)
    print("ALL PASSED" if allok else "SOME FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
