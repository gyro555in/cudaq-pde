"""Joint statistics of mid-circuit records: ``cudaq.run`` against ``cudaq.sample``.

Circuit (``mid_circuit_probe_kernels``): ``ry(a)`` on q0, ``ry.ctrl(b)`` q0 -> anc,
``mz`` + ``reset`` of anc (m1), ``ry.ctrl(t)`` q0 -> anc, ``mz`` + ``reset`` (m2),
``mz`` of q0 (out). The anc is only rotated when q0 = 1, and q0 = 1 forces out = 1, so a
record with ``out = 0`` and (``m1 = 1`` or ``m2 = 1``) is IMPOSSIBLE.

Run pinned on the login node (about 15 s of CPU):

    taskset -c 0 python benchmarks/mid_circuit_probe_stats.py

Writes benchmarks/output/mid_circuit_probe_stats.json with run metadata.

``exact_record_distribution`` derives the exact probabilities of the records by branch
enumeration of the circuit in numpy (no number is hard coded); the joint all-zero
probability comes out as ``cos(a/2)**2``.
"""

import json
import sys
from pathlib import Path

import cudaq
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mid_circuit_probe_kernels import (  # noqa: E402
    two_factor_probe,
    two_factor_probe_sample,
)

from cudaq_pde.metadata import collect_metadata  # noqa: E402
from cudaq_pde.solvers import sampling  # noqa: E402

A, B, T = 1.0, 1.2, 0.9
SHOTS = 100_000
SEED = 100
OUT = Path("benchmarks/output/mid_circuit_probe_stats.json")


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]])


def exact_record_distribution(a: float, b: float, t: float) -> dict[tuple, float]:
    """Exact ``P(m1, m2, out)`` of the circuit by enumerating measurement branches.

    State ``psi[q0, anc]``; ``ry.ctrl`` applies ``ry`` to the anc when q0 = 1; a
    measurement of the anc projects it and ``reset`` flips it back to 0.
    """
    psi = np.zeros((2, 2))
    psi[0, 0] = 1.0
    psi[:, 0] = ry(a) @ psi[:, 0]  # ry(a) on q0
    dist: dict[tuple, float] = {}

    def controlled(psi: np.ndarray, angle: float) -> np.ndarray:
        out = psi.copy()
        out[1, :] = ry(angle) @ psi[1, :]
        return out

    def measure_anc(psi: np.ndarray):
        for m in (0, 1):
            proj = np.zeros_like(psi)
            proj[:, m] = psi[:, m]
            p = float(np.sum(proj**2))
            if p > 0.0:
                after = proj / np.sqrt(p)
                if m == 1:  # reset: |1> -> |0> on the anc
                    after = np.stack([after[:, 1], after[:, 0]], axis=1)
                yield m, p, after

    for m1, p1, s1 in measure_anc(controlled(psi, b)):
        for m2, p2, s2 in measure_anc(controlled(s1, t)):
            for out in (0, 1):
                p3 = float(np.sum(s2[out, :] ** 2))
                if p3 > 0.0:
                    key = (m1, m2, out)
                    dist[key] = dist.get(key, 0.0) + p1 * p2 * p3
    return dist


def impossible_mass(dist: dict) -> float:
    return sum(p for (m1, m2, out), p in dist.items() if out == 0 and (m1 or m2))


def record_stats(records: np.ndarray, exact: dict, a: float) -> dict:
    """Frequencies of per-shot ``(m1, m2, out)`` records against the exact values."""
    shots = len(records)
    all_zero = float(np.mean(np.all(records == 0, axis=1)))
    impossible = float(
        np.mean((records[:, 2] == 0) & ((records[:, 0] == 1) | (records[:, 1] == 1)))
    )
    p0 = exact[(0, 0, 0)]
    sigma = float(np.sqrt(p0 * (1.0 - p0) / shots))
    return {
        "shots": shots,
        "impossible_fraction": impossible,
        "joint_all_zero": all_zero,
        "joint_all_zero_exact": p0,
        "cos2_a_over_2": float(np.cos(a / 2.0) ** 2),
        "binomial_sigma": sigma,
        "z": (all_zero - p0) / sigma,
    }


def run_stats(shots: int = SHOTS, seed: int = SEED) -> dict:
    """Both readouts on qpp-cpu at fixed seeds; returns JSON-serializable numbers."""
    cudaq.set_target("qpp-cpu")
    exact = exact_record_distribution(A, B, T)
    records = sampling.run_records(two_factor_probe, A, B, T, shots=shots, seed=seed)
    run = record_stats(records, exact, A)
    cudaq.set_random_seed(seed)
    result = cudaq.sample(
        two_factor_probe_sample, A, B, T, shots_count=shots, explicit_measurements=True
    )
    counts = {bits: int(c) for bits, c in result.items()}
    rows = np.array(
        [[int(ch) for ch in bits] for bits, c in counts.items() for _ in range(c)],
        dtype=np.int8,
    )
    explicit = record_stats(rows, exact, A)
    explicit["counts"] = counts
    return {
        "params": {"a": A, "b": B, "t": T, "shots": shots, "seed": seed},
        "exact": {"".join(map(str, k)): v for k, v in sorted(exact.items())},
        "exact_impossible_mass": impossible_mass(exact),
        "run": run,
        "sample_explicit_measurements": explicit,
    }


def write_stats(stats: dict, path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": collect_metadata(), **stats}, indent=1))
    return path


def main() -> None:
    stats = run_stats()
    print(json.dumps({k: v for k, v in stats.items() if k != "exact"}, indent=1))
    print("wrote", write_stats(stats), file=sys.stderr)


if __name__ == "__main__":
    main()
