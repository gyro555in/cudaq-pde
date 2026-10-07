"""Power-spectrum readout error vs shots, n = 3..8 (informational table).

For each n and shots, the RMS L2 error of the sampled power spectrum over several seeds
is compared with the multinomial prediction ``sqrt((1 - sum p^2) / N)``, and the
log-log slope over shots is reported (expected -1/2). Pinned on the login node:

    taskset -c 0 python benchmarks/spectrum_shots.py
"""

import json
from pathlib import Path

import cudaq
import numpy as np

from cudaq_pde import classical
from cudaq_pde.metadata import collect_metadata
from cudaq_pde.solvers import spectral_measure as sm

L = classical.TWO_PI
SHOTS = [10**2, 10**3, 10**4, 10**5, 10**6]
SEEDS = 20


def pulse(n: int) -> np.ndarray:
    x = classical.grid(n, L)
    d = (x - np.pi + np.pi) % L - np.pi
    return np.exp(-0.5 * (d / 0.5) ** 2) + 0.2 * np.cos(3 * x)


def main() -> None:
    cudaq.set_target("qpp-cpu")
    rows = []
    print(
        f"{'n':>2} | "
        + " ".join(f"{s:>8.0e}" for s in SHOTS)
        + " | slope | predicted at 1e4"
    )
    for n in range(3, 9):
        u0, c, T = pulse(n), 1.0, 1.0
        ref = sm.classical_power_spectrum(classical.evolve_spectral(u0, T, c, 0.0, L))
        rms = []
        for shots in SHOTS:
            errs = [
                np.sqrt(
                    np.sum(
                        (sm.estimate_power_spectrum(u0, c, L, T, shots, seed=s) - ref)
                        ** 2
                    )
                )
                for s in range(SEEDS)
            ]
            rms.append(float(np.sqrt(np.mean(np.square(errs)))))
        slope = float(np.polyfit(np.log(SHOTS), np.log(rms), 1)[0])
        pred = sm.expected_rms_error(ref, 10**4)
        print(
            f"{n:>2} | "
            + " ".join(f"{r:>8.1e}" for r in rms)
            + f" | {slope:>5.2f} | {pred:.1e}"
        )
        rows.append(
            {
                "n": n,
                "shots": SHOTS,
                "rms_error": rms,
                "slope": slope,
                "predicted_rms_at_1e4": pred,
            }
        )
    path = Path(__file__).parent / "output" / "spectrum_shots.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps({"metadata": collect_metadata(), "rows": rows}, indent=2) + "\n"
    )
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
