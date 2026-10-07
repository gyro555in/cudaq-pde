#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=rz
#SBATCH --gpus=1
#SBATCH --exclude=ga008
#SBATCH --time=15
#SBATCH --job-name=cudaq-pde-mcm-probe
#SBATCH --output=logs/%x_%j.out

# Mid-circuit mz + reset probe and one-shot routes on one A100 (nvidia, fp64).
# Submit from the repo root:
#   sbatch benchmarks/mid_circuit_probe.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source env/activate_rosi.sh

nvidia-smi

python benchmarks/mid_circuit_probe.py --target nvidia --shots 20000

python - <<'PY'
import numpy as np
import cudaq
from cudaq_pde import classical
from cudaq_pde.solvers import oneshot_sim

cudaq.set_target("nvidia", option="fp64")
print("fp64 state-vector agreement of the one-shot routes with classical.evolve_spectral")
rng = np.random.default_rng(0)
worst = 0.0
for n in (3, 4):
    u0 = rng.normal(size=2**n)
    for nyq in ("zero", "negative"):
        ref = classical.evolve_spectral(u0, 0.8, 0.7, 0.05, 2 * np.pi, nyq)
        for route in "ABC":
            r = oneshot_sim.evolve_oneshot(u0, 0.7, 0.05, 2 * np.pi, 0.8, route, nyq)
            err = float(np.max(np.abs(r.u - ref)))
            worst = max(worst, err)
            print(f"n={n} {nyq:8s} route {route}: max error {err:.2e}")
print("worst:", worst, "PASS" if worst < 1e-12 else "FAIL")
raise SystemExit(0 if worst < 1e-12 else 1)
PY
