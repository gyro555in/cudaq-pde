#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --account=rz
#SBATCH --gpus=1
#SBATCH --time=15
#SBATCH --job-name=cudaq-pde-gpu-tests
#SBATCH --output=logs/%x_%j.out

# GPU tests on one A100. Submit from the repo root:
#   sbatch benchmarks/gpu_tests.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source env/activate_rosi.sh

nvidia-smi

python - <<PY
import cudaq
from cudaq_pde.metadata import write_metadata
cudaq.set_target("nvidia", option="fp64")
meta = write_metadata("logs/gpu_tests_${SLURM_JOB_ID}_metadata.json")
print("metadata:", meta)
PY

pytest --run-gpu -m gpu -v
